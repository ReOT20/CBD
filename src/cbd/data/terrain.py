from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import rasterio
from pydantic import BaseModel, Field
from rasterio.errors import RasterioError
from rasterio.mask import mask
from rasterio.transform import array_bounds
from scipy.ndimage import generic_filter
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from cbd.data.common import clean_geometries, read_vector
from cbd.manifests import (
    SUPPORTED_TERRAIN_SOURCE_IDS,
    AoiManifest,
    DataManifest,
    DataSource,
    get_enabled_aois,
    get_enabled_terrain_sources,
    validate_supported_terrain_sources,
)


class TerrainResolutionError(ValueError):
    pass


class TerrainPreprocessingError(ValueError):
    pass


class TerrainDerivativesError(ValueError):
    pass


class ResolvedTerrainSource(BaseModel):
    id: str
    role: str
    provider: str
    source_type: str
    source_format: str
    local_reference: str
    path_kind: Literal["expected_root", "expected_path"]
    resolved_path: str
    raster_files: list[str] = Field(default_factory=list)
    raster_count: int


class ResolvedAoi(BaseModel):
    id: str
    split: str
    geometry_path: str
    notes: list[str] = Field(default_factory=list)


class AoiTerrainAssociation(BaseModel):
    aoi_id: str
    split: str
    geometry_path: str
    terrain_source_id: str
    terrain_role: str
    terrain_path: str
    raster_count: int


class TerrainResolutionSummary(BaseModel):
    project_name: str
    project_root: str
    data_manifest_path: str
    aoi_manifest_path: str
    enabled_aoi_count: int
    enabled_terrain_source_count: int
    aois: list[ResolvedAoi]
    terrain_sources: list[ResolvedTerrainSource]
    associations: list[AoiTerrainAssociation]


class PreprocessedRasterRecord(BaseModel):
    aoi_id: str
    split: str
    terrain_source_id: str
    input_raster_path: str
    output_raster_path: str | None = None
    status: Literal["written", "skipped"]
    skip_reason: str | None = None
    source_crs: str
    output_crs: str | None = None
    width: int | None = None
    height: int | None = None
    count: int | None = None
    dtype: str | None = None
    nodata: float | None = None
    bounds: list[float] | None = None


class TerrainPreprocessingSummary(BaseModel):
    project_name: str
    project_root: str
    terrain_resolution_artifact: str
    output_root: str
    output_summary_path: str
    total_aois_processed: int
    total_terrain_sources_processed: int
    total_raster_outputs_written: int
    records: list[PreprocessedRasterRecord]


class DerivedRasterRecord(BaseModel):
    aoi_id: str
    split: str
    terrain_source_id: str
    input_raster_path: str
    derivative_name: Literal["slope", "local_relief"]
    output_raster_path: str
    width: int
    height: int
    dtype: str
    nodata: float
    relief_window_size: int | None = None


class TerrainDerivativesSummary(BaseModel):
    project_name: str
    project_root: str
    terrain_preprocessing_artifact: str
    output_root: str
    output_summary_path: str
    total_input_rasters_processed: int
    total_derivative_rasters_written: int
    relief_window_size: int
    records: list[DerivedRasterRecord]


DERIVATIVE_OUTPUT_NODATA = -9999.0


def infer_project_root(manifest_path: str | Path, project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).expanduser().resolve()

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest_dir = manifest_file.parent
    if manifest_dir.name == "manifests":
        return manifest_dir.parent.resolve()
    return manifest_dir.resolve()


def resolve_manifest_path(project_root: Path, raw_path: str) -> Path:
    return (project_root / raw_path).expanduser().resolve()


def _validate_aoi_paths(project_root: Path, manifest: AoiManifest) -> list[ResolvedAoi]:
    resolved: list[ResolvedAoi] = []
    for aoi in get_enabled_aois(manifest):
        geometry_path = resolve_manifest_path(project_root, aoi.geometry_path)
        if not geometry_path.exists():
            raise TerrainResolutionError(
                f"AOI geometry path not found for '{aoi.id}': {geometry_path}"
            )
        resolved.append(
            ResolvedAoi(
                id=aoi.id,
                split=aoi.split,
                geometry_path=str(geometry_path),
                notes=list(aoi.notes),
            )
        )
    return resolved


def _source_local_reference(
    source: DataSource,
) -> tuple[str, Literal["expected_root", "expected_path"]]:
    if source.local.expected_root:
        return source.local.expected_root, "expected_root"
    if source.local.expected_path:
        return source.local.expected_path, "expected_path"
    raise TerrainResolutionError(
        f"Terrain source '{source.id}' must declare local.expected_root or local.expected_path."
    )


def _discover_raster_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in {".tif", ".tiff"}:
            raise TerrainResolutionError(f"Terrain file is not a TIFF raster: {path}")
        return [path.resolve()]

    if not path.is_dir():
        raise TerrainResolutionError(f"Terrain path is neither a file nor a directory: {path}")

    raster_files = sorted(
        candidate.resolve()
        for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
        for candidate in path.rglob(pattern)
        if candidate.is_file()
    )
    unique_files: list[Path] = []
    seen: set[Path] = set()
    for candidate in raster_files:
        if candidate not in seen:
            unique_files.append(candidate)
            seen.add(candidate)
    return unique_files


def _resolve_terrain_sources(
    project_root: Path, manifest: DataManifest
) -> list[ResolvedTerrainSource]:
    validate_supported_terrain_sources(manifest)

    resolved_sources: list[ResolvedTerrainSource] = []
    for source in get_enabled_terrain_sources(manifest):
        local_reference, path_kind = _source_local_reference(source)
        resolved_path = resolve_manifest_path(project_root, local_reference)

        if not resolved_path.exists():
            raise TerrainResolutionError(
                f"Terrain path not found for source '{source.id}': {resolved_path}"
            )

        raster_files = _discover_raster_files(resolved_path)
        if not raster_files:
            raise TerrainResolutionError(
                f"No TIFF rasters found for terrain source '{source.id}' under {resolved_path}"
            )

        resolved_sources.append(
            ResolvedTerrainSource(
                id=source.id,
                role=source.role,
                provider=source.provider,
                source_type=source.type,
                source_format=source.format,
                local_reference=local_reference,
                path_kind=path_kind,
                resolved_path=str(resolved_path),
                raster_files=[str(path) for path in raster_files],
                raster_count=len(raster_files),
            )
        )

    required_source_id = manifest.requirements.get("primary_terrain_source")
    resolved_source_ids = {source.id for source in resolved_sources}
    if required_source_id and required_source_id not in resolved_source_ids:
        raise TerrainResolutionError(
            f"Primary terrain source '{required_source_id}' is not enabled and resolvable."
        )

    if not resolved_sources:
        supported = ", ".join(sorted(SUPPORTED_TERRAIN_SOURCE_IDS))
        raise TerrainResolutionError(
            "No enabled terrain sources found. "
            f"Supported terrain sources for this stage: {supported}."
        )

    return resolved_sources


def resolve_terrain_inputs(
    data_manifest: DataManifest,
    aoi_manifest: AoiManifest,
    *,
    data_manifest_path: str | Path,
    aoi_manifest_path: str | Path,
    project_root: str | Path | None = None,
) -> TerrainResolutionSummary:
    root = infer_project_root(data_manifest_path, project_root)
    resolved_aois = _validate_aoi_paths(root, aoi_manifest)
    resolved_sources = _resolve_terrain_sources(root, data_manifest)

    associations = [
        AoiTerrainAssociation(
            aoi_id=aoi.id,
            split=aoi.split,
            geometry_path=aoi.geometry_path,
            terrain_source_id=source.id,
            terrain_role=source.role,
            terrain_path=source.resolved_path,
            raster_count=source.raster_count,
        )
        for aoi in resolved_aois
        for source in resolved_sources
    ]

    project_name = data_manifest.project.get("name", "unknown")
    return TerrainResolutionSummary(
        project_name=project_name,
        project_root=str(root),
        data_manifest_path=str(Path(data_manifest_path).expanduser().resolve()),
        aoi_manifest_path=str(Path(aoi_manifest_path).expanduser().resolve()),
        enabled_aoi_count=len(resolved_aois),
        enabled_terrain_source_count=len(resolved_sources),
        aois=resolved_aois,
        terrain_sources=resolved_sources,
        associations=associations,
    )


def write_terrain_resolution_summary(
    summary: TerrainResolutionSummary, output_path: str | Path
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_terrain_resolution_summary(path: str | Path) -> TerrainResolutionSummary:
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Terrain resolution artifact not found: {artifact_path}")
    try:
        return TerrainResolutionSummary.model_validate_json(
            artifact_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise TerrainPreprocessingError(
            f"Failed to parse terrain resolution artifact: {artifact_path}"
        ) from exc


def load_terrain_preprocessing_summary(path: str | Path) -> TerrainPreprocessingSummary:
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Terrain preprocessing artifact not found: {artifact_path}")
    try:
        return TerrainPreprocessingSummary.model_validate_json(
            artifact_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise TerrainDerivativesError(
            f"Failed to parse terrain preprocessing artifact: {artifact_path}"
        ) from exc


def _load_clip_geometry(geometry_path: str | Path) -> tuple[BaseGeometry, str]:
    gdf = read_vector(geometry_path)
    gdf = clean_geometries(gdf)
    if gdf.empty:
        raise TerrainPreprocessingError(
            f"AOI geometry dataset is empty after cleaning: {geometry_path}"
        )
    if gdf.crs is None:
        raise TerrainPreprocessingError(f"AOI geometry has no CRS: {geometry_path}")
    geometry = unary_union(list(gdf.geometry))
    if geometry.is_empty:
        raise TerrainPreprocessingError(
            f"AOI geometry is empty after union: {geometry_path}"
        )
    return cast(BaseGeometry, geometry), str(gdf.crs)


def _coerce_nodata(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _default_preprocess_root(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / "outputs" / "interim" / "terrain"


def _validate_relief_window_size(relief_window_size: int) -> None:
    if relief_window_size < 3 or relief_window_size % 2 == 0:
        raise TerrainDerivativesError(
            f"Relief window size must be an odd integer >= 3, got {relief_window_size}."
        )


def _nanmax_filter(window: np.ndarray) -> float:
    if np.isnan(window).all():
        return np.nan
    return float(np.nanmax(window))


def _nanmin_filter(window: np.ndarray) -> float:
    if np.isnan(window).all():
        return np.nan
    return float(np.nanmin(window))


def _build_nodata_mask(array: np.ndarray, nodata: float | int | str | None) -> np.ndarray:
    if nodata is None:
        return np.zeros(array.shape, dtype=bool)
    if isinstance(nodata, str):
        try:
            nodata = float(nodata)
        except ValueError:
            return np.zeros(array.shape, dtype=bool)
    nodata_float = float(nodata)
    if np.isnan(nodata_float):
        return np.isnan(array)
    return np.isclose(array, nodata_float)


def _compute_slope(
    array: np.ndarray,
    nodata_mask: np.ndarray,
    *,
    xres: float,
    yres: float,
) -> np.ndarray:
    work = array.astype(np.float32, copy=True)
    if nodata_mask.any():
        valid = work[~nodata_mask]
        fill_value = float(valid.mean()) if valid.size > 0 else 0.0
        work[nodata_mask] = fill_value
    grad_y, grad_x = np.gradient(work, yres, xres)
    slope = np.sqrt((grad_x ** 2) + (grad_y ** 2)).astype(np.float32)
    slope[nodata_mask] = np.float32(DERIVATIVE_OUTPUT_NODATA)
    return slope


def _compute_local_relief(
    array: np.ndarray,
    nodata_mask: np.ndarray,
    *,
    relief_window_size: int,
) -> np.ndarray:
    work = array.astype(np.float32, copy=True)
    work[nodata_mask] = np.nan
    local_max = generic_filter(
        work,
        _nanmax_filter,
        size=relief_window_size,
        mode="nearest",
    )
    local_min = generic_filter(
        work,
        _nanmin_filter,
        size=relief_window_size,
        mode="nearest",
    )
    relief = (local_max - local_min).astype(np.float32)
    relief[np.isnan(relief)] = np.float32(DERIVATIVE_OUTPUT_NODATA)
    relief[nodata_mask] = np.float32(DERIVATIVE_OUTPUT_NODATA)
    return relief


def _write_derivative_raster(
    output_path: Path,
    source_profile: dict[str, Any],
    derivative_array: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = source_profile.copy()
    profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "float32",
            "nodata": float(DERIVATIVE_OUTPUT_NODATA),
        }
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(derivative_array.astype(np.float32), 1)


def preprocess_terrain_inputs(
    terrain_resolution_summary: TerrainResolutionSummary,
    *,
    output_root: str | Path | None = None,
) -> TerrainPreprocessingSummary:
    project_root = Path(terrain_resolution_summary.project_root).expanduser().resolve()
    preprocess_root = (
        _default_preprocess_root(project_root)
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    summary_path = preprocess_root / "terrain_preprocessing_summary.json"
    raster_root = preprocess_root / "preprocessed"

    records: list[PreprocessedRasterRecord] = []
    written_count = 0

    source_by_id = {source.id: source for source in terrain_resolution_summary.terrain_sources}
    aoi_by_id = {aoi.id: aoi for aoi in terrain_resolution_summary.aois}

    for association in terrain_resolution_summary.associations:
        aoi = aoi_by_id.get(association.aoi_id)
        source = source_by_id.get(association.terrain_source_id)
        if aoi is None or source is None:
            raise TerrainPreprocessingError(
                "Terrain resolution artifact associations are inconsistent."
            )

        geometry_path = Path(aoi.geometry_path).expanduser().resolve()
        if not geometry_path.exists():
            raise TerrainPreprocessingError(
                f"AOI geometry path not found during preprocessing for '{aoi.id}': {geometry_path}"
            )
        clip_geometry, aoi_crs = _load_clip_geometry(geometry_path)

        for raster_path_str in source.raster_files:
            raster_path = Path(raster_path_str).expanduser().resolve()
            if not raster_path.exists():
                raise TerrainPreprocessingError(
                    f"Input raster path not found during preprocessing: {raster_path}"
                )
            try:
                with rasterio.open(raster_path) as src:
                    raster_crs = src.crs
                    if raster_crs is None:
                        raise TerrainPreprocessingError(
                            f"Raster has no CRS: {raster_path}"
                        )
                    raster_crs_str = str(raster_crs)
                    if raster_crs_str != aoi_crs:
                        raise TerrainPreprocessingError(
                            f"CRS mismatch for AOI '{aoi.id}' and raster '{raster_path}': "
                            f"AOI={aoi_crs}, raster={raster_crs_str}"
                        )

                    try:
                        clipped, transform = mask(
                            src,
                            [mapping(clip_geometry)],
                            crop=True,
                        )
                    except ValueError:
                        records.append(
                            PreprocessedRasterRecord(
                                aoi_id=aoi.id,
                                split=aoi.split,
                                terrain_source_id=source.id,
                                input_raster_path=str(raster_path),
                                status="skipped",
                                skip_reason="no_intersection",
                                source_crs=raster_crs_str,
                            )
                        )
                        continue

                    output_path = (
                        raster_root / aoi.id / source.id / f"{raster_path.stem}.tif"
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    profile = src.profile.copy()
                    profile.update(
                        {
                            "driver": "GTiff",
                            "height": int(clipped.shape[1]),
                            "width": int(clipped.shape[2]),
                            "transform": transform,
                        }
                    )

                    with rasterio.open(output_path, "w", **profile) as dst:
                        dst.write(clipped)

                    bounds = array_bounds(
                        int(clipped.shape[1]),
                        int(clipped.shape[2]),
                        transform,
                    )
                    bounds_list = [
                        float(value)
                        for value in cast(tuple[float, float, float, float], bounds)
                    ]
                    records.append(
                        PreprocessedRasterRecord(
                            aoi_id=aoi.id,
                            split=aoi.split,
                            terrain_source_id=source.id,
                            input_raster_path=str(raster_path),
                            output_raster_path=str(output_path),
                            status="written",
                            source_crs=raster_crs_str,
                            output_crs=raster_crs_str,
                            width=int(clipped.shape[2]),
                            height=int(clipped.shape[1]),
                            count=int(clipped.shape[0]),
                            dtype=str(clipped.dtype),
                            nodata=_coerce_nodata(cast(float | int | str | None, src.nodata)),
                            bounds=bounds_list,
                        )
                    )
                    written_count += 1
            except RasterioError as exc:
                raise TerrainPreprocessingError(
                    f"Raster preprocessing failed for {raster_path}: {exc}"
                ) from exc

    return TerrainPreprocessingSummary(
        project_name=terrain_resolution_summary.project_name,
        project_root=str(project_root),
        terrain_resolution_artifact="",
        output_root=str(preprocess_root),
        output_summary_path=str(summary_path),
        total_aois_processed=len(terrain_resolution_summary.aois),
        total_terrain_sources_processed=len(terrain_resolution_summary.terrain_sources),
        total_raster_outputs_written=written_count,
        records=records,
    )


def write_terrain_preprocessing_summary(
    summary: TerrainPreprocessingSummary, output_path: str | Path
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return path


def derive_terrain_features(
    terrain_preprocessing_summary: TerrainPreprocessingSummary,
    *,
    output_root: str | Path | None = None,
    relief_window_size: int = 5,
) -> TerrainDerivativesSummary:
    _validate_relief_window_size(relief_window_size)

    project_root = Path(terrain_preprocessing_summary.project_root).expanduser().resolve()
    derivatives_root = (
        _default_preprocess_root(project_root)
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    output_summary_path = derivatives_root / "terrain_derivatives_summary.json"
    raster_root = derivatives_root / "derivatives"

    records: list[DerivedRasterRecord] = []
    processed_inputs = 0

    for record in terrain_preprocessing_summary.records:
        if record.status != "written":
            continue

        input_raster_path = Path(record.output_raster_path or "").expanduser().resolve()
        if not input_raster_path.exists():
            raise TerrainDerivativesError(
                f"Preprocessed raster path not found for derivatives: {input_raster_path}"
            )

        try:
            with rasterio.open(input_raster_path) as src:
                if src.count != 1:
                    raise TerrainDerivativesError(
                        "Derivative stage only supports single-band rasters, "
                        f"got {src.count} for {input_raster_path}"
                    )
                xres = float(abs(src.transform.a))
                yres = float(abs(src.transform.e))
                if xres == 0.0 or yres == 0.0:
                    raise TerrainDerivativesError(
                        f"Raster has invalid pixel size for derivatives: {input_raster_path}"
                    )

                data = src.read(1).astype(np.float32)
                nodata_mask = _build_nodata_mask(
                    data,
                    cast(float | int | str | None, src.nodata),
                )
                processed_inputs += 1

                slope = _compute_slope(data, nodata_mask, xres=xres, yres=yres)
                relief = _compute_local_relief(
                    data,
                    nodata_mask,
                    relief_window_size=relief_window_size,
                )

                for derivative_name, derivative_array in (
                    ("slope", slope),
                    ("local_relief", relief),
                ):
                    output_path = (
                        raster_root
                        / record.aoi_id
                        / record.terrain_source_id
                        / f"{input_raster_path.stem}__{derivative_name}.tif"
                    )
                    _write_derivative_raster(output_path, src.profile, derivative_array)
                    records.append(
                        DerivedRasterRecord(
                            aoi_id=record.aoi_id,
                            split=record.split,
                            terrain_source_id=record.terrain_source_id,
                            input_raster_path=str(input_raster_path),
                            derivative_name=cast(Literal["slope", "local_relief"], derivative_name),
                            output_raster_path=str(output_path),
                            width=src.width,
                            height=src.height,
                            dtype="float32",
                            nodata=float(DERIVATIVE_OUTPUT_NODATA),
                            relief_window_size=(
                                relief_window_size if derivative_name == "local_relief" else None
                            ),
                        )
                    )
        except RasterioError as exc:
            raise TerrainDerivativesError(
                f"Derivative computation failed for {input_raster_path}: {exc}"
            ) from exc

    return TerrainDerivativesSummary(
        project_name=terrain_preprocessing_summary.project_name,
        project_root=str(project_root),
        terrain_preprocessing_artifact="",
        output_root=str(derivatives_root),
        output_summary_path=str(output_summary_path),
        total_input_rasters_processed=processed_inputs,
        total_derivative_rasters_written=len(records),
        relief_window_size=relief_window_size,
        records=records,
    )


def write_terrain_derivatives_summary(
    summary: TerrainDerivativesSummary, output_path: str | Path
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return path
