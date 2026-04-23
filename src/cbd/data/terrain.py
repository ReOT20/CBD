from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pydantic import BaseModel, Field
from pyproj import CRS  # pyright: ignore[reportMissingImports]
from rasterio import DatasetReader
from rasterio.errors import RasterioError
from rasterio.features import shapes
from rasterio.mask import mask
from rasterio.transform import array_bounds
from scipy.ndimage import generic_filter, label
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from sklearn.linear_model import LogisticRegression  # pyright: ignore[reportMissingImports]
from sklearn.metrics import (  # pyright: ignore[reportMissingImports]
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from cbd.data.common import (
    clean_geometries,
    read_vector,
    validate_split_value,
    validate_split_values,
)
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


class TerrainCandidatesError(ValueError):
    pass


class TerrainReviewError(ValueError):
    pass


class TerrainBaselineEvaluationError(ValueError):
    pass


class TerrainFinalInventoryError(ValueError):
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


class TerrainCandidateRecord(BaseModel):
    candidate_id: str
    aoi_id: str
    split: str
    terrain_source_id: str
    source_raster_stem: str
    output_vector_path: str
    pixel_count: int
    area_map_units: float
    bbox_width: float
    bbox_height: float
    bbox_aspect_ratio: float
    mean_local_relief: float
    max_local_relief: float
    mean_slope: float
    max_slope: float


class CandidateVectorArtifactRecord(BaseModel):
    aoi_id: str
    split: str
    terrain_source_id: str
    source_raster_stem: str
    input_raster_path: str
    candidate_vector_path: str
    candidate_count: int


class TerrainCandidatesSummary(BaseModel):
    project_name: str
    project_root: str
    terrain_derivatives_artifact: str
    output_root: str
    output_summary_path: str
    total_input_groups_processed: int
    total_candidate_vectors_written: int
    total_candidates: int
    relief_threshold: float
    min_pixels: int
    vectors: list[CandidateVectorArtifactRecord]
    records: list[TerrainCandidateRecord]


class ReviewTableArtifactRecord(BaseModel):
    aoi_id: str
    split: str
    terrain_source_id: str
    source_raster_stem: str | None = None
    row_count: int
    output_table_path: str


class ReviewOverlayArtifactRecord(BaseModel):
    aoi_id: str
    split: str
    terrain_source_id: str
    candidate_count: int
    output_vector_path: str


class TerrainReviewSummary(BaseModel):
    project_name: str
    project_root: str
    terrain_candidates_artifact: str
    output_root: str
    output_summary_path: str
    curated_manifest_path: str
    total_candidate_rows: int
    total_review_tables_written: int
    total_review_overlays_written: int
    total_curated_tables_written: int
    tables: list[ReviewTableArtifactRecord]
    overlays: list[ReviewOverlayArtifactRecord]


class CuratedReviewArtifactRecord(BaseModel):
    artifact_kind: Literal["top_candidates", "large_candidates", "small_candidates", "index"]
    split: str
    row_count: int
    output_table_path: str
    sort_policy: str


class TerrainCuratedReviewManifest(BaseModel):
    project_name: str
    project_root: str
    output_root: str
    output_manifest_path: str
    total_candidate_rows: int
    total_curated_tables_written: int
    artifacts: list[CuratedReviewArtifactRecord]


class TerrainBaselineRowRecord(BaseModel):
    candidate_id: str
    aoi_id: str
    split: str
    terrain_source_id: str
    source_raster_stem: str
    output_vector_path: str
    pixel_count: int
    area_map_units: float
    bbox_width: float
    bbox_height: float
    bbox_aspect_ratio: float
    mean_local_relief: float
    max_local_relief: float
    mean_slope: float
    max_slope: float
    matched_label_id: str | None = None
    best_iou: float = 0.0
    matched_negative_label_id: str | None = None
    best_negative_iou: float = 0.0
    is_hard_negative_match: int
    training_weight: float
    target_label: int
    score: float
    predicted_label: int


class TerrainBaselinePredictionArtifactRecord(BaseModel):
    split: str
    row_count: int
    positive_count: int
    output_path: str


class TerrainBaselineMetrics(BaseModel):
    threshold: float
    reviewed_hard_negative_weight: float
    train_row_count: int
    val_row_count: int
    train_positive_count: int
    train_negative_count: int
    val_positive_count: int
    val_negative_count: int
    val_precision: float
    val_recall: float
    val_f1: float
    val_roc_auc: float | None = None


class TerrainBaselineEvaluationSummary(BaseModel):
    project_name: str
    project_root: str
    terrain_candidates_artifact: str
    normalized_labels_path: str
    output_root: str
    output_summary_path: str
    rows_output_path: str
    metrics_output_path: str
    failure_analysis_output_path: str
    match_iou_threshold: float
    classification_threshold: float
    prediction_artifacts: list[TerrainBaselinePredictionArtifactRecord]
    metrics: TerrainBaselineMetrics


class FinalInventoryFeatureRecord(BaseModel):
    candidate_id: str
    aoi_id: str
    split: str
    terrain_source_id: str
    source_raster_stem: str
    output_vector_path: str
    pixel_count: int
    area_map_units: float
    bbox_width: float
    bbox_height: float
    bbox_aspect_ratio: float
    mean_local_relief: float
    max_local_relief: float
    mean_slope: float
    max_slope: float
    matched_label_id: str | None = None
    best_iou: float = 0.0
    matched_negative_label_id: str | None = None
    best_negative_iou: float = 0.0
    is_hard_negative_match: int
    target_label: int
    score: float
    predicted_label: int


class FinalInventoryArtifactRecord(BaseModel):
    artifact_kind: Literal["geojson", "csv"]
    output_path: str
    row_count: int


class TerrainFinalInventorySummary(BaseModel):
    project_name: str
    project_root: str
    terrain_baseline_evaluation_artifact: str
    terrain_candidates_artifact: str
    output_root: str
    output_summary_path: str
    geojson_output_path: str
    csv_output_path: str
    total_exported_features: int
    split_counts: dict[str, int]
    predicted_positive_count: int
    predicted_negative_count: int
    classification_threshold: float
    match_iou_threshold: float
    artifacts: list[FinalInventoryArtifactRecord]
    records: list[FinalInventoryFeatureRecord]


DERIVATIVE_OUTPUT_NODATA = -9999.0
CURATED_REVIEW_SORT_POLICY = (
    "max_local_relief_desc__mean_local_relief_desc__pixel_count_desc__candidate_id_asc"
)
CURATED_REVIEW_LARGE_CANDIDATE_MIN_PIXELS = 16
BASELINE_CLASSIFICATION_THRESHOLD = 0.5
BASELINE_RANDOM_STATE = 42


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
        try:
            split = validate_split_value(
                aoi.split,
                context=f"AOI manifest split for '{aoi.id}'",
            )
        except ValueError as exc:
            raise TerrainResolutionError(str(exc)) from exc
        geometry_path = resolve_manifest_path(project_root, aoi.geometry_path)
        if not geometry_path.exists():
            raise TerrainResolutionError(
                f"AOI geometry path not found for '{aoi.id}': {geometry_path}"
            )
        resolved.append(
            ResolvedAoi(
                id=aoi.id,
                split=split,
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


def load_terrain_derivatives_summary(path: str | Path) -> TerrainDerivativesSummary:
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Terrain derivatives artifact not found: {artifact_path}")
    try:
        return TerrainDerivativesSummary.model_validate_json(
            artifact_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise TerrainCandidatesError(
            f"Failed to parse terrain derivatives artifact: {artifact_path}"
        ) from exc


def load_terrain_candidates_summary(path: str | Path) -> TerrainCandidatesSummary:
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Terrain candidates artifact not found: {artifact_path}")
    try:
        return TerrainCandidatesSummary.model_validate_json(
            artifact_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise TerrainReviewError(
            f"Failed to parse terrain candidates artifact: {artifact_path}"
        ) from exc


def load_terrain_baseline_evaluation_summary(
    path: str | Path,
) -> TerrainBaselineEvaluationSummary:
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Terrain baseline evaluation artifact not found: {artifact_path}"
        )
    try:
        return TerrainBaselineEvaluationSummary.model_validate_json(
            artifact_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise TerrainFinalInventoryError(
            f"Failed to parse terrain baseline evaluation artifact: {artifact_path}"
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


def _reproject_geometry(
    geometry: BaseGeometry,
    *,
    source_crs: str,
    target_crs: str,
) -> BaseGeometry:
    if CRS.from_user_input(source_crs) == CRS.from_user_input(target_crs):
        return geometry
    geometry_frame = gpd.GeoDataFrame(geometry=[geometry], crs=source_crs)
    reprojected = geometry_frame.to_crs(target_crs).geometry.iloc[0]
    return cast(BaseGeometry, reprojected)


def _coerce_nodata(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _default_preprocess_root(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / "outputs" / "interim" / "terrain"


def _default_candidates_root(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / "outputs" / "interim" / "terrain"


def _default_review_root(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / "outputs" / "interim" / "terrain"


def _default_evaluation_root(project_root: str | Path) -> Path:
    return (
        Path(project_root).expanduser().resolve()
        / "outputs"
        / "interim"
        / "terrain"
        / "evaluation"
    )


def _default_final_inventory_root(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / "outputs" / "final" / "terrain"


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


def _validate_candidate_parameters(relief_threshold: float, min_pixels: int) -> None:
    if relief_threshold < 0.0:
        raise TerrainCandidatesError(
            f"Relief threshold must be >= 0, got {relief_threshold}."
        )
    if min_pixels < 1:
        raise TerrainCandidatesError(f"Minimum pixels must be >= 1, got {min_pixels}.")


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
                    clip_geometry_in_raster_crs = _reproject_geometry(
                        clip_geometry,
                        source_crs=aoi_crs,
                        target_crs=raster_crs_str,
                    )

                    try:
                        clipped, transform = mask(
                            src,
                            [mapping(clip_geometry_in_raster_crs)],
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


def _group_derivative_records(
    terrain_derivatives_summary: TerrainDerivativesSummary,
) -> dict[tuple[str, str, str, str], dict[str, DerivedRasterRecord]]:
    groups: dict[tuple[str, str, str, str], dict[str, DerivedRasterRecord]] = {}
    for record in terrain_derivatives_summary.records:
        input_raster_path = Path(record.input_raster_path).expanduser().resolve()
        group_key = (
            record.aoi_id,
            record.split,
            record.terrain_source_id,
            input_raster_path.stem,
        )
        derivative_group = groups.setdefault(group_key, {})
        derivative_group[record.derivative_name] = record
    return groups


def _ensure_matching_raster_grids(
    reference_dataset: DatasetReader,
    candidate_dataset: DatasetReader,
    *,
    reference_label: str,
    candidate_label: str,
) -> None:
    if (
        reference_dataset.width != candidate_dataset.width
        or reference_dataset.height != candidate_dataset.height
    ):
        raise TerrainCandidatesError(
            f"Mismatched raster shape for {reference_label} and {candidate_label}."
        )
    if reference_dataset.transform != candidate_dataset.transform:
        raise TerrainCandidatesError(
            f"Mismatched raster transform for {reference_label} and {candidate_label}."
        )
    if reference_dataset.crs != candidate_dataset.crs:
        raise TerrainCandidatesError(
            f"Mismatched raster CRS for {reference_label} and {candidate_label}."
        )


def _component_geometries(
    labeled_components: np.ndarray,
    transform: Any,
) -> dict[int, BaseGeometry]:
    component_shapes: dict[int, list[BaseGeometry]] = {}
    for geom_mapping, value in shapes(
        labeled_components.astype(np.int32),
        mask=labeled_components > 0,
        transform=transform,
    ):
        component_id = int(value)
        if component_id <= 0:
            continue
        component_shapes.setdefault(component_id, []).append(
            cast(BaseGeometry, shape(geom_mapping))
        )
    return {
        component_id: cast(BaseGeometry, unary_union(parts))
        for component_id, parts in component_shapes.items()
    }


def _empty_candidate_gdf(crs: Any) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "candidate_id": [],
            "aoi_id": [],
            "split": [],
            "terrain_source_id": [],
            "source_raster_stem": [],
            "pixel_count": np.array([], dtype=np.int32),
            "area_map_units": np.array([], dtype=np.float64),
            "bbox_width": np.array([], dtype=np.float64),
            "bbox_height": np.array([], dtype=np.float64),
            "bbox_aspect_ratio": np.array([], dtype=np.float64),
            "mean_local_relief": np.array([], dtype=np.float64),
            "max_local_relief": np.array([], dtype=np.float64),
            "mean_slope": np.array([], dtype=np.float64),
            "max_slope": np.array([], dtype=np.float64),
            "geometry": gpd.GeoSeries([], crs=crs),
        },
        geometry="geometry",
        crs=crs,
    )


def _write_candidate_vector(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")


def _write_review_table(
    rows: list[dict[str, object]],
    output_path: Path,
    *,
    fieldnames: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dataframe_record_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], df.to_dict(orient="records"))


def _candidate_row_sort_key(row: dict[str, object]) -> tuple[float, float, int, str]:
    return (
        -cast(float, row["max_local_relief"]),
        -cast(float, row["mean_local_relief"]),
        -cast(int, row["pixel_count"]),
        cast(str, row["candidate_id"]),
    )


def _curated_review_rows(
    candidate_rows: list[dict[str, object]],
    *,
    split: str,
) -> dict[str, list[dict[str, object]]]:
    if split == "all":
        split_rows = list(candidate_rows)
    else:
        split_rows = [row for row in candidate_rows if row["split"] == split]

    sorted_rows = sorted(split_rows, key=_candidate_row_sort_key)
    return {
        "top_candidates": sorted_rows,
        "large_candidates": [
            row
            for row in sorted_rows
            if cast(int, row["pixel_count"]) >= CURATED_REVIEW_LARGE_CANDIDATE_MIN_PIXELS
        ],
        "small_candidates": [
            row
            for row in sorted_rows
            if cast(int, row["pixel_count"]) < CURATED_REVIEW_LARGE_CANDIDATE_MIN_PIXELS
        ],
    }


def _write_curated_review_manifest(
    manifest: TerrainCuratedReviewManifest, output_path: Path
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return output_path


def _write_json_artifact(payload: BaseModel | dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        output_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    else:
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _validate_match_iou_threshold(match_iou: float) -> None:
    if match_iou < 0.0 or match_iou > 1.0:
        raise TerrainBaselineEvaluationError(
            f"Match IoU threshold must be between 0 and 1, got {match_iou}."
        )


def _validate_reviewed_hard_negative_weight(weight: float) -> None:
    if weight <= 0.0:
        raise TerrainBaselineEvaluationError(
            "Reviewed hard-negative weight must be greater than 0, "
            f"got {weight}."
        )


def _load_normalized_labels(labels_path: str | Path) -> gpd.GeoDataFrame:
    gdf = read_vector(labels_path)
    gdf = clean_geometries(gdf)
    required_columns = {
        "label_id",
        "class_name",
        "source_id",
        "split",
        "review_status",
        "notes",
    }
    missing_columns = required_columns - set(gdf.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise TerrainBaselineEvaluationError(
            f"Normalized labels vector is missing required column(s): {missing}."
        )
    if gdf.crs is None:
        raise TerrainBaselineEvaluationError("Normalized labels vector has no CRS.")
    try:
        normalized_splits = validate_split_values(
            gdf["split"].tolist(),
            context="Normalized labels split column",
        )
    except ValueError as exc:
        raise TerrainBaselineEvaluationError(str(exc)) from exc
    gdf = gdf.copy()
    gdf["split"] = normalized_splits
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf.crs)


def _load_candidate_geometries(
    terrain_candidates_summary: TerrainCandidatesSummary,
) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for vector_record in terrain_candidates_summary.vectors:
        candidate_vector_path = Path(vector_record.candidate_vector_path).expanduser().resolve()
        if not candidate_vector_path.exists():
            raise TerrainBaselineEvaluationError(
                f"Candidate vector path not found for evaluation: {candidate_vector_path}"
            )
        try:
            gdf = gpd.read_file(candidate_vector_path)
        except Exception as exc:
            raise TerrainBaselineEvaluationError(
                f"Candidate vector could not be read for evaluation: {candidate_vector_path}"
            ) from exc

        required_columns = {
            "candidate_id",
            "aoi_id",
            "split",
            "terrain_source_id",
            "source_raster_stem",
        }
        missing_columns = required_columns - set(gdf.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise TerrainBaselineEvaluationError(
                f"Candidate vector is missing required column(s): {missing}."
            )
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf.crs)
        if gdf.crs is None:
            raise TerrainBaselineEvaluationError(
                f"Candidate vector has no CRS: {candidate_vector_path}"
            )
        try:
            normalized_splits = validate_split_values(
                gdf["split"].tolist(),
                context=f"Candidate vector split column for {candidate_vector_path}",
            )
        except ValueError as exc:
            raise TerrainBaselineEvaluationError(str(exc)) from exc
        gdf = gdf.copy()
        gdf["split"] = normalized_splits
        gdf["output_vector_path"] = str(candidate_vector_path)
        frames.append(gdf)

    if not frames:
        return gpd.GeoDataFrame(
            {
                "candidate_id": [],
                "aoi_id": [],
                "split": [],
                "terrain_source_id": [],
                "source_raster_stem": [],
                "output_vector_path": [],
            },
            geometry=gpd.GeoSeries([], crs=None),
            crs=None,
        )

    return gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=frames[0].crs,
    )


def _load_baseline_rows(
    evaluation_summary: TerrainBaselineEvaluationSummary,
) -> pd.DataFrame:
    rows_path = Path(evaluation_summary.rows_output_path).expanduser().resolve()
    if not rows_path.exists():
        raise FileNotFoundError(f"Terrain baseline rows artifact not found: {rows_path}")
    try:
        rows = pd.read_csv(rows_path)
    except Exception as exc:
        raise TerrainFinalInventoryError(
            f"Terrain baseline rows artifact could not be read: {rows_path}"
        ) from exc

    required_columns = {
        "candidate_id",
        "aoi_id",
        "split",
        "terrain_source_id",
        "source_raster_stem",
        "output_vector_path",
        "pixel_count",
        "area_map_units",
        "bbox_width",
        "bbox_height",
        "bbox_aspect_ratio",
        "mean_local_relief",
        "max_local_relief",
        "mean_slope",
        "max_slope",
        "matched_label_id",
        "best_iou",
        "matched_negative_label_id",
        "best_negative_iou",
        "is_hard_negative_match",
        "target_label",
        "score",
        "predicted_label",
    }
    missing_columns = required_columns - set(rows.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        hard_negative_diagnostic_columns = {
            "matched_negative_label_id",
            "best_negative_iou",
            "is_hard_negative_match",
        }
        missing_hard_negative_columns = hard_negative_diagnostic_columns & missing_columns
        if missing_hard_negative_columns:
            missing_hard_negative = ", ".join(sorted(missing_hard_negative_columns))
            raise TerrainFinalInventoryError(
                "Terrain baseline rows artifact is missing required hard-negative "
                "diagnostic column(s) for final inventory export: "
                f"{missing_hard_negative}. Regenerate the baseline rows artifact with "
                "matched_negative_label_id, best_negative_iou, and "
                "is_hard_negative_match."
            )
        raise TerrainFinalInventoryError(
            f"Terrain baseline rows artifact is missing required column(s): {missing}."
        )

    normalized = rows.copy()
    normalized["candidate_id"] = normalized["candidate_id"].astype(str)
    normalized["aoi_id"] = normalized["aoi_id"].astype(str)
    normalized["split"] = normalized["split"].astype(str)
    normalized["terrain_source_id"] = normalized["terrain_source_id"].astype(str)
    normalized["source_raster_stem"] = normalized["source_raster_stem"].astype(str)
    normalized["output_vector_path"] = normalized["output_vector_path"].astype(str)
    _normalize_optional_text_column(normalized, "matched_label_id")
    _normalize_optional_text_column(normalized, "matched_negative_label_id")
    try:
        normalized["split"] = validate_split_values(
            normalized["split"].tolist(),
            context="Terrain baseline rows split column",
        )
    except ValueError as exc:
        raise TerrainFinalInventoryError(str(exc)) from exc

    int_columns = ["pixel_count", "is_hard_negative_match", "target_label", "predicted_label"]
    float_columns = [
        "area_map_units",
        "bbox_width",
        "bbox_height",
        "bbox_aspect_ratio",
        "mean_local_relief",
        "max_local_relief",
        "mean_slope",
        "max_slope",
        "best_iou",
        "best_negative_iou",
        "score",
    ]
    try:
        for column in int_columns:
            normalized[column] = normalized[column].astype(int)
        for column in float_columns:
            normalized[column] = normalized[column].astype(float)
    except ValueError as exc:
        raise TerrainFinalInventoryError(
            f"Terrain baseline rows artifact has invalid numeric values: {rows_path}"
        ) from exc

    return normalized


def _candidate_feature_frame(
    terrain_candidates_summary: TerrainCandidatesSummary,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            record.model_dump()
            for record in terrain_candidates_summary.records
        ]
    )


def _best_label_match(
    geometry: BaseGeometry,
    split_labels: gpd.GeoDataFrame,
) -> tuple[str | None, float]:
    if split_labels.empty:
        return None, 0.0

    intersecting = split_labels.loc[split_labels.intersects(geometry)].copy()
    if intersecting.empty:
        return None, 0.0

    best_label_id: str | None = None
    best_iou = 0.0
    geometry_area = float(geometry.area)
    for label_row in intersecting.itertuples():
        label_geometry = cast(BaseGeometry, label_row.geometry)
        intersection_area = float(geometry.intersection(label_geometry).area)
        if intersection_area <= 0.0:
            continue
        union_area = geometry_area + float(label_geometry.area) - intersection_area
        if union_area <= 0.0:
            continue
        iou = float(intersection_area / union_area)
        if iou > best_iou:
            best_iou = iou
            best_label_id = cast(str, label_row.label_id)
    return best_label_id, best_iou


def _normalize_optional_text_column(df: pd.DataFrame, column: str) -> None:
    df[column] = df[column].astype(object).where(
        pd.notna(df[column]),
        None,
    )


def _labels_by_split(
    labels: gpd.GeoDataFrame,
    *,
    class_name: str,
    review_status: str | None = None,
) -> dict[str, gpd.GeoDataFrame]:
    filtered = labels.loc[labels["class_name"] == class_name].copy()
    if review_status is not None:
        filtered = filtered.loc[filtered["review_status"] == review_status].copy()

    labels_by_split: dict[str, gpd.GeoDataFrame] = {}
    for split_value, frame in cast(Any, filtered.groupby("split")):
        labels_by_split[str(split_value)] = gpd.GeoDataFrame(
            frame,
            geometry="geometry",
            crs=filtered.crs,
        )
    return labels_by_split


def generate_terrain_candidates(
    terrain_derivatives_summary: TerrainDerivativesSummary,
    *,
    output_root: str | Path | None = None,
    relief_threshold: float = 1.0,
    min_pixels: int = 4,
) -> TerrainCandidatesSummary:
    _validate_candidate_parameters(relief_threshold, min_pixels)

    project_root = Path(terrain_derivatives_summary.project_root).expanduser().resolve()
    candidates_root = (
        _default_candidates_root(project_root)
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    summary_path = candidates_root / "terrain_candidates_summary.json"

    vector_records: list[CandidateVectorArtifactRecord] = []
    candidate_records: list[TerrainCandidateRecord] = []
    grouped_records = _group_derivative_records(terrain_derivatives_summary)

    for (
        aoi_id,
        split,
        terrain_source_id,
        source_raster_stem,
    ), derivatives in grouped_records.items():
        local_relief_record = derivatives.get("local_relief")
        slope_record = derivatives.get("slope")
        if local_relief_record is None:
            raise TerrainCandidatesError(
                "Missing required local_relief derivative for "
                f"group aoi_id={aoi_id}, terrain_source_id={terrain_source_id}, "
                f"source_raster_stem={source_raster_stem}."
            )
        if slope_record is None:
            raise TerrainCandidatesError(
                "Missing required slope derivative for "
                f"group aoi_id={aoi_id}, terrain_source_id={terrain_source_id}, "
                f"source_raster_stem={source_raster_stem}."
            )

        local_relief_path = Path(local_relief_record.output_raster_path).expanduser().resolve()
        slope_path = Path(slope_record.output_raster_path).expanduser().resolve()
        if not local_relief_path.exists():
            raise TerrainCandidatesError(
                f"Derivative raster not found: {local_relief_path}"
            )
        if not slope_path.exists():
            raise TerrainCandidatesError(f"Derivative raster not found: {slope_path}")

        candidate_vector_path = (
            candidates_root
            / "candidates"
            / aoi_id
            / terrain_source_id
            / f"{source_raster_stem}__candidates.geojson"
        )

        try:
            with rasterio.open(local_relief_path) as relief_src:
                with rasterio.open(slope_path) as slope_src:
                    _ensure_matching_raster_grids(
                        relief_src,
                        slope_src,
                        reference_label=str(local_relief_path),
                        candidate_label=str(slope_path),
                    )

                    relief = relief_src.read(1).astype(np.float32)
                    slope = slope_src.read(1).astype(np.float32)
                    relief_nodata_mask = _build_nodata_mask(
                        relief, cast(float | int | str | None, relief_src.nodata)
                    )
                    slope_nodata_mask = _build_nodata_mask(
                        slope, cast(float | int | str | None, slope_src.nodata)
                    )
                    valid_mask = ~relief_nodata_mask & ~slope_nodata_mask
                    candidate_mask = valid_mask & (relief >= np.float32(relief_threshold))

                    structure = np.ones((3, 3), dtype=np.int8)
                    labeled_components, component_count = cast(
                        tuple[np.ndarray, int],
                        label(candidate_mask, structure=structure),
                    )

                    retained_labels: list[int] = []
                    for component_id in range(1, int(component_count) + 1):
                        pixel_count = int(np.count_nonzero(labeled_components == component_id))
                        if pixel_count >= min_pixels:
                            retained_labels.append(component_id)

                    retained_component_set = set(retained_labels)
                    filtered_labels = np.where(
                        np.isin(labeled_components, retained_labels),
                        labeled_components,
                        0,
                    ).astype(np.int32)
                    component_geometries = _component_geometries(
                        filtered_labels, relief_src.transform
                    )

                    gdf = _empty_candidate_gdf(relief_src.crs)
                    for ordinal, component_id in enumerate(
                        sorted(retained_component_set), start=1
                    ):
                        component_mask = filtered_labels == component_id
                        geometry = component_geometries.get(component_id)
                        if geometry is None or geometry.is_empty:
                            continue

                        relief_values = relief[component_mask]
                        slope_values = slope[component_mask]
                        minx, miny, maxx, maxy = geometry.bounds
                        bbox_width = float(maxx - minx)
                        bbox_height = float(maxy - miny)
                        bbox_aspect_ratio = (
                            float(bbox_width / bbox_height) if bbox_height > 0.0 else 0.0
                        )
                        candidate_record = TerrainCandidateRecord(
                            candidate_id=f"{source_raster_stem}__cand_{ordinal:04d}",
                            aoi_id=aoi_id,
                            split=split,
                            terrain_source_id=terrain_source_id,
                            source_raster_stem=source_raster_stem,
                            output_vector_path=str(candidate_vector_path),
                            pixel_count=int(component_mask.sum()),
                            area_map_units=float(geometry.area),
                            bbox_width=bbox_width,
                            bbox_height=bbox_height,
                            bbox_aspect_ratio=bbox_aspect_ratio,
                            mean_local_relief=float(np.mean(relief_values)),
                            max_local_relief=float(np.max(relief_values)),
                            mean_slope=float(np.mean(slope_values)),
                            max_slope=float(np.max(slope_values)),
                        )
                        candidate_records.append(candidate_record)
                        gdf.loc[len(gdf)] = {
                            **candidate_record.model_dump(exclude={"output_vector_path"}),
                            "geometry": geometry,
                        }

                    _write_candidate_vector(gdf, candidate_vector_path)
        except RasterioError as exc:
            raise TerrainCandidatesError(
                f"Candidate generation failed for {local_relief_path}: {exc}"
            ) from exc

        vector_records.append(
            CandidateVectorArtifactRecord(
                aoi_id=aoi_id,
                split=split,
                terrain_source_id=terrain_source_id,
                source_raster_stem=source_raster_stem,
                input_raster_path=str(
                    Path(local_relief_record.input_raster_path).expanduser().resolve()
                ),
                candidate_vector_path=str(candidate_vector_path),
                candidate_count=sum(
                    1
                    for record in candidate_records
                    if record.output_vector_path == str(candidate_vector_path)
                ),
            )
        )

    return TerrainCandidatesSummary(
        project_name=terrain_derivatives_summary.project_name,
        project_root=str(project_root),
        terrain_derivatives_artifact="",
        output_root=str(candidates_root),
        output_summary_path=str(summary_path),
        total_input_groups_processed=len(grouped_records),
        total_candidate_vectors_written=len(vector_records),
        total_candidates=len(candidate_records),
        relief_threshold=relief_threshold,
        min_pixels=min_pixels,
        vectors=vector_records,
        records=candidate_records,
    )


def write_terrain_candidates_summary(
    summary: TerrainCandidatesSummary, output_path: str | Path
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return path


def prepare_terrain_review_artifacts(
    terrain_candidates_summary: TerrainCandidatesSummary,
    *,
    output_root: str | Path | None = None,
) -> TerrainReviewSummary:
    project_root = Path(terrain_candidates_summary.project_root).expanduser().resolve()
    review_root = (
        _default_review_root(project_root)
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    summary_path = review_root / "terrain_review_summary.json"
    curated_manifest_path = review_root / "terrain_curated_review_manifest.json"
    artifacts_root = review_root / "review"

    candidate_rows = [
        {
            "candidate_id": record.candidate_id,
            "aoi_id": record.aoi_id,
            "split": record.split,
            "terrain_source_id": record.terrain_source_id,
            "source_raster_stem": record.source_raster_stem,
            "output_vector_path": record.output_vector_path,
            "pixel_count": record.pixel_count,
            "area_map_units": record.area_map_units,
            "bbox_width": record.bbox_width,
            "bbox_height": record.bbox_height,
            "bbox_aspect_ratio": record.bbox_aspect_ratio,
            "mean_local_relief": record.mean_local_relief,
            "max_local_relief": record.max_local_relief,
            "mean_slope": record.mean_slope,
            "max_slope": record.max_slope,
        }
        for record in sorted(
            terrain_candidates_summary.records,
            key=lambda record: (
                -record.max_local_relief,
                -record.mean_local_relief,
                -record.pixel_count,
                record.candidate_id,
            ),
        )
    ]
    fieldnames = [
        "candidate_id",
        "aoi_id",
        "split",
        "terrain_source_id",
        "source_raster_stem",
        "output_vector_path",
        "pixel_count",
        "area_map_units",
        "bbox_width",
        "bbox_height",
        "bbox_aspect_ratio",
        "mean_local_relief",
        "max_local_relief",
        "mean_slope",
        "max_slope",
    ]

    table_records: list[ReviewTableArtifactRecord] = []
    overlay_records: list[ReviewOverlayArtifactRecord] = []
    curated_records: list[CuratedReviewArtifactRecord] = []

    overall_table_path = artifacts_root / "terrain_candidate_review_table.csv"
    _write_review_table(candidate_rows, overall_table_path, fieldnames=fieldnames)
    table_records.append(
        ReviewTableArtifactRecord(
            aoi_id="all",
            split="all",
            terrain_source_id="all",
            source_raster_stem=None,
            row_count=len(candidate_rows),
            output_table_path=str(overall_table_path),
        )
    )

    rows_by_raster: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in candidate_rows:
        group_key = (
            cast(str, row["aoi_id"]),
            cast(str, row["split"]),
            cast(str, row["terrain_source_id"]),
            cast(str, row["source_raster_stem"]),
        )
        rows_by_raster.setdefault(group_key, []).append(row)

    for (aoi_id, split, terrain_source_id, source_raster_stem), rows in rows_by_raster.items():
        table_path = (
            artifacts_root
            / split
            / aoi_id
            / terrain_source_id
            / f"{source_raster_stem}__review_table.csv"
        )
        _write_review_table(rows, table_path, fieldnames=fieldnames)
        table_records.append(
            ReviewTableArtifactRecord(
                aoi_id=aoi_id,
                split=split,
                terrain_source_id=terrain_source_id,
                source_raster_stem=source_raster_stem,
                row_count=len(rows),
                output_table_path=str(table_path),
            )
        )

    vectors_by_group: dict[tuple[str, str, str], list[CandidateVectorArtifactRecord]] = {}
    for vector_record in terrain_candidates_summary.vectors:
        group_key = (
            vector_record.aoi_id,
            vector_record.split,
            vector_record.terrain_source_id,
        )
        vectors_by_group.setdefault(group_key, []).append(vector_record)

    for (aoi_id, split, terrain_source_id), vector_records in vectors_by_group.items():
        frames: list[gpd.GeoDataFrame] = []
        total_candidates = 0
        overlay_path = (
            artifacts_root
            / split
            / aoi_id
            / terrain_source_id
            / "terrain_candidate_review_overlay.geojson"
        )
        for vector_record in vector_records:
            candidate_vector_path = Path(vector_record.candidate_vector_path).expanduser().resolve()
            if not candidate_vector_path.exists():
                raise TerrainReviewError(
                    f"Candidate vector path not found for review: {candidate_vector_path}"
                )
            try:
                gdf = gpd.read_file(candidate_vector_path)
            except Exception as exc:
                raise TerrainReviewError(
                    f"Candidate vector could not be read for review: {candidate_vector_path}"
                ) from exc
            frames.append(gdf)
            total_candidates += len(gdf)

        if frames:
            merged = gpd.GeoDataFrame(
                pd.concat(frames, ignore_index=True),
                geometry="geometry",
                crs=frames[0].crs,
            )
        else:
            merged = _empty_candidate_gdf(None)
        _write_candidate_vector(merged, overlay_path)
        overlay_records.append(
            ReviewOverlayArtifactRecord(
                aoi_id=aoi_id,
                split=split,
                terrain_source_id=terrain_source_id,
                candidate_count=total_candidates,
                output_vector_path=str(overlay_path),
            )
        )

    curated_root = artifacts_root / "curated"
    splits = sorted({cast(str, row["split"]) for row in candidate_rows})
    for split in ["all", *splits]:
        curated_tables = _curated_review_rows(candidate_rows, split=split)
        split_root = curated_root / split
        index_rows: list[dict[str, object]] = []
        for artifact_kind, rows in curated_tables.items():
            output_table_path = split_root / f"{artifact_kind}.csv"
            _write_review_table(rows, output_table_path, fieldnames=fieldnames)
            curated_records.append(
                CuratedReviewArtifactRecord(
                    artifact_kind=cast(
                        Literal["top_candidates", "large_candidates", "small_candidates"],
                        artifact_kind,
                    ),
                    split=split,
                    row_count=len(rows),
                    output_table_path=str(output_table_path),
                    sort_policy=CURATED_REVIEW_SORT_POLICY,
                )
            )
            index_rows.append(
                {
                    "artifact_kind": artifact_kind,
                    "row_count": len(rows),
                    "output_table_path": str(output_table_path),
                    "sort_policy": CURATED_REVIEW_SORT_POLICY,
                }
            )

        index_path = split_root / "curated_index.csv"
        _write_review_table(
            index_rows,
            index_path,
            fieldnames=["artifact_kind", "row_count", "output_table_path", "sort_policy"],
        )
        curated_records.append(
            CuratedReviewArtifactRecord(
                artifact_kind="index",
                split=split,
                row_count=len(index_rows),
                output_table_path=str(index_path),
                sort_policy=CURATED_REVIEW_SORT_POLICY,
            )
        )

    _write_curated_review_manifest(
        TerrainCuratedReviewManifest(
            project_name=terrain_candidates_summary.project_name,
            project_root=str(project_root),
            output_root=str(review_root),
            output_manifest_path=str(curated_manifest_path),
            total_candidate_rows=len(candidate_rows),
            total_curated_tables_written=len(curated_records),
            artifacts=curated_records,
        ),
        curated_manifest_path,
    )

    return TerrainReviewSummary(
        project_name=terrain_candidates_summary.project_name,
        project_root=str(project_root),
        terrain_candidates_artifact="",
        output_root=str(review_root),
        output_summary_path=str(summary_path),
        curated_manifest_path=str(curated_manifest_path),
        total_candidate_rows=len(candidate_rows),
        total_review_tables_written=len(table_records),
        total_review_overlays_written=len(overlay_records),
        total_curated_tables_written=len(curated_records),
        tables=table_records,
        overlays=overlay_records,
    )


def write_terrain_review_summary(summary: TerrainReviewSummary, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return path


def evaluate_terrain_baseline(
    terrain_candidates_summary: TerrainCandidatesSummary,
    *,
    normalized_labels_path: str | Path,
    output_root: str | Path | None = None,
    match_iou: float = 0.10,
    reviewed_hard_negative_weight: float = 3.0,
) -> TerrainBaselineEvaluationSummary:
    _validate_match_iou_threshold(match_iou)
    _validate_reviewed_hard_negative_weight(reviewed_hard_negative_weight)

    project_root = Path(terrain_candidates_summary.project_root).expanduser().resolve()
    evaluation_root = (
        _default_evaluation_root(project_root)
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    summary_path = evaluation_root / "terrain_baseline_evaluation_summary.json"
    rows_path = evaluation_root / "terrain_baseline_rows.csv"
    metrics_path = evaluation_root / "terrain_baseline_metrics.json"
    failure_analysis_path = evaluation_root / "terrain_baseline_failure_analysis.csv"
    prediction_root = evaluation_root / "predictions"

    labels_path = Path(normalized_labels_path).expanduser().resolve()
    if not labels_path.exists():
        raise FileNotFoundError(f"Normalized labels vector not found: {labels_path}")

    labels_gdf = _load_normalized_labels(labels_path)
    candidate_geometry_gdf = _load_candidate_geometries(terrain_candidates_summary)
    candidate_features = _candidate_feature_frame(terrain_candidates_summary)

    join_columns = [
        "candidate_id",
        "aoi_id",
        "split",
        "terrain_source_id",
        "source_raster_stem",
        "output_vector_path",
    ]
    merged = candidate_features.merge(
        pd.DataFrame(candidate_geometry_gdf[join_columns + ["geometry"]]),
        on=join_columns,
        how="left",
        validate="one_to_one",
    )
    if bool(merged["geometry"].isnull().any()):
        raise TerrainBaselineEvaluationError(
            "Failed to align candidate feature rows with candidate geometries."
        )
    candidate_rows = gpd.GeoDataFrame(merged, geometry="geometry", crs=candidate_geometry_gdf.crs)
    if candidate_rows.crs is None:
        raise TerrainBaselineEvaluationError("Candidate geometries have no CRS for evaluation.")

    labels_in_candidate_crs = labels_gdf.to_crs(candidate_rows.crs)
    positive_labels_by_split = _labels_by_split(
        labels_in_candidate_crs,
        class_name="positive_complete",
    )
    reviewed_negative_labels_by_split = _labels_by_split(
        labels_in_candidate_crs,
        class_name="negative_hard",
        review_status="reviewed",
    )

    matched_label_ids: list[str | None] = []
    best_ious: list[float] = []
    matched_negative_label_ids: list[str | None] = []
    best_negative_ious: list[float] = []
    is_hard_negative_matches: list[int] = []
    target_labels: list[int] = []
    candidate_ids = cast(list[str], candidate_rows["candidate_id"].astype(str).tolist())
    split_values = cast(list[str], candidate_rows["split"].tolist())
    geometries = cast(list[BaseGeometry], candidate_rows.geometry.tolist())
    for candidate_id, split_value, geometry in zip(
        candidate_ids,
        split_values,
        geometries,
        strict=False,
    ):
        split_positive_labels = positive_labels_by_split.get(split_value)
        if split_positive_labels is None:
            matched_label_id, best_iou = None, 0.0
        else:
            matched_label_id, best_iou = _best_label_match(
                geometry,
                split_positive_labels,
            )
        split_negative_labels = reviewed_negative_labels_by_split.get(split_value)
        if split_negative_labels is None:
            matched_negative_label_id, best_negative_iou = None, 0.0
        else:
            matched_negative_label_id, best_negative_iou = _best_label_match(
                geometry,
                split_negative_labels,
            )
        matched_label_ids.append(matched_label_id)
        best_ious.append(best_iou)
        matched_negative_label_ids.append(matched_negative_label_id)
        best_negative_ious.append(best_negative_iou)

        is_positive_match = matched_label_id is not None and best_iou >= match_iou
        is_hard_negative_match = (
            matched_negative_label_id is not None and best_negative_iou >= match_iou
        )
        if is_positive_match and is_hard_negative_match:
            raise TerrainBaselineEvaluationError(
                "Candidate matches both positive and reviewed hard-negative labels in the "
                f"same split: candidate_id={candidate_id}, split={split_value}, "
                f"positive_label_id={matched_label_id}, "
                f"negative_label_id={matched_negative_label_id}."
            )

        is_hard_negative_matches.append(int(is_hard_negative_match))
        if split_value == "val":
            target_labels.append(1 if is_positive_match else 0)
        elif is_positive_match:
            target_labels.append(1)
        elif is_hard_negative_match:
            target_labels.append(0)
        else:
            target_labels.append(0)

    candidate_rows["matched_label_id"] = matched_label_ids
    candidate_rows["best_iou"] = best_ious
    candidate_rows["matched_negative_label_id"] = matched_negative_label_ids
    candidate_rows["best_negative_iou"] = best_negative_ious
    candidate_rows["is_hard_negative_match"] = is_hard_negative_matches
    candidate_rows["target_label"] = target_labels

    try:
        candidate_rows["split"] = validate_split_values(
            candidate_rows["split"].tolist(),
            context="Candidate row split column for evaluation",
        )
    except ValueError as exc:
        raise TerrainBaselineEvaluationError(str(exc)) from exc

    train_mask = candidate_rows["split"] == "train"
    val_mask = candidate_rows["split"] == "val"
    candidate_rows["training_weight"] = 1.0
    candidate_rows.loc[
        train_mask & (candidate_rows["is_hard_negative_match"] == 1),
        "training_weight",
    ] = reviewed_hard_negative_weight
    train_rows = candidate_rows.loc[train_mask].copy()
    val_rows = candidate_rows.loc[val_mask].copy()

    if train_rows.empty:
        raise TerrainBaselineEvaluationError(
            "No train candidate rows available for training."
        )
    if val_rows.empty:
        raise TerrainBaselineEvaluationError(
            "No val candidate rows available for evaluation."
        )
    train_unique_targets = sorted(set(cast(list[int], train_rows["target_label"].tolist())))
    if len(train_unique_targets) < 2:
        raise TerrainBaselineEvaluationError(
            "Training rows must include both positive and negative targets for logistic regression."
        )

    feature_columns = [
        "pixel_count",
        "area_map_units",
        "bbox_width",
        "bbox_height",
        "bbox_aspect_ratio",
        "mean_local_relief",
        "max_local_relief",
        "mean_slope",
        "max_slope",
    ]
    model = LogisticRegression(
        random_state=BASELINE_RANDOM_STATE,
        max_iter=1000,
        class_weight=None,
    )
    model.fit(
        train_rows[feature_columns],
        train_rows["target_label"],
        sample_weight=train_rows["training_weight"],
    )

    scores = model.predict_proba(candidate_rows[feature_columns])[:, 1]
    candidate_rows["score"] = scores
    candidate_rows["predicted_label"] = (
        candidate_rows["score"] >= BASELINE_CLASSIFICATION_THRESHOLD
    ).astype(int)

    row_fields = [
        "candidate_id",
        "aoi_id",
        "split",
        "terrain_source_id",
        "source_raster_stem",
        "output_vector_path",
        "pixel_count",
        "area_map_units",
        "bbox_width",
        "bbox_height",
        "bbox_aspect_ratio",
        "mean_local_relief",
        "max_local_relief",
        "mean_slope",
        "max_slope",
        "matched_label_id",
        "best_iou",
        "matched_negative_label_id",
        "best_negative_iou",
        "is_hard_negative_match",
        "training_weight",
        "target_label",
        "score",
        "predicted_label",
    ]
    row_payload = cast(pd.DataFrame, candidate_rows[row_fields].copy())
    _write_review_table(
        _dataframe_record_rows(row_payload),
        rows_path,
        fieldnames=row_fields,
    )

    prediction_artifacts: list[TerrainBaselinePredictionArtifactRecord] = []
    for split_name in sorted(cast(set[str], set(candidate_rows["split"].tolist()))):
        split_rows = cast(
            pd.DataFrame,
            candidate_rows.loc[candidate_rows["split"] == split_name, row_fields].copy(),
        )
        prediction_path = prediction_root / f"{split_name}_predictions.csv"
        _write_review_table(
            _dataframe_record_rows(split_rows),
            prediction_path,
            fieldnames=row_fields,
        )
        prediction_artifacts.append(
            TerrainBaselinePredictionArtifactRecord(
                split=split_name,
                row_count=len(split_rows),
                positive_count=int(cast(pd.Series, split_rows["target_label"]).sum()),
                output_path=str(prediction_path),
            )
        )

    val_target = cast(list[int], val_rows["target_label"].tolist())
    val_predicted = cast(list[int], candidate_rows.loc[val_mask, "predicted_label"].tolist())
    val_scores = cast(list[float], candidate_rows.loc[val_mask, "score"].tolist())
    val_precision = (
        float(precision_score(val_target, val_predicted, zero_division=0))
        if len(val_target) > 0
        else 0.0
    )
    val_recall = (
        float(recall_score(val_target, val_predicted, zero_division=0))
        if len(val_target) > 0
        else 0.0
    )
    val_f1 = (
        float(f1_score(val_target, val_predicted, zero_division=0))
        if len(val_target) > 0
        else 0.0
    )
    val_roc_auc: float | None = None
    if len(set(val_target)) >= 2:
        val_roc_auc = float(roc_auc_score(val_target, val_scores))

    metrics = TerrainBaselineMetrics(
        threshold=BASELINE_CLASSIFICATION_THRESHOLD,
        reviewed_hard_negative_weight=reviewed_hard_negative_weight,
        train_row_count=len(train_rows),
        val_row_count=len(val_rows),
        train_positive_count=int(train_rows["target_label"].sum()),
        train_negative_count=int(len(train_rows) - int(train_rows["target_label"].sum())),
        val_positive_count=int(val_rows["target_label"].sum()),
        val_negative_count=int(len(val_rows) - int(val_rows["target_label"].sum())),
        val_precision=val_precision,
        val_recall=val_recall,
        val_f1=val_f1,
        val_roc_auc=val_roc_auc,
    )
    _write_json_artifact(metrics, metrics_path)

    failure_rows = candidate_rows.loc[val_mask].copy()
    false_positives = failure_rows.loc[
        (failure_rows["target_label"] == 0) & (failure_rows["predicted_label"] == 1)
    ].copy()
    false_negatives = failure_rows.loc[
        (failure_rows["target_label"] == 1) & (failure_rows["predicted_label"] == 0)
    ].copy()
    false_positives["error_type"] = "false_positive"
    false_negatives["error_type"] = "false_negative"
    failure_analysis = cast(
        pd.DataFrame,
        pd.concat(
        [
            false_positives.sort_values("score", ascending=False),
            false_negatives.sort_values("score"),
        ],
        ignore_index=True,
        ),
    )
    failure_fields = row_fields + ["error_type"]
    _write_review_table(
        _dataframe_record_rows(failure_analysis.reindex(columns=failure_fields)),
        failure_analysis_path,
        fieldnames=failure_fields,
    )

    return TerrainBaselineEvaluationSummary(
        project_name=terrain_candidates_summary.project_name,
        project_root=str(project_root),
        terrain_candidates_artifact="",
        normalized_labels_path=str(labels_path),
        output_root=str(evaluation_root),
        output_summary_path=str(summary_path),
        rows_output_path=str(rows_path),
        metrics_output_path=str(metrics_path),
        failure_analysis_output_path=str(failure_analysis_path),
        match_iou_threshold=match_iou,
        classification_threshold=BASELINE_CLASSIFICATION_THRESHOLD,
        prediction_artifacts=prediction_artifacts,
        metrics=metrics,
    )


def write_terrain_baseline_evaluation_summary(
    summary: TerrainBaselineEvaluationSummary, output_path: str | Path
) -> Path:
    return _write_json_artifact(summary, Path(output_path).expanduser().resolve())


def export_final_inventory(
    terrain_baseline_evaluation_summary: TerrainBaselineEvaluationSummary,
    terrain_candidates_summary: TerrainCandidatesSummary,
    *,
    output_root: str | Path | None = None,
) -> TerrainFinalInventorySummary:
    project_root = Path(terrain_candidates_summary.project_root).expanduser().resolve()
    final_root = (
        _default_final_inventory_root(project_root)
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    summary_path = final_root / "terrain_final_inventory_summary.json"
    geojson_path = final_root / "terrain_final_inventory.geojson"
    csv_path = final_root / "terrain_final_inventory.csv"

    evaluation_project_root = (
        Path(terrain_baseline_evaluation_summary.project_root).expanduser().resolve()
    )
    if evaluation_project_root != project_root:
        raise TerrainFinalInventoryError(
            "Terrain baseline evaluation and candidate artifacts must share the same project root."
        )

    evaluation_rows = _load_baseline_rows(terrain_baseline_evaluation_summary)
    try:
        candidate_geometry_gdf = _load_candidate_geometries(terrain_candidates_summary)
    except (TerrainBaselineEvaluationError, TerrainReviewError) as exc:
        raise TerrainFinalInventoryError(str(exc)) from exc

    join_columns = [
        "candidate_id",
        "aoi_id",
        "split",
        "terrain_source_id",
        "source_raster_stem",
    ]
    geometry_columns = join_columns + ["output_vector_path", "geometry"]
    candidate_geometry_frame = (
        pd.DataFrame(candidate_geometry_gdf[geometry_columns])
        if not candidate_geometry_gdf.empty
        else pd.DataFrame(columns=geometry_columns)
    )

    merged = evaluation_rows.merge(
        candidate_geometry_frame,
        on=join_columns,
        how="left",
        suffixes=("_eval", ""),
        validate="one_to_one",
    )

    missing_geometry_rows = merged.loc[merged["geometry"].isnull(), join_columns].copy()
    if not missing_geometry_rows.empty:
        missing_pairs = ", ".join(
            f"{row.candidate_id}/{row.aoi_id}/{row.split}/{row.terrain_source_id}/{row.source_raster_stem}"
            for row in missing_geometry_rows.itertuples(index=False)
        )
        raise TerrainFinalInventoryError(
            "Failed to match scored evaluation rows back to candidate geometries for: "
            f"{missing_pairs}"
        )

    if "output_vector_path_eval" in merged.columns:
        merged = merged.drop(columns=["output_vector_path_eval"])

    export_fields = [
        "candidate_id",
        "aoi_id",
        "split",
        "terrain_source_id",
        "source_raster_stem",
        "output_vector_path",
        "pixel_count",
        "area_map_units",
        "bbox_width",
        "bbox_height",
        "bbox_aspect_ratio",
        "mean_local_relief",
        "max_local_relief",
        "mean_slope",
        "max_slope",
        "matched_label_id",
        "best_iou",
        "matched_negative_label_id",
        "best_negative_iou",
        "is_hard_negative_match",
        "target_label",
        "score",
        "predicted_label",
    ]
    ordered = cast(
        pd.DataFrame,
        merged[export_fields + ["geometry"]],
    ).sort_values(
        by=["split", "aoi_id", "terrain_source_id", "source_raster_stem", "candidate_id"],
        ascending=True,
        kind="mergesort",
    )
    _normalize_optional_text_column(ordered, "matched_label_id")
    _normalize_optional_text_column(ordered, "matched_negative_label_id")
    inventory_gdf = gpd.GeoDataFrame(
        ordered,
        geometry="geometry",
        crs=candidate_geometry_gdf.crs,
    )
    if inventory_gdf.crs is None and not inventory_gdf.empty:
        raise TerrainFinalInventoryError("Candidate geometries have no CRS for final inventory.")

    _write_candidate_vector(inventory_gdf, geojson_path)
    _write_review_table(
        _dataframe_record_rows(pd.DataFrame(inventory_gdf.drop(columns="geometry"))),
        csv_path,
        fieldnames=export_fields,
    )

    split_counts_series = cast(pd.Series, inventory_gdf["split"].value_counts(sort=False))
    split_counts = {
        str(split): int(count)
        for split, count in sorted(split_counts_series.to_dict().items())
    }
    predicted_positive_count = int(cast(pd.Series, inventory_gdf["predicted_label"]).sum())
    total_exported_features = int(len(inventory_gdf))
    predicted_negative_count = total_exported_features - predicted_positive_count
    summary_records_df = pd.DataFrame(inventory_gdf[export_fields]).copy()
    _normalize_optional_text_column(summary_records_df, "matched_label_id")
    _normalize_optional_text_column(summary_records_df, "matched_negative_label_id")

    summary = TerrainFinalInventorySummary(
        project_name=terrain_candidates_summary.project_name,
        project_root=str(project_root),
        terrain_baseline_evaluation_artifact="",
        terrain_candidates_artifact="",
        output_root=str(final_root),
        output_summary_path=str(summary_path),
        geojson_output_path=str(geojson_path),
        csv_output_path=str(csv_path),
        total_exported_features=total_exported_features,
        split_counts=split_counts,
        predicted_positive_count=predicted_positive_count,
        predicted_negative_count=predicted_negative_count,
        classification_threshold=terrain_baseline_evaluation_summary.classification_threshold,
        match_iou_threshold=terrain_baseline_evaluation_summary.match_iou_threshold,
        artifacts=[
            FinalInventoryArtifactRecord(
                artifact_kind="geojson",
                output_path=str(geojson_path),
                row_count=total_exported_features,
            ),
            FinalInventoryArtifactRecord(
                artifact_kind="csv",
                output_path=str(csv_path),
                row_count=total_exported_features,
            ),
        ],
        records=[
            FinalInventoryFeatureRecord.model_validate(record)
            for record in _dataframe_record_rows(summary_records_df)
        ],
    )
    return summary


def write_final_inventory_summary(
    summary: TerrainFinalInventorySummary, output_path: str | Path
) -> Path:
    return _write_json_artifact(summary, Path(output_path).expanduser().resolve())
