from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

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
