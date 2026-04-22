from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError


class SourceAccess(BaseModel):
    method: str
    auth_required: bool = False


class SourceLocal(BaseModel):
    expected_path: str | None = None
    expected_root: str | None = None


class DataSource(BaseModel):
    id: str
    role: str
    enabled: bool = True
    type: str
    format: str
    provider: str
    access: SourceAccess
    local: SourceLocal = Field(default_factory=SourceLocal)
    notes: list[str] = Field(default_factory=list)


class DataManifest(BaseModel):
    version: int
    project: dict[str, Any]
    sources: list[DataSource]
    defaults: dict[str, Any] = Field(default_factory=dict)
    requirements: dict[str, Any] = Field(default_factory=dict)
    non_goals: list[str] = Field(default_factory=list)


class AoiRecord(BaseModel):
    id: str
    split: str
    geometry_path: str
    enabled: bool = True
    notes: list[str] = Field(default_factory=list)


class AoiManifest(BaseModel):
    version: int
    aoi_sets: list[AoiRecord]
    rules: dict[str, Any] = Field(default_factory=dict)


SUPPORTED_TERRAIN_SOURCE_IDS = {
    "nc_dem_10m_opentopography",
    "nc_lidar_1m_validation",
    "copernicus_dem_30m_fallback",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Manifest not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a mapping: {file_path}")
    return data


def load_data_manifest(path: str | Path) -> DataManifest:
    raw = load_yaml(path)
    return DataManifest.model_validate(raw)


def load_aoi_manifest(path: str | Path) -> AoiManifest:
    raw = load_yaml(path)
    return AoiManifest.model_validate(raw)


def summarize_data_manifest(manifest: DataManifest) -> dict[str, Any]:
    enabled_sources = [src for src in manifest.sources if src.enabled]
    return {
        "project": manifest.project.get("name", "unknown"),
        "total_sources": len(manifest.sources),
        "enabled_sources": len(enabled_sources),
        "source_ids": [src.id for src in enabled_sources],
    }


def summarize_aoi_manifest(manifest: AoiManifest) -> dict[str, Any]:
    enabled_aois = [aoi for aoi in manifest.aoi_sets if aoi.enabled]
    split_counts: dict[str, int] = {}
    for aoi in enabled_aois:
        split_counts[aoi.split] = split_counts.get(aoi.split, 0) + 1
    return {
        "total_aois": len(manifest.aoi_sets),
        "enabled_aois": len(enabled_aois),
        "split_counts": split_counts,
    }


def format_validation_error(exc: ValidationError) -> str:
    return exc.json(indent=2)


def get_enabled_aois(manifest: AoiManifest) -> list[AoiRecord]:
    return [aoi for aoi in manifest.aoi_sets if aoi.enabled]


def get_enabled_terrain_sources(manifest: DataManifest) -> list[DataSource]:
    return [
        source
        for source in manifest.sources
        if source.enabled and source.id in SUPPORTED_TERRAIN_SOURCE_IDS
    ]


def validate_supported_terrain_sources(manifest: DataManifest) -> None:
    unsupported_enabled_terrain = [
        source.id
        for source in manifest.sources
        if source.enabled
        and source.type == "raster_dem"
        and source.id not in SUPPORTED_TERRAIN_SOURCE_IDS
    ]
    if unsupported_enabled_terrain:
        supported = ", ".join(sorted(SUPPORTED_TERRAIN_SOURCE_IDS))
        unsupported = ", ".join(sorted(unsupported_enabled_terrain))
        raise ValueError(
            f"Unsupported enabled terrain source(s): {unsupported}. "
            f"Supported terrain sources for this stage: {supported}."
        )
