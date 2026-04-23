from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from cbd.data.common import (
    clean_geometries,
    ensure_crs,
    read_vector,
    validate_split_value,
    validate_split_values,
    write_vector,
)
from cbd.manifests import AoiManifest, get_enabled_aois

DEFAULT_LABEL_CLASS = "positive_complete"
DEFAULT_REVIEW_STATUS = "seed"
DEFAULT_HARD_NEGATIVE_CLASS = "negative_hard"


def _normalize_source_attr_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        value = cast(Any, value).item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _source_record_attrs(gdf: gpd.GeoDataFrame) -> list[str]:
    attribute_columns = [column for column in gdf.columns if column != "geometry"]
    rows: list[str] = []
    records = cast(Any, gdf[attribute_columns]).to_dict(orient="records")
    for record in records:
        normalized = {
            key: _normalize_source_attr_value(value)
            for key, value in record.items()
        }
        rows.append(json.dumps(normalized, sort_keys=True))
    return rows


def _seed_parent_source_record(row: object, available_columns: set[str]) -> str:
    row_any = cast(Any, row)
    parts = [f"split={row_any.split}"]
    for column in ["aoi_id", "terrain_source_id", "source_raster_stem"]:
        if column in available_columns:
            parts.append(f"{column}={getattr(row_any, column)}")
    parts.append(f"candidate_id={row_any.candidate_id}")
    return ";".join(parts)


def normalize_labels(
    input_path: str | Path,
    output_path: str | Path,
    target_crs: str = "EPSG:4326",
    source_id: str = "carolina_bays_labels",
    split: str = "train",
) -> Path:
    split = validate_split_value(split, context="Label split")
    gdf = read_vector(input_path)
    gdf = ensure_crs(gdf, target_crs)
    gdf = clean_geometries(gdf)

    if gdf.empty:
        raise ValueError("Input label dataset is empty after geometry cleaning.")

    normalized = gpd.GeoDataFrame(geometry=gdf.geometry.copy(), crs=gdf.crs)

    normalized["label_id"] = [f"{source_id}_{i:06d}" for i in range(len(normalized))]
    normalized["source_record_id"] = [f"{source_id}__src_{i:06d}" for i in range(len(normalized))]
    normalized["source_record_attrs"] = _source_record_attrs(gdf)
    normalized["class_name"] = DEFAULT_LABEL_CLASS
    normalized["source_id"] = source_id
    normalized["split"] = split
    normalized["review_status"] = DEFAULT_REVIEW_STATUS
    normalized["notes"] = ""

    return write_vector(normalized, output_path)


def _infer_project_root(manifest_path: str | Path) -> Path:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest_dir = manifest_file.parent
    if manifest_dir.name == "manifests":
        return manifest_dir.parent.resolve()
    return manifest_dir.resolve()


def _load_aoi_geometry_index(
    aoi_manifest: AoiManifest,
    *,
    aoi_manifest_path: str | Path,
    target_crs: str,
) -> list[tuple[str, BaseGeometry]]:
    project_root = _infer_project_root(aoi_manifest_path)
    indexed_aois: list[tuple[str, BaseGeometry]] = []
    for aoi in get_enabled_aois(aoi_manifest):
        split = validate_split_value(aoi.split, context=f"AOI manifest split for '{aoi.id}'")
        geometry_path = (project_root / aoi.geometry_path).expanduser().resolve()
        aoi_gdf = read_vector(geometry_path)
        aoi_gdf = ensure_crs(aoi_gdf, target_crs)
        aoi_gdf = clean_geometries(aoi_gdf)
        if aoi_gdf.empty:
            raise ValueError(f"AOI dataset is empty after geometry cleaning: {geometry_path}")
        geometry = aoi_gdf.union_all()
        if geometry.is_empty:
            raise ValueError(f"AOI dataset is empty after union: {geometry_path}")
        indexed_aois.append((split, cast(BaseGeometry, geometry)))
    return indexed_aois


def normalize_labels_by_aoi(
    input_path: str | Path,
    output_path: str | Path,
    *,
    aoi_manifest: AoiManifest,
    aoi_manifest_path: str | Path,
    target_crs: str = "EPSG:4326",
    source_id: str = "carolina_bays_labels",
) -> Path:
    gdf = read_vector(input_path)
    gdf = ensure_crs(gdf, target_crs)
    gdf = clean_geometries(gdf)

    if gdf.empty:
        raise ValueError("Input label dataset is empty after geometry cleaning.")

    aoi_index = _load_aoi_geometry_index(
        aoi_manifest,
        aoi_manifest_path=aoi_manifest_path,
        target_crs=target_crs,
    )

    kept_geometries: list[BaseGeometry] = []
    source_record_ids: list[str] = []
    source_record_attrs: list[str] = []
    splits: list[str] = []
    source_attrs_by_row = _source_record_attrs(gdf)
    for row_index, geometry in enumerate(gdf.geometry.tolist()):
        label_geometry = cast(BaseGeometry, geometry)
        matching_splits = sorted(
            split
            for split, aoi_geometry in aoi_index
            if label_geometry.intersects(aoi_geometry)
        )
        if not matching_splits:
            continue
        if len(set(matching_splits)) > 1:
            raise ValueError(
                "A label intersects AOIs from multiple splits. "
                "Real-data smoke labels must map to exactly one split."
            )
        kept_geometries.append(label_geometry)
        source_record_ids.append(f"{source_id}__src_{row_index:06d}")
        source_record_attrs.append(source_attrs_by_row[row_index])
        splits.append(matching_splits[0])

    if not kept_geometries:
        raise ValueError("No labels intersect the enabled AOIs in the AOI manifest.")

    normalized = gpd.GeoDataFrame(geometry=kept_geometries, crs=target_crs)
    normalized["label_id"] = [f"{source_id}_{i:06d}" for i in range(len(normalized))]
    normalized["source_record_id"] = source_record_ids
    normalized["source_record_attrs"] = source_record_attrs
    normalized["class_name"] = DEFAULT_LABEL_CLASS
    normalized["source_id"] = source_id
    normalized["split"] = splits
    normalized["review_status"] = DEFAULT_REVIEW_STATUS
    normalized["notes"] = ""

    return write_vector(normalized, output_path)


def seed_hard_negative_labels(
    input_path: str | Path,
    output_path: str | Path,
    *,
    split: str = "val",
    source_id: str = "hard_negatives_seed",
    target_crs: str = "EPSG:4326",
    top_n: int = 25,
    min_score: float = 0.05,
    min_pixels: int = 50,
    min_max_local_relief: float = 2.0,
) -> Path:
    split = validate_split_value(split, context="Hard-negative seed split")
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}.")
    if min_score < 0.0 or min_score > 1.0:
        raise ValueError(f"min_score must be between 0 and 1, got {min_score}.")
    if min_pixels < 1:
        raise ValueError(f"min_pixels must be >= 1, got {min_pixels}.")
    if min_max_local_relief < 0.0:
        raise ValueError(
            f"min_max_local_relief must be >= 0, got {min_max_local_relief}."
        )

    gdf = read_vector(input_path)
    gdf = ensure_crs(gdf, target_crs)
    gdf = clean_geometries(gdf)

    required_columns = {
        "candidate_id",
        "split",
        "target_label",
        "score",
        "pixel_count",
        "max_local_relief",
    }
    missing_columns = required_columns - set(gdf.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Input inventory is missing required column(s): {missing}."
        )
    validate_split_values(gdf["split"].tolist(), context="Input inventory split column")

    candidate_rows = gdf.loc[
        (gdf["split"] == split)
        & (gdf["target_label"] == 0)
        & (gdf["score"] >= min_score)
        & (gdf["pixel_count"] >= min_pixels)
        & (gdf["max_local_relief"] >= min_max_local_relief)
    ].copy()

    if candidate_rows.empty:
        raise ValueError(
            "No candidate rows matched the hard-negative seed criteria."
        )

    candidate_rows = candidate_rows.sort_values(
        by=["score", "max_local_relief", "pixel_count"],
        ascending=[False, False, False],
        kind="mergesort",
    ).head(top_n)
    available_columns = set(candidate_rows.columns)

    normalized = gpd.GeoDataFrame(geometry=candidate_rows.geometry.copy(), crs=gdf.crs)
    normalized["label_id"] = [f"{source_id}_{i:06d}" for i in range(len(normalized))]
    normalized["class_name"] = DEFAULT_HARD_NEGATIVE_CLASS
    normalized["source_id"] = source_id
    normalized["split"] = split
    normalized["review_status"] = DEFAULT_REVIEW_STATUS
    normalized["notes"] = [
        (
            f"seeded_from_candidate_id={row.candidate_id};"
            f"score={float(row.score):.6f};"
            f"pixel_count={int(row.pixel_count)};"
            f"max_local_relief={float(row.max_local_relief):.6f}"
        )
        for row in candidate_rows.itertuples()
    ]
    normalized["error_category"] = "terrain_large_steep_nonbay"
    normalized["parent_source_record"] = cast(
        list[str],
        [
            _seed_parent_source_record(row, available_columns)
            for row in candidate_rows.itertuples(index=False)
        ],
    )

    return write_vector(normalized, output_path)
