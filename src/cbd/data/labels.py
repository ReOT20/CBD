from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from cbd.data.common import clean_geometries, ensure_crs, read_vector, write_vector


DEFAULT_LABEL_CLASS = "positive_complete"
DEFAULT_REVIEW_STATUS = "seed"


def normalize_labels(
    input_path: str | Path,
    output_path: str | Path,
    target_crs: str = "EPSG:4326",
    source_id: str = "carolina_bays_labels",
    split: str = "train",
) -> Path:
    gdf = read_vector(input_path)
    gdf = ensure_crs(gdf, target_crs)
    gdf = clean_geometries(gdf)

    if gdf.empty:
        raise ValueError("Input label dataset is empty after geometry cleaning.")

    normalized = gpd.GeoDataFrame(geometry=gdf.geometry.copy(), crs=gdf.crs)

    normalized["label_id"] = [f"{source_id}_{i:06d}" for i in range(len(normalized))]
    normalized["class_name"] = DEFAULT_LABEL_CLASS
    normalized["source_id"] = source_id
    normalized["split"] = split
    normalized["review_status"] = DEFAULT_REVIEW_STATUS
    normalized["notes"] = ""

    return write_vector(normalized, output_path)
