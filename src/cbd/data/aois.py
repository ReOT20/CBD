from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from cbd.data.common import clean_geometries, ensure_crs, read_vector, write_vector


def normalize_aoi(
    input_path: str | Path,
    output_path: str | Path,
    target_crs: str = "EPSG:4326",
    aoi_id: str = "aoi_0001",
    split: str = "train",
) -> Path:
    gdf = read_vector(input_path)
    gdf = ensure_crs(gdf, target_crs)
    gdf = clean_geometries(gdf)

    if gdf.empty:
        raise ValueError("Input AOI dataset is empty after geometry cleaning.")

    normalized = gpd.GeoDataFrame(geometry=gdf.geometry.copy(), crs=gdf.crs)
    normalized["aoi_id"] = [f"{aoi_id}_{i:04d}" for i in range(len(normalized))]
    normalized["split"] = split
    normalized["notes"] = ""

    return write_vector(normalized, output_path)

