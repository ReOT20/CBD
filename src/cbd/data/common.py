from __future__ import annotations

from pathlib import Path

import geopandas as gpd


def read_vector(path: str | Path) -> gpd.GeoDataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Vector file not found: {file_path}")
    return gpd.read_file(file_path)


def ensure_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(target_crs)
    return gdf.to_crs(target_crs)


def clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = gdf[gdf.geometry.notnull()].copy()
    result = result[result.is_valid].copy()
    result = result[~result.geometry.is_empty].copy()
    return result.reset_index(drop=True)


def write_vector(gdf: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path)
    return path
