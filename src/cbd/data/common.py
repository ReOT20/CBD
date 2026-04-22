from __future__ import annotations

from pathlib import Path
from typing import cast

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
    mask = gdf.geometry.notnull() & gdf.is_valid & ~gdf.geometry.is_empty
    filtered = gdf.loc[mask].copy()
    result = gpd.GeoDataFrame(filtered, geometry="geometry", crs=gdf.crs)
    return cast(gpd.GeoDataFrame, result.reset_index(drop=True))


def write_vector(gdf: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path)
    return path
