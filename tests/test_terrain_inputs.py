from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import yaml
from shapely.geometry import Polygon
from typer.testing import CliRunner

from cbd.cli import app

runner = CliRunner()


def _write_test_geojson(path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {"name": ["x"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:4326",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path)


def _write_manifest_pair(tmp_path: Path) -> tuple[Path, Path]:
    data_manifest = {
        "version": 2,
        "project": {"name": "carolina-bays-mvp"},
        "sources": [
            {
                "id": "nc_dem_10m_opentopography",
                "role": "primary_terrain",
                "enabled": True,
                "type": "raster_dem",
                "format": "geotiff_or_cloud_optimized_geotiff",
                "provider": "opentopography",
                "access": {"method": "local_assets_or_scripted_fetch", "auth_required": False},
                "local": {"expected_root": "data/raw/dem/nc_10m"},
                "notes": [],
            },
            {
                "id": "nc_lidar_1m_validation",
                "role": "validation_terrain",
                "enabled": True,
                "type": "raster_dem",
                "format": "geotiff_or_cloud_optimized_geotiff",
                "provider": "opentopography",
                "access": {"method": "local_assets_or_scripted_fetch", "auth_required": False},
                "local": {"expected_root": "data/raw/dem/nc_1m_validation"},
                "notes": [],
            },
            {
                "id": "copernicus_dem_30m_fallback",
                "role": "fallback_terrain",
                "enabled": True,
                "type": "raster_dem",
                "format": "geotiff_or_cloud_optimized_geotiff",
                "provider": "copernicus",
                "access": {"method": "local_assets_or_scripted_fetch", "auth_required": False},
                "local": {"expected_root": "data/raw/dem/copernicus_30m"},
                "notes": [],
            },
            {
                "id": "sentinel2_sr",
                "role": "auxiliary_optical_context",
                "enabled": True,
                "type": "raster_multispectral",
                "format": "geotiff_or_stac_assets",
                "provider": "copernicus_or_prepared_local_assets",
                "access": {"method": "stac_or_prepared_local_assets", "auth_required": False},
                "local": {"expected_root": "data/raw/imagery/sentinel2"},
                "notes": [],
            },
        ],
        "defaults": {},
        "requirements": {"primary_terrain_source": "nc_dem_10m_opentopography"},
        "non_goals": [],
    }

    aoi_manifest = {
        "version": 1,
        "aoi_sets": [
            {
                "id": "train_aoi_01",
                "split": "train",
                "geometry_path": "data/raw/aoi/train_aoi_01.geojson",
                "enabled": True,
                "notes": [],
            },
            {
                "id": "val_aoi_01",
                "split": "val",
                "geometry_path": "data/raw/aoi/val_aoi_01.geojson",
                "enabled": True,
                "notes": [],
            },
        ],
        "rules": {"geographic_holdout": True},
    }

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    data_path = manifests_dir / "data_manifest.yaml"
    aoi_path = manifests_dir / "aoi_manifest.yaml"
    data_path.write_text(yaml.safe_dump(data_manifest), encoding="utf-8")
    aoi_path.write_text(yaml.safe_dump(aoi_manifest), encoding="utf-8")
    return data_path, aoi_path


def _touch_raster(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"TIFF")


def test_resolve_terrain_inputs_command_success(tmp_path: Path) -> None:
    data_path, aoi_path = _write_manifest_pair(tmp_path)
    _write_test_geojson(tmp_path / "data" / "raw" / "aoi" / "train_aoi_01.geojson")
    _write_test_geojson(tmp_path / "data" / "raw" / "aoi" / "val_aoi_01.geojson")
    _touch_raster(tmp_path / "data" / "raw" / "dem" / "nc_10m" / "tile_001.tif")
    _touch_raster(tmp_path / "data" / "raw" / "dem" / "nc_1m_validation" / "tile_101.tif")
    _touch_raster(tmp_path / "data" / "raw" / "dem" / "copernicus_30m" / "tile_201.tif")

    result = runner.invoke(app, ["resolve-terrain-inputs", str(data_path), str(aoi_path)])

    assert result.exit_code == 0
    assert "Terrain inputs resolved successfully" in result.stdout

    artifact_path = (
        tmp_path / "outputs" / "interim" / "terrain" / "terrain_input_resolution.json"
    )
    assert artifact_path.exists()

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["enabled_aoi_count"] == 2
    assert payload["enabled_terrain_source_count"] == 3
    assert len(payload["terrain_sources"]) == 3
    assert len(payload["associations"]) == 6
    assert {source["id"] for source in payload["terrain_sources"]} == {
        "nc_dem_10m_opentopography",
        "nc_lidar_1m_validation",
        "copernicus_dem_30m_fallback",
    }


def test_resolve_terrain_inputs_command_fails_for_missing_terrain_path(tmp_path: Path) -> None:
    data_path, aoi_path = _write_manifest_pair(tmp_path)
    _write_test_geojson(tmp_path / "data" / "raw" / "aoi" / "train_aoi_01.geojson")
    _write_test_geojson(tmp_path / "data" / "raw" / "aoi" / "val_aoi_01.geojson")
    _touch_raster(tmp_path / "data" / "raw" / "dem" / "nc_10m" / "tile_001.tif")
    _touch_raster(tmp_path / "data" / "raw" / "dem" / "copernicus_30m" / "tile_201.tif")

    result = runner.invoke(app, ["resolve-terrain-inputs", str(data_path), str(aoi_path)])

    assert result.exit_code == 4
    assert "nc_lidar_1m_validation" in result.stdout
    assert "Terrain path not found" in result.stdout


def test_resolve_terrain_inputs_command_fails_for_missing_aoi_geometry(tmp_path: Path) -> None:
    data_path, aoi_path = _write_manifest_pair(tmp_path)
    _write_test_geojson(tmp_path / "data" / "raw" / "aoi" / "train_aoi_01.geojson")
    _touch_raster(tmp_path / "data" / "raw" / "dem" / "nc_10m" / "tile_001.tif")
    _touch_raster(tmp_path / "data" / "raw" / "dem" / "nc_1m_validation" / "tile_101.tif")
    _touch_raster(tmp_path / "data" / "raw" / "dem" / "copernicus_30m" / "tile_201.tif")

    result = runner.invoke(app, ["resolve-terrain-inputs", str(data_path), str(aoi_path)])

    assert result.exit_code == 4
    assert "val_aoi_01" in result.stdout
    assert "AOI geometry path not found" in result.stdout


def test_resolve_terrain_inputs_command_fails_for_empty_terrain_root(tmp_path: Path) -> None:
    data_path, aoi_path = _write_manifest_pair(tmp_path)
    _write_test_geojson(tmp_path / "data" / "raw" / "aoi" / "train_aoi_01.geojson")
    _write_test_geojson(tmp_path / "data" / "raw" / "aoi" / "val_aoi_01.geojson")
    _touch_raster(tmp_path / "data" / "raw" / "dem" / "nc_10m" / "tile_001.tif")
    (tmp_path / "data" / "raw" / "dem" / "nc_1m_validation").mkdir(parents=True, exist_ok=True)
    _touch_raster(tmp_path / "data" / "raw" / "dem" / "copernicus_30m" / "tile_201.tif")

    result = runner.invoke(app, ["resolve-terrain-inputs", str(data_path), str(aoi_path)])

    assert result.exit_code == 4
    assert "nc_lidar_1m_validation" in result.stdout
    assert "No TIFF rasters found" in result.stdout
