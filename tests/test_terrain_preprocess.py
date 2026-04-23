from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon
from typer.testing import CliRunner

from cbd.cli import app

runner = CliRunner()


def _write_test_geojson(path: Path, polygon: Polygon, crs: str = "EPSG:4326") -> None:
    gdf = gpd.GeoDataFrame({"name": ["x"]}, geometry=[polygon], crs=crs)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path)


def _write_test_raster(
    path: Path,
    *,
    crs: str = "EPSG:4326",
    nodata: float | None = -9999.0,
    transform=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if transform is None:
        transform = from_origin(0.0, 4.0, 1.0, 1.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(
            np.array(
                [[
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]],
                dtype="float32",
            )
        )


def _write_resolution_artifact(
    tmp_path: Path,
    *,
    aoi_path: Path,
    raster_path: Path,
    project_root: Path | None = None,
) -> Path:
    root = tmp_path if project_root is None else project_root
    artifact = {
        "project_name": "carolina-bays-mvp",
        "project_root": str(root.resolve()),
        "data_manifest_path": str((tmp_path / "manifests" / "data_manifest.yaml").resolve()),
        "aoi_manifest_path": str((tmp_path / "manifests" / "aoi_manifest.yaml").resolve()),
        "enabled_aoi_count": 1,
        "enabled_terrain_source_count": 1,
        "aois": [
            {
                "id": "train_aoi_01",
                "split": "train",
                "geometry_path": str(aoi_path.resolve()),
                "notes": [],
            }
        ],
        "terrain_sources": [
            {
                "id": "nc_dem_10m_opentopography",
                "role": "primary_terrain",
                "provider": "opentopography",
                "source_type": "raster_dem",
                "source_format": "geotiff_or_cloud_optimized_geotiff",
                "local_reference": "data/raw/dem/nc_10m",
                "path_kind": "expected_root",
                "resolved_path": str(raster_path.parent.resolve()),
                "raster_files": [str(raster_path.resolve())],
                "raster_count": 1,
            }
        ],
        "associations": [
            {
                "aoi_id": "train_aoi_01",
                "split": "train",
                "geometry_path": str(aoi_path.resolve()),
                "terrain_source_id": "nc_dem_10m_opentopography",
                "terrain_role": "primary_terrain",
                "terrain_path": str(raster_path.parent.resolve()),
                "raster_count": 1,
            }
        ],
    }
    artifact_path = tmp_path / "outputs" / "interim" / "terrain" / "terrain_input_resolution.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact_path


def test_preprocess_terrain_command_success(tmp_path: Path) -> None:
    aoi_path = tmp_path / "data" / "raw" / "aoi" / "train_aoi_01.geojson"
    raster_path = tmp_path / "data" / "raw" / "dem" / "nc_10m" / "tile_001.tif"
    _write_test_geojson(aoi_path, Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))
    _write_test_raster(raster_path)
    artifact_path = _write_resolution_artifact(tmp_path, aoi_path=aoi_path, raster_path=raster_path)

    result = runner.invoke(app, ["preprocess-terrain", str(artifact_path)])

    assert result.exit_code == 0
    assert "Terrain preprocessing completed successfully" in result.stdout

    summary_path = (
        tmp_path / "outputs" / "interim" / "terrain" / "terrain_preprocessing_summary.json"
    )
    output_raster = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "preprocessed"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001.tif"
    )
    assert summary_path.exists()
    assert output_raster.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["total_raster_outputs_written"] == 1
    assert payload["records"][0]["status"] == "written"
    assert payload["records"][0]["output_raster_path"] == str(output_raster.resolve())


def test_preprocess_terrain_command_skips_non_intersection(tmp_path: Path) -> None:
    aoi_path = tmp_path / "data" / "raw" / "aoi" / "train_aoi_01.geojson"
    raster_path = tmp_path / "data" / "raw" / "dem" / "nc_10m" / "tile_001.tif"
    _write_test_geojson(aoi_path, Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]))
    _write_test_raster(raster_path)
    artifact_path = _write_resolution_artifact(tmp_path, aoi_path=aoi_path, raster_path=raster_path)

    result = runner.invoke(app, ["preprocess-terrain", str(artifact_path)])

    assert result.exit_code == 0
    summary_path = (
        tmp_path / "outputs" / "interim" / "terrain" / "terrain_preprocessing_summary.json"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["total_raster_outputs_written"] == 0
    assert payload["records"][0]["status"] == "skipped"
    assert payload["records"][0]["skip_reason"] == "no_intersection"


def test_preprocess_terrain_command_reprojects_aoi_to_raster_crs(tmp_path: Path) -> None:
    aoi_path = tmp_path / "data" / "raw" / "aoi" / "train_aoi_01.geojson"
    raster_path = tmp_path / "data" / "raw" / "dem" / "nc_10m" / "tile_001.tif"
    _write_test_geojson(aoi_path, Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), crs="EPSG:3857")
    _write_test_raster(raster_path, crs="EPSG:4326")
    artifact_path = _write_resolution_artifact(tmp_path, aoi_path=aoi_path, raster_path=raster_path)

    result = runner.invoke(app, ["preprocess-terrain", str(artifact_path)])

    assert result.exit_code == 0
    assert "Terrain preprocessing completed successfully" in result.stdout

    summary_path = (
        tmp_path / "outputs" / "interim" / "terrain" / "terrain_preprocessing_summary.json"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["total_raster_outputs_written"] == 1
    assert payload["records"][0]["status"] == "written"
    assert payload["records"][0]["source_crs"] == "EPSG:4326"
    assert payload["records"][0]["output_crs"] == "EPSG:4326"


def test_preprocess_terrain_command_fails_for_missing_raster(tmp_path: Path) -> None:
    aoi_path = tmp_path / "data" / "raw" / "aoi" / "train_aoi_01.geojson"
    raster_path = tmp_path / "data" / "raw" / "dem" / "nc_10m" / "tile_001.tif"
    _write_test_geojson(aoi_path, Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))
    artifact_path = _write_resolution_artifact(tmp_path, aoi_path=aoi_path, raster_path=raster_path)

    result = runner.invoke(app, ["preprocess-terrain", str(artifact_path)])

    assert result.exit_code == 3
    assert "Input raster path not found" in result.stdout
