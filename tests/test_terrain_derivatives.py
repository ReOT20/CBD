from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from typer.testing import CliRunner

from cbd.cli import app

runner = CliRunner()


def _write_test_raster(
    path: Path,
    *,
    count: int = 1,
    nodata: float | None = -9999.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype="float32",
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=count,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dst:
        for band_index in range(1, count + 1):
            dst.write(data, band_index)


def _write_preprocessing_artifact(
    tmp_path: Path,
    *,
    raster_path: Path,
) -> Path:
    artifact = {
        "project_name": "carolina-bays-mvp",
        "project_root": str(tmp_path.resolve()),
        "terrain_resolution_artifact": str(
            (
                tmp_path
                / "outputs"
                / "interim"
                / "terrain"
                / "terrain_input_resolution.json"
            ).resolve()
        ),
        "output_root": str((tmp_path / "outputs" / "interim" / "terrain").resolve()),
        "output_summary_path": str(
            (
                tmp_path
                / "outputs"
                / "interim"
                / "terrain"
                / "terrain_preprocessing_summary.json"
            ).resolve()
        ),
        "total_aois_processed": 1,
        "total_terrain_sources_processed": 1,
        "total_raster_outputs_written": 1,
        "records": [
            {
                "aoi_id": "train_aoi_01",
                "split": "train",
                "terrain_source_id": "nc_dem_10m_opentopography",
                "input_raster_path": str(raster_path.resolve()),
                "output_raster_path": str(raster_path.resolve()),
                "status": "written",
                "skip_reason": None,
                "source_crs": "EPSG:4326",
                "output_crs": "EPSG:4326",
                "width": 4,
                "height": 4,
                "count": 1,
                "dtype": "float32",
                "nodata": -9999.0,
                "bounds": [0.0, 0.0, 4.0, 4.0],
            }
        ],
    }
    artifact_path = (
        tmp_path / "outputs" / "interim" / "terrain" / "terrain_preprocessing_summary.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact_path


def test_derive_terrain_features_command_success(tmp_path: Path) -> None:
    raster_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "preprocessed"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001.tif"
    )
    _write_test_raster(raster_path)
    artifact_path = _write_preprocessing_artifact(tmp_path, raster_path=raster_path)

    result = runner.invoke(app, ["derive-terrain-features", str(artifact_path)])

    assert result.exit_code == 0
    assert "Terrain derivatives completed successfully" in result.stdout

    summary_path = (
        tmp_path / "outputs" / "interim" / "terrain" / "terrain_derivatives_summary.json"
    )
    slope_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "derivatives"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__slope.tif"
    )
    relief_path = slope_path.with_name("tile_001__local_relief.tif")

    assert summary_path.exists()
    assert slope_path.exists()
    assert relief_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["total_input_rasters_processed"] == 1
    assert payload["total_derivative_rasters_written"] == 2
    assert {record["derivative_name"] for record in payload["records"]} == {
        "slope",
        "local_relief",
    }


def test_derive_terrain_features_preserves_nodata(tmp_path: Path) -> None:
    raster_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "preprocessed"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001.tif"
    )
    _write_test_raster(raster_path)
    with rasterio.open(raster_path, "r+") as dst:
        data = dst.read(1)
        data[0, 0] = -9999.0
        dst.write(data, 1)
    artifact_path = _write_preprocessing_artifact(tmp_path, raster_path=raster_path)

    result = runner.invoke(app, ["derive-terrain-features", str(artifact_path)])

    assert result.exit_code == 0
    slope_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "derivatives"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__slope.tif"
    )
    with rasterio.open(slope_path) as src:
        data = src.read(1)
        assert src.nodata == -9999.0
        assert data[0, 0] == -9999.0


def test_derive_terrain_features_fails_for_invalid_relief_window(tmp_path: Path) -> None:
    raster_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "preprocessed"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001.tif"
    )
    _write_test_raster(raster_path)
    artifact_path = _write_preprocessing_artifact(tmp_path, raster_path=raster_path)

    result = runner.invoke(
        app,
        ["derive-terrain-features", str(artifact_path), "--relief-window-size", "4"],
    )

    assert result.exit_code == 3
    assert "Relief window size must be an odd integer >= 3" in result.stdout


def test_derive_terrain_features_fails_for_multiband_raster(tmp_path: Path) -> None:
    raster_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "preprocessed"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001.tif"
    )
    _write_test_raster(raster_path, count=2)
    artifact_path = _write_preprocessing_artifact(tmp_path, raster_path=raster_path)

    result = runner.invoke(app, ["derive-terrain-features", str(artifact_path)])

    assert result.exit_code == 3
    assert "only supports single-band rasters" in result.stdout
