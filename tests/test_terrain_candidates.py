from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from typer.testing import CliRunner

from cbd.cli import app

runner = CliRunner()


def _write_derivative_raster(
    path: Path,
    data: np.ndarray,
    *,
    crs: str = "EPSG:4326",
    transform=None,
    nodata: float = -9999.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if transform is None:
        transform = from_origin(0.0, 5.0, 1.0, 1.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=int(data.shape[0]),
        width=int(data.shape[1]),
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"), 1)


def _write_derivatives_artifact(
    tmp_path: Path,
    *,
    local_relief_path: Path | None,
    slope_path: Path,
) -> Path:
    records: list[dict[str, object]] = [
        {
            "aoi_id": "train_aoi_01",
            "split": "train",
            "terrain_source_id": "nc_dem_10m_opentopography",
            "input_raster_path": str(
                (
                    tmp_path
                    / "outputs"
                    / "interim"
                    / "terrain"
                    / "preprocessed"
                    / "train_aoi_01"
                    / "nc_dem_10m_opentopography"
                    / "tile_001.tif"
                ).resolve()
            ),
            "derivative_name": "slope",
            "output_raster_path": str(slope_path.resolve()),
            "width": 5,
            "height": 5,
            "dtype": "float32",
            "nodata": -9999.0,
            "relief_window_size": None,
        }
    ]
    if local_relief_path is not None:
        records.append(
            {
                "aoi_id": "train_aoi_01",
                "split": "train",
                "terrain_source_id": "nc_dem_10m_opentopography",
                "input_raster_path": str(
                    (
                        tmp_path
                        / "outputs"
                        / "interim"
                        / "terrain"
                        / "preprocessed"
                        / "train_aoi_01"
                        / "nc_dem_10m_opentopography"
                        / "tile_001.tif"
                    ).resolve()
                ),
                "derivative_name": "local_relief",
                "output_raster_path": str(local_relief_path.resolve()),
                "width": 5,
                "height": 5,
                "dtype": "float32",
                "nodata": -9999.0,
                "relief_window_size": 5,
            }
        )

    artifact = {
        "project_name": "carolina-bays-mvp",
        "project_root": str(tmp_path.resolve()),
        "terrain_preprocessing_artifact": str(
            (
                tmp_path
                / "outputs"
                / "interim"
                / "terrain"
                / "terrain_preprocessing_summary.json"
            ).resolve()
        ),
        "output_root": str((tmp_path / "outputs" / "interim" / "terrain").resolve()),
        "output_summary_path": str(
            (
                tmp_path
                / "outputs"
                / "interim"
                / "terrain"
                / "terrain_derivatives_summary.json"
            ).resolve()
        ),
        "total_input_rasters_processed": 1,
        "total_derivative_rasters_written": len(records),
        "relief_window_size": 5,
        "records": records,
    }
    artifact_path = (
        tmp_path / "outputs" / "interim" / "terrain" / "terrain_derivatives_summary.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact_path


def test_generate_terrain_candidates_success(tmp_path: Path) -> None:
    local_relief_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "derivatives"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__local_relief.tif"
    )
    slope_path = local_relief_path.with_name("tile_001__slope.tif")
    relief = np.array(
        [
            [0.2, 0.4, 0.3, 0.1, 0.0],
            [0.5, 1.4, 1.5, 0.6, 0.2],
            [0.3, 1.6, 1.7, 0.7, 0.1],
            [0.1, 0.5, 0.8, 0.4, 0.0],
            [0.0, 0.1, 0.2, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    slope = np.array(
        [
            [0.2, 0.4, 0.5, 0.3, 0.1],
            [0.4, 3.0, 3.5, 0.8, 0.3],
            [0.3, 2.8, 3.2, 0.9, 0.2],
            [0.1, 0.7, 1.0, 0.4, 0.1],
            [0.0, 0.2, 0.3, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    _write_derivative_raster(local_relief_path, relief)
    _write_derivative_raster(slope_path, slope)
    artifact_path = _write_derivatives_artifact(
        tmp_path, local_relief_path=local_relief_path, slope_path=slope_path
    )

    result = runner.invoke(app, ["generate-terrain-candidates", str(artifact_path)])

    assert result.exit_code == 0
    assert "Terrain candidate generation completed successfully" in result.stdout

    summary_path = (
        tmp_path / "outputs" / "interim" / "terrain" / "terrain_candidates_summary.json"
    )
    candidate_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__candidates.geojson"
    )

    assert summary_path.exists()
    assert candidate_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["total_input_groups_processed"] == 1
    assert payload["total_candidate_vectors_written"] == 1
    assert payload["total_candidates"] == 1
    assert payload["vectors"][0]["candidate_count"] == 1
    assert payload["records"][0]["pixel_count"] == 4
    assert payload["records"][0]["mean_local_relief"] == pytest.approx(1.55)
    assert payload["records"][0]["max_slope"] == 3.5

    gdf = gpd.read_file(candidate_path)
    assert len(gdf) == 1
    assert gdf.loc[0, "candidate_id"] == "tile_001__cand_0001"
    assert float(gdf.loc[0, "area_map_units"]) == 4.0


def test_generate_terrain_candidates_empty_result(tmp_path: Path) -> None:
    local_relief_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "derivatives"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__local_relief.tif"
    )
    slope_path = local_relief_path.with_name("tile_001__slope.tif")
    relief = np.full((5, 5), 0.4, dtype=np.float32)
    slope = np.full((5, 5), 1.5, dtype=np.float32)
    _write_derivative_raster(local_relief_path, relief)
    _write_derivative_raster(slope_path, slope)
    artifact_path = _write_derivatives_artifact(
        tmp_path, local_relief_path=local_relief_path, slope_path=slope_path
    )

    result = runner.invoke(app, ["generate-terrain-candidates", str(artifact_path)])

    assert result.exit_code == 0
    summary_path = (
        tmp_path / "outputs" / "interim" / "terrain" / "terrain_candidates_summary.json"
    )
    candidate_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__candidates.geojson"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert payload["total_candidates"] == 0
    assert payload["vectors"][0]["candidate_count"] == 0
    gdf = gpd.read_file(candidate_path)
    assert gdf.empty


def test_generate_terrain_candidates_fails_for_missing_artifact(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.json"

    result = runner.invoke(app, ["generate-terrain-candidates", str(missing_path)])

    assert result.exit_code == 2
    assert "Terrain derivatives artifact not found" in result.stdout


def test_generate_terrain_candidates_fails_for_malformed_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "terrain_derivatives_summary.json"
    artifact_path.write_text("{not-json", encoding="utf-8")

    result = runner.invoke(app, ["generate-terrain-candidates", str(artifact_path)])

    assert result.exit_code == 3
    assert "Failed to parse terrain derivatives artifact" in result.stdout


def test_generate_terrain_candidates_fails_for_missing_referenced_raster(tmp_path: Path) -> None:
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
    _write_derivative_raster(slope_path, np.ones((5, 5), dtype=np.float32))
    missing_relief_path = slope_path.with_name("tile_001__local_relief.tif")
    artifact_path = _write_derivatives_artifact(
        tmp_path, local_relief_path=missing_relief_path, slope_path=slope_path
    )

    result = runner.invoke(app, ["generate-terrain-candidates", str(artifact_path)])

    assert result.exit_code == 3
    assert "Derivative raster not found" in result.stdout


def test_generate_terrain_candidates_fails_for_mismatched_raster_grids(tmp_path: Path) -> None:
    local_relief_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "derivatives"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__local_relief.tif"
    )
    slope_path = local_relief_path.with_name("tile_001__slope.tif")
    _write_derivative_raster(local_relief_path, np.ones((5, 5), dtype=np.float32))
    _write_derivative_raster(
        slope_path,
        np.ones((5, 5), dtype=np.float32),
        transform=from_origin(100.0, 5.0, 1.0, 1.0),
    )
    artifact_path = _write_derivatives_artifact(
        tmp_path, local_relief_path=local_relief_path, slope_path=slope_path
    )

    result = runner.invoke(app, ["generate-terrain-candidates", str(artifact_path)])

    assert result.exit_code == 3
    assert "Mismatched raster transform" in result.stdout


def test_generate_terrain_candidates_fails_for_missing_local_relief_derivative(
    tmp_path: Path,
) -> None:
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
    _write_derivative_raster(slope_path, np.ones((5, 5), dtype=np.float32))
    artifact_path = _write_derivatives_artifact(
        tmp_path, local_relief_path=None, slope_path=slope_path
    )

    result = runner.invoke(app, ["generate-terrain-candidates", str(artifact_path)])

    assert result.exit_code == 3
    assert "Missing required local_relief derivative" in result.stdout


def test_generate_terrain_candidates_fails_for_invalid_parameters(tmp_path: Path) -> None:
    local_relief_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "derivatives"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__local_relief.tif"
    )
    slope_path = local_relief_path.with_name("tile_001__slope.tif")
    _write_derivative_raster(local_relief_path, np.ones((5, 5), dtype=np.float32))
    _write_derivative_raster(slope_path, np.ones((5, 5), dtype=np.float32))
    artifact_path = _write_derivatives_artifact(
        tmp_path, local_relief_path=local_relief_path, slope_path=slope_path
    )

    negative_threshold_result = runner.invoke(
        app,
        [
            "generate-terrain-candidates",
            str(artifact_path),
            "--relief-threshold",
            "-1",
        ],
    )
    min_pixels_result = runner.invoke(
        app,
        [
            "generate-terrain-candidates",
            str(artifact_path),
            "--min-pixels",
            "0",
        ],
    )

    assert negative_threshold_result.exit_code == 3
    assert "Relief threshold must be >= 0" in negative_threshold_result.stdout
    assert min_pixels_result.exit_code == 3
    assert "Minimum pixels must be >= 1" in min_pixels_result.stdout
