from __future__ import annotations

from pathlib import Path

import geopandas as gpd
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


def test_normalize_labels_command(tmp_path: Path) -> None:
    input_path = tmp_path / "input_labels.geojson"
    output_path = tmp_path / "normalized" / "labels.geojson"
    _write_test_geojson(input_path)

    result = runner.invoke(
        app,
        [
            "normalize-labels",
            str(input_path),
            str(output_path),
            "--source-id",
            "carolina_bays_labels",
            "--split",
            "train",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    gdf = gpd.read_file(output_path)
    assert "label_id" in gdf.columns
    assert "class_name" in gdf.columns
    assert len(gdf) == 1


def test_normalize_aoi_command(tmp_path: Path) -> None:
    input_path = tmp_path / "input_aoi.geojson"
    output_path = tmp_path / "normalized" / "aoi.geojson"
    _write_test_geojson(input_path)

    result = runner.invoke(
        app,
        [
            "normalize-aoi",
            str(input_path),
            str(output_path),
            "--aoi-id",
            "val_aoi_01",
            "--split",
            "val",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    gdf = gpd.read_file(output_path)
    assert "aoi_id" in gdf.columns
    assert "split" in gdf.columns
    assert len(gdf) == 1
