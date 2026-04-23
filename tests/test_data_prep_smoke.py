from __future__ import annotations

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


def test_normalize_labels_by_aoi_command(tmp_path: Path) -> None:
    input_path = tmp_path / "input_labels.geojson"
    output_path = tmp_path / "normalized" / "labels_by_aoi.geojson"
    train_aoi_path = tmp_path / "data" / "raw" / "aoi" / "train_aoi_01.geojson"
    val_aoi_path = tmp_path / "data" / "raw" / "aoi" / "val_aoi_01.geojson"
    manifests_dir = tmp_path / "manifests"
    aoi_manifest_path = manifests_dir / "aoi_manifest.yaml"

    input_gdf = gpd.GeoDataFrame(
        {"name": ["train_label", "val_label", "outside_label"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]),
            Polygon([(20, 20), (21, 20), (21, 21), (20, 21)]),
        ],
        crs="EPSG:4326",
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_gdf.to_file(input_path)

    train_aoi_gdf = gpd.GeoDataFrame(
        {"name": ["train"]},
        geometry=[Polygon([(-1, -1), (2, -1), (2, 2), (-1, 2)])],
        crs="EPSG:4326",
    )
    train_aoi_path.parent.mkdir(parents=True, exist_ok=True)
    train_aoi_gdf.to_file(train_aoi_path)

    val_aoi_gdf = gpd.GeoDataFrame(
        {"name": ["val"]},
        geometry=[Polygon([(9, 9), (12, 9), (12, 12), (9, 12)])],
        crs="EPSG:4326",
    )
    val_aoi_gdf.to_file(val_aoi_path)

    manifests_dir.mkdir(parents=True, exist_ok=True)
    aoi_manifest_path.write_text(
        yaml.safe_dump(
            {
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
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "normalize-labels-by-aoi",
            str(input_path),
            str(aoi_manifest_path),
            str(output_path),
            "--source-id",
            "carolina_bays_labels",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    gdf = gpd.read_file(output_path)
    assert len(gdf) == 2
    assert sorted(gdf["split"].tolist()) == ["train", "val"]
    assert "label_id" in gdf.columns
    assert "class_name" in gdf.columns


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
