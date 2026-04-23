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


def test_normalize_labels_command_fails_for_invalid_split(tmp_path: Path) -> None:
    input_path = tmp_path / "input_labels.geojson"
    output_path = tmp_path / "normalized" / "labels.geojson"
    _write_test_geojson(input_path)

    result = runner.invoke(
        app,
        [
            "normalize-labels",
            str(input_path),
            str(output_path),
            "--split",
            "dev",
        ],
    )

    assert result.exit_code == 3
    assert "Label split must use one of the allowed split values" in result.stdout


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


def test_normalize_aoi_command_fails_for_invalid_split(tmp_path: Path) -> None:
    input_path = tmp_path / "input_aoi.geojson"
    output_path = tmp_path / "normalized" / "aoi.geojson"
    _write_test_geojson(input_path)

    result = runner.invoke(
        app,
        [
            "normalize-aoi",
            str(input_path),
            str(output_path),
            "--split",
            "test",
        ],
    )

    assert result.exit_code == 3
    assert "AOI split must use one of the allowed split values" in result.stdout


def test_seed_hard_negatives_command(tmp_path: Path) -> None:
    input_path = tmp_path / "inventory.geojson"
    output_path = tmp_path / "normalized" / "hard_negatives.geojson"
    inventory = gpd.GeoDataFrame(
        {
            "candidate_id": ["cand_high", "cand_low", "cand_train"],
            "aoi_id": ["val_aoi_01", "val_aoi_01", "train_aoi_01"],
            "split": ["val", "val", "train"],
            "terrain_source_id": [
                "nc_dem_10m_opentopography",
                "nc_dem_10m_opentopography",
                "nc_dem_10m_opentopography",
            ],
            "source_raster_stem": ["tile_val", "tile_val", "tile_train"],
            "target_label": [0, 0, 0],
            "score": [0.22, 0.01, 0.45],
            "pixel_count": [700, 20, 900],
            "max_local_relief": [4.5, 1.1, 5.0],
        },
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            Polygon([(4, 4), (5, 4), (5, 5), (4, 5)]),
        ],
        crs="EPSG:4326",
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_file(input_path)

    result = runner.invoke(
        app,
        [
            "seed-hard-negatives",
            str(input_path),
            str(output_path),
            "--split",
            "val",
            "--top-n",
            "5",
            "--min-score",
            "0.05",
            "--min-pixels",
            "50",
            "--min-max-local-relief",
            "2.0",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    gdf = gpd.read_file(output_path)
    assert len(gdf) == 1
    assert gdf.loc[0, "class_name"] == "negative_hard"
    assert gdf.loc[0, "split"] == "val"
    assert gdf.loc[0, "source_id"] == "hard_negatives_seed"
    assert (
        gdf.loc[0, "parent_source_record"]
        == "split=val;aoi_id=val_aoi_01;terrain_source_id=nc_dem_10m_opentopography;"
        "source_raster_stem=tile_val;candidate_id=cand_high"
    )


def test_seed_hard_negatives_command_fails_for_invalid_split(tmp_path: Path) -> None:
    input_path = tmp_path / "inventory.geojson"
    output_path = tmp_path / "normalized" / "hard_negatives.geojson"
    inventory = gpd.GeoDataFrame(
        {
            "candidate_id": ["cand_high"],
            "aoi_id": ["val_aoi_01"],
            "split": ["val"],
            "terrain_source_id": ["nc_dem_10m_opentopography"],
            "source_raster_stem": ["tile_val"],
            "target_label": [0],
            "score": [0.22],
            "pixel_count": [700],
            "max_local_relief": [4.5],
        },
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:4326",
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_file(input_path)

    result = runner.invoke(
        app,
        [
            "seed-hard-negatives",
            str(input_path),
            str(output_path),
            "--split",
            "holdout",
        ],
    )

    assert result.exit_code == 3
    assert "Hard-negative seed split must use one of the allowed split values" in result.stdout
