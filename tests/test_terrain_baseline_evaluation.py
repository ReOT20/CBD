from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from typer.testing import CliRunner

from cbd.cli import app

runner = CliRunner()


def _candidate_record(
    *,
    candidate_id: str,
    aoi_id: str,
    split: str,
    source_raster_stem: str,
    output_vector_path: Path,
    pixel_count: int,
    mean_local_relief: float,
    max_local_relief: float,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "aoi_id": aoi_id,
        "split": split,
        "terrain_source_id": "nc_dem_10m_opentopography",
        "source_raster_stem": source_raster_stem,
        "output_vector_path": str(output_vector_path.resolve()),
        "pixel_count": pixel_count,
        "area_map_units": float(pixel_count),
        "bbox_width": 2.0,
        "bbox_height": 2.0,
        "bbox_aspect_ratio": 1.0,
        "mean_local_relief": mean_local_relief,
        "max_local_relief": max_local_relief,
        "mean_slope": mean_local_relief / 2.0,
        "max_slope": max_local_relief / 2.0,
    }


def _vector_record(
    *,
    tmp_path: Path,
    aoi_id: str,
    split: str,
    source_raster_stem: str,
    candidate_vector_path: Path,
    candidate_count: int,
) -> dict[str, object]:
    return {
        "aoi_id": aoi_id,
        "split": split,
        "terrain_source_id": "nc_dem_10m_opentopography",
        "source_raster_stem": source_raster_stem,
        "input_raster_path": str(
            (
                tmp_path
                / "outputs"
                / "interim"
                / "terrain"
                / "preprocessed"
                / aoi_id
                / "nc_dem_10m_opentopography"
                / f"{source_raster_stem}.tif"
            ).resolve()
        ),
        "candidate_vector_path": str(candidate_vector_path.resolve()),
        "candidate_count": candidate_count,
    }


def _write_candidate_vector(
    path: Path,
    *,
    candidate_id: str,
    aoi_id: str,
    split: str,
    source_raster_stem: str,
    geometry: Polygon,
    pixel_count: int,
    mean_local_relief: float,
    max_local_relief: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {
            "candidate_id": [candidate_id],
            "aoi_id": [aoi_id],
            "split": [split],
            "terrain_source_id": ["nc_dem_10m_opentopography"],
            "source_raster_stem": [source_raster_stem],
            "pixel_count": [pixel_count],
            "area_map_units": [float(pixel_count)],
            "bbox_width": [2.0],
            "bbox_height": [2.0],
            "bbox_aspect_ratio": [1.0],
            "mean_local_relief": [mean_local_relief],
            "max_local_relief": [max_local_relief],
            "mean_slope": [mean_local_relief / 2.0],
            "max_slope": [max_local_relief / 2.0],
        },
        geometry=[geometry],
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GeoJSON")


def _write_candidates_artifact(
    tmp_path: Path,
    *,
    vectors: list[dict[str, object]],
    records: list[dict[str, object]],
) -> Path:
    artifact = {
        "project_name": "carolina-bays-mvp",
        "project_root": str(tmp_path.resolve()),
        "terrain_derivatives_artifact": str(
            (
                tmp_path
                / "outputs"
                / "interim"
                / "terrain"
                / "terrain_derivatives_summary.json"
            ).resolve()
        ),
        "output_root": str((tmp_path / "outputs" / "interim" / "terrain").resolve()),
        "output_summary_path": str(
            (
                tmp_path
                / "outputs"
                / "interim"
                / "terrain"
                / "terrain_candidates_summary.json"
            ).resolve()
        ),
        "total_input_groups_processed": len(vectors),
        "total_candidate_vectors_written": len(vectors),
        "total_candidates": len(records),
        "relief_threshold": 1.0,
        "min_pixels": 4,
        "vectors": vectors,
        "records": records,
    }
    artifact_path = (
        tmp_path / "outputs" / "interim" / "terrain" / "terrain_candidates_summary.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact_path


def _write_normalized_labels(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf.to_file(path, driver="GeoJSON")


def test_evaluate_terrain_baseline_success(tmp_path: Path) -> None:
    train_pos_vector = (
        tmp_path / "outputs" / "interim" / "terrain" / "candidates" / "train_aoi_01"
        / "nc_dem_10m_opentopography" / "train_pos__candidates.geojson"
    )
    train_neg_vector = train_pos_vector.with_name("train_neg__candidates.geojson")
    val_pos_vector = train_pos_vector.with_name("val_pos__candidates.geojson")
    val_neg_vector = train_pos_vector.with_name("val_neg__candidates.geojson")

    _write_candidate_vector(
        train_pos_vector,
        candidate_id="train_pos__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_pos",
        geometry=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        pixel_count=20,
        mean_local_relief=2.5,
        max_local_relief=3.0,
    )
    _write_candidate_vector(
        train_neg_vector,
        candidate_id="train_neg__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_neg",
        geometry=Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
        pixel_count=8,
        mean_local_relief=0.8,
        max_local_relief=1.0,
    )
    _write_candidate_vector(
        val_pos_vector,
        candidate_id="val_pos__cand_0001",
        aoi_id="val_aoi_01",
        split="val",
        source_raster_stem="val_pos",
        geometry=Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]),
        pixel_count=18,
        mean_local_relief=2.2,
        max_local_relief=2.9,
    )
    _write_candidate_vector(
        val_neg_vector,
        candidate_id="val_neg__cand_0001",
        aoi_id="val_aoi_01",
        split="val",
        source_raster_stem="val_neg",
        geometry=Polygon([(20, 20), (22, 20), (22, 22), (20, 22)]),
        pixel_count=6,
        mean_local_relief=0.7,
        max_local_relief=0.9,
    )

    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                candidate_vector_path=train_pos_vector,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_neg",
                candidate_vector_path=train_neg_vector,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_pos",
                candidate_vector_path=val_pos_vector,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_neg",
                candidate_vector_path=val_neg_vector,
                candidate_count=1,
            ),
        ],
        records=[
            _candidate_record(
                candidate_id="train_pos__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                output_vector_path=train_pos_vector,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            ),
            _candidate_record(
                candidate_id="train_neg__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_neg",
                output_vector_path=train_neg_vector,
                pixel_count=8,
                mean_local_relief=0.8,
                max_local_relief=1.0,
            ),
            _candidate_record(
                candidate_id="val_pos__cand_0001",
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_pos",
                output_vector_path=val_pos_vector,
                pixel_count=18,
                mean_local_relief=2.2,
                max_local_relief=2.9,
            ),
            _candidate_record(
                candidate_id="val_neg__cand_0001",
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_neg",
                output_vector_path=val_neg_vector,
                pixel_count=6,
                mean_local_relief=0.7,
                max_local_relief=0.9,
            ),
        ],
    )

    labels_path = tmp_path / "data" / "labels" / "normalized_labels.geojson"
    _write_normalized_labels(
        labels_path,
        [
            {
                "label_id": "label_train_000001",
                "class_name": "positive_complete",
                "source_id": "carolina_bays_labels",
                "split": "train",
                "review_status": "seed",
                "notes": "",
                "geometry": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            },
            {
                "label_id": "label_val_000001",
                "class_name": "positive_complete",
                "source_id": "carolina_bays_labels",
                "split": "val",
                "review_status": "seed",
                "notes": "",
                "geometry": Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]),
            },
        ],
    )

    result = runner.invoke(
        app,
        ["evaluate-terrain-baseline", str(artifact_path), str(labels_path)],
    )

    assert result.exit_code == 0
    assert "Terrain baseline evaluation completed successfully" in result.stdout

    summary_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "evaluation"
        / "terrain_baseline_evaluation_summary.json"
    )
    rows_path = summary_path.with_name("terrain_baseline_rows.csv")
    metrics_path = summary_path.with_name("terrain_baseline_metrics.json")
    failure_path = summary_path.with_name("terrain_baseline_failure_analysis.csv")
    train_predictions_path = summary_path.parent / "predictions" / "train_predictions.csv"
    val_predictions_path = summary_path.parent / "predictions" / "val_predictions.csv"

    assert summary_path.exists()
    assert rows_path.exists()
    assert metrics_path.exists()
    assert failure_path.exists()
    assert train_predictions_path.exists()
    assert val_predictions_path.exists()

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["metrics"]["train_row_count"] == 2
    assert summary_payload["metrics"]["val_row_count"] == 2
    assert summary_payload["metrics"]["val_roc_auc"] is not None

    with rows_path.open(encoding="utf-8", newline="") as handle:
        row_table = list(csv.DictReader(handle))
    assert len(row_table) == 4
    train_positive = next(row for row in row_table if row["candidate_id"] == "train_pos__cand_0001")
    val_positive = next(row for row in row_table if row["candidate_id"] == "val_pos__cand_0001")
    assert train_positive["target_label"] == "1"
    assert train_positive["matched_label_id"] == "label_train_000001"
    assert train_positive["matched_negative_label_id"] == ""
    assert train_positive["best_negative_iou"] == "0.0"
    assert train_positive["is_hard_negative_match"] == "0"
    assert val_positive["target_label"] == "1"
    assert val_positive["matched_label_id"] == "label_val_000001"
    assert val_positive["matched_negative_label_id"] == ""
    assert val_positive["best_negative_iou"] == "0.0"
    assert val_positive["is_hard_negative_match"] == "0"


def test_evaluate_terrain_baseline_uses_reviewed_hard_negatives_for_train_only(
    tmp_path: Path,
) -> None:
    train_pos_vector = (
        tmp_path / "outputs" / "interim" / "terrain" / "candidates" / "train_aoi_01"
        / "nc_dem_10m_opentopography" / "train_pos__candidates.geojson"
    )
    train_hard_neg_vector = train_pos_vector.with_name("train_hard_neg__candidates.geojson")
    val_hard_neg_vector = train_pos_vector.with_name("val_hard_neg__candidates.geojson")
    val_pos_vector = train_pos_vector.with_name("val_pos__candidates.geojson")

    _write_candidate_vector(
        train_pos_vector,
        candidate_id="train_pos__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_pos",
        geometry=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        pixel_count=20,
        mean_local_relief=2.5,
        max_local_relief=3.0,
    )
    _write_candidate_vector(
        train_hard_neg_vector,
        candidate_id="train_hard_neg__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_hard_neg",
        geometry=Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
        pixel_count=15,
        mean_local_relief=2.1,
        max_local_relief=2.8,
    )
    _write_candidate_vector(
        val_hard_neg_vector,
        candidate_id="val_hard_neg__cand_0001",
        aoi_id="val_aoi_01",
        split="val",
        source_raster_stem="val_hard_neg",
        geometry=Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]),
        pixel_count=14,
        mean_local_relief=2.0,
        max_local_relief=2.4,
    )
    _write_candidate_vector(
        val_pos_vector,
        candidate_id="val_pos__cand_0001",
        aoi_id="val_aoi_01",
        split="val",
        source_raster_stem="val_pos",
        geometry=Polygon([(20, 20), (22, 20), (22, 22), (20, 22)]),
        pixel_count=18,
        mean_local_relief=2.4,
        max_local_relief=3.1,
    )

    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                candidate_vector_path=train_pos_vector,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_hard_neg",
                candidate_vector_path=train_hard_neg_vector,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_hard_neg",
                candidate_vector_path=val_hard_neg_vector,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_pos",
                candidate_vector_path=val_pos_vector,
                candidate_count=1,
            ),
        ],
        records=[
            _candidate_record(
                candidate_id="train_pos__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                output_vector_path=train_pos_vector,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            ),
            _candidate_record(
                candidate_id="train_hard_neg__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_hard_neg",
                output_vector_path=train_hard_neg_vector,
                pixel_count=15,
                mean_local_relief=2.1,
                max_local_relief=2.8,
            ),
            _candidate_record(
                candidate_id="val_hard_neg__cand_0001",
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_hard_neg",
                output_vector_path=val_hard_neg_vector,
                pixel_count=14,
                mean_local_relief=2.0,
                max_local_relief=2.4,
            ),
            _candidate_record(
                candidate_id="val_pos__cand_0001",
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_pos",
                output_vector_path=val_pos_vector,
                pixel_count=18,
                mean_local_relief=2.4,
                max_local_relief=3.1,
            ),
        ],
    )

    labels_path = tmp_path / "data" / "labels" / "normalized_labels.geojson"
    _write_normalized_labels(
        labels_path,
        [
            {
                "label_id": "label_train_pos_0001",
                "class_name": "positive_complete",
                "source_id": "carolina_bays_labels",
                "split": "train",
                "review_status": "seed",
                "notes": "",
                "geometry": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            },
            {
                "label_id": "label_train_neg_0001",
                "class_name": "negative_hard",
                "source_id": "hard_negatives_seed",
                "split": "train",
                "review_status": "reviewed",
                "notes": "",
                "geometry": Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
            },
            {
                "label_id": "label_val_neg_0001",
                "class_name": "negative_hard",
                "source_id": "hard_negatives_seed",
                "split": "val",
                "review_status": "reviewed",
                "notes": "",
                "geometry": Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]),
            },
            {
                "label_id": "label_val_pos_0001",
                "class_name": "positive_complete",
                "source_id": "carolina_bays_labels",
                "split": "val",
                "review_status": "seed",
                "notes": "",
                "geometry": Polygon([(20, 20), (22, 20), (22, 22), (20, 22)]),
            },
        ],
    )

    result = runner.invoke(
        app,
        ["evaluate-terrain-baseline", str(artifact_path), str(labels_path)],
    )

    assert result.exit_code == 0

    rows_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "evaluation"
        / "terrain_baseline_rows.csv"
    )
    metrics_path = rows_path.with_name("terrain_baseline_metrics.json")
    with rows_path.open(encoding="utf-8", newline="") as handle:
        row_table = list(csv.DictReader(handle))

    train_hard_negative = next(
        row for row in row_table if row["candidate_id"] == "train_hard_neg__cand_0001"
    )
    val_hard_negative = next(
        row for row in row_table if row["candidate_id"] == "val_hard_neg__cand_0001"
    )
    val_positive = next(row for row in row_table if row["candidate_id"] == "val_pos__cand_0001")

    assert train_hard_negative["target_label"] == "0"
    assert train_hard_negative["matched_label_id"] == ""
    assert train_hard_negative["matched_negative_label_id"] == "label_train_neg_0001"
    assert train_hard_negative["best_negative_iou"] == "1.0"
    assert train_hard_negative["is_hard_negative_match"] == "1"
    assert train_hard_negative["training_weight"] == "3.0"

    assert val_hard_negative["target_label"] == "0"
    assert val_hard_negative["matched_label_id"] == ""
    assert val_hard_negative["matched_negative_label_id"] == "label_val_neg_0001"
    assert val_hard_negative["best_negative_iou"] == "1.0"
    assert val_hard_negative["is_hard_negative_match"] == "1"
    assert val_hard_negative["training_weight"] == "1.0"

    assert val_positive["target_label"] == "1"
    assert val_positive["matched_label_id"] == "label_val_pos_0001"
    assert val_positive["matched_negative_label_id"] == ""
    assert val_positive["is_hard_negative_match"] == "0"
    assert val_positive["training_weight"] == "1.0"

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["reviewed_hard_negative_weight"] == 3.0
    assert metrics_payload["train_positive_count"] == 1
    assert metrics_payload["train_negative_count"] == 1
    assert metrics_payload["val_positive_count"] == 1
    assert metrics_payload["val_negative_count"] == 1


def test_evaluate_terrain_baseline_ignores_seed_hard_negatives_for_target_assignment(
    tmp_path: Path,
) -> None:
    train_pos_vector = (
        tmp_path / "outputs" / "interim" / "terrain" / "candidates" / "train_aoi_01"
        / "nc_dem_10m_opentopography" / "train_pos__candidates.geojson"
    )
    train_seed_neg_vector = train_pos_vector.with_name("train_seed_neg__candidates.geojson")
    val_pos_vector = train_pos_vector.with_name("val_pos__candidates.geojson")

    _write_candidate_vector(
        train_pos_vector,
        candidate_id="train_pos__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_pos",
        geometry=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        pixel_count=20,
        mean_local_relief=2.5,
        max_local_relief=3.0,
    )
    _write_candidate_vector(
        train_seed_neg_vector,
        candidate_id="train_seed_neg__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_seed_neg",
        geometry=Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
        pixel_count=8,
        mean_local_relief=0.8,
        max_local_relief=1.0,
    )
    _write_candidate_vector(
        val_pos_vector,
        candidate_id="val_pos__cand_0001",
        aoi_id="val_aoi_01",
        split="val",
        source_raster_stem="val_pos",
        geometry=Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]),
        pixel_count=18,
        mean_local_relief=2.4,
        max_local_relief=3.1,
    )

    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                candidate_vector_path=train_pos_vector,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_seed_neg",
                candidate_vector_path=train_seed_neg_vector,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_pos",
                candidate_vector_path=val_pos_vector,
                candidate_count=1,
            ),
        ],
        records=[
            _candidate_record(
                candidate_id="train_pos__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                output_vector_path=train_pos_vector,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            ),
            _candidate_record(
                candidate_id="train_seed_neg__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_seed_neg",
                output_vector_path=train_seed_neg_vector,
                pixel_count=8,
                mean_local_relief=0.8,
                max_local_relief=1.0,
            ),
            _candidate_record(
                candidate_id="val_pos__cand_0001",
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_pos",
                output_vector_path=val_pos_vector,
                pixel_count=18,
                mean_local_relief=2.4,
                max_local_relief=3.1,
            ),
        ],
    )

    labels_path = tmp_path / "data" / "labels" / "normalized_labels.geojson"
    _write_normalized_labels(
        labels_path,
        [
            {
                "label_id": "label_train_pos_0001",
                "class_name": "positive_complete",
                "source_id": "carolina_bays_labels",
                "split": "train",
                "review_status": "seed",
                "notes": "",
                "geometry": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            },
            {
                "label_id": "label_train_neg_seed_0001",
                "class_name": "negative_hard",
                "source_id": "hard_negatives_seed",
                "split": "train",
                "review_status": "seed",
                "notes": "",
                "geometry": Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
            },
            {
                "label_id": "label_val_pos_0001",
                "class_name": "positive_complete",
                "source_id": "carolina_bays_labels",
                "split": "val",
                "review_status": "seed",
                "notes": "",
                "geometry": Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]),
            },
        ],
    )

    result = runner.invoke(
        app,
        ["evaluate-terrain-baseline", str(artifact_path), str(labels_path)],
    )

    assert result.exit_code == 0
    rows_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "evaluation"
        / "terrain_baseline_rows.csv"
    )
    with rows_path.open(encoding="utf-8", newline="") as handle:
        row_table = list(csv.DictReader(handle))

    seed_negative = next(
        row for row in row_table if row["candidate_id"] == "train_seed_neg__cand_0001"
    )
    assert seed_negative["target_label"] == "0"
    assert seed_negative["matched_negative_label_id"] == ""
    assert seed_negative["best_negative_iou"] == "0.0"
    assert seed_negative["is_hard_negative_match"] == "0"
    assert seed_negative["training_weight"] == "1.0"


def test_evaluate_terrain_baseline_passes_reviewed_hard_negative_sample_weight(
    tmp_path: Path,
) -> None:
    train_pos_vector = (
        tmp_path / "outputs" / "interim" / "terrain" / "candidates" / "train_aoi_01"
        / "nc_dem_10m_opentopography" / "train_pos__candidates.geojson"
    )
    train_hard_neg_vector = train_pos_vector.with_name("train_hard_neg__candidates.geojson")
    val_pos_vector = train_pos_vector.with_name("val_pos__candidates.geojson")

    _write_candidate_vector(
        train_pos_vector,
        candidate_id="train_pos__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_pos",
        geometry=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        pixel_count=20,
        mean_local_relief=2.5,
        max_local_relief=3.0,
    )
    _write_candidate_vector(
        train_hard_neg_vector,
        candidate_id="train_hard_neg__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_hard_neg",
        geometry=Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
        pixel_count=15,
        mean_local_relief=2.1,
        max_local_relief=2.8,
    )
    _write_candidate_vector(
        val_pos_vector,
        candidate_id="val_pos__cand_0001",
        aoi_id="val_aoi_01",
        split="val",
        source_raster_stem="val_pos",
        geometry=Polygon([(20, 20), (22, 20), (22, 22), (20, 22)]),
        pixel_count=18,
        mean_local_relief=2.4,
        max_local_relief=3.1,
    )

    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                candidate_vector_path=train_pos_vector,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_hard_neg",
                candidate_vector_path=train_hard_neg_vector,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_pos",
                candidate_vector_path=val_pos_vector,
                candidate_count=1,
            ),
        ],
        records=[
            _candidate_record(
                candidate_id="train_pos__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                output_vector_path=train_pos_vector,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            ),
            _candidate_record(
                candidate_id="train_hard_neg__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_hard_neg",
                output_vector_path=train_hard_neg_vector,
                pixel_count=15,
                mean_local_relief=2.1,
                max_local_relief=2.8,
            ),
            _candidate_record(
                candidate_id="val_pos__cand_0001",
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_pos",
                output_vector_path=val_pos_vector,
                pixel_count=18,
                mean_local_relief=2.4,
                max_local_relief=3.1,
            ),
        ],
    )

    labels_path = tmp_path / "data" / "labels" / "normalized_labels.geojson"
    _write_normalized_labels(
        labels_path,
        [
            {
                "label_id": "label_train_pos_0001",
                "class_name": "positive_complete",
                "source_id": "carolina_bays_labels",
                "split": "train",
                "review_status": "seed",
                "notes": "",
                "geometry": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            },
            {
                "label_id": "label_train_neg_0001",
                "class_name": "negative_hard",
                "source_id": "hard_negatives_seed",
                "split": "train",
                "review_status": "reviewed",
                "notes": "",
                "geometry": Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
            },
            {
                "label_id": "label_val_pos_0001",
                "class_name": "positive_complete",
                "source_id": "carolina_bays_labels",
                "split": "val",
                "review_status": "seed",
                "notes": "",
                "geometry": Polygon([(20, 20), (22, 20), (22, 22), (20, 22)]),
            },
        ],
    )

    captured: dict[str, object] = {}

    class FakeLogisticRegression:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["init_kwargs"] = kwargs

        def fit(
            self,
            x: Any,
            y: Any,
            sample_weight: Any = None,
        ) -> FakeLogisticRegression:
            captured["fit_row_count"] = len(x)
            captured["target_labels"] = list(y)
            captured["sample_weight"] = list(sample_weight) if sample_weight is not None else None
            return self

        def predict_proba(self, x: Any) -> np.ndarray:
            return np.tile(np.array([[0.9, 0.1]]), (len(x), 1))

    with patch("cbd.data.terrain.LogisticRegression", FakeLogisticRegression):
        result = runner.invoke(
            app,
            [
                "evaluate-terrain-baseline",
                str(artifact_path),
                str(labels_path),
                "--reviewed-hard-negative-weight",
                "5.0",
            ],
        )

    assert result.exit_code == 0
    assert captured["fit_row_count"] == 2
    assert captured["target_labels"] == [1, 0]
    assert captured["sample_weight"] == [1.0, 5.0]

    metrics_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "evaluation"
        / "terrain_baseline_metrics.json"
    )
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["reviewed_hard_negative_weight"] == 5.0


def test_evaluate_terrain_baseline_fails_for_conflicting_positive_and_hard_negative_matches(
    tmp_path: Path,
) -> None:
    train_pos_vector = (
        tmp_path / "outputs" / "interim" / "terrain" / "candidates" / "train_aoi_01"
        / "nc_dem_10m_opentopography" / "train_pos__candidates.geojson"
    )
    train_neg_vector = train_pos_vector.with_name("train_neg__candidates.geojson")

    _write_candidate_vector(
        train_pos_vector,
        candidate_id="train_pos__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_pos",
        geometry=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        pixel_count=20,
        mean_local_relief=2.5,
        max_local_relief=3.0,
    )
    _write_candidate_vector(
        train_neg_vector,
        candidate_id="train_neg__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_neg",
        geometry=Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
        pixel_count=8,
        mean_local_relief=0.8,
        max_local_relief=1.0,
    )

    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                candidate_vector_path=train_pos_vector,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_neg",
                candidate_vector_path=train_neg_vector,
                candidate_count=1,
            ),
        ],
        records=[
            _candidate_record(
                candidate_id="train_pos__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                output_vector_path=train_pos_vector,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            ),
            _candidate_record(
                candidate_id="train_neg__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_neg",
                output_vector_path=train_neg_vector,
                pixel_count=8,
                mean_local_relief=0.8,
                max_local_relief=1.0,
            ),
        ],
    )

    labels_path = tmp_path / "data" / "labels" / "normalized_labels.geojson"
    overlap_geometry = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    _write_normalized_labels(
        labels_path,
        [
            {
                "label_id": "label_train_pos_0001",
                "class_name": "positive_complete",
                "source_id": "carolina_bays_labels",
                "split": "train",
                "review_status": "seed",
                "notes": "",
                "geometry": overlap_geometry,
            },
            {
                "label_id": "label_train_neg_0001",
                "class_name": "negative_hard",
                "source_id": "hard_negatives_seed",
                "split": "train",
                "review_status": "reviewed",
                "notes": "",
                "geometry": overlap_geometry,
            },
        ],
    )

    result = runner.invoke(
        app,
        ["evaluate-terrain-baseline", str(artifact_path), str(labels_path)],
    )

    assert result.exit_code == 3
    assert "Candidate matches both positive and reviewed hard-negative labels" in result.stdout


def test_evaluate_terrain_baseline_ignores_cross_split_label_matches(tmp_path: Path) -> None:
    train_vector = (
        tmp_path / "outputs" / "interim" / "terrain" / "candidates" / "train_aoi_01"
        / "nc_dem_10m_opentopography" / "train_shared__candidates.geojson"
    )
    train_neg_vector = train_vector.with_name("train_neg__candidates.geojson")
    val_vector = train_vector.with_name("val_shared__candidates.geojson")
    _write_candidate_vector(
        train_vector,
        candidate_id="train_shared__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_shared",
        geometry=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        pixel_count=12,
        mean_local_relief=2.0,
        max_local_relief=2.5,
    )
    _write_candidate_vector(
        train_neg_vector,
        candidate_id="train_neg__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_neg",
        geometry=Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
        pixel_count=7,
        mean_local_relief=0.9,
        max_local_relief=1.0,
    )
    _write_candidate_vector(
        val_vector,
        candidate_id="val_shared__cand_0001",
        aoi_id="val_aoi_01",
        split="val",
        source_raster_stem="val_shared",
        geometry=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        pixel_count=11,
        mean_local_relief=1.8,
        max_local_relief=2.0,
    )

    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_shared",
                candidate_vector_path=train_vector,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_neg",
                candidate_vector_path=train_neg_vector,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_shared",
                candidate_vector_path=val_vector,
                candidate_count=1,
            ),
        ],
        records=[
            _candidate_record(
                candidate_id="train_shared__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_shared",
                output_vector_path=train_vector,
                pixel_count=12,
                mean_local_relief=2.0,
                max_local_relief=2.5,
            ),
            _candidate_record(
                candidate_id="train_neg__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_neg",
                output_vector_path=train_neg_vector,
                pixel_count=7,
                mean_local_relief=0.9,
                max_local_relief=1.0,
            ),
            _candidate_record(
                candidate_id="val_shared__cand_0001",
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_shared",
                output_vector_path=val_vector,
                pixel_count=11,
                mean_local_relief=1.8,
                max_local_relief=2.0,
            ),
        ],
    )

    labels_path = tmp_path / "data" / "labels" / "normalized_labels.geojson"
    _write_normalized_labels(
        labels_path,
        [
            {
                "label_id": "label_train_only",
                "class_name": "positive_complete",
                "source_id": "carolina_bays_labels",
                "split": "train",
                "review_status": "seed",
                "notes": "",
                "geometry": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            }
        ],
    )

    result = runner.invoke(
        app,
        ["evaluate-terrain-baseline", str(artifact_path), str(labels_path)],
    )

    assert result.exit_code == 0
    rows_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "evaluation"
        / "terrain_baseline_rows.csv"
    )
    with rows_path.open(encoding="utf-8", newline="") as handle:
        row_table = list(csv.DictReader(handle))
    val_row = next(row for row in row_table if row["candidate_id"] == "val_shared__cand_0001")
    assert val_row["target_label"] == "0"
    assert val_row["matched_label_id"] == ""


def test_evaluate_terrain_baseline_sets_roc_auc_null_for_single_validation_class(
    tmp_path: Path,
) -> None:
    train_pos_vector = (
        tmp_path / "outputs" / "interim" / "terrain" / "candidates" / "train_aoi_01"
        / "nc_dem_10m_opentopography" / "train_pos__candidates.geojson"
    )
    train_neg_vector = train_pos_vector.with_name("train_neg__candidates.geojson")
    val_neg_vector = train_pos_vector.with_name("val_neg__candidates.geojson")
    _write_candidate_vector(
        train_pos_vector,
        candidate_id="train_pos__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_pos",
        geometry=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        pixel_count=20,
        mean_local_relief=2.5,
        max_local_relief=3.0,
    )
    _write_candidate_vector(
        train_neg_vector,
        candidate_id="train_neg__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_neg",
        geometry=Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
        pixel_count=8,
        mean_local_relief=0.8,
        max_local_relief=1.0,
    )
    _write_candidate_vector(
        val_neg_vector,
        candidate_id="val_neg__cand_0001",
        aoi_id="val_aoi_01",
        split="val",
        source_raster_stem="val_neg",
        geometry=Polygon([(20, 20), (22, 20), (22, 22), (20, 22)]),
        pixel_count=6,
        mean_local_relief=0.7,
        max_local_relief=0.9,
    )

    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                candidate_vector_path=train_pos_vector,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_neg",
                candidate_vector_path=train_neg_vector,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_neg",
                candidate_vector_path=val_neg_vector,
                candidate_count=1,
            ),
        ],
        records=[
            _candidate_record(
                candidate_id="train_pos__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                output_vector_path=train_pos_vector,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            ),
            _candidate_record(
                candidate_id="train_neg__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_neg",
                output_vector_path=train_neg_vector,
                pixel_count=8,
                mean_local_relief=0.8,
                max_local_relief=1.0,
            ),
            _candidate_record(
                candidate_id="val_neg__cand_0001",
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_neg",
                output_vector_path=val_neg_vector,
                pixel_count=6,
                mean_local_relief=0.7,
                max_local_relief=0.9,
            ),
        ],
    )

    labels_path = tmp_path / "data" / "labels" / "normalized_labels.geojson"
    _write_normalized_labels(
        labels_path,
        [
            {
                "label_id": "label_train_000001",
                "class_name": "positive_complete",
                "source_id": "carolina_bays_labels",
                "split": "train",
                "review_status": "seed",
                "notes": "",
                "geometry": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            }
        ],
    )

    result = runner.invoke(
        app,
        ["evaluate-terrain-baseline", str(artifact_path), str(labels_path)],
    )

    assert result.exit_code == 0
    metrics_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "evaluation"
        / "terrain_baseline_metrics.json"
    )
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["val_roc_auc"] is None


def test_evaluate_terrain_baseline_fails_for_missing_label_columns(tmp_path: Path) -> None:
    vector_path = (
        tmp_path / "outputs" / "interim" / "terrain" / "candidates" / "train_aoi_01"
        / "nc_dem_10m_opentopography" / "train_pos__candidates.geojson"
    )
    _write_candidate_vector(
        vector_path,
        candidate_id="train_pos__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_pos",
        geometry=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        pixel_count=20,
        mean_local_relief=2.5,
        max_local_relief=3.0,
    )
    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                candidate_vector_path=vector_path,
                candidate_count=1,
            )
        ],
        records=[
            _candidate_record(
                candidate_id="train_pos__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                output_vector_path=vector_path,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            )
        ],
    )
    labels_path = tmp_path / "data" / "labels" / "broken_labels.geojson"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {"label_id": ["label_0001"], "split": ["train"]},
        geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])],
        crs="EPSG:4326",
    )
    gdf.to_file(labels_path, driver="GeoJSON")

    result = runner.invoke(
        app,
        ["evaluate-terrain-baseline", str(artifact_path), str(labels_path)],
    )

    assert result.exit_code == 3
    assert "Normalized labels vector is missing required column" in result.stdout


def test_evaluate_terrain_baseline_fails_for_invalid_match_iou(tmp_path: Path) -> None:
    vector_path = (
        tmp_path / "outputs" / "interim" / "terrain" / "candidates" / "train_aoi_01"
        / "nc_dem_10m_opentopography" / "train_pos__candidates.geojson"
    )
    _write_candidate_vector(
        vector_path,
        candidate_id="train_pos__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_pos",
        geometry=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        pixel_count=20,
        mean_local_relief=2.5,
        max_local_relief=3.0,
    )
    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                candidate_vector_path=vector_path,
                candidate_count=1,
            )
        ],
        records=[
            _candidate_record(
                candidate_id="train_pos__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                output_vector_path=vector_path,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            )
        ],
    )
    labels_path = tmp_path / "data" / "labels" / "normalized_labels.geojson"
    _write_normalized_labels(
        labels_path,
        [
            {
                "label_id": "label_train_000001",
                "class_name": "positive_complete",
                "source_id": "carolina_bays_labels",
                "split": "train",
                "review_status": "seed",
                "notes": "",
                "geometry": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            }
        ],
    )

    result = runner.invoke(
        app,
        [
            "evaluate-terrain-baseline",
            str(artifact_path),
            str(labels_path),
            "--match-iou",
            "1.5",
        ],
    )

    assert result.exit_code == 3
    assert "Match IoU threshold must be between 0 and 1" in result.stdout


def test_evaluate_terrain_baseline_fails_for_invalid_reviewed_hard_negative_weight(
    tmp_path: Path,
) -> None:
    vector_path = (
        tmp_path / "outputs" / "interim" / "terrain" / "candidates" / "train_aoi_01"
        / "nc_dem_10m_opentopography" / "train_pos__candidates.geojson"
    )
    _write_candidate_vector(
        vector_path,
        candidate_id="train_pos__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_pos",
        geometry=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        pixel_count=20,
        mean_local_relief=2.5,
        max_local_relief=3.0,
    )
    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                candidate_vector_path=vector_path,
                candidate_count=1,
            )
        ],
        records=[
            _candidate_record(
                candidate_id="train_pos__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                output_vector_path=vector_path,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            )
        ],
    )
    labels_path = tmp_path / "data" / "labels" / "normalized_labels.geojson"
    _write_normalized_labels(
        labels_path,
        [
            {
                "label_id": "label_train_000001",
                "class_name": "positive_complete",
                "source_id": "carolina_bays_labels",
                "split": "train",
                "review_status": "seed",
                "notes": "",
                "geometry": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            }
        ],
    )

    result = runner.invoke(
        app,
        [
            "evaluate-terrain-baseline",
            str(artifact_path),
            str(labels_path),
            "--reviewed-hard-negative-weight",
            "0",
        ],
    )

    assert result.exit_code == 3
    assert "Reviewed hard-negative weight must be greater than 0" in result.stdout


def test_evaluate_terrain_baseline_fails_for_invalid_candidate_split(tmp_path: Path) -> None:
    vector_path = (
        tmp_path / "outputs" / "interim" / "terrain" / "candidates" / "train_aoi_01"
        / "nc_dem_10m_opentopography" / "train_pos__candidates.geojson"
    )
    _write_candidate_vector(
        vector_path,
        candidate_id="train_pos__cand_0001",
        aoi_id="train_aoi_01",
        split="holdout",
        source_raster_stem="train_pos",
        geometry=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        pixel_count=20,
        mean_local_relief=2.5,
        max_local_relief=3.0,
    )
    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="holdout",
                source_raster_stem="train_pos",
                candidate_vector_path=vector_path,
                candidate_count=1,
            )
        ],
        records=[
            _candidate_record(
                candidate_id="train_pos__cand_0001",
                aoi_id="train_aoi_01",
                split="holdout",
                source_raster_stem="train_pos",
                output_vector_path=vector_path,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            )
        ],
    )
    labels_path = tmp_path / "data" / "labels" / "normalized_labels.geojson"
    _write_normalized_labels(
        labels_path,
        [
            {
                "label_id": "label_train_000001",
                "class_name": "positive_complete",
                "source_id": "carolina_bays_labels",
                "split": "train",
                "review_status": "seed",
                "notes": "",
                "geometry": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            }
        ],
    )

    result = runner.invoke(
        app,
        ["evaluate-terrain-baseline", str(artifact_path), str(labels_path)],
    )

    assert result.exit_code == 3
    assert "Candidate vector split column" in result.stdout
    assert "allowed split values" in result.stdout


def test_evaluate_terrain_baseline_fails_for_missing_val_rows(tmp_path: Path) -> None:
    train_pos_vector = (
        tmp_path / "outputs" / "interim" / "terrain" / "candidates" / "train_aoi_01"
        / "nc_dem_10m_opentopography" / "train_pos__candidates.geojson"
    )
    train_neg_vector = train_pos_vector.with_name("train_neg__candidates.geojson")

    _write_candidate_vector(
        train_pos_vector,
        candidate_id="train_pos__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_pos",
        geometry=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        pixel_count=20,
        mean_local_relief=2.5,
        max_local_relief=3.0,
    )
    _write_candidate_vector(
        train_neg_vector,
        candidate_id="train_neg__cand_0001",
        aoi_id="train_aoi_01",
        split="train",
        source_raster_stem="train_neg",
        geometry=Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
        pixel_count=8,
        mean_local_relief=0.8,
        max_local_relief=1.0,
    )
    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                candidate_vector_path=train_pos_vector,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_neg",
                candidate_vector_path=train_neg_vector,
                candidate_count=1,
            ),
        ],
        records=[
            _candidate_record(
                candidate_id="train_pos__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                output_vector_path=train_pos_vector,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            ),
            _candidate_record(
                candidate_id="train_neg__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_neg",
                output_vector_path=train_neg_vector,
                pixel_count=8,
                mean_local_relief=0.8,
                max_local_relief=1.0,
            ),
        ],
    )
    labels_path = tmp_path / "data" / "labels" / "normalized_labels.geojson"
    _write_normalized_labels(
        labels_path,
        [
            {
                "label_id": "label_train_000001",
                "class_name": "positive_complete",
                "source_id": "carolina_bays_labels",
                "split": "train",
                "review_status": "seed",
                "notes": "",
                "geometry": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            }
        ],
    )

    result = runner.invoke(
        app,
        ["evaluate-terrain-baseline", str(artifact_path), str(labels_path)],
    )

    assert result.exit_code == 3
    assert "No val candidate rows available for evaluation" in result.stdout


def test_evaluate_terrain_baseline_fails_for_unreadable_candidate_vector(tmp_path: Path) -> None:
    vector_path = (
        tmp_path / "outputs" / "interim" / "terrain" / "candidates" / "train_aoi_01"
        / "nc_dem_10m_opentopography" / "train_pos__candidates.geojson"
    )
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    vector_path.write_text("{bad-geojson", encoding="utf-8")
    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                candidate_vector_path=vector_path,
                candidate_count=1,
            )
        ],
        records=[
            _candidate_record(
                candidate_id="train_pos__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_pos",
                output_vector_path=vector_path,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            )
        ],
    )
    labels_path = tmp_path / "data" / "labels" / "normalized_labels.geojson"
    _write_normalized_labels(
        labels_path,
        [
            {
                "label_id": "label_train_000001",
                "class_name": "positive_complete",
                "source_id": "carolina_bays_labels",
                "split": "train",
                "review_status": "seed",
                "notes": "",
                "geometry": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            }
        ],
    )

    result = runner.invoke(
        app,
        ["evaluate-terrain-baseline", str(artifact_path), str(labels_path)],
    )

    assert result.exit_code == 3
    assert "Candidate vector could not be read for evaluation" in result.stdout
