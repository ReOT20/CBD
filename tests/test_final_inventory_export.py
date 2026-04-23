from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import cast

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
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
    wetlands_any_overlap: int | None = None,
    wetlands_overlap_area: float | None = None,
    wetlands_overlap_fraction: float | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
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
    if wetlands_any_overlap is not None:
        record["wetlands_any_overlap"] = wetlands_any_overlap
    if wetlands_overlap_area is not None:
        record["wetlands_overlap_area"] = wetlands_overlap_area
    if wetlands_overlap_fraction is not None:
        record["wetlands_overlap_fraction"] = wetlands_overlap_fraction
    return record


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


def _evaluation_row(
    *,
    candidate_id: str,
    aoi_id: str,
    split: str,
    source_raster_stem: str,
    output_vector_path: Path,
    pixel_count: int,
    mean_local_relief: float,
    max_local_relief: float,
    matched_label_id: str | None,
    best_iou: float,
    matched_negative_label_id: str | None = None,
    best_negative_iou: float = 0.0,
    is_hard_negative_match: int = 0,
    training_weight: float = 1.0,
    wetlands_any_overlap: int | None = None,
    wetlands_overlap_area: float | None = None,
    wetlands_overlap_fraction: float | None = None,
    target_label: int,
    score: float,
    predicted_label: int,
) -> dict[str, object]:
    record: dict[str, object] = {
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
        "matched_label_id": matched_label_id,
        "best_iou": best_iou,
        "matched_negative_label_id": matched_negative_label_id,
        "best_negative_iou": best_negative_iou,
        "is_hard_negative_match": is_hard_negative_match,
        "training_weight": training_weight,
        "target_label": target_label,
        "score": score,
        "predicted_label": predicted_label,
    }
    if wetlands_any_overlap is not None:
        record["wetlands_any_overlap"] = wetlands_any_overlap
    if wetlands_overlap_area is not None:
        record["wetlands_overlap_area"] = wetlands_overlap_area
    if wetlands_overlap_fraction is not None:
        record["wetlands_overlap_fraction"] = wetlands_overlap_fraction
    return record


def _write_candidate_vector(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    geometry = [cast(BaseGeometry, row["geometry"]) for row in rows]
    payload = [{key: value for key, value in row.items() if key != "geometry"} for row in rows]
    gdf = gpd.GeoDataFrame(payload, geometry=geometry, crs="EPSG:4326")
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


def _write_evaluation_artifact(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    evaluation_root = tmp_path / "outputs" / "interim" / "terrain" / "evaluation"
    rows_path = evaluation_root / "terrain_baseline_rows.csv"
    metrics_path = evaluation_root / "terrain_baseline_metrics.json"
    failure_path = evaluation_root / "terrain_baseline_failure_analysis.csv"
    summary_path = evaluation_root / "terrain_baseline_evaluation_summary.json"
    fieldnames = [
        "candidate_id",
        "aoi_id",
        "split",
        "terrain_source_id",
        "source_raster_stem",
        "output_vector_path",
        "pixel_count",
        "area_map_units",
        "bbox_width",
        "bbox_height",
        "bbox_aspect_ratio",
        "mean_local_relief",
        "max_local_relief",
        "mean_slope",
        "max_slope",
        "wetlands_any_overlap",
        "wetlands_overlap_area",
        "wetlands_overlap_fraction",
        "matched_label_id",
        "best_iou",
        "matched_negative_label_id",
        "best_negative_iou",
        "is_hard_negative_match",
        "training_weight",
        "target_label",
        "score",
        "predicted_label",
    ]

    evaluation_root.mkdir(parents=True, exist_ok=True)
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics_path.write_text(
        json.dumps(
            {
                "threshold": 0.5,
                "reviewed_hard_negative_weight": 3.0,
                "train_row_count": sum(1 for row in rows if row["split"] != "val"),
                "val_row_count": sum(1 for row in rows if row["split"] == "val"),
                "train_positive_count": 1,
                "train_negative_count": 1,
                "val_positive_count": 0,
                "val_negative_count": 1,
                "val_precision": 0.0,
                "val_recall": 0.0,
                "val_f1": 0.0,
                "val_roc_auc": None,
            }
        ),
        encoding="utf-8",
    )
    with failure_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + ["error_type"])
        writer.writeheader()

    summary_path.write_text(
        json.dumps(
            {
                "project_name": "carolina-bays-mvp",
                "project_root": str(tmp_path.resolve()),
                "terrain_candidates_artifact": str(
                    (
                        tmp_path
                        / "outputs"
                        / "interim"
                        / "terrain"
                        / "terrain_candidates_summary.json"
                    ).resolve()
                ),
                "normalized_labels_path": str(
                    (tmp_path / "data" / "labels" / "normalized_labels.geojson").resolve()
                ),
                "output_root": str(evaluation_root.resolve()),
                "output_summary_path": str(summary_path.resolve()),
                "rows_output_path": str(rows_path.resolve()),
                "metrics_output_path": str(metrics_path.resolve()),
                "failure_analysis_output_path": str(failure_path.resolve()),
                "match_iou_threshold": 0.1,
                "classification_threshold": 0.5,
                "prediction_artifacts": [],
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def test_export_final_inventory_success(tmp_path: Path) -> None:
    train_vector = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "train_source__candidates.geojson"
    )
    val_vector = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "val_aoi_01"
        / "nc_dem_10m_opentopography"
        / "val_source__candidates.geojson"
    )

    _write_candidate_vector(
        train_vector,
        [
            {
                **_candidate_record(
                    candidate_id="train_source__cand_0001",
                    aoi_id="train_aoi_01",
                    split="train",
                    source_raster_stem="train_source",
                    output_vector_path=train_vector,
                    pixel_count=20,
                    mean_local_relief=2.5,
                    max_local_relief=3.0,
                ),
                "geometry": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            },
            {
                **_candidate_record(
                    candidate_id="train_source__cand_0002",
                    aoi_id="train_aoi_01",
                    split="train",
                    source_raster_stem="train_source",
                    output_vector_path=train_vector,
                    pixel_count=9,
                    mean_local_relief=1.0,
                    max_local_relief=1.4,
                ),
                "geometry": Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            },
        ],
    )
    _write_candidate_vector(
        val_vector,
        [
            {
                **_candidate_record(
                    candidate_id="val_source__cand_0001",
                    aoi_id="val_aoi_01",
                    split="val",
                    source_raster_stem="val_source",
                    output_vector_path=val_vector,
                    pixel_count=7,
                    mean_local_relief=0.9,
                    max_local_relief=1.1,
                ),
                "geometry": Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]),
            }
        ],
    )

    candidates_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_source",
                candidate_vector_path=train_vector,
                candidate_count=2,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_source",
                candidate_vector_path=val_vector,
                candidate_count=1,
            ),
        ],
        records=[
            _candidate_record(
                candidate_id="train_source__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_source",
                output_vector_path=train_vector,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            ),
            _candidate_record(
                candidate_id="train_source__cand_0002",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_source",
                output_vector_path=train_vector,
                pixel_count=9,
                mean_local_relief=1.0,
                max_local_relief=1.4,
            ),
            _candidate_record(
                candidate_id="val_source__cand_0001",
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_source",
                output_vector_path=val_vector,
                pixel_count=7,
                mean_local_relief=0.9,
                max_local_relief=1.1,
            ),
        ],
    )
    evaluation_path = _write_evaluation_artifact(
        tmp_path,
        [
            _evaluation_row(
                candidate_id="train_source__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_source",
                output_vector_path=train_vector,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
                matched_label_id="label_train_0001",
                best_iou=0.9,
                wetlands_any_overlap=1,
                wetlands_overlap_area=2.0,
                wetlands_overlap_fraction=0.1,
                target_label=1,
                score=0.91,
                predicted_label=1,
            ),
            _evaluation_row(
                candidate_id="val_source__cand_0001",
                aoi_id="val_aoi_01",
                split="val",
                source_raster_stem="val_source",
                output_vector_path=val_vector,
                pixel_count=7,
                mean_local_relief=0.9,
                max_local_relief=1.1,
                matched_label_id=None,
                best_iou=0.0,
                target_label=0,
                score=0.12,
                predicted_label=0,
            ),
        ],
    )

    result = runner.invoke(
        app,
        ["export-final-inventory", str(evaluation_path), str(candidates_path)],
    )

    assert result.exit_code == 0
    assert "Final scored candidate inventory exported successfully" in result.stdout

    geojson_path = tmp_path / "outputs" / "final" / "terrain" / "terrain_final_inventory.geojson"
    csv_path = tmp_path / "outputs" / "final" / "terrain" / "terrain_final_inventory.csv"
    summary_path = (
        tmp_path / "outputs" / "final" / "terrain" / "terrain_final_inventory_summary.json"
    )

    assert geojson_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()

    gdf = gpd.read_file(geojson_path)
    assert len(gdf) == 2
    assert sorted(gdf["candidate_id"].tolist()) == [
        "train_source__cand_0001",
        "val_source__cand_0001",
    ]
    train_row = gdf.loc[gdf["candidate_id"] == "train_source__cand_0001"].iloc[0]
    assert float(train_row["score"]) == 0.91
    assert str(train_row["output_vector_path"]) == str(train_vector.resolve())
    assert float(train_row.geometry.area) == 4.0
    assert int(train_row["wetlands_any_overlap"]) == 1
    assert float(train_row["wetlands_overlap_area"]) == 2.0
    assert float(train_row["wetlands_overlap_fraction"]) == 0.1
    assert train_row["matched_negative_label_id"] is None
    assert float(train_row["best_negative_iou"]) == 0.0
    assert int(train_row["is_hard_negative_match"]) == 0

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert sorted(row["candidate_id"] for row in rows) == [
        "train_source__cand_0001",
        "val_source__cand_0001",
    ]
    assert "train_source__cand_0002" not in {row["candidate_id"] for row in rows}
    val_row = next(row for row in rows if row["candidate_id"] == "val_source__cand_0001")
    assert val_row["wetlands_any_overlap"] == "0"
    assert val_row["wetlands_overlap_area"] == "0.0"
    assert val_row["wetlands_overlap_fraction"] == "0.0"
    assert val_row["matched_negative_label_id"] == ""
    assert val_row["best_negative_iou"] == "0.0"
    assert val_row["is_hard_negative_match"] == "0"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_exported_features"] == 2
    assert summary["split_counts"] == {"train": 1, "val": 1}
    assert summary["predicted_positive_count"] == 1
    assert summary["predicted_negative_count"] == 1


def test_export_final_inventory_propagates_hard_negative_diagnostics(tmp_path: Path) -> None:
    train_vector = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "train_source__candidates.geojson"
    )
    _write_candidate_vector(
        train_vector,
        [
            {
                **_candidate_record(
                    candidate_id="train_source__cand_0001",
                    aoi_id="train_aoi_01",
                    split="train",
                    source_raster_stem="train_source",
                    output_vector_path=train_vector,
                    pixel_count=20,
                    mean_local_relief=2.5,
                    max_local_relief=3.0,
                ),
                "geometry": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            }
        ],
    )
    candidates_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_source",
                candidate_vector_path=train_vector,
                candidate_count=1,
            )
        ],
        records=[
            _candidate_record(
                candidate_id="train_source__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_source",
                output_vector_path=train_vector,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            )
        ],
    )
    evaluation_path = _write_evaluation_artifact(
        tmp_path,
        [
            _evaluation_row(
                candidate_id="train_source__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_source",
                output_vector_path=train_vector,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
                matched_label_id=None,
                best_iou=0.0,
                matched_negative_label_id="label_train_neg_0001",
                best_negative_iou=0.82,
                is_hard_negative_match=1,
                target_label=0,
                score=0.18,
                predicted_label=0,
            )
        ],
    )

    result = runner.invoke(
        app,
        ["export-final-inventory", str(evaluation_path), str(candidates_path)],
    )

    assert result.exit_code == 0

    geojson_path = tmp_path / "outputs" / "final" / "terrain" / "terrain_final_inventory.geojson"
    csv_path = tmp_path / "outputs" / "final" / "terrain" / "terrain_final_inventory.csv"
    summary_path = (
        tmp_path / "outputs" / "final" / "terrain" / "terrain_final_inventory_summary.json"
    )

    gdf = gpd.read_file(geojson_path)
    row = gdf.iloc[0]
    assert row["matched_negative_label_id"] == "label_train_neg_0001"
    assert float(row["best_negative_iou"]) == 0.82
    assert int(row["is_hard_negative_match"]) == 1

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["matched_negative_label_id"] == "label_train_neg_0001"
    assert rows[0]["best_negative_iou"] == "0.82"
    assert rows[0]["is_hard_negative_match"] == "1"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["records"][0]["matched_negative_label_id"] == "label_train_neg_0001"
    assert float(summary["records"][0]["best_negative_iou"]) == 0.82
    assert summary["records"][0]["is_hard_negative_match"] == 1


def test_export_final_inventory_fails_for_missing_evaluation_artifact(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "export-final-inventory",
            str(tmp_path / "missing_evaluation_summary.json"),
            str(tmp_path / "missing_candidates_summary.json"),
        ],
    )

    assert result.exit_code == 2
    assert "Terrain baseline evaluation artifact not found" in result.stdout


def test_export_final_inventory_fails_for_malformed_evaluation_artifact(
    tmp_path: Path,
) -> None:
    evaluation_path = tmp_path / "terrain_baseline_evaluation_summary.json"
    evaluation_path.write_text("{bad-json", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "export-final-inventory",
            str(evaluation_path),
            str(tmp_path / "missing_candidates_summary.json"),
        ],
    )

    assert result.exit_code == 3
    assert "Failed to parse terrain baseline evaluation artifact" in result.stdout


def test_export_final_inventory_fails_for_missing_candidates_artifact(tmp_path: Path) -> None:
    evaluation_path = _write_evaluation_artifact(tmp_path, rows=[])

    result = runner.invoke(
        app,
        [
            "export-final-inventory",
            str(evaluation_path),
            str(tmp_path / "missing_candidates_summary.json"),
        ],
    )

    assert result.exit_code == 2
    assert "Terrain candidates artifact not found" in result.stdout


def test_export_final_inventory_fails_for_unmatched_scored_row(tmp_path: Path) -> None:
    vector_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "train_source__candidates.geojson"
    )
    _write_candidate_vector(
        vector_path,
        [
            {
                **_candidate_record(
                    candidate_id="train_source__cand_9999",
                    aoi_id="train_aoi_01",
                    split="train",
                    source_raster_stem="train_source",
                    output_vector_path=vector_path,
                    pixel_count=20,
                    mean_local_relief=2.5,
                    max_local_relief=3.0,
                ),
                "geometry": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            }
        ],
    )
    candidates_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_source",
                candidate_vector_path=vector_path,
                candidate_count=1,
            )
        ],
        records=[
            _candidate_record(
                candidate_id="train_source__cand_9999",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_source",
                output_vector_path=vector_path,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            )
        ],
    )
    evaluation_path = _write_evaluation_artifact(
        tmp_path,
        [
            _evaluation_row(
                candidate_id="train_source__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_source",
                output_vector_path=vector_path,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
                matched_label_id="label_train_0001",
                best_iou=0.8,
                target_label=1,
                score=0.91,
                predicted_label=1,
            )
        ],
    )

    result = runner.invoke(
        app,
        ["export-final-inventory", str(evaluation_path), str(candidates_path)],
    )

    assert result.exit_code == 3
    assert "Failed to match scored evaluation rows back to candidate geometries" in result.stdout


def test_export_final_inventory_fails_for_unreadable_candidate_vector(tmp_path: Path) -> None:
    vector_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "train_source__candidates.geojson"
    )
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    vector_path.write_text("{bad-geojson", encoding="utf-8")

    candidates_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_source",
                candidate_vector_path=vector_path,
                candidate_count=1,
            )
        ],
        records=[
            _candidate_record(
                candidate_id="train_source__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_source",
                output_vector_path=vector_path,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            )
        ],
    )
    evaluation_path = _write_evaluation_artifact(
        tmp_path,
        [
            _evaluation_row(
                candidate_id="train_source__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_source",
                output_vector_path=vector_path,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
                matched_label_id="label_train_0001",
                best_iou=0.9,
                target_label=1,
                score=0.91,
                predicted_label=1,
            )
        ],
    )

    result = runner.invoke(
        app,
        ["export-final-inventory", str(evaluation_path), str(candidates_path)],
    )

    assert result.exit_code == 3
    assert "Candidate vector could not be read for evaluation" in result.stdout


def test_export_final_inventory_fails_for_old_rows_artifact_missing_hard_negative_columns(
    tmp_path: Path,
) -> None:
    vector_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "train_source__candidates.geojson"
    )
    _write_candidate_vector(
        vector_path,
        [
            {
                **_candidate_record(
                    candidate_id="train_source__cand_0001",
                    aoi_id="train_aoi_01",
                    split="train",
                    source_raster_stem="train_source",
                    output_vector_path=vector_path,
                    pixel_count=20,
                    mean_local_relief=2.5,
                    max_local_relief=3.0,
                ),
                "geometry": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            }
        ],
    )
    candidates_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_source",
                candidate_vector_path=vector_path,
                candidate_count=1,
            )
        ],
        records=[
            _candidate_record(
                candidate_id="train_source__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_source",
                output_vector_path=vector_path,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
            )
        ],
    )
    evaluation_path = _write_evaluation_artifact(
        tmp_path,
        [
            _evaluation_row(
                candidate_id="train_source__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                source_raster_stem="train_source",
                output_vector_path=vector_path,
                pixel_count=20,
                mean_local_relief=2.5,
                max_local_relief=3.0,
                matched_label_id="label_train_0001",
                best_iou=0.9,
                matched_negative_label_id="label_train_neg_0001",
                best_negative_iou=0.82,
                is_hard_negative_match=1,
                target_label=1,
                score=0.91,
                predicted_label=1,
            )
        ],
    )

    rows_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "evaluation"
        / "terrain_baseline_rows.csv"
    )
    with rows_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    legacy_fieldnames = [
        field
        for field in rows[0].keys()
        if field
        not in {
            "matched_negative_label_id",
            "best_negative_iou",
            "is_hard_negative_match",
        }
    ]
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=legacy_fieldnames)
        writer.writeheader()
        writer.writerows(
            [{field: row[field] for field in legacy_fieldnames} for row in rows]
        )

    result = runner.invoke(
        app,
        ["export-final-inventory", str(evaluation_path), str(candidates_path)],
    )

    assert result.exit_code == 3
    assert "Terrain baseline rows artifact is missing required hard-negative diagnostic" in (
        result.stdout
    )
    assert "final inventory export" in result.stdout
    assert "matched_negative_label_id" in result.stdout
    assert "best_negative_iou" in result.stdout
    assert "is_hard_negative_match" in result.stdout
