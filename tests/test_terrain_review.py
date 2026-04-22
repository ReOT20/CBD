from __future__ import annotations

import csv
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon
from typer.testing import CliRunner

from cbd.cli import app

runner = CliRunner()


def _candidate_record(
    *,
    candidate_id: str,
    aoi_id: str,
    split: str,
    terrain_source_id: str,
    source_raster_stem: str,
    output_vector_path: Path,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "aoi_id": aoi_id,
        "split": split,
        "terrain_source_id": terrain_source_id,
        "source_raster_stem": source_raster_stem,
        "output_vector_path": str(output_vector_path.resolve()),
        "pixel_count": 4,
        "area_map_units": 4.0,
        "bbox_width": 2.0,
        "bbox_height": 2.0,
        "bbox_aspect_ratio": 1.0,
        "mean_local_relief": 1.55,
        "max_local_relief": 1.7,
        "mean_slope": 3.125,
        "max_slope": 3.5,
    }


def _vector_record(
    *,
    tmp_path: Path,
    aoi_id: str,
    split: str,
    terrain_source_id: str,
    source_raster_stem: str,
    candidate_vector_path: Path,
    candidate_count: int,
) -> dict[str, object]:
    return {
        "aoi_id": aoi_id,
        "split": split,
        "terrain_source_id": terrain_source_id,
        "source_raster_stem": source_raster_stem,
        "input_raster_path": str(
            (
                tmp_path
                / "outputs"
                / "interim"
                / "terrain"
                / "preprocessed"
                / aoi_id
                / terrain_source_id
                / f"{source_raster_stem}.tif"
            ).resolve()
        ),
        "candidate_vector_path": str(candidate_vector_path.resolve()),
        "candidate_count": candidate_count,
    }


def _write_candidate_vector(
    path: Path,
    *,
    with_candidate: bool,
    aoi_id: str = "train_aoi_01",
    split: str = "train",
    terrain_source_id: str = "nc_dem_10m_opentopography",
    source_raster_stem: str = "tile_001",
    candidate_id: str = "tile_001__cand_0001",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if with_candidate:
        gdf = gpd.GeoDataFrame(
            {
                "candidate_id": [candidate_id],
                "aoi_id": [aoi_id],
                "split": [split],
                "terrain_source_id": [terrain_source_id],
                "source_raster_stem": [source_raster_stem],
                "pixel_count": [4],
                "area_map_units": [4.0],
                "bbox_width": [2.0],
                "bbox_height": [2.0],
                "bbox_aspect_ratio": [1.0],
                "mean_local_relief": [1.55],
                "max_local_relief": [1.7],
                "mean_slope": [3.125],
                "max_slope": [3.5],
            },
            geometry=[Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])],
            crs="EPSG:4326",
        )
    else:
        gdf = gpd.GeoDataFrame(
            {
                "candidate_id": [],
                "aoi_id": [],
                "split": [],
                "terrain_source_id": [],
                "source_raster_stem": [],
                "pixel_count": [],
                "area_map_units": [],
                "bbox_width": [],
                "bbox_height": [],
                "bbox_aspect_ratio": [],
                "mean_local_relief": [],
                "max_local_relief": [],
                "mean_slope": [],
                "max_slope": [],
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
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


def test_prepare_terrain_review_success(tmp_path: Path) -> None:
    vector_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__candidates.geojson"
    )
    _write_candidate_vector(vector_path, with_candidate=True)
    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                terrain_source_id="nc_dem_10m_opentopography",
                source_raster_stem="tile_001",
                candidate_vector_path=vector_path,
                candidate_count=1,
            )
        ],
        records=[
            _candidate_record(
                candidate_id="tile_001__cand_0001",
                aoi_id="train_aoi_01",
                split="train",
                terrain_source_id="nc_dem_10m_opentopography",
                source_raster_stem="tile_001",
                output_vector_path=vector_path,
            )
        ],
    )

    result = runner.invoke(app, ["prepare-terrain-review", str(artifact_path)])

    assert result.exit_code == 0
    assert "Terrain review artifact preparation completed successfully" in result.stdout

    summary_path = tmp_path / "outputs" / "interim" / "terrain" / "terrain_review_summary.json"
    overall_table_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "review"
        / "terrain_candidate_review_table.csv"
    )
    per_raster_table_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "review"
        / "train"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__review_table.csv"
    )
    overlay_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "review"
        / "train"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "terrain_candidate_review_overlay.geojson"
    )

    assert summary_path.exists()
    assert overall_table_path.exists()
    assert per_raster_table_path.exists()
    assert overlay_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["total_candidate_rows"] == 1
    assert payload["total_review_tables_written"] == 2
    assert payload["total_review_overlays_written"] == 1

    with overall_table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "tile_001__cand_0001"

    overlay_gdf = gpd.read_file(overlay_path)
    assert len(overlay_gdf) == 1
    assert overlay_gdf.loc[0, "terrain_source_id"] == "nc_dem_10m_opentopography"


def test_prepare_terrain_review_empty_candidates(tmp_path: Path) -> None:
    vector_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__candidates.geojson"
    )
    _write_candidate_vector(vector_path, with_candidate=False)
    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                terrain_source_id="nc_dem_10m_opentopography",
                source_raster_stem="tile_001",
                candidate_vector_path=vector_path,
                candidate_count=0,
            )
        ],
        records=[],
    )

    result = runner.invoke(app, ["prepare-terrain-review", str(artifact_path)])

    assert result.exit_code == 0
    summary_path = tmp_path / "outputs" / "interim" / "terrain" / "terrain_review_summary.json"
    overall_table_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "review"
        / "terrain_candidate_review_table.csv"
    )
    overlay_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "review"
        / "train"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "terrain_candidate_review_overlay.geojson"
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["total_candidate_rows"] == 0
    assert payload["total_review_tables_written"] == 1
    assert payload["total_review_overlays_written"] == 1

    with overall_table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == []

    overlay_gdf = gpd.read_file(overlay_path)
    assert overlay_gdf.empty


def test_prepare_terrain_review_preserves_split_in_output_paths(tmp_path: Path) -> None:
    train_vector_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "shared_aoi"
        / "nc_dem_10m_opentopography"
        / "tile_001__train_candidates.geojson"
    )
    val_vector_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "shared_aoi"
        / "nc_dem_10m_opentopography"
        / "tile_001__val_candidates.geojson"
    )
    _write_candidate_vector(
        train_vector_path,
        with_candidate=True,
        aoi_id="shared_aoi",
        split="train",
        candidate_id="tile_001__cand_train",
    )
    _write_candidate_vector(
        val_vector_path,
        with_candidate=True,
        aoi_id="shared_aoi",
        split="val",
        candidate_id="tile_001__cand_val",
    )
    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="shared_aoi",
                split="train",
                terrain_source_id="nc_dem_10m_opentopography",
                source_raster_stem="tile_001",
                candidate_vector_path=train_vector_path,
                candidate_count=1,
            ),
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="shared_aoi",
                split="val",
                terrain_source_id="nc_dem_10m_opentopography",
                source_raster_stem="tile_001",
                candidate_vector_path=val_vector_path,
                candidate_count=1,
            ),
        ],
        records=[
            _candidate_record(
                candidate_id="tile_001__cand_train",
                aoi_id="shared_aoi",
                split="train",
                terrain_source_id="nc_dem_10m_opentopography",
                source_raster_stem="tile_001",
                output_vector_path=train_vector_path,
            ),
            _candidate_record(
                candidate_id="tile_001__cand_val",
                aoi_id="shared_aoi",
                split="val",
                terrain_source_id="nc_dem_10m_opentopography",
                source_raster_stem="tile_001",
                output_vector_path=val_vector_path,
            ),
        ],
    )

    result = runner.invoke(app, ["prepare-terrain-review", str(artifact_path)])

    assert result.exit_code == 0

    train_table_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "review"
        / "train"
        / "shared_aoi"
        / "nc_dem_10m_opentopography"
        / "tile_001__review_table.csv"
    )
    val_table_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "review"
        / "val"
        / "shared_aoi"
        / "nc_dem_10m_opentopography"
        / "tile_001__review_table.csv"
    )
    train_overlay_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "review"
        / "train"
        / "shared_aoi"
        / "nc_dem_10m_opentopography"
        / "terrain_candidate_review_overlay.geojson"
    )
    val_overlay_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "review"
        / "val"
        / "shared_aoi"
        / "nc_dem_10m_opentopography"
        / "terrain_candidate_review_overlay.geojson"
    )

    assert train_table_path.exists()
    assert val_table_path.exists()
    assert train_overlay_path.exists()
    assert val_overlay_path.exists()

    summary_path = tmp_path / "outputs" / "interim" / "terrain" / "terrain_review_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    logical_table_paths = {
        record["output_table_path"]
        for record in payload["tables"]
        if record["aoi_id"] == "shared_aoi"
    }
    logical_overlay_paths = {
        record["output_vector_path"]
        for record in payload["overlays"]
        if record["aoi_id"] == "shared_aoi"
    }
    assert logical_table_paths == {str(train_table_path), str(val_table_path)}
    assert logical_overlay_paths == {str(train_overlay_path), str(val_overlay_path)}


def test_prepare_terrain_review_fails_for_missing_artifact(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["prepare-terrain-review", str(tmp_path / "missing_candidates_summary.json")],
    )

    assert result.exit_code == 2
    assert "Terrain candidates artifact not found" in result.stdout


def test_prepare_terrain_review_fails_for_malformed_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "terrain_candidates_summary.json"
    artifact_path.write_text("{bad-json", encoding="utf-8")

    result = runner.invoke(app, ["prepare-terrain-review", str(artifact_path)])

    assert result.exit_code == 3
    assert "Failed to parse terrain candidates artifact" in result.stdout


def test_prepare_terrain_review_fails_for_missing_candidate_vector(tmp_path: Path) -> None:
    vector_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__candidates.geojson"
    )
    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                terrain_source_id="nc_dem_10m_opentopography",
                source_raster_stem="tile_001",
                candidate_vector_path=vector_path,
                candidate_count=0,
            )
        ],
        records=[],
    )

    result = runner.invoke(app, ["prepare-terrain-review", str(artifact_path)])

    assert result.exit_code == 3
    assert "Candidate vector path not found for review" in result.stdout


def test_prepare_terrain_review_fails_for_unreadable_candidate_vector(tmp_path: Path) -> None:
    vector_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__candidates.geojson"
    )
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    vector_path.write_text("{not-valid-geojson", encoding="utf-8")
    artifact_path = _write_candidates_artifact(
        tmp_path,
        vectors=[
            _vector_record(
                tmp_path=tmp_path,
                aoi_id="train_aoi_01",
                split="train",
                terrain_source_id="nc_dem_10m_opentopography",
                source_raster_stem="tile_001",
                candidate_vector_path=vector_path,
                candidate_count=0,
            )
        ],
        records=[],
    )

    result = runner.invoke(app, ["prepare-terrain-review", str(artifact_path)])

    assert result.exit_code == 3
    assert "Candidate vector could not be read for review" in result.stdout
    assert "tile_001__candidates.geojson" in result.stdout.replace("\n", "")
