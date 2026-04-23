from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import yaml
from shapely.geometry import Polygon
from typer.testing import CliRunner

from cbd.cli import app

runner = CliRunner()


def _write_candidate_vector(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {
            "candidate_id": ["tile_001__cand_0001", "tile_001__cand_0002"],
            "aoi_id": ["train_aoi_01", "train_aoi_01"],
            "split": ["train", "train"],
            "terrain_source_id": ["nc_dem_10m_opentopography", "nc_dem_10m_opentopography"],
            "source_raster_stem": ["tile_001", "tile_001"],
            "pixel_count": [10, 12],
            "area_map_units": [4.0, 4.0],
            "bbox_width": [2.0, 2.0],
            "bbox_height": [2.0, 2.0],
            "bbox_aspect_ratio": [1.0, 1.0],
            "mean_local_relief": [2.0, 1.0],
            "max_local_relief": [2.5, 1.5],
            "mean_slope": [1.0, 0.5],
            "max_slope": [1.25, 0.75],
        },
        geometry=[
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]),
        ],
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GeoJSON")


def _write_candidates_artifact(tmp_path: Path, candidate_vector_path: Path) -> Path:
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
        "total_input_groups_processed": 1,
        "total_candidate_vectors_written": 1,
        "total_candidates": 2,
        "relief_threshold": 1.0,
        "min_pixels": 4,
        "vectors": [
            {
                "aoi_id": "train_aoi_01",
                "split": "train",
                "terrain_source_id": "nc_dem_10m_opentopography",
                "source_raster_stem": "tile_001",
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
                "candidate_vector_path": str(candidate_vector_path.resolve()),
                "candidate_count": 2,
            }
        ],
        "records": [
            {
                "candidate_id": "tile_001__cand_0001",
                "aoi_id": "train_aoi_01",
                "split": "train",
                "terrain_source_id": "nc_dem_10m_opentopography",
                "source_raster_stem": "tile_001",
                "output_vector_path": str(candidate_vector_path.resolve()),
                "pixel_count": 10,
                "area_map_units": 4.0,
                "bbox_width": 2.0,
                "bbox_height": 2.0,
                "bbox_aspect_ratio": 1.0,
                "mean_local_relief": 2.0,
                "max_local_relief": 2.5,
                "mean_slope": 1.0,
                "max_slope": 1.25,
            },
            {
                "candidate_id": "tile_001__cand_0002",
                "aoi_id": "train_aoi_01",
                "split": "train",
                "terrain_source_id": "nc_dem_10m_opentopography",
                "source_raster_stem": "tile_001",
                "output_vector_path": str(candidate_vector_path.resolve()),
                "pixel_count": 12,
                "area_map_units": 4.0,
                "bbox_width": 2.0,
                "bbox_height": 2.0,
                "bbox_aspect_ratio": 1.0,
                "mean_local_relief": 1.0,
                "max_local_relief": 1.5,
                "mean_slope": 0.5,
                "max_slope": 0.75,
            },
        ],
    }
    artifact_path = (
        tmp_path / "outputs" / "interim" / "terrain" / "terrain_candidates_summary.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact_path


def _write_data_manifest(
    tmp_path: Path,
    *,
    source_id: str = "wetlands",
    source_type: str = "vector",
) -> Path:
    manifest = {
        "version": 2,
        "project": {"name": "carolina-bays-mvp"},
        "sources": [
            {
                "id": source_id,
                "role": "hard_negative_context",
                "enabled": True,
                "type": source_type,
                "format": "geojson_or_geopackage",
                "provider": "external_reference_dataset",
                "access": {"method": "local_project_files", "auth_required": False},
                "local": {"expected_path": "data/raw/context/wetlands.gpkg"},
                "notes": [],
            }
        ],
        "defaults": {},
        "requirements": {},
        "non_goals": [],
    }
    path = tmp_path / "manifests" / "data_manifest.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def test_derive_context_features_success_with_crs_reprojection(tmp_path: Path) -> None:
    candidate_vector_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__candidates.geojson"
    )
    _write_candidate_vector(candidate_vector_path)
    candidates_artifact = _write_candidates_artifact(tmp_path, candidate_vector_path)

    wetlands_path = tmp_path / "data" / "raw" / "context" / "wetlands.gpkg"
    wetlands_path.parent.mkdir(parents=True, exist_ok=True)
    wetlands = gpd.GeoDataFrame(
        {"name": ["wetland_001"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    wetlands.to_file(wetlands_path, driver="GPKG")

    manifest_path = _write_data_manifest(tmp_path)

    result = runner.invoke(
        app,
        [
            "derive-context-features",
            str(candidates_artifact),
            str(manifest_path),
        ],
    )

    assert result.exit_code == 0
    assert "Terrain context feature derivation completed successfully" in result.stdout

    summary_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "terrain_candidates_with_wetlands_summary.json"
    )
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["context_source_id"] == "wetlands"

    enriched_vector_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "context_candidates"
        / "wetlands"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__candidates.geojson"
    )
    enriched = gpd.read_file(enriched_vector_path)
    overlapping = enriched.loc[enriched["candidate_id"] == "tile_001__cand_0001"].iloc[0]
    disjoint = enriched.loc[enriched["candidate_id"] == "tile_001__cand_0002"].iloc[0]
    assert int(overlapping["wetlands_any_overlap"]) == 1
    assert float(overlapping["wetlands_overlap_area"]) > 0.0
    assert float(overlapping["wetlands_overlap_fraction"]) > 0.0
    assert int(disjoint["wetlands_any_overlap"]) == 0
    assert float(disjoint["wetlands_overlap_area"]) == 0.0
    assert float(disjoint["wetlands_overlap_fraction"]) == 0.0


def test_derive_context_features_fails_for_missing_context_source(tmp_path: Path) -> None:
    candidate_vector_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__candidates.geojson"
    )
    _write_candidate_vector(candidate_vector_path)
    candidates_artifact = _write_candidates_artifact(tmp_path, candidate_vector_path)
    manifest_path = _write_data_manifest(tmp_path, source_id="hydrology")

    result = runner.invoke(
        app,
        [
            "derive-context-features",
            str(candidates_artifact),
            str(manifest_path),
        ],
    )

    assert result.exit_code == 4
    assert "Context source 'wetlands' is not declared" in result.stdout


def test_derive_context_features_fails_for_non_vector_context_source(tmp_path: Path) -> None:
    candidate_vector_path = (
        tmp_path
        / "outputs"
        / "interim"
        / "terrain"
        / "candidates"
        / "train_aoi_01"
        / "nc_dem_10m_opentopography"
        / "tile_001__candidates.geojson"
    )
    _write_candidate_vector(candidate_vector_path)
    candidates_artifact = _write_candidates_artifact(tmp_path, candidate_vector_path)
    manifest_path = _write_data_manifest(tmp_path, source_type="raster_dem")

    result = runner.invoke(
        app,
        [
            "derive-context-features",
            str(candidates_artifact),
            str(manifest_path),
        ],
    )

    assert result.exit_code == 4
    assert "must be a vector source" in result.stdout
