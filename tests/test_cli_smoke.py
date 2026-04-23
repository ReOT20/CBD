from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from cbd.cli import app

runner = CliRunner()


def test_health_command() -> None:
    result = runner.invoke(app, ["health"])
    assert result.exit_code == 0
    assert "CBD CLI is available" in result.stdout


def test_validate_manifests_command(tmp_path: Path) -> None:
    data_manifest = {
        "version": 1,
        "project": {"name": "carolina-bays-mvp"},
        "sources": [
            {
                "id": "sentinel2_sr",
                "role": "imagery",
                "enabled": True,
                "type": "raster_multispectral",
                "format": "geotiff_or_stac_assets",
                "provider": "open_imagery",
                "access": {"method": "stac_or_prepared_local_assets", "auth_required": False},
                "local": {"expected_root": "data/raw/imagery/sentinel2"},
                "notes": [],
            }
        ],
        "defaults": {},
        "requirements": {},
        "non_goals": [],
    }

    aoi_manifest = {
        "version": 1,
        "aoi_sets": [
            {
                "id": "val_aoi_01",
                "split": "val",
                "geometry_path": "data/raw/aoi/val_aoi_01.geojson",
                "enabled": True,
                "notes": [],
            }
        ],
        "rules": {"geographic_holdout": True},
    }

    data_path = tmp_path / "data_manifest.yaml"
    aoi_path = tmp_path / "aoi_manifest.yaml"

    data_path.write_text(yaml.safe_dump(data_manifest), encoding="utf-8")
    aoi_path.write_text(yaml.safe_dump(aoi_manifest), encoding="utf-8")

    result = runner.invoke(app, ["validate-manifests", str(data_path), str(aoi_path)])

    assert result.exit_code == 0
    assert "Validation successful" in result.stdout
    assert "carolina-bays-mvp" in result.stdout


def test_seed_hard_negatives_help_describes_scored_inventory_input() -> None:
    result = runner.invoke(app, ["seed-hard-negatives", "--help"])

    assert result.exit_code == 0
    assert "Input scored final inventory GeoJSON/GeoPackage" in result.stdout
    assert "candidate vector file" not in result.stdout


def test_checked_in_configs_use_internal_repo_paths() -> None:
    config_root = Path(__file__).resolve().parents[1] / "configs"
    for config_name in ("mvp_baseline.yaml", "real_data_smoke.yaml"):
        payload = yaml.safe_load((config_root / config_name).read_text(encoding="utf-8"))
        path_values = payload.get("paths", {}).values()
        for value in path_values:
            assert isinstance(value, str)
            assert not value.startswith("../")
