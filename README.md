# CBD

Implementation repository for the Carolina Bays DEM-first baseline / benchmark pipeline.

## Current scope

This repository contains the executable implementation state for:
- label ingestion
- AOI ingestion
- manifest validation
- DEM-first terrain preprocessing and derivatives
- terrain-driven candidate generation
- review artifact export
- baseline evaluation artifact export
- final inventory export

## Method summary

Current MVP direction:
- DEM-first / LiDAR-first
- Carolina Bays as the primary reference dataset
- geographic hold-out evaluation
- Sentinel-2 used as auxiliary context, not primary modality

## Current status

Implemented:
- package bootstrap
- CLI
- manifest validation
- label normalization
- AOI normalization
- DEM / LiDAR input resolution
- DEM preprocessing
- morphometric derivative generation
- terrain-driven candidate generation
- review artifact export
- baseline evaluation artifact export
- final inventory export
- smoke tests

## Development checks

When changing code in this repository, run:

```bash
ruff check src tests
basedpyright --level error
PYTHONPATH=src python -m pytest -q
```

These checks are the default verification path for CLI, data-layer, and pipeline-stage changes.

## Repo Layout

This public repo tracks the implementation contract for data and runs:
- `manifests/` contains checked-in manifest examples and smoke-run entrypoints
- `docs/` contains implementation-facing runbooks and schema notes
- `configs/` contains repo-root-relative config examples

This repo does not track full runtime datasets or generated artifacts. Keep these local workspace
paths under `CBD/`:
- `data/raw/`
- `data/interim/`
- `data/cache/`
- `outputs/`

Only tiny placeholders and test fixtures should be committed there.

## First Real-Data Run

For the first narrow real-data MVP integration run, use:
- [`configs/real_data_smoke.yaml`](configs/real_data_smoke.yaml)
- [`docs/first_real_data_smoke_run.md`](docs/first_real_data_smoke_run.md)
- [`manifests/real_data_smoke_data_manifest.yaml`](manifests/real_data_smoke_data_manifest.yaml)
- [`manifests/real_data_smoke_aoi_manifest.yaml`](manifests/real_data_smoke_aoi_manifest.yaml)

Run the commands from the `CBD/` repo root. The checked-in config and runbook use internal
repo-relative paths and should not depend on an outer thesis repo layout.

That run is intentionally limited to:
- one train AOI
- one val AOI
- NC 10 m DEM only
- normalized labels

Related implementation docs:
- [`docs/label_schema.md`](docs/label_schema.md)
- [`docs/hard_negative_curation.md`](docs/hard_negative_curation.md)
