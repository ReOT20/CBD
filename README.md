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

## First Real-Data Run

For the first narrow real-data MVP integration run, use:
- [real_data_smoke.yaml](</home/type7/Documents/spbu/thesis/CBD/configs/real_data_smoke.yaml>)
- [first_real_data_smoke_run.md](</home/type7/Documents/spbu/thesis/docs/first_real_data_smoke_run.md>)

That run is intentionally limited to:
- one train AOI
- one val AOI
- NC 10 m DEM only
- normalized labels
