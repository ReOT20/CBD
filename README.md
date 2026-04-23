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
- smoke tests

Planned next:
- final inventory export

## Development checks

When changing code in this repository, run:

```bash
ruff check src tests
basedpyright --level error
PYTHONPATH=src python -m pytest -q
```

These checks are the default verification path for CLI, data-layer, and pipeline-stage changes.
