# CBD

Implementation repository for the Carolina Bays DEM-first baseline / benchmark pipeline.

## Current scope

This repository contains the executable implementation state for:
- label ingestion
- AOI ingestion
- manifest validation
- future DEM-first candidate generation pipeline

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
- smoke tests

Planned next:
- DEM preprocessing
- morphometric derivatives
- terrain-driven candidate generation
- evaluation and export

## Development checks

When changing code in this repository, run:

```bash
ruff check src tests
basedpyright --level error
PYTHONPATH=src python -m pytest -q
```

These checks are the default verification path for CLI, data-layer, and pipeline-stage changes.
