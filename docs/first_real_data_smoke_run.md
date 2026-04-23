# First Real-Data Smoke Run

This run is the first narrow integration test on actual data for the terrain-first MVP.

Use it to confirm that the current pipeline works on real files and produces interpretable artifacts.

## Scope

Use only:
- one `train` AOI
- one `val` AOI
- `nc_dem_10m_opentopography`
- Carolina Bays labels

Do not use yet:
- Sentinel-2
- `nc_lidar_1m_validation`
- `copernicus_dem_30m_fallback`
- broad multi-AOI runs

Optional first auxiliary pass:
- wetlands context may now be derived after terrain candidate generation
- this remains a secondary path and is not required for the terrain-only smoke run

## Required Inputs

Prepare these local workspace files before running the pipeline:
- `data/raw/dem/nc_10m/` with one or more NC 10 m DEM TIFFs
- `data/raw/labels/carolina_bays.geojson`
- `data/raw/aoi/train_aoi_01.geojson`
- `data/raw/aoi/val_aoi_01.geojson`

The dedicated manifests and config for this run are:
- `manifests/real_data_smoke_data_manifest.yaml`
- `manifests/real_data_smoke_aoi_manifest.yaml`
- `configs/real_data_smoke.yaml`

Important:
- run these commands from the `CBD/` repo root
- keep manifest files directly under `manifests/`
- the current terrain CLI infers the project root from the manifest directory name
- moving them into a nested subdirectory will break relative path resolution unless code changes

## Commands

Normalize labels with split assignment from the AOI manifest:

```bash
PYTHONPATH=src python -m cbd.cli normalize-labels-by-aoi \
  data/raw/labels/carolina_bays.geojson \
  manifests/real_data_smoke_aoi_manifest.yaml \
  data/interim/real_data_smoke/labels/normalized_labels.geojson \
  --source-id carolina_bays_labels
```

Normalize the AOIs into the exact paths referenced by the real-data smoke AOI manifest:

```bash
PYTHONPATH=src python -m cbd.cli normalize-aoi \
  data/raw/aoi/train_aoi_01.geojson \
  data/interim/real_data_smoke/aois/train_aoi_01.geojson \
  --aoi-id train_aoi_01 \
  --split train

PYTHONPATH=src python -m cbd.cli normalize-aoi \
  data/raw/aoi/val_aoi_01.geojson \
  data/interim/real_data_smoke/aois/val_aoi_01.geojson \
  --aoi-id val_aoi_01 \
  --split val
```

Validate manifests:

```bash
PYTHONPATH=src python -m cbd.cli validate-manifests \
  manifests/real_data_smoke_data_manifest.yaml \
  manifests/real_data_smoke_aoi_manifest.yaml
```

Run the terrain-first pipeline:

```bash
PYTHONPATH=src python -m cbd.cli resolve-terrain-inputs \
  manifests/real_data_smoke_data_manifest.yaml \
  manifests/real_data_smoke_aoi_manifest.yaml \
  --output-path outputs/real_data_smoke/interim/terrain/terrain_input_resolution.json

PYTHONPATH=src python -m cbd.cli preprocess-terrain \
  outputs/real_data_smoke/interim/terrain/terrain_input_resolution.json \
  --output-root outputs/real_data_smoke/interim/terrain

PYTHONPATH=src python -m cbd.cli derive-terrain-features \
  outputs/real_data_smoke/interim/terrain/terrain_preprocessing_summary.json \
  --output-root outputs/real_data_smoke/interim/terrain

PYTHONPATH=src python -m cbd.cli generate-terrain-candidates \
  outputs/real_data_smoke/interim/terrain/terrain_derivatives_summary.json \
  --output-root outputs/real_data_smoke/interim/terrain

PYTHONPATH=src python -m cbd.cli prepare-terrain-review \
  outputs/real_data_smoke/interim/terrain/terrain_candidates_summary.json \
  --output-root outputs/real_data_smoke/interim/terrain

PYTHONPATH=src python -m cbd.cli evaluate-terrain-baseline \
  outputs/real_data_smoke/interim/terrain/terrain_candidates_summary.json \
  data/interim/real_data_smoke/labels/normalized_labels.geojson \
  --output-root outputs/real_data_smoke/interim/terrain/evaluation

PYTHONPATH=src python -m cbd.cli export-final-inventory \
  outputs/real_data_smoke/interim/terrain/evaluation/terrain_baseline_evaluation_summary.json \
  outputs/real_data_smoke/interim/terrain/terrain_candidates_summary.json \
  --output-root outputs/real_data_smoke/final/terrain
```

Optional wetlands-context branch:

```bash
PYTHONPATH=src python -m cbd.cli derive-context-features \
  outputs/real_data_smoke/interim/terrain/terrain_candidates_summary.json \
  manifests/real_data_smoke_data_manifest.yaml \
  --context-source-id wetlands \
  --output-root outputs/real_data_smoke/interim/terrain

PYTHONPATH=src python -m cbd.cli prepare-terrain-review \
  outputs/real_data_smoke/interim/terrain/terrain_candidates_with_wetlands_summary.json \
  --output-root outputs/real_data_smoke/interim/terrain

PYTHONPATH=src python -m cbd.cli evaluate-terrain-baseline \
  outputs/real_data_smoke/interim/terrain/terrain_candidates_with_wetlands_summary.json \
  data/interim/real_data_smoke/labels/normalized_labels.geojson \
  --output-root outputs/real_data_smoke/interim/terrain/evaluation

PYTHONPATH=src python -m cbd.cli export-final-inventory \
  outputs/real_data_smoke/interim/terrain/evaluation/terrain_baseline_evaluation_summary.json \
  outputs/real_data_smoke/interim/terrain/terrain_candidates_with_wetlands_summary.json \
  --output-root outputs/real_data_smoke/final/terrain
```

## What To Inspect

Inspect these artifacts first:
- `outputs/real_data_smoke/interim/terrain/review/`
- `outputs/real_data_smoke/interim/terrain/evaluation/terrain_baseline_rows.csv`
- `outputs/real_data_smoke/interim/terrain/evaluation/terrain_baseline_metrics.json`
- `outputs/real_data_smoke/final/terrain/terrain_final_inventory.geojson`
- `outputs/real_data_smoke/final/terrain/terrain_final_inventory.csv`
- `outputs/real_data_smoke/final/terrain/terrain_final_inventory_summary.json`

Check for:
- non-empty candidate output on at least one AOI
- candidate density that is reviewable by a human
- candidates appearing in plausible terrain locations
- at least some known positives having nearby or overlapping candidates
- correct `split`, `aoi_id`, `terrain_source_id`, and `source_raster_stem`
- populated `score`, `predicted_label`, `target_label`, `matched_label_id`, and `best_iou`
- exported hard-negative diagnostics: `matched_negative_label_id`, `best_negative_iou`, `is_hard_negative_match`
- optional wetlands-context diagnostics when the context stage was used: `wetlands_any_overlap`, `wetlands_overlap_area`, `wetlands_overlap_fraction`
- final inventory row count matching scored evaluation rows
- both `train` and `val` rows appearing in normalized labels where expected

## Pass / Fail Rule

Treat the run as a pass if:
- every command completes
- there are no CRS or path failures
- train and validation outputs remain split-separated
- review artifacts are interpretable
- baseline evaluation and final inventory export complete without join failures

Treat the run as a failure if:
- labels or AOIs cannot be normalized cleanly
- manifest-relative paths do not resolve
- candidate generation is empty everywhere
- candidate volume is obviously unusable for review
- evaluation rows cannot be aligned back to candidate geometry
- final inventory export is missing geometry or score fields

## Hard-Negative Seeding

If the real-data run is dominated by high-score non-matching terrain structures, seed a first hard-negative layer from the final inventory:

```bash
PYTHONPATH=src python -m cbd.cli seed-hard-negatives \
  outputs/real_data_smoke/final/terrain/terrain_final_inventory.geojson \
  data/interim/real_data_smoke/labels/hard_negatives_seed.geojson \
  --split val \
  --source-id hard_negatives_seed \
  --target-crs EPSG:4269 \
  --top-n 25 \
  --min-score 0.05 \
  --min-pixels 50 \
  --min-max-local-relief 2.0
```

Review the exported negatives before treating them as curated truth.

`seed-hard-negatives` consumes a scored final inventory export. It is not a raw candidate-vector ingestion step.

Recommended artifact roles for this smoke workflow:
- `data/interim/real_data_smoke/labels/hard_negatives_seed.geojson`: machine-suggested review queue
- `data/interim/real_data_smoke/labels/reviewed_hard_negatives.geojson`: manually curated `negative_hard` rows only
- `data/interim/real_data_smoke/labels/normalized_labels_with_reviewed_hard_negatives.geojson`: derived evaluation input built from positives plus reviewed negatives

Do not treat auto-promoted or convenience-generated reviewed files as final curated truth. For the current smoke milestone, Task 5 was closed only after a human review pass produced a curated `reviewed_hard_negatives.geojson` and the reviewed rerun completed successfully.

To activate reviewed hard negatives in baseline training:

1. Open the seeded GeoJSON in a GIS editor.
2. Keep only clear non-target structures that are useful counterexamples.
3. Assign an explicit `error_category` to each approved row using the first-pass class set: `wetland`, `pond`, `oxbow`, `non_target_depression`, or `target_like_noise` when applicable.
4. Set `review_status=reviewed` for approved rows and leave unreviewed seeds as `seed` or `needs_review`.
5. Save the approved rows into `reviewed_hard_negatives.geojson`.
6. Merge the reviewed `negative_hard` rows into the same normalized labels file used by `evaluate-terrain-baseline`.
7. Rerun baseline evaluation with that mixed labels file.
8. Export a reviewed final inventory from the rerun evaluation artifact.

At this stage, reviewed hard negatives change train target assignment only. Validation metrics stay based on positive overlap vs non-positive candidates, while row exports and final inventory include hard-negative match diagnostics for analysis.

Current smoke-run result:
- `reviewed_hard_negatives.geojson` contains 9 reviewed negatives
- the current curated classes are `non_target_depression` and `target_like_noise`
- the reviewed rerun completed successfully with 9 hard-negative matches in the exported rows
- validation thresholded metrics remained unchanged, so the next problem is model/training behavior rather than absence of curated negatives

Final inventory export now also expects those diagnostic columns to already exist in `terrain_baseline_rows.csv`. If you point it at an older rows artifact without them, the export is expected to fail and the baseline evaluation must be rerun.

For a fuller review checklist and artifact discipline, use [hard_negative_curation.md](hard_negative_curation.md).

