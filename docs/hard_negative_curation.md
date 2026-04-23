# Hard Negative Curation

This document defines the minimum reproducible workflow for completing Task 5 in the real-data smoke setup.

## Purpose

Use this workflow to turn seeded false-positive candidates into a maintained reviewed `negative_hard` dataset.

The goal is not to auto-promote rows. The goal is to create one manually confirmed artifact that can be reused across reruns.

## Canonical Artifacts

Use these files with distinct roles:
- `data/interim/real_data_smoke/labels/hard_negatives_seed.geojson`: machine-suggested review queue
- `data/interim/real_data_smoke/labels/reviewed_hard_negatives.geojson`: manually approved negatives only
- `data/interim/real_data_smoke/labels/normalized_labels_with_reviewed_hard_negatives.geojson`: derived evaluation input

Only `reviewed_hard_negatives.geojson` should be treated as the maintained negative dataset.

## Review Rules

Approve a row only if it is a clear non-target structure that is useful as a counterexample for the terrain-first baseline.

Reject or leave as `needs_review` when:
- the geometry appears to overlap a plausible Carolina Bay
- the morphology is too ambiguous to support confident negative labeling
- the candidate is mostly an artifact of poor geometry or obvious preprocessing error

When approving a row:
- keep `class_name=negative_hard`
- set `review_status=reviewed`
- preserve `parent_source_record`
- preserve geometry unless there is a documented reason to refine it
- assign one explicit `error_category`

## First-Pass Error Categories

Use these categories where applicable:
- `wetland`
- `pond`
- `oxbow`
- `non_target_depression`
- `target_like_noise`

If none fit cleanly, keep the row out of the curated set until the taxonomy is revised.

## Rerun Workflow

1. Generate or refresh `hard_negatives_seed.geojson` from the current final inventory.
2. Review the seeded rows in GIS.
3. Save approved rows into `reviewed_hard_negatives.geojson`.
4. Build `normalized_labels_with_reviewed_hard_negatives.geojson` from the normalized positives plus the reviewed negatives.
5. Rerun `evaluate-terrain-baseline`.
6. Rerun `export-final-inventory`.
7. Inspect whether hard-negative diagnostics and ranking errors become easier to interpret.

## Completion Criteria For Task 5

Task 5 is complete only when:
- a curated reviewed-negative artifact exists
- at least one explicit false-positive class is represented
- the artifact follows the documented schema
- the baseline has been rerun with that manually curated artifact

## Current Smoke-Run Status

The first smoke-run curation cycle is complete:
- `reviewed_hard_negatives.geojson` contains 9 manually reviewed negatives
- the current curated classes are `non_target_depression` and `target_like_noise`
- the derived merged labels file was rebuilt from the normalized positives plus those reviewed negatives
- `evaluate-terrain-baseline` and `export-final-inventory` both succeeded on the reviewed rerun artifacts
