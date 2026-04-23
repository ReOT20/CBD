# label_schema.md

## Purpose

This document defines the label semantics used by the MVP.

## Core principle

Labels describe observable morphology only.
They do not encode causal geological claims.

## Primary classes

- `positive_complete`
  - clear closed or near-closed Carolina Bay-like structure

- `positive_partial`
  - plausible but incomplete or degraded structure

- `ambiguous`
  - uncertain case that should not be treated as a clean positive

- `negative_hard`
  - clear non-target with target-like appearance

## Geometry

Allowed geometry types for baseline:
- polygon
- ellipse-like polygon approximation
- bounding polygon derived from source labels

The baseline inventory may later export simplified ellipse approximations, but source normalization must preserve the original geometry where available.

## Required fields

Each normalized label record should contain:
- `label_id`
- `class_name`
- `source_id`
- `split`
- `geometry`
- `review_status`
- `notes`

## Optional fields

- `source_record_id`
- `source_record_attrs`
- `confidence`
- `reviewer`
- `review_date`
- `error_category`
- `parent_source_record`

`source_record_id` should be a deterministic back-reference to the original source-row identity used during normalization. The current normalization commands produce it, but downstream stages should not require it for label loading.

`source_record_attrs` should preserve a compact serialized snapshot of the original non-geometry source attributes. The current normalization commands produce it, but downstream stages should not depend on arbitrary raw source fields directly; they should continue to use the normalized semantic fields above.

For seeded review artifacts, `parent_source_record` should preserve stable provenance back to the scored inventory row. At minimum it should include the candidate split and candidate identity, and it may also include AOI or terrain-source context when available.

For curated `negative_hard` rows, prefer explicit `error_category` values that reflect the first false-positive taxonomy used in review:
- `wetland`
- `pond`
- `oxbow`
- `non_target_depression`
- `target_like_noise`

For the first real-data smoke curation pass, the maintained reviewed set currently uses `non_target_depression` and `target_like_noise`.

## Review status values

- `seed`
- `reviewed`
- `needs_review`

## Notes

`ambiguous` must not be silently converted into a positive or negative class.
Any remapping must be explicit in code and config.
For the current terrain baseline milestone, only reviewed `negative_hard` labels are active, and they are used only for train-time supervision. Seeded negatives are analysis artifacts.

