# Feature GUI v0.56.0 — ML Export Builder foundation

This patch moves ML-ready feature export responsibility into the Feature Analysis GUI. The acoustic GUI remains responsible for acoustic feature computation, and the kinematics GUI remains responsible for kinematic feature/aggregation outputs. The Feature GUI now serves as the curation bridge that prepares model-facing tables.

## Why this belongs in Feature GUI

The acoustic output is already a per-file scalar feature table with rich registries and audit tables. It does not have, and does not need, an acoustic aggregation tab. The kinematic output can include aggregated per-video scalar tables. The Feature GUI is the correct place to harmonize these different outputs, classify columns, attach family/role metadata, and write ML-ready exports.

## New GUI page

A new sidebar page is added:

- **ML Export Builder**

It accepts:

- `acoustic_features_per_file.csv`
- `selected_acoustic_feature_registry.csv`
- `acoustic_feature_measurement_scale_registry.csv`
- `kinematic_aggregated_features.csv`
- optional `kinematic_feature_manifest.csv`
- optional metadata/labels CSV

It writes outputs under:

```text
<output>/feature_analysis/ml_export_builder/tables/
```

## Outputs

Core outputs:

- `acoustic_ml_ready.csv`
- `acoustic_features_only.csv`
- `kinematic_ml_ready.csv`
- `kinematic_features_only.csv`
- `multimodal_early_fusion_ml_ready.csv` when a safe shared key exists
- `feature_manifest_unified.csv`
- `row_alignment_report.csv`
- `row_exclusions.csv`
- `ml_export_manifest.json`

## Design boundary

This stage does not train models. It also does not impute, scale, optimize, split, or select features inside validation folds. Those operations belong in the later ML GUI to avoid leakage.

## Multimodal joining

Early fusion is created only when a safe shared row key exists. Supported key patterns are:

1. `record_key`
2. `subject_id + session_id + task`
3. `subject_id + visit_id + task`
4. `subject_id + task`
5. `subject_id`

If no safe key exists, the acoustic-only and kinematic-only exports are still written, and the early-fusion table is left empty with an explanation in `row_exclusions.csv`.

## Important next step

For real multimodal ML, provide a metadata/labels table that maps acoustic and kinematic recordings to common IDs such as:

- `subject_id`
- `session_id` or `visit_id`
- `task`
- `diagnosis`
- `severity_score`
- `recording_date` or `days_from_baseline`

This enables reliable row alignment and subject-grouped ML validation.
