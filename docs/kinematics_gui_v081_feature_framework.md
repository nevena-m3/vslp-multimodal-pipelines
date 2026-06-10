# Kinematics GUI v0.81 — Video QC placeholder refinement and feature framework start

## Purpose

This patch keeps the current automated video/landmark QC separate from the future acoustic-style kinematics QC framework, and begins organizing the Feature stage using the uploaded ALS/video kinematic field maps.

## Video QC change

The previous QC placeholder had too much detail for a future framework that is not yet implemented. It is now deliberately a degradation-family scaffold only:

- lighting / exposure
- frame freezing / dropped frames
- camera instability / motion
- multiple people / face confusion
- occlusion / cropping
- pose / face angle
- focus / resolution
- task compliance / usable behavior

Each row is marked `placeholder_to_be_built_in`. No thresholds, plots, reviewer workflow, or acoustic-style final QC report are implied.

The implemented QC remains limited to automated extraction-risk summaries:

- face-detected fraction
- no-face gap structure
- selected-landmark frame-to-frame displacement
- conservative pass/review/fail triage

## Feature-stage change

The Feature tab now distinguishes:

1. the current implemented oral-motor computational kernel; and
2. the broader field-map roadmap from the uploaded ALS/video kinematic maps.

The GUI now shows feature framework/provenance tables before the selector. This is intentionally honest: the uploaded maps describe a larger feature universe, but the current backend should not pretend full 65-feature field-map parity until each feature is implemented, tested, and validated.

## New feature provenance outputs

The feature stage can now write:

```text
kinematics/006_features/tables/kinematic_feature_framework.csv
kinematics/006_features/tables/kinematic_feature_implementation_audit.csv
kinematics/006_features/tables/kinematic_feature_framework.json
```

These files document feature-family coverage and current implementation status. They are not diagnostic outputs.

## Scientific caution

The uploaded feature map uses a legacy/field-map ICD convention based on landmarks 243/463. The current GUI normalization convention uses 133/362 as preferred inner-canthus anchors with 33/263 fallback. This difference is now explicitly surfaced as an implementation-audit item so the project can lock the convention later instead of hiding it.

Disease relevance comes from the feature family and task design, not from a single landmark index or scalar value. ALS/PD thresholds require labeled validation data and are not implemented in this GUI.
