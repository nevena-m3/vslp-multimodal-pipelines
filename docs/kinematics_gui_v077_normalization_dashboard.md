# Kinematics GUI v0.77 - Normalization Dashboard

This patch refines the Normalization menu so the stage is educational, auditable, and scientifically explicit before downstream feature computation.

## Rationale

Normalization is required because raw MediaPipe coordinates are image/model coordinates. They are useful for tracking relative movement, but they remain affected by camera distance, face size, and recording setup. The default intercanthal/inner-eye method centers coordinates and divides by a stable eye-corner denominator, producing face-scale units for oral-motor kinematics.

This does not create true millimeter displacement and does not correct head rotation, depth motion, occlusion, or tracking failure. Those limitations are now visible in the GUI and must be reviewed with QC.

## GUI changes

The Normalization tab now includes:

- normalization readiness cards;
- a compact schematic visual that updates with the selected method;
- method-specific explanation of what changes, best use, caution, and evidence level;
- an anchor/method audit panel;
- a clearer output diagnostics table with scale source, scale valid percentage, scale CV, max scale jump, selected landmark completeness, and QC flags;
- refresh support for existing normalization results;
- warning stage status when normalization completed with QC flags or errors.

## Scientific status of default methods

| Method | Status |
|---|---|
| intercanthal_distance | Preferred research default for ALS/oral-motor kinematics; uses 133/362 with 33/263 fallback. |
| interpupillary_or_outer_eye | Fallback visual audit proxy; more pose-sensitive. |
| face_bbox_width | Engineering fallback when named anchors are unreliable. |
| face_height_nose_chin | Exploratory vertical scale; can be contaminated by lower-face motion. |
| procrustes_head_stabilized | Future placeholder; full rigid stabilization is not yet active. |
| raw_normalized_coordinates | Audit/debug only; not recommended for final biomarkers. |

## Validation

Validated in the development sandbox with:

```text
PYTHONPATH=src python -m pytest tests/unit/test_kinematics_*.py -q
47 passed

PYTHONPATH=src python -m pytest tests/integration/test_acoustic_preprocess_stage.py -q
1 passed
```
