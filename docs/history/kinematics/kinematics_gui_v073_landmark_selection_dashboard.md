# Kinematics GUI v0.73 - Landmark Selection Dashboard

This patch refines the Landmark Selection menu after the Setup and Landmark Extraction dashboard passes.

## Purpose

The Landmark Selection tab now acts as an anatomical selection workstation rather than only a list of landmark IDs. The GUI still uses the real video frame plus actual MediaPipe overlay as the primary selection surface. This patch adds auditable selection diagnostics so the user can see whether the chosen landmark subset supports normalization, oral aperture, lip spread, jaw/lower-face tracking, and symmetry review before moving downstream.

## New backend module

`src/vslp/analysis/kinematics/selection.py`

The module provides:

- `analyze_landmark_selection(indices)`
- `write_selected_landmarks(output_root, selected_landmarks, ...)`
- region coverage summaries
- requirement-level PASS/REVIEW/FAIL diagnostics
- selected landmark JSON, summary CSV, and requirement CSV writing

## New GUI behavior

The Landmark Selection tab now includes a dashboard named:

`1. Selection readiness and scientific coverage`

Dashboard cards show:

- Preset
- Selected landmark count
- Number of represented anatomical regions
- Anchor readiness
- Mouth-aperture readiness
- Jaw/lower-face readiness
- Overall selection status
- Next recommended step

The right-side workstation panel now includes a requirement table with:

- Requirement
- Status
- Missing required landmarks
- Missing recommended landmarks
- Reason

## Output files

Saving selected landmarks now writes:

```text
kinematics/003_selection/tables/selected_landmarks.json
kinematics/003_selection/tables/selected_landmark_summary.csv
kinematics/003_selection/tables/selected_landmark_requirements.csv
kinematics/003_selection/figures/selected_landmark_mesh_preview.png
```

The JSON now includes selection status, region counts, requirement diagnostics, warning flags, app version, mesh source, and preview path.

## Scientific notes

This patch intentionally does not replace analyst review. The requirement checks are conservative guardrails. They ensure the selected set includes the landmarks needed to audit major downstream feature families.

The GUI marks the selection stage as complete when all required and recommended checks pass. If required or recommended landmarks are missing, the selection is saved but marked as completed with warnings so the left sidebar shows a review state instead of silent success.

## Tests

Added:

`tests/unit/test_kinematics_selection_dashboard.py`

Validated in the sandbox with:

```text
python -m pytest tests/unit -k kinematics -q
40 passed

PYTHONPATH=src python -m pytest tests/integration/test_acoustic_preprocess_stage.py -q
1 passed
```
