# VSLP Feature Analysis GUI v0.47 — Overview visual dashboard refinement

This patch upgrades the Overview page from a basic orientation page into a visual dataset-readiness dashboard.

## Purpose

The Overview page should answer: “Do I understand the structure, coverage, and first-pass readiness of this feature dataset before I inspect missingness, distributions, QC, or ML export?”

## New outputs

Tables:

- `overview_readiness_summary.csv`
- `overview_feature_quality_landscape.csv`

Plots:

- `overview_readiness_scorecard.png`
- `overview_design_tiles.png`
- `overview_role_counts.png`
- `overview_group_counts.png`
- `overview_subject_task_matrix.png`
- `overview_feature_family_counts.png`
- `overview_feature_family_quality.png`
- `overview_feature_quality_landscape.png`
- `feature_availability_heatmap.png`
- `missingness_top_features.png`

## Design notes

- The readiness scorecard is descriptive, not predictive.
- The feature quality landscape combines missingness and robust outlier burden to show which features need review.
- The subject × task matrix is shown only when both subject and task columns are available.
- Family-level plots become more informative when a registry / computation-policy table supplies subsystem labels.
- This patch does not add ML modeling, imputation, scaling, or feature selection.
