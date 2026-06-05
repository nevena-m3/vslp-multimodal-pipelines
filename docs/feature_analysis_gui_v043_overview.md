# VSLP Feature Analysis GUI v0.43 — Overview refinement

This update upgrades the Overview page from a table preview to a true dataset-orientation dashboard.

## Purpose

The Overview page should answer whether the uploaded feature dataset is ready for deeper statistical review. It summarizes table dimensions, mapped column roles, detected design variables, feature-family coverage, and optional QC/metadata availability.

## New outputs

The Feature Analysis GUI now writes additional overview tables:

- `dataset_inventory.csv`
- `feature_role_summary.csv`
- `dataset_design_overview.csv`
- `feature_family_overview.csv`
- `group_counts.csv`

It also writes overview plots:

- `overview_role_counts.png`
- `overview_group_counts.png`
- `overview_feature_family_counts.png`
- `feature_availability_heatmap.png`
- `missingness_top_features.png`

## Interpretation

The Overview page is descriptive only. It does not perform inference, feature selection, or ML training. Its purpose is to confirm that the feature table, optional QC table, optional metadata table, and column mapping are coherent before moving to missingness, distribution, QC-integration, reliability, and export modules.
