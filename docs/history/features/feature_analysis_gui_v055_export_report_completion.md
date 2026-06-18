# VSLP Feature Analysis GUI v0.55 — Export / Report Completion

This patch completes the Feature Analysis GUI export/report stage.

## Purpose

The Export / Report stage is the final packaging layer after Project, Column Mapping, Overview, Missingness, Distributions / Outliers, QC Integration, Feature Relationships, Group / Outcome Screening, Reliability / Repeatability, and Feature Recommendation.

It does not train a model. It creates transparent downstream-analysis files and documents why each feature is included, held for review, or excluded by default.

## New GUI behavior

The Export / Report page now provides:

- Export profile selector.
- Color-coded feature manifest.
- Feature inclusion / hold decision column.
- Readiness summary table.
- Exported-table profile summary.
- Decision legend.
- Plot preview and full-resolution plot opening.
- HTML report opening.
- Output-folder opening.
- One-click export package creation.

## Export profiles

### Recommended only (strict)

Includes only features labeled `recommended`.

### Recommended + caution (default ML starting set)

Includes features labeled `recommended` and `recommended_with_caution`. This is the default suggested starting set for future ML, because caution features are retained but their risks are documented.

### Review set

Includes `recommended`, `recommended_with_caution`, and `review_before_use` features. This is useful for sensitivity analyses or expert review, but it is not the conservative ML default.

### Full audit set

Includes all features. This is useful for documentation and manual exploration only. It should not be treated as the default ML feature matrix.

## Color coding

- Green: recommended; include by default.
- Light green: recommended with caution; include but document risk.
- Gold: review before use; hold from default ML unless justified.
- Orange: exclude or recompute; technical/scientific concern.
- Red: exclude by default; major support problem.

## Exported files

Each created export package includes:

- `ml_ready_feature_matrix.csv`
- `ml_target_table.csv`
- `ml_covariate_table.csv`
- `ml_qc_covariate_table_from_feature_table.csv`
- `linked_qc_table.csv` when a QC table was loaded
- `linked_metadata_table.csv` when metadata was loaded
- `feature_export_manifest.csv`
- `feature_recommendation_summary.csv`
- `feature_recommendation_legend.csv`
- `export_profile_summary.csv`
- `export_config.json`
- `README.md`
- `vslp_feature_analysis_export_report.html`
- zipped copy of the export package

## Scientific boundary

The export package is a feature-analysis product, not a machine-learning product. Imputation, scaling, transformations, feature selection, and model training must occur inside the downstream ML pipeline to avoid data leakage.
