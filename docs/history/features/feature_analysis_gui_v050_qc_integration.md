# VSLP Feature Analysis GUI v0.50 — QC Integration

This release replaces the placeholder QC Integration page with a professional artifact-aware review module.

## Purpose

The QC Integration page evaluates whether acoustic feature values, feature missingness, or feature outliers may be associated with recording-quality artifacts. QC is treated as a multidimensional acquisition profile rather than a single global pass/fail score.

The implemented artifact families follow the VSLP multidimensional QC framework:

- Additive interference
- Gain / level dynamics
- Reverberation / echo
- Channel / device / platform
- Nonlinear distortion
- Temporal discontinuities

## New tables

- `qc_integration_summary.csv`
- `qc_metric_catalog.csv`
- `qc_family_burden_summary.csv`
- `qc_row_burden_summary.csv`
- `feature_qc_spearman_correlation.csv`
- `feature_qc_family_association.csv`
- `qc_missingness_associations.csv`
- `qc_outlier_associations.csv`

## New plots

- `qc_artifact_model.png`
- `qc_family_burden.png`
- `qc_metric_distributions.png`
- `qc_feature_association_heatmap.png`
- `qc_top_feature_associations.png`
- `qc_missingness_associations.png`
- `qc_row_burden.png`
- `selected_feature_qc_scatter.png`

## Interpretation policy

QC associations are descriptive screening signals. They do not prove artifact causation and must not be used as automatic exclusion rules. A feature-QC association should prompt contextual review of task, cohort, severity, device/acquisition setup, segmentation, raw audio, and expected feature behavior.

QC integration is intended to support measurement-validity review before ML export. Any final feature exclusion, QC covariate adjustment, sensitivity analysis, imputation, scaling, or feature selection should be handled explicitly downstream in leakage-safe statistical or ML workflows.
