# VSLP Feature Analysis GUI v0.39

This update introduces the first substantial Feature Analysis GUI. It is separate from the Acoustic GUI and is intended to inspect feature tables produced by acoustic, kinematic, mixed-modality, or generic pipelines.

## Purpose

The Feature Analysis GUI answers whether the feature table is statistically usable before downstream ML. It audits missingness, distributions, outliers, feature redundancy, QC associations, and preliminary feature reliability.

It does not train models. The output should feed later ML workflows.

## Launch

```bash
python -m vslp.gui.features.app
```

or:

```bash
python tools/run_feature_analysis_gui.py
```

## Inputs

Primary feature table is required. Optional inputs include QC table, metadata table, and feature registry or computation policy table.

Supported table formats in this first pass: CSV, TSV/TXT, and Parquet.

## Included modules in v0.39

1. Upload tables
2. Auto column-role mapping
3. Dataset overview
4. Missingness analysis
5. Distribution summaries
6. Robust outlier detection
7. Spearman feature correlation
8. Feature-QC correlation if QC table is supplied
9. Initial feature reliability screen
10. HTML report and output export

## Output folder

Outputs are written under:

```text
feature_analysis/
  tables/
  plots/
  reports/
  exports/
```

Key tables:

- `dataset_inventory.csv`
- `feature_column_mapping.csv`
- `feature_distribution_summary.csv`
- `missingness_by_row.csv`
- `robust_outlier_flags.csv`
- `feature_spearman_correlation.csv`
- `feature_qc_spearman_correlation.csv`
- `feature_reliability_screen.csv`

Key plots:

- `missingness_top_features.png`
- `feature_availability_heatmap.png`
- `feature_distribution_grid.png`
- `feature_correlation_heatmap.png`
- `outlier_counts.png`

## Interpretation constraints

The reliability screen is descriptive. It is not an automatic feature-exclusion rule. It flags high missingness, zero variance, robust outliers, and strong QC associations so the researcher can review the feature before ML.
