# VSLP Feature Analysis GUI v0.46 — Distributions / Outliers

This update completes the Distributions / Outliers screen as a descriptive feature-review module.

## Purpose

The module helps users evaluate whether feature values are plausible, stable, and suitable for downstream analysis. It does not automatically exclude features or rows. Flags are review prompts.

## Added outputs

Tables written to `feature_analysis/tables/`:

- `feature_distribution_summary.csv`
- `distribution_review_summary.csv`
- `robust_outlier_flags.csv`
- `feature_expected_range_flags.csv`

Plots written to `feature_analysis/plots/`:

- `distribution_review_status.png`
- `expected_range_flags.png`
- `outlier_counts.png`
- `feature_distribution_grid.png`
- `selected_feature_distribution.png`
- `selected_feature_by_group.png`

## Statistical approach

- Numeric feature columns are summarized using mean, median, SD, IQR, 5th/95th percentiles, min, and max.
- Robust outliers are identified using median/MAD robust z-scores when possible.
- If MAD is zero but IQR is non-zero, an IQR-scaled robust score is used.
- Expected-range flags are only applied when a registry/policy file supplies usable expected low/high bounds.
- Status labels are descriptive: `ok`, `monitor`, or `review`.

## Interpretation

A flagged value may reflect true physiology, task effects, device/recording quality, segmentation error, feature extraction failure, or sample-size effects. Users should inspect flags alongside QC, metadata, task, and implementation-status outputs.
