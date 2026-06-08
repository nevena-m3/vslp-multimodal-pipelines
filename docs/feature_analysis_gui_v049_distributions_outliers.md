# VSLP Feature Analysis GUI v0.49 — Distributions / Outliers Professional Review

This version upgrades the Distributions / Outliers menu into a feature-plausibility review screen. It is intended to help the analyst determine whether feature values are numerically stable, scientifically plausible, and interpretable before downstream ML.

The menu now emphasizes:

- feature-level distribution summary statistics;
- robust outlier burden;
- expected-range violations when a registry/policy table supplies bounds;
- distribution-shape audit: skew, tail heaviness, floor/ceiling effects, zero/near-zero variance, small valid n;
- row-level outlier burden to detect recordings that accumulate many feature flags;
- selected-feature diagnostic plots with histogram, box/strip view, empirical CDF, and QQ-style normality check;
- group-overlay plots for task, diagnosis, severity, modality, sex/gender, session, or subject when available;
- interpretation guidance beside every distribution plot.

These outputs are descriptive review tools only. They do not exclude, transform, impute, scale, or select features. Any transformation or model-based selection must occur inside the downstream ML pipeline to avoid leakage.
