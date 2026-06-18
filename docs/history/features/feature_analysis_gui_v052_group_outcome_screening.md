# VSLP Feature Analysis GUI v0.52 — Group / Outcome Screening

Adds a descriptive, univariate Group / Outcome Screening module.

This module is intentionally not a modelling screen. It does not train classifiers, fit final clinical models, perform leakage-prone feature selection, or make biomarker claims. It provides a structured first-pass review of whether features show descriptive relationships with mapped targets, outcomes, tasks, diagnosis/group labels, severity variables, visit/session variables, device variables, and covariates.

Outputs:

- `screening_summary.csv`
- `screening_variable_catalog.csv`
- `screening_continuous_outcome_associations.csv`
- `screening_categorical_group_associations.csv`
- `screening_group_balance.csv`

Plots:

- `screening_group_balance.png`
- `screening_effect_ranking.png`
- `screening_continuous_heatmap.png`
- `screening_group_heatmap.png`
- `screening_effect_landscape.png`
- `selected_feature_outcome.png`

Continuous outcomes are screened using Spearman feature-outcome association. Categorical groups are screened using robust descriptive contrasts based on median differences scaled by pooled IQR/SD, with Cliff's delta reported for binary contrasts.

Interpretation: all results are descriptive screening signals. Candidate effects must be checked against missingness, distributional plausibility, QC sensitivity, redundancy, group balance, task structure, covariates, and repeated-measures design before clinical interpretation or ML export.
