# VSLP Feature Analysis GUI v0.54 — Feature Recommendation

This release adds the Feature Recommendation stage. The stage integrates prior diagnostics into transparent feature-readiness labels for downstream export planning.

The stage does not perform final feature selection, machine learning, imputation, scaling, or transformation. It combines evidence from missingness, distribution/outlier review, QC sensitivity, redundancy, group/outcome screening, and reliability/repeatability into a conservative recommendation table.

Primary outputs:

- `feature_analysis/tables/feature_recommendations.csv`
- `feature_analysis/tables/feature_recommendation_summary.csv`
- `feature_analysis/tables/feature_recommendation_reason_counts.csv`
- `feature_analysis/tables/feature_recommendation_family_summary.csv`
- `feature_analysis/tables/ml_export_manifest.csv`

Primary plots:

- `feature_analysis/plots/recommendation_counts.png`
- `feature_analysis/plots/recommendation_score_landscape.png`
- `feature_analysis/plots/recommendation_reason_counts.png`
- `feature_analysis/plots/recommendation_family_summary.png`
- `feature_analysis/plots/ml_export_manifest_summary.png`

Readiness labels:

- `recommended`: clean default candidate for downstream export.
- `recommended_with_caution`: usable, but requires documented caution.
- `review_before_use`: hold for manual review/sensitivity analysis before downstream use.
- `exclude_or_recompute`: not recommended by default; review or recompute.
- `exclude_by_default`: generally hold out except for audit or explicit scientific justification.

The ML export manifest is a transparent review default, not final ML feature selection. Final preprocessing and feature selection must occur in the future ML GUI inside leakage-safe training/validation folds.
