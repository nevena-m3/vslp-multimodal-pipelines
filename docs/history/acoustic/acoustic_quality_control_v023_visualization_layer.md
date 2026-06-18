# VSLP v0.23 Quality Control visualization layer

This update expands the segmentation-informed Quality Control stage with a descriptive statistical visualization layer. The purpose is to help researchers understand artifact burden, recording-level review priority, and the internal structure of QC features before acoustic feature extraction.

The QC plots are descriptive and audit-oriented. They do not perform clinical inference and do not claim diagnosis or severity effects. Group-stratified plots are produced only as descriptive views and include sample-size awareness. Small, imbalanced groups should be interpreted cautiously.

## New tables

- `acoustic_quality_distribution_summary.csv`: per-feature distribution, missingness, transformation, winsorization, and robust-scaling metadata.
- `acoustic_quality_processed_features.csv`: transformed, winsorized, and robust-scaled QC features.
- `acoustic_quality_family_scores.csv`: robust family scores computed as the median of available processed features within each artifact family.
- `acoustic_quality_feature_spearman_correlation.csv`: feature-level Spearman correlation matrix.
- `acoustic_quality_family_spearman_correlation.csv`: family-score Spearman correlation matrix.
- `acoustic_quality_recording_review_rank.csv`: recording-level review ranking using warning count and median absolute family-score burden.
- `acoustic_quality_pca_variance.csv`: PCA variance summary when enough recordings/features exist.
- `acoustic_quality_pca_scores.csv`: PCA scores with metadata when available.
- `acoustic_quality_group_counts.csv`: diagnosis/severity/task group counts used for descriptive plotting.

## New plots

- `quality_family_score_distributions.png`: violin + point distributions of robust artifact-family scores.
- `quality_recording_review_rank.png`: top recordings ranked for review.
- `quality_feature_correlation_heatmap.png`: Spearman correlation matrix across usable QC features.
- `quality_family_correlation_heatmap.png`: Spearman correlation matrix across artifact-family scores.
- `quality_missingness_feature_coverage.png`: feature availability across uploaded recordings.
- `quality_pca_scree.png`: PCA variance summary when applicable.
- `quality_pca_embedding.png`: PCA embedding colored/sized by QC warning count when applicable.
- `quality_group_stratified_family_scores.png`: descriptive group-stratified family scores when metadata support it.

## Statistical guardrails

The stage uses robust descriptive procedures only:

- skew-aware monotonic transforms (`log1p`, `asinh`, or none),
- 1st/99th percentile winsorization,
- median/IQR robust scaling,
- family scores as median of available processed features,
- Spearman correlations because QC features may be non-Gaussian and zero-inflated,
- PCA only when enough recordings and variable features exist.

No inferential tests are run in this GUI stage. QC-feature/acoustic-feature relationships belong in the separate Feature Analysis GUI.
