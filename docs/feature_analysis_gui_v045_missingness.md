# VSLP Feature Analysis GUI v0.45 — Missingness module

This update completes the first dedicated Missingness screen for the Feature Analysis GUI.

## Purpose

Clinical speech, kinematic, and multimodal datasets rarely have random missingness. A missing feature can reflect task incompatibility, failed segmentation, low signal quality, disease severity, subject fatigue, device differences, or implementation limits. The missingness module is designed to audit those patterns before downstream feature selection or ML.

## New missingness outputs

Tables are written to `feature_analysis/tables/`:

- `missingness_by_feature.csv`
- `missingness_by_row.csv`
- `missingness_by_group.csv`
- `missingness_by_family.csv`
- `missingness_comissing_pairs.csv`

Plots are written to `feature_analysis/plots/`:

- `missingness_top_features.png`
- `missingness_row_distribution.png`
- `missingness_by_group.png`
- `missingness_by_family.png`
- `feature_availability_heatmap.png`
- `missingness_comissing_heatmap.png`

## Interpretation policy

The GUI uses conservative descriptive categories:

- `<20% missing`: low
- `20–50% missing`: monitor
- `50–80% missing`: review
- `>=80% missing`: high review

These are not automatic exclusion rules. They are prompts for scientific review.

## Recommended workflow

1. Load feature, QC, metadata, and optional registry/policy tables.
2. Review and accept Column Mapping.
3. Run Feature Analysis.
4. Open Missingness.
5. Inspect feature-level missingness first.
6. Inspect row-level missingness to identify poor recordings/sessions.
7. Inspect group-level missingness to identify potential bias by task, diagnosis, severity, subject, session, device, or modality.
8. Inspect co-missingness to find groups of features that fail together.

## Biostatistical caution

The module intentionally does not run p-value-driven missingness tests. With small, imbalanced, repeated-measure biomedical datasets, descriptive patterns, sample sizes, and mechanism review are more useful at this stage.
