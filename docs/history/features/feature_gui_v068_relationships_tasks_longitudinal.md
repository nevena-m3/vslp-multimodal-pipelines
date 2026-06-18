# Feature GUI v0.68 - Relationships visual parity, Task Review, and Longitudinal / Iterations

This patch does three UI-level things.

## Feature Relationships
- Bumps visible version to v0.68.0.
- Brings Feature Relationships into the same plot-first visual pattern as Overview, Missingness, Distributions, and QC Integration.
- Adds a stacked Relationship snapshot panel.
- Keeps the plot list restricted to feature-feature structure, redundancy, family/block structure, and PCA.

## Task Review
- Adds a new sidebar menu dedicated to task-aware dataset review.
- Reports when task metadata is unavailable.
- Summarizes task counts, task x subject coverage, task x diagnosis/severity context, and task-level feature support.
- Adds task focus controls for task-centered review.

## Longitudinal / Iterations
- Adds a new sidebar menu dedicated to repeated-subject, session/visit, iteration, and date review.
- Reports when subject/session/iteration/date fields are unavailable.
- Summarizes repeated subjects and available longitudinal design fields.

## No modeling changes
No ML training, feature selection, imputation, scaling, or model evaluation is added here.
