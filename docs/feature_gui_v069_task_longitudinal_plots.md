# Feature GUI v0.69 - Task and Longitudinal plots

This patch turns Task Review and Longitudinal / Iterations into visual review menus.

## Task Review plots
- Rows by task
- Task x subject coverage
- Task x diagnosis/severity context
- Task-level feature support

## Longitudinal / Iterations plots
- Records per subject
- Subject x session/visit coverage
- Subject x iteration counts
- Visit/recording date timeline

## Full-dataset plotting
The feature availability heatmap now prefers full-dataset display when feasible instead of silently capping rows/features by default. Very large visualizations can still be controlled by optional arguments in the plotting function.

## No modeling changes
No ML training, imputation, scaling, or validation logic is added here.
