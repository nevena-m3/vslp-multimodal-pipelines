# Feature GUI v0.67 - QC Integration visual parity

This patch brings QC Integration into the same visual pattern as Overview, Missingness, and Distributions.

## Changes
- Bumps visible version to v0.67.0.
- Uses the same plot-first skeleton: main plot panel left, stacked snapshot cards right, detailed tables below.
- Standardizes the toolbar: plot dropdown, Show, Regenerate, Open current plot on the far right.
- Keeps selected feature and selected QC metric controls inside the plot panel.
- Simplifies QC snapshot card labels.
- Keeps the QC plot list focused on acquisition-sensitivity only.

## No logic changes
This patch does not change QC calculations, metadata joins, missingness, distributions, or exports.
