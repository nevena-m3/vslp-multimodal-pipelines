# Feature GUI v0.66 - Distributions visual parity

This patch brings the Distributions / Outliers menu into the same visual pattern as Overview and Missingness.

## Changes
- Bumps visible version to v0.66.0.
- Uses the same plot-first skeleton: main plot panel on the left, stacked snapshot cards on the right, detailed tables below.
- Standardizes the toolbar: plot dropdown, Show, Regenerate, and Open current plot on the far right.
- Keeps selected-feature and group-overlay controls inside the plot panel.
- Simplifies distribution snapshot card labels.
- Keeps the distribution plot list focused on value-shape and plausibility only.

## No logic changes
This patch does not change distribution calculations, metadata joins, QC logic, missingness logic, or exports.
