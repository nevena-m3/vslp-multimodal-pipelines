# Feature GUI v0.73 - Distributions scope cleanup

This patch keeps the v0.72 task/clinical scope behavior but removes the redundant and confusing Group overlay control.

## Changes
- Removes the separate Group overlay dropdown from Distributions.
- Clinical context is now the single grouping source for selected-feature plots.
- If no clinical context is available, selected-feature grouping falls back to task when possible.
- Clinical context changes now refresh values and re-preview the active distribution plot.
- Base analysis is unchanged.

## Rationale
The old Group overlay only affected selected-feature plots and overlapped with the newer Clinical context control. This made the interface look like a control should affect every plot when it did not. The new design is simpler: task filters scope; clinical context filters and groups.
