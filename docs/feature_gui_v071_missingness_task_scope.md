# Feature GUI v0.71 - Missingness local task scope

Built from stable v0.70 safe local metadata context.

## Scope
This patch only touches Missingness and shared plot-preview rendering.

## Changes
- Adds a local Task focus dropdown to Missingness.
- Missingness plots can be regenerated for All tasks or one detected task.
- The base analysis is not rerun and `analysis_df` is not modified.
- Missingness heatmaps request all selected feature columns when feasible.
- Co-missingness heatmap defaults to all selected feature columns instead of only the most-missing subset.
- Adds stable plot-preview rendering to stop repeated Show clicks from visually zooming the preview.

## Not changed
- Project, Column Mapping, Overview, Distributions, QC, Relationships, Screening, Reliability, Recommendations, ML Export.
- No global task/group filter.
- No clinical-context derivation in the run-analysis path.
