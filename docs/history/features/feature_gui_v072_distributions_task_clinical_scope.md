# Feature GUI v0.72 - Distributions local task and clinical scope

Built from the stable v0.71 Missingness task-scope line.

## Scope
This patch only adds local task/clinical controls to Distributions and reuses the stable plot-preview helper.

## Adds
- Task focus dropdown for Distributions.
- Clinical context dropdown for Distributions.
- Clinical value dropdown for Distributions.
- Local detection for diagnosis, ALSFRS bulbar severity, ALSFRS total severity, ALSBDI placeholder bins, sex/gender, session/visit, and iteration.
- Task/clinical-scoped distribution plots without modifying the base analysis table.

## Safety
- No global filter.
- No rerun of the whole analysis for plot scope changes.
- No mutation of `analysis_df`.
- No warning-handling changes.
- If metadata is absent or unfamiliar, controls show unavailable and base plots still work.
