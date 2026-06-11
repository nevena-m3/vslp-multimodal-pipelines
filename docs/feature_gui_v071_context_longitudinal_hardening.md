# Feature GUI v0.71 - Context, longitudinal, and plot-preview hardening

This patch addresses several usability/scientific issues.

## Fixes
- Suppresses/avoids constant-input Spearman correlation warnings by skipping constant vectors before correlation.
- Uses a stable plot-display helper so repeated Show/Regenerate clicks do not make previews grow or zoom.
- Adds a selected-feature trajectory view to Longitudinal / Iterations.
- Adds a feature selector to Longitudinal / Iterations so a user can inspect how one feature changes across visits/sessions/iterations for one subject.
- Keeps task/group context filtering available through the global Analysis context bar.

## Longitudinal interpretation
The longitudinal menu is now structured around:
- repeated records per subject,
- subject x session/visit coverage,
- subject x iteration coverage,
- visit/date timeline,
- selected feature trajectory.

The selected feature trajectory is the main view for asking: what changed in this subject from visit to visit?

## Boundary
This is still descriptive review. Formal longitudinal modeling remains a later ML/statistical modeling step.
