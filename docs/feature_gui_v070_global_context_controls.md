# Feature GUI v0.70 - Global task/group analysis context

This patch adds a persistent Analysis Context bar across the Feature GUI.

## Purpose
All review plots and tables can be affected by task composition and group composition. The global context controls let the user regenerate the analysis for:

- all tasks or one selected task
- all groups or one selected group value

## Controls
- Task
- Group variable
- Group value
- Apply context + regenerate
- Clear context

## Behavior
After applying a context, Feature GUI rebuilds outputs and regenerates plots/tables using the filtered table. The run log records the active context. If the selected context has zero rows, the GUI warns instead of generating misleading empty outputs.

## Boundary
This is an analysis/review filter, not ML validation. ML splits and leakage-safe group handling must still happen later inside the ML GUI.
