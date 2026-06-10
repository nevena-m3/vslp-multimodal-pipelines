# Feature GUI v0.60 — Overview Redesign

This patch redesigns the Feature Analysis GUI Overview page as a compact orientation screen rather than a redundant plot gallery.

## Changes

- Bumps visible Feature GUI version to `v0.60.0`.
- Moves summary tiles into a right-side `Dataset snapshot` panel so they no longer consume the top of the page.
- Replaces the previous left plot-button stack with a single top toolbar: plot dropdown, `Show`, `Regenerate`, and `Open current plot` on the far right.
- Removes redundant overview plot choices that belong in deeper menus, especially top missing features and feature availability heatmap.
- Keeps the Overview plot set limited to six orientation plots:
  - Readiness scorecard
  - Design context
  - Role mapping summary
  - Feature-family coverage
  - Feature-quality landscape
  - Subject × task coverage
- Keeps detailed overview tables below the plot area.
- Makes overview tables horizontally scrollable/interactively resizable rather than forced-stretch.

## Boundary

This patch changes Overview layout and plot selection only. It does not change feature calculations, QC logic, ML export behavior, or metadata parsing.
