# VSLP Feature Analysis GUI v0.48 — Missingness visuals and Windows readability refinement

This patch refines the Missingness menu and hardens the Qt stylesheet so ordinary controls remain light and readable on Windows.

## Styling fixes

- Forces buttons, combo boxes, tab bars, list views, menus, tooltips, radio buttons, checkboxes, and table selections to use light backgrounds and dark readable text.
- Adds explicit hover, focus, pressed, selected, checked, and disabled states.
- Keeps the navy left navigation intentional while preventing working controls from appearing black before activation.

## Missingness menu refinements

- Adds a compact interpretation guide to the plot-control panel.
- Adds a plot-specific interpretation caption next to the preview.
- Clarifies that missingness review is about missing-data mechanism and feature support, not automatic exclusion.
- Keeps the existing tables and plot outputs unchanged so downstream compatibility is preserved.

## Interpretation principle

The Missingness screen should answer whether missingness is isolated, row-wide, task/group-associated, family/subsystem-associated, or clustered among related features. This is the required review step before complete-case analysis, imputation, feature exclusion, or ML export.
