# VSLP Feature Analysis GUI v0.44 — Overview Plot Completion

This patch completes the Overview plotting layer for the Feature Analysis GUI.

## Key changes

- Overview plot buttons now generate missing plot files automatically when feature tables and output folder are available.
- The Overview page includes a two-panel plot gallery: plot controls on the left and an embedded preview canvas on the right.
- Users can regenerate all Overview plots from the GUI.
- Users can open the currently previewed plot in the native operating-system image viewer.
- Overview plots are generated under `feature_analysis/plots/` and tables under `feature_analysis/tables/`.
- Plot generation is robust to empty data, missing registry files, missing grouping variables, and small datasets.

## Overview plots

- `overview_role_counts.png`
- `overview_group_counts.png`
- `overview_feature_family_counts.png`
- `feature_availability_heatmap.png`
- `missingness_top_features.png`

## Intended workflow

1. Load the primary feature table.
2. Optionally load QC, metadata, and feature registry/policy tables.
3. Review and accept column mapping.
4. Select an output folder.
5. Run Feature Analysis or use the Overview plot buttons to generate/preview plots.

The Overview stage is an orientation layer. It helps the user confirm that the dataset was loaded correctly before deeper missingness, distribution, QC, reliability, or ML analysis.
