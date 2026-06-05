# VSLP v0.24 — Quality Control workstation layout

This update refines the Quality Control GUI tab without changing the underlying QC feature formulas.

## Intent

The Quality Control tab is now a compact workstation for:

- selecting artifact-family QC features;
- inspecting feature meaning and parameter dependencies;
- tuning transparent computation parameters;
- running segmentation-informed QC;
- previewing QC tables and plots from one place;
- reading a descriptive dataset-level QC snapshot.

## Scientific guardrails

QC warnings are screening outputs. They are not automatic exclusion rules and should not be interpreted as clinical decisions. Downstream reliability analysis between QC metrics and acoustic/kinematic biomarkers belongs in the separate Feature Analysis GUI.

## Layout

The tab now contains two subtabs:

1. Configure
2. Outputs

The Configure tab contains feature selection, feature details, parameter relevance, computation parameters, and the run button.

The Outputs tab contains preview buttons for core QC tables and plots, plus a compact interpretation summary.
