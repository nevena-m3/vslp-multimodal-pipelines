# VSLP Feature Analysis GUI v0.53

## Scope

This release fixes two interface/runtime issues and adds a professional Reliability / Repeatability review module.

## Fixes

- Adds a compatibility `_make_table()` method used by newer GUI pages so Group / Outcome Screening can initialize correctly.
- Fixes the selected-feature distribution preview recursion that could break the interface when plotting distributions.
- Keeps the Feature Relationships open-current-plot workflow explicit through the existing full-resolution open button.

## Reliability / Repeatability module

The Reliability / Repeatability page evaluates whether features are stable across repeated recordings or sessions. It is descriptive and should not be treated as a formal longitudinal mixed-effects analysis.

New tables:

- `reliability_design_summary.csv`
- `reliability_subject_record_counts.csv`
- `feature_repeatability_summary.csv`
- `reliability_family_summary.csv`

New plots:

- `reliability_status_counts.png`
- `reliability_icc_ranking.png`
- `reliability_variance_landscape.png`
- `reliability_family_summary.png`
- `reliability_subject_counts.png`
- `selected_feature_reliability.png`

## Interpretation

Reliability summaries require a detected subject or participant identifier. ICC-style values are screening proxies based on repeated-record variance structure. High stability may reflect physiology, stable participant traits, or stable device/setup effects; therefore reliability findings should be interpreted with QC Integration and task/session context.
