# VSLP Feature Analysis GUI v0.42

This update fixes the Feature Analysis public API and improves the Column Mapping workflow.

## Changes

- Restores `AnalysisInputs` and `run_feature_analysis` as public imports from `vslp.analysis.features`.
- Adds a backward-compatible `core.py` API shim.
- Preserves the v0.41 visual design while improving the mapping workflow.
- Adds manual role-edit controls on the Column Mapping page.
- Adds `Accept Mapping and Continue` before analysis.
- Keeps feature-column detection conservative: numeric primary-table columns are features unless there is strong evidence otherwise.

## Column roles

Supported roles include Identifier, Feature, QC feature, Target / label, Covariate, Task, Time / visit, Audit / status, and Ignore.

Use `Ignore / Exclude` for columns that should not be analyzed.
