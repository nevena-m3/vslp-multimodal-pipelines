# Feature GUI v0.59 — Column Mapping and Metadata-Aware Review

This patch hardens the Feature Analysis GUI after v0.58.

## User-facing changes

- Visible Feature GUI version is now `v0.59.0`.
- Column Mapping table uses a more compact Role column and a wider Reason / rationale column.
- Role drop-downs no longer change when the user scrolls the mouse wheel over the table.
- If the user modifies any proposed column roles, the GUI asks for explicit confirmation before continuing.
- Accepted mappings are written to `feature_analysis/tables/accepted_column_mapping.csv`.
- Overview, Missingness, and Distribution pages now use a plot-first layout with detailed tables below plots.
- Feature Analysis now builds a metadata-aware analysis context when a metadata table is supplied.

## Metadata context rules

The GUI tries metadata joins in this order:

1. `record_key`
2. `recording_id`
3. `file_name`
4. `source_file_path`
5. `subject_id + session_id + task`
6. `subject_id + visit_id + task`
7. `subject_id + session_id`
8. `subject_id + visit_id`
9. `subject_id + task`
10. `subject_id`

A key join is used only when the metadata key is unique. If no safe key exists but the metadata table has the same row count as the feature table, the GUI uses a row-order metadata context and records that strategy in the run log and outputs. If no safe context exists, metadata remains loaded but unmerged.

The GUI remains descriptive only. No ML training, imputation, scaling, feature selection, or validation is performed here.
