# v0.84 Kinematics GUI — Inspector and Reports Completion

This patch completes the remaining two end-of-workflow interface tabs: **Inspector** and **Reports & Outputs**.

## Design goal

The final two tabs should make the pipeline testable and auditable. They do not add new biomarker math. They answer:

1. Which stage artifacts exist?
2. Which required outputs are missing?
3. Which table is the GUI previewing?
4. Can the current run be packaged into a reviewable report?
5. Where are the source CSV/JSON artifacts?

## Inspector

The Inspector tab is now a read-only artifact audit view. It writes:

```text
kinematics/008_inspector/tables/stage_status.csv
kinematics/008_inspector/tables/artifact_inventory.csv
kinematics/008_inspector/tables/inspector_manifest.json
```

The GUI shows:

- stage status cards,
- a stage status table,
- an artifact inventory table,
- a latest table preview,
- inspector notes.

The inspector does not validate clinical or scientific interpretation. It only inventories pipeline outputs.

## Reports & Outputs

The Reports tab now supports two report types:

1. **Workflow Outline Report** — a scaffold/methods-style overview of the intended workflow.
2. **Pipeline Summary Report** — a current-output report based on the latest inspector inventory and available ingest/QC/features/aggregation tables.

Outputs:

```text
kinematics/009_reports/kinematics_gui_workflow_outline_report.html
kinematics/009_reports/kinematics_pipeline_summary_report.html
kinematics/009_reports/report_manifest.json
```

The pipeline summary report includes stage status, artifact inventory, ingest summary, video/landmark QC summary, feature preview, and aggregated output preview when those tables exist.

## Scientific guardrail

Reports are provenance and review artifacts. They are not diagnostic outputs and should not be treated as the primary data source. CSV/JSON outputs remain the source of truth.

## Validation

Validated in sandbox with:

```text
60 kinematics unit tests passed
1 acoustic integration test passed
compile checks passed
```
