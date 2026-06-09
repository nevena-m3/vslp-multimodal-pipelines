# Kinematics GUI v0.71 - Setup / Project Dashboard

This patch refines the first kinematics GUI menu into a clearer project dashboard and structural ingest gate.

## Scope

The patch focuses only on the **Setup / Project** menu. It does not alter landmark extraction, landmark selection, normalization, QC, features, aggregation, inspector, or reports behavior.

## Backend additions

`src/vslp/analysis/kinematics/ingest.py` now includes dashboard helpers:

- `summarize_ingest_manifest(df)`
- `build_format_summary(df)`
- `build_warning_summary(df)`

`run_ingest()` now writes additional structural ingest artifacts:

- `kinematics/000_ingest/tables/video_ingest_summary.csv`
- `kinematics/000_ingest/tables/video_ingest_summary.json`
- `kinematics/000_ingest/tables/video_format_summary.csv`
- `kinematics/000_ingest/tables/video_ingest_warnings.csv`
- `kinematics/000_ingest/logs/video_ingest.log`

The original manifest paths are preserved:

- `kinematics/000_ingest/tables/video_ingest_manifest.csv`
- `kinematics/000_ingest/tables/video_ingest_manifest.json`

## Readiness logic

The Setup dashboard reports:

- `PASS`: videos found, structurally readable, no ingest warnings.
- `REVIEW`: at least one readable video exists, but some videos have warnings or probe issues.
- `FAIL`: no supported videos were found, or no videos were structurally readable.

This is intentionally structural readiness only. It does not replace landmark QC, normalization QC, or feature-level QC.

## GUI additions

The Setup tab now has five sections:

1. Project definition
2. Project gate
3. Dataset readiness
4. Video format summary
5. Ingest warnings

The readiness section shows compact metric cards:

- Status
- Videos
- Readable
- Warnings
- Median FPS
- Median duration
- Estimated frames
- FPS range

The GUI also shows next-step guidance, such as whether to continue to landmark extraction or inspect warnings first.

## Project manifest update

Project initialization now writes a richer `kinematics/project_manifest.json` containing:

- schema
- app version
- UTC timestamp
- project name
- task label
- input video folder
- output project folder
- stage order

## Tests

New tests were added in:

- `tests/unit/test_kinematics_setup_dashboard_ingest.py`

Validated locally in the sandbox with:

```powershell
python -m pytest tests/unit/test_kinematics_setup_dashboard_ingest.py tests/unit/test_kinematics_scaffold.py -q
python -m pytest tests/unit/test_kinematics_*.py -q
```

The kinematics unit test subset passed.
