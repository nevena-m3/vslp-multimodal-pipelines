# Project Workspace Layout

## Rule

Acoustic Setup accepts a parent output folder and creates a separate `TaskSlug_YYYYMMDD_HHMMSS` run folder beneath it. Select that run folder when loading its acoustic handoff into Feature Analysis. Kinematics retains its own workspace setup.

Acoustic Setup requires project name, task name, input folder, and output parent. It creates the manifest, setup config, log, and `acoustic/` root. Setup then locks. Acoustic stage folders appear as stages run, and empty stage directories are removed afterward. Downstream Feature Analysis may later add `feature_analysis/` when the run folder is selected as its workspace. Kinematics and ML are not created by acoustic Setup.

## Acoustic Run Layout

```text
TaskSlug_YYYYMMDD_HHMMSS/
|-- project_manifest.json
|-- configs/
|   `-- setup_config.json
|-- logs/
|   `-- setup.log
`-- acoustic/
    |-- 000_ingest/
    |-- 001_preprocess/
    |-- 002_segmentation/
    |-- 003_quality_control/
    |-- 004_features/
    |-- 005_run_summary/
    `-- feature_handoff/
        |-- main/
        `-- supplementary/
```

The root manifest stores the project name, human-readable task, safe task slug, run ID, local and exact UTC creation times, resolved paths, versions, and acoustic modality. Existing run names receive `_02`, `_03`, etc., and are never reused. The setup config and log record the project, task, source folder, and run folder. Stage folders appear as stages run. The acoustic GUI does not create `kinematics/`, `ml/`, or `feature_analysis/` during Setup. Clinical metadata is joined in Feature Analysis. Recording IDs are stable original-source SHA-256 values; source filename remains a separate human-readable field. The Acoustic GUI does not read demographics or clinical labels.

Kinematics uses its own component workflow and may write a component manifest under `kinematics/` in a workspace chosen for that GUI.

## When to Create Another Workspace

Create a separate workspace when any of the following changes:

- study or cohort identity;
- data-governance boundary;
- incompatible acquisition or processing protocol;
- pilot versus locked production processing;
- independent analysis that must not share outputs or provenance.

Different acoustic tasks from the same governed study may share a parent output folder but receive separate task-stamped run folders. Their task identity remains explicit in metadata and exported tables.

## Feature Analysis

Feature Analysis processes one modality at a time. Selecting the modality is required because it determines the isolated output folder:

- Acoustic -> `feature_analysis/acoustic/`
- Kinematic -> `feature_analysis/kinematics/`
- Generic -> `feature_analysis/generic/`

This prevents an acoustic analysis from replacing kinematic tables, plots, reports, or export packages. The Advanced ML Export Builder may create a separate multimodal package under `feature_analysis/ml_export_builder/`; it does not change the single-modality analysis folders.

## Raw Data

Raw audio, raw video, and identifiable metadata may remain outside the workspace in approved source storage. The workspace manifest records the selected source location, while generated outputs remain under the shared root. Never copy sensitive source data into Git.
