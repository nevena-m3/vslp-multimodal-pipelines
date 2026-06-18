# Project Workspace Layout

## Rule

Use one VSLP study workspace for a study or processing cohort. Select that same root folder in the Acoustic, Kinematics, and Feature Analysis GUIs.

Do not create unrelated top-level output folders for each GUI. The applications isolate their own artifacts below the shared root.

## Canonical Layout

```text
study_workspace/
|-- project_manifest.json
|-- configs/
|-- acoustic/
|   |-- 000_metadata/
|   |-- 001_ingest/
|   `-- ...
|-- kinematics/
|   |-- project_manifest.json
|   |-- 000_ingest/
|   `-- ...
|-- feature_analysis/
|   |-- acoustic/
|   |   |-- tables/
|   |   |-- plots/
|   |   |-- reports/
|   |   `-- exports/
|   |-- kinematics/
|   `-- generic/
|-- ml/
`-- logs/
```

The root `project_manifest.json` is the workspace identity and component registry. Component initialization is additive: opening an existing workspace from another GUI preserves its original project name, creation timestamp, and registered components.

The Kinematics GUI also writes a component manifest under `kinematics/` because its stage workflow requires kinematics-specific configuration. This supplements the root manifest; it does not define a second project.

## When to Create Another Workspace

Create a separate workspace when any of the following changes:

- study or cohort identity;
- data-governance boundary;
- incompatible acquisition or processing protocol;
- pilot versus locked production processing;
- independent analysis that must not share outputs or provenance.

Different tasks from the same governed study may share a workspace. Their task identity must remain explicit in metadata and exported tables.

## Feature Analysis

Feature Analysis processes one modality at a time. Selecting the modality is required because it determines the isolated output folder:

- Acoustic -> `feature_analysis/acoustic/`
- Kinematic -> `feature_analysis/kinematics/`
- Generic -> `feature_analysis/generic/`

This prevents an acoustic analysis from replacing kinematic tables, plots, reports, or export packages. The Advanced ML Export Builder may create a separate multimodal package under `feature_analysis/ml_export_builder/`; it does not change the single-modality analysis folders.

## Raw Data

Raw audio, raw video, and identifiable metadata may remain outside the workspace in approved source storage. The workspace manifest records the selected source location, while generated outputs remain under the shared root. Never copy sensitive source data into Git.
