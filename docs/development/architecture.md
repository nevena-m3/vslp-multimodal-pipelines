# VSLP Architecture

## Design Goals

VSLP is a local-first, stage-oriented research platform. Scientific computation lives in backend modules; PySide6 applications coordinate inputs, configurations, execution, review, and output navigation.

Primary design goals are reproducibility, traceability, conservative defaults, file-level fault isolation, and explicit separation between measurement engineering and machine learning.

## Application Layers

```text
Desktop GUIs / CLI
        |
        v
Stage orchestration and typed configuration
        |
        v
Acoustic, kinematic, and feature-analysis computation
        |
        v
Tables, plots, reports, logs, errors, artifacts, manifests
```

## Stage Contract

Conceptually, every executable stage follows:

```text
StageInput + StageConfig -> StageOutput + StageReport + StageManifest
```

Stages should:

- validate required inputs before computation;
- preserve raw source media;
- write deterministic stage directories;
- record configuration and provenance;
- emit explicit file-level errors;
- avoid silently dropping rows;
- expose machine-readable tables independently of HTML reports;
- remain callable without GUI state where practical.

## Data Boundaries

### Acoustic

The acoustic backend produces processed derivatives, segmentation, QC evidence, and feature-specific scalar outputs. Original media are never modified.

### Kinematics

The kinematic backend produces landmark trajectories, selected-landmark policy, normalized trajectories, QC evidence, canonical features, timeseries, and temporal aggregation.

### Feature Analysis

Feature Analysis accepts one modality at a time. It audits dataset support and creates task-specific manifests and handoff tables. It does not train models or learn preprocessing parameters.

### Future ML

The ML layer will consume validated task-specific exports. Subject-grouped splitting, imputation, scaling, transformations, feature selection, model fitting, tuning, and validation must be implemented inside training folds.

## Identifiers and Joins

Preferred record-level key is `record_key`. When unavailable, joins may use a validated composite such as `subject_id + session_id + task` or `subject_id + visit_id + task`.

Many-to-many joins are unsafe by default. A join must preserve row meaning and document unmatched, duplicate, and excluded rows.

## Privacy and Security

Processing is local. Source media, metadata, outputs, and model artifacts remain on the operator's machine unless explicitly moved. The repository must never contain participant data, credentials, or generated study outputs.

## Scientific Safety

- QC is evidence for review, not an automatic exclusion engine.
- Screening is exploratory and not confirmatory inference.
- Reliability is reported only when repeated-measure support is adequate.
- Task-specific results are not generalized across tasks without evidence.
- Feature-readiness recommendations are transparent rules, not performance estimates.
- Clinical interpretation requires independent validation.

## Versioning

Application versions currently evolve independently while sharing one package version. Stage manifests and export configurations record application/schema details required to interpret outputs. Breaking data-contract changes should introduce a new explicit schema identifier.
