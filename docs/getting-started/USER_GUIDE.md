# VSLP Desktop Suite User Guide

## Purpose and Boundaries

VSLP supports reproducible local processing of acoustic recordings and facial-kinematic videos, followed by modality-specific feature review. It is designed for research operations and method development.

The software does not provide clinical diagnoses, automatic subject exclusion, or validated treatment recommendations. Screening, QC, reliability, and readiness outputs require scientific review.

## Project Planning

Before processing data, define:

- the row/recording unit;
- stable `subject_id`, task, session/visit, and source-file identifiers;
- target/outcome and covariate definitions;
- the modality and task being processed;
- the output root and study naming convention;
- approved handling of identifiable or sensitive media;
- the analysis and exclusion protocol.

Use separate output projects for distinct studies or incompatible processing configurations. Never write generated outputs over raw source media.

Use the same study workspace root in all three GUIs. VSLP separates Acoustic, Kinematics, and modality-specific Feature Analysis outputs inside that root. See the [project workspace layout](../reference/project_workspace.md).

## Acoustic Workflow

```text
Setup -> Metadata -> Preprocess -> Segmentation -> Quality Control
      -> Feature Extraction -> Inspector -> Reports & Outputs
```

Use the Acoustic Pipeline GUI for audio. Review each stage before proceeding. The final downstream folder is `acoustic/feature_handoff/main`; detailed stage artifacts remain indexed under `supplementary`.

Detailed procedure: [Acoustic Pipeline SOP](../sops/ACOUSTIC_PIPELINE_SOP.md).

## Kinematics Workflow

```text
Setup -> Metadata -> Face Landmarks -> Landmark Selection -> Normalization
      -> Video/Landmark QC -> Feature Computation -> Temporal Aggregation
      -> Inspector -> Reports & Outputs
```

Use the Kinematics Pipeline GUI for videos. Confirm real-frame landmark overlays and normalization anchors before computing features. The final downstream folder is `kinematics/feature_handoff/main`; timeseries, engineering exports, and audit evidence remain indexed under `supplementary`.

Detailed procedure: [Kinematics Pipeline SOP](../sops/KINEMATICS_PIPELINE_SOP.md).

## Feature Analysis Workflow

Load one modality at a time. Map the feature, metadata, QC, and registry tables, then run the full analysis once.

```text
Load/Map -> Overview -> Missingness -> Distributions -> QC -> Relationships
         -> Task Review -> Longitudinal -> Screening -> Reliability
         -> Recommendations -> Export / Report
```

Recommendations and export decisions are task-specific. Build one Export / Report package per intended task. Use All Tasks only for audit review.

Detailed procedure: [Feature Analysis SOP](../sops/FEATURE_ANALYSIS_GUI_SOP.md).

## Cross-Application Data Handoff

### Acoustic to Feature Analysis

Select `acoustic/feature_handoff/main` with **Load Main Handoff Folder**. Use Direct Table Import only for legacy or independently prepared tables.

### Kinematics to Feature Analysis

Select `kinematics/feature_handoff/main` with **Load Main Handoff Folder**. Include separately curated metadata if optional mapped context was unavailable during extraction. Do not treat every wide engineering column as a predictor.

### Feature Analysis to Future ML

Use the task-specific Export / Report ZIP or directory. Preserve:

- ML-ready feature matrix;
- target, covariate, and QC tables;
- feature decision manifest;
- export configuration;
- README and HTML report.

The future ML workflow must group splits by subject and fit imputation, scaling, transformation, and feature selection inside training folds.

## Run Logs and Errors

Each application exposes status, logs, tables, and reports. A completed button press is not sufficient evidence of a valid stage. Confirm:

- expected inputs were found;
- output row counts are plausible;
- file-level failures are reviewed;
- stage manifests record the configuration;
- downstream outputs are refreshed after upstream reruns;
- no unsupported scientific claims are made from warning or screening tables.

## Data Protection

Keep identifiable media and metadata in approved local storage. Do not commit data, generated project outputs, credentials, participant identifiers, or model files containing sensitive information to Git. See [Security](../../SECURITY.md).
