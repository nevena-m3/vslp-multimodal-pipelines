# VSLP Acoustic Pipeline Standard Operating Procedure

## 1. Purpose

This SOP defines the controlled use of the VSLP Acoustic Pipeline GUI to create auditable per-recording acoustic features and QC outputs for research. It covers project setup, non-destructive preprocessing, segmentation, quality control, feature extraction, inspection, and reporting.

The pipeline is not a diagnostic system. QC warnings are review evidence, not automatic exclusion rules. Feature values require study-specific validation before clinical interpretation.

## 2. Inputs and Preconditions

Required:

- source audio or audio-bearing media readable by FFmpeg/FFprobe;
- a writable parent output folder outside the source-media directory;
- a required task name and defined row/recording unit;
- Python 3.11 VSLP environment with GUI dependencies.

Recommended:

- documented filename convention;
- approved study-specific preprocessing, segmentation, QC, and feature policy;
- a representative pilot subset for validation before batch processing.

Do not proceed if source files are still being modified, identifiers are ambiguous, or the output location is not approved for sensitive data.

## 3. Launch

```text
vslp gui acoustic
```

## 4. Controlled Workflow

### Step 1: Setup

1. Select the input media folder.
2. Select a parent output folder outside the source-media directory.
3. Enter a required project name and required task name. The human-readable task name is preserved in provenance, while a filesystem-safe slug names the run folder.
4. Confirm recursive discovery is appropriate for the folder structure.
5. Click Initialize Project. The GUI creates a new `TaskSlug_YYYYMMDD_HHMMSS` folder below the selected parent, then displays that active run folder in Setup. All later acoustic stages use it.
6. Confirm that the new folder contains `project_manifest.json`, `configs/setup_config.json`, `logs/setup.log`, and `acoustic/`. A collision receives `_02`, `_03`, etc. Setup fields lock for this run; open a new GUI window for a different project, task, or input. Review discovered, accepted, duplicate-skipped, and failed counts after Run Ingest.

Acceptance checks:

- all expected files are discovered once;
- unsupported or unreadable files are reported;
- output is not inside the raw-data folder and the task-stamped run folder is new;
- the required task reflects the recordings selected for this run.

### Step 2: Preprocess

Use conservative defaults unless the protocol specifies otherwise:

- remove DC offset;
- create a 16 kHz segmentation WAV;
- preserve source sample rate for feature WAVs unless resampling is scientifically required;
- leave amplitude normalization and filters off unless documented;
- never modify originals.

Review SNR, clipping, DC offset, sample rate, duration, and conversion failures. Filtering or normalization can alter amplitude-, spectral-, and voice-quality features and must be recorded.

### Step 3: Data Segmentation

Run Silero VAD or another implemented method using a locked configuration. Review:

- speech and pause region plausibility;
- total speech/pause durations;
- files with no detected speech;
- boundary plots on representative and difficult recordings;
- task types for which generic speech VAD may be inappropriate.

Do not treat segmentation success as task-compliance validation.

### Step 4: Quality Control

Run selected QC families and inspect recording-level and family-level evidence. Current families include additive interference, gain dynamics, reverberation/echo, channel/device effects, nonlinear distortion, and temporal discontinuity.

Record any study-specific exclusion or sensitivity decision separately. Do not delete rows solely because a QC score is high.

### Step 5: Feature Extraction

1. Select implemented feature families or individual features.
2. Confirm the computation mode and acoustic region policy.
3. Review native measurement scale, support requirements, scalar-reduction policy, and implementation status.
4. Run extraction.
5. Inspect status, validity, expected-range, and reduction-audit tables.

Feature subsystems include timing/respiratory, rhythm/envelope modulation, phonatory, articulatory/formant, resonatory/nasality, and coordination features.

### Step 6: Inspector

Inspect representative tables and plots. Confirm that row counts, filenames, task context, QC linkage, and feature coverage are plausible. Investigate file-level errors before creating a handoff.

### Step 7: Reports and Outputs

Generate or refresh the run summary. Preserve configuration, manifests, errors, tables, plots, and reports with the study record.

## 5. Main and Supplementary Outputs

In **Reports & Outputs**, use **Open Main Feature GUI Handoff** for downstream Feature Analysis. Use **Open Supplementary Outputs** for audit, diagnostics, plots, reports, and troubleshooting.

Main downstream folder:

```text
acoustic/feature_handoff/main/
|-- feature_values.csv
|-- feature_registry.csv
|-- feature_status.csv
|-- feature_export_manifest.json
|-- qc_features.csv              # when available
`-- README.md
```

The four canonical files are required. QC is optional. Clinical metadata, participant identity, diagnosis, outcomes, and visit structure are joined in Feature Analysis. Acoustic handoff carries only SHA-based recording identity, Setup task/run provenance, measured features, registry, status, and optional acoustic QC.

Supplementary index:

```text
acoustic/feature_handoff/supplementary/artifact_catalog.csv
```

The established stage outputs remain in place:

```text
acoustic/000_ingest/
acoustic/001_preprocess/
acoustic/002_segmentation/
acoustic/003_quality_control/
acoustic/004_features/
acoustic/005_run_summary/
```

Legacy and detailed feature-stage files include:

```text
acoustic/004_features/tables/acoustic_features_per_file.csv
acoustic/004_features/tables/selected_acoustic_feature_registry.csv
acoustic/004_features/tables/acoustic_feature_status_long.csv
acoustic/004_features/tables/acoustic_feature_computation_policy.csv
acoustic/004_features/tables/acoustic_feature_scalar_reduction_audit.csv
```

Do not use report HTML, plots, identifiers, file paths, or processing status as predictor inputs.

## 6. Stop Conditions

Stop and resolve the issue when:

- source files cannot be probed or decoded;
- identifiers create ambiguous joins;
- segmentation is systematically inappropriate for the task;
- most files fail QC or feature support;
- upstream outputs are older than rerun source stages;
- feature status indicates widespread computation failure;
- output row counts cannot be reconciled with accepted inputs.

## 7. Completion Checklist

- Source media preserved unchanged.
- Project and task scope documented.
- Required task is present on acoustic feature rows; clinical metadata is joined in Feature Analysis.
- Preprocessing configuration recorded.
- Segmentation spot-checked.
- QC evidence reviewed without automatic exclusion.
- Feature policy and status audited.
- File-level errors resolved or documented.
- Run summary and manifests generated.
- Main handoff opened and its four canonical files verified.
- Optional QC context presence or absence documented; load clinical metadata in Feature Analysis.
- Supplementary artifact catalog retained with the study record.
