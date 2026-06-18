# VSLP Acoustic Pipeline Standard Operating Procedure

## 1. Purpose

This SOP defines the controlled use of the VSLP Acoustic Pipeline GUI to create auditable per-recording acoustic features and QC outputs for research. It covers project setup, metadata linkage, non-destructive preprocessing, segmentation, quality control, feature extraction, inspection, and reporting.

The pipeline is not a diagnostic system. QC warnings are review evidence, not automatic exclusion rules. Feature values require study-specific validation before clinical interpretation.

## 2. Inputs and Preconditions

Required:

- source audio or audio-bearing media readable by FFmpeg/FFprobe;
- a writable output root outside the source-media directory;
- a defined task and row/recording unit;
- Python 3.11 VSLP environment with GUI dependencies.

Recommended:

- metadata CSV containing stable subject, session/visit, task, and source-file linkage;
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
2. Select a separate output project folder.
3. Enter a stable project name and task fallback.
4. Confirm recursive discovery is appropriate for the folder structure.
5. Initialize the project and review discovered file counts.

Acceptance checks:

- all expected files are discovered once;
- unsupported or unreadable files are reported;
- output is not inside the raw-data folder;
- task fallback is not being used to overwrite valid metadata.

### Step 2: Metadata

1. Load the metadata CSV when available.
2. Review detected subject, task, visit/date, diagnosis, severity, and filename columns.
3. Run metadata indexing/linkage.
4. Inspect unmatched and duplicate records.

Metadata may contain a larger cohort than the selected media subset. Unmatched clinical fields should remain blank rather than being guessed.

### Step 3: Preprocess

Use conservative defaults unless the protocol specifies otherwise:

- remove DC offset;
- create a 16 kHz segmentation WAV;
- preserve source sample rate for feature WAVs unless resampling is scientifically required;
- leave amplitude normalization and filters off unless documented;
- never modify originals.

Review SNR, clipping, DC offset, sample rate, duration, and conversion failures. Filtering or normalization can alter amplitude-, spectral-, and voice-quality features and must be recorded.

### Step 4: Data Segmentation

Run Silero VAD or another implemented method using a locked configuration. Review:

- speech and pause region plausibility;
- total speech/pause durations;
- files with no detected speech;
- boundary plots on representative and difficult recordings;
- task types for which generic speech VAD may be inappropriate.

Do not treat segmentation success as task-compliance validation.

### Step 5: Quality Control

Run selected QC families and inspect recording-level and family-level evidence. Current families include additive interference, gain dynamics, reverberation/echo, channel/device effects, nonlinear distortion, and temporal discontinuity.

Record any study-specific exclusion or sensitivity decision separately. Do not delete rows solely because a QC score is high.

### Step 6: Feature Extraction

1. Select implemented feature families or individual features.
2. Confirm the computation mode and acoustic region policy.
3. Review native measurement scale, support requirements, scalar-reduction policy, and implementation status.
4. Run extraction.
5. Inspect status, validity, expected-range, and reduction-audit tables.

Feature subsystems include timing/respiratory, rhythm/envelope modulation, phonatory, articulatory/formant, resonatory/nasality, and coordination features.

### Step 7: Inspector

Inspect representative tables and plots. Confirm that row counts, filenames, task context, QC linkage, and feature coverage are plausible. Investigate file-level errors before creating a handoff.

### Step 8: Reports and Outputs

Generate or refresh the run summary. Preserve configuration, manifests, errors, tables, plots, and reports with the study record.

## 5. Primary Outputs

```text
acoustic/000_metadata/
acoustic/001_ingest/
acoustic/002_preprocess/
acoustic/003_segmentation/
acoustic/003_quality_control/
acoustic/004_features/
acoustic/007_run_summary/
```

Feature Analysis handoff files normally include:

```text
acoustic/004_features/tables/acoustic_features_per_file.csv
acoustic/004_features/tables/selected_acoustic_feature_registry.csv
acoustic/004_features/tables/acoustic_feature_status_long.csv
acoustic/004_features/tables/acoustic_feature_computation_policy.csv
acoustic/004_features/tables/acoustic_feature_scalar_reduction_audit.csv
```

Include relevant QC and metadata outputs. Do not use report HTML as a data input.

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
- Metadata linkage reviewed.
- Preprocessing configuration recorded.
- Segmentation spot-checked.
- QC evidence reviewed without automatic exclusion.
- Feature policy and status audited.
- File-level errors resolved or documented.
- Run summary and manifests generated.
- Acoustic feature, registry, QC, and metadata handoff verified.
