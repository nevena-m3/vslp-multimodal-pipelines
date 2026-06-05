# VSLP Multimodal Pipelines

**VSLP Multimodal Pipelines** is a local research software platform for reproducible acoustic and facial-kinematic analysis in clinical speech and motor-assessment studies. The current completed application is the **VSLP Acoustic Pipeline GUI**, designed for local processing of speech recordings from remote or lab-based clinical research workflows.

This repository is currently for **research use only**. It is not a validated medical device and must not be used for clinical decision-making without formal validation, documentation lock, risk management, and the appropriate regulatory pathway.

## Current acoustic GUI workflow

The Acoustic Pipeline GUI currently supports this workflow:

```text
Setup
→ Metadata
→ Preprocess
→ Data Segmentation
→ Quality Control
→ Feature Extraction
→ Inspector
→ Reports & Outputs
```

The removed Aggregation tab is intentional. Feature Extraction now performs feature-specific scalar reduction from each feature's native measurement scale into one file-level value. Cross-file, cross-session, longitudinal, or ML-oriented aggregation should be handled later in the Feature Analysis GUI or ML GUI.

## What each stage does

### Setup

Selects the input audio folder, output project folder, project name, and task being analyzed. Subfolders are searched recursively. One task per input folder is recommended when possible. The task field is used only as a fallback when task information is not available from metadata or filename parsing.

### Metadata

Optionally links recordings to a metadata CSV. The metadata table may contain many more rows than the uploaded audio subset. VSLP detects flexible column names such as `SubjectID`, `ID`, `Clinical Visit ID`, `Protocol ID`, `Task Name`, `Recording date`, `ALSFRS total score`, and related variants. If no metadata CSV is supplied, VSLP continues with filename parsing and leaves clinical labels blank when they cannot be inferred.

### Preprocess

Creates non-destructive processed audio derivatives. Originals are never modified. The default behavior removes DC offset, creates a 16 kHz segmentation WAV for Silero, and creates a feature WAV while preserving the source sample rate unless the user explicitly enables resampling. Optional peak normalization, Butterworth high-pass/low-pass/band-pass filtering, and 50/60 Hz notch filtering are available but off by default.

Preprocess outputs include estimated SNR, clipping, DC offset before/after correction, powerline flags, generated WAV paths, plots, and reports.

### Data Segmentation

Runs speech/pause segmentation. Silero VAD is currently implemented. SPA, Energy/RMS, and custom segmentation are reserved plugin slots for future methods. Segmentation outputs speech regions, pause regions, segment summaries, plots, and reports.

### Quality Control

Computes segmentation-informed audio-quality features after segmentation and before feature extraction. It includes six artifact families: additive interference, gain dynamics, reverberation/echo, channel/device, nonlinear distortion, and temporal discontinuity. Outputs include QC feature tables, warnings, recommendations, family scores, review rankings, descriptive plots, and reports. These are screening aids, not automatic exclusion rules.

### Feature Extraction

Computes acoustic features by subsystem:

```text
respiratory/timing
rhythm / envelope modulation
phonatory
articulatory / formant
resonatory / nasality
coordination
```

The Feature Extraction stage outputs one row per file, but each scalar is produced by a documented feature-specific computation policy. The GUI and audit tables record native scale, analysis region, scalar reduction, computation mode, validity warnings, and implementation status.

Examples:

```text
Timing/respiratory: speech/pause event summaries
Rhythm/EMS: effective task region, preserving internal pauses
Phonatory: voiced-frame support
Articulatory/formant: valid LPC formant trajectories
Resonatory/nasality: valid spectral frames
Coordination: aligned CPP/F1/F2 trajectories and lagged eigenspectrum summaries
```

### Inspector

Provides in-GUI preview of key tables and plots.

### Reports & Outputs

Opens generated reports, primary tables, stage folders, and creates the final run-summary manifest.

## Installation

Recommended Python version: **3.11**.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,gui,silero]'
```

Install FFmpeg/FFprobe before running the pipeline.

macOS:

```bash
brew install ffmpeg
```

Windows users should install FFmpeg/FFprobe using a trusted package manager or a vetted binary distribution.

## Launching the Acoustic GUI

From the repository root:

```bash
source .venv/bin/activate
vslp gui acoustic
```

## Full-pipeline behavior

The **Run Full Acoustic Workflow** button runs:

```text
Metadata
→ Ingest
→ Preprocess
→ Data Segmentation
→ Quality Control
→ Feature Extraction
→ Run Summary
```

If no metadata CSV is selected, the GUI asks whether the user wants to select metadata or continue without it. The pipeline can run without metadata, but diagnosis, severity, and other clinical labels remain blank unless they can be parsed or are provided later.

## Primary output folders

Each project output folder contains stage-organized outputs under:

```text
acoustic/000_metadata/
acoustic/001_ingest/
acoustic/002_preprocess/
acoustic/003_segmentation/
acoustic/003_quality_control/
acoustic/004_features/
acoustic/007_run_summary/
```

Each stage writes some combination of:

```text
tables/
plots/
reports/
logs/
errors/
manifests/
artifacts/
```

## Main acoustic feature outputs

The most important feature outputs are:

```text
acoustic/004_features/tables/acoustic_features_per_file.csv
acoustic/004_features/tables/acoustic_feature_status_long.csv
acoustic/004_features/tables/acoustic_feature_scalar_reduction_audit.csv
acoustic/004_features/tables/acoustic_feature_computation_policy.csv
acoustic/004_features/reports/acoustic_feature_report.html
```

## Project principles

1. Local processing of sensitive data.
2. Non-destructive audio handling.
3. Explicit stage inputs and outputs.
4. Reproducibility through manifests and audit tables.
5. Conservative preprocessing defaults.
6. Segmentation-informed QC.
7. Feature-specific computation policies.
8. No automatic clinical interpretation or exclusion.
9. GUI usability with CLI reproducibility.

## Repository layout

```text
src/vslp/core              Shared schemas, manifests, project utilities
src/vslp/acoustic          Acoustic metadata, ingest, preprocess, segmentation, QC, features
src/vslp/gui               PySide6 GUI applications
src/vslp/cli               Command-line interface
configs                    Task and feature configuration files
docs                       User and developer documentation
tests                      Unit and integration tests
```

## Citation / attribution

Current GUI attribution:

```text
© 2026 Nevena Musikic & Yana Yunusova
Speech Production Lab, University of Toronto
```

Final ownership, licensing, distribution, and release wording should be confirmed with the Speech Production Lab and University of Toronto policies before public release.
