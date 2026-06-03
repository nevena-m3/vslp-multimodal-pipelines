# VSLP Multimodal Pipelines

**VSLP** is a local-first research software platform for reproducible acoustic and facial-kinematic analysis in clinical speech and motor assessment studies.

The current design target is remote longitudinal monitoring of ALS and related neuromotor populations using speech/audio recordings and camera-derived facial kinematics.

> Status: early scaffold. This repository is not a validated medical device. Research use only.

## Core design principles

1. Local-only processing of sensitive patient data.
2. Non-destructive ingest and preprocessing.
3. Stage-level modularity: every stage has explicit input, config, output, logs, reports, and manifests.
4. Audit-grade provenance: hashes, configuration, software versions, timestamps, warnings, and errors.
5. Subject-level split enforcement for ML to prevent leakage across longitudinal sessions.
6. GUI-first usability, CLI-first reproducibility.

## Four applications

1. **Acoustic Pipeline GUI**: ingest, preprocessing, segmentation, QC, feature extraction, aggregation.
2. **Kinematic Pipeline GUI**: video ingest, MediaPipe landmarks, normalization, QC, kinematic features, aggregation.
3. **Feature Analysis GUI**: missingness, distributions, correlations, outliers, reliability, task/diagnosis stratification.
4. **ML GUI**: classical ML training, pretrained local inference, evaluation, explainability, model registry.

## Installation

Recommended Python: **3.11**.

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
pip install -e ".[dev,gui,silero,kinematic]"
```

Install FFmpeg/FFprobe before running ingest. The code checks for them and records the detected versions in manifests.

macOS:

```bash
brew install ffmpeg
```

Windows recommendation: install FFmpeg via winget/chocolatey or bundle a vetted binary in a future packaged release.

## First local development run

Detailed instructions are in `docs/development_first_run.md`. The shortest path is:

```bash
# 1) Create optional synthetic local test files.
python scripts/create_test_audio.py

# 2) Create a project/output folder.
vslp project init examples/test_runs/run_001 --project-name "VSLP smoke test"

# 3) Digest the media files with ffprobe.
vslp acoustic ingest examples/test_data/audio_inputs examples/test_runs/run_001

# 4) Decode/canonicalize/preprocess with ffmpeg and write QC outputs.
vslp acoustic preprocess examples/test_data/audio_inputs examples/test_runs/run_001

# 5) Run Silero segmentation after installing the optional Silero dependency.
vslp acoustic segment-silero \
  examples/test_runs/run_001/acoustic/002_preprocess/tables/acoustic_preprocess_summary.csv \
  examples/test_runs/run_001
```

For initial development without Silero, run:

```bash
vslp acoustic run-v1 examples/test_data/audio_inputs examples/test_runs/run_002 --skip-segmentation
```

## Repository layout

```text
src/vslp/core              Shared schemas, manifests, provenance, logging, project state
src/vslp/acoustic          Acoustic ingest/preprocess/segment/QC/features/aggregation
src/vslp/kinematic         Video/MediaPipe/landmark normalization/kinematic features
src/vslp/feature_analysis  Cross-modal feature analysis
src/vslp/ml                Classical ML, pretrained local inference, explainability
src/vslp/gui               Four PySide6 GUI applications
src/vslp/cli               Reproducible command-line interface
```

## Clinical/product caution

This software is currently designed for research workflows. It should not be used for clinical decision-making until validated, locked, risk-managed, and documented under the appropriate regulatory pathway.


## Launch the acoustic desktop GUI

Install GUI dependencies:

```bash
pip install -e '.[gui,silero]'
```

Launch:

```bash
vslp gui acoustic
```

See `docs/gui_first_run.md` for step-by-step instructions.
