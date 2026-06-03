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

## First smoke test

```bash
vslp project init --output-root examples/demo_project/output --project-name demo
vslp acoustic ingest --input examples/demo_project/input --output-root examples/demo_project/output
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
