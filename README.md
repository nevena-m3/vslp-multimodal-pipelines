# VSLP Multimodal Pipelines

VSLP is a local-first research software suite for reproducible acoustic, facial-kinematic, and feature-readiness analysis in clinical speech and motor-assessment studies.

The repository currently contains three working desktop applications:

| Application | Status | Purpose | Primary handoff |
|---|---|---|---|
| Acoustic Pipeline GUI | Workflow complete; validation ongoing | Audio ingest, preprocessing, segmentation, QC, and acoustic feature extraction | Per-recording acoustic feature and QC tables |
| Kinematics Pipeline GUI | Workflow complete; validation ongoing | Video ingest, face landmarks, normalization, QC, kinematic features, and temporal aggregation | Per-video kinematic feature and QC tables |
| Feature Analysis GUI | Workflow complete; validation ongoing | Modality-specific audit, task review, reliability, recommendations, and ML-ready export | Task-specific feature package for the future ML GUI |
| ML GUI | In development | Leakage-safe dataset design, validation, modelling, and interpretation | Not yet release-ready |

VSLP is research software, not a validated medical device. It must not be used for diagnosis, treatment decisions, or regulated clinical claims without independent validation, locked specifications, risk management, and the appropriate regulatory pathway.

## System Workflow

```text
Audio recordings  -> Acoustic Pipeline  --+
                                            +-> Feature Analysis -> Task-specific ML handoff
Video recordings  -> Kinematics Pipeline --+
```

Run Feature Analysis separately for each modality. The Feature Analysis GUI does not combine raw acoustic and kinematic engineering tables and does not train models. Its Advanced ML Export Builder may align completed modality exports when a safe shared recording key exists.

Use one output workspace per study. The GUIs write to isolated component folders within that shared root; see the [project workspace layout](docs/reference/project_workspace.md).

## Requirements

- Python 3.11
- Windows 10/11, macOS, or Linux
- FFmpeg and FFprobe for acoustic workflows
- A local writable project/output directory
- MediaPipe runtime and a Face Landmarker model for kinematic landmark extraction

See [Installation](docs/getting-started/INSTALLATION.md) for platform-specific setup and verification.

## Quick Start

### Windows PowerShell

```powershell
git clone https://github.com/nevena-m3/vslp-multimodal-pipelines.git
cd vslp-multimodal-pipelines
py -3.11 -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[gui,silero,kinematic,dev]"
vslp doctor
```

### macOS or Linux

```bash
git clone https://github.com/nevena-m3/vslp-multimodal-pipelines.git
cd vslp-multimodal-pipelines
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[gui,silero,kinematic,dev]'
vslp doctor
```

## Launch the Applications

```text
vslp gui acoustic
vslp gui kinematics
vslp gui features
```

Dedicated aliases are also installed:

```text
vslp-acoustic-gui
vslp-kinematics-gui
vslp-features-gui
```

Windows users may alternatively run the module commands from the repository environment:

```powershell
python -m vslp.gui.acoustic_app.app
python -m vslp.gui.kinematics.app
python -m vslp.gui.features.app
```

## Recommended Operating Sequence

1. Process audio in the Acoustic Pipeline or video in the Kinematics Pipeline.
2. Inspect stage reports, QC outputs, errors, and manifests.
3. In Feature Analysis, load one modality's `feature_handoff/main` folder; use direct table import only for legacy inputs.
4. Map identifiers, task context, outcomes, covariates, QC variables, and predictors.
5. Run Feature Analysis and review quality, structure, task support, screening, and reliability.
6. Review task-specific recommendations.
7. Create one task-specific Export / Report package per intended ML task.
8. Preserve all transformations and feature selection for the future ML workflow, inside training or cross-validation folds.

## Documentation

- [Documentation index](docs/README.md)
- [Installation and environment verification](docs/getting-started/INSTALLATION.md)
- [Suite user guide](docs/getting-started/USER_GUIDE.md)
- [Standard operating procedures](docs/sops/)
- [Data contracts and technical references](docs/reference/)
- [Architecture and development](docs/development/)
- [Historical implementation records](docs/history/)
- [Contributing](CONTRIBUTING.md)
- [Security and sensitive-data handling](SECURITY.md)

Current guidance is separated from versioned implementation history. Start with the documentation index and use historical records only for provenance.

## Repository Layout

```text
src/vslp/acoustic          Acoustic backend stages
src/vslp/analysis/kinematics Kinematic backend stages
src/vslp/analysis/features Feature audit and export logic
src/vslp/gui               PySide6 desktop applications
src/vslp/cli               Command-line interface
src/vslp/core              Shared project, schema, and manifest utilities
configs                    Versioned default configuration
docs                       Structured user, SOP, reference, development, and history documentation
tests                      Unit and integration tests
```

## Reproducibility and Safety Principles

- Local processing of sensitive data
- Non-destructive source-media handling
- Explicit stage inputs, outputs, configurations, and manifests
- File-level errors without silent data loss
- Conservative preprocessing defaults
- QC as review evidence, not automatic exclusion
- Stable subject identifiers and subject-grouped ML splitting
- Task-specific feature review and export
- No learned preprocessing or feature selection before ML validation folds

## Testing

```powershell
python -m pytest -q
python -m ruff check src tests
```

Focused GUI smoke tests and platform notes are described in [Development](docs/development/DEVELOPMENT.md).

## Attribution and License

Copyright 2026 Nevena Musikic and Yana Yunusova, Speech Production Lab, University of Toronto.

The current repository license is research-use-only and pending final institutional approval. See [LICENSE](LICENSE). Confirm ownership, distribution, data-governance, and release terms with the Speech Production Lab and University of Toronto before public or clinical deployment.
