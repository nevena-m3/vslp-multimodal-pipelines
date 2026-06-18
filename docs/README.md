# VSLP Documentation

This index identifies the current authoritative documentation for the VSLP desktop suite.

## Start Here

| Document | Audience | Purpose |
|---|---|---|
| [Installation](INSTALLATION.md) | All users | Supported Python version, optional dependencies, FFmpeg, MediaPipe, verification, and troubleshooting |
| [User Guide](USER_GUIDE.md) | Researchers and clinicians | End-to-end workflow across the three completed GUIs |
| [Acoustic Pipeline SOP](ACOUSTIC_PIPELINE_SOP.md) | Acoustic operators | Controlled audio-processing procedure and acceptance checks |
| [Kinematics Pipeline SOP](KINEMATICS_PIPELINE_SOP.md) | Kinematic operators | Controlled video/landmark procedure and acceptance checks |
| [Feature Analysis SOP](FEATURE_ANALYSIS_GUI_SOP.md) | Analysts | Task-specific audit, recommendations, and ML handoff procedure |
| [Data Dictionary](data_dictionary.md) | Analysts and developers | Canonical identifiers, context, targets, covariates, QC, and feature roles |
| [Architecture](architecture.md) | Developers and reviewers | Component boundaries, stage contracts, provenance, and leakage controls |
| [Development](DEVELOPMENT.md) | Developers | Environment setup, tests, quality checks, and release procedure |

## Policies

- [Contributing](../CONTRIBUTING.md)
- [Security and sensitive-data handling](../SECURITY.md)
- [Research-use license](../LICENSE)

## Historical Documents

Version-labelled files such as `acoustic_gui_v038_final_polish.md`, `kinematics_gui_v087_feature_exports.md`, and `feature_gui_v133_recommendations_workstation.md` are engineering history. They record incremental implementation decisions and are retained for provenance.

Historical files are not authoritative installation or operating instructions. When a historical note conflicts with a durable document listed above, follow the durable document.
