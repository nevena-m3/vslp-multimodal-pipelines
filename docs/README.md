# VSLP Documentation

This directory separates current operating guidance from technical references and implementation history.

## Start Here

| Document | Audience | Purpose |
|---|---|---|
| [Installation](getting-started/INSTALLATION.md) | All users | Environment setup, optional dependencies, verification, and troubleshooting |
| [User Guide](getting-started/USER_GUIDE.md) | Researchers and clinicians | End-to-end workflow across the completed GUIs |
| [Acoustic Pipeline SOP](sops/ACOUSTIC_PIPELINE_SOP.md) | Acoustic operators | Controlled audio-processing procedure and acceptance checks |
| [Kinematics Pipeline SOP](sops/KINEMATICS_PIPELINE_SOP.md) | Kinematic operators | Controlled video/landmark procedure and acceptance checks |
| [Feature Analysis SOP](sops/FEATURE_ANALYSIS_GUI_SOP.md) | Analysts | Task-specific audit, recommendations, and ML handoff procedure |

## Documentation Map

| Section | Contents |
|---|---|
| [`getting-started/`](getting-started/) | Installation and suite-level use instructions |
| [`sops/`](sops/) | Controlled operating procedures for each completed GUI |
| [`reference/`](reference/) | Data dictionary, feature definitions, policies, registries, and ML handoff contract |
| [`development/`](development/) | Architecture, contributor environment, tests, and plugin development |
| [`history/`](history/) | Versioned engineering records retained for provenance |

## Policies

- [Contributing](../CONTRIBUTING.md)
- [Security and sensitive-data handling](../SECURITY.md)
- [Research-use license](../LICENSE)

Only the documents linked under **Start Here**, `reference/`, and `development/` are maintained as current guidance. Historical records may describe superseded behavior and must not be used as operating instructions.
