# Technical Reference

| Document | Purpose |
|---|---|
| [Data dictionary](data_dictionary.md) | Canonical identifiers, context, outcomes, covariates, QC, and feature roles |
| [Acoustic features](acoustic_features.md) | Acoustic feature definitions and interpretation |
| [Acoustic feature registry](acoustic_feature_registry.csv) | Machine-readable acoustic feature inventory |
| [Acoustic formula traceability](ACOUSTIC_FEATURE_SOURCE_TRACEABILITY.md) | Source hashes, formula families, implementation locations, and source ambiguities |
| [Kinematic feature registry](kinematic_feature_registry.csv) | Machine-readable canonical kinematic inventory with formulas and source mapping |
| [Kinematic source traceability](KINEMATIC_FEATURE_SOURCE_TRACEABILITY.md) | Workbook-to-code mapping, aggregation policy, and unresolved definitions |
| [Cross-modal feature output contract](FEATURE_OUTPUT_CONTRACT.md) | Shared acoustic/kinematic handoff consumed by the Feature GUI |
| [Feature region policy](feature_region_policy.md) | Region and landmark inclusion policy |
| [GUI design](gui_design.md) | Shared desktop-interface conventions |
| [ML GUI data contract](ml_gui_v001_data_contract.md) | Planned ML input and handoff contract |
| [Project workspace layout](project_workspace.md) | Shared study root, component ownership, and modality-scoped analysis outputs |

Reference documents define interfaces and semantics. When an output schema changes, update the relevant reference in the same change as the implementation and tests.
