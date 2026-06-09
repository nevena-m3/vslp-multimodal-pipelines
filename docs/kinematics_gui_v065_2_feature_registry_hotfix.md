# v0.65.2 Kinematics Feature Registry Hotfix

This hotfix restores the declarative kinematic feature registry symbols required by the Feature Selector tests and GUI while preserving the computational feature implementation layer.

It provides:

- `KinematicFeatureSpec`
- `KINEMATIC_FEATURE_SPECS`
- `KINEMATIC_FEATURE_GROUPS`
- `DEFAULT_KINEMATIC_FEATURE_IDS`
- `QC_FEATURE_REQUIREMENTS`
- `feature_registry_dataframe()`
- `selected_specs()`

The registry documents feature groups, landmark dependencies, intercanthal-distance normalization expectations, aggregation policy, and QC preconditions.
