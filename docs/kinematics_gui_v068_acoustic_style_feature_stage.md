# VSLP Kinematics GUI v0.68 - Acoustic-Style Feature Stage

This patch reorganizes the Kinematics **Features** tab to match the Acoustic Pipeline feature-computation workflow more closely.

## What changed

- Adds a grouped feature selector tree by biomechanical subsystem.
- Shows implementation status, tier, landmark dependencies, native signal/unit, normalization policy and scalar aggregation policy.
- Adds quick selection buttons: All, Default oral-motor, Tier B only and Clear.
- Keeps real numerical feature computation connected through **Run Kinematic Feature Computation**.
- Adds selected-feature summary and detailed interpretation panel.
- Adds a QC requirements table that explains how normalization/QC gates affect feature validity.
- Preserves the existing normalized-landmark input and kinematic feature output files.

## Scientific policy

Default feature computation expects computational normalization to have produced ICD-normalized landmark trajectories. The current default normalization uses intercanthal distance from MediaPipe landmarks 243 and 463 as the scale anchor. This is a scale normalization layer, not a cure for poor tracking or head-pose artifacts. Feature validity still depends on detected-frame fraction, anchor availability, long no-face gaps, jitter and frame timing.

## Outputs

The feature stage writes:

- `kinematics/006_features/tables/kinematic_features.csv`
- `kinematics/006_features/tables/kinematic_feature_registry.csv`
- `kinematics/006_features/tables/feature_computation_manifest.json`
- `kinematics/006_features/timeseries/<video_id>-kinematic-timeseries.csv`
