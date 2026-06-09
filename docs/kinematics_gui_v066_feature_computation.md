# VSLP Kinematics GUI v0.66 - Feature Computation

This update adds the first functional kinematic feature-computation stage.

## Inputs

The stage expects normalized landmark trajectories from:

`kinematics/004_normalization/tables/normalized_landmarks_manifest.csv`

Each per-video normalized landmark table is expected to contain normalized columns such as:

`13_x_norm`, `13_y_norm`, `13_z_norm`, etc.

## Outputs

The stage writes:

- `kinematics/006_features/tables/kinematic_features.csv`
- `kinematics/006_features/tables/kinematic_feature_registry.csv`
- `kinematics/006_features/tables/feature_computation_manifest.json`
- `kinematics/006_features/timeseries/<video_id>-kinematic-timeseries.csv`

## Feature families

The initial implemented feature set includes:

- mouth aperture
- outer and inner lip spread
- lip aspect ratio
- jaw-to-nose displacement
- lower-lip-to-chin distance
- mouth-corner vertical asymmetry
- mouth-corner lateral asymmetry
- absolute speed summaries
- path-length summaries
- movement-range summaries

## Processing policy

Feature computation follows the uploaded laboratory script structure:

1. interpolate missing samples;
2. remove distributional outliers with a two-pass sigma rule;
3. optionally smooth cleaned trajectories using a low-pass Butterworth filter;
4. compute interpretable geometry signals;
5. segment movement windows from mouth aperture;
6. summarize time series using robust scalar statistics.

## QC behavior

Feature computation flags risk but does not automatically exclude videos. The feature table preserves rows with `status = qc_flagged` and records reasons in `feature_qc_flags`.

