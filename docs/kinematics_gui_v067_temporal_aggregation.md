# VSLP Kinematics GUI v0.67 - Temporal Aggregation

This update turns the Aggregation tab from a plan-only stage into a functional temporal aggregation layer.

## Purpose

The Feature stage writes both per-video scalar features and frame-level kinematic time series. The Aggregation stage now explicitly collapses those frame-level time-series outputs into an auditable per-video table using a selected profile.

## Inputs

The stage expects the Feature stage output:

- `kinematics/006_features/tables/kinematic_features.csv`
- `kinematics/006_features/timeseries/<video_id>-kinematic-timeseries.csv`

## Outputs

The stage writes:

- `kinematics/007_aggregation/tables/aggregation_config.json`
- `kinematics/007_aggregation/tables/kinematic_aggregated_features.csv`
- `kinematics/007_aggregation/tables/movement_level_features.csv`
- `kinematics/007_aggregation/tables/aggregation_manifest.json`

## GUI behavior

The Aggregation tab now includes:

- aggregation profile selector;
- minimum valid-fraction threshold;
- minimum detected-face fraction threshold;
- option to include raw unsmoothed feature signals;
- option to include velocity-derived signals;
- Run Temporal Aggregation;
- Refresh Aggregation Outputs;
- results preview table.

## Aggregation profiles

The profile selector uses the existing project policy definitions:

- `robust_default`: median, IQR, 5th/95th percentiles, and robust spread;
- `movement_segmented`: movement-level rows plus per-video movement summaries;
- `full_timeseries_summary`: whole-video trajectory summaries;
- `clinically_sensitive`: robust summaries with additional distributional statistics;
- `exploratory_dense`: a wider exploratory scalar table for later feature-review stages.

## QC behavior

Aggregation flags low detected-face fraction, low valid signal fraction, missing time series, and empty signal sets. It does not automatically exclude videos. The output table retains `status`, `aggregation_qc_flags`, `feature_status`, and `feature_qc_flags` so downstream feature-analysis and ML stages can decide how to filter or stratify.
