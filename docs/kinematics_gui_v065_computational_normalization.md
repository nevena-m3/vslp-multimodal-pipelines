# VSLP Kinematics GUI v0.65 - Computational Normalization

This patch makes the Normalization stage computational rather than configuration-only.

## Purpose

The stage reads the full MediaPipe landmark CSV files from `kinematics/002_landmarks/tables`, uses the visual landmark selection from `kinematics/003_selection/tables/selected_landmarks.json`, and writes normalized landmark trajectories for downstream QC and feature computation.

## Output

The stage writes to:

- `kinematics/004_normalization/tables/normalization_config.json`
- `kinematics/004_normalization/tables/normalized_landmarks_manifest.csv`
- `kinematics/004_normalization/tables/<video_id>-norm-lmks.csv`

## Default method

The default method is `intercanthal_distance`. It scales coordinates by the robust median distance between MediaPipe landmarks 133 and 362, which are used as inner eye/canthus anchors. Coordinates are centered on landmark 1 by default and expressed in units of the selected scale.

## QC fields

The manifest reports:

- `face_detected_fraction`
- `scale_valid_fraction`
- `selected_complete_frame_fraction`
- `mean_selected_landmark_availability`
- `missing_selected_landmarks`
- `qc_flags`

These fields do not automatically exclude videos. They provide transparent risk evidence for downstream Video QC and feature computation.

## GUI behavior

The Normalization tab now includes:

- Write Normalization Config Only
- Run Computational Normalization
- center landmark control
- overwrite control
- output results table

The bottom run log and progress bar update during normalization.
