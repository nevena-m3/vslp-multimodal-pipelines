# Kinematics GUI v0.65.1 normalization/QC merge hotfix

This hotfix preserves computational landmark normalization while restoring the Landmark / Video QC GUI stage and backend summary table. It combines the computational normalization layer with automated video/landmark QC.

Normalization outputs:

- `kinematics/004_normalization/tables/normalization_config.json`
- `kinematics/004_normalization/tables/normalized_landmarks_manifest.csv`
- `kinematics/004_normalization/tables/<video_id>-normalized-lmks.csv`

QC outputs:

- `kinematics/005_video_qc/tables/landmark_video_qc_summary.csv`
- `kinematics/005_video_qc/tables/landmark_video_qc_summary.json`

The GUI exposes `Run Landmark / Video QC` on the Video QC tab.
