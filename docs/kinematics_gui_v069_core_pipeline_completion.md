# VSLP Kinematics GUI v0.69 Core Pipeline Completion

This release consolidates the post-landmark kinematics workflow after the real-frame MediaPipe landmark workstation.

## Included stages

- Real-frame MediaPipe landmark overlay and landmark selection.
- Progress-log and progress-bar behavior for long-running kinematics stages.
- Computational normalization from selected landmarks.
- Landmark / video QC summary.
- Acoustic-style kinematic feature selector.
- Kinematic feature computation from normalized landmark trajectories.
- Temporal aggregation to participant/video-level feature tables.

## Main output folders

- `kinematics/002_landmarks/tables/` contains full MediaPipe landmark CSVs.
- `kinematics/003_selection/tables/` contains selected landmark profiles.
- `kinematics/004_normalization/tables/` contains normalized landmark CSVs and manifest.
- `kinematics/005_video_qc/tables/` contains video/landmark QC summaries.
- `kinematics/006_features/tables/` contains per-video kinematic features.
- `kinematics/006_features/timeseries/` contains per-frame kinematic feature trajectories.
- `kinematics/007_aggregation/tables/` contains aggregated feature tables.

## Scientific guardrails

QC is advisory and auditable. It flags videos for review based on face-detection burden, long no-face gaps, landmark stability, and normalization adequacy. It does not silently exclude data.

Normalization should be checked before feature computation. The default intercanthal scaling policy uses eye-corner anchors as a camera-distance normalization reference. Raw normalized coordinates can be used for debugging or when anchors are invalid, but should not be treated as the preferred scientific output without justification.

Feature computation depends on selected landmarks and available normalized trajectories. Missing required landmarks are reported as QC flags rather than silently imputed.

## Recommended sequence

1. Run video ingest.
2. Run MediaPipe landmark extraction.
3. Inspect real-frame overlay and save landmark selection.
4. Run computational normalization.
5. Run landmark/video QC.
6. Select kinematic features.
7. Run kinematic feature computation.
8. Run temporal aggregation.
9. Inspect outputs and export reports.
