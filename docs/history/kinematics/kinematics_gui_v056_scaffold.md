# VSLP Kinematics GUI v0.56 Scaffold

This patch introduces the first user-facing Kinematics GUI scaffold. It mirrors the acoustic workflow while preparing for video-based facial kinematic analysis.

## Launch

```powershell
python -m vslp.gui.kinematics.app
```

## Implemented menu scaffold

1. Setup / Ingest
2. Metadata
3. Face Landmarks
4. Landmark Selection
5. Normalization
6. Video QC
7. Feature Computation
8. Temporal Aggregation
9. Data Inspector
10. Reports & Outputs

## Current implemented functionality

- Recursive video discovery.
- Broad container support: MP4, WebM, MOV, MKV, AVI, WMV, MPG/MPEG, M4V, 3GP.
- ffprobe-based structural video manifest when ffprobe is available.
- Metadata loading and conservative link preview.
- MediaPipe Face Landmarker configuration planning.
- Landmark preset selection, including ALS oral-motor core 15 and additional lower-face/symmetry/hypomimia configurations.
- Normalization method selection.
- Video QC placeholder taxonomy.
- Feature computation placeholder aligned with uploaded kinematic feature modules.
- Temporal aggregation profile selection.
- Initial scaffold HTML report.

## Scientific guardrails

Frame-level landmark trajectories are not scalar biomarkers until they have passed video QC, landmark tracking review, normalization, feature computation, and temporal aggregation. The GUI does not yet compute final kinematic biomarkers; it establishes the professional workflow and configuration surfaces.

## MediaPipe quality note

The GUI records face-detection/presence/tracking thresholds and plans to derive quality metrics from face-detected fraction, missing landmark frames, long gaps, jitter, pose/head movement, and tracking stability. The scaffold does not assume that a per-landmark confidence interval is available.
