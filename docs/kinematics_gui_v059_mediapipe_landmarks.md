# VSLP Kinematics GUI v0.59 — MediaPipe Landmark Extraction

This patch connects the Kinematics GUI Landmarks tab to Google MediaPipe Face Landmarker.

## What is implemented

- Optional runtime dependency check for `opencv-python` and `mediapipe`.
- Download button for the public Google FaceLandmarker `.task` model.
- Landmark extraction plan from the video ingest manifest.
- Real per-video MediaPipe extraction from accepted/probed videos.
- One `<video_id>-lmks.csv` landmark table per video.
- `landmarks_manifest.csv` and `landmarks_manifest.json` with extraction status, frame counts, detected-face counts, dropped-frame counts, fps, detected-frame fraction, and errors.
- GUI preview of landmark extraction results.

## Output location

The stage writes to:

```text
kinematics/002_landmarks/tables/
```

Main files:

```text
landmark_extraction_plan.csv
landmark_config.json
landmarks_manifest.csv
landmarks_manifest.json
<video_id>-lmks.csv
```

## Landmark table schema

Each landmark CSV uses:

```text
frame, timestamp_ms, face_detected, 0_x, 0_y, 0_z, 1_x, 1_y, 1_z, ...
```

Frames with no detected face remain in the table with `face_detected=False` and NaN coordinates. This preserves tracking gaps for later QC.

## Dependency notes

The GUI can open without MediaPipe installed. Real extraction requires:

```powershell
python -m pip install opencv-python mediapipe
```

The FaceLandmarker model can be downloaded from the GUI using **Download Default Model**. The default path is:

```text
models/face_landmarker.task
```

## Quality interpretation

MediaPipe Face Landmarker does not provide a simple per-landmark confidence interval in the exported table. This pipeline therefore tracks landmark quality using derived indicators such as face-detected fraction, dropped frames, long gaps, tracking stability, interpolation burden, jitter, and the configured detection/presence/tracking thresholds.

## Scientific guardrail

Landmark extraction is not yet feature computation. A successful landmark CSV only means MediaPipe produced frame-level coordinates. The next stages must still review landmark selection, normalization, video QC, trajectory cleaning, feature computation, and temporal aggregation before kinematic features are interpreted.
