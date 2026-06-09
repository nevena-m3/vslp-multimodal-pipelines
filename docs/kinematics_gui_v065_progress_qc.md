# VSLP Kinematics GUI v0.65 — Progress Logging and Landmark / Video QC

This patch improves user visibility during long-running kinematics stages and adds the first functional Video QC stage.

## User-facing changes

- The bottom Run Log now receives explicit stage start, completion, and failure messages for threaded long-running operations.
- The bottom progress bar is now used during long-running stages. For extraction and QC it switches to an indeterminate busy state while the task is running, then returns to idle when complete.
- Real-frame landmark overlay loading now briefly advances the progress bar during landmark-row lookup, video-frame decoding, and overlay rendering.
- Repetitive OpenCV/FFmpeg WebM console warnings are suppressed where possible using `OPENCV_FFMPEG_LOGLEVEL=-8`. Decode failures are still surfaced through the GUI when they affect outputs.
- The Video QC tab now computes automated landmark/video QC instead of writing only a placeholder plan.

## New Video QC outputs

After running `Video QC -> Run Landmark / Video QC`, outputs are written to:

```text
kinematics/005_video_qc/tables/landmark_video_qc_summary.csv
kinematics/005_video_qc/tables/landmark_video_qc_summary.json
```

The QC summary currently includes:

- video ID and landmark CSV path
- face-detected fraction
- number of decoded frames
- number of detected-face frames
- number of no-face runs
- maximum consecutive no-face gap
- selected-landmark frame-to-frame displacement statistics
- automated `pass` / `review` / `fail` status
- written rationale for the status

## Scientific interpretation

The automated QC stage is conservative. It identifies videos that may be risky for downstream kinematic feature computation, but it does not automatically exclude any video. Analyst review remains required before final feature computation or model export.

The current QC layer focuses on extraction quality from MediaPipe landmarks. Later versions can add illumination, head pose, task adherence, and task-specific movement-structure checks.
