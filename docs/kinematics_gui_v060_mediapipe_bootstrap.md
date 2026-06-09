# VSLP Kinematics GUI v0.60 — MediaPipe Runtime Bootstrap

This patch turns the Landmarks tab into a self-contained MediaPipe setup and extraction screen.

## Key changes

- Adds an **Install / Verify MediaPipe Runtime** button that installs `opencv-python` and `mediapipe` into the active Python environment using `sys.executable -m pip`.
- Adds a preflight check before extraction so missing MediaPipe/OpenCV/model files are surfaced before batch processing.
- Keeps the **Download Default Model** button for Google's `face_landmarker.task` model bundle.
- Adds an **Auto-prepare runtime/model before extraction** option. When enabled, the GUI will download the model if missing and will prompt to install runtime packages if they are not available.
- Prevents silent all-error extraction manifests by showing a critical dialog when no video succeeds.

## Required output

Successful extraction writes per-video `<video_id>-lmks.csv` files under:

```text
kinematics/002_landmarks/tables/
```

Each landmark CSV contains one row per video frame, a `face_detected` flag, and x/y/z columns for all expected landmarks. Frames with no face are kept as rows with NaN landmarks so tracking gaps remain auditable.
