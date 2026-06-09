# VSLP Kinematics GUI v0.60.1 — MediaPipe Runtime Button Hotfix

This hotfix makes the MediaPipe runtime setup explicit and visible in the Landmarks tab.

## Changes

- Adds a visible **Install / Verify MediaPipe Runtime** button to the MediaPipe Face Landmarker configuration panel.
- Adds an **Auto-install missing runtime and download model before extraction** checkbox, enabled by default.
- Before extraction, the GUI now checks for `opencv-python` and `mediapipe` in the active VSLP virtual environment.
- If packages are missing and auto-prepare is enabled, the GUI prompts the user and installs the packages with the Python interpreter that launched VSLP.
- If packages remain unavailable, landmark extraction stops with a clear message instead of writing an all-error manifest.

## User-facing behavior

A user does not need to manually know how to install Google MediaPipe. The recommended workflow is:

1. Run Setup / Ingest.
2. Open Landmarks.
3. Click **Install / Verify MediaPipe Runtime**.
4. Click **Download Default Model**.
5. Click **Run MediaPipe Landmark Extraction**.

The extraction writes one `<video_id>-lmks.csv` table per video, preserving no-face frames as `face_detected=False` with NaN landmark coordinates for downstream QC.
