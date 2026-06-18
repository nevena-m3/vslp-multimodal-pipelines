# VSLP Kinematics GUI v0.63 — Landmark Workstation Plotting and Selection Fix

This patch refines the Landmark Selection workstation after real-world testing.

## Purpose

The landmark selector must behave like a professional annotation workstation. Users need to select videos reliably, refresh extracted landmark options, inspect a real video frame, zoom to the face, pan the view, and click directly on Google MediaPipe landmarks without ambiguity.

## Fixes and improvements

- Adds explicit **Overlay view controls** to the Landmark Selection tab.
- Adds **Auto-zoom to detected face** so the face is not tiny when a video frame contains a large background.
- Adds **Zoom In**, **Zoom Out**, **Zoom to Face**, **Reset View**, and **Reload Current Frame** buttons.
- Adds mouse wheel zoom support.
- Adds drag-to-pan support on the overlay canvas.
- Fixes video selection refresh behavior so choosing a different extracted video updates the valid frame range.
- Uses a representative middle frame by default after selecting a video; if no face is detected in that frame, loading falls back to the nearest detected-face frame.
- Keeps selected landmarks highlighted and labelled over the actual video frame.
- Keeps the saved preview image tied to the current real-frame overlay view.

## User workflow

1. Run Setup/Ingest.
2. Run MediaPipe landmark extraction.
3. Go to Landmark Selection.
4. Click **Refresh Extracted Videos**.
5. Choose a video.
6. Click **Load Real Frame + MediaPipe Overlay**.
7. Use **Zoom to Face** or mouse wheel if the face appears small.
8. Click landmarks directly to select or deselect.
9. Save the selected landmark set.

## Notes

The overlay is based on the actual MediaPipe coordinates from the selected video and frame. Frames with no detected face are avoided when possible by falling back to the nearest detected frame.
