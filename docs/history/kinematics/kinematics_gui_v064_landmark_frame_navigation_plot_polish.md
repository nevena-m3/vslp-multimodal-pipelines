# VSLP Kinematics GUI v0.64 — Landmark frame navigation and overlay polish

This hotfix improves the real-frame landmark workstation. It fixes frame navigation so the frame slider and numeric frame control can reload the displayed MediaPipe overlay, adds debounced auto-reload, updates the selected-video reload behavior, and restyles the overlay canvas to remove the distracting white plotting background.

Key changes:

- Frame slider now synchronizes with the numeric frame box.
- Numeric frame edits now synchronize with the slider.
- Auto-reload can update the real frame and landmark overlay after the first frame is loaded.
- Changing video updates the frame range and reloads when auto-reload is active.
- Reload Current Frame still forces an immediate redraw.
- The overlay canvas now uses a dark workstation background so the video frame is visually isolated without a white border.
- Frame fallback to nearest detected-face frame no longer triggers a reload loop.

This patch changes GUI behavior only. It does not change MediaPipe extraction, landmark CSVs, feature computation, or downstream data outputs.
