# VSLP Kinematics GUI v0.62 — Real-Frame Landmark Selection Workstation

This patch replaces the abstract landmark mesh selector with a real-frame MediaPipe overlay workstation.

## Main change

The Landmark Selection tab now allows the analyst to:

1. choose an extracted video from the landmark manifest,
2. choose a frame number,
3. load the actual source video frame,
4. overlay the actual Google MediaPipe landmarks from the corresponding `*-lmks.csv` row,
5. click directly on overlaid landmark points to select or deselect them,
6. use region quick-select buttons for mouth/lips, jaw/chin, eye anchors, nose/midline, and brows/upper face,
7. view immediate selection feedback describing region coverage and downstream suitability,
8. save a PNG preview of the selected real-frame overlay,
9. save `selected_landmarks.json` for downstream normalization and feature computation.

## Why this matters

MediaPipe landmark indices are not interpretable by index number alone. The user must see landmark placement on a real video frame before trusting a selected subset. This workstation makes the selection process visual, auditable, and anatomically interpretable.

## Outputs

The selection stage writes:

- `kinematics/003_selection/tables/selected_landmarks.json`
- `kinematics/003_selection/figures/selected_landmark_video_overlay_preview.png`

The original full landmark CSV files remain unchanged. The selected subset only defines the working landmark set for downstream normalization and feature computation.

## Recommended use

Use the ALS oral-motor core preset as a starting point, load a representative frame, and verify that selected landmarks lie on expected mouth, jaw, lip, and eye-anchor locations. Adjust manually when MediaPipe placement, occlusion, pose, or task demands require a different subset.
