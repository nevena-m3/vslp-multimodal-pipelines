# VSLP Kinematics GUI v0.61 — Visual Landmark Selector

This patch improves the Landmark Selection stage by adding an interactive MediaPipe face-mesh viewer.

## Purpose

MediaPipe Face Landmarker extraction writes hundreds of per-frame landmark coordinates. Analysts should not have to select landmarks only by numeric index. The visual selector provides a face-mesh canvas so users can load an extracted landmark file, inspect the median face mesh, and click points to include or exclude landmarks.

## Workflow

1. Run Setup and Video Ingest.
2. Run MediaPipe landmark extraction.
3. Open Landmark Selection.
4. Click **Load Mesh From Extracted Landmarks**.
5. Choose a preset or click landmarks directly on the visual face mesh.
6. Click **Apply / Save Selected Landmarks**.

## Outputs

The selection stage writes:

- `kinematics/003_selection/tables/selected_landmarks.json`
- `kinematics/003_selection/figures/selected_landmark_mesh_preview.png`

## Interpretation

The selected landmarks define the default subset for later normalization and feature computation. Full MediaPipe landmarks remain available in the original landmark CSV files for audit, QC, and future exploratory analysis.

The canvas uses median x/y coordinates from the most recent extracted `*-lmks.csv` file when available. If no extraction has been run, it shows a generic template so the user can still review presets, but real extracted landmarks are preferred.
