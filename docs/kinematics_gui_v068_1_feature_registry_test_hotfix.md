# VSLP Kinematics GUI v0.68.1 — Feature Registry Test Hotfix

This hotfix restores the uploaded laboratory feature-family names in the GUI-facing kinematic feature registry while preserving the v0.68 acoustic-style feature tab and computation path.

Fixes:

- Restores registry groups: Vertical lip/jaw displacement, Horizontal lip spread, Lip aperture geometry, Jaw lateralization, Lip symmetry, Bilateral coordination.
- Restores expected feature identifiers including sLL_vert, aLL_horz, lip_aspect, jaw_lateralization, lip_symmetry, and lat_xcorr.
- Restores the QC table parameter label `ICD availability`.
- Adds a visible `Write Feature Computation Plan` button to the Features tab.

The computation implementation still runs from the normalized landmark layer and writes kinematic feature outputs under `kinematics/006_features`.
