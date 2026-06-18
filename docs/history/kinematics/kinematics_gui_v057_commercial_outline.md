# VSLP Kinematics GUI v0.57 — Commercial Workflow Outline

This patch upgrades the initial kinematics GUI scaffold into a polished, professional workflow outline aligned with the completed acoustic GUI style.

## Scope

The GUI remains a scaffold. It does not yet run full MediaPipe extraction or final kinematic feature computation. It now provides the high-level user-facing structure, visual workflow, branding, stage objectives, outputs, decision rules, landmark preset guidance, normalization guidance, QC taxonomy, feature-family outline, aggregation strategy outline, inspector page, and scaffold report.

## Menus

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

## Branding and ownership

The GUI includes Speech Production Lab and University of Toronto logos in the stage headers, plus a visible copyright/intended-use statement in the Reports & Outputs stage.

## Scientific guardrails

The interface emphasizes that video landmark trajectories are frame-level time series and should not be interpreted as scalar biomarkers until landmark extraction, normalization, QC review, feature computation, and temporal aggregation have been completed and documented.

## Next recommended patch

v0.58 should connect the real MediaPipe landmark extraction backend and write per-video landmark CSV outputs, manifests, and logs.
