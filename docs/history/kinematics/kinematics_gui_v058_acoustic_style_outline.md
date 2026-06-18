# VSLP Kinematics GUI v0.58 — Acoustic-style workflow outline

This patch replaces the previous commercial-card kinematics outline with an acoustic-pipeline-matched GUI shell.

## Intent

The kinematics GUI now intentionally mirrors the VSLP Acoustic Pipeline interface:

- left VSLP sidebar with stage status cards;
- compact institutional branding bar with Speech Production Lab and University of Toronto logos;
- top tab workflow;
- bottom run log and progress bar;
- dark scientific workstation theme;
- same stage naming philosophy as the acoustic GUI, adapted for video/facial kinematics.

## Kinematics tabs

1. Setup
2. Metadata
3. Landmarks
4. Landmark Selection
5. Normalization
6. Video QC
7. Features
8. Aggregation
9. Inspector
10. Reports & Outputs

## Notes

This is still an outline/scaffold. It keeps lightweight ingest, metadata linking, landmark plan writing, landmark selection, normalization config writing, placeholder QC/features/aggregation, table inspection, and scaffold report generation. MediaPipe execution and full kinematic feature computation will be connected in later patches.
