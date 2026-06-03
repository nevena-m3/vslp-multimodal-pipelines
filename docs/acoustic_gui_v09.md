# VSLP Acoustic GUI v0.9

V0.9 is a GUI polish and inspection release. It does not change the validated backend stage order. It improves usability around stage execution, feature selection, and output inspection.

## Main additions

- Feature-level selection within each subsystem.
- Quick selectors: All, Implemented, Implemented + Proxy, Clear.
- Feature detail panel showing subsystem, status, unit, meaning, and implementation note.
- Embedded CSV previews for metadata, ingest, preprocessing, segmentation, and feature tables.
- Embedded plot previews for segmentation and feature extraction plots.
- Refresh Latest Outputs button to detect existing output files from disk.
- Clearer stage workflow guidance.

## Feature statuses

`implemented` means the feature is computed by a current plugin.

`proxy` means the feature is computed by an engineering proxy and should not be treated as clinically validated until matched against a reference implementation.

`pending` means the feature is registered and selectable, but currently emitted as an explicit NaN placeholder with status `not_implemented_yet`.

## Recommended user workflow

1. Select input folder.
2. Select output project folder.
3. Run Metadata.
4. Run Ingest.
5. Run Preprocess.
6. Run Silero Segmentation.
7. Select features in the Features tab.
8. Run Feature Extraction.
9. Inspect tables/plots in the Inspector tab.
10. Open full reports from Reports & Outputs.

## Developer note

Feature-level selection is passed into `FeatureExtractionConfig.selected_features`. The backend already supported selected feature whitelisting; v0.9 exposes this safely in the GUI.
