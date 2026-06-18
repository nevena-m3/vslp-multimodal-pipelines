# VSLP Acoustic GUI v0.38 Final Polish

V0.38 applies final user-facing GUI polish before SOP documentation.

## Changes

- The top institutional branding bar now uses a white background so the Speech Production Lab and University of Toronto logos remain visible.
- The GUI version label is updated to v0.38.
- `Run Full Acoustic Workflow` now checks for metadata before starting.
- If no metadata CSV is selected, the user can:
  - select a metadata CSV,
  - run without metadata,
  - cancel.
- Running without metadata remains supported. VSLP uses filename parsing and the Setup task field where possible; clinical labels remain blank if unavailable.
- Full workflow still runs Metadata, Ingest, Preprocess, Data Segmentation, Quality Control, Feature Extraction, and Run Summary.
- README was rewritten to reflect the current acoustic workflow accurately.

## Notes

This update does not change scientific feature formulas. It is a GUI and documentation polish update.
