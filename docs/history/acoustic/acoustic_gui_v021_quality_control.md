# VSLP v0.21 - Segmentation-informed Quality Control

This release adds a Quality Control stage immediately after Data Segmentation and before acoustic feature extraction.

## Purpose

The QC module extracts artifact-oriented quality features from the segmentation outputs. These features are intended to flag recordings for review, not to automatically reject clinical data.

## QC feature families

1. Additive interference: background noise, hum, and pause-region contamination.
2. Gain dynamics: amplitude instability, level drift, and automatic-gain-control-like behavior.
3. Reverberation / echo: post-speech offset tails and pause energy consistent with room echo.
4. Channel / device: spectral coloration, bandwidth limits, and device-response differences.
5. Nonlinear distortion: clipping, saturation, near-clipping, and peak flattening.
6. Temporal discontinuity: dropouts, silent holes, repeated windows, and abrupt waveform breaks.

## Inputs

The stage consumes the Silero/Data Segmentation summary table and its referenced frame tables, segment tables, and segmentation WAV files.

## Outputs

Outputs are written under `acoustic/003_quality_control/`:

- `tables/acoustic_quality_features.csv`
- `tables/acoustic_quality_main_summary.csv`
- `tables/acoustic_quality_family_status.csv`
- `tables/acoustic_quality_family_summary.csv`
- `plots/quality_family_status.png`
- `plots/quality_main_features_overview.png`
- `reports/acoustic_quality_control_report.html`
- `logs/stage_manifest.json`

## Notes

The implementations are conservative proxy/screening features based on the uploaded QC notebooks. They require validation on representative recordings before being used as formal exclusion criteria.
