# Acoustic GUI v0.19 - Preprocessing refinement

This update refines preprocessing controls and outputs. The stage remains non-destructive. Source files are never modified.

## Default policy

- Remove DC offset by default.
- Create a 16 kHz segmentation WAV for Silero by default.
- Create a feature WAV by default while preserving the original sample rate unless feature resampling is explicitly enabled.
- Do not normalize by default, because normalization can change amplitude and intensity features.
- Do not filter by default, because filtering can alter acoustic features.
- Always compute QC measurements: estimated SNR, clipping, DC offset, and 50/60 Hz powerline flags.

## Optional transformations

The user may enable:

- Peak normalization with selectable target peak.
- Feature-WAV resampling with selectable sample rate.
- Stable Butterworth high-pass, low-pass, and band-pass filters.
- 50 Hz or 60 Hz notch filtering.

## Output tables

Primary table:

- `acoustic_preprocess_main_summary.csv`

Detailed technical table used by downstream stages:

- `acoustic_preprocess_summary.csv`

QC flag table:

- `acoustic_preprocess_qc_flags.csv`

Per-file JSON records:

- `reports/per_file_qc_json/`

## Plots

- `preprocess_snr_distribution.png`
- `preprocess_clipping_distribution.png`
- `preprocess_dc_offset_before_after.png`
