# VSLP Acoustic GUI v0.13 — Visual and Scientific Rationale Pass

This update focuses on GUI quality rather than new feature formulas.

## Goals

- Make the Acoustic GUI more professional, readable, and visually coherent.
- Make the scientific rationale for every major control visible inside the interface.
- Preserve conservative starting values and explain why they are defaults.
- Improve table and plot preview usability.
- Keep backend execution local, reproducible, and audit-friendly.

## Key defaults and rationale

### Segmentation sample rate: 16 kHz
Silero VAD is designed for 8 kHz or 16 kHz speech inputs. VSLP therefore creates a 16 kHz segmentation WAV by default while preserving the original sampling rate for feature WAVs unless the user explicitly requests resampling.

### Feature sample rate: blank by default
Leaving this blank preserves the original sampling rate for feature computation. This avoids unnecessary resampling artifacts in acoustic measures that may be sensitive to bandwidth.

### Silero threshold: 0.50
A 0.50 speech probability threshold is a neutral starting point. Increasing it makes detection more conservative; decreasing it may increase sensitivity at the cost of more false speech detections.

### Minimum speech duration: 250 ms
This suppresses very short transient detections while preserving short speech bursts relevant to DDK and dysarthric speech.

### Minimum silence duration: 100 ms
This preserves clinically relevant short pauses while avoiding excessive fragmentation.

### Speech padding: 50 ms
Small padding reduces boundary truncation at speech onsets and offsets.

### Diagnostic frame size: 30 ms
Thirty milliseconds is a conventional speech-analysis time scale and supports interpretable frame-level diagnostic plots.

### Feature region policy: speech_only
Most signal-derived features should not be computed blindly over the full file. The default `speech_only` policy computes signal features from detected speech regions. Timing and pause features use the Silero segment tables directly.

### Minimum pause duration: 150 ms
This is a conservative initial threshold for internal pause features. It should be adjusted per task/study if needed.

### QC SNR threshold: 10 dB
This is a research-screening flag, not a clinical exclusion criterion. VSLP reports estimated SNR as a proxy because clean-reference SNR is not available in remote recordings.

### Maximum clipping fraction: 0.001
A fraction of 0.001 corresponds to 0.1% of samples near clipping and is used as an initial review flag.

## GUI improvements

- Refined navy clinical/research theme.
- Information panels on each stage tab.
- Better stage status badges.
- Control tooltips with practical scientific rationale.
- Feature-status coloring in the feature selector.
- Improved feature detail panel.
- Inspector plot preview now rescales to available space.
- Inspector controls are grouped into table previews and plot previews.

## Important limitation

This release does not validate every acoustic feature formula. It improves the GUI and makes the current feature state more transparent. Feature validation remains the next major scientific task.
