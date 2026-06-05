# VSLP v0.30 — Resonatory / Nasality Feature Validation

This pass implements the resonatory/nasality feature family using a conservative, auditable, single-microphone spectral method.

## Implemented features

Primary raw nasality ratios:

- `A1P0`: first-formant amplitude A1 minus low-frequency nasal pole amplitude P0.
- `A1P1`: first-formant amplitude A1 minus ~1 kHz nasal pole amplitude P1.
- `A3P0`: third-formant amplitude A3 minus low-frequency nasal pole amplitude P0.

Compensated variants:

- `A1P0comp`
- `A1P1comp`

These are emitted as `computed_proxy` because exact Chen/Praat-style compensation requires a reference-validated vowel-targeted path. The raw A1-P0/A1-P1/A3-P0 values are the primary validated-local outputs.

Support features:

- `P0freq`, `P0amp`, `P0prom`, `P1amp`
- `F1freq`, `F1amp`, `F1width`
- `F2freq`, `F2amp`, `F2width`
- `F3freq`, `F3amp`, `F3width`
- `RMSamp`

## Signal path

1. Select the configured audio region, default `speech_only`.
2. Frame the signal with 40 ms frames and 10 ms hop.
3. Reject very low-energy frames.
4. Compute a zero-padded Hann-windowed magnitude spectrum.
5. Smooth the dB spectrum with a conservative moving-average smoother.
6. Estimate spectral peaks in broad windows:
   - P0: 180–500 Hz
   - P1: 790–1100 Hz
   - F1: 250–1100 Hz, adjusted upward when P0 would otherwise be reused as F1
   - F2: 800–2600 Hz
   - F3: 1800–3600 Hz
7. Compute amplitude differences and support variables.
8. Summarize per-frame values with the median.

## Interpretation cautions

Nasality from one microphone is fragile. A1-P0/A1-P1/A3-P0 depend on:

- vowel identity;
- whether the selected region contains the intended oral vowel;
- F0 and harmonic placement;
- formant/nasal-pole overlap;
- recording bandwidth and spectral smoothing;
- room/device contamination;
- lack of nasometer/oral-nasal channel separation.

Lower A1-P0/A1-P1/A3-P0 values are generally interpreted as greater nasal coupling, but VSLP treats these as recording- and task-dependent acoustic screening features, not clinical cutoffs.

## Status behavior

- `computed`: sufficient spectral frames and no frequent P0/F1 overlap.
- `computed_with_warnings`: frequent P0/F1 overlap.
- `low_validity`: too few valid spectral frames.
- `computed_proxy`: compensated variants requiring external validation.
- `failed`: missing/empty/too-short input.

## Validation

Unit tests verify:

- finite output for synthetic spectral signals containing P0/F1/P1/F3 components;
- A1-P0 decreases when P0 is strengthened and A1 is dampened;
- resonatory features are registered as implemented/default plugin features.

This is not Praat-identical and should be externally validated against vowel-targeted reference measurements before publication-level clinical claims.
