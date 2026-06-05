# VSLP v0.29 — Articulatory / formant feature validation pass

This update implements the registered articulatory/formant feature group using a conservative LPC-root tracking path.

## Implemented features

- `f1`–`f5`: median formant frequencies.
- `f1_bw`–`f5_bw`: median formant bandwidths.
- `f1_range`, `f2_range`, `f3_range`: robust 95th-minus-5th percentile formant range.
- `f1_d_dx_*`, `f2_d_dx_*`, `f3_d_dx_*`: first-derivative summaries from smoothed formant trajectories.

## Default signal path

1. Use speech-region audio by default.
2. Downsample to approximately 10 kHz for LPC formant tracking when the source sample rate is higher.
3. Frame using 25 ms frames and 10 ms hop.
4. Apply DC removal, pre-emphasis, and Hamming windowing.
5. Estimate LPC coefficients by the autocorrelation method.
6. Convert complex roots into formant frequencies and bandwidths.
7. Apply broad adult-speech plausibility filters.
8. Summarize only valid frames.

## Important interpretation note

These are auditable local LPC estimates. They are internally tested on synthetic all-pole vowels and protected by validity filters, but they are not claimed to be numerically identical to Praat or any other reference engine. External reference validation is still recommended before clinical interpretation.

## Why this is conservative

Formants are fragile in remote patient recordings. F0 errors, noise, breathiness, low bandwidth, speech task, sex/anatomy, vocal tract length, frame voicing, and LPC order can all affect estimates. VSLP therefore writes per-feature status notes, including the fraction of frames with valid F1/F2.
