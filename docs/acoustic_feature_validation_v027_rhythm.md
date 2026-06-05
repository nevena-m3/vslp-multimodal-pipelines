# VSLP v0.27 Rhythm / Envelope Modulation Feature Validation

This update validates the rhythm feature group derived from the envelope modulation spectrum (EMS).

## Validated features

- `intensity_CV`
- `fft_peaks1`
- `fft_peaks2`
- `fft_ampli1`
- `fft_ampli2`
- `nrj_below_boundary`
- `nrj_above_boundary`
- `nrj_3_6`
- `ratio_below_above`

## Scientific policy

Rhythm features are computed over the **effective task interval** by default, not concatenated speech-only samples. This preserves internal pauses and speech/pause alternation, which are part of connected-speech rhythm.

Default EMS signal path:

1. Select effective task region from first speech onset to last speech offset.
2. Preserve internal pauses.
3. Apply 300--1000 Hz Butterworth speech-band prefilter.
4. Compute Hilbert amplitude envelope.
5. Resample envelope to 100 Hz.
6. Detrend and apply Tukey window.
7. Compute FFT over 0--10 Hz.
8. Split modulation power at the 4 Hz boundary.

Band-energy features are proportions of total 0--10 Hz modulation power.

## Important limitation

These rhythm features are **not syllable counts**, **not DDK rate**, and **not SPA peak rate**. They quantify low-frequency amplitude-envelope structure in connected speech.
