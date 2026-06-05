# VSLP v0.28 — Phonatory feature validation pass

This update implements the complete registered phonatory feature group using transparent local signal-processing algorithms.

## Implemented features

- `f0_mean`
- `f0_std`
- `CPP_mean`
- `HNR`
- `localJitter`
- `localabsoluteJitter`
- `rapJitter`
- `ppq5Jitter`
- `ddpJitter`
- `localShimmer`
- `localdbShimmer`
- `apq3Shimmer`
- `apq5Shimmer`
- `apq11Shimmer`
- `num_voicebreaks`
- `H1freq`
- `H1amp`
- `H2freq`
- `H2amp`

## Implementation principles

- Region-aware extraction defaults to speech/voiced regions.
- F0 and HNR are computed from normalized autocorrelation.
- CPP is computed as a line-normalized cepstral peak prominence estimate.
- Jitter and shimmer families are computed from voiced-frame period and RMS-amplitude tracks.
- H1/H2 are estimated from spectral peaks near F0 and 2×F0.
- Voice breaks are counted as internal unvoiced runs between voiced regions.

## Scientific caution

The formulas are explicit and internally tested on synthetic signals. However, jitter/shimmer are not guaranteed to match Praat, MDVP, or other commercial implementations exactly because VSLP does not yet estimate individual glottal-cycle boundaries. Treat these as local, auditable implementations pending external reference validation.

## Recommended use

- Best used for sustained-vowel recordings.
- `f0_mean`, `f0_std`, `CPP_mean`, and `HNR` may be useful on connected speech, but perturbation features are most defensible on stable sustained phonation.
- Interpretation must account for sex, age, microphone/acquisition quality, SNR, clipping, and voicing-tracker reliability.
