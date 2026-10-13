# Family 11 — intensity and amplitude: implementation review

**Status:** scientific definitions frozen; downstream implementation tested with synthetic data. Source parity, device robustness, reference, real-data, and clinical validation remain pending. These digital amplitudes are **not calibrated physiological intensity**. Microphone distance/sensitivity, gain, AGC, compression, codec, clipping, and noise can change them.

## Scope and regions

Five historical constructs (C060–C064) produce 20 selectable, task-specific leaves. S5 is “We were away a year ago.” C060/C061 use sustained /a/, S5, and Bamboo. C062/C064 use sustained /a/ and S5. C063 uses sustained /a/ and Bamboo. All consume frozen reviewed timing and the original/native source waveform; none require MFA, TextGrid, or Final Alignment.

- Sustained /a/: full accepted phonation onset through termination, never F01's centered stable window.
- S5/Bamboo C060/C061/C062/C064: chronological eligible reviewed speech intervals, with hard exclusions removed. Concatenation is a sample-statistics operation; it does not assert continuous physical time. C061 divides by this concatenated support duration.
- Bamboo C063: first through last eligible reviewed speech, including ordinary internal pauses. Any hard exclusion inside the passage span makes C063 unavailable with `hard_exclusion_ambiguity`; the waveform is not concatenated across it.
- Every reviewed trial is measured separately. The recording output is the median of valid trial scalars. The native trial audit retains identity, bounds, eligible intervals, source audio, and parameters.

## Exact formulas

For C060/C061 only, each eligible reviewed speech interval is independently resampled to 48,000 Hz by `scipy.signal.resample_poly` using rational factors. Energy and sample counts are then summed. **The resampler never filters across an artificial concatenation boundary.** The native source remains untouched. This removes differences caused solely by 44.1 versus 48 kHz sample counts, but does not remove channel effects. No peak normalization or F11 DC subtraction is applied.

- C060: `E = Σ x48k[n]²`, digital-amplitude² × samples, arbitrary digital units.
- C061: `P = E / (N48k / 48000)`, arbitrary-scale digital power. It is neither watts, Pa², nor SPL.
- C062: `scipy.stats.skew(x, bias=False)` and `scipy.stats.kurtosis(x, fisher=True, bias=False)` on native samples. The latter is **excess** kurtosis. Positive linear gain leaves these ideally unchanged; clipping, AGC, codec distortion, and preprocessing can change them. They are relatively gain robust, not device invariant.
- C063: native waveform → complete nonoverlapping 50 ms windows → sample SD (`ddof=1`) per window → **natural** visibility graph with strict line-of-sight slope → `2m/[M(M−1)]`. Sample width is `round(0.050 × native_sample_rate_hz)` (2205 at 44.1 kHz, 2400 at 48 kHz); incomplete final samples are discarded and audited. The Bamboo pathway follows connected-speech source lineage; the vowel pathway is a VSLP task adaptation. MATLAB/source parity remains pending.
- C064: `mean(x)`, sample `SD(x)` with `ddof=1`, `min(x)`, and `max(x)` on native samples. Mean is mainly a DC/asymmetry diagnostic, not loudness. Min/max are gain-sensitive, clipping-sensitive, and polarity-sensitive: reversing signal polarity swaps and negates the extrema. All four are channel sensitive.

The native region audit retains DC offset, peak absolute amplitude, fraction with `|x| ≥ 0.999`, and fraction with `|x| ≥ 0.98` on nominal digital full scale. Clipping is flagged in audit; mathematically valid raw measurements are not silently discarded. Nonfinite/empty audio, near-zero variance for moments, insufficient C063 windows, and invalid trial/audio bounds yield NaN with an explicit reason.

The old generic IDs and single-recording pooling path remain legacy/nonselectable. Current public IDs are task-specific in `family11.json`. C060/C061 also retain nonselectable per-trial `mean_square_power_support`, plus participant/session/recording/task/trial and algorithm/sample-rate provenance for later analysis.

## Deferred Feature Analysis design

Potential same-session derived measures are `s5_relative_power_vs_vowel_db` and `bamboo_relative_power_vs_vowel_db`, using `10 log10(P_task/P_vowel)` where each `P` is F11 mean-square-power support. Matching would require the same participant, session, acquisition setup where possible, and F11 preprocessing/version. These ratios are **not** extraction-time F11 leaves.
