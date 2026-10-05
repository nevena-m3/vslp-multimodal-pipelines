# F02 implementation for scientific review (superseded draft)

The active Group 2 contract is [family02_scientific_freeze.md](family02_scientific_freeze.md). The 700-Hz GNE band, fourth-order PVI filter, Hann/512 H1–H8 analysis, and N−2 DFP denominator below describe the prior draft and are superseded.

This is an engineering implementation, not independent reference parity, real-data
validation, or clinical validation. F02 uses the canonical acoustic analysis WAV,
frozen reviewed segmentation, upstream task identity, and original-time trial
coordinates. Linguistic Alignment is not required. The current sustained-vowel
protocol contains /a/ only.

## Task and aggregation contract

| Construct | Sustained /a/ | S5 | S6 | Bamboo |
| --- | --- | --- | --- | --- |
| Jitter, shimmer, HNR | Recommended | Recommended | Available, not recommended | Available, not recommended |
| GNE, PVI | Research replication | Unavailable for this domain | Unavailable | Unavailable |
| Unsmoothed CPP | Recommended | Recommended | Recommended | Recommended |
| DFP, H1–H8 | Recommended | Recommended VSLP extension | Available, not recommended | Available, not recommended |

S5 is “We were away a year ago”; S6 is “My mama makes lemon jam”. Other
sentences and DDK retain executability where the algorithm has valid evidence,
separate from recommendation. GNE and PVI are restricted to reviewed sustained
/a/ because their present source-replication domain is a stable vowel.

Each sustained-/a/ trial reuses F01's longest contiguous reviewed stable
phonation interval after exclusions, then its centered **up to 2.0 s**. Connected
speech uses contiguous valid voiced islands within reviewed speech and trial
bounds. No pulse or spectral window bridges an unvoiced gap or a manual
exclusion. Island values are weighted by valid cycles or frames within a trial;
the recording result is the median of valid trial values, feature by feature.
The [F02 registry profile](../src/vslp/acoustic/features/data/family02.json)
records the parameters. `voice_quality_trial_summaries.csv` records trial
values, islands, cycle/frame counts, failures, and source-parity state.

## Established Praat paths

- Jitter retains local %, absolute seconds, RAP %, PPQ5 %, and DDP %. Praat
  PointProcess uses pitch 60–500 Hz, periods 0.0016–0.020833 s, and maximum
  period factor 1.3. Sustained /a/ retains the existing 100-usable-cycle
  technical gate; disconnected sentence islands use a 4-cycle computational
  gate, with Praat undefined outputs remaining failures. Sentence and vowel
  contexts are identified in provenance and should not be pooled as if
  physiologically equivalent.
- Shimmer retains local %, local dB, APQ3 %, APQ5 %, APQ11 %, and DDA %.
  Maximum amplitude factor is 1.6; other pulse constraints match jitter.
- HNR remains mean valid-frame **cross-correlation Harmonicity in dB**, using
  60 Hz minimum pitch, 10 ms step, silence threshold 0.1, and 4.5 periods per
  window. NHR is never substituted.
- DFP remains sign reversals of adjacent period differences divided by
  `N−2`, ×100. Zero differences do not count as reversals. S5 recommendation
  is a lab extension beyond the source's sustained-vowel task support.
- `cpp_db` is **unsmoothed CPP**, not CPPS. The pinned Parselmouth 0.4.7
  embeds Praat 6.1.38. The path is Sound → PowerCepstrogram (60 Hz floor,
  2 ms step, 5 kHz maximum, 50 Hz pre-emphasis) → unsmoothed PowerCepstrum
  slices → peak prominence (60–500 Hz, parabolic interpolation, 1–50 ms
  straight trend, robust-slow fit), averaged across valid voiced-frame slices.
  `cpps_db` remains a separate unavailable smoothed variant.

## Explicit research replication choices

These are executable **research/source-replication** leaves. None has verified
source numerical parity. The source document supplies the concepts and the
parameters labeled “source” below. All other choices are VSLP replication
parameters, frozen for reproducibility but not asserted to be source values.

| Measure | Source-specified | VSLP replication choice | Remaining parity question |
| --- | --- | --- | --- |
| GNE | Hilbert-envelope cross-band correlation; centers near 500, 1500, 2500 Hz | 700 Hz bands; fourth-order Butterworth; zero-phase SOS filtering; Hilbert magnitude, demeaned without smoothing; maximum normalized correlation across distinct band pairs over ±1 ms lag | Original bandwidth/order, envelope treatment, lag and aggregation |
| PVI | F0 modulation 9–14 Hz; 1 s Welch window, 95% overlap | Fourth-order Butterworth zero-phase bandpass; interpolate at most 20 ms contiguous gaps and 10% missing frames; reject missing edges; sum square root of Welch spectrum power in band | Original filter order, missing-F0 handling and amplitude convention |
| H1–H8 | Eight harmonics, eight-period pitch-synchronous windows, 512 spectral samples; `RelH=1/(abs(mean)+SD)` | One-period hop; resample each eight-period window to 512 points; Hann window; linear peak amplitude divided by Hann coherent gain; sample SD across valid windows | Original amplitude scaling, normalization, window shape and window hop |

H1–H8 are 24 scalar leaves (`h1_mean` … `h8_relh`), retaining every harmonic
and every statistic. The three older broad vector IDs remain nonselectable
placeholders to avoid changing their meaning silently. Research output units
describe the VSLP scale, not a source-calibrated unit.

## Remaining validation

Independent Praat/reference comparisons are pending for jitter, shimmer, HNR,
CPP, and DFP. GNE, PVI, and H1–H8 require source-parameter recovery and
numerical parity testing before production interpretation. No F02 output has
been clinically validated by these software tests.
