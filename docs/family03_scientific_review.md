# Family 03 scientific review build — open glottal and tremor measures

**Status:** engineering implementation for scientific review. No Voice Clinical Systems (VCS) Pr output is numerically reproduced, and no F03 output has independent source-package, real-data, or clinical validation. The four active parent constructs are historical concepts 13, 14, 16, and 20. Historical constructs 15, 17, 18, and 19 remain in the source matrix for provenance but are hidden from the active GUI and have no selectable leaves.

## Input and task contract

All runnable F03 leaves are recommended for **sustained /a/ only**. S5, S6, Bamboo, DDK, and other sentences are not recommended. F03 uses canonical native-rate audio, final reviewed segmentation, the explicit upstream /a/ task identity, and independently reviewed trials. It reuses the F01/F02 centered contiguous stable-phonation selection: up to 2.0 seconds from each reviewed /a/ attempt after hard manual exclusions. This is a VSLP protocol adaptation; some source studies use 3–4 seconds. The application does not add MFA, a task selector, denoising, AGC, or global amplitude normalization.

For each feature, frames/cycles are summarized **within each trial first**, then the recording result is the median of valid trial values. Native timing and trial IDs survive in `family03_trial_summaries.csv` and `family03_cycle_measurements.csv`. No cycle can cross a reviewed exclusion; the GIF calculator processes each eligible voiced island separately. Missing or invalid evidence produces NaN with a reason.

## Shared QCP-lineage glottal-source backbone

The primary implementation follows the weighted-linear-prediction and attenuated-main-excitation (AME) method in [Airaksinen, Story, and Alku (2013)](https://www.isca-archive.org/interspeech_2013/airaksinen13_interspeech.pdf). A Praat periodic-CC pulse track supplies *acoustic GCI candidates*. These are not the synthesis-known GCIs used in the published benchmark. Each eligible island is polyphase-resampled internally to 8 kHz; native audio is left unchanged. Model estimation uses 30-ms frames every 10 ms, prediction order 10, pre-emphasis zero 0.99, AME floor `10⁻⁵`, position quotient 0.05, seven-sample ramps, and pitch-adaptive duration quotient 0.95 below 190 Hz, 0.55 from 190 to below 280 Hz, and 0.70 at or above 280 Hz. Weighted least squares implements the paper's WLP normal equation; unstable all-pole roots are reflected inside the unit circle. The tract inverse is applied to un-emphasized speech pressure, overlap-added, integrated for lip-radiation compensation, and linearly detrended to estimate glottal flow. The method/version and 8-kHz internal rate are recorded.

**Limits:** this is a source-lineage engineering implementation, not demonstrated QCP parity. Acoustic GCI error, inverse-filter drift, and source-filter interaction can change cycle measurements. An IAIF comparator was not added in this build. Independent QCP/IAIF/OPENGLOT comparison and consented real-recording review are required before clinical interpretation. The recovered flow amplitude is uncalibrated; MFDR is a relative flow-rate proxy, never Newtons or an absolute volume-flow unit.

## Construct 13 — glottal closure / phonatory effort

| Aspect | Scientific review definition |
|---|---|
| Source concept | VCS Pr9 “glottal closure force”; its commercial transform is unavailable. |
| Open implementation | `glottal_mfdr = median_cycle(abs(min(dg/dt)))` and `glottal_naq = median_cycle(A_ac/(T0×MFDR))`, with `A_ac` the cycle peak-to-peak estimated flow. |
| Equivalence | **Not numerically equivalent to Pr9.** MFDR and NAQ are standard open-method primitives on a QCP-lineage flow estimate. NAQ is dimensionless; MFDR is uncalibrated. |
| Failure/QC | Invalid pulses, cycle duration, nonfinite flow, absent negative declination, and zero amplitude are excluded or reported. |

## Construct 14 — glottal cycle phase quotients

The current explicit **VSLP 5%-AC event convention** uses each pair of successive acoustic GCI candidates as a cycle. On the estimated glottal-flow cycle, opening is the first upward crossing of `minimum + 0.05×peak-to-peak amplitude`; maximum flow ends the opening phase and begins the closing phase; the next GCI ends closing. Initial GCI-to-opening time is the closed phase. Invalid event order yields NaN.

| Leaf | Calculation | Unit |
|---|---|---|
| `glottal_open_quotient` | opening-to-next-GCI duration / T0 | 0–1 proportion |
| `glottal_closed_quotient` | current-GCI-to-opening duration / T0 | 0–1 proportion |
| `glottal_closing_quotient` | maximum-flow-to-next-GCI duration / T0 | 0–1 proportion |
| `glottal_speed_quotient` | opening duration / closing duration | dimensionless |

The 5% landmark is an explicitly versioned operational convention that requires scientific/reference review; it is **not** a recovered VCS Pr4/Pr5/Pr6/Pr7 phase estimator. No QOQ leaf was added. Trial values are medians of valid cycle values, with cycle counts and individual values retained.

## Construct 16 — amplitude tremor

The source concept is amplitude tremor/PAT or dominant modulation-depth amplitude tremor (MDAT), **distinct from F02 PVI**, which analyzes F0 modulation at 9–14 Hz. The reusable calculator extracts a Hilbert amplitude envelope, resamples it to 50 Hz, detrends relative to local mean amplitude, applies a Gaussian window, and examines 1–20-Hz FFT peaks in 2-second windows stepped by 100 ms. Up to five candidates per window are linked across adjacent windows when frequencies differ by at most 1 Hz; the largest cumulative-depth qualifying track yields the median depth. The exact candidate-link tolerance and envelope/FFT normalization are VSLP choices pending independent Dysarthria Analyzer parity.

The active ≤2-second /a/ domain has **one window** and cannot form a multiwindow source track. Therefore `amplitude_tremor_mdat_pct` is visible as duration-blocked and nonselectable for the current protocol. Its calculator can return a source-lineage track value for separately approved longer input, but the current stage never substitutes a short estimate under this ID. The runnable `amplitude_tremor_depth_short_v1_pct` reports the strongest normalized 1–20-Hz modulation depth in the available 2-second window. It is a **VSLP operational analogue, not source-equivalent PAT/MDAT**. Shorter or unstable regions return NaN; no synthetic 3-second track is fabricated. Audit records modulation rate, windows, peak frequency, track duration/fraction, mode, and source-equivalence flag.

## Construct 20 — vibratory instability / blockage

These are three explicit VSLP operational analogues, **not numerically equivalent to proprietary VCS Pr13/Pr14/Pr15**.

| Leaf | Calculation | Distinction |
|---|---|---|
| `cycle_waveform_instability` | Median `1−Pearson r` for adjacent time-normalized, centered, amplitude-normalized inverse-filtered flow cycles within one contiguous voiced run | Shape similarity, not acoustic jitter |
| `cycle_amplitude_instability_pct` | `100×1.4826×MAD(Aᵢ)/median(Aᵢ)` for positive finite cycle peak-to-peak flow amplitudes across a trial | Longer-scale robust source-amplitude variation, not shimmer |
| `vibration_blockage_pct` | `100×interior periodic-loss duration / eligible stable-phonation duration`, counting only contiguous loss lasting at least three expected cycles | Operational interruption, not a VCS blockage score |

The blockage detector uses the shared F0/GIF pulse-cycle evidence. It excludes onset/offset and manual exclusions. A low-amplitude dropout relative to the trial RMS or digital clipping causes `acquisition_dropout_requires_review` rather than a biological blockage result. Other unrecognized acquisition defects remain possible; the number must not be read as a diagnosis without listening and inspection.

## Provenance and unresolved scientific questions

Every F03 result retains feature/task/trial identity, native sample rate, reviewed boundaries, stage versions, GIF method and internal rate, aggregation, source method class, non-equivalence flag, QC/failure reason, and reviewed-segmentation hashes. Historical VCS source names are provenance only. The implementation is **engineering tested**, with independent QCP/VoiceSauce/Dysarthria Analyzer parity, OPENGLOT reference testing, real VSLP recording review, and ALS-specific validity all pending.

Scientific review should decide whether the 5%-AC opening landmark and acoustic GCI candidates are adequate; whether the ≤2-second protocol can support any clinical amplitude-tremor interpretation; whether MFDR should remain an uncalibrated proxy; and whether the operational blockage rule can separate phonatory breaks from all relevant recording artifacts. Those decisions do not alter the proprietary non-equivalence statement.
