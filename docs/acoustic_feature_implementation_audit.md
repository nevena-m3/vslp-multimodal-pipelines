# Acoustic Features implementation audit

Snapshot: current local `september-review` checkout, 2026-09-23. The checkout
already contained uncommitted work. This audit changes documentation only. It
compares the 79-construct master matrix, all 13 detailed family DOCX files,
the loaded catalog and family JSON files, calculator ID sets, stage executors,
mixed dispatch, GUI, and tests. **IMPLEMENTED means an approved exact ID is
selectable and has a registered calculator and dispatcher path. It does not
mean the feature succeeds on every recording or has clinical validation.**

The machine-readable, one-row-per-ID/template inventory is
[`acoustic_feature_implementation_inventory.csv`](acoustic_feature_implementation_inventory.csv).
It contains the exact public IDs, source-only IDs/templates, scientific metadata,
upstream dependencies, executor groups, and reasons for unavailable outputs.

## Executive summary and counting rule

| Scope | Count |
|---|---:|
| Scientific constructs in master matrix | 79 |
| Entries loaded in active registry | 130 |
| Concrete exact IDs loaded in active registry | 128 |
| Unresolved template placeholders loaded in active registry | 2 |
| Source-defined IDs/templates absent from active registry | 25 |
| Reconciled inventory rows | 155 |
| IMPLEMENTED rows | 115 |
| NOT_IMPLEMENTED rows | 21 |
| UNRESOLVED_TEMPLATE rows | 19 |

The 155-row total is **source-reconciled**, not a claim that 155 public features
are selectable: only 115 are selectable. Of the 25 source-only rows, nine are
Family 04/05/06 target templates and 16 are Family 13 IDs/templates. Family 03
has eight blocked constructs but its document repeats the generic
`blocked_proprietary` placeholder for each; it provides no unique public leaf
ID, so those eight placeholders are not counted as leaves. The registry's 130
entries include two disabled templates, `vowel_duration_<phone>_s` and
`opensmile_1_4khz_<functional>`.

Constructs and outputs differ because jitter, shimmer, MFCC, spectral contrast,
and other constructs expand into several distinct IDs. Targeted constructs can
remain templates; blocked constructs can have no reproducible public output.

## Family-by-family inventory

The exact ID list for every family appears below and in the CSV. Counts use the
source-reconciled inventory; a dash means no source-defined unique public leaf.

| Family | Name | Constructs | IMPLEMENTED / NOT_IMPLEMENTED / UNRESOLVED_TEMPLATE | Executor / mixed group | Current upstream dependency | Profile and audit artifact | Coverage and principal limitation |
|---|---|---:|---:|---|---|---|---|
| F01 | F0 and pitch variability | 4 | 8 / 1 / 0 | Yes / `voice` | Frozen review; stable region where required | `family01_f0_praat_ac_v1`; `f0_tracks.csv` | Unit + reviewed integration; factor transform missing; intonation variants lack individual numeric reference assertions. |
| F02 | Voice quality, perturbation and noise | 8 | 13 / 7 / 0 | Yes / `voice` | Frozen review; stable phonation/pulses | `family02_stable_vowel_praat_v1`; `voice_pulses.csv` | Unit + reviewed integration; seven source-parity outputs lack frozen parameters; several perturbation variants lack individual hand-calculation tests. |
| F03 | Laryngeal biomechanical measures | 8 | 0 / 0 / 0 | No | Not executable | None | Eight proprietary constructs; source repeats `blocked_proprietary`, not eight distinct IDs. |
| F04 | Formant and vowel-space measures | 12 | 12 / 1 / 2 | Yes / `formants` | Frozen Alignment; vowel mapping for named centroids | `family04_burg_native_5500_v1` (5000 sensitivity profile); token/vowel CSVs and frame NPZ | Unit + aligned integration + mixed tests; two target templates and source envelope distance unresolved. |
| F05 | Formant dynamics | 2 | 0 / 0 / 4 | No public executor mapping | Frozen Alignment and approved formant targets | Internal `family05_targeted_formant_v1`; reads shared F04 frame NPZ | Internal slope/range unit tests only; no approved concrete target/ID or public dispatch. |
| F06 | Segmental articulatory / phonetic contrasts | 8 | 3 / 3 / 3 | Yes / `segmental` | Frozen Alignment **and** validated acoustic sub-events | `family06_native_subevents_v1`; `family06_segmental_targets.csv` | Unit + mixed integration for M1; proprietary precision, incomplete contrasts, and target templates remain unavailable. |
| F07 | Segment duration and temporal variability | 5 | 6 / 0 / 1 | Yes / `timing` | Reviewed timing for two IDs; frozen Alignment for four IDs | `family07_reviewed_timing_v1`, `family07_alignment_tokens_v1`; `family07_duration_events.csv` | Unit + aligned/reviewed integration; phone-specific duration ID awaits frozen phone inventory. |
| F08 | Speech and articulation rate | 2 | 3 / 0 / 0 | Yes / `timing` | Reviewed speech/pause timing plus versioned prompt counts | `family08_bamboo_reviewed_v1`; shared timing/prompt audit | Hand-calculated unit tests; prompt manifest missing affects only these three outputs. |
| F09 | Pause and phrasing | 8 | 10 / 1 / 0 | Yes / `timing` | Reviewed speech/pause timing | `family09_bamboo_reviewed_v1`; `pause_phrase_events.csv` | Hand-calculated unit tests and timing integration; factor loadings not frozen. |
| F10 | DDK performance | 2 | 2 / 0 / 0 | Yes / `ddk` | Frozen reviewed DDK events | `family10_ddk_reviewed_v1`; `ddk_feature_events.csv` | Hand-calculated event/unit tests and mixed missing-input isolation; no successful Family 10 full-stage mixed integration fixture. |
| F11 | Intensity and amplitude | 5 | 9 / 0 / 0 | Yes / `spectrum` | Frozen review supplies analysis region/audio; normalization provenance | `family11_amplitude_native_v1`; waveform/spectral audit CSV | Unit + integration; digital scale and channel/device sensitivity require empirical validation. |
| F12 | Spectral and cepstral descriptors | 7 | 49 / 0 / 1 | Yes / `spectrum` | Frozen review supplies region/audio; private 48-kHz representation | `family12_mfcc_48k_v1`, `family12_spectral_48k_v1`; MFCC NPZ and spectral audit CSV | Unit + integration; openSMILE functional unresolved; many coefficients/bands lack individual numeric references. |
| F13 | Nonlinear complexity, rhythm and regularity composites | 8 | 0 / 8 / 8 | No | Not yet established in active executor | No public profile or audit artifact | Detailed source document exists, but family is not loaded/implemented; several transforms and factor definitions require freezing. |

### Exact ID appendix

The CSV is authoritative for one ID per row. Source-only IDs in F13 and
templates in F04/05/06 are **not** active public GUI leaves. Internal Family 05
slope/range helpers and Family 06 duration/contrast helpers are not counted as
implemented public outputs. The appendix below is generated from the CSV to
avoid shortening or aliasing any code name.

**F01.**
- IMPLEMENTED: `f0_iqr_st`, `f0_mean_hz`, `f0_median_hz`, `f0_range_st`, `f0_sd_st`, `intonation_f0_iqr_st`, `intonation_f0_sd_st`, `pfr_maxmin_st`.
- NOT_IMPLEMENTED: `intonation_factor_score_if_frozen`.

**F02.**
- IMPLEMENTED: `dfp_pct`, `hnr_mean_db`, `jitter_absolute_s`, `jitter_ddp_pct`, `jitter_local_pct`, `jitter_ppq5_pct`, `jitter_rap_pct`, `shimmer_apq11_pct`, `shimmer_apq3_pct`, `shimmer_apq5_pct`, `shimmer_dda_pct`, `shimmer_local_db`, `shimmer_local_pct`.
- NOT_IMPLEMENTED: `cpp_db`, `cpps_db`, `gne_source_replication_only`, `harmonic_h1_h8_mean`, `harmonic_h1_h8_sd`, `pvi_9_14hz_source`, `relh_h1_h8`.

**F03.**
No unique public source IDs. Eight F03 constructs use the repeated `blocked_proprietary` placeholder.

**F04.**
- IMPLEMENTED: `f1_token_hz`, `f1_vowel_median_hz`, `f2_distance_i_a_hz`, `f2_token_hz`, `f2_vowel_median_hz`, `f3_token_hz`, `f3_vowel_median_hz`, `fcr_iau`, `fri_iau`, `sfri_iau`, `vai_iau`, `vsa3_iau_hz2`.
- NOT_IMPLEMENTED: `vowel_envelope_distance_ai_source`.
- UNRESOLVED_TEMPLATE: `vowel_dispersion_<variant>_hz`, `vowel_distance_<x>_<y>_hz`.

**F05.**
- UNRESOLVED_TEMPLATE: `f1_range_<diphthong>_hz`, `f1_slope_<target>_hz_s`, `f2_range_<diphthong>_hz`, `f2_slope_<target>_hz_s`.

**F06.**
- IMPLEMENTED: `m1_t_minus_k_hz`, `stop_burst_spectral_tilt_db_khz`, `wideband_noise_energy_0_10khz`.
- NOT_IMPLEMENTED: `band_noise_contrast_variant1`, `band_noise_contrast_variant2`, `blocked_aural_analytics_ap`.
- UNRESOLVED_TEMPLATE: `noise_duration_<target>_s`, `noise_rise_time_<definition>_s`, `normalized_duration_contrast_<A>_<B>`.

**F07.**
- IMPLEMENTED: `delta_v_s`, `mean_word_duration_s`, `npvi_v_pct`, `speech_time_s`, `task_elapsed_duration_s`, `varco_v_pct`.
- UNRESOLVED_TEMPLATE: `vowel_duration_<phone>_s`.

**F08.**
- IMPLEMENTED: `articulation_rate_syll_s`, `speaking_rate_syll_s`, `speaking_rate_words_min`.

**F09.**
- IMPLEMENTED: `cv_pause_duration`, `cv_pause_duration_pct`, `cv_phrase_duration`, `cv_phrase_duration_pct`, `mean_pause_duration_s`, `mean_phrase_duration_s`, `pause_count`, `pause_pattern_components`, `percent_pause_ge300ms`, `total_pause_duration_s`.
- NOT_IMPLEMENTED: `pause_pattern_factor_if_frozen`.

**F10.**
- IMPLEMENTED: `ddk_cycle_mad_s`, `ddk_rate_syll_s`.

**F11.**
- IMPLEMENTED: `absolute_energy_fs2`, `amp_max_fs`, `amp_mean_fs`, `amp_min_fs`, `amp_sd_fs`, `sound_power_digital`, `visibility_graph_density_amp`, `wave_amp_excess_kurtosis`, `wave_amp_skew`.

**F12.**
- IMPLEMENTED: `mfcc01_mean`, `mfcc01_sd`, `mfcc02_mean`, `mfcc02_sd`, `mfcc03_mean`, `mfcc03_sd`, `mfcc04_mean`, `mfcc04_sd`, `mfcc05_mean`, `mfcc05_sd`, `mfcc06_mean`, `mfcc06_sd`, `mfcc07_mean`, `mfcc07_sd`, `mfcc08_mean`, `mfcc08_sd`, `mfcc09_mean`, `mfcc09_sd`, `mfcc10_mean`, `mfcc10_sd`, `mfcc11_mean`, `mfcc11_sd`, `mfcc12_mean`, `mfcc12_sd`, `mfcc13_mean`, `mfcc13_sd`, `spec_m1_hz`, `spec_m2_hz`, `spec_m3_skew`, `spec_m4_kurtosis`, `spectral_bandwidth_p2_mean_hz`, `spectral_bandwidth_p2_median_hz`, `spectral_centroid_hz`, `spectral_contrast_band0_db`, `spectral_contrast_band1_db`, `spectral_contrast_band2_db`, `spectral_contrast_band3_db`, `spectral_contrast_band4_db`, `spectral_contrast_band5_db`, `spectral_contrast_band6_db`, `spectral_crest`, `spectral_flatness`, `spectral_flux`, `spectral_kurtosis`, `spectral_skew`, `spectral_slope_db_khz`, `spectral_spread_hz`, `zcr_mean_fraction`, `zcr_median_fraction`.
- UNRESOLVED_TEMPLATE: `opensmile_1_4khz_<functional>`.

**F13.**
- NOT_IMPLEMENTED: `dynamics_articulation_rate`, `dynamics_det_dmfcc_components`, `dynamics_factor_if_frozen`, `ppe_source_replication_only`, `regularity_factor_if_frozen`, `regularity_psi_components`, `regularity_visibility_density`, `rhythm_factor_if_frozen`.
- UNRESOLVED_TEMPLATE: `psd_spectral_entropy_<config>`, `rhythm_moddepth_<band>`, `rhythm_psi_<coupling>`, `rqa_det_mfcc<k>_<config>`, `sample_entropy_<config>`, `shannon_signal_entropy_<config>`, `wavelet_energy_<wavelet>_L<level>_<node>`, `wpd_shannon_entropy_<wavelet>_L<level>`.


## Dependency matrix from current execution paths

All 115 implemented IDs currently pass through a **stage-level frozen reviewed
segmentation gate**, even waveform-only Families 11/12. Thus there is no
implemented segmentation-free Acoustic Features path today. The boundary column
in the CSV records the most specific dependency, not the entire chain.

| Most specific source | Implemented IDs | Additional dependencies |
|---|---:|---|
| Reviewed segmentation/audio | 94 | F08's three rate IDs also require versioned prompt counts; several voice outputs need a stable phonation region. |
| Frozen linguistic Alignment | 16 | F04's nine named-vowel/derived IDs also require a vowel-category manifest. F07 is hybrid: `mean_word_duration_s`, `delta_v_s`, `varco_v_pct`, `npvi_v_pct` require Alignment; `speech_time_s`, `task_elapsed_duration_s` use reviewed timing. |
| Frozen reviewed DDK events | 2 | `ddk_rate_syll_s`, `ddk_cycle_mad_s`; no detector is rerun in Features. |
| Validated acoustic sub-event annotations | 3 | F06 also requires frozen Alignment and an approved segmental target manifest. |
| No segmentation/Alignment prerequisite at the current stage entry point | 0 | The universal review gate is an architectural constraint, including for waveform-only outputs. |

No feature Python module invokes MFA CLI or parses TextGrid. The search found
MFA wording only in the historical `master_matrix.json` estimator descriptions.
F04/F06/F07 consume the frozen Alignment API; MFA, if used, belongs in the
Alignment provider layer.

## Checkpoints

**Family 10.** Both `ddk_rate_syll_s` and `ddk_cycle_mad_s` are registered,
selectable, implemented in `family10.py`, and mapped once to `ddk` by mixed
dispatch. The stage requires frozen final DDK decisions/intervals and writes
`ddk_feature_events.csv`. Unit tests cover counts, interval MAD, exclusions,
sequence breaks, and NaN reasons. Mixed integration tests currently verify
that an absent DDK prerequisite does not block pause features; a successful
frozen-DDK mixed stage integration case remains a validation gap.

**Family 12 spectral contrast.** All seven concrete IDs,
`spectral_contrast_band0_db` through `spectral_contrast_band6_db`, are
IMPLEMENTED and mapped to `spectrum`. The frozen private working sample rate is
48,000 Hz, with `n_fft=2048`, 1,200-sample (25-ms) Hann window, `hop=480`
(10 ms), `center=False`, `fmin=200 Hz`, `n_bands=6`, `quantile=0.02`,
`linear=False`, STFT magnitude input, and librosa 0.11.0's
`power_to_db(peak) - power_to_db(valley)` convention. The effective band edges
are 0, 200, 400, 800, 1,600, 3,200, 6,400, and 24,000 Hz; the last band
extends to Nyquist. Seven outputs result from **six** octave boundaries, so
the earlier claim that this configuration is invalid at 48 kHz was incorrect.
The existing unit test checks seven IDs, these edges, synthetic tonal/noise
direction, and minimum-frame failure. Individual band reference values remain
untested against a frozen external calculation.

**Target templates.** F04 needs a frozen vowel pair, inventory, coordinate
scale, dispersion variant, and concrete IDs for
`vowel_distance_<x>_<y>_hz` and `vowel_dispersion_<variant>_hz`.
F05 needs an approved task/word/phone/transition target and IDs for
`f1_slope_<target>_hz_s`, `f2_slope_<target>_hz_s`, plus a named diphthong,
glide-exclusion convention, and IDs for `f1_range_<diphthong>_hz`,
`f2_range_<diphthong>_hz`. F06 needs target-specific noise onset/offset and
ID for `noise_duration_<target>_s`, source rise endpoints and ID for
`noise_rise_time_<definition>_s`, and approved A/B task/pairing/ID for
`normalized_duration_contrast_<A>_<B>`. Internal calculators exist for F05
slope/range and F06 noise duration/normalized contrast; no concrete public
leaf is instantiated by those helpers.

## Dispatch and GUI consistency

The mixed dispatcher maps each of the 115 implemented IDs to exactly one of
`voice`, `formants`, `segmental`, `timing`, `ddk`, or `spectrum`. Audit result:
**zero unmapped IDs, zero multiply mapped IDs, zero duplicate registry IDs**.
Calculator ID sets matched registered implemented IDs for F01, F02, F04, F06,
F07, F08, F10, F11, and F12. F09's calculator constant also names the
nonselectable `pause_pattern_factor_if_frozen`; the catalog blocks its execution.
The dispatcher rejects an unselected emitted row and synthesizes an explicit
failure if a selected executor omits a result; mixed tests cover those contracts.

An offscreen GUI audit checked all 130 registered entries: Code name,
Availability, Evidence, Recommended tasks, Unit, Range, Analysis unit, and checkbox state matched
the catalog with zero mismatches. Task indicator mappings matched the catalog
for all four registered exact tasks (520 checks). Implemented entries are
selectable; unavailable entries are not. Two unresolved templates, Family 07's
`vowel_duration_<phone>_s` and Family 12's
`opensmile_1_4khz_<functional>`, appear as disabled **leaf rows** rather than
solely as construct metadata. They cannot run and create no invented concrete
ID, but this is a presentation inconsistency to consider later.

## Provenance and test gaps

Native audit artifacts exist for every implemented executor group: F0 tracks,
voice pulses, F04 formant token/vowel tables and frame NPZ, F06 target/sub-event
CSV, F07 duration events, F09 pause/phrase events, F10 DDK events, F11/12
waveform/spectral audit CSV, and MFCC matrices. F08 shares timing and prompt
provenance. No implemented family lacks an audit artifact, though F11/12 do not
persist every spectral frame/bin and F06 does not duplicate full spectra.

Unit files: `test_family01_f0.py`, `test_family02_voice_quality.py`,
`test_family04_formants.py`, `test_family06_segmental.py`,
`test_family07_duration.py`, `test_family08_rate.py`, `test_family09_pause.py`,
`test_family10_ddk.py`, `test_family11_amplitude.py`, and
`test_family12_spectral.py`. F05 has internal-only tests in
`test_family05_formant_dynamics.py`. Integration coverage is in
`test_family01_reviewed_f0.py`, `test_family04_aligned_formants.py`,
`test_family07_reviewed_duration.py`, `test_family11_12_reviewed_audio.py`,
and `test_mixed_acoustic_features.py`. Mixed tests cover successful F01+F09,
F04+F09, F11+F12, F06 M1+F09, and failure isolation; successful F10+other-family
execution is not directly integration-tested.

Implemented leaves without an **individual frozen numeric reference** include
F01's `intonation_f0_sd_st` and `intonation_f0_iqr_st`; F02's RAP/PPQ5/DDP
jitter and APQ3/APQ5/APQ11/DDA shimmer variants; F06's absolute wideband
energy and burst-tilt values; F11's visibility-graph density; and F12's 26
individual MFCC aggregates, seven individual contrast bands, and most spectral
shape descriptors. Family-level synthetic directions, IDs, or output shapes are
tested for many of these. Those tests do not establish source parity or clinical
validity. Live MFA provider validation, validated segmental target annotations,
cross-device amplitude robustness, and VSLP/manual-reference comparisons also
remain outstanding.

## Remaining work classified once

The 40 unavailable inventory rows are classified by their **primary** blocker:

| Classification | Items |
|---|---|
| CODE NOT YET IMPLEMENTED | Four source-only F13 exact IDs: `dynamics_det_dmfcc_components`, `dynamics_articulation_rate`, `regularity_visibility_density`, `regularity_psi_components`. This classification does not approve them for production without Family 13 review. |
| SCIENTIFIC DEFINITION NOT FROZEN | The other 35 unavailable inventory rows: 19 templates; F01 factor (1); F02 GNE/PVI/CPP/CPPS/harmonic outputs (7); F04 envelope distance (1); F06 band-noise variants (2); F09 factor (1); F13 PPE and three factor IDs (4). |
| PROPRIETARY / UNREPRODUCIBLE | `blocked_aural_analytics_ap` (1 public ID) plus eight F03 proprietary constructs that have no unique public ID. |
| EXTERNAL INFRASTRUCTURE NOT VALIDATED | No additional leaf is classified here; this is a cross-cutting validation dependency: live MFA/model/dictionary operation and approved external target/sub-event resources have not been demonstrated on local reference data. |
| IMPLEMENTED BUT NEEDS EMPIRICAL VALIDATION | All 115 implemented leaves. Code and synthetic tests are not VSLP, clinical, device, or source-package parity validation. |

The source-only F13 rows are included to show what the document names; they are
not registered or executable. The current feature stage's universal frozen
review gate is an additional engineering constraint to revisit if truly
segmentation-independent waveform extraction is required.

**Recommended next step:** review/approve this inventory, then ingest Family 13
with explicit decisions for each source-defined transform, parameterized ID,
and factor loading. Separately add a successful frozen-DDK mixed integration
fixture and numeric reference fixtures for the high-dimensional outputs before
claiming broader validation.

## Validation

Validation command results are recorded in the task response for this audit.
No feature algorithm, registry, dispatcher, or GUI behavior was changed.
