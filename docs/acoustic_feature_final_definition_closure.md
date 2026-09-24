# Acoustic feature final definition closure

Scope: the current `september-review` checkout, the 13 detailed family DOCX specifications, the active registry, executors, mixed dispatcher, task registry and tests. This is an implementation and definition audit, not clinical validation. The row-level inventory is [acoustic_feature_implementation_inventory.csv](acoustic_feature_implementation_inventory.csv). No feature formula was changed.

## Counts and identity provenance

There are **79 scientific constructs**, **136 concrete exact registry leaves** (115 `IMPLEMENTED` and selectable; 21 `NOT_IMPLEMENTED` and nonselectable), and **19 `UNRESOLVED_TEMPLATE` identities**. The inventory has 155 rows because it includes both concrete leaves and templates. Two templates are retained as disabled registry output records; the other 17 are construct-level source templates. Family 03 has eight constructs and no unique public leaves.

| Family | Implemented | Exact not implemented | Unresolved templates |
|---|---:|---:|---:|
| 01 | 8 | 1 | 0 |
| 02 | 13 | 7 | 0 |
| 03 | 0 | 0 | 0 |
| 04 | 12 | 1 | 2 |
| 05 | 0 | 0 | 4 |
| 06 | 3 | 3 | 3 |
| 07 | 6 | 0 | 1 |
| 08 | 3 | 0 | 0 |
| 09 | 10 | 1 | 0 |
| 10 | 2 | 0 | 0 |
| 11 | 9 | 0 | 0 |
| 12 | 49 | 0 | 1 |
| 13 | 0 | 8 | 8 |
| **Total** | **115** | **21** | **19** |

The inventory now distinguishes `SOURCE_DEFINED_ID` (136 concrete IDs), `PROJECT_TEMPLATE` (19 unresolved identities from source templates), `PROJECT_DEFINED_ID` (zero active), `LEGACY_ID` (zero active), and `NO_PUBLIC_ID` (Family 03's eight constructs). The 19 template strings appear in detailed source `FEATURE ID(S)` fields, but their placeholders are **not concrete source-defined IDs**. Family 12's MFCC and spectral-contrast concrete IDs are deterministic expansions of source-defined coefficient/band ranges. Family 13's DOCX **does** contain `FEATURE ID(S)` fields in Word text boxes. Earlier paragraph/table extraction missed them. The Family 13 source metadata and closure documentation now state the correct origin. Family 03 repeats `blocked_proprietary` for all eight constructs; it is a source placeholder, not eight distinct public IDs.

## Family 03: final proprietary closure

All eight have `NOT_IMPLEMENTED` status, `NO_PUBLIC_ID`, no selectable leaf, recommendation **Sustained /a/**, evidence **LIMITED** (one study each). The repeated source identity is `blocked_proprietary`. The registry now carries the source evidence, entry counts and limitations without inventing a calculator.

| Construct | Source terminology / registry identity | Entries | Source unit and range | Exact block |
|---|---|---:|---|---|
| C013 | Glottal closure force; `blocked_proprietary` | 1 | Vendor index; no universal bound | No public estimator, glottal landmarks or scale. |
| C014 | Open/closed/approximation quotient; `blocked_proprietary` | 4 | % or ratio; 0–100% or 0–1 depending on scale | OQ/CQ concepts are given, but phase-boundary estimator and scale are undisclosed. |
| C015 | Mucosal wave correlate in opening phase; `blocked_proprietary` | 1 | Vendor index; no universal bound | Acoustic-to-mucosal-wave transform and scale undisclosed. |
| C016 | Proportion of Amplitude Tremor (Dysarthria Analyzer); `blocked_proprietary` | 1 | % or ratio, unconfirmed; no universal bound | Proprietary event rule, denominator and scale undisclosed. |
| C017 | Vocal fold edge separation; `blocked_proprietary` | 1 | Vendor index; no universal bound | No derivation of fold-edge separation/glottal gap from audio. |
| C018 | Vocal fold oedema correlate in opening phase; `blocked_proprietary` | 1 | Vendor index; no universal bound | Estimator, phase definition and scale undisclosed. |
| C019 | Vocal fold structural imbalance index; `blocked_proprietary` | 1 | Vendor index; no universal bound | Calculation and scale undisclosed. |
| C020 | Vocal fold vibratory instability and blockage index; `blocked_proprietary` | 3 | Vendor index; no universal bound | Within-cycle landmarks, aggregation and scale undisclosed. |

No generic glottal, Praat or open-source substitute is equivalent to these named source measures.

## Unresolved templates

Every template below remains nonselectable in this release. “Lab” means an approved, versioned scientific contract or task inventory is needed; no current authoritative task definition supplies it. Internal mathematical helpers, where present, do not constitute public implemented leaves.

| Family / template | Missing information and class | Existing authoritative task definition? | Lab input / release state |
|---|---|---|---|
| F04 `vowel_distance_<x>_<y>_hz` | Exact vowel pair, inventory, coordinate scale, aggregation and concrete ID; **linguistic inventory, task target**. | No | Lab must freeze; unresolved for this release. |
| F04 `vowel_dispersion_<variant>_hz` | Exact vowel inventory, dispersion variant/scale and ID; **linguistic inventory, parameter configuration**. | No | Lab must freeze; unresolved. |
| F05 `f1_slope_<target>_hz_s`, `f2_slope_<target>_hz_s` | Controlled target, task, token rule, transition window and concrete IDs; **task target, acoustic landmark**. | No | Lab must freeze target manifest; unresolved. Regression helper alone is not a public feature. |
| F05 `f1_range_<diphthong>_hz`, `f2_range_<diphthong>_hz` | Diphthong, onset/offset-glide exclusion, target rule and IDs; **task target, acoustic landmark**. | No | Lab must freeze; unresolved. |
| F06 `noise_duration_<target>_s` | Target plus validated acoustic noise onset/offset and ID; **task target, acoustic landmark**. | No | Lab annotations/definition required; unresolved. Phone duration is not a substitute. |
| F06 `noise_rise_time_<definition>_s` | Source-specific rise endpoints/threshold and ID; **source definition, acoustic landmark**. | No | Lab/source contract required; unresolved. |
| F06 `normalized_duration_contrast_<A>_<B>` | A/B controlled targets, pairing, task and ID; **task target, linguistic inventory**. | No | Lab must freeze; unresolved. Generic formula alone is not a leaf. |
| F07 `vowel_duration_<phone>_s` | Explicit target phone/inventory and concrete output identity; **linguistic inventory**. | No | Lab must freeze; unresolved. Existing aligned-vowel metrics remain distinct. |
| F12 `opensmile_1_4khz_<functional>` | Exact openSMILE functional, package/config parity and concrete ID; **source definition, parameter configuration**. | No | Source/config required; unresolved. |
| F13 `shannon_signal_entropy_<config>` | Signal/domain, discretization, bins, log base, aggregation; **parameter configuration**. | No | Lab must freeze; unresolved. |
| F13 `sample_entropy_<config>` | Sequence, embedding, tolerance/scaling, minimum length, aggregation; **parameter configuration**. | No | Lab must freeze; unresolved. |
| F13 `wpd_shannon_entropy_<wavelet>_L<level>` | Wavelet, level, nodes, boundary mode, energy/log/aggregation; **parameter configuration**. | No | Lab must freeze; unresolved. |
| F13 `psd_spectral_entropy_<config>` | PSD estimator, band, frame, probability/log convention, aggregation; **parameter configuration**. | No | Lab must freeze; unresolved. |
| F13 `wavelet_energy_<wavelet>_L<level>_<node>` | Wavelet, level, node, extension, energy normalization and aggregation; **parameter configuration**. | No | Lab must freeze; unresolved. |
| F13 `rqa_det_mfcc<k>_<config>` | MFCC component, embedding/delay, metric, recurrence rule, Theiler/minimum diagonal, aggregation; **parameter configuration**. | No | Lab/source parity contract required; unresolved. |
| F13 `rhythm_moddepth_<band>` | Envelope/modulation estimator, exact band, normalization and aggregation; **parameter configuration, source definition**. | No | Lab must freeze; unresolved. |
| F13 `rhythm_psi_<coupling>` | Exact bands/coupling, PSI definition and aggregation; **parameter configuration, source definition**. | No | Lab must freeze; unresolved. |

## Exact not-implemented leaves

All 21 are registered with exact code names and are **nonselectable**. Each has exactly one primary reason class, recorded row by row in the CSV.

| Reason class | Exact IDs |
|---|---|
| `PROPRIETARY_UNREPRODUCIBLE` | `blocked_aural_analytics_ap` |
| `FACTOR_TRANSFORM_NOT_FROZEN` | `intonation_factor_score_if_frozen`, `pause_pattern_factor_if_frozen`, `dynamics_factor_if_frozen`, `rhythm_factor_if_frozen`, `regularity_factor_if_frozen` |
| `SOURCE_PACKAGE_PARITY_UNAVAILABLE` | `vowel_envelope_distance_ai_source` |
| `SOURCE_DEFINITION_INCOMPLETE` | `gne_source_replication_only`, `pvi_9_14hz_source`, `cpp_db`, `cpps_db`, `harmonic_h1_h8_mean`, `harmonic_h1_h8_sd`, `relh_h1_h8`, `band_noise_contrast_variant1`, `band_noise_contrast_variant2`, `ppe_source_replication_only`, `dynamics_det_dmfcc_components`, `dynamics_articulation_rate`, `regularity_visibility_density`, `regularity_psi_components` |
| `OTHER_EXPLICIT_REASON` | None |

The detailed reason for each exact ID is in the CSV `reason_if_not_implemented_or_unresolved` field. Registry unavailability is intentional; an exact name alone does not supply the absent source transform, scale, landmark or estimator.

## Task and prompt registry

The authoritative task registry is `src/vslp/acoustic/alignment/task_prompts.json`. It contains no inferred filename prompts. Task type is explicit metadata.

| Task ID / type | Linguistic Alignment | Expected stimuli | Registered | Authoritative text and current workflow |
|---|---|---:|---:|---|
| `bamboo_passage` / Passage / connected speech | Yes | 1 canonical passage | 0 | Exact lab passage text, ID and version are missing. Alignment cannot run end to end until supplied. |
| `wstg` / Sentence / controlled speech | Yes | Total **unknown** | 1 | `wstg_we_see_three_geese` v1 = “We see three geese.”, supplied by user. This item can follow the GUI alignment workflow when recording, speaker, reviewed segmentation and MFA are available. Other WSTG item IDs/text/versions are unknown and cannot be guessed. |
| `ddk` / DDK | No | 0 linguistic prompts | 0 | Linguistic MFA Alignment is not required. DDK event segmentation/review is separate; exact task elicitation variants still require authoritative task metadata if future prompt-specific mapping is needed. |
| `sustained_a` / Sustained phonation | No | 0 linguistic prompts | 0 | Linguistic MFA Alignment is not required. The sustained /a/ target is task identity, not a word/phone prompt manifest. |

## Final checkpoints

**Family 10.** `ddk_rate_syll_s` and `ddk_cycle_mad_s` are both registered, selectable and mapped to the DDK mixed executor. The executor consumes final reviewed DDK events and emits `ddk_feature_events.csv`; it does not call MFA. Unit formula/event tests exist. The mixed integration suite covers a **missing DDK segmentation** result isolated from `pause_count`. It still lacks a successful full-stage mixed run using a frozen reviewed DDK fixture: **ENGINEERING TEST GAP**, not a scientific feature gap.

**Family 12 spectral contrast.** All seven `spectral_contrast_band0_db` through `spectral_contrast_band6_db` leaves are implemented, selectable and dispatched. The frozen working rate is **48,000 Hz** (private working signal); `n_fft=2048`, Hann `win_length=1200` samples (25 ms), `hop_length=480` (10 ms), `center=False`, `fmin=200 Hz`, `n_bands=6`, `quantile=0.02`, magnitude input, `linear=False` logarithmic contrast. Librosa's seven band edges at this rate are 0, 200, 400, 800, 1600, 3200, 6400 and 24000 Hz. Seven outputs with `n_bands=6` are valid. The spectral-contrast unit tests cover the configuration and synthetic behavior.

## Release classification and decisions

| Remaining item | Primary release category | Meaning |
|---|---|---|
| Implemented, selectable leaves with verified registry/dispatch definitions | `NO RELEASE BLOCKER` | Software definition closure for the implemented subset; still subject to scientific validation. |
| Bamboo canonical prompt and missing WSTG stimulus inventory | `AUTHORITATIVE TASK CONTENT REQUIRED` | Lab must supply exact text, IDs and versions; no online substitute. |
| 19 unresolved templates and 20 nonproprietary exact unavailable leaves | `SCIENTIFIC DEFINITION REQUIRED` | Freeze the specific source/target/transform contracts in the tables above. This count includes factor and source-parity cases. |
| Eight Family 03 constructs and Aural Analytics articulatory precision | `PROPRIETARY / UNREPRODUCIBLE` | Deliberately not selectable; no proxies. |
| Successful full-stage mixed DDK fixture | `ENGINEERING TEST GAP` | Add a reviewed-event success integration test before claiming that path is end-to-end regression covered. |
| Real source parity, manual boundary comparison, clinical/device robustness and ALS measurement validation | `EMPIRICAL VALIDATION REQUIRED` | Tests of formulas and synthetic behavior do not establish biomarker validity. |

The **scientific definition status is closed**: every known construct, concrete ID and template has an explicit defensible state. The **entire 79-construct battery is not fully implementable or release-ready** from current source information. The implemented subset is software definition-complete, with the task-content, engineering test and empirical validation work above still open. No speculative replacement is proposed.
