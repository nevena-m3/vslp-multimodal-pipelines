# F04 — formant and vowel-space scientific review build

**Status:** IMPLEMENTATION FROZEN; REFERENCE VALIDATION PENDING; REAL-DATA VALIDATION PENDING; CLINICAL VALIDATION PENDING; SPECTRAL-ENVELOPE SOURCE PARITY PENDING. The S3 envelope remains an implemented VSLP source-lineage analogue. No F04 result is a diagnosis.

## Scientist sign-off checklist (current code)

The registry has **12 constructs, 27 concrete leaves**: 21 selectable recording outputs and six nonselectable token/vowel audit outputs. All 27 have implemented calculation paths. The S3 envelope leaf is **implemented but source parity pending**; independent reference validation and real-data checks are **NOT DONE for every leaf**. “Implemented” here describes code, not clinical validity.

| Construct and IDs | Scientific target / task | Input and region | Aggregation / unit | Design match and next action |
|---|---|---|---|---|
| F2: `f2_token_hz`, `f2_vowel_median_hz`, `f2_median_hz` | Vowel F2; S5, Bamboo, sustained /a/ | Final Alignment V1 central 50% for speech; reviewed stable midpoint for /a/ | Token median → trial median → recording median; Hz | MATCH in code; independent Praat check needed |
| F1: `f1_token_hz`, `f1_vowel_median_hz`, `f1_median_hz` | Vowel F1; S5, Bamboo | Final Alignment V1 central 50% | Token median → trial median → recording median; Hz | MATCH; reference check needed |
| F3: `f3_token_hz`, `f3_vowel_median_hz`, `f3_median_hz` | Vowel F3; WSTG “We see three geese.” | Final Alignment V1 central 50% | Token median → trial median → recording median; Hz | MATCH; F3/B3 track review needed |
| `vsa3_iau_hz2` | /a,i,u/ Bamboo corners | Shared trial centroids | Shoelace triangle area, then trial median; Hz² | MATCH; reference check needed |
| `fri_iau`, `sfri_iau` | Bamboo corner ratios | Shared trial centroids | Frozen ratios, then trial median; dimensionless | MATCH; reference check needed |
| `vai_iau`, `fcr_iau` | Bamboo articulation/centralization | Shared trial centroids | Frozen ratios, then trial median; dimensionless | MATCH; reference check needed |
| Eight `vowel_dispersion_*` IDs | Bamboo three-corner distances | Shared trial centroids | Pairwise/radial Euclidean distances, then trial median; Hz | MATCH as VSLP adaptation; reference check needed |
| `vowel_distance_ai_hz`, `_au_hz`, `_iu_hz` | S3 spot/AA, key/IY, blue/UW | Final Alignment V1 central 50%; exact parent word and phone | Target-token median within trial → distance → trial median; Hz | MATCH; target and reference check needed |
| `f2_distance_i_a_hz` | S3 key/IY vs spot/AA | Same S3 tokens | Absolute F2 difference per trial → trial median; Hz | MATCH; reference check needed |
| `vowel_envelope_distance_ai_source` | S3 spot/AA vs key/IY | Native audio, versioned midpoint pulse/LPC envelope | Median target envelope per trial → L1 distance → trial median; mean absolute dB | IMPLEMENTED, SOURCE PARITY PENDING; compare with source implementation before scientific sign-off |

The six audit IDs are the three `f1/f2/f3_token_hz` and three `f1/f2/f3_vowel_median_hz` outputs. Connected tokens retain recording/trial/sequence and final word/phone IDs, final boundary source, native audio sampling rate, and Alignment run identity. The `phone_index_in_trial` and `phone_index_in_word` fields come directly from V1; F04 does not read the legacy `phone_index` alias. Token exclusion reasons include `TOKEN_TOO_SHORT` and `INSUFFICIENT_VALID_FRAMES`; missing corners and targets produce explicit unavailable reasons. Reference validation remains NOT DONE for all rows above.

## Tasks and recommendations

| Construct | Recommended task |
|---|---|
| F2 | Sustained /a/, S5, Bamboo Passage |
| F1 | S5, Bamboo Passage; **not** sustained /a/ |
| VSA, FRI, SFRI, VAI, FCR, 3-corner dispersion | Bamboo Passage |
| F3 | WSTG/VC3 KEYS: “We see three geese” |
| Spectral-envelope distance, F1/F2 Euclidean distances, F2 convergence | S3: “The blue spot is on the key again” |

S5 is “We were away a year ago.” The existing Alignment registry retains the registered WSTG punctuation (“We see three geese.”); the F04 task identity is the same word sequence. Recommendation is independent of technical executability. No additional task selector is introduced.

**S5 F1/F2:** Static sentence-level formant summaries from every final usable aligned vowel token in each S5 trial. Take middle-half token medians, then the median across valid vowel tokens within trial, then the median across valid trials. This VSLP task-specific raw-formant implementation is consistent with the all-vowel-segment variant in the review literature. It is not a minimal-pair contrast; no word pair is imposed.

**WSTG F3:** Static sentence-level aligned-vowel F3 summary from every final usable vowel in “We see three geese.” Each repetition remains independent until its trial median exists; recording output is the median of valid trial medians. This is a **VSLP TASK ADAPTATION**, not Moore & Rong's historical /r/-/l/ minimal-pair F3 contrast. No F3 target pair is imposed.

**S3:** The pair/contrast constructs use only blue/UW, spot/AA, and key/IY. More than one final usable occurrence of any expected word+phone within a trial gives `AMBIGUOUS_CANONICAL_TARGET` for metrics requiring it; independent pairs remain available.

## One shared token layer

**Window containment:** A candidate Burg frame centre must lie in the aligned phone's middle 50%. Its complete 25-ms support must also remain inside that phone, its trial, native audio, and the exclusion-free domain. A rejected frame is never clipped or concatenated. Fewer than three remaining valid frames produces `INSUFFICIENT_VALID_FRAMES`.

Connected speech uses the **final frozen Alignment** word/phone boundaries and canonical native-rate waveform. Only aligned vowel nuclei are measured. Consonants and pauses are excluded logically by token selection; audio is never concatenated across artificial joins. Final reviewed segmentation and its hard manual exclusions remain authoritative. Token and frame audits retain original recording times, `trial_id`, task, word, original/stress-stripped phone label, canonical vowel class, F1/F2/F3, B1/B2/B3 where Praat returns positive finite bandwidths, counts, QC flags, and estimator provenance.

Praat/Parselmouth Burg production settings: **25-ms window, 5-ms step, five formants, 50-Hz pre-emphasis, 5500-Hz ceiling**. The optional sensitivity ceiling is 5000 Hz. Native 44.1- or 48-kHz audio is analyzed directly; a native rate at or below twice the selected ceiling fails explicitly. No sex inference, denoising, AGC, peak normalization, or forced 48-kHz conversion is applied. Parser and Praat versions are written to run provenance.

Connected-speech tokens under 50 ms fail QC. From each longer token, the middle 50% is measured. A token needs at least three valid frames with finite ordered F1<F2<F3 inside technical review bounds F1 150–1200 Hz, F2 500–3500 Hz, F3 1200–5000 Hz. Out-of-range frames are rejected and flagged, never clipped or converted to zero. Token F1/F2/F3 are medians of valid frames; token IQR and positive finite bandwidth medians are retained. Track jumps need independent review; there is no automatic interpolation or repair.

The frozen corner inventory is **/a,i,u/**: AA→/a/, IY→/i/, UW→/u/. The final Alignment's normalized label strips lexical stress, while the original phone label remains in the audit. AH, AE, IH, and UH are not silently mapped to these corners. The task registry supplies this versioned mapping in normal use; a legacy explicit mapping manifest remains compatible.

Generic S5 and Bamboo F1/F2 summarize **all** QC-passing vowel tokens: median token value within each trial. WSTG F3 similarly summarizes all valid aligned vowel nuclei, retaining each vowel identity and B3 audit. Sustained /a/ F2 uses a different domain: the **single temporal midpoint** of each reviewed centered stable ≤2-s /a/ trial, evaluated on the same Burg formant object. It is not a mean of a 2-s contour. Multiple attempts remain separate. For any scalar, recording value is the median of valid trial values; `formant_trial_summaries.csv` records the trial values.

## Bamboo /a,i,u/ centroids and formulas

Within each Bamboo trial, the authoritative centroid table takes the median token F1/F2 for each AA/IY/UW class. All derived measures below reuse this **same** table. One valid token per class allows calculation with `low_n_vowel_tokens`; two or more are preferred. A missing class yields NaN and `missing_required_vowel`. Trial values are aggregated by median across valid trials.

Let F1x and F2x be the centroid coordinates for vowel x. All coordinates are in Hz.

| Output | Frozen calculation |
|---|---|
| `vsa3_iau_hz2` | ½·abs[F1i(F2a−F2u)+F1a(F2u−F2i)+F1u(F2i−F2a)] |
| `fri_iau` | (F1a+F1i+F1u+F2a+F2i)/F2u |
| `sfri_iau` | (F2a+F2i)/F2u |
| `vai_iau` | (F2i+F1a)/(F1i+F1u+F2u+F2a) |
| `fcr_iau` | (F2u+F2a+F1i+F1u)/(F2i+F1a) |

Denominators must be positive. VSA has units Hz²; the ratios are dimensionless.

The eight `vowel_dispersion_*` leaves are explicitly a **VSLP three-corner adaptation**, not the historical ten-vowel source measure. `vowel_dispersion_ai_hz`, `_au_hz`, and `_iu_hz` are Euclidean F1/F2 pair distances. `vowel_dispersion_mean_3corner_hz` is their arithmetic mean. The global centroid is the unweighted mean of the three vowel centroids on each axis. `vowel_dispersion_a_to_global_hz`, `_i_to_global_hz`, and `_u_to_global_hz` are Euclidean distances to it; `vowel_dispersion_global_mean_hz` is their mean. No misleading front/back source labels are emitted.

## S3 lexical targets and envelopes

For S3, only **blue–UW (/u/), spot–AA (/a/), key–IY (/i/)** are approved targets. Other occurrences of the same phone in the sentence are excluded. `vowel_distance_ai_hz`, `_au_hz`, and `_iu_hz` are Euclidean F1/F2 distances between named target centroids. `f2_distance_i_a_hz` is abs(F2key−F2spot), not an F1/F2 Euclidean or signed difference. Repeated equivalent target tokens within a trial are summarized by median; missing targets yield NaN with an explicit reason.

The source documents specify an all-pole /a/-/i/ envelope and L1 distance, `mean_k |Ei(k)−Ea(k)|`, but do **not** recover the full pitch-synchronous frame, LPC order, grid, amplitude scale, and normalization settings. The executable `vowel_envelope_distance_ai_source` is therefore a **versioned VSLP source-lineage replication**, with source parity pending. Its explicit settings are: a frame centered on the Praat periodic-CC pulse nearest the token midpoint and spanning four local F0 periods, bounded below by 25 ms and above by token duration; Hamming window; 50-Hz first-order pre-emphasis; 14th-order autocorrelation LPC; 256 matched bins from 0 to 5000 Hz; dB all-pole magnitude centered to zero mean; median envelope across repeated valid target tokens; then mean absolute binwise difference. The required voiced F0 is obtained from Praat autocorrelation. The ID is historical; provenance states this is not independently source-equivalent. It is **not** substituted with F1/F2 Euclidean distance.

## GUI and outputs

Twelve historical F04 constructs appear in the feature catalog. **Twenty-one** recording-level leaves are selectable. Six previous token/named-vowel IDs remain as audit/intermediate identities but are hidden from the normal menu. Token, vowel-centroid, trial-summary, and compact frame-track artifacts remain available for review. The normal GUI shows task recommendations from the registry. Connected-speech outputs request final Alignment; the sustained /a/ midpoint scalar is exempt from linguistic Alignment.

F3/B3 and other credible bandwidth primitives are retained for a future resonance review. No hypernasality, nasal emission, or oral/nasal contrast composite is calculated in F04.

## Validation required

Engineering tests check task routing, hand-calculated formulas, QC behavior, trial separation, selected-only dispatch, and artifact provenance. They do not establish reference parity. Next scientific checks: inspect Burg tracks and bandwidths on consented recordings; compare to independent Praat analyses; validate S3 spectral-envelope settings against source code or reference outputs if obtainable; assess device/channel and sex/age sensitivity downstream; and evaluate low-token-count stability.
