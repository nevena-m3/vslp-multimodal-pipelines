# Family 06 — segmental articulatory and phonetic contrasts

**Build status:** implementation tested with synthetic, nonclinical fixtures. Independent source parity, real-data landmark validation, and clinical validation are pending. An automatic detector score is signal prominence, not a confidence probability or evidence that a dysarthric boundary is accurate.

## Source and task contract

The family retains eight historical constructs (C035–C042). C035 remains unavailable because the proprietary normative likelihood-ratio model is unavailable; no GOP surrogate is substituted. C040 is historical and inactive. C041 0–10 kHz wideband noise energy is historical and inactive because its task is not frozen. The active constructs are C036, C037, C038, C039, and C042, with 16 selectable outputs in `data/family06.json`.

| Construct | Frozen task and targets | Method | Interpretation |
|---|---|---|---|
| C036 bandpassed noise energy | WSTG: see/S, geese/S, three/TH | Native-rate one-sided Hann PSD integrated over bands; A = E(720–10000) − E(4200–10000), B = E(720–6500) − E(720–2150) | Task-specific source-lineage analogue; /S/ and /TH/ measured separately. Requires Nyquist >10 kHz for A, >6.5 kHz for B. |
| C037 first spectral moment difference | S3: spot/T minus key/K | Burst onset; 20-ms native window; common 500–5000 Hz power centroid ΣfP/ΣP; signed T−K | Source low-level numerical parity pending. |
| C038 fricative noise duration | WSTG: see/S, geese/S, three/TH | Accepted aperiodic-noise interval duration in ms; /S/ median of see and geese, /TH/ separate, optional /S/−/TH/ | VSLP task adaptation; no affricate duration substituted. |
| C039 noise-energy rise time | S7: judge/JH, first sentenced/T | Native high-band power envelope, five-frame smoothing, local first-fifth median baseline, 95th-percentile peak, first persistent 10% to 90% crossing; ms | `method_class=VSLP_OPERATIONAL_ANALOGUE`; `source_numerical_parity=pending`. The historical source used a manually annotated stop-versus-affricate contrast; VSLP uses an automatic 10%–90% rise. WSTG exploratory branch is not public. |
| C042 stop-burst spectral tilt | S9: keep/K minus at/T | Burst onset; 10-ms Hann native window; 1500–5000 Hz log-amplitude OLS slope in dB/kHz; signed K−T | VSLP task-specific contrast. S2 is technically extensible but no public S2 C042 output is frozen. |

The registered scientific texts are WSTG “We see three geese.”, S3 “The blue spot is on the key again.”, S7 “The municipal judge sentenced the criminal.”, and S9 “Peter will keep at the peak.” S2 is “Buy Bobby a puppy.” and is **not** a substitute for S9. S4 is not an active public F06 task. Recommendation metadata is independent of technical executability.

The frozen `english_us_arpa` dictionary gives `sentenced` as `S EH1 N T AH0 N S T`. The canonical first /T/ is at **1-based** position 4; the second is at position 8. Final Alignment V1 `phone_index_in_word` enumerates aligned provider phones in 1-based order, so C039 requires both position 4 and the observed `S EH N T` prefix. A missing earlier phone cannot silently shift the final /T/ into the target position. The medial stop is selected over the complex word-final cluster. If that prefix is absent, the target is unavailable. This operational rule does not establish source-equivalent numerical results.

The 16 active IDs are `band_noise_a_see_s`, `band_noise_a_geese_s`, `band_noise_a_three_th`, `band_noise_b_see_s`, `band_noise_b_geese_s`, `band_noise_b_three_th`, `band_noise_a_s_minus_th`, `band_noise_b_s_minus_th`, `spectral_m1_t_minus_k_hz`, `fricative_noise_duration_s_ms`, `fricative_noise_duration_th_ms`, `fricative_noise_duration_s_minus_th_ms`, `noise_rise_time_judge_jh_ms`, `noise_rise_time_sentenced_t_ms`, `noise_rise_time_t_minus_jh_ms`, and `spectral_tilt_k_minus_t_db_per_khz`. The three historical nonselectable records are `blocked_aural_analytics_ap`, `normalized_temporal_contrast_historical`, and `wideband_noise_energy_0_10khz`.

## Dependency and provenance

The production executor uses only `load_final_alignment()` and its feature-facing `words_for_trial` and `phones_for_trial` accessors for exact canonical word, parent word, phone, trial, sequence, and original-time boundaries. It never reads TextGrid, provider tokens, or MFA working audio and never invokes MFA. Acoustic measurement reads the reviewed canonical/native waveform at its native sample rate. A selected task with no current frozen Alignment yields an unavailable leaf with a reason; unrelated feature families can still run.

The `family06_subevents.py` detector version `f06-native-subevents-v1` emits three kinds: `STOP_BURST`, `APERIODIC_NOISE_INTERVAL`, and `NOISE_RISE_INTERVAL`. The digital power and threshold rules are versioned in that module. The detector finds candidates; it does not certify a clinically accurate landmark. An approved manual annotation in original-recording seconds, keyed to recording, trial, phone and event kind, overrides the automatic event. Invalid or ambiguous manual annotations fail explicitly. Hard exclusions remain forbidden. Token, trial and subevent audit tables retain trial, sequence, word, phone, original-time event and measurement bounds, detector/source, sample rate, and QC reason.

Each target token is measured independently. Equivalent repeated tokens within a trial are median-combined, then trial-level results are median-combined across valid trials. Signed differences are formed only between target-specific trial summaries; trajectories and phone classes are never concatenated. Failures are NaN with machine-readable reasons, never zero-filled. The canonical final Alignment is a prerequisite, not an acoustic burst/noise detector.

## Source evidence and limits

C035 has strong parent-construct evidence but is blocked by the proprietary source model. C036–C039 have limited parent-construct evidence; C042 is not established in the master matrix. The VSLP signal-processing and target choices above are explicit reproducible adaptations. No independent numerical parity with the historical sources has been demonstrated. Synthetic tests check implementation invariants, not ALS/MND landmark accuracy. Real remote dysarthric recordings require independent human landmark comparison before scientific or clinical claims.
