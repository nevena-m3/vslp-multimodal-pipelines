# F05 — Formant dynamics implementation review

**Status:** Implemented; independent numerical/source parity and real-data validation pending. Neither construct is clinically validated.

## Frozen outputs and tasks

| Construct | Task | Canonical target | Public output IDs | Evidence/lineage |
| --- | --- | --- | --- | --- |
| C033 signed transition slope | S2, “Buy Bobby a puppy.” | buy /AY/ (/aɪ/) | `f1_transition_slope_buy_ay_hz_s`, `f2_transition_slope_buy_ay_hz_s` | Strong parent-construct evidence; source-lineage target |
| C033 signed transition slope | S5, “We were away a year ago” | away /EY/ (/eɪ/) | `f1_transition_slope_away_ey_hz_s`, `f2_transition_slope_away_ey_hz_s` | VSLP task adaptation |
| C033 signed transition slope | S5, “We were away a year ago” | ago /OW/ (/oʊ/) | `f1_transition_slope_ago_ow_hz_s`, `f2_transition_slope_ago_ow_hz_s` | VSLP task adaptation |
| C034 fitted frequency range | S2 only | buy /AY/ | `f1_range_buy_ay_hz`, `f2_range_buy_ay_hz` | Limited evidence; source-lineage target |

There are two constructs and eight selectable leaves. Recommendation is specific to the table above; other tasks are not recommended. AY, EY, and OW are never combined. S5 away and ago produce separate outputs.

The current VSLP task inventory names S2 “Buy Bobby a Puppy 3 times”; its canonical text is “Buy Bobby a puppy.” and `current_protocol_expected_repetitions = 3`. Jafari et al. (2023) described a separate orofacial-kinematics BBP protocol with `historical_jafari_protocol_repetitions = 10`. `historical_protocol_is_not_current_task_contract = true`; the historical count does not control VSLP trial generation or QC.

## Inputs and estimator

Final Alignment Contract V1 provides accepted, current canonical parent-word and phone identities, direct trial and sequence IDs, final original-time bounds, boundary source, and correction authority. F05 uses the canonical/native `analysis_wav_path` at its native sample rate. It never reads raw TextGrid, provider tokens, or MFA's private working WAV, and never invokes MFA. Missing or stale Final Alignment makes the F05 leaf unavailable.

F05 reuses F04 `track_aligned_vowels` and `profile_by_id`. The production profile is Praat/Parselmouth Burg, 25-ms nominal window, 5-ms step, five maximum formants, 50-Hz pre-emphasis, 5500-Hz ceiling; the existing 5000-Hz sensitivity profile remains available. F04's ordered-track and physiological plausibility checks apply. Complete nominal frame support must remain inside the phone, trial, native audio, and outside manual exclusions. F05 adds a 40-ms minimum final phone duration and requires at least five valid frames.

For phone `[t0,t1]`, `D=t1-t0`, the analysis domain is `[t0+0.20D, t0+0.80D]`. First and final 20% are excluded. Each formant is fit separately against original-time seconds using MAD-scaled Tukey bisquare iteratively reweighted least squares (`bisquare_irls_mad_4.685_v1`): tuning constant 4.685, up to 50 iterations, relative coefficient tolerance `1e-8`. The signed slope is `β1` Hz/s. The fitted range is `|β1| × (0.60D)` Hz, equivalent to the endpoint max-minus-min of the fitted line. Raw-frame max-minus-min is not used. An unconverged fit fails explicitly.

Each valid target token is measured independently. Equivalent target tokens within a trial combine by median; valid trial values combine by median to one recording result. Token and trial audit tables preserve identities, timestamps, frame counts, fit coefficients, residual MAD, provenance, and failure reasons. Failures are NaN plus machine-readable reason, never zero.

## Source relationship and remaining validation

The reviewed S2 source lineage uses sentence reading, `/ai/` in “buy”, robust-fit F1/F2 trajectory slopes and fitted-line max-minus-min with on/off-glides excluded. VSLP retains the construct and target but uses the frozen F04 Burg settings rather than the historical 10-ms Hamming, 0.1-ms step, 22.05-kHz settings. `source_low_level_parity = pending`; exact numerical Rong parity is not claimed. S5 away/EY and ago/OW use the same VSLP estimator as a task adaptation; Rong did not establish those S5 targets. Independent numerical reference comparison, real-data checks, and clinical validation remain pending.
