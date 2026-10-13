# F09 — Bamboo pause and phrasing

**Scientific definition frozen; implementation frozen; focused engineering testing passed.** Reference, real-data, clinical, and Silero-parameter validation remain pending. Seven selectable leaves use the Bamboo Passage only. The eight historical constructs C050–C057 remain identifiable; C057 pause-pattern composite is deferred and nonselectable because the exact source scaling, loadings, intercept, and normalization are unavailable (`exact_composite_definition_not_frozen`). The older component vector is an audit concept, not an active numeric leaf. No composite formula has been invented.

## Timing authority and events

F09 consumes final reviewed segmentation in original recording seconds. It calls F07 `reviewed_timing()` for eligible reviewed speech and the effective first-to-last speech duration. Silero only proposes upstream speech segments: its probability threshold, minimum silence, and padding settings are separate from the **0.300 s F09 production pause threshold**. A 0.250 s threshold is reserved for future sensitivity analysis and is not used in production outputs.

Within each Bamboo trial, an internal reviewed nonspeech gap must lie between eligible speech intervals. A gap **≥0.300 s** is a qualifying PAUSE and splits phrases. A shorter gap is not a qualifying pause; the adjacent speech spans form one PHRASE whose duration includes that short gap. Leading and trailing nonspeech never enter the event list. Each output derives from the same per-trial PAUSE/PHRASE table, with original-time boundaries, trial identity, source speech indices, review provenance, and validity fields.

An internal hard manual exclusion is not physiological silence. The current conservative rule returns `hard_exclusion_ambiguity` for that trial's F09 values and does not count contamination as a pause or join a phrase across it. Missing or inconsistent reviewed timing similarly gives NaN plus a reason.

## Seven production outputs

Let qualifying pause durations be \(P_1,\ldots,P_N\), phrase span durations be \(R_1,\ldots,R_M\), and \(T\) be **F07 `bamboo_total_duration_s`**. All calculations occur per trial before recording medians are taken.

| Public ID | Trial statistic | Unit |
|---|---|---|
| `bamboo_pause_duration_median_s` | median(P), N≥1 | s |
| `bamboo_pause_duration_cv` | sample SD(P, ddof=1) / mean(P), N≥2 | ratio |
| `bamboo_phrase_duration_cv` | sample SD(R, ddof=1) / mean(R), M≥2 | ratio |
| `bamboo_pause_count` | N | count |
| `bamboo_percent_pause_pct` | 100 × sum(P) / T | % |
| `bamboo_mean_pause_duration_s` | mean(P), N≥1 | s |
| `bamboo_mean_phrase_duration_s` | mean(R), M≥1 | s |

Support values in the trial audit include `bamboo_total_pause_duration_s = sum(P)` and `bamboo_phrase_count = M`. These are nonselectable. With no qualifying pause, count, total pause, and percent pause are valid zero; one uninterrupted phrase remains. Mean/median pause and the pause CV are unavailable with explicit reasons, as is phrase CV when only one phrase exists. CVs are ratios, not percentages. The recording value is the median of valid trial statistics; pause or phrase lists are never concatenated across trials.

F09 has no MFA, TextGrid, Final Alignment, or private MFA audio dependency. Engineering tests are not reference, real-data, or clinical validation. A later segmentation review should audit Silero speech probability threshold, minimum speech duration, minimum silence duration, speech padding, timestamp mapping, and post-review merge/split behavior without changing F09's clinical threshold implicitly.
