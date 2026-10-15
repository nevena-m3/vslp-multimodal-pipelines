# F08 — Bamboo speaking and articulation rate

Implementation status: **implementation ready for review**. Reference, real-data, and clinical validation remain pending.

The two active constructs are C048 speaking rate and C049 articulation rate. Their only selectable outputs are `bamboo_speaking_rate_wpm` and `bamboo_articulation_rate_wpm`. Bamboo Passage is the sole recommended and executable task. The former syllable-per-second outputs and generic `speaking_rate_words_min` are deprecated and nonselectable; their legacy calculator remains for historical compatibility, while the active dispatcher uses `family08_v2`.

The registered `bamboo_passage_v1` text has SHA-256 `a132fa58fe945d86812dec3cce3936f6f586a24527105f3bc9009f18efdb5f67`, one expected repetition, and **97 scientific words**. `good-looking` is one word. The numerator is this versioned *expected prompt count*, never MFA tokens, aligned words, ASR output, or an observed production count. Prompt completion is not verified by F08. These are **fixed-prompt, nominal WPM** measures and assume approximately complete production of the 97-word passage.

For each reviewed Bamboo trial, F08 calls F07's `reviewed_timing` on the trial's reviewed intervals. It does not implement separate timing or pause-removal logic:

- Speaking rate: `60 × 97 / bamboo_total_duration_s`. F07's total duration is last eligible reviewed speech offset minus first eligible reviewed speech onset; internal pauses remain in the denominator.
- Articulation rate: `60 × 97 / bamboo_speech_duration_s`. F07 sums eligible reviewed speech intervals after hard exclusions. There is no second 300-ms pause subtraction.

The canonical one-production Bamboo domain is the first through last eligible speech boundary in final reviewed timing, whether or not a legacy Alignment trial-proposal file exists. An AUTO proposal is never treated as human confirmation. If multiple trial proposals are present, current-hash human confirmation is required to preserve separate trial measurements; rates are then calculated per trial and the recording output is their median. Unverified multiple proposals remain unavailable. The trial audit retains identity, bounds, both denominators, both rates, failure reasons, and review source.

Both leaves require frozen reviewed segmentation/timing. Neither requires Final Alignment, MFA, TextGrid, or acoustic waveform. Output provenance records `expected_prompt_count`, `prompt_completion_verified=false`, prompt version/hash, and the frozen reviewed timing source. Missing or nonpositive denominators, inconsistent speech and total durations, invalid registered prompt count, and nonfinite rates yield NaN with explicit reasons. One missing denominator does not erase the other valid leaf. Normally articulation WPM is at least speaking WPM.

The literature supports the parent rate constructs; implementation tests do not establish prompt completeness, normative validity, ALS sensitivity, reference parity, or clinical validity.
