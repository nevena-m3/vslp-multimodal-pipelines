# Family 08 — Speech and articulation rate

Source: `08_Speech_Articulation_Rate_Codex_Implementation.docx`.
Only constructs C048 (Speaking rate) and C049 (Articulation rate) are implemented.
The exact source IDs are `speaking_rate_syll_s`,
`speaking_rate_words_min`, and `articulation_rate_syll_s`.

The feature stage requires frozen reviewed decisions and intervals from
`acoustic/003_segmentation_review/final/`. It does not run Silero or modify audio.
Task time begins at the first final patient-speech onset and ends at the last
final patient-speech offset, retaining pauses between them. Internal reviewed
nonspeech intervals of at least 300 ms count as pauses. Shorter gaps remain in
speech time. Manual contamination intervals inside that span are subtracted
from **both** denominators, following the user's explicit decision. They are
not physiological pauses.

- Speaking syllables/s = manifest syllable count / analyzed elapsed task seconds.
- Speaking words/min = 60 × manifest word count / analyzed elapsed task seconds.
- Articulation syllables/s = manifest syllable count / (analyzed elapsed task
  seconds − internal pause seconds at least 300 ms).

The immutable parameter set is `family08_bamboo_reviewed_v1`; algorithm version
is `family08-rate-1.0.0`. No rate is computed from an inferred prompt count.
Supply a JSON manifest using
`docs/family08_prompt_manifest.example.json`: set the task ID, prompt version,
positive integer counts, count source, and explicitly confirm whether the prompt
applies to all recordings. A missing count yields NaN and a per-output reason.
The word count is needed only for `speaking_rate_words_min`.

The detailed family document recommends Bamboo Passage only; this overrides the
broader WSTG checkmarks for C048/C049 in the master matrix. WSTG has no Family 08
recommendation in the active registry. The family document's old 50 ms Silero
padding and 48 kHz master-branch text conflict with the current pipeline.
The user confirmed retaining current frozen reviewed 0 ms boundaries; the
canonical native-rate audio contract also remains authoritative.
