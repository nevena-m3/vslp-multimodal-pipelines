# One-time batch alignment (whole dataset)

`python -m vslp.acoustic.alignment.batch` runs Montreal Forced Aligner once over
every fixed-prompt recording and writes one `VSLP_ALIGNMENT_PACKAGE_V1` per task
code. It uses the same workbench and package code as the VSLP Alignment app, so
the Acoustic GUI's **Import Alignment Package** accepts the result unchanged.
The Acoustic GUI itself never loads or runs MFA.

Three steps (the middle one runs in the separate `vslp-whisper` environment):

```
# 1. extract the selected WAVs (read only from the zip)
python -m vslp.acoustic.alignment.batch --inventory WAVS.csv --zip WAVs.zip \
    --workspace W --extract-only [--limit-per-task 10]
# 2. count how many times each sentence was read (faster-whisper, local)
python -m vslp.acoustic.alignment.readings --audio-dir W\audio --out W\readings.csv
# 3. align: one MFA job per task code and reading count
python -m vslp.acoustic.alignment.batch --inventory WAVS.csv --zip WAVs.zip \
    --workspace W --readings W\readings.csv [--limit-per-task 10]
```

Sentence tasks are read a different number of times by protocol and speaker
(the 2026-11 pilot found 3 readings in protocol 1075 and 3 to ~8 for "Buy
Bobby a puppy"), so their count is taken per recording. Speech-to-text only
counts readings; MFA always aligns the canonical prompt text. Each speech
chunk is transcribed separately; a chunk much longer than the file's typical
one-reading chunk counts as several readings (repeats without a pause). If no
reading is recognised, the whole file is tried once. A recording with speech
(>= 1.5 s) but no recognised reading is aligned with the protocol's usual count
and left to QC; only recordings with no count or no speech are held back
(`held_back_recordings.csv`). With `--audio-dir` alone, only files present in
that folder are considered.
Pilots (`--limit-per-task`) draw one file per participant, spread over protocols.

* Which recordings and prompts: `src/vslp/acoustic/alignment/batch_tasks.yaml`
  (task code in the file name → registered prompt and repetition count).
  DDK and vowel codes are never aligned; SENT_CAPEV and SIT need their sentence
  identified first and are not part of the fixed-prompt batch.
* Speaker adaptation groups recordings by `SubjectID`.
* Default profile: `mfa_vslp_en_us_arpa_334_wide_beam_v1` (same model and
  dictionary as the engineering profile, `--beam 100 --retry_beam 400`).
* Only needed WAVs are extracted from the zip into `<workspace>/audio`; sources
  are never modified. Finished tasks are skipped on re-run (`batch_state.json`).
* Outputs: `packages/<CODE>__<task_id>/`, `batch_recordings.csv` (state and QC
  flags per recording), `plots/` (waveform, spectrogram, word and phone tiers in
  original recording time) for recordings that need review.
* QC also flags `long_pause_within_reading_requires_review` when consecutive
  words of one reading of a short sentence are more than 1 s apart (one reading
  spread over several).
* A recording whose alignment fails structural QC is left out of the package and
  listed as `NEEDS_REVIEW`; it is never forced.
* Packages cover the whole dataset; importing one into a project brings in only
  that project's recordings, matched by decoded audio content.
