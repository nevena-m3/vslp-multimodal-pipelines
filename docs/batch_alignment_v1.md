# One-time batch alignment (whole dataset)

`python -m vslp.acoustic.alignment.batch` runs Montreal Forced Aligner once over
every fixed-prompt recording and writes one `VSLP_ALIGNMENT_PACKAGE_V1` per task
code. It uses the same workbench and package code as the VSLP Alignment app, so
the Acoustic GUI's **Import Alignment Package** accepts the result unchanged.
The Acoustic GUI itself never loads or runs MFA.

```
python -m vslp.acoustic.alignment.batch --inventory WAVS.csv --zip WAVs.zip \
    --workspace C:\VSLP_alignment_batch [--codes 240_PSG_BAMBOO] [--limit-per-task 10]
```

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
* A recording whose alignment fails structural QC is left out of the package and
  listed as `NEEDS_REVIEW`; it is never forced.
* Packages cover the whole dataset; importing one into a project brings in only
  that project's recordings, matched by decoded audio content.
