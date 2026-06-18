# VSLP Acoustic GUI v0.17.1 - Setup cleanup and duplicate media policy

This patch cleans the Setup screen and adds deterministic duplicate-media handling.

## Setup screen cleanup

Removed extra explanatory panels from Setup. The screen now focuses only on:

1. Project setup
2. Project gate
3. Ingest summary

Metadata remains only on the Metadata tab.

## Duplicate media policy

When VSLP discovers multiple supported files with exactly the same filename stem and different extensions, it keeps one file only.

Priority:

1. `.wav`
2. `.mp4`
3. `.webm`
4. other supported extensions

Example:

```text
SUBJ001_BAMBOO.wav
SUBJ001_BAMBOO.mp4
SUBJ001_BAMBOO.webm
```

VSLP keeps `SUBJ001_BAMBOO.wav` and writes skipped duplicate records to:

```text
acoustic/001_ingest/tables/audio_ingest_skipped_duplicates.csv
```

This policy is also used by preprocessing through the shared file-discovery utility.
