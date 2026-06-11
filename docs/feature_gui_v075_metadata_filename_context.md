# Feature GUI v0.75 - Metadata matching and filename context fallback

Built from v0.74 QC task/framework.

## Why
Kinematic feature tables may contain `video_id` stems without file extensions, while metadata often contains `.webm` and `.wav` rows. The old metadata merge could choose basename matching first, even when basename produced zero matches, and never reach stem matching. Acoustic rows with `.wav` names often worked, while kinematic rows did not.

## Changes
- Metadata join now chooses the best actual matching key, not the first existing key.
- File-stem metadata duplicates, such as parallel `.webm` and `.wav` rows for the same stem, are safely deduplicated for context merge.
- Project page adds Filename context inference controls.
- Filename fallback can infer:
  - subject_id
  - protocol_id
  - iteration
  - recording_date
  - task
- The task fallback uses the final textual filename block after date/index tokens, for names like `MIBD09_270_2_20230619_1004_NSM_PUFF`.
- Metadata values take priority; filename parsing fills only empty/missing canonical context fields.
- Output table `filename_context_parse.csv` records parser results.

## Safety
No base feature values are changed. Parsed context is only used for subject/task/date/iteration context fields.
