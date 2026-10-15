# Alignment trial proposal policy

Policy version: `task_specific_structural_review_v3`.

This policy classifies candidate *trial structure* before linguistic Alignment. It does not validate the words produced or their boundaries. Reviewed speech intervals do not provide semantic evidence that distinguishes an ordinary within-utterance pause from a restart or false start. `READY_FOR_ALIGNMENT` means only that the checks below found no deterministic reason preventing an Alignment attempt. It does not certify trial identity or linguistic correctness. A scientist can inspect and override any ready proposal. `AUTO_ACCEPTED_STRUCTURAL` is reserved for a post-MFA Alignment result that passes structural QC.

## States and provenance

| Pre-MFA `trial_status` | Meaning | Alignment input? |
| --- | --- | --- |
| `PROPOSED` | Candidate awaiting structural classification | No |
| `READY_FOR_ALIGNMENT` | Registered prompt, valid eligible reviewed bounds, and no hard exclusion or blocking structural exception; or explicit human confirmation after these checks | Yes |
| `NEEDS_TRIAL_REVIEW` | A deterministic exception needs a scientist's decision | No |
| `OMITTED` | Candidate rejected or omitted by a scientist | No |
| `STALE` | Upstream review, prompt, or trial geometry changed | No |

Project trial choices retain `recording_id`, stable `trial_id`, `task_id`, `prompt_id`, `prompt_version`, original-time start/end, `proposal_source`, `trial_status`, `trial_review_source`, `trial_review_reason`, `trial_review_timestamp` when human-confirmed, `review_policy_version`, and `segmentation_review_hash`. The trial manifest carries these fields. Existing legacy `HUMAN_CONFIRMED` and `HUMAN_ACCEPTED` choices map to ready with HUMAN provenance. Legacy `AUTO_ACCEPTED_STRUCTURAL` is ambiguous across stages and maps to STALE, requiring review. Unknown or missing legacy values also map to STALE. Post-MFA review is stored separately as `alignment_review_status`; only post-MFA structural QC may assign `AUTO_ACCEPTED_STRUCTURAL`.

## Deterministic rules

- Bamboo: propose one candidate envelope from the first to the last reviewed speech interval, retaining ordinary internal pauses. It becomes ready only with the registered Bamboo prompt, valid reviewed bounds, and no intersecting manual exclusion, explicit multiple-attempt annotation, or stale identity/review state. Multiple speech intervals alone do not establish multiple passage attempts or prove no restart occurred.
- Controlled sentences: offer each reviewed speech interval as a candidate hypothesis. Valid, nonexcluded candidates may become ready regardless of expected count. A count mismatch preserves every candidate and flags `REPETITION_COUNT_MISMATCH` without blocking an Alignment attempt. The count never creates or deletes a trial; a match does not prove attempt identity.
- An explicit `linguistic_attempt_id` column, if present in frozen reviewed intervals, can flag multiple annotated attempts. Current ordinary Segmentation Review output does not supply this column. No restart detector, pause-duration threshold, or linguistic completeness inference is used.
- Manual exclusions intersecting a candidate produce `TRIAL_CROSSES_EXCLUSION`; trial validation blocks that geometry from Alignment input. Edits invalidate prior structural acceptance and require human confirmation. Changed frozen review or prompt version makes a saved proposal stale.
- Sustained vowels and DDK do not enter the linguistic MFA proposal path.

Only `NEEDS_TRIAL_REVIEW`, `STALE`, `PROPOSED`, missing bounds, and other unresolved states enter the pre-Alignment trial review queue. Ready cases remain available for inspection. This policy is workflow triage, not boundary validation in ALS/MND speech.
