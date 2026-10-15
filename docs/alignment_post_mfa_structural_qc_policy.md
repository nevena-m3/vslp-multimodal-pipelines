# Post-MFA structural Alignment review policy

Policy version: `alignment_structural_review_v2`.

`AUTO_ACCEPTED_STRUCTURAL` is assigned only after an admitted `READY_FOR_ALIGNMENT` trial has a parsed MFA result and every deterministic check below passes. It means **structurally coherent under this policy**. It does not establish word or phone boundary accuracy, correct Alignment in dysarthric speech, clinical validity, or human review. `NEEDS_REVIEW` enters the exception queue; `STALE` cannot be frozen. MFA does not provide a validated confidence score in this pipeline.

## Checks and reason codes

| Condition | Reason code |
| --- | --- |
| MFA failed, omitted a TextGrid, or preflight prevented execution | `MFA_EXECUTION_FAILED` |
| No parsed word or phone tier | `EMPTY_ALIGNMENT`, `MISSING_WORD_TOKENS`, `MISSING_PHONE_TOKENS` |
| Expected sequence not covered; required or extra token | `INCOMPLETE_TRANSCRIPT_COVERAGE`, `MISSING_REQUIRED_TOKEN`, `UNEXPECTED_TOKEN` |
| Unresolved dictionary OOV or unidentified word | `UNRESOLVED_OOV` |
| Nonfinite or nonpositive duration | `INVALID_BOUNDARY` |
| Word or phone exceeds reviewed trial bounds after allowing half a working sample of mapping tolerance | `TRIAL_TOKEN_BOUNDARY_MISMATCH` |
| Phone outside parent word or missing parent | `PHONE_WORD_RELATIONSHIP_INVALID` |
| Token order goes backward or overlaps | `NONMONOTONIC_WORDS`, `NONMONOTONIC_PHONES` |
| Frozen hard exclusion intersects trial | `TRIAL_CROSSES_EXCLUSION` |
| Trial state or frozen Segmentation Review hash changed | `STALE_TRIAL` |
| Registered prompt ID or version changed | `STALE_PROMPT` |
| Human transcript override is present | `TRANSCRIPT_CORRECTION_REQUIRED` |
| Observed candidate count differs from protocol count | `PROTOCOL_TRIAL_COUNT_MISMATCH` |

Existing parser/diagnostic codes such as `REVIEW_REQUIRED`, `TOKEN_IDENTITY_MISMATCH`, and `INVALID_BOUNDARIES` remain available. The count mismatch is a post-MFA review flag, never a pre-MFA admission block. Expected count never creates or deletes trials. Hard exclusions prevent corpus creation before MFA and are checked again against final trials. Original-time remapping retains a per-trial working-time map; a token that cannot map into its reviewed trial fails structural processing.

The run-level QC index and summary are written for every trial-aware run. Flagged trials receive plots; clean trials do not receive per-trial plots by default. Scientific Bamboo word count remains **97**; the MFA-normalized transcript has **98** tokens because `good-looking` becomes `GOOD LOOKING`.
