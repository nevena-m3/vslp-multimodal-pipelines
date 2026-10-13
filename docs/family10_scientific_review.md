# Family 10 scientific review — downstream reviewed-event freeze

## Scope and status

C058 DDK rate and C059 cycle-to-cycle temporal variability are the two active constructs. Each has `/pa/`, `/ta/`, `/ka/`, and `/pataka/` children (eight selectable leaves). Raw cycle count is QC support only. The historical generic `ddk_rate_syll_s` and `ddk_cycle_mad_s` IDs are nonselectable legacy outputs; their older five-event/analysis-window formulas are not the current production contract.

**Scientific definition frozen; downstream feature implementation frozen; synthetic engineering testing passed.** The upstream DDK segmentation/event detector, hand-label comparison, real-data, reference, and clinical validation remain pending. No detector, MFA, TextGrid, Silero, or Final Alignment is used by F10.

## Input and target identity

The feature reads `acoustic/003_segmentation_review/final/final_ddk_events.csv` within the frozen final DDK review context. The shared feature dispatcher uses final review decisions to identify kept recordings. This is not a Silero prerequisite. F10 does not derive cycle events from final speech segments. Rows require `recording_id`, `trial_id`, `sequence_id`, `task`, `ddk_target`, `ddk_mode`, `event_index`, `start_sec`, `center_sec`, `end_sec`, `valid`, `review_status`, and `rejection_reason`. `event_id` is retained when supplied. Accepted review states are `HUMAN_ACCEPTED`, `REVIEWED_ACCEPTED`, and `FINAL_ACCEPTED`. Only valid accepted events enter calculation. Missing table gives `missing_reviewed_ddk_events`.

`pa`, `ta`, and `ka` use AMR syllable cycles. `pataka` uses SMR complete `pa-ta-ka` triplet sequences, requiring `sequence_label=pa-ta-ka` and `sequence_complete=true`. The reviewed center must consistently denote the center of the whole cycle/sequence. Distinct sequence IDs prevent calculation across an exclusion or sequence break.

## First ten and equations

Within each target and trial, sort valid accepted events by `center_sec`, then `event_index` and `event_id` for deterministic tie handling. Select exactly the first ten. Later events cannot alter the output. Fewer than ten gives NaN and `insufficient_valid_ddk_events`. A tied selected center gives `invalid_ddk_event_order`.

For AMR, `rate = 10 / (end of tenth accepted cycle − start of first accepted cycle)` syllables/s. For SMR, `rate = 30 / (end of tenth complete sequence − start of first complete sequence)` syllables/s; 30 is the syllable-equivalent count. The denominator is onset-to-offset, never center-to-center.

For centers `t₁…t₁₀`, compute nine intervals `Iᵢ=tᵢ₊₁−tᵢ`, then `cTV=(1/8) Σᵢ₌₁⁸ |Iᵢ₊₁−Iᵢ|` seconds. For SMR the centers are complete-sequence centers. Larger cTV means less regular timing.

Each trial is measured separately. The recording result is the median of valid trial results for the same target. Targets and trials are never pooled before measurement. Audit tables retain all reviewed candidate rows, selection rank, original-time bounds, counts, trial values, and failure reasons.

## Validation limits

The reviewed-event writer and detector are not yet production-frozen. Synthetic tables establish downstream arithmetic and routing, not event accuracy. No biological, reference, or clinical validity is claimed.
