# Alignment trial review contract

The task-first Alignment workflow treats a recording as a container for one or
more explicitly reviewed trials. A reviewed speech interval is offered as a
boundary candidate; it is not automatically called a repetition. The scientist
confirms each trial's original-recording start and end times before MFA runs.
The current `wstg_we_see_three_geese` protocol metadata expects three trials.
This is an expectation for review, not a fabricated third trial: two confirmed
trials remain two trials. The run manifest records expected and observed counts
and the shortfall for each recording.

For MFA tasks, each confirmed trial becomes one corpus WAV and one copy of its
approved alignment transcript. MFA still runs once at corpus level. The
per-trial working time map sends every returned word and phone boundary back to
the original recording clock. The established word and phone CSV columns stay
unchanged. `alignment_trial_tokens.csv` links globally unique word and phone
indices to `recording_id` and `trial_id`; `alignment_trial_diagnostics.csv`
records coverage for each trial. The trial manifest and hashes are retained in
the run. Frozen runs additionally contain the three final trial artifacts.
Historical runs without this manifest retain `legacy_single_unit` provenance;
they are not retroactively interpreted as complete multi-trial alignments.
When a new reviewed run replaces a frozen legacy run, the GUI asks for explicit
confirmation. The previous final product is kept under `archived_final/` and
the new run becomes the authoritative `final/` product. Both run IDs remain
traceable.

The scientist reviews waveform, word tier, phone tier, and trial boundaries,
listens on the original time axis, then marks each trial `ACCEPTED` or
`NEEDS_REVIEW`. MFA completion leaves trials `UNREVIEWED`. A new-style run
cannot freeze unless every planned trial produced results and every trial was
accepted. Imported and older frozen Alignment products retain their existing
contract.

Review now distinguishes two repairs. A wrong trial definition is edited on the
waveform; confirming it marks that recording's old trial results
`STALE_AFTER_TRIAL_EDIT`. A targeted rerun aligns that recording in a new
historical run and reuses the unchanged recordings from the prior run. The new
manifest names the source run and the rerun recording; unchanged review
decisions and manual corrections carry forward. No historical MFA word or
phone table is overwritten.

If the trial is correct but a word or phone boundary is wrong, the scientist
selects the corresponding tier and drags a highlighted token handle. The run's
`alignment_manual_corrections.csv` records original and corrected times, token
identity, trial, reason, timestamp, and approval state. Reset token/trial
removes only this overlay. Any edit after acceptance reopens that trial as
`NEEDS_REVIEW`. Freeze validates reviewed boundaries and publishes the same
word/phone schemas with approved corrections applied. It also publishes
`final_alignment_manual_corrections.csv`; final trial review distinguishes
`ACCEPTED_MFA` and `ACCEPTED_MANUAL`. The original run MFA tables and their
hashes remain unchanged. Stale, unreviewed, or unresolved trials cannot freeze.
Published frozen runs remain immutable; corrections are made before freeze or
through a new Alignment run that can supersede the old frozen product after
review.

Expected repetition count is shown beside the confirmed count. A mismatch
requires explicit reviewer acknowledgement in the GUI but does not manufacture
or forbid a real missing/restarted trial. Diagnostics open in a separate,
scrollable dialog so the review waveform remains the main screen.

Speaker identity comes first from explicit deidentified project metadata. In
its absence, a random, stable project-local technical key is persisted per
recording. That key asserts no participant identity and is never inferred from
filenames. All repetitions of a recording share the same key. MFA's speaker
grouping can affect normalization and adaptation, so this fallback is recorded
in the run manifest and still needs empirical boundary review before research
use. See the [MFA corpus documentation](https://montreal-forced-aligner.readthedocs.io/en/v3.3.4/user_guide/corpus_structure.html)
and [speaker adaptation guidance](https://montreal-forced-aligner.readthedocs.io/en/v3.0.7/user_guide/concepts/speaker_adaptation.html).

Sustained phonation and DDK do not pass through MFA. The same task-neutral
trial markers can be confirmed and persisted for those recordings. Reviewed
segmentation preserves separate speech/event intervals; confirmed trial IDs
are carried into the F01/F02 native track/pulse audit and the F10 event audit
when a frame/event is wholly inside one trial. Current F01/F02 sustained
recording-level measurements still use one frozen stable phonation region per
recording, so per-trial acoustic aggregation is not provided by this Alignment
workflow. DDK preserves individual reviewed events and sequence breaks, but a
sequence ID is not automatically a protocol trial ID. Trial-specific feature
aggregation for these tasks needs a separate, scientifically approved feature
contract; this pass does not change their formulas.
