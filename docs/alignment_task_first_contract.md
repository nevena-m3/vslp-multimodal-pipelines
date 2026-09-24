# Task-first Alignment workflow

Alignment reads the selected task from the project manifest and the frozen
reviewed segmentation. Its versioned task registry is
`src/vslp/acoustic/alignment/task_prompts.json`. The only currently registered
WSTG stimulus is the user-supplied **“We see three geese.”** (`wstg_we_see_three_geese`,
v1). Other WSTG items have not been supplied. The complete canonical Bamboo
Passage text was not found in repository configuration or source files; Bamboo
therefore reports `MISSING_CANONICAL_PROMPT` until that text and version are
provided. Sustained phonation and DDK are explicitly marked as not requiring
linguistic Alignment.

For multiple registered stimuli, each recording needs an explicit stimulus
selection. The GUI persists prompt selection, speaker ID, and any reviewed
alignment transcript in `configs/alignment_choices.json`. The user edits these
in the project GUI; no JSON browsing is needed. A transcript correction needs
both a reason and reviewer ID. The original expected prompt and both transcript
hashes are retained in the run configuration. Task prompt manifests, speaker
mapping, and transcript overrides are generated internally and snapshotted
under each Alignment run. The frozen word and phone schemas remain unchanged.

Speaker IDs are taken automatically only from the project's metadata index
when the individual `subject_id_source` field is an explicit metadata CSV.
Older indexes without this per-field provenance require scientist confirmation.
Filename-derived or generated
subject IDs are not trusted for MFA speaker adaptation. Missing IDs require
scientist entry in the Alignment table and raise `SPEAKER_ID_REQUIRED` until
resolved. Repeated recordings can share the same deidentified ID.

MFA 3.3.4 `validate` and `align` each receive a distinct
`--temporary_directory` and `--clean`. The short directory lives under the
system temporary folder at `vslp_mfa_work/<alignment-run-id>/<command>/<nonce>`
because MFA's Windows sqlite/Kaldi tools fail with deep project paths. A
successful command removes its temporary files; a failed command retains its
isolated workspace for diagnostics. The path and command are captured in run
provenance. Global MFA models and default workspaces are untouched.

The GUI sends environment checks, self-tests, and Alignment runs to the
existing Qt worker thread. MFA remains one corpus-level subprocess per
validation and alignment phase. Progress identifies stages rather than
pretending to report individual recordings during the MFA corpus command.
Inspection shows the waveform, reviewed domain, manual exclusions, word and
phone regions and labels, and token timing/source. The GUI requires an Inspect
action before enabling Freeze. Run selection and inspection state persist in
the project's Alignment choices; historical runs and frozen outputs retain
their existing contracts. Cancellation is not yet offered because safe
termination of the Conda/MFA process tree across Windows child processes has
not been validated; closing the GUI is not a supported cancellation method.

The synthetic WSTG fixture is an engineering integration test. It does not
validate clinical boundary accuracy or downstream biomarker equivalence.
