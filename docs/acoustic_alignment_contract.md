# Acoustic Alignment contract

Alignment is optional and follows frozen Segmentation Manual Review. The stage
reads `003_segmentation_review/final/final_segmentation_decisions.csv` and
`final_segmentation_intervals.csv`; it never rewrites either file or canonical
audio. Families requiring words or phones read only the frozen product under
`004_alignment/final/`.

## Prompt manifest

The user supplies JSON with `manifest_version`, `task_id`, `prompt_version`,
`language`, `transcript`, `expected_words` (ordered string array), and
`phone_set`. Version 1 accepts `phone_set = ARPABET_CMU_39`. The transcript and
expected words must agree after case and punctuation normalization. No prompt
text is inferred from the task name, filename, or audio. Each alignment run
contains a snapshot and SHA-256 of the supplied manifest.

## Sources and time axis

`external` imports validated word and/or phone CSVs with `recording_id`,
`start_sec`, `end_sec`, and `word`/`phone` (or `label`). A combined legacy-style
CSV with `token_type` can also feed this stage. Imported timestamps are on the
original recording clock. Tokens outside the reviewed analysis window or across
manual contamination are rejected and counted in diagnostics.

`mfa` uses a single corpus-level `mfa validate` followed by a single `mfa align`
for an explicitly mapped set of deidentified speaker directories. Supported
engineering versions are pinned to 3.3.4 and 3.4.0 in legacy mode. The 3.4.0
`align_hf` command can be constructed, but production HF mode is held at
`HF_PREFLIGHT_NOT_SUPPORTED` until an equivalent validated OOV/feature-generation
preflight has been tested. Other versions fail closed. No model, dictionary,
MFA package, or G2P model is installed or downloaded by the application.

An explicit versioned engineering profile supplies model and dictionary paths,
ARPABET phone set, 16-kHz working rate, text-normalization profile, and OOV
policy. See `docs/examples/mfa_engineering_profile.example.json`. A separate
recording-to-speaker manifest is mandatory. The local dictionary is checked
for every normalized alignment-transcript word before MFA validation; unresolved
OOVs stop the batch. Model/dictionary hashes are recorded and optionally pinned
in the profile. MFA validation tests model/dictionary/phone compatibility and
feature generation. This is an engineering profile, not empirical validation.

The expected prompt remains immutable. Optional reviewed transcript overrides
are per recording, versioned, and require reason, reviewer, and timestamp.
`vslp_prompt_nfkc_v1` applies Unicode NFKC, uppercases, removes punctuation
except apostrophes, splits hyphens, collapses whitespace, and rejects numeric
digits pending a reviewed spoken-form transcription. Both expected prompt and
actual normalized MFA transcript are recorded. There is no ASR fallback.

Private working WAVs are mono and resampled to 16 kHz by SciPy polyphase
resampling. Reviewed analysis bounds are honored; manually excluded spans are
removed and remaining pieces concatenated because MFA expects a continuous
utterance WAV. The saved piecewise map returns token times to the original
recording clock. A token crossing a removed span is invalid. Source audio is
never modified. Exclusions, leading/trailing material outside the analysis
window, and their source/working hashes are retained in run provenance.

## Output

Each execution has a unique `alignment_run_id` and writes to
`004_alignment/runs/<alignment_run_id>/`. Normalized word and phone tables,
per-recording diagnostics, the working time map when applicable, the prompt
snapshot, and a stage manifest remain in that historical run. The run manifest
records source reviewed-segmentation hashes, prompt and imported-table hashes,
provider/model/dictionary versions, sample-rate policy, counts, and output
hashes. Missing tools/resources and per-recording failures are explicit.
`logs/run_state.json` records `RUNNING`, `PREFLIGHT_FAILED`, `COMPLETED`, or
`COMPLETED_WITH_FLAGS`; the final manifest records `FROZEN` only after structural
checks. An interrupted run retains its last state for inspection.

Freezing a selected run publishes `004_alignment/final/`:

- `final_alignment_words.csv`
- `final_alignment_phones.csv`
- `final_alignment_diagnostics.csv`
- `final_alignment_manifest.json`

The manifest is written last and is the freeze marker. A frozen alignment is
immutable. The loader verifies word/phone/diagnostic hashes and the current
frozen segmentation hashes before returning tokens.

Before freeze, the stage also verifies run-table hashes, recording/task/prompt
identity, finite positive non-overlapping intervals within the reviewed domain,
and phone containment in the indexed word. A structural failure prevents freeze.
Low coverage and unusual duration/likelihood diagnostics are review signals,
not universal rejection thresholds for dysarthric speech.

Word fields include run/recording/file/task/prompt identity, word index and
label, original-clock start/end/duration, original sample indices, sequence ID,
status, and source. Phone fields add parent word index, phone index, raw and
normalized ARPABET labels, stress digit, and deterministic vowel classification.
Diagnostics include expected and matched words, coverage, missing/extra words,
invalid/overlapping/nonmonotonic/out-of-domain intervals, and sample rates.
Coverage below 80% is reported as `LOW_COVERAGE`; it is not a universal token
exclusion rule. Individual features apply their own coverage requirements.
Word-only alignment is explicitly `PARTIAL_ALIGNMENT` with
`no_aligned_phones`; it can be frozen for word features, while phone-dependent
features still fail their own prerequisite.

`AlignmentStore` provides `get_word_tokens`, `get_phone_tokens`,
`get_vowel_tokens`, and `get_tokens_by_label`. Family 07 uses this shared loader
and no longer accepts a separate feature-local alignment CSV. Family 07's
existing 80% word-coverage gate and duration formulas remain unchanged.

## Automated real MFA engineering self-test

1. Install Miniforge or Miniconda explicitly, then open **Miniforge Prompt**.
2. Create a dedicated environment and pin the tested MFA version: `mamba create
   -n vslp-mfa-334 -c conda-forge montreal-forced-aligner=3.3.4` (use `conda`
   if `mamba` is unavailable). Run `conda activate vslp-mfa-334` and `mfa version`;
   verify **3.3.4**. This environment/version is engineering supported by
   command construction here; it has not been run locally on this PC.
3. Obtain an explicitly approved legacy English acoustic model and matching
   ARPABET dictionary separately. Store them outside Git and record their
   versions and SHA-256 hashes. Do not use clinical recordings for the smoke test.
4. Copy and edit the engineering profile example with exact local paths and
   hashes. In Alignment choose **Montreal Forced Aligner**, select that profile,
   click **Check MFA environment**, then **Run MFA Self-Test**. A participant
   project, prompt manifest, speaker mapping, and corpus setup are not needed.
   If offline Windows System.Speech is unavailable, optionally select a
   *nonclinical* WAV speaking “We are testing speech today.”
5. The application creates a temporary one-speaker corpus, transcript, reviewed
   analysis domain, and time map. It runs the normal provider's dictionary OOV
   check, `mfa validate`, and corpus-level `mfa align`; normalizes word/phone
   tables; freezes and reloads the temporary Alignment output. On success it
   deletes the full working corpus and retains only a JSON result under the
   system temporary directory. On failure it deletes the corpus and may retain
   provider logs in a separate temporary diagnostics folder.

The same service is available at:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path  # when running from an uninstalled checkout
python -m vslp.acoustic.alignment.self_test --profile C:\path\to\mfa_profile.json
```

Add `--nonclinical-wav C:\path\to\test.wav` only if offline TTS fails. The GUI
and CLI use the same implementation. To run its real opt-in integration test,
set `VSLP_RUN_REAL_MFA_SELF_TEST=1` and `VSLP_REAL_MFA_PROFILE` to the profile
path, then run `pytest tests/integration/test_mfa_self_test_real_opt_in.py`.
No default test downloads models or requires MFA.

**PASS means only that the installed MFA environment and VSLP provider completed
a real end-to-end engineering alignment.** Synthetic speech does not establish
ALS validity, clinical boundary accuracy, or downstream feature equivalence.
Those require manually annotated reference recordings and separate analysis.

For the older optional multiple-speaker integration test, set
`VSLP_REAL_ALIGNMENT_ROOT` to a *nonclinical fixture root* containing
`project_manifest.json`, `prompt.json`, `speakers.json`, source WAVs, and frozen
reviewed segmentation. Set `VSLP_REAL_MFA_PROFILE` to the edited profile. Run
`python -m pytest tests/integration/test_mfa_real_opt_in.py -q` in the MFA
environment with this repository installed. The test copies the fixture,
runs corpus MFA, freezes Alignment, reloads hashes, and exercises Family 07
word duration plus Family 04 token formant dispatch. Without both variables
it skips. No participant recordings are bundled. This separate test checks
multiple speaker organization; the normal one-click self-test uses one speaker.

For manual-reference assessment, run
`python -m vslp.acoustic.alignment.compare_reference word|phone MFA.csv MANUAL.csv REPORT.csv`
with optional `--tolerance` values specified by the study team. For feature
sensitivity, use `features` with feature-value tables computed separately from
MFA versus manual boundaries. Reports include median, IQR, 95th percentile,
unmatched counts, and proportions within supplied tolerances. No acceptance
tolerance is imposed by this code.

## Current limits

MFA must be installed separately with a compatible model and dictionary. A live
MFA installation was unavailable for an end-to-end execution test. Manual word/phone
boundary editing is outside this stage; the viewer provides zoom, pan, token
inspection, and diagnostic coverage. The phone classifier currently supports
explicit ARPABET/CMU labels only. Other phone sets need a declared mapping.
