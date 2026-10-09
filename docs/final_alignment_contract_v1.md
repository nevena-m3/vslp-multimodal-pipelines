# Final Alignment Contract V1

Status: engineering interface frozen for feature development. This is not a
claim of MFA boundary accuracy in ALS/MND speech.

## Input boundary

`FINAL_ALIGNMENT_CONTRACT_V1` is produced only by freezing a reviewed Alignment
run. All exported `start_sec` and `end_sec` values use original recording time.
The private 16 kHz MFA WAV, LAB and trial-local TextGrid times are never feature
measurement inputs. Waveform features reopen the reviewed native/canonical audio
using these original-time coordinates.

The loader checks frozen status, schema identity, hashed tables and evidence,
frozen Segmentation Review, source audio, current trial selection, pinned MFA
profile, structural review policy and registered prompt identity. A missing, stale or unresolved product yields a structured
prerequisite issue; it never invokes MFA.

## Files in `acoustic/004_alignment/final`

| File | Role |
| --- | --- |
| `final_alignment_contract_trials.csv` | Feature-facing trial identity, status, original-time bounds and unique sequence ID. |
| `final_alignment_canonical_words.csv` | Feature-facing scientific lexical units. |
| `final_alignment_canonical_phones.csv` | Feature-facing phones with direct trial, sequence and parent canonical word. |
| `alignment_provider_words.csv`, `alignment_provider_phones.csv` | Uncorrected provider-token provenance only. |
| `final_alignment_words.csv`, `final_alignment_phones.csv` | Legacy final provider-token tables, retained for backward compatibility and audit. Do not use as a new feature input. |
| `final_alignment_trial_tokens.csv` | Legacy provider token-to-trial link table. |
| `final_alignment_manual_corrections.csv` | Original and approved corrected boundaries with reviewer state and timestamp. |
| `final_alignment_trial_review.csv` | Final review decisions. |
| `final_alignment_evidence_index.csv` | Trial-to-plot paths and SHA-256 hashes. |
| `evidence/*_overview.png` | Whole-trial waveform, spectrogram, reviewed speech, exclusions, canonical words and final phones. |
| `evidence/*_detail.pdf` | Bamboo-only paginated detail in original time. |
| `final_alignment_manifest.json` | Contract version, run/profile/dependency identities and artifact hashes. |

The scientific Bamboo passage has **97 canonical lexical units**. The MFA
normalization profile splits `good-looking` into `GOOD` and `LOOKING`, yielding
**98 provider tokens**. The two tokens map to one canonical word whose bounds
span the first token onset through the second token offset. The provider tokens
remain available in provenance. Any token sequence that cannot be mapped in
order to the canonical transcript blocks this contract; the mapping is never
guessed.

Manual corrections are applied before canonical tables and evidence plots are
built. Feature-facing rows carry `boundary_source`; provider rows and the
correction table retain original MFA boundaries. `sequence_id` includes both
recording and trial identity. F07 adjacency cannot bridge that ID or a hard
exclusion. The frozen trial gate rejects unresolved or stale review states.

## Feature prerequisites

Alignment is optional infrastructure at concrete-leaf and task level.
F01/F02/F08/F09/F10/F11/F12 and other nonlinguistic leaves continue from their
own reviewed inputs without loading Alignment. Sustained-vowel F04 uses its
reviewed phonation region. Connected-speech F04 uses final canonical phones and
native waveform for formants. F05 may use the same target identity and ordered
context; it defines its own trajectory windows. F06 uses the aligned phone only
to localize a target, then requires a separately validated acoustic subevent.
F07 word/vowel duration and rhythm use the canonical tables; speech time and
task elapsed duration use frozen reviewed timing. F08/F09 are not migrated to
MFA. A mixed selection may run its non-Alignment leaves while Alignment-dependent
leaves report an unmet prerequisite.

For F06 localization, a target's `word_index` identifies the canonical word
within its trial and `phone_index` identifies the phone within that word.
Repeated trials require explicit trial identity or produce an ambiguous-target
failure; provider-wide token indices are not scientific target IDs.

Every mixed-dispatch status row records `input_contract`. Alignment-dependent
rows also record the final Alignment run and contract version. The stage manifest
records canonical word/phone hashes only when an Alignment-dependent leaf was
selected.

## Compatibility and scientific limits

Existing frozen packages without `FINAL_ALIGNMENT_CONTRACT_V1` remain archived,
but the default feature-facing loader reports `unsupported_final_alignment_contract`.
New freezes preserve the legacy provider-token filenames; their meaning has not
changed. Existing feature tests requiring the full project environment may not
collect in the local `.venv-win` because `librosa` is absent; no environment was
changed for this work.

Structural coherence, including a deterministic canonical mapping, does not
prove a complete production, correct lexical perception, or accurate word/phone
boundaries in dysarthric speech. Sentence target inventories and any mapping of
human transcript overrides to scientific canonical units require explicit lab
review; a mismatched sequence must not be forced into this contract.
