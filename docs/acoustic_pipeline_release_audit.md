# VSLP Acoustic Pipeline: final software release audit

Audit scope: current `september-review` checkout, active 79-construct catalog, family metadata, mixed dispatcher, task/prompt registry, GUI stage wiring, Git index and tests. This audit does not establish source-package parity or clinical validity. The detailed row-level [implementation inventory](acoustic_feature_implementation_inventory.csv) and [definition closure](acoustic_feature_final_definition_closure.md) remain authoritative for individual IDs and unresolved contracts. No acoustic formula was changed in this pass.

## Decision

The **implemented subset is software definition-complete**: each implemented public ID is registered, selectable, assigned to exactly one executor group, and carries evidence, unit, analysis unit, algorithm version and parameter-set metadata. The 21 unavailable exact leaves and 19 unresolved templates remain nonselectable. A successful mixed reviewed-DDK fixture was added.

**Post-audit privacy resolution:** The user approved a conservative history cleanup. The 53 files identified in the original release audit, 24 additional questionable plots/notebooks/documentation files, and one older path alias were removed from all advertised branch and tag histories. A fresh clone has zero matching paths. The private backup and scope limits are described in the [data-hygiene audit](acoustic_data_hygiene_audit.md). The original audit's data blocker is closed for controlled Git refs; unresolved scientific/task-content requirements remain below.

## Recomputed feature and dispatch inventory

The code has **79 constructs**, **138 output records with 138 unique IDs**, of which two records are disabled placeholder templates. Thus there are **136 concrete exact registry leaves**: **115 implemented/selectable** and **21 exact not implemented/nonselectable**. The full unresolved-template inventory is **19** (two disabled output records and 17 construct-level templates). No `F13` executor or selectable leaf exists.

| Family | Constructs | Concrete leaves | Implemented | Exact unavailable | Templates | Executor group | Direct unit tests / integration evidence |
|---|---:|---:|---:|---:|---:|---|---|
| F01 F0/pitch | 4 | 9 | 8 | 1 | 0 | voice | `test_family01_f0.py`; reviewed F0 and mixed execution |
| F02 voice quality | 8 | 20 | 13 | 7 | 0 | voice | `test_family02_voice_quality.py`; shared voice/mixed path |
| F03 laryngeal biomechanics | 8 | 0 | 0 | 0 | 0 | none | Blocked construct metadata tests; no executable source method |
| F04 formants/vowel space | 12 | 13 | 12 | 1 | 2 | formants | `test_family04_formants.py`; aligned formants and mixed tests |
| F05 formant dynamics | 2 | 0 | 0 | 0 | 4 | none | Internal algorithm tests only; no approved concrete target IDs |
| F06 segmental contrasts | 8 | 6 | 3 | 3 | 3 | segmental | `test_family06_segmental.py`; matched target and mixed tests |
| F07 duration | 5 | 6 | 6 | 0 | 1 | timing | `test_family07_duration.py`; reviewed duration and mixed tests |
| F08 rate | 2 | 3 | 3 | 0 | 0 | timing | `test_family08_rate.py`; missing-prompt mixed isolation |
| F09 pause/phrase | 8 | 11 | 10 | 1 | 0 | timing | `test_family09_pause.py`; mixed pause tests |
| F10 DDK | 2 | 2 | 2 | 0 | 0 | ddk | `test_family10_ddk.py`; successful reviewed-DDK + F12 mixed test added |
| F11 amplitude | 5 | 9 | 9 | 0 | 0 | spectrum | `test_family11_amplitude.py`; reviewed-audio and mixed normalization tests |
| F12 spectral | 7 | 49 | 49 | 0 | 1 | spectrum | `test_family12_spectral.py`; reviewed-audio and mixed tests |
| F13 nonlinear/composites | 8 | 8 | 0 | 8 | 8 | none | Closure tests; no source-complete executable leaf |
| **Total** | **79** | **136** | **115** | **21** | **19** | **6 groups** | Formula/synthetic tests are engineering tests |

Registry checks found no duplicate ID, implemented leaf without a group, multiply mapped family, availability/selectability mismatch, or selectable template. All 115 implemented leaves carry nonempty evidence, unit, analysis unit, algorithm version and parameter-set ID. The Range field is `—` for 78 implemented leaves because the *ingested registry metadata* has no numerical bound for them. This is an explicit unspecified range, not a numeric QC rule; source-document range reconciliation remains a documentation review item, and no bound was invented during the audit. Explicit exact-task/task-type recommendation remains advisory and independent of selection. Family 03 construct evidence is LIMITED; Family 13 unavailable leaves retain their evidence. The implementation inventory still enumerates every concrete and template identity.

## Mixed-family execution and outputs

The dispatcher resolves selected IDs from the approved registry, groups F01/F02 as `voice`, F04 as `formants`, F06 as `segmental`, F07/F08/F09 as `timing`, F10 as `ddk`, and F11/F12 as `spectrum`. It invokes each selected group once, rejects unselected executor rows, checks duplicate recording/feature keys and emits one authoritative `acoustic_features_per_file.csv`, `acoustic_feature_status_long.csv`, selected registry snapshot and run manifest. Token-level F04 measurements remain in native long-form audit tables. Family failures yield local NaN/failure rows; unrelated groups continue. Family-specific native artifacts and manifests remain under `family_runs/<group>`.

The new synthetic/reviewed fixture demonstrates **five frozen DDK events** yielding `ddk_rate_syll_s=2.5` and `ddk_cycle_mad_s=0`, alongside finite `mfcc01_mean`, in one selected-only mixed run with no duplicate rows. A second ten-family selection covers F01/F02/F04/F06/F07/F08/F09/F10/F11/F12 in all six groups; absent alignment/target/prompt inputs fail locally while DDK, timing and spectrum complete. F01 voice tracking can legitimately fail on short DDK events (`analysis_region_too_short`); task recommendations do not block manual selection.

## Upstream and task workflow

Active feature executors consume frozen reviewed segmentation, canonical/preprocessed analysis audio and, when required, frozen Alignment. F04 and F06 use the shared Alignment loader; F07 word/vowel outputs use frozen Alignment while its speech/task duration outputs use reviewed timing. F10 consumes reviewed DDK events only. F06's relevant outputs require approved target and validated acoustic sub-event annotations in addition to Alignment. Active mixed executors do not call MFA, parse TextGrid, rerun Silero, or write to frozen segmentation/Alignment. Dormant historical plugin files still refer to raw Silero tables, but cannot be selected through the approved 79-construct path; they are a maintenance distinction, not evidence of active use.

| Task | GUI path and present limit |
|---|---|
| Sustained phonation | Setup → optional Preprocess → Segmentation/Review → Acoustic Features. Linguistic MFA Alignment is not required; stable phonation region remains a feature-specific prerequisite. |
| Bamboo Passage | The lab-supplied canonical `bamboo_passage_v1` prompt resolves automatically in task-first Alignment. Word/phone outputs still require frozen reviewed segmentation, speaker identity and Alignment. Family 08 counts remain a separate scientific prerequisite. |
| DDK | Dedicated DDK segmentation and Manual Review → F10; no MFA dependence. Other implemented features remain manually selectable, subject to their own technical prerequisites. |
| WSTG | The one registered item `wstg_we_see_three_geese` v1 uses “We see three geese.” Task-first Alignment can resolve that item if speaker, reviewed segmentation and MFA resources exist. Remaining stimulus IDs/text/versions are unknown. |

Setup stores explicit task type; recommendation is advisory. The GUI's normal Alignment fields resolve task, prompt and speaker from the project; engineering paths are under Advanced. MFA environment checks, self-test and alignment run on a Qt worker thread. The Alignment viewer presents waveform, reviewed domain, words and phones for inspection before freezing. Quality Control is visible but disabled and does not gate Acoustic Features. `Run Acoustic Features` accepts mixed implemented selections and reports per-output technical failures. The post-review navigation opens Alignment even for tasks that do not need it, but explicitly labels Alignment optional; it does not enforce an MFA gate.

**Workflow resolution:** The shortcut now reads “Run to Manual Review,” which accurately describes its stopping point. “Continue after Review” routes to Alignment only when the task supports linguistic Alignment and a selected output requires it; otherwise it opens Acoustic Features. Review and Alignment freeze remain human gates. The remaining normal feature input browsers are Family 08 approved prompt counts, Family 04 vowel-category mapping, Family 06 target definitions, and Family 06 validated acoustic sub-event annotations. These are specialized scientific inputs that cannot be inferred safely from a filename or task name. Engineering Alignment paths remain under Advanced.

**Bamboo count finding:** The exact canonical text has 97 whitespace-delimited orthographic tokens. The already-frozen `vslp_prompt_nfkc_v1` Alignment normalizer splits the hyphen in `good-looking`, producing 98 MFA transcript words. Family 08 does not use either count automatically: it requires a versioned approved count manifest containing a `word_count` and/or `syllable_count`. The legacy `configs/acoustic_task_word_counts.yaml` and `configs/default_acoustic.yaml` state 99, but neither supplies the active Family 08 count contract. No authoritative general Family 08 rule for hyphenated words, contractions or syllables has been supplied. Its Bamboo numerator remains `PROMPT_COUNT_DEFINITION_REQUIRED`; the historic 99 is not justified by this text. Alignment's 98 is for matching MFA transcript tokens and is not a rate numerator by implication.

**Data history resolution:** After explicit approval, 78 path identities were removed with `git-filter-repo` in two passes; all affected remote branches and tags were updated atomically with explicit leases. The active index and a fresh remote clone have zero purge-path and participant-like identifier matches. No original local research file was deleted. External clones/caches cannot be certified by this repository check.

## Provenance and test quality

The unified long status table retains recording/file, task ID/type, exact feature ID, family, value, unit, status, failure reason, algorithm/version, parameter set and family-manifest reference. The run manifest records selected IDs, groups/families, success/failure counts, prerequisite definitions, parameter IDs, reviewed-segmentation hashes and frozen-Alignment hashes where present. Family audits add native-region, preprocessing, target/annotation, voice-track, timing, DDK-event, formant-token/frame and MFCC provenance as applicable. F04 token/vowel IDs are deliberately not coerced to one recording scalar.

Every implemented family has direct unit tests; integration is strongest for F01, F04, F07, F10, F11/F12 and selected mixed combinations. The ten-family fixture is a dispatch/prerequisite isolation test, not proof that all 115 outputs succeed on one task. Qt worker wiring is visible in code, but there is no automated live-GUI event-loop responsiveness stress test during a long MFA command; the real WSTG opt-in test exercises the workflow function rather than clicking the GUI. Exact formula, synthetic and integration tests demonstrate **implementation behavior**, not source-package parity, ALS discrimination, longitudinal sensitivity, device robustness or clinical validity. No manual gold-standard boundary or clinical measurement dataset was supplied for this audit.

## Release issues: one primary category each

| Item | Category | Required next action |
|---|---|---|
| Formerly tracked participant-like audio, metadata, generated outputs, plots and notebooks | `NO RELEASE BLOCKER` for controlled Git refs | Sanitized after explicit approval; keep private backups outside Git and prevent old clones from repushing old history. |
| Family 08 Bamboo word and syllable counts | `SCIENTIFIC DEFINITION REQUIRED` | Lab freezes versioned orthographic counting and authoritative syllable count; do not infer from Alignment tokenization or legacy 99. |
| Remaining WSTG inventory | `AUTHORITATIVE TASK CONTENT REQUIRED` | Lab supplies exact remaining stimulus IDs, text and versions. The known item remains usable. |
| 19 unresolved templates and source/transform-incomplete exact leaves | `SCIENTIFIC DEFINITION REQUIRED` | Freeze only approved source, target and factor contracts; keep unavailable meanwhile. |
| Eight Family 03 proprietary constructs and Aural Analytics articulatory precision | `PROPRIETARY / UNREPRODUCIBLE` | Remain visibly unavailable; no substitute. |
| Real external Conda MFA single-task execution | `NO RELEASE BLOCKER` | Cross-environment self-test and task-first WSTG engineering alignment passed in this audit. |
| Real two-speaker corpus-to-feature fixture | `ENGINEERING TEST GAP` | Opt-in test remains skipped because no explicitly approved nonclinical multi-speaker fixture/profile paths were supplied. |
| Long-running MFA GUI responsiveness under live interaction | `ENGINEERING TEST GAP` | Worker-thread dispatch is code-reviewed; add a live event-loop/cancellation stress check before a broad GUI release claim. |
| Formula/source parity and clinical/device validation | `EMPIRICAL VALIDATION REQUIRED` | Separate controlled reference and clinical protocol. |
| Registry uniqueness, mixed dispatch, successful DDK fixture, task recommendation separation and disabled QC gate | `NO RELEASE BLOCKER` | Preserve contracts and tests. |

## Validation

Validation after the Bamboo/workflow pass: `pytest tests/unit` **344 passed**; `pytest tests/integration` **71 passed, 4 skipped**; Ruff **passed**; `git diff --check` **passed**. This pass reran the opt-in real external-Conda MFA engineering self-test: **1 passed** in 236 seconds. The prior release-audit pass also ran task-first WSTG engineering alignment successfully. These are engineering checks, not boundary-accuracy or clinical validation. No commit is part of this pass.
