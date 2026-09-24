# VSLP Acoustic Pipeline release candidate readiness

Assessment of the current `september-review` working tree, 2026-09-24. This is a release decision record, not clinical validation. No scientific algorithm, feature definition, Git history, or tracked data was changed for this assessment. The working tree also contains earlier uncommitted release-audit changes.

## Decision

**Privacy cleanup is complete for the Git refs controlled here.** The user approved conservative removal of 77 potentially participant-derived current files and one additional historical path alias. The active index, all advertised remote branches and tags, and a fresh clone of their reachable history contain none of the 78 path identities. The original local research files and pre-rewrite Git bundles are retained outside the distributable repository; see the [data-hygiene audit](acoustic_data_hygiene_audit.md). External clones and service caches remain outside this verification. Release freezing still requires the separate scientific/task-content decisions below.

The Family 08 Bamboo prompt-count definition is also not frozen. The canonical lab-supplied text has 97 whitespace tokens; Alignment's existing hyphen-splitting transcript normalization yields 98 MFA words; old configs say 99. The active Family 08 executor requires an approved versioned word/syllable-count manifest and consumes neither the Alignment count nor the old 99 configs automatically. The lab must freeze the counting convention and authoritative syllable count before claiming fully automatic Bamboo rate outputs.

## Scientific inventory

The active catalog has **79 constructs** and 138 output records with unique IDs. Two output records are unresolved placeholders, leaving **136 concrete public leaves**: **115 implemented and selectable**, **21 exact not implemented and nonselectable**. The full inventory has **19 unresolved templates** (the two placeholders plus 17 construct-level templates), all nonselectable. Family 13 has no executable leaf. These totals are checked against the registry and [row-level inventory](acoustic_feature_implementation_inventory.csv); no unavailable output was enabled.

## Supported task content and workflow

| Task | Current path/content | Limit |
|---|---|---|
| Sustained phonation | Setup → optional Preprocess → Segmentation → Manual Review → Acoustic Features | Linguistic MFA is not required; feature-specific stable-region prerequisites still apply. |
| DDK | Setup → optional Preprocess → dedicated Segmentation/Review of DDK events → Acoustic Features | F10 reads reviewed DDK events, not MFA. |
| Bamboo Passage | The lab-supplied `bamboo_passage_v1` resolves automatically from the task registry; Alignment is used when selected outputs require words/phones. | Family 08 approved word/syllable counts remain outstanding. |
| WSTG | `wstg_we_see_three_geese` v1 (“We see three geese.”) resolves through task-first Alignment. | Other WSTG stimulus IDs/text/versions have not been supplied; no additional sentence is inferred. |

The GUI order is Setup → optional Preprocess → Segmentation → Segmentation Manual Review → Alignment where needed → Acoustic Features → outputs. “Run to Manual Review” accurately stops at the human gate. “Continue after Review” routes to Alignment only for selected linguistic-boundary features on an alignment-applicable task; otherwise it opens Acoustic Features. The scientist inspects and freezes Alignment before consuming it. Quality Control remains disabled and does not gate features. Active executors use frozen reviewed segmentation or Alignment, not raw Silero output or direct MFA/TextGrid parsing. Recommendations remain advisory; technical prerequisites produce local feature failures. The mixed dispatcher produces selected-only results and preserves unrelated family results when one prerequisite fails.

## Specialized scientific inputs

The normal Acoustic Features view shows these controls only for relevant selected leaves, using registry prerequisite flags. They are not generated from task names or filenames:

| Input | Visibility and meaning |
|---|---|
| Family 08 approved prompt counts | Visible when a selected output requires a prompt manifest; the placeholder says approved task word/syllable counts are required. |
| Family 04 vowel-category mapping | Visible when a selected output requires a versioned vowel-category map; mapping must be scientifically approved. |
| Family 06 target definitions | Visible for selectable F06 leaves; all current executable F06 outputs require an approved target manifest. |
| Family 06 validated acoustic sub-events | Visible for selectable F06 leaves; all current executable F06 outputs require validated burst/noise annotations. |

Missing inputs are reported as current-run prerequisites. Unimplemented leaves and unresolved templates remain nonselectable. These browsers are specialized scientific inputs, not normal Alignment engineering configuration.

## Output and provenance review

One mixed run writes `acoustic_features_per_file.csv`, `acoustic_feature_status_long.csv`, `selected_acoustic_feature_registry.csv`, and `stage_manifest.json`, plus family-specific audits under `family_runs/<executor_group>`. The manifest records selected IDs, families and groups invoked, family manifests, parameter-set IDs, success/failure counts, prerequisites, frozen segmentation hashes, and frozen Alignment word/phone hashes and run ID where available. Long status rows carry recording, task, feature, family, value, unit, status/failure reason, algorithm/version and parameter set; family audits retain region, preprocessing, target, and event provenance where applicable. This is an engineering reproducibility contract, not proof of clinical validity.

## Blockers and limitations

1. **Privacy cleanup:** the 78 reviewed path identities are absent from controlled reachable Git history and current tracking. Collaborators must not repush pre-rewrite commits.
2. **Scientific-definition requirement for Bamboo rates:** approved orthographic word-count convention and authoritative syllable count, versioned against `bamboo_passage_v1`.
3. **Task-content scope:** remaining WSTG inventory requires authoritative lab text and versions. The known WSTG item remains usable.
4. **Intentionally unavailable:** 21 exact leaves and 19 templates remain nonselectable by design; see the implementation inventory and [known limitations](release_known_limitations.md).
5. **Empirical work remains:** manual boundary-reference accuracy, MFA feature sensitivity, source-package parity where applicable, ALS/clinical validity, longitudinal sensitivity, and device robustness. Passing synthetic and software integration tests does not establish these properties.

## Validation

Default test suites: `pytest tests/unit` **345 passed**; `pytest tests/integration` **71 passed, 4 skipped**. Ruff **passed** and `git diff --check` **passed**. Two opt-in real MFA tests also **passed** (external-Conda self-test and task-first WSTG alignment; 433 seconds). These use nonclinical engineering audio and do not establish boundary accuracy or clinical validity. The other opt-in scenarios require separate fixture/flags and were not run. No commit or tag was created.
