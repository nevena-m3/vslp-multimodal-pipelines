# Acoustic Features registry contract

The active registry uses only the supplied Family Index and 79-feature master
matrix. It contains 13 families and exactly 79 constructs. The master
documents do not supply exact FEATURE ID(S). The subsequently supplied Family
08 specification provides **three approved leaf outputs** under C048/C049.
Family 09 provides **eleven exact leaf IDs** under C050–C057; its frozen
pause-pattern factor remains visible but cannot execute without its source
transform. Family 09 uses the final reviewed segmentation and a shared
pause/phrase event table. No old pause aliases are reactivated.
Family 01 now registers eight reproducible exact outputs under C001-C004 using one
native-rate Praat autocorrelation track. The two intonation components are emitted;
the factor-score ID remains visible but unavailable because the source
standardization/factor transform is not frozen. Family 02 registers thirteen reproducible Praat PointProcess,
Harmonicity, and DFP outputs under C005-C007/C011. GNE, PVI, CPP/CPPS, and the
H1-H8 source-parity outputs remain visible but unavailable because their complete
source parameterization or scaling is not supplied.

Family 07 registers six executable scalar outputs under C043-C045/C047. The two
reviewed-timing outputs reuse the same >=300 ms pause and manual-contamination
semantics as Families 08/09. Word and vowel-variability calculations consume an
explicit validated alignment CSV and return `missing_alignment` when it is absent;
the feature calculator never launches an aligner. The source ID pattern
`vowel_duration_<phone>_s` remains visible but unavailable until a target-phone
inventory supplies concrete leaf IDs.

Family 10 provides **two exact leaf IDs** under C058/C059. Its calculation
uses only final reviewed DDK events and writes a separate event/cycle audit.
Other constructs cannot be selected or executed as scalar features.
Previously exposed IDs from the old catalog and earlier examples are removed
from the active registry. The feature stage rejects unapproved or legacy IDs.

The 31 removed legacy-only active-catalog leaves are:

`f0_mean_hz`, `f0_median_hz`, `f0_sd_st`, `f0_iqr_st`, `f0_range_st`,
`jitter_local_pct`, `jitter_absolute_s`, `jitter_rap_pct`,
`jitter_ppq5_pct`, `jitter_ddp_pct`, `localShimmer`, `localdbShimmer`,
`apq3Shimmer`, `apq5Shimmer`, `apq11Shimmer`, `HNR`, `CPP_mean`,
`f2`, `f1`, `f3`, `speech_dur`, `speech_rate`,
`total_pause_dur`, `cv_pause_dur`,
`cv_phrase_dur`, `num_pause`, `percent_pause`, `mean_pause_dur`,
`mean_phrase_dur`, `DDKrate`, `DDKregularity`.

The other 67 constructs currently lack exact family-spec leaf IDs.

Construct code names display as “—”; implemented leaves always display their
exact registered feature IDs. Evidence remains “—” until a detailed family
specification provides it, then propagates from the construct to its leaves.
Families 08 and 09 supply exact IDs and evidence for their outputs.
Use retains the matrix categories internally: PROD, VARIANT,
TASK, VALIDATE, RESEARCH, BLOCKED. The matrix's task flags are the sole source
of task fit. The exact registered Setup task names are Sustained /a/, Bamboo
Passage, DDK, and WSTG. No sentence, passage, or other semantic task class is
inferred. An unmatched Setup name has no recommendations.

For each future family document, add exact output leaves under the existing
construct IDs, preserve supplied FEATURE ID(S), evidence counts, units, ranges,
prerequisites and frozen parameters, then implement and validate that family.
The family/construct IDs remain stable as outputs are added. Do this one family
at a time.

The current upstream pipeline contract still controls audio provenance,
optional preprocessing, and frozen reviewed segmentation. The master matrix's
older 48 kHz branch, segmentation settings and feature-local DDK detection do
not override that contract.

Setup records an explicitly selected `task_type` separately from exact
`task_id` and `task_display_name`. Older runs with no type display “Not
specified”; no type is inferred from the task name. Task recommendation is a
graphical green/amber/gray dot derived from explicit source mapping.
Availability is only “Implemented” or “Not implemented” and depends on a
registered exact executable leaf, not the selected task or this run's inputs.
Current-run prerequisites appear separately in the detail panel and feature
status output. Implemented features remain selectable outside their recommended
tasks; a versioned task-matched prompt count is still required to compute rate
values. The Family 10 code remains in local review and is not yet registered
as a GUI-available executor.
