# Acoustic Features registry contract

The active registry uses only the supplied Family Index and 79-feature master
matrix. It contains 13 families and exactly 79 constructs. The master
documents do not supply exact FEATURE ID(S). The subsequently supplied Family
08 specification provides **three approved leaf outputs** under C048/C049.
Family 09 provides **eleven exact leaf IDs** under C050–C057; its frozen
pause-pattern factor remains visible but cannot execute without its source
transform. Family 09 uses the final reviewed segmentation and a shared
pause/phrase event table. No old pause aliases are reactivated.
Other constructs cannot be selected or executed as scalar features.
Previously exposed IDs from the old catalog and earlier examples are removed
from the active registry. The feature stage rejects unapproved or legacy IDs.

The 32 removed active-catalog leaves are:

`f0_mean_hz`, `f0_median_hz`, `f0_sd_st`, `f0_iqr_st`, `f0_range_st`,
`jitter_local_pct`, `jitter_absolute_s`, `jitter_rap_pct`,
`jitter_ppq5_pct`, `jitter_ddp_pct`, `localShimmer`, `localdbShimmer`,
`apq3Shimmer`, `apq5Shimmer`, `apq11Shimmer`, `HNR`, `CPP_mean`,
`f2`, `f1`, `f3`, `speech_dur`, `speech_rate`,
`articulation_rate_syll_s`, `total_pause_dur`, `cv_pause_dur`,
`cv_phrase_dur`, `num_pause`, `percent_pause`, `mean_pause_dur`,
`mean_phrase_dur`, `DDKrate`, `DDKregularity`.

The other 69 constructs currently lack exact family-spec leaf IDs.

Code name and Evidence display as “—” until an individual family specification
provides them. Family 08 supplies exact IDs and evidence for its three outputs.
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
