# Family 10 reviewed DDK feature contract

Family 10 exposes only `ddk_rate_syll_s` (C058) and `ddk_cycle_mad_s`
(C059). Both consume the frozen authoritative segmentation decisions and
intervals from `003_segmentation_review/final/`. No detector, waveform
filter, envelope, or peak search runs in Acoustic Features.

The frozen `family10_ddk_reviewed_v1` profile uses the midpoint of each
final reviewed DDK event interval as its cycle timestamp. This matches the
review stage's midpoint timing for edited DDK intervals and gives automatic
and manual events one consistent final-layer landmark. The rate denominator
is the reviewed analysis-window duration minus the duration of manually
excluded contamination within that window. This is the user's explicit
choice because the family document says only “analyzed DDK duration” and
does not fix its endpoints. Rate is valid event count divided by that
effective duration in seconds.

Manual contamination splits the DDK event train into separate sequences.
Adjacent cycle intervals are calculated within a sequence only. Temporal
variability is the unnormalized mean of absolute differences between
consecutive cycle intervals within each sequence; differences are pooled
across sequences and stored in seconds. A manually excluded span is never
a cycle. If an exclusion splits one original DDK event into two boundary
fragments, both fragments are marked unavailable for feature counting.
At least five valid final events are required for production values. Cycle
MAD additionally needs at least two adjacent cycle intervals within a
single continuous sequence; otherwise it is NaN with an explicit reason.

`acoustic/005_features/tables/native_measurements/ddk_feature_events.csv`
records event index, sequence ID, reviewed onset/offset, midpoint timestamp,
next within-sequence cycle interval, cycle interval index, boundary source,
reviewer, exclusion status/reason, and automatic source interval ID where
traceable. The per-file feature table stores raw analysis duration,
excluded-contamination duration, effective duration, valid event count,
valid sequence count, final segmentation run IDs and table hashes. The long
status table stores exact feature ID, value, unit, algorithm/profile,
analysis region, status, and failure reason.

The family document's 48 kHz master branch and band-pass/envelope/find-peaks
detector guidance predates the current upstream architecture. The current
native-rate audio and reviewed DDK segmentation contract takes precedence.
The document's recommended tasks include individual `/pa`, `/ta`, `/ka`, and
`/pataka` labels. The current explicit Setup task registry contains only
`DDK`, so Family 10 is recommended for that registered task; no syllable task
is inferred from a filename or free-text label. The specification requires
hand-label validation for stable metrics. Passing synthetic and pipeline
contract tests is not a clinical validation claim.
