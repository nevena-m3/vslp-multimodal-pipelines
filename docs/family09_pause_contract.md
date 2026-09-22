# Family 09 pause and phrasing contract

Family 09 consumes only frozen, authoritative reviewed decisions and intervals.
It never reruns Silero or changes the canonical audio. The shared event builder
also supplies Family 08's articulation-rate pause duration.

An eligible pause is an internal nonspeech interval of at least 300 ms between
adjacent patient-speech regions. Leading/trailing silence, content outside the
reviewed analysis window, and manually excluded contamination are never pause
events. A phrase is a patient-speech group between eligible pauses; short gaps
inside a phrase stay in its elapsed phrase span. Contamination terminates a
phrase without becoming a physiological pause. The shared event table at
`acoustic/005_features/tables/native_measurements/pause_phrase_events.csv`
contains pause, subthreshold internal gap, excluded contamination, and phrase
rows with exact start/end times, durations, source role, boundary source, and
the frozen parameter-set ID.

The frozen profile `family09_bamboo_reviewed_v1` uses a 300 ms minimum internal
pause and sample SD (`ddof=1`) for coefficient of variation. The leaf IDs are
`total_pause_duration_s`, `cv_pause_duration`, `cv_pause_duration_pct`,
`cv_phrase_duration`, `cv_phrase_duration_pct`, `pause_count`,
`percent_pause_ge300ms`, `mean_pause_duration_s`, `mean_phrase_duration_s`,
`pause_pattern_components`, and `pause_pattern_factor_if_frozen`.
The factor output cannot execute because the source standardization and frozen
loadings were not supplied. Components are a research-only JSON vector.

No pauses yield zero for total duration, count, and percent pause. Mean pause
duration and pause CV are undefined and receive NaN plus a machine-readable
reason. CV requires at least two eligible events. The per-file table retains
raw duration, reviewed analysis bounds, excluded-contamination duration,
pause/phrase event counts, segmentation/review run IDs, and final-table hashes.
The long status table records feature-specific units, algorithm version,
parameter-set ID, analysis region, status, and failure reason.

The Family 09 document's global 48 kHz branch, 50 ms Silero padding, and
200 ms inward-boundary guard predate the current reviewed native-rate pipeline.
Those upstream settings are not applied here. The detailed Family 09 task
scope marks Bamboo Passage only; the broader master-matrix WSTG cell is not
used for Family 09 recommendation or execution.
