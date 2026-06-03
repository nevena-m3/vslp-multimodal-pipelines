# VSLP Developer Guide: Acoustic Features

VSLP treats features as registered, auditable units. A feature should not be added as an anonymous column in a notebook-style script. It should be added through the feature registry and a validated implementation.

## Where feature definitions live

Feature metadata is in:

```text
src/vslp/acoustic/features/registry.py
```

The feature registry describes:

```text
feature
subsystem
formula_or_definition
units
implementation_status
notes
```

The extraction stage is in:

```text
src/vslp/acoustic/features/stage.py
```

## Correct workflow for adding a feature

1. Add or update the feature entry in `registry.py`.
2. Implement the calculation in a small function in `stage.py` or a dedicated plugin module.
3. Add the feature name to `IMPLEMENTED_FEATURES` only after it is validated.
4. Add a test with a synthetic or known-answer signal.
5. Confirm the feature appears in:

```text
acoustic/004_features/tables/acoustic_features_per_file.csv
acoustic/004_features/tables/acoustic_feature_status_long.csv
```

## Rules

- Do not silently compute approximate clinical features.
- If a formula is uncertain, leave the feature as `NaN` and mark it `not_implemented_yet`.
- Every feature must have clear units.
- Every feature must define whether it is computed per file, per segment, per task, or per subject/session.
- Features relying on task metadata must fail gracefully when metadata is missing.

## Current implemented family

The first implemented family is timing/respiratory features derived from Silero speech/non-speech segments.

Implemented now:

```text
total_dur
speech_dur
percent_pause
num_pause
mean_pause_dur
mean_phrase_dur
cv_pause_dur
cv_phrase_dur
total_pause_dur
speech_rate
```

`speech_rate` requires task word count metadata. If no word count is available, it remains `NaN`.

## Future plugin structure

As the feature library grows, implementations should be split by subsystem:

```text
src/vslp/acoustic/features/plugins/
  timing.py
  phonatory.py
  articulatory.py
  resonatory.py
  rhythm.py
  coordination.py
```

Each plugin should expose a function that accepts validated stage inputs and returns a dictionary of feature values.
