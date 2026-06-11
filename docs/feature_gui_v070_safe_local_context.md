# Feature GUI v0.70 - Safe local metadata context

This patch is rebuilt from stable v0.69.

## Design rule
The base analysis must run even with no metadata, incomplete metadata, unfamiliar metadata, or malformed clinical columns. Optional metadata must not mutate the base analysis table or control the whole GUI.

## What changed
- Adds local Task Review clinical-context controls only.
- Adds local detection for diagnosis, ALSFRS total, ALSFRS bulbar, ALSBDI, sex/gender, session/visit, and iteration.
- Adds project ALSFRS bulbar bins:
  - near_normal_11_12
  - mild_9_10
  - moderate_6_8
  - severe_0_5
  - invalid_above_12
  - invalid_below_0
- Adds Task x clinical context and Task counts within clinical value plots.
- Adds Context detection and Severity presets tables in Task Review.

## What did not change
- No global context bar.
- No global filtering.
- No modification of `analysis_df` during Load/Run Feature Analysis.
- No new clinical-context backend dependency in the run-analysis path.
- No warning-handling changes.
