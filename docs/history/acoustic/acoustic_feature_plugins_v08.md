# VSLP Acoustic Feature Plugins v0.8

This update combines the planned v0.7 and v0.8 work:

- v0.7: modular feature plugin architecture
- v0.8: additional implemented feature families

## Plugin location

Feature plugins live here:

```text
src/vslp/acoustic/features/plugins/
```

Current plugins:

```text
base.py          shared plugin interfaces
 timing.py       respiratory/timing features from Silero segment tables
 phonatory.py    F0 and CPP engineering proxies from canonical WAVs
 rhythm.py       envelope modulation / rhythm features
 resonatory.py   global RMS amplitude baseline
```

## Current implemented or proxy-computed features

Timing/respiratory:

```text
speech_rate
total_dur
speech_dur
percent_pause
num_pause
mean_pause_dur
mean_phrase_dur
cv_pause_dur
cv_phrase_dur
total_pause_dur
```

Phonatory engineering proxies:

```text
f0_mean
f0_std
CPP_mean
```

Rhythm/envelope:

```text
intensity_CV
fft_peaks1
fft_peaks2
fft_ampli1
fft_ampli2
nrj_below_boundary
nrj_above_boundary
nrj_3_6
ratio_below_above
```

Resonatory/intensity baseline:

```text
RMSamp
```

## Important validation note

The phonatory values are currently local engineering proxies. They should be used for
pipeline testing and exploratory research only until validated against the uploaded
reference notebook and/or a Praat-style implementation. The feature status table marks
these as `computed_proxy`.

## How to add a feature

1. Add or confirm the feature in:

```text
src/vslp/acoustic/features/registry.py
```

2. Add the formula to an existing plugin or create a new plugin in:

```text
src/vslp/acoustic/features/plugins/
```

3. Add the plugin to:

```text
src/vslp/acoustic/features/plugins/__init__.py
```

4. Add a unit test.

5. Run:

```bash
pytest
```

6. Run the GUI and inspect:

```text
acoustic/004_features/tables/acoustic_features_per_file.csv
acoustic/004_features/tables/acoustic_feature_status_long.csv
acoustic/004_features/reports/acoustic_feature_report.html
```

## Feature status values

- `computed`: implemented feature computed normally.
- `computed_proxy`: computed engineering proxy; needs formula/reference validation before clinical interpretation.
- `failed`: attempted but failed for this file.
- `not_implemented_yet`: feature is registered but no validated plugin implementation exists yet.
- `not_selected_or_missing_input`: plugin exists, but required input was missing or feature was not selected.
