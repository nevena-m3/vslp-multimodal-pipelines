# VSLP v0.32 Multiscale Feature Aggregation

This update changes the interpretation of aggregation.

Feature extraction should preserve the physiologic scale of a measurement as long as possible. A single file-level scalar is useful for simple ML tables, but it is not always the native feature scale.

Examples:

- Timing features originate from speech and pause segment events.
- Phonatory features originate from voiced frames or pseudo-period tracks.
- Formant features originate from valid LPC frame trajectories.
- Rhythm features originate from an effective-task amplitude envelope and modulation spectrum.
- Resonatory/nasality features originate from valid spectral frames and vowel/task-dependent peak contrasts.
- Coordination features originate from aligned multitrajectory windows and lagged correlation matrices.

## New outputs

Feature extraction now writes:

```text
acoustic/004_features/tables/acoustic_feature_measurement_scale_registry.csv
acoustic/004_features/tables/native_measurements/acoustic_native_segment_events.csv
```

Aggregation now writes:

```text
acoustic/005_aggregation/tables/acoustic_features_aggregated.csv
acoustic/005_aggregation/tables/acoustic_features_aggregated_multistat.csv
acoustic/005_aggregation/tables/acoustic_aggregation_strategy.csv
```

The compact table preserves the previous interface. The multi-stat table writes, for each feature and aggregation group:

```text
mean
median
sd
iqr
q05
q25
q75
q95
min
max
n
missing_fraction
```

This avoids forcing every downstream model to use only mean or median.

## Current limitation

Native segment events are now persisted. Frame-level F0/CPP/formant/spectral/coordination tracks are documented in the scale registry and should be persisted in the next architecture extension. Until then, file-level scalar features remain the primary feature table, but aggregation is now explicit about the native scale and recommended reducers.
