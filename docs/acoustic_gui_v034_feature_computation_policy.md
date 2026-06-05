# VSLP v0.34 - Feature Computation Policy

This update clarifies how a single audio file is converted into one scalar value per selected acoustic feature.

The scalar feature table remains one row per file, but the GUI and backend now explicitly document the native measurement scale and the reduction rule for each feature family. This prevents the misleading assumption that all features are computed over the same signal region or reduced by a generic mean/median.

## Policy by feature family

| Family | Best tasks | Native scale | Default region | File-level scalar reduction |
|---|---|---|---|---|
| Timing / respiratory | connected speech, passage, reading, free speech | speech/pause segment events | Silero segment table | sums, counts, percent of effective task, mean/CV of event distributions |
| Rhythm / EMS | connected speech | amplitude envelope and modulation spectrum | first speech onset to last speech offset, internal pauses preserved | modulation peaks and normalized band powers |
| Phonatory | sustained vowel; voiced connected speech with caution | voiced frames / period support | speech-only with voiced-frame filtering | F0 summaries, perturbation formulas, CPP/HNR summaries, voice-break count |
| Articulatory / formant | vowels, sentences, passage | valid LPC formant trajectories | valid speech frames | robust medians, 5-95 ranges, slope percentiles |
| Resonatory / nasality | controlled sentences / vowel-targeted frames | valid spectral frames | valid spectral speech frames | median spectral contrasts and support variables with warnings |
| Coordination | connected speech with stable CPP/F1/F2 tracks | aligned multitrajectory windows | effective-task trajectories | lagged correlation eigenspectrum summary |

## New output

Feature extraction now writes:

```text
acoustic/004_features/tables/acoustic_feature_computation_policy.csv
```

This table documents feature-level policy with task scope, default region, native measurements, scalar reduction, interpretation rationale, and operations to avoid.

## GUI change

The Feature Extraction tab now shows a compact computation-policy table instead of a paragraph block. The previous explanatory paragraph about the removed aggregation tab was removed from the GUI.
