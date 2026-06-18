# VSLP v0.35 - Feature Extraction Completion Pass

This release completes the current acoustic Feature Extraction block by making feature-computation modes explicit and auditable.

## Key design principle

Feature Extraction produces one file-level scalar per selected feature, but each scalar must be traceable to its native physiological measurement scale.

The scalar is not a generic average. It is produced from a family-specific computation policy:

- timing/respiratory: segment-event summaries;
- rhythm/EMS: effective-task envelope and modulation spectrum;
- phonatory: voiced-frame/period support;
- articulatory/formant: valid LPC formant trajectories;
- resonatory/nasality: valid spectral frames;
- coordination: aligned CPP/F1/F2 trajectories and lagged eigenspectrum summaries.

## GUI changes

The Feature Extraction tab now includes a computation-mode selector:

- validated_default
- speech_only_concatenated
- effective_task_with_pauses
- per_segment_robust
- full_file_exploratory

Validated default is recommended. Other modes are applied only where scientifically appropriate and are documented in the scalar-reduction audit output.

## New output table

```text
acoustic/004_features/tables/acoustic_feature_scalar_reduction_audit.csv
```

This table records, for every selected feature:

- requested_global_mode
- applied_family_mode
- analysis_region
- native_measurement_scale
- file_level_scalar_reduction
- scientific_warning

## Important limitation

Per-segment robust reduction is now represented as a computation-policy option, but signal-family per-segment native track persistence is still planned. Timing features already preserve segment events. The next major architecture step would be persistence of native phonatory/formant/resonatory/coordination tracks.
