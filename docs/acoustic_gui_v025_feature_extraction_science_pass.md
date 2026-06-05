# VSLP v0.25 Feature Extraction Science Pass

This update refines the Acoustic Feature Extraction block using the uploaded Bamboo Acoustic Feature Map and ALS Speech Biomarkers Map.

## Main design changes

- The feature registry now carries subsystem, task scope, evidence tier, formula/definition, implementation note, expected orientation range, and ALS-direction notes.
- Feature detail panels show formula and task compatibility, not only feature names.
- The backend continues to separate implemented, proxy, and pending features.
- Expected-range flags are descriptive screening aids only. They are not clinical cutoffs and must be re-derived for each dataset/control cohort.

## New outputs

- `acoustic_feature_distribution_audit.csv`
- `acoustic_feature_expected_range_flags.csv`
- `feature_distribution_audit.png`
- `feature_expected_range_flags.png`
- `feature_correlation_heatmap.png`
- `feature_subsystem_distributions.png`

## Interpretation

Flagged distributions mean “review this feature and its upstream QC,” not “exclude this feature.”

Common reasons for flags:

- wrong task type for feature
- segmentation failure or region-policy mismatch
- poor SNR / clipping / device artifact
- too-small sample size
- expected range not suitable for the current cohort
- proxy feature not yet validated

## Current limitation

The registry is scientifically richer, but this update does not fully implement every pending formula. The next step is subsystem-by-subsystem formula validation.
