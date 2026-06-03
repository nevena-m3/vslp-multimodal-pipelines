# VSLP Acoustic GUI v0.5 - Feature Layer

This update adds the first feature-extraction layer to the Acoustic Pipeline GUI.

## Scope

The GUI now includes a **Features** tab after Segmentation.

The backend writes:

```text
acoustic/004_features/tables/acoustic_features_per_file.csv
acoustic/004_features/tables/acoustic_feature_status_long.csv
acoustic/004_features/tables/selected_acoustic_feature_registry.csv
acoustic/004_features/plots/feature_missingness.png
acoustic/004_features/plots/feature_subsystem_implementation_status.png
acoustic/004_features/reports/acoustic_feature_report.html
acoustic/004_features/logs/stage_manifest.json
```

## Current implementation

V0.5 computes the validated timing/respiratory feature subset from Silero segment tables:

- `total_dur`
- `speech_dur`
- `percent_pause`
- `num_pause`
- `mean_pause_dur`
- `mean_phrase_dur`
- `cv_pause_dur`
- `cv_phrase_dur`
- `total_pause_dur`
- `speech_rate`, only when a task word-count map is supplied; otherwise NaN

All 73 registered acoustic features are available in the feature registry. Features not yet formula-validated are emitted as explicit NaN placeholders with `not_implemented_yet` status. This is intentional and safer than silently computing approximate clinical features.

## GUI workflow

1. Run Ingest.
2. Run Preprocess.
3. Run Silero Segmentation.
4. Open the **Features** tab.
5. Select feature subsystems.
6. Set the minimum internal pause duration.
7. Click **Run Feature Extraction**.
8. Open the feature table and report from **Reports & Outputs**.

## Next feature-layer step

Next update should implement formula-validated feature plugins from the uploaded Bamboo Passage notebook, starting with robust low-risk features and adding validation tests feature by feature.
