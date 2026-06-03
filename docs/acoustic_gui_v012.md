# VSLP Acoustic GUI v0.12

This release combines the previously planned v0.10, v0.11, and v0.12 work:

1. Aggregation stage and GUI.
2. QC dashboard stage and GUI.
3. Region-aware feature extraction.
4. Improved embedded previews and reports.

## Region-aware feature extraction

The major scientific correction in v0.12 is that signal features are no longer conceptually treated as full-file measurements by default.

The default region policy is:

```text
speech_only
```

This means phonatory, rhythm, and baseline acoustic signal features are computed from concatenated Silero speech regions. Timing and pause features always use the segment table directly.

Available signal-region policies:

```text
speech_only     Recommended default for most speech signal features.
effective_task  Speech + internal pauses, excluding leading/trailing silence.
full_file       Whole canonical segmentation WAV. Use only when explicitly justified.
```

Some features are pause-specific or timing-specific. These are computed from Silero nonspeech/speech segment intervals, not from waveform-only summaries.

## Aggregation stage

The aggregation stage writes:

```text
acoustic/005_aggregation/tables/acoustic_features_aggregated.csv
acoustic/005_aggregation/tables/aggregation_missingness.csv
acoustic/005_aggregation/reports/acoustic_aggregation_report.html
acoustic/005_aggregation/plots/aggregation_missingness_top30.png
acoustic/005_aggregation/plots/aggregation_group_counts.png
```

Default aggregation grouping:

```text
subject_id, session_id, iteration, task
```

Supported numeric policies:

```text
mean
median
```

Supported missing-value policies:

```text
preserve
drop_features_over_threshold
impute_group_median
```

## QC dashboard

The QC dashboard writes:

```text
acoustic/006_qc_dashboard/tables/acoustic_qc_dashboard.csv
acoustic/006_qc_dashboard/tables/qc_flag_counts.csv
acoustic/006_qc_dashboard/reports/acoustic_qc_dashboard.html
acoustic/006_qc_dashboard/plots/qc_pass_review_counts.png
acoustic/006_qc_dashboard/plots/qc_snr_distribution.png
acoustic/006_qc_dashboard/plots/qc_speech_fraction_distribution.png
acoustic/006_qc_dashboard/plots/qc_flag_counts.png
```

QC flags are research-screening indicators, not clinical acceptance criteria.

## Recommended workflow

```text
Metadata → Ingest → Preprocess → Silero Segmentation → Feature Extraction → Aggregation → QC Dashboard
```

## CLI commands

```bash
vslp acoustic metadata INPUT_FOLDER OUTPUT_ROOT
vslp acoustic ingest INPUT_FOLDER OUTPUT_ROOT
vslp acoustic preprocess INPUT_FOLDER OUTPUT_ROOT
vslp acoustic segment-silero OUTPUT_ROOT/acoustic/002_preprocess/tables/acoustic_preprocess_summary.csv OUTPUT_ROOT
vslp acoustic extract-features OUTPUT_ROOT/acoustic/003_segmentation/tables/acoustic_segmentation_summary.csv OUTPUT_ROOT --acoustic-region-policy speech_only
vslp acoustic aggregate OUTPUT_ROOT/acoustic/004_features/tables/acoustic_features_per_file.csv OUTPUT_ROOT
vslp acoustic qc-dashboard OUTPUT_ROOT
```
