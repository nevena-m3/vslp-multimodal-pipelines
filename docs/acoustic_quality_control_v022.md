# VSLP Acoustic Quality Control v0.22

This update refines the segmentation-informed Quality Control stage.

## Purpose

Quality Control extracts artifact-oriented features after data segmentation and before acoustic feature extraction. These values are screening/proxy measurements for researcher review. They do not automatically reject recordings.

## Families

- Additive interference: noise or hum added to the recording, especially visible during internal pauses.
- Gain dynamics: unstable recording level, level drift, or automatic gain control effects.
- Reverberation / echo: speech energy persisting into pauses after speech offset.
- Channel / device: device or microphone spectral response differences.
- Nonlinear distortion: clipping, saturation, near-clipping, or high-amplitude overload.
- Temporal discontinuity: dropouts, silent holes, frozen windows, or abrupt waveform breaks.

## Parameters

- minimum_internal_pause_sec: minimum nonspeech duration used as internal-pause support for pause-region QC features.
- high_level_percentile: speech-frame percentile used to define high-level speech windows for gain and distortion features.
- hard_clip_threshold: absolute waveform threshold for hard clipping/saturation detection.
- near_clip_threshold: absolute waveform threshold for near-clipping/overload risk.
- zero_threshold: absolute waveform threshold for near-zero/dropout detection.
- abrupt_jump_db: frame-to-frame RMS dB jump threshold for temporal discontinuity screening.

## Outputs

- acoustic_quality_features.csv
- acoustic_quality_main_summary.csv
- acoustic_quality_feature_registry.csv
- acoustic_quality_warnings.csv
- acoustic_quality_recommendations.csv
- acoustic_quality_family_status.csv
- acoustic_quality_family_summary.csv
- acoustic_quality_control_report.html

## Plots

- quality_feature_distributions.png
- quality_warning_heatmap.png
- quality_recommendation_summary.png
- quality_family_status.png
- quality_main_features_overview.png
