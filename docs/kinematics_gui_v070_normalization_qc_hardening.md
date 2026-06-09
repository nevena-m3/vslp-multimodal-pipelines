# Kinematics GUI v0.70.0 - Normalization and QC hardening

This patch strengthens the scientific audit layer between MediaPipe landmark extraction and downstream kinematic feature computation.

## Why this patch exists

The GUI already had computational normalization and landmark/video QC. This patch makes those stages more conservative and more interpretable by exposing the actual stability diagnostics that decide whether normalized trajectories are trustworthy.

Feature computation should continue to depend on normalization and QC outputs rather than raw landmark CSVs.

## Normalization changes

The default normalization method remains `intercanthal_distance`, using MediaPipe landmarks `133` and `362` as the stable eye/canthus anchor distance.

Normalization now writes additional audit columns to `kinematics/004_normalization/tables/normalized_landmarks_manifest.csv`:

- `scale_anchor_landmarks`
- `scale_value_iqr`
- `scale_value_cv`
- `scale_frame_to_frame_max_jump_fraction`
- `scale_frame_to_frame_p95_jump_fraction`
- `n_scale_valid_frames`
- expanded `qc_flags`

Normalized per-frame CSVs now include `scale_value_frame_to_video_ratio`, which makes frame-wise scale instability visible during inspection.

New normalization flags include:

- `normalization_method_raw_only`
- `normalization_anchor_missing`
- `normalization_invalid_scale`
- `normalization_insufficient_face_frames`
- `normalization_insufficient_valid_frames`
- `normalization_high_scale_cv`
- `normalization_high_frame_to_frame_jump`
- `normalization_scale_unstable`

## Video QC changes

The QC backend now loads the selected landmark set from `003_selection/tables/selected_landmarks.json` when no explicit config is passed. This makes QC reflect the analyst's actual landmark selection rather than only the default preset.

QC now measures:

- selected landmark valid rate
- selected complete-frame fraction
- missing selected landmark columns
- longest no-face gap in frames and seconds
- timestamp monotonicity
- median frame delta and estimated FPS from timestamps
- out-of-range selected x/y coordinate burden
- normalization status and normalization QC flags when `normalized_landmarks_manifest.csv` exists

QC still writes the existing GUI-compatible file:

```text
kinematics/005_video_qc/tables/landmark_video_qc_summary.csv
```

It also writes canonical aliases:

```text
kinematics/005_video_qc/tables/video_qc_summary.csv
kinematics/005_video_qc/tables/video_qc_summary.json
kinematics/005_video_qc/tables/video_qc_long_gaps.csv
```

## GUI changes

The Normalization results table now shows:

- scale CV
- maximum frame-to-frame scale jump
- QC flags

The Video QC table now shows:

- selected landmark valid rate
- longest no-face gap in seconds
- out-of-range coordinate burden
- timestamp monotonicity
- normalization status
- normalization QC flags

## Tests

New tests cover:

- unstable normalization anchor scale detection
- QC selected-landmark validity/timestamp reporting
- QC propagation of normalization flags
- canonical QC output aliases

Validated in the uploaded repo snapshot with:

```powershell
python -m pytest tests/unit -q
python -m pytest tests/integration -q
```

Results in the sandbox:

```text
67 passed in 13.16s
1 passed in 7.58s
```
