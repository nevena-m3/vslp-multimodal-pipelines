# Kinematics GUI v0.80 — Video / Landmark QC framework placeholder

## Purpose

This patch refines the **Video QC** menu without presenting the current automated checks as a completed clinical QC framework.

Current automated QC is useful for extraction-risk triage before feature computation. It summarizes face visibility, no-face gaps, and selected-landmark frame-to-frame displacement. These checks help determine whether downstream kinematic features are likely to be trustworthy.

The full acoustic-style kinematics QC framework is **not complete yet**. The GUI now shows an explicit **QC TO BE BUILT IN** placeholder so users and reports do not confuse implemented automated checks with a complete validated QC protocol.

## What is implemented now

The current automated QC computes:

- face-detected fraction;
- number of detected frames;
- number of no-face runs;
- longest no-face gap in frames;
- longest no-face gap as a fraction of video length;
- selected-landmark median frame-to-frame displacement;
- selected-landmark p95 frame-to-frame displacement;
- large-jump fraction;
- automated pass/review/fail recommendation with rationale.

These checks are written to:

```text
kinematics/005_video_qc/tables/landmark_video_qc_summary.csv
kinematics/005_video_qc/tables/landmark_video_qc_summary.json
```

## What is explicitly a placeholder

The future acoustic-style kinematics QC framework placeholder is written to:

```text
kinematics/005_video_qc/tables/video_qc_framework_placeholder.csv
kinematics/005_video_qc/tables/video_qc_framework_placeholder.json
```

The placeholder is deliberately labeled:

```text
PLACEHOLDER_TO_BE_BUILT_IN
```

It is not a completed QC framework and should not be interpreted as a validated exclusion protocol.

## Where the QC thresholds come from

The default thresholds are conservative engineering defaults, not disease-specific ALS/Parkinson's cutoffs.

| Parameter | Default | Source | Meaning |
|---|---:|---|---|
| `pass_face_fraction` | 0.90 | engineering default | At least 90% detected frames is treated as clean enough for PASS if other checks are clean. |
| `review_face_fraction` | 0.75 | engineering default | Below this level, automated QC fails the video. |
| `max_gap_review_fraction` | 0.10 | engineering default | Longest no-face gap >=10% of video triggers REVIEW. |
| `max_gap_fail_fraction` | 0.25 | engineering default | Longest no-face gap >=25% of video triggers FAIL. |
| `jitter_review_p95` | 0.080 | engineering default | High p95 frame-to-frame displacement triggers REVIEW. |
| `jitter_fail_p95` | 0.140 | engineering default | Very high p95 frame-to-frame displacement triggers FAIL. |

The correct long-term workflow is to run QC on a representative dataset, visually inspect PASS/REVIEW/FAIL cases, calibrate thresholds, freeze a versioned QC policy, and then report that policy with outputs.

## GUI changes

The Video QC menu now includes:

- status cards for QC status, videos, PASS, REVIEW, FAIL, mean detection, framework status, and next step;
- a tab for current automated QC;
- a tab for the future acoustic-style QC framework placeholder;
- a tab explaining threshold provenance;
- a button to write the placeholder even before running QC.

## Scientific caveat

Current automated QC is a triage layer. It should help decide what to inspect before feature computation. It does not diagnose ALS, Parkinson's disease, or any other condition, and it does not automatically exclude videos.
