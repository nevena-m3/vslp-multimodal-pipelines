# Family 01 scientific contract — 5 October 2026

This is the current VSLP implementation contract, not evidence of reference or clinical validity. Implementation tests are present; independent Praat parity and real-data validation remain pending.

## Task use

| Construct | Sustained /a/ | S5 | Bamboo | Other sentences and DDK |
| --- | --- | --- | --- | --- |
| Fundamental frequency | Recommended | Recommended | Recommended | Executable, not recommended |
| F0 variability | Recommended | Recommended | Recommended | Executable, not recommended |
| Observed F0 range | Executable, not recommended | Recommended | Recommended | Executable, not recommended |
| Intonation components | Executable, not recommended | Recommended | Recommended | Executable, not recommended |

S5 is “We were away a year ago”. The current sustained-vowel registry contains /a/ only. The F01 task code can retain a different explicit vowel identity if a future approved task entry supplies it. The current battery does not elicit physiological maximum phonatory frequency range.

## Measurement and aggregation

For each reviewed sustained-vowel trial, F01 intersects its reviewed stable interval with eligible speech, its trial boundaries, and manual exclusions. It chooses the longest contiguous eligible part and analyzes up to its centered 2.0 seconds. A shorter part remains eligible if the existing F0 frame and tracking checks pass. There is no rule that discards the first 2 seconds. If an additional vowel attempt lacks reviewed stable bounds, it cannot be treated as a valid F01 trial by assumption.

Praat autocorrelation settings remain 60–500 Hz, 5 ms time step, at least 20 valid frames, and at least 80% tracking yield for sustained phonation. Technical octave and boundary flags do not repair the contour.

The public recording value for each F01 leaf is the median of valid trial-level values; `f0_trial_summaries.csv` retains each attempt. `f0_legacy_pooled_recording.csv` separately retains the former pooled-frame calculation, explicitly labeled legacy. This change in the public recording estimator is versioned as `family01_median_valid_trial_values_v1`. Original-time frame rows remain in `f0_tracks.csv`.

`pfr_maxmin_st` remains the available but nonrecommended literal max/min observed-range variant. The new project-defined preferred `observed_f0_range_p05_p95_st` is Q95–Q05 of valid semitone F0 within each trial and is recommended for S5 and Bamboo. Both describe observed range in the analyzed task, not elicited physiological range. The intonation factor score remains unavailable.

For Bamboo, F01 uses reviewed speech spans without concatenating audio across pauses or exclusions. Only valid voiced F0 frames contribute. If final reviewed segmentation contains explicit approved sentence labels covering the trial's voiced frames, intonation components are computed per sentence and summarized by their median. Current standard reviewed-segmentation output has no such sentence annotations; consequently the ordinary path uses the eligible voiced passage contour and records `intonation_analysis_unit=passage_fallback`. No linguistic Alignment is required for F01.
