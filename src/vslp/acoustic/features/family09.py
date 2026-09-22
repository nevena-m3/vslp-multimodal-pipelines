"""Family 09 pause and phrasing values from shared frozen-review events."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

FAMILY09_IDS = frozenset({
    "total_pause_duration_s",
    "cv_pause_duration",
    "cv_pause_duration_pct",
    "cv_phrase_duration",
    "cv_phrase_duration_pct",
    "pause_count",
    "percent_pause_ge300ms",
    "mean_pause_duration_s",
    "mean_phrase_duration_s",
    "pause_pattern_components",
    "pause_pattern_factor_if_frozen",
})
ALGORITHM_VERSION = "family09-pause-1.0.0"
PARAMETER_SET_ID = "family09_bamboo_reviewed_v1"


def calculate_family09(
    events: pd.DataFrame, timing: dict[str, float], selected: set[str],
) -> dict[str, tuple[float | str, str]]:
    """Calculate every requested exact ID from one event table."""
    results: dict[str, tuple[float | str, str]] = {}
    if not timing:
        return {feature_id: (np.nan, "reviewed_timing_unavailable")
                for feature_id in sorted(selected)}
    pauses = events.loc[events["event_type"].eq("pause")]
    phrases = events.loc[events["event_type"].eq("phrase")]
    pause_durations = pauses["duration_sec"].to_numpy(dtype=float)
    phrase_durations = phrases["duration_sec"].to_numpy(dtype=float)
    n_pause, n_phrase = len(pause_durations), len(phrase_durations)
    pause_total = float(np.sum(pause_durations))
    elapsed = float(timing["elapsed_task_sec"])
    if elapsed <= 0:
        return {feature_id: (np.nan, "nonpositive_analyzed_duration")
                for feature_id in sorted(selected)}
    percent = 100.0 * pause_total / elapsed
    pause_mean = float(np.mean(pause_durations)) if n_pause else np.nan
    phrase_mean = float(np.mean(phrase_durations)) if n_phrase else np.nan
    pause_sd = float(np.std(pause_durations, ddof=1)) if n_pause >= 2 else np.nan
    phrase_sd = float(np.std(phrase_durations, ddof=1)) if n_phrase >= 2 else np.nan
    pause_cv = pause_sd / pause_mean if n_pause >= 2 and pause_mean > 0 else np.nan
    phrase_cv = phrase_sd / phrase_mean if n_phrase >= 2 and phrase_mean > 0 else np.nan
    for feature_id in sorted(selected):
        if feature_id == "total_pause_duration_s":
            results[feature_id] = (pause_total, "")
        elif feature_id == "pause_count":
            results[feature_id] = (int(n_pause), "")
        elif feature_id == "percent_pause_ge300ms":
            results[feature_id] = (percent, "")
        elif feature_id == "mean_pause_duration_s":
            results[feature_id] = (pause_mean, "" if n_pause else "no_pause")
        elif feature_id in {"cv_pause_duration", "cv_pause_duration_pct"}:
            value = pause_cv * (100.0 if feature_id.endswith("_pct") else 1.0)
            reason = "" if n_pause >= 2 else "no_pause" if n_pause == 0 else "insufficient_pause_events"
            results[feature_id] = (value, reason)
        elif feature_id == "mean_phrase_duration_s":
            results[feature_id] = (phrase_mean, "" if n_phrase else "no_phrase")
        elif feature_id in {"cv_phrase_duration", "cv_phrase_duration_pct"}:
            value = phrase_cv * (100.0 if feature_id.endswith("_pct") else 1.0)
            reason = "" if n_phrase >= 2 else "insufficient_phrase_events"
            if n_pause == 0 and n_phrase >= 2:
                value, reason = np.nan, "phrases_split_only_by_contamination"
            results[feature_id] = (value, reason)
        elif feature_id == "pause_pattern_components":
            if n_pause < 2:
                results[feature_id] = (np.nan, "insufficient_pause_events_for_composite_components")
            else:
                components: dict[str, Any] = {
                    "mean_pause_duration_s": pause_mean,
                    "sd_pause_duration_s": pause_sd,
                    "cv_pause_duration": pause_cv,
                    "percent_pause_ge300ms": percent,
                }
                results[feature_id] = (json.dumps(components, sort_keys=True), "")
        elif feature_id == "pause_pattern_factor_if_frozen":
            results[feature_id] = (np.nan, "frozen_source_transform_unavailable")
        else:
            raise ValueError(f"Unregistered Family 09 output: {feature_id}")
    return results
