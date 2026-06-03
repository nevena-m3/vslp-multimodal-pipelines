"""Respiratory/timing acoustic features derived from Silero segments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vslp.acoustic.features.plugins.base import AcousticFeaturePlugin, FeatureContext, FeatureValue, as_feature_values

TIMING_FEATURES = (
    "total_dur",
    "speech_dur",
    "percent_pause",
    "num_pause",
    "mean_pause_dur",
    "mean_phrase_dur",
    "cv_pause_dur",
    "cv_phrase_dur",
    "total_pause_dur",
    "speech_rate",
)


def _cv(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    mean = float(np.mean(arr))
    if abs(mean) < 1e-12:
        return np.nan
    return float(np.std(arr, ddof=0) / mean)


@dataclass(frozen=True)
class TimingPlugin(AcousticFeaturePlugin):
    subsystem: str = "respiratory_timing"
    feature_names: tuple[str, ...] = TIMING_FEATURES

    def compute(self, context: FeatureContext) -> dict[str, FeatureValue]:
        cfg = context.config
        min_pause = float(getattr(cfg, "minimum_pause_duration_sec", 0.15))
        task_word_counts = getattr(cfg, "task_word_counts", {}) or {}

        if context.segments_csv is None or not context.segments_csv.exists():
            return {
                name: FeatureValue(name, np.nan, "failed", "segments_csv_missing")
                for name in self.feature_names
            }

        segs = pd.read_csv(context.segments_csv)
        if segs.empty:
            return {name: FeatureValue(name, np.nan, "failed", "empty_segments_csv") for name in self.feature_names}

        speech = segs.loc[segs["segment_type"] == "speech"].copy()
        internal_pause = segs.loc[segs["segment_role"] == "internal_nonspeech"].copy()
        if "duration_sec" in internal_pause.columns:
            internal_pause = internal_pause.loc[internal_pause["duration_sec"] >= min_pause]

        leading = float(segs.loc[segs["segment_role"] == "leading_nonspeech", "duration_sec"].sum()) if "segment_role" in segs else 0.0
        trailing = float(segs.loc[segs["segment_role"] == "trailing_nonspeech", "duration_sec"].sum()) if "segment_role" in segs else 0.0
        duration = float(context.duration_sec) if context.duration_sec is not None else np.nan
        effective_dur = duration - leading - trailing
        if not np.isfinite(effective_dur) or effective_dur <= 0:
            effective_dur = duration if np.isfinite(duration) and duration > 0 else np.nan

        speech_dur = float(speech["duration_sec"].sum()) if not speech.empty else 0.0
        total_pause_dur = float(internal_pause["duration_sec"].sum()) if not internal_pause.empty else 0.0

        features = {
            "total_dur": effective_dur,
            "speech_dur": speech_dur,
            "total_pause_dur": total_pause_dur,
            "percent_pause": float(total_pause_dur / effective_dur) if np.isfinite(effective_dur) and effective_dur > 0 else np.nan,
            "num_pause": float(len(internal_pause)),
            "mean_pause_dur": float(internal_pause["duration_sec"].mean()) if not internal_pause.empty else 0.0,
            "mean_phrase_dur": float(speech["duration_sec"].mean()) if not speech.empty else 0.0,
            "cv_pause_dur": _cv(internal_pause["duration_sec"]) if not internal_pause.empty else np.nan,
            "cv_phrase_dur": _cv(speech["duration_sec"]) if not speech.empty else np.nan,
            "speech_rate": np.nan,
        }

        normalized_task = str(context.task).strip().lower() if context.task is not None and pd.notna(context.task) else ""
        word_count = task_word_counts.get(normalized_task)
        if word_count is not None and np.isfinite(effective_dur) and effective_dur > 0:
            features["speech_rate"] = float(word_count / effective_dur * 60.0)

        return as_feature_values(features, note="computed_from_silero_segments")
