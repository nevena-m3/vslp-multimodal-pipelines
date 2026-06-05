"""Validated respiratory/timing acoustic features derived from segmentation tables.

This module is intentionally conservative. It computes only features whose formulas
can be determined from the Silero speech/nonspeech segment table. It does not infer
syllable counts, peak rates, or articulation rates unless the required task counts
are explicitly provided.
"""

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
    """Population coefficient of variation, undefined for <2 finite observations."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    mean = float(np.mean(arr))
    if abs(mean) < 1e-12:
        return np.nan
    return float(np.std(arr, ddof=0) / mean)


def _normalize_task_name(task: object) -> str:
    if task is None or pd.isna(task):
        return ""
    return str(task).strip().lower().replace(" ", "_").replace("-", "_")


def _task_word_count(task_word_counts: dict, task: object) -> float | None:
    if not task_word_counts:
        return None
    norm = _normalize_task_name(task)
    # Support exact normalized task names and raw keys from YAML/GUI.
    normalized_map = {_normalize_task_name(k): v for k, v in task_word_counts.items()}
    val = normalized_map.get(norm)
    if val is None:
        return None
    try:
        val_f = float(val)
    except Exception:
        return None
    return val_f if np.isfinite(val_f) and val_f > 0 else None


def _effective_interval_from_segments(segs: pd.DataFrame, fallback_duration: float) -> tuple[float, float, float, str]:
    """Return first-speech start, last-speech end, effective duration, note."""
    speech = segs.loc[segs["segment_type"] == "speech"].copy()
    if not speech.empty and {"start_sec", "end_sec"}.issubset(speech.columns):
        start = float(speech["start_sec"].min())
        end = float(speech["end_sec"].max())
        dur = end - start
        if np.isfinite(dur) and dur > 0:
            return start, end, dur, "effective_interval=first_speech_start_to_last_speech_end"

    # Fallback for legacy segment tables without timestamps.
    leading = 0.0
    trailing = 0.0
    if "segment_role" in segs.columns and "duration_sec" in segs.columns:
        leading = float(segs.loc[segs["segment_role"] == "leading_nonspeech", "duration_sec"].sum())
        trailing = float(segs.loc[segs["segment_role"] == "trailing_nonspeech", "duration_sec"].sum())
    dur = fallback_duration - leading - trailing if np.isfinite(fallback_duration) else np.nan
    if np.isfinite(dur) and dur > 0:
        return 0.0, dur, dur, "effective_interval=duration_minus_edge_nonspeech_fallback"
    return np.nan, np.nan, np.nan, "effective_interval_unavailable"


def _internal_pauses(segs: pd.DataFrame, start: float, end: float, min_pause: float) -> pd.DataFrame:
    """Select internal nonspeech intervals meeting the minimum duration threshold."""
    if segs.empty or "duration_sec" not in segs.columns:
        return pd.DataFrame(columns=segs.columns)
    nonspeech = segs.loc[segs["segment_type"] == "nonspeech"].copy()
    if nonspeech.empty:
        return nonspeech
    if {"start_sec", "end_sec"}.issubset(nonspeech.columns) and np.isfinite(start) and np.isfinite(end):
        # Nonspeech intervals must be inside the first-to-last speech interval.
        eps = 1e-9
        nonspeech = nonspeech.loc[(nonspeech["start_sec"] > start + eps) & (nonspeech["end_sec"] < end - eps)]
    elif "segment_role" in nonspeech.columns:
        nonspeech = nonspeech.loc[nonspeech["segment_role"] == "internal_nonspeech"]
    return nonspeech.loc[nonspeech["duration_sec"] >= min_pause].copy()


@dataclass(frozen=True)
class TimingPlugin(AcousticFeaturePlugin):
    subsystem: str = "respiratory_timing"
    feature_names: tuple[str, ...] = TIMING_FEATURES

    def compute(self, context: FeatureContext) -> dict[str, FeatureValue]:
        cfg = context.config
        min_pause = float(getattr(cfg, "minimum_pause_duration_sec", 0.15))
        task_word_counts = getattr(cfg, "task_word_counts", {}) or {}

        if context.segments_csv is None or not context.segments_csv.exists():
            return {name: FeatureValue(name, np.nan, "failed", "segments_csv_missing") for name in self.feature_names}

        segs = pd.read_csv(context.segments_csv)
        required = {"segment_type", "duration_sec"}
        missing = sorted(required.difference(segs.columns))
        if missing:
            return {name: FeatureValue(name, np.nan, "failed", f"segments_csv_missing_required_columns:{','.join(missing)}") for name in self.feature_names}
        if segs.empty:
            return {name: FeatureValue(name, np.nan, "failed", "empty_segments_csv") for name in self.feature_names}

        duration = float(context.duration_sec) if context.duration_sec is not None and np.isfinite(context.duration_sec) else np.nan
        start, end, effective_dur, interval_note = _effective_interval_from_segments(segs, duration)
        speech = segs.loc[segs["segment_type"] == "speech"].copy()
        if speech.empty or not np.isfinite(effective_dur) or effective_dur <= 0:
            base = {
                "total_dur": effective_dur,
                "speech_dur": np.nan,
                "total_pause_dur": np.nan,
                "percent_pause": np.nan,
                "num_pause": np.nan,
                "mean_pause_dur": np.nan,
                "mean_phrase_dur": np.nan,
                "cv_pause_dur": np.nan,
                "cv_phrase_dur": np.nan,
                "speech_rate": np.nan,
            }
            return as_feature_values(base, status="failed", note="no_valid_speech_interval; " + interval_note)

        internal_pause = _internal_pauses(segs, start, end, min_pause)
        speech_dur = float(speech["duration_sec"].sum())
        total_pause_dur = float(internal_pause["duration_sec"].sum()) if not internal_pause.empty else 0.0
        percent_pause = float(100.0 * total_pause_dur / effective_dur) if effective_dur > 0 else np.nan

        word_count = _task_word_count(task_word_counts, context.task)
        speech_rate = float(word_count / effective_dur * 60.0) if word_count is not None and effective_dur > 0 else np.nan
        speech_rate_note = "speech_rate=word_count_per_effective_minute" if word_count is not None else "speech_rate_not_computed_no_task_word_count"

        features = {
            "total_dur": float(effective_dur),
            "speech_dur": speech_dur,
            "total_pause_dur": total_pause_dur,
            "percent_pause": percent_pause,
            "num_pause": float(len(internal_pause)),
            "mean_pause_dur": float(internal_pause["duration_sec"].mean()) if not internal_pause.empty else np.nan,
            "mean_phrase_dur": float(speech["duration_sec"].mean()) if not speech.empty else np.nan,
            "cv_pause_dur": _cv(internal_pause["duration_sec"]) if not internal_pause.empty else np.nan,
            "cv_phrase_dur": _cv(speech["duration_sec"]),
            "speech_rate": speech_rate,
        }
        note = (
            "validated_timing_v0.26; computed_from_segmentation_segments; "
            f"minimum_internal_pause_sec={min_pause}; {interval_note}; {speech_rate_note}; "
            "percent_pause_units=percent_0_to_100"
        )
        return as_feature_values(features, note=note)
