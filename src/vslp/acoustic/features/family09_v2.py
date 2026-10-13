"""Bamboo pause and phrase events from F07's final reviewed timing authority."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vslp.acoustic.features.family07_v2 import reviewed_timing


VERSION = "family09-reviewed-pause-2.0.0"
PAUSE_THRESHOLD_SEC = 0.300
PUBLIC_IDS = frozenset({"bamboo_pause_duration_median_s", "bamboo_pause_duration_cv",
                        "bamboo_phrase_duration_cv", "bamboo_pause_count",
                        "bamboo_percent_pause_pct", "bamboo_mean_pause_duration_s",
                        "bamboo_mean_phrase_duration_s"})
EVENT_COLUMNS = ("recording_id", "trial_id", "task_id", "event_type", "event_index",
                 "start_sec", "end_sec", "duration_sec", "threshold_sec",
                 "source_speech_interval_indices", "internal_subthreshold_gap_count",
                 "boundary_source", "review_provenance", "valid", "rejection_reason")


def _unavailable(reason: str) -> dict[str, tuple[float, str]]:
    return {feature: (np.nan, reason) for feature in PUBLIC_IDS}


def reviewed_pause_events(intervals: pd.DataFrame, recording_id: str,
                          trial_id: str) -> tuple[pd.DataFrame, dict[str, tuple[float, str]], dict]:
    """Derive one event set; hard exclusions never become physiological pauses."""
    timing, audit = reviewed_timing(intervals, "bamboo_passage")
    if not timing or audit.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS), _unavailable("missing_reviewed_segmentation"), {}
    total, timing_reason = timing.get("bamboo_total_duration_s", (np.nan, "missing_reviewed_segmentation"))
    if timing_reason or not np.isfinite(total):
        reason = "invalid_timing_interval" if timing_reason == "invalid_speech_interval" else (
            "no_valid_speech_intervals" if timing_reason == "no_valid_speech_intervals" else timing_reason)
        return pd.DataFrame(columns=EVENT_COLUMNS), _unavailable(reason), {}
    if total <= 0:
        return pd.DataFrame(columns=EVENT_COLUMNS), _unavailable("total_duration_nonpositive"), {}
    speech = audit.loc[audit.segment_role.astype(str).eq("eligible_speech")].sort_values("start_sec")
    if speech.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS), _unavailable("no_valid_speech_intervals"), {}
    first, last = float(speech.start_sec.min()), float(speech.end_sec.max())
    exclusions = intervals.loc[intervals.segment_role.astype(str).eq("manual_exclusion")]
    if any(float(row.start_sec) < last and float(row.end_sec) > first
           for row in exclusions.itertuples()):
        return pd.DataFrame(columns=EVENT_COLUMNS), _unavailable("hard_exclusion_ambiguity"), {
            "bamboo_total_duration_s": total}
    segments = [(float(row.start_sec), float(row.end_sec), index)
                for index, row in enumerate(speech.itertuples(), 1)]
    internal = intervals.loc[intervals.segment_role.astype(str).eq("internal_nonspeech")]
    rows = []
    phrase_start, phrase_end, indices, short_count = segments[0][0], segments[0][1], [1], 0
    pause_index = phrase_index = 0

    def append(event_type: str, start: float, end: float, index: int,
               source_indices: list[int], short_gaps: int) -> None:
        rows.append({"recording_id": recording_id, "trial_id": trial_id,
                     "task_id": "bamboo_passage", "event_type": event_type,
                     "event_index": index, "start_sec": start, "end_sec": end,
                     "duration_sec": end - start, "threshold_sec": PAUSE_THRESHOLD_SEC,
                     "source_speech_interval_indices": ";".join(map(str, source_indices)),
                     "internal_subthreshold_gap_count": short_gaps,
                     "boundary_source": "FINAL_REVIEWED_TIMING",
                     "review_provenance": "FINAL_REVIEWED_SEGMENTATION",
                     "valid": True, "rejection_reason": ""})

    for left, right in zip(segments, segments[1:], strict=False):
        gap_start, gap_end = left[1], right[0]
        if gap_end < gap_start - 1e-8:
            return pd.DataFrame(columns=EVENT_COLUMNS), _unavailable("invalid_timing_interval"), {}
        if gap_end <= gap_start + 1e-8:
            phrase_end = right[1]
            indices.append(right[2])
            continue
        matching = internal.loc[(internal.start_sec.astype(float) - gap_start).abs().lt(1e-7)
                                & (internal.end_sec.astype(float) - gap_end).abs().lt(1e-7)]
        if len(matching) != 1:
            return pd.DataFrame(columns=EVENT_COLUMNS), _unavailable("invalid_internal_gap"), {}
        if round(gap_end - gap_start, 9) >= PAUSE_THRESHOLD_SEC:
            phrase_index += 1
            append("PHRASE", phrase_start, phrase_end, phrase_index, indices, short_count)
            pause_index += 1
            append("PAUSE", gap_start, gap_end, pause_index, [], 0)
            phrase_start, phrase_end, indices, short_count = right[0], right[1], [right[2]], 0
        else:
            phrase_end = right[1]
            indices.append(right[2])
            short_count += 1
    phrase_index += 1
    append("PHRASE", phrase_start, phrase_end, phrase_index, indices, short_count)
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    pauses = events.loc[events.event_type.eq("PAUSE"), "duration_sec"].to_numpy(dtype=float)
    phrases = events.loc[events.event_type.eq("PHRASE"), "duration_sec"].to_numpy(dtype=float)
    if not np.isfinite(pauses).all() or (pauses <= 0).any():
        return events, _unavailable("nonfinite_pause_duration"), {}
    if not np.isfinite(phrases).all() or (phrases <= 0).any():
        return events, _unavailable("nonfinite_phrase_duration"), {}
    pause_total = float(pauses.sum())
    if pause_total > total + 1e-8:
        return events, _unavailable("pause_duration_exceeds_total"), {}
    percent = 100.0 * pause_total / total
    if not np.isfinite(percent) or percent < 0 or percent > 100 + 1e-8:
        return events, _unavailable("percent_pause_out_of_range"), {}
    result = {
        "bamboo_pause_count": (float(len(pauses)), ""),
        "bamboo_percent_pause_pct": (percent, ""),
        "bamboo_pause_duration_median_s": ((float(np.median(pauses)), "") if len(pauses)
                                            else (np.nan, "no_qualifying_pause")),
        "bamboo_mean_pause_duration_s": ((float(np.mean(pauses)), "") if len(pauses)
                                         else (np.nan, "no_qualifying_pause")),
        "bamboo_mean_phrase_duration_s": (float(np.mean(phrases)), ""),
    }
    if len(pauses) < 2:
        result["bamboo_pause_duration_cv"] = (np.nan, "insufficient_pause_events_for_cv")
    else:
        mean = float(np.mean(pauses))
        value = float(np.std(pauses, ddof=1) / mean) if mean > 0 else np.nan
        result["bamboo_pause_duration_cv"] = ((value, "") if np.isfinite(value) else
                                               (np.nan, "zero_mean_pause_duration" if mean <= 0
                                                else "nonfinite_pause_cv"))
    if len(phrases) < 2:
        result["bamboo_phrase_duration_cv"] = (np.nan, "insufficient_phrases_for_cv")
    else:
        mean = float(np.mean(phrases))
        value = float(np.std(phrases, ddof=1) / mean) if mean > 0 else np.nan
        result["bamboo_phrase_duration_cv"] = ((value, "") if np.isfinite(value) else
                                                (np.nan, "zero_mean_phrase_duration" if mean <= 0
                                                 else "nonfinite_feature"))
    support = {"bamboo_total_duration_s": total,
               "bamboo_total_pause_duration_s": pause_total,
               "bamboo_phrase_count": len(phrases)}
    return events, result, support
