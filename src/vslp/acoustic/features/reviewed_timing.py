"""Shared reviewed speech, pause, and phrase events for Families 08 and 09."""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_INTERNAL_PAUSE_SEC = 0.300
EVENT_COLUMNS = (
    "recording_id", "file_name", "event_type", "event_index",
    "start_sec", "end_sec", "duration_sec", "source", "source_role",
    "qualifies_as_pause", "boundary_source", "parameter_set_id",
)


def derive_reviewed_timing(
    intervals: pd.DataFrame, *, recording_id: str = "", file_name: str = "",
    recording_duration_sec: float | None = None,
    parameter_set_id: str = "reviewed_pause_ge300ms_v1",
) -> tuple[pd.DataFrame, dict[str, float], str]:
    """Derive one reusable event set without changing frozen segmentation."""
    required = {"view", "segment_role", "start_sec", "end_sec"}
    if not required <= set(intervals):
        return pd.DataFrame(columns=EVENT_COLUMNS), {}, "final_interval_schema_missing"
    timeline = intervals.loc[intervals["view"].eq("authoritative")].copy()
    if timeline.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS), {}, "no_final_patient_speech"
    timeline["start_sec"] = pd.to_numeric(timeline["start_sec"], errors="coerce")
    timeline["end_sec"] = pd.to_numeric(timeline["end_sec"], errors="coerce")
    if timeline[["start_sec", "end_sec"]].isna().any().any() or (
            timeline["end_sec"] <= timeline["start_sec"]).any():
        return pd.DataFrame(columns=EVENT_COLUMNS), {}, "invalid_final_interval_boundaries"
    timeline = timeline.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)
    starts = timeline["start_sec"].to_numpy(dtype=float)
    ends = timeline["end_sec"].to_numpy(dtype=float)
    if np.any(starts[1:] < ends[:-1] - 1e-7):
        return pd.DataFrame(columns=EVENT_COLUMNS), {}, "overlapping_final_intervals"
    speech = timeline.loc[timeline["segment_role"].eq("speech")]
    if speech.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS), {}, "no_final_patient_speech"
    utterance_start = float(speech["start_sec"].min())
    utterance_end = float(speech["end_sec"].max())
    def reviewed_bound(column: str, fallback: float) -> float:
        if column not in timeline:
            return fallback
        values = pd.to_numeric(timeline[column], errors="coerce").dropna()
        return float(values.iloc[0]) if not values.empty else fallback

    analysis_start = reviewed_bound("analysis_start_sec", float(starts[0]))
    analysis_end = reviewed_bound("analysis_end_sec", float(ends[-1]))
    if not (0 <= analysis_start < analysis_end and
            analysis_start <= utterance_start < utterance_end <= analysis_end):
        return pd.DataFrame(columns=EVENT_COLUMNS), {}, "invalid_reviewed_analysis_window"
    raw_duration = (float(recording_duration_sec) if recording_duration_sec is not None
                    else float(ends[-1] - starts[0]))
    if not np.isfinite(raw_duration) or raw_duration <= 0:
        return pd.DataFrame(columns=EVENT_COLUMNS), {}, "invalid_recording_duration"

    rows: list[dict[str, object]] = []
    counters: dict[str, int] = {}

    def append_event(event_type: str, start: float, end: float, role: str,
                     qualifies: bool, boundary_source: str = "") -> None:
        counters[event_type] = counters.get(event_type, 0) + 1
        rows.append({
            "recording_id": recording_id, "file_name": file_name,
            "event_type": event_type, "event_index": counters[event_type],
            "start_sec": start, "end_sec": end, "duration_sec": end - start,
            "source": "final_reviewed_segmentation", "source_role": role,
            "qualifies_as_pause": qualifies, "boundary_source": boundary_source,
            "parameter_set_id": parameter_set_id,
        })

    for index, row in timeline.iterrows():
        role = str(row.segment_role)
        a, b = float(row.start_sec), float(row.end_sec)
        if role == "internal_nonspeech" and utterance_start <= a and b <= utterance_end:
            adjacent_speech = (
                index > 0 and index + 1 < len(timeline)
                and timeline.iloc[index - 1].segment_role == "speech"
                and timeline.iloc[index + 1].segment_role == "speech"
                and abs(float(timeline.iloc[index - 1].end_sec) - a) <= 1e-6
                and abs(float(timeline.iloc[index + 1].start_sec) - b) <= 1e-6
            )
            qualifies = adjacent_speech and b - a >= MIN_INTERNAL_PAUSE_SEC - 1e-9
            append_event("pause" if qualifies else "internal_gap", a, b, role,
                         qualifies, str(row.get("boundary_source", "")))
        elif role == "manual_exclusion":
            clipped_start, clipped_end = max(a, analysis_start), min(b, analysis_end)
            if clipped_end > clipped_start:
                append_event("excluded_contamination", clipped_start, clipped_end,
                             role, False, str(row.get("boundary_source", "")))

    phrase_start: float | None = None
    phrase_end: float | None = None
    for index, row in timeline.iterrows():
        a, b = float(row.start_sec), float(row.end_sec)
        if b <= utterance_start or a >= utterance_end:
            continue
        role = str(row.segment_role)
        if role == "speech":
            if phrase_start is None:
                phrase_start = a
            phrase_end = b
            continue
        is_short_gap = role == "internal_nonspeech" and b - a < MIN_INTERNAL_PAUSE_SEC - 1e-9
        if is_short_gap and phrase_start is not None:
            continue
        if phrase_start is not None and phrase_end is not None:
            append_event("phrase", phrase_start, phrase_end, "speech_group", False,
                         str(row.get("boundary_source", "")))
            phrase_start = phrase_end = None
    if phrase_start is not None and phrase_end is not None:
        append_event("phrase", phrase_start, phrase_end, "speech_group", False)

    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    pauses = events.loc[events["event_type"].eq("pause")]
    phrases = events.loc[events["event_type"].eq("phrase")]
    exclusions = events.loc[events["event_type"].eq("excluded_contamination")]
    excluded_total = float(exclusions["duration_sec"].sum()) if not exclusions.empty else 0.0
    excluded_inside = sum(
        max(0.0, min(utterance_end, float(row.end_sec)) -
            max(utterance_start, float(row.start_sec)))
        for row in exclusions.itertuples())
    effective_elapsed = utterance_end - utterance_start - excluded_inside
    pause_total = float(pauses["duration_sec"].sum()) if not pauses.empty else 0.0
    speech_time = effective_elapsed - pause_total
    if effective_elapsed <= 0 or speech_time <= 0:
        return events, {}, "nonpositive_analyzable_or_speech_time"
    summary = {
        "raw_task_duration_sec": raw_duration,
        "analysis_start_sec": analysis_start,
        "analysis_end_sec": analysis_end,
        "analysis_window_duration_sec": analysis_end - analysis_start,
        "utterance_start_sec": utterance_start,
        "utterance_end_sec": utterance_end,
        "elapsed_task_sec": effective_elapsed,
        "internal_pauses_ge_300ms_sec": pause_total,
        "manual_exclusion_sec": excluded_inside,
        "excluded_contamination_duration_sec": excluded_total,
        "articulation_speech_sec": speech_time,
        "n_pause_events": int(len(pauses)),
        "n_phrase_events": int(len(phrases)),
    }
    return events, summary, ""
