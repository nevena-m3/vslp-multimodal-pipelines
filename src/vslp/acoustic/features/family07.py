"""Family 07 durations from shared reviewed timing and optional aligned tokens."""

from __future__ import annotations

import numpy as np
import pandas as pd


ALGORITHM_VERSION = "family07-reviewed-duration-1.0.0"
TIMING_PARAMETER_SET_ID = "family07_reviewed_timing_v1"
ALIGNMENT_PARAMETER_SET_ID = "family07_alignment_tokens_v1"
MIN_ALIGNMENT_COVERAGE = 0.80
MIN_VOWEL_DURATION_SEC = 0.030
FAMILY07_IDS = frozenset({"mean_word_duration_s", "speech_time_s",
                          "task_elapsed_duration_s", "delta_v_s",
                          "varco_v_pct", "npvi_v_pct"})
ALIGNMENT_IDS = frozenset({"mean_word_duration_s", "delta_v_s",
                           "varco_v_pct", "npvi_v_pct"})
EVENT_COLUMNS = (
    "recording_id", "file_name", "feature_source", "event_type", "event_index",
    "sequence_id", "label", "start_sec", "end_sec", "duration_sec",
    "boundary_source", "alignment_source", "valid_for_feature", "exclusion_reason",
    "review_flag",
)


def _overlaps(a: float, b: float, exclusions: pd.DataFrame) -> bool:
    return any(float(row.start_sec) < b and float(row.end_sec) > a
               for row in exclusions.itertuples())


def derive_duration_events(recording_id: str, file_name: str,
                           reviewed_events: pd.DataFrame,
                           alignment: pd.DataFrame | None) -> pd.DataFrame:
    """Create an audit layer without changing reviewed segmentation or alignment."""
    rows: list[dict[str, object]] = []
    exclusions = reviewed_events.loc[
        reviewed_events.event_type.eq("excluded_contamination")].copy()
    for event in reviewed_events.itertuples():
        rows.append({
            "recording_id": recording_id, "file_name": file_name,
            "feature_source": "final_reviewed_segmentation",
            "event_type": event.event_type, "event_index": int(event.event_index),
            "sequence_id": "", "label": event.source_role,
            "start_sec": float(event.start_sec), "end_sec": float(event.end_sec),
            "duration_sec": float(event.duration_sec),
            "boundary_source": event.boundary_source,
            "alignment_source": "", "valid_for_feature": event.event_type != "excluded_contamination",
            "exclusion_reason": "manual_contamination" if event.event_type == "excluded_contamination" else "",
            "review_flag": "",
        })
    if alignment is None or alignment.empty:
        return pd.DataFrame(rows, columns=EVENT_COLUMNS)
    subset = alignment.loc[alignment.recording_id.astype(str).eq(str(recording_id))].copy()
    subset = subset.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)
    previous_valid_end: float | None = None
    sequence = 1
    previous_source_sequence = None
    counters: dict[str, int] = {}
    for token in subset.itertuples():
        token_type = str(token.token_type).strip().lower()
        counters[token_type] = counters.get(token_type, 0) + 1
        start, end = float(token.start_sec), float(token.end_sec)
        duration = end - start
        valid_flag = str(getattr(token, "valid", "true")).strip().lower() not in {"0", "false", "no"}
        reason = ""
        if end <= start:
            valid_flag, reason = False, "invalid_alignment_boundary"
        elif _overlaps(start, end, exclusions):
            valid_flag, reason = False, "manual_contamination_overlap"
        elif token_type == "vowel" and duration < MIN_VOWEL_DURATION_SEC:
            valid_flag, reason = False, "vowel_duration_below_30ms"
        source_sequence = getattr(token, "sequence_id", "")
        contamination_break = (previous_valid_end is not None and any(
            float(item.start_sec) < start and float(item.end_sec) > previous_valid_end
            for item in exclusions.itertuples()))
        if previous_valid_end is not None and (
                contamination_break or (source_sequence != "" and
                                         previous_source_sequence not in {None, ""}
                                         and source_sequence != previous_source_sequence)):
            sequence += 1
        if valid_flag:
            previous_valid_end = end
            previous_source_sequence = source_sequence
        rows.append({
            "recording_id": recording_id, "file_name": file_name,
            "feature_source": "alignment", "event_type": token_type,
            "event_index": counters[token_type], "sequence_id": sequence,
            "label": str(token.label), "start_sec": start, "end_sec": end,
            "duration_sec": duration,
            "boundary_source": "ALIGNMENT",
            "alignment_source": str(getattr(token, "alignment_source", "unspecified")),
            "valid_for_feature": valid_flag, "exclusion_reason": reason,
            "review_flag": ("vowel_duration_below_50ms_review" if valid_flag
                            and token_type == "vowel" and duration < 0.050 else ""),
        })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def calculate_family07(events: pd.DataFrame, timing: dict[str, float],
                       alignment_rows: pd.DataFrame | None,
                       selected: set[str]) -> tuple[dict[str, tuple[float, str]], dict[str, object]]:
    """Calculate exact Family 07 leaves and return per-recording support counts."""
    result: dict[str, tuple[float, str]] = {}
    support = {"n_events": 0, "n_expected_words": np.nan, "n_aligned_words": 0,
               "alignment_coverage": np.nan, "n_valid_vowels": 0,
               "n_adjacent_vowel_pairs": 0,
               "task_boundary_method": "reviewed_speech_boundary_fallback"}
    if not timing:
        return ({feature_id: (np.nan, "reviewed_timing_unavailable")
                 for feature_id in selected}, support)
    if "speech_time_s" in selected:
        result["speech_time_s"] = (float(timing["articulation_speech_sec"]), "")
    aligned = events.loc[(events.feature_source.eq("alignment")) &
                         events.valid_for_feature.astype(bool)] if not events.empty else pd.DataFrame()
    words = aligned.loc[aligned.event_type.eq("word")] if not aligned.empty else pd.DataFrame()
    vowels = aligned.loc[aligned.event_type.eq("vowel")] if not aligned.empty else pd.DataFrame()
    support["n_aligned_words"] = int(len(words))
    support["n_valid_vowels"] = int(len(vowels))
    expected = np.nan
    if alignment_rows is not None and not alignment_rows.empty and "expected_count" in alignment_rows:
        values = pd.to_numeric(alignment_rows.loc[
            alignment_rows.token_type.astype(str).str.lower().eq("word"), "expected_count"],
            errors="coerce").dropna()
        if not values.empty:
            expected = float(values.max())
    support["n_expected_words"] = expected
    coverage = len(words) / expected if np.isfinite(expected) and expected > 0 else np.nan
    support["alignment_coverage"] = coverage
    if "task_elapsed_duration_s" in selected:
        task_elapsed = float(timing["elapsed_task_sec"])
        if not words.empty and np.isfinite(coverage) and coverage >= MIN_ALIGNMENT_COVERAGE:
            start, end = float(words.start_sec.min()), float(words.end_sec.max())
            exclusions = events.loc[events.event_type.eq("excluded_contamination")]
            excluded = sum(max(0.0, min(end, float(item.end_sec)) -
                               max(start, float(item.start_sec)))
                           for item in exclusions.itertuples())
            if end - start - excluded > 0:
                task_elapsed = end - start - excluded
                support["task_boundary_method"] = "first_to_last_valid_aligned_word"
        result["task_elapsed_duration_s"] = (task_elapsed, "")
    if "mean_word_duration_s" in selected:
        reason = ("missing_alignment" if alignment_rows is None or alignment_rows.empty else
                  "expected_word_count_missing" if not np.isfinite(expected) else
                  "low_alignment_coverage" if coverage < MIN_ALIGNMENT_COVERAGE else
                  "no_valid_word_tokens" if words.empty else "")
        result["mean_word_duration_s"] = (
            np.nan if reason else float(words.duration_sec.mean()), reason)
    variability = selected & {"delta_v_s", "varco_v_pct", "npvi_v_pct"}
    if variability:
        if alignment_rows is None or alignment_rows.empty:
            result.update({feature_id: (np.nan, "missing_alignment") for feature_id in variability})
        elif not alignment_rows.token_type.astype(str).str.lower().eq("vowel").any():
            result.update({feature_id: (np.nan, "missing_phone_alignment") for feature_id in variability})
        elif len(vowels) < 2:
            result.update({feature_id: (np.nan, "insufficient_vowel_tokens") for feature_id in variability})
        else:
            durations = vowels.duration_sec.to_numpy(dtype=float)
            sample_sd = float(np.std(durations, ddof=1))
            mean = float(np.mean(durations))
            values = {"delta_v_s": sample_sd,
                      "varco_v_pct": 100.0 * sample_sd / mean if mean > 0 else np.nan}
            pair_terms = []
            for _, group in vowels.groupby("sequence_id", sort=False):
                d = group.sort_values("start_sec").duration_sec.to_numpy(dtype=float)
                if len(d) >= 2:
                    pair_terms.extend(np.abs(np.diff(d)) / ((d[:-1] + d[1:]) / 2.0))
            support["n_adjacent_vowel_pairs"] = len(pair_terms)
            values["npvi_v_pct"] = 100.0 * float(np.mean(pair_terms)) if len(pair_terms) >= 2 else np.nan
            for feature_id in variability:
                reason = ("nonpositive_mean_vowel_duration" if feature_id == "varco_v_pct"
                          and not np.isfinite(values[feature_id]) else
                          "insufficient_adjacent_vowel_pairs" if feature_id == "npvi_v_pct"
                          and len(pair_terms) < 2 else "")
                result[feature_id] = (values[feature_id], reason)
    support["n_events"] = int(len(events))
    return result, support
