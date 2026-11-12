"""Reviewed, original-time stable phonation spans shared by F01 and F04."""

from __future__ import annotations

import json

import pandas as pd


def _stable_regions(row) -> list[tuple[float, float]]:
    """Per-repetition stable windows from the frozen review (may be absent)."""
    raw = getattr(row, "stable_regions_json", "")
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        return [(float(a), float(b)) for a, b in json.loads(raw)]
    except (TypeError, ValueError):
        return []


def repetition_trials(row, intervals: pd.DataFrame, duration_sec: float) -> list[dict]:
    """One trial per reviewed /a/ repetition that has its own stable window.

    Used when no trials were confirmed by hand, so trial-level features are
    computed per repetition and summarised across repetitions (median), as
    for confirmed trials. Empty when the review has no per-repetition windows.
    """
    regions = [(a, b) for a, b in _stable_regions(row) if 0 <= a < b <= duration_sec]
    if not regions:
        return []
    trials = []
    speech = intervals.loc[intervals.segment_role.eq("speech")].sort_values("start_sec")
    for number, item in enumerate(speech.itertuples(), 1):
        left, right = float(item.start_sec), float(item.end_sec)
        own = [region for region in regions if left <= region[0] < region[1] <= right]
        if own:
            trials.append({"trial_id": f"repetition_{number:02d}", "start_sec": left,
                           "end_sec": right, "stable_region_start": own[0][0],
                           "stable_region_end": own[0][1]})
    return trials


def centered_stable_spans(row, intervals: pd.DataFrame, duration_sec: float,
                          trials: list[dict]) -> list[tuple[float, float, str, float, float]]:
    """Return one centred, exclusion-free span per trial without acoustic tracking.

    Without confirmed trials, every reviewed /a/ repetition (speech interval)
    is its own trial, with its own stable window when the segmentation gave
    one. Before 2026-11-12 all repetitions shared one trial id, so only the
    longest one reached the features.
    """
    try:
        start, end = float(row.stable_region_start), float(row.stable_region_end)
    except (TypeError, ValueError):
        start = end = -1.0
    domain = (start, end) if 0 <= start < end <= duration_sec else None
    regions = [(a, b) for a, b in _stable_regions(row) if 0 <= a < b <= duration_sec]
    if domain is None and not regions and not any("stable_region_start" in trial and
                                  "stable_region_end" in trial for trial in trials):
        raise ValueError("stable_phonation_region_unavailable")
    exclusions = [(float(item.start_sec), float(item.end_sec)) for item in
                  intervals.loc[intervals.segment_role.eq("manual_exclusion")].itertuples()]
    spans = []
    speech = intervals.loc[intervals.segment_role.eq("speech")].sort_values("start_sec")
    for number, item in enumerate(speech.itertuples(), 1):
        left, right = float(item.start_sec), float(item.end_sec)
        if left >= right:
            continue
        own = [region for region in regions if left <= region[0] < region[1] <= right]
        repetition = {"trial_id": f"repetition_{number:02d}" if regions else "",
                      "start_sec": left, "end_sec": right}
        if own:
            repetition.update(stable_region_start=own[0][0], stable_region_end=own[0][1])
        elif regions:
            continue  # a repetition without a usable steady window is not measured
        trial_domains = trials or [repetition]
        for trial in trial_domains:
            trial_id = str(trial["trial_id"])
            trial_start, trial_end = float(trial["start_sec"]), float(trial["end_sec"])
            stable = ((float(trial["stable_region_start"]), float(trial["stable_region_end"]))
                      if "stable_region_start" in trial and "stable_region_end" in trial
                      else domain)
            if stable is None:
                continue
            a, b = max(left, trial_start, stable[0]), min(right, trial_end, stable[1])
            fragments = [(a, b)] if a < b else []
            for excluded_start, excluded_end in exclusions:
                fragments = ([(x, min(y, excluded_start)) for x, y in fragments
                              if x < min(y, excluded_start)] +
                             [(max(x, excluded_end), y) for x, y in fragments
                              if max(x, excluded_end) < y])
            spans.extend((x, y, trial_id, trial_start, trial_end) for x, y in fragments)
    by_trial: dict[str, list[tuple[float, float, str, float, float]]] = {}
    for span in spans:
        by_trial.setdefault(span[2], []).append(span)
    centered = []
    for candidates in by_trial.values():
        a, b, trial_id, trial_start, trial_end = max(
            candidates, key=lambda span: (span[1] - span[0], -span[0]))
        width, middle = min(2.0, b - a), (a + b) / 2.0
        centered.append((middle - width / 2.0, middle + width / 2.0,
                         trial_id, trial_start, trial_end))
    if not centered:
        raise ValueError("final_reviewed_speech_region_unavailable")
    return sorted(centered)
