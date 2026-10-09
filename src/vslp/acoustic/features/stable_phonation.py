"""Reviewed, original-time stable phonation spans shared by F01 and F04."""

from __future__ import annotations

import pandas as pd


def centered_stable_spans(row, intervals: pd.DataFrame, duration_sec: float,
                          trials: list[dict]) -> list[tuple[float, float, str, float, float]]:
    """Return one centred, exclusion-free span per trial without acoustic tracking."""
    start, end = float(row.stable_region_start), float(row.stable_region_end)
    domain = (start, end) if 0 <= start < end <= duration_sec else None
    if domain is None and not any("stable_region_start" in trial and
                                  "stable_region_end" in trial for trial in trials):
        raise ValueError("stable_phonation_region_unavailable")
    exclusions = [(float(item.start_sec), float(item.end_sec)) for item in
                  intervals.loc[intervals.segment_role.eq("manual_exclusion")].itertuples()]
    spans = []
    speech = intervals.loc[intervals.segment_role.eq("speech")].sort_values("start_sec")
    for item in speech.itertuples():
        left, right = float(item.start_sec), float(item.end_sec)
        if left >= right:
            continue
        trial_domains = trials or [{"trial_id": "", "start_sec": left, "end_sec": right}]
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
