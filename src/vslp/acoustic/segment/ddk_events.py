"""Reviewed DDK syllable intervals -> final_ddk_events.csv (the F10 input contract).

Finalizing segmentation review used to freeze DDK intervals without writing
the event table that the DDK features (F10) read. As a result, DDK rate and
cycle variability could never be computed. This module writes that table from
the reviewed intervals.

- One row per syllable for /pa/, /ta/ and /ka/ (AMR).
- One row per complete pa-ta-ka triplet for /pataka/ (SMR). Syllables are
  labelled from their release bursts (see ddk_v2.label_pataka) and grouped into
  triplets in order. Incomplete groups are listed but not valid.
- Each event's centre is the energy peak (syllable nucleus) inside the reviewed
  interval, so manual corrections are handled the same way.
- Sequences are split at gaps > max(1 s, 3 x median), as in ddk_v2. Only the
  longest sequence is valid; the rest are kept with a reason.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .ddk_v2 import _burst_balance, _contour, label_pataka

EVENT_COLUMNS = [
    "recording_id", "file_name", "trial_id", "sequence_id", "task", "ddk_target", "ddk_mode",
    "event_id", "event_index", "start_sec", "center_sec", "end_sec", "valid", "review_status",
    "reviewer", "rejection_reason", "sequence_label", "sequence_complete", "syllable_labels",
    "timing_source",
]
MODES = {"pa": "AMR", "ta": "AMR", "ka": "AMR", "pataka": "SMR"}


def ddk_target(file_name: str, task_name: str = "") -> str:
    """pa / ta / ka / pataka from the task name or file-name code (e.g. 242_DDK_PA)."""
    text = f"{task_name} {file_name}".casefold()
    if "pataka" in text or "pa-ta-ka" in text or "pa ta ka" in text:
        return "pataka"
    for target in ("pa", "ta", "ka"):
        if re.search(rf"(?:^|[^a-z]){target}(?:[^a-z]|$)", text):
            return target
    return ""


def _nucleus(x: np.ndarray, sr: int, start: float, end: float) -> float:
    a, b = int(start * sr), max(int(start * sr) + 1, int(end * sr))
    segment = np.asarray(x[a:b], dtype=np.float64)
    window = max(1, int(0.02 * sr))
    if len(segment) <= window:
        return (start + end) / 2
    power = np.convolve(segment ** 2, np.ones(window) / window, mode="same")
    center = start + int(np.argmax(power)) / sr
    return float(min(max(center, start + 1e-4), end - 1e-4))


def _sequences(centers: np.ndarray) -> list[list[int]]:
    if not len(centers):
        return []
    gaps = np.diff(centers)
    limit = max(1.0, 3 * float(np.median(gaps))) if len(gaps) else np.inf
    groups, current = [], [0]
    for index, gap in enumerate(gaps, 1):
        if gap > limit:
            groups.append(current)
            current = [index]
        else:
            current.append(index)
    groups.append(current)
    return groups


def ddk_event_rows(x: np.ndarray, sr: int, intervals: list[tuple[float, float]], *,
                   recording_id: str, file_name: str, task_name: str = "",
                   reviewer: str = "") -> list[dict]:
    target = ddk_target(file_name, task_name)
    if not target or not intervals:
        return []
    intervals = sorted((float(a), float(b)) for a, b in intervals if b > a)
    centers = np.array([_nucleus(x, sr, a, b) for a, b in intervals])
    status = "HUMAN_ACCEPTED" if reviewer else "FINAL_ACCEPTED"
    groups = _sequences(centers)
    main = max(groups, key=len)
    base = {"recording_id": recording_id, "file_name": file_name, "trial_id": "trial_01",
            "task": "ddk", "ddk_target": target, "ddk_mode": MODES[target],
            "review_status": status, "reviewer": reviewer,
            "timing_source": "reviewed_interval_energy_peak"}
    rows: list[dict] = []
    if target != "pataka":
        for number, group in enumerate(groups, 1):
            for index in group:
                a, b = intervals[index]
                rows.append({**base, "sequence_id": f"{recording_id}:seq_{number:02d}",
                             "event_id": f"{recording_id}:syl_{index + 1:03d}",
                             "event_index": index + 1, "start_sec": a,
                             "center_sec": centers[index], "end_sec": b,
                             "valid": group is main,
                             "rejection_reason": "" if group is main else "outside_main_ddk_sequence",
                             "sequence_label": target, "sequence_complete": True,
                             "syllable_labels": target})
        return rows
    work = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    _centers, _contour_db, band = _contour(work, sr, 20.0, 5.0)
    event_number = 0
    for number, group in enumerate(groups, 1):
        balances = [_burst_balance(band, sr, int(intervals[i][0] * sr), int(centers[i] * sr))
                    for i in group]
        labels, fit = label_pataka(balances)
        position = 0
        while position < len(group):
            chunk = group[position:position + 3]
            chunk_labels = labels[position:position + 3]
            complete = chunk_labels == ["pa", "ta", "ka"]
            if not complete and chunk_labels and chunk_labels[0] != "pa":
                # Realign to the next "pa" so triplets stay pa-ta-ka.
                skip = 1
                event_number += 1
                index = group[position]
                rows.append({**base, "sequence_id": f"{recording_id}:seq_{number:02d}",
                             "event_id": f"{recording_id}:pataka_{event_number:03d}",
                             "event_index": event_number, "start_sec": intervals[index][0],
                             "center_sec": centers[index], "end_sec": intervals[index][1],
                             "valid": False, "rejection_reason": "incomplete_pataka_sequence",
                             "sequence_label": labels[position], "sequence_complete": False,
                             "syllable_labels": labels[position]})
                position += skip
                continue
            event_number += 1
            first, last = chunk[0], chunk[-1]
            middle = chunk[1] if len(chunk) > 1 else chunk[0]
            valid = complete and group is main
            reason = ("" if valid else "incomplete_pataka_sequence" if not complete
                      else "outside_main_ddk_sequence")
            rows.append({**base, "sequence_id": f"{recording_id}:seq_{number:02d}",
                         "event_id": f"{recording_id}:pataka_{event_number:03d}",
                         "event_index": event_number, "start_sec": intervals[first][0],
                         "center_sec": centers[middle], "end_sec": intervals[last][1],
                         "valid": valid, "rejection_reason": reason,
                         "sequence_label": "pa-ta-ka" if complete else "-".join(chunk_labels),
                         "sequence_complete": complete,
                         "syllable_labels": "-".join(chunk_labels),
                         "pataka_label_fit": fit})
            position += len(chunk)
    return rows


def ddk_events_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    columns = EVENT_COLUMNS + [c for c in frame.columns if c not in EVENT_COLUMNS]
    return frame.reindex(columns=columns)
