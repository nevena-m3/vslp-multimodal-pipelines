"""Frozen downstream F10 measurements from accepted, reviewed DDK events."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "family10-reviewed-events-2.0.0"
PARAMETER_SET_ID = "family10_first10_reviewed_v2"
TARGETS = {"pa": "AMR", "ta": "AMR", "ka": "AMR", "pataka": "SMR"}
RATE_IDS = {target: f"ddk_{target}_rate_syll_s" for target in TARGETS}
VARIABILITY_IDS = {
    target: f"ddk_{target}_cycle_temporal_variability_s" for target in TARGETS
}
PUBLIC_IDS = frozenset([*RATE_IDS.values(), *VARIABILITY_IDS.values()])
ACCEPTED_STATUSES = frozenset({"HUMAN_ACCEPTED", "REVIEWED_ACCEPTED", "FINAL_ACCEPTED"})
REQUIRED_COLUMNS = frozenset({
    "recording_id", "trial_id", "sequence_id", "task", "ddk_target", "ddk_mode",
    "event_index", "start_sec", "center_sec", "end_sec", "valid", "review_status",
    "rejection_reason",
})
AUDIT_COLUMNS = (
    "recording_id", "trial_id", "sequence_id", "task", "ddk_target", "ddk_mode",
    "event_id", "event_index", "selected_rank", "start_sec", "center_sec",
    "end_sec", "duration_sec", "valid", "review_status", "reviewer", "confidence",
    "sequence_label",
    "sequence_complete", "rejection_reason", "selected", "selection_reason",
)


def _bool(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _failure(reason: str) -> dict[str, tuple[float, str]]:
    return {"rate": (math.nan, reason), "variability": (math.nan, reason)}


def measure_trial(events: pd.DataFrame, target: str) -> tuple[dict, pd.DataFrame, dict]:
    """Select chronological first ten accepted cycles in one trial and one sequence."""
    target = str(target).strip().casefold().strip("/")
    if target not in TARGETS:
        return _failure("unsupported_ddk_target"), pd.DataFrame(columns=AUDIT_COLUMNS), {}
    if not REQUIRED_COLUMNS <= set(events):
        return _failure("invalid_ddk_event"), pd.DataFrame(columns=AUDIT_COLUMNS), {}
    own = events.copy()
    own["ddk_target"] = own.ddk_target.astype(str).str.casefold().str.strip("/")
    own = own.loc[own.ddk_target.eq(target)].copy()
    if own.empty:
        return _failure("missing_reviewed_ddk_events"), pd.DataFrame(columns=AUDIT_COLUMNS), {}
    own["reviewer"] = own.reviewer.astype(str) if "reviewer" in own else ""
    own["confidence"] = (pd.to_numeric(own.confidence, errors="coerce")
                         if "confidence" in own else np.nan)
    own["event_id"] = (own.event_id.astype(str) if "event_id" in own
                       else own.event_index.astype(str))
    own["sequence_label"] = (own.sequence_label.astype(str) if "sequence_label" in own
                             else "")
    own["sequence_complete"] = (own.sequence_complete.map(_bool)
                                 if "sequence_complete" in own else False)
    own["selected"] = False
    own["selected_rank"] = pd.NA
    own["selection_reason"] = "not_accepted"
    own["valid"] = own.valid.map(_bool)
    own["review_status"] = own.review_status.astype(str).str.upper().str.strip()
    accepted = (own.valid & own.review_status.isin(ACCEPTED_STATUSES)
                & own.task.astype(str).str.casefold().eq("ddk"))
    accepted &= own.ddk_mode.astype(str).str.upper().eq(TARGETS[target])
    if target == "pataka":
        accepted &= own.sequence_complete & own.sequence_label.eq("pa-ta-ka")
    own.loc[own.valid & ~accepted & own.ddk_mode.astype(str).str.upper().ne(TARGETS[target]),
            "selection_reason"] = "invalid_ddk_event"
    if target == "pataka":
        own.loc[own.valid & ~own.sequence_complete, "selection_reason"] = (
            "incomplete_pataka_sequence")
    count_valid = int(accepted.sum())
    own.loc[accepted, "selection_reason"] = "after_first_10"
    summary = {
        "n_detected_events": len(own), "n_valid_events": count_valid,
        "analysis_target_n": 10, "n_events_used": 0,
        "n_detected_sequences": len(own) if target == "pataka" else 0,
        "n_complete_valid_sequences": count_valid if target == "pataka" else 0,
        "n_sequences_used": 0, "syllable_equivalent_count": 0,
    }
    if count_valid < 10:
        return _failure("insufficient_valid_ddk_events"), own.reindex(columns=AUDIT_COLUMNS), summary
    for column in ("start_sec", "center_sec", "end_sec"):
        own[column] = pd.to_numeric(own[column], errors="coerce")
    eligible = own.loc[accepted].sort_values(
        ["center_sec", "event_index", "event_id"], kind="stable").head(10)
    times = eligible[["start_sec", "center_sec", "end_sec"]].to_numpy(dtype=float)
    if not np.isfinite(times).all() or (times[:, 0] >= times[:, 1]).any() or (
            times[:, 1] >= times[:, 2]).any() or (times[:, 0] < 0).any():
        return _failure("invalid_ddk_event_bounds"), own.reindex(columns=AUDIT_COLUMNS), summary
    centers = times[:, 1]
    if (np.diff(centers) <= 0).any():
        return _failure("invalid_ddk_event_order"), own.reindex(columns=AUDIT_COLUMNS), summary
    if eligible.sequence_id.astype(str).nunique() != 1:
        return _failure("invalid_ddk_event_order"), own.reindex(columns=AUDIT_COLUMNS), summary
    duration = float(times[-1, 2] - times[0, 0])
    if duration <= 0:
        return _failure("analyzed_duration_nonpositive"), own.reindex(columns=AUDIT_COLUMNS), summary
    intervals = np.diff(centers)
    if (intervals <= 0).any():
        return _failure("nonpositive_cycle_interval"), own.reindex(columns=AUDIT_COLUMNS), summary
    multiplier = 3 if target == "pataka" else 1
    rate = 10 * multiplier / duration
    variability = float(np.mean(np.abs(np.diff(intervals))))
    if not np.isfinite(rate):
        return _failure("nonfinite_ddk_rate"), own.reindex(columns=AUDIT_COLUMNS), summary
    if not np.isfinite(variability):
        return _failure("nonfinite_ddk_temporal_variability"), own.reindex(columns=AUDIT_COLUMNS), summary
    own.loc[eligible.index, "selected"] = True
    own.loc[eligible.index, "selected_rank"] = range(1, 11)
    own.loc[eligible.index, "selection_reason"] = "first_10_valid"
    own["duration_sec"] = own.end_sec - own.start_sec
    summary.update(n_events_used=10, n_sequences_used=10 if target == "pataka" else 0,
                   syllable_equivalent_count=10 * multiplier,
                   analyzed_duration_sec=duration,
                   rate_denominator="first_accepted_onset_to_tenth_accepted_offset")
    return {"rate": (rate, ""), "variability": (variability, "")}, (
        own.reindex(columns=AUDIT_COLUMNS)), summary


def load_reviewed_ddk_events(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Read the explicit final review product; never derive events from candidates."""
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(columns=AUDIT_COLUMNS), "missing_reviewed_ddk_events"
    try:
        events = pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame(columns=AUDIT_COLUMNS), "invalid_ddk_event"
    if not REQUIRED_COLUMNS <= set(events):
        return pd.DataFrame(columns=AUDIT_COLUMNS), "invalid_ddk_event"
    return events, ""
