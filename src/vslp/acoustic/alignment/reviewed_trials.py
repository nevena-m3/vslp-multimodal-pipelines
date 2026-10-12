"""Reviewed Trial Confirmation Contract V1, independent of Alignment GUI/MFA."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path


CONTRACT_VERSION = "REVIEWED_TRIAL_CONFIRMATION_V1"


def segmentation_review_hash(root: str | Path) -> str:
    """Hash the frozen decisions and intervals in their established byte order."""
    final = Path(root) / "acoustic" / "003_segmentation_review" / "final"
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    if not decisions.is_file() or not intervals.is_file():
        raise ValueError("missing_frozen_segmentation_review")
    return sha256(decisions.read_bytes() + intervals.read_bytes()).hexdigest()


def confirmed_trial_rows(root: str | Path, *, recording_id: str, task_id: str,
                         prompt_id: str, bounds: list[tuple[float, float]],
                         trial_ids: list[str] | None = None) -> list[dict]:
    """Persist actual reviewed bounds; protocol repetition count never creates rows."""
    digest = segmentation_review_hash(root)
    timestamp = datetime.now(timezone.utc).isoformat()
    trial_ids = trial_ids or [f"trial_{index:03d}" for index in range(1, len(bounds) + 1)]
    if len(trial_ids) != len(bounds) or len(set(trial_ids)) != len(trial_ids):
        raise ValueError("invalid_trial_ids")
    rows = []
    for trial_id, (start, end) in zip(trial_ids, bounds, strict=True):
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("invalid_trial_bounds")
        rows.append({"trial_id": trial_id,
                     "sequence_id": f"{recording_id}:{trial_id}",
                     "task_id": task_id, "prompt_id": prompt_id,
                     "start_sec": start, "end_sec": end,
                     "trial_status": "READY_FOR_ALIGNMENT",
                     "trial_review_source": "HUMAN",
                     "segmentation_review_hash": digest,
                     "trial_review_timestamp": timestamp,
                     "reviewed_trial_contract_version": CONTRACT_VERSION})
    return rows


def load_confirmed_trials(root: str | Path, recording_id: str,
                          task_id: str) -> tuple[list[dict], str]:
    """Reject legacy, stale, and nonhuman rows for downstream reviewed timing."""
    root = Path(root)
    path = root / "configs" / "alignment_choices.json"
    if not path.is_file():
        return [], "missing_trial_bounds"
    choices = json.loads(path.read_text(encoding="utf-8"))
    if (choices.get("pending_trial_revision")
            or choices.get("trial_edit_pending", {}).get(recording_id)):
        return [], "invalid_trial_bounds"
    trials = choices.get("trials", {}).get(recording_id, [])
    if not trials:
        return [], "missing_trial_bounds"
    try:
        current_hash = segmentation_review_hash(root)
    except ValueError:
        return [], "missing_frozen_segmentation_review"
    verified = []
    for trial in trials:
        required = ("trial_id", "sequence_id", "task_id", "prompt_id", "start_sec",
                    "end_sec", "trial_status", "trial_review_source",
                    "segmentation_review_hash", "reviewed_trial_contract_version")
        if any(key not in trial for key in required):
            return [], "legacy_unverified_trial_confirmation"
        if trial["reviewed_trial_contract_version"] != CONTRACT_VERSION:
            return [], "legacy_unverified_trial_confirmation"
        if trial["trial_review_source"] != "HUMAN":
            return [], "trial_not_human_confirmed"
        if trial["trial_status"] != "READY_FOR_ALIGNMENT":
            return [], "trial_not_ready_for_alignment"
        if trial["segmentation_review_hash"] != current_hash:
            return [], "stale_trial_confirmation"
        if trial["task_id"] != task_id or not trial["trial_id"] or not trial["sequence_id"]:
            return [], "invalid_trial_identity"
        try:
            start, end = float(trial["start_sec"]), float(trial["end_sec"])
        except (TypeError, ValueError):
            return [], "invalid_trial_bounds"
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            return [], "invalid_trial_bounds"
        verified.append({**trial, "start_sec": start, "end_sec": end})
    return verified, ""
