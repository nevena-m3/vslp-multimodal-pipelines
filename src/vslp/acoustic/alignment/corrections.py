"""Reviewer boundary overlays; immutable MFA tables remain the original evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = ("recording_id", "trial_id", "tier", "token_index", "token_label",
           "original_start_sec", "original_end_sec", "corrected_start_sec",
           "corrected_end_sec", "reviewer_state", "correction_utc", "reason")


def correction_path(root: str | Path, run_id: str) -> Path:
    return Path(root) / "acoustic" / "004_alignment" / "runs" / run_id / "tables" / "alignment_manual_corrections.csv"


def _ensure_editable(root: Path, run_id: str) -> None:
    frozen = root / "acoustic" / "004_alignment" / "final" / "final_alignment_manifest.json"
    if (frozen.is_file() and
            json.loads(frozen.read_text(encoding="utf-8")).get("alignment_run_id") == run_id):
        raise ValueError("frozen_alignment_is_immutable")


def load_corrections(root: str | Path, run_id: str) -> pd.DataFrame:
    path = correction_path(root, run_id)
    return pd.read_csv(path, keep_default_na=False) if path.is_file() else pd.DataFrame(columns=COLUMNS)


def _require_reacceptance(root: str | Path, run_id: str, recording_id: str,
                          trial_id: str) -> None:
    path = correction_path(root, run_id).with_name("alignment_trial_review.csv")
    if not path.is_file():
        return
    table = pd.read_csv(path, keep_default_na=False)
    mask = table.recording_id.astype(str).eq(str(recording_id)) & table.trial_id.astype(str).eq(
        str(trial_id)) & table.review_status.isin(["ACCEPTED", "AUTO_ACCEPTED_STRUCTURAL"])
    if mask.any():
        table.loc[mask, "review_status"] = "NEEDS_REVIEW"
        table.loc[mask, "reviewed_utc"] = ""
        if "review_mode" in table:
            table.loc[mask, "review_mode"] = "HUMAN_CORRECTION_PENDING"
        if "queued_for_review" in table:
            table.loc[mask, "queued_for_review"] = True
        table.to_csv(path, index=False)
        corrections = load_corrections(root, run_id)
        selected = (corrections.recording_id.astype(str).eq(str(recording_id))
                    & corrections.trial_id.astype(str).eq(str(trial_id)))
        corrections.loc[selected, "reviewer_state"] = "PENDING_ACCEPTANCE"
        corrections.to_csv(correction_path(root, run_id), index=False)


def _immutable_token(root: Path, run_id: str, recording_id: str, tier: str,
                     token_index: int) -> pd.Series:
    if tier not in {"word", "phone"}:
        raise ValueError("invalid_alignment_tier")
    table = pd.read_csv(root / "acoustic" / "004_alignment" / "runs" / run_id /
                        "tables" / f"alignment_{tier}s.csv", keep_default_na=False)
    key = f"{tier}_index"
    found = table.loc[table.recording_id.astype(str).eq(str(recording_id))
                      & table[key].astype(int).eq(int(token_index))]
    if len(found) != 1:
        raise ValueError("unknown_alignment_token")
    return found.iloc[0]


def set_correction(root: str | Path, run_id: str, recording_id: str, trial_id: str,
                   tier: str, token_index: int, start_sec: float, end_sec: float,
                   reason: str = "") -> None:
    root = Path(root)
    _ensure_editable(root, run_id)
    original = _immutable_token(root, run_id, recording_id, tier, token_index)
    start, end = float(start_sec), float(end_sec)
    if not np.isfinite([start, end]).all() or not 0 <= start < end:
        raise ValueError("invalid_corrected_boundary")
    trials = pd.read_csv(root / "acoustic" / "004_alignment" / "runs" / run_id /
                         "tables" / "alignment_trial_diagnostics.csv", keep_default_na=False)
    trial = trials.loc[trials.recording_id.astype(str).eq(str(recording_id))
                       & trials.trial_id.astype(str).eq(str(trial_id))]
    if len(trial) != 1 or not (float(trial.iloc[0].start_sec) <= start < end <=
                              float(trial.iloc[0].end_sec)):
        raise ValueError("correction_outside_trial")
    intervals = pd.read_csv(root / "acoustic" / "003_segmentation_review" / "final" /
                            "final_segmentation_intervals.csv", keep_default_na=False)
    excluded = intervals.loc[intervals.recording_id.astype(str).eq(str(recording_id))
                             & intervals.segment_role.eq("manual_exclusion")]
    if any(float(row.start_sec) < end and float(row.end_sec) > start
           for row in excluded.itertuples()):
        raise ValueError("correction_crosses_manual_exclusion")
    rows = load_corrections(root, run_id)
    mask = (rows.recording_id.astype(str).eq(str(recording_id))
            & rows.trial_id.astype(str).eq(str(trial_id))
            & rows.tier.eq(tier) & rows.token_index.astype(int).eq(int(token_index)))
    rows = rows.loc[~mask]
    label = str(original["word" if tier == "word" else "phone"])
    new = {"recording_id": recording_id, "trial_id": trial_id, "tier": tier,
           "token_index": token_index, "token_label": label,
           "original_start_sec": float(original.start_sec),
           "original_end_sec": float(original.end_sec),
           "corrected_start_sec": start, "corrected_end_sec": end,
           "reviewer_state": "PENDING_ACCEPTANCE",
           "correction_utc": datetime.now(timezone.utc).isoformat(), "reason": reason}
    pd.concat([rows, pd.DataFrame([new])], ignore_index=True).to_csv(
        correction_path(root, run_id), index=False)
    _require_reacceptance(root, run_id, recording_id, trial_id)


def reset_corrections(root: str | Path, run_id: str, recording_id: str,
                      trial_id: str, tier: str = "", token_index: int | None = None) -> None:
    _ensure_editable(Path(root), run_id)
    rows = load_corrections(root, run_id)
    if rows.empty:
        return
    mask = (rows.recording_id.astype(str).eq(str(recording_id))
            & rows.trial_id.astype(str).eq(str(trial_id)))
    if tier:
        mask &= rows.tier.eq(tier)
    if token_index is not None:
        mask &= rows.token_index.astype(int).eq(int(token_index))
    if mask.any():
        rows.loc[~mask].to_csv(correction_path(root, run_id), index=False)
        _require_reacceptance(root, run_id, recording_id, trial_id)


def reviewed_tables(root: str | Path, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(root) / "acoustic" / "004_alignment" / "runs" / run_id / "tables"
    words = pd.read_csv(base / "alignment_words.csv", keep_default_na=False)
    phones = pd.read_csv(base / "alignment_phones.csv", keep_default_na=False)
    diagnostics = pd.read_csv(base / "alignment_diagnostics.csv", keep_default_na=False)
    for correction in load_corrections(root, run_id).itertuples():
        table = words if correction.tier == "word" else phones
        key = f"{correction.tier}_index"
        mask = (table.recording_id.astype(str).eq(str(correction.recording_id))
                & table[key].astype(int).eq(int(correction.token_index)))
        if mask.sum() != 1:
            raise ValueError("correction_token_mismatch")
        source = table.loc[mask].iloc[0]
        if (float(source.start_sec) != float(correction.original_start_sec) or
                float(source.end_sec) != float(correction.original_end_sec)):
            raise ValueError("correction_original_boundary_mismatch")
        table.loc[mask, "start_sec"] = float(correction.corrected_start_sec)
        table.loc[mask, "end_sec"] = float(correction.corrected_end_sec)
        table.loc[mask, "duration_sec"] = (float(correction.corrected_end_sec) -
                                           float(correction.corrected_start_sec))
        rates = diagnostics.loc[diagnostics.recording_id.astype(str).eq(
            str(correction.recording_id)), "canonical_sample_rate_hz"]
        if len(rates) != 1 or int(rates.iloc[0]) <= 0:
            raise ValueError("correction_native_sample_rate_missing")
        rate = int(rates.iloc[0])
        table.loc[mask, "start_sample"] = round(float(correction.corrected_start_sec) * rate)
        table.loc[mask, "end_sample"] = round(float(correction.corrected_end_sec) * rate)
        table.loc[mask, "alignment_source"] = "reviewed_manual_correction"
    return words, phones
