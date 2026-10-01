"""Persist human acceptance separately from MFA execution status."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


STATUSES = frozenset({"UNREVIEWED", "ACCEPTED", "AUTO_ACCEPTED_STRUCTURAL",
                      "NEEDS_REVIEW", "STALE_AFTER_TRIAL_EDIT"})


def trial_review_table(root: str | Path, run_id: str) -> pd.DataFrame:
    base = Path(root) / "acoustic" / "004_alignment" / "runs" / run_id / "tables"
    diagnostics = base / "alignment_trial_diagnostics.csv"
    if not diagnostics.is_file():
        return pd.DataFrame(columns=["recording_id", "trial_id", "review_status", "reviewed_utc"])
    trials = pd.read_csv(diagnostics, keep_default_na=False)[["recording_id", "trial_id"]]
    path = base / "alignment_trial_review.csv"
    if path.is_file():
        saved = pd.read_csv(path, keep_default_na=False)
        trials = trials.merge(saved, on=["recording_id", "trial_id"], how="left")
    if "review_status" not in trials:
        trials["review_status"] = "UNREVIEWED"
    if "reviewed_utc" not in trials:
        trials["reviewed_utc"] = ""
    trials["review_status"] = trials.review_status.fillna("UNREVIEWED")
    trials["reviewed_utc"] = trials.reviewed_utc.fillna("")
    return trials


def set_trial_review(root: str | Path, run_id: str, recording_id: str,
                     trial_id: str, status: str) -> None:
    if status not in {"ACCEPTED", "NEEDS_REVIEW"}:
        raise ValueError("invalid_alignment_review_status")
    base = Path(root) / "acoustic" / "004_alignment"
    frozen = base / "final" / "final_alignment_manifest.json"
    if (frozen.exists() and
            json.loads(frozen.read_text(encoding="utf-8")).get("alignment_run_id") == run_id):
        raise ValueError("frozen_alignment_is_immutable")
    table = trial_review_table(root, run_id)
    matches = table.recording_id.astype(str).eq(recording_id) & table.trial_id.astype(str).eq(trial_id)
    if matches.sum() != 1:
        raise ValueError("unknown_alignment_trial")
    if table.loc[matches, "review_status"].iloc[0] == "STALE_AFTER_TRIAL_EDIT":
        raise ValueError("stale_alignment_requires_rerun")
    if status == "ACCEPTED":
        from .corrections import reviewed_tables
        words, phones = reviewed_tables(root, run_id)
        links = pd.read_csv(base / "runs" / run_id / "tables" /
                            "alignment_trial_tokens.csv", keep_default_na=False)
        linked = links.loc[links.recording_id.astype(str).eq(recording_id)
                           & links.trial_id.astype(str).eq(trial_id)]
        subset = {}
        for kind, source in (("word", words), ("phone", phones)):
            indices = linked.loc[linked.token_type.eq(kind), "token_index"]
            rows = source.loc[source.recording_id.astype(str).eq(recording_id)
                              & source[f"{kind}_index"].isin(indices)].sort_values("start_sec")
            if (rows.empty or not (rows.end_sec.astype(float) > rows.start_sec.astype(float)).all()
                    or any(float(a) > float(b) for a, b in zip(
                        rows.end_sec.iloc[:-1], rows.start_sec.iloc[1:]))):
                raise ValueError("corrected_alignment_boundaries_invalid")
            subset[kind] = rows
        for phone in subset["phone"].itertuples():
            owners = subset["word"].loc[subset["word"].word_index.astype(int).eq(
                int(phone.word_index))]
            if len(owners) != 1 or not (float(owners.iloc[0].start_sec) <= float(phone.start_sec)
                                        < float(phone.end_sec) <= float(owners.iloc[0].end_sec)):
                raise ValueError("corrected_phone_word_relationship_invalid")
    table.loc[matches, "review_status"] = status
    table.loc[matches, "reviewed_utc"] = datetime.now(timezone.utc).isoformat()
    for name, value in (("review_mode", "HUMAN"), ("review_policy_version", "")):
        if name not in table:
            table[name] = ""
        table.loc[matches, name] = value
    table.to_csv(base / "runs" / run_id / "tables" / "alignment_trial_review.csv", index=False)
    from .corrections import correction_path, load_corrections
    correction_file = correction_path(root, run_id)
    if correction_file.is_file():
        corrections = load_corrections(root, run_id)
        selected = (corrections.recording_id.astype(str).eq(recording_id)
                    & corrections.trial_id.astype(str).eq(trial_id))
        corrections.loc[selected, "reviewer_state"] = (
            "APPROVED" if status == "ACCEPTED" else "PENDING_ACCEPTANCE")
        corrections.to_csv(correction_file, index=False)


def mark_recording_stale(root: str | Path, run_id: str, recording_id: str) -> None:
    """A changed trial plan cannot be accepted against the old MFA output."""
    base = Path(root) / "acoustic" / "004_alignment" / "runs" / run_id / "tables"
    table = trial_review_table(root, run_id)
    mask = table.recording_id.astype(str).eq(str(recording_id))
    if mask.any():
        table.loc[mask, "review_status"] = "STALE_AFTER_TRIAL_EDIT"
        table.loc[mask, "reviewed_utc"] = datetime.now(timezone.utc).isoformat()
        table.to_csv(base / "alignment_trial_review.csv", index=False)
