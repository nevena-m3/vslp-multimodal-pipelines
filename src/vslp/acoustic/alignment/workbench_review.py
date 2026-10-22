"""Non-destructive reviewer corrections for standalone MFA jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .mfa_provider import parse_textgrid
from .workbench import load_job


def correction_path(job_root: str | Path, record_key: str) -> Path:
    return Path(job_root) / "review" / f"{record_key}_corrections.json"


def load_corrections(job_root: str | Path, record_key: str) -> dict:
    path = correction_path(job_root, record_key)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "record_key": record_key, "removed_word_indices": [], "word_bounds": {},
        "phone_bounds": {}, "issue": "", "note": "", "reviewer": "",
        "decision": "",
        "history": []}


def save_corrections(job_root: str | Path, record_key: str, changes: dict,
                     reviewer: str, reason: str) -> Path:
    if not reviewer.strip() or not reason.strip():
        raise ValueError("reviewer_and_reason_required")
    path = correction_path(job_root, record_key)
    before = load_corrections(job_root, record_key)
    data = {**before, **changes, "record_key": record_key, "reviewer": reviewer.strip()}
    data["history"] = [*before.get("history", []), {
        "at_utc": datetime.now(timezone.utc).isoformat(), "reviewer": reviewer.strip(),
        "reason": reason.strip(), "previous": {key: before.get(key)
                                         for key in changes}}]
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def reviewed_tokens(job_root: str | Path, record_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    job = load_job(job_root)
    row = next((item for item in job["recordings"] if item["record_key"] == record_key), None)
    if row is None or not row.get("textgrid_path"):
        raise ValueError("reviewed_recording_unavailable")
    words, phones = parse_textgrid(Path(row["textgrid_path"]))
    offset = float(row.get("alignment_scope", {}).get("start_sec", 0.0))
    for table in (words, phones):
        if not table.empty:
            table[["start_sec", "end_sec"]] = table[["start_sec", "end_sec"]].astype(float) + offset
    edits = load_corrections(job_root, record_key)
    words = words.copy()
    phones = phones.copy()
    words["provider_index"] = range(1, len(words) + 1)
    phones["provider_index"] = range(1, len(phones) + 1)
    for kind, table in (("word", words), ("phone", phones)):
        for index_text, pair in edits.get(f"{kind}_bounds", {}).items():
            index = int(index_text)
            match = table.provider_index.eq(index)
            if match.sum() != 1 or len(pair) != 2 or not 0 <= float(pair[0]) < float(pair[1]):
                raise ValueError("invalid_reviewed_boundary_correction")
            table.loc[match, ["start_sec", "end_sec"]] = [float(pair[0]), float(pair[1])]
    removed = set(map(int, edits.get("removed_word_indices", [])))
    if removed:
        raise ValueError("lexical_edit_requires_realign")
    return words.reset_index(drop=True), phones.reset_index(drop=True)
