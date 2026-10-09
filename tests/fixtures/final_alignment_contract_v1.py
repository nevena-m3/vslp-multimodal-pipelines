"""Write a loadable, nonclinical final Alignment package without MFA."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from vslp.acoustic.alignment.final_contract import (
    FINAL_ALIGNMENT_CONTRACT_VERSION, build_feature_tables, canonical_units,
)
from vslp.acoustic.alignment.stage import (DIAGNOSTIC_COLUMNS, PHONE_COLUMNS,
                                           WORD_COLUMNS, normalize_transcript)
from vslp.acoustic.alignment.task_workflow import task_registry


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_nonclinical_final_alignment(root: Path, task_id: str,
                                      repetitions: int = 1,
                                      corrected_phone: bool = False) -> Path:
    """Create WSTG/S3/S5/Bamboo token fixtures for downstream family tests."""
    task = next(item for item in task_registry()["tasks"] if item["task_id"] == task_id)
    prompt = task["prompts"][0]
    trial_rows, link_rows, word_rows, phone_rows, reviews = [], [], [], [], []
    index = 0
    for trial_number in range(1, repetitions + 1):
        trial_id = f"trial_{trial_number:03d}"
        trial_rows.append({"recording_id": "nonclinical", "trial_id": trial_id,
                           "prompt_id": prompt["prompt_id"],
                           "start_sec": 10. * trial_number,
                           "end_sec": 10. * trial_number + 8.})
        reviews.append({"recording_id": "nonclinical", "trial_id": trial_id,
                        "alignment_review_status": "HUMAN_ACCEPTED"})
        for unit in canonical_units(prompt["exact_expected_text"]):
            for part in normalize_transcript(unit).split():
                index += 1
                start = 10. * trial_number + .02 * (index % 100)
                word_rows.append({"recording_id": "nonclinical", "word_index": index,
                                  "word": part, "start_sec": start,
                                  "end_sec": start + .015,
                                  "alignment_source": "synthetic_fixture"})
                phone_rows.append({"recording_id": "nonclinical", "word_index": index,
                                   "phone_index": index, "phone": "UW1",
                                   "phone_raw": "UW1", "phone_normalized": "UW",
                                   "stress": "1", "is_vowel": True,
                                   "start_sec": start, "end_sec": start + .015,
                                   "alignment_source": (
                                       "reviewed_manual_correction" if corrected_phone and index == 1
                                       else "synthetic_fixture")})
                link_rows.extend({"recording_id": "nonclinical", "trial_id": trial_id,
                                  "token_type": kind, "token_index": index}
                                 for kind in ("word", "phone"))
    provider_words = pd.DataFrame(word_rows)
    provider_phones = pd.DataFrame(phone_rows)
    trials, words, phones = build_feature_tables(
        pd.DataFrame(trial_rows), pd.DataFrame(link_rows), provider_words,
        provider_phones, pd.DataFrame(reviews),
        {"nonclinical": {**prompt, "task_id": task_id}}, "nonclinical_run")
    review = root / "acoustic" / "003_segmentation_review" / "final"
    review.mkdir(parents=True, exist_ok=True)
    decisions_path = review / "final_segmentation_decisions.csv"
    intervals_path = review / "final_segmentation_intervals.csv"
    pd.DataFrame([{"recording_id": "nonclinical", "final_decision": "KEEP_MANUAL"}]).to_csv(
        decisions_path, index=False)
    pd.DataFrame([{"recording_id": "nonclinical", "segment_role": "speech",
                   "start_sec": 10., "end_sec": 10. * repetitions + 8.}]).to_csv(
        intervals_path, index=False)
    final = root / "acoustic" / "004_alignment" / "final"
    final.mkdir(parents=True, exist_ok=True)
    files = {
        "final_alignment_words.csv": provider_words.reindex(columns=WORD_COLUMNS, fill_value=""),
        "final_alignment_phones.csv": provider_phones.reindex(columns=PHONE_COLUMNS, fill_value=""),
        "final_alignment_diagnostics.csv": pd.DataFrame(columns=DIAGNOSTIC_COLUMNS),
        "final_alignment_contract_trials.csv": trials,
        "final_alignment_canonical_words.csv": words,
        "final_alignment_canonical_phones.csv": phones,
    }
    for name, table in files.items():
        table.to_csv(final / name, index=False)
    manifest = {
        "alignment_status": "FROZEN", "alignment_run_id": "nonclinical_run",
        "final_alignment_contract_version": FINAL_ALIGNMENT_CONTRACT_VERSION,
        "source_final_decisions_sha256": _hash(decisions_path),
        "source_final_intervals_sha256": _hash(intervals_path),
        "final_words_sha256": _hash(final / "final_alignment_words.csv"),
        "final_phones_sha256": _hash(final / "final_alignment_phones.csv"),
        "final_diagnostics_sha256": _hash(final / "final_alignment_diagnostics.csv"),
        "final_alignment_contract_artifact_sha256": {
            name: _hash(final / name) for name in (
                "final_alignment_contract_trials.csv",
                "final_alignment_canonical_words.csv",
                "final_alignment_canonical_phones.csv")},
        "source_audio_sha256_by_recording": {},
    }
    (final / "final_alignment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return root
