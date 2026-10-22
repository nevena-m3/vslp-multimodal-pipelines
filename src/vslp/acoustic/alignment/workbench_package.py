"""Additive Package V1 producer for standalone, structurally accepted MFA jobs."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import soundfile as sf

from .audio_identity import audio_content_sha256
from .final_contract import FINAL_ALIGNMENT_CONTRACT_VERSION, canonical_units
from .package_v1 import PACKAGE_VERSION, _validate_canonical
from .stage import normalize_phone, normalize_transcript
from .workbench import load_job
from .workbench_review import load_corrections, reviewed_tokens


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows_for_record(record: dict, spec: dict, run_id: str,
                     job_root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Map exact provider tokens to canonical units; refuse ambiguous phones."""
    provider_words, provider_phones = reviewed_tokens(job_root, record["record_key"])
    reviewed = record["state"] == "REVIEWED"
    review_status = "HUMAN_ACCEPTED" if reviewed else "AUTO_ACCEPTED_STRUCTURAL"
    corrections = load_corrections(job_root, record["record_key"])
    if reviewed and corrections.get("decision") != "ACCEPT":
        raise ValueError(f"reviewed_alignment_not_available:{record['record_key']}")
    units = canonical_units(spec["canonical_transcript"])
    token_groups = [normalize_transcript(unit).split() for unit in units]
    expected = [token for group in token_groups for token in group] * spec["expected_repetitions"]
    actual = [normalize_transcript(str(label)) for label in provider_words.label]
    if expected != actual:
        raise ValueError(f"canonical_provider_token_mapping_requires_review:{record['record_key']}")
    source = Path(record["source_path"])
    info = sf.info(source)
    trial_rows: list[dict] = []
    word_rows: list[dict] = []
    phone_rows: list[dict] = []
    token_cursor = 0
    used_phones: set[int] = set()
    tolerance = .5 / info.samplerate
    for trial_index in range(1, spec["expected_repetitions"] + 1):
        trial_id = f"trial_{trial_index:03d}"
        sequence_id = f"{record['record_key']}:{trial_id}"
        count = sum(map(len, token_groups))
        part = provider_words.iloc[token_cursor:token_cursor + count]
        if part.empty:
            raise ValueError("missing_repetition_alignment")
        start, end = float(part.start_sec.iloc[0]), float(part.end_sec.iloc[-1])
        common = {"recording_id": record["record_key"], "trial_id": trial_id,
                  "sequence_id": sequence_id, "task_id": spec["task_id"],
                  "prompt_id": spec["prompt_id"], "prompt_version": spec["prompt_version"],
                  "alignment_run_id": run_id,
                  "alignment_review_status": review_status,
                  "alignment_contract_version": FINAL_ALIGNMENT_CONTRACT_VERSION}
        trial_rows.append({**common, "trial_index": trial_index, "start_sec": start,
                           "end_sec": end, "prompt_sha256": spec["transcript_hash"],
                           "protocol_expected_repetitions": spec["expected_repetitions"],
                           "observed_trial_count": spec["expected_repetitions"],
                           "production_structure": "known_repetition_sequence",
                           "alignment_source_summary": ("HUMAN_CORRECTED_MFA" if reviewed
                                                        else "MFA_STRUCTURAL_QC"),
                           "qc_flags": "", "segmentation_review_hash": "",
                           "trial_manifest_hash": ""})
        phone_index_in_trial = 0
        for index, (unit, group) in enumerate(zip(units, token_groups), 1):
            parts = provider_words.iloc[token_cursor:token_cursor + len(group)]
            word_start, word_end = float(parts.start_sec.iloc[0]), float(parts.end_sec.iloc[-1])
            word_id = f"{sequence_id}:word_{index:03d}"
            provider_indices = parts.provider_index.astype(int).tolist()
            word_corrected = any(str(value) in corrections.get("word_bounds", {})
                                 for value in provider_indices)
            word_source = "REVIEWED_MANUAL_CORRECTION" if word_corrected else "MFA"
            word_rows.append({**common, "word_id": word_id,
                              "canonical_word_index": index, "canonical_word_text": unit,
                              "word_normalized": normalize_transcript(unit),
                              "provider_token_ids": json.dumps(provider_indices),
                              "provider_token_count": len(group), "word_index": index,
                              "word": unit, "start_sec": word_start, "end_sec": word_end,
                              "duration_sec": word_end - word_start,
                              "boundary_source": word_source, "alignment_source": word_source,
                              "qc_flags": ""})
            phone_subset = provider_phones.loc[
                (provider_phones.start_sec.astype(float) >= word_start - tolerance) &
                (provider_phones.end_sec.astype(float) <= word_end + tolerance)]
            for local_index, (phone_row_index, phone) in enumerate(phone_subset.iterrows(), 1):
                if int(phone_row_index) in used_phones:
                    raise ValueError("ambiguous_phone_parent_word")
                used_phones.add(int(phone_row_index))
                phone_index_in_trial += 1
                phone_start, phone_end = float(phone.start_sec), float(phone.end_sec)
                normalized, stress, vowel = normalize_phone(str(phone.label))
                phone_source = ("REVIEWED_MANUAL_CORRECTION" if str(int(phone.provider_index))
                                in corrections.get("phone_bounds", {}) else "MFA")
                phone_rows.append({**common,
                                   "phone_id": f"{sequence_id}:phone_{phone_index_in_trial:04d}",
                                   "phone_index_in_trial": phone_index_in_trial,
                                   "phone_index_in_word": local_index,
                                   "parent_word_id": word_id, "parent_word_index": index,
                                   "parent_word_text": unit,
                                   "word_index": index, "phone_index": phone_index_in_trial,
                                   "word": unit, "phone_raw": str(phone.label),
                                   "phone": str(phone.label), "phone_normalized": normalized,
                                   "stress": stress, "is_vowel": vowel,
                                   "start_sec": phone_start, "end_sec": phone_end,
                                   "duration_sec": phone_end - phone_start,
                                   "boundary_source": phone_source,
                                   "alignment_source": phone_source,
                                   "qc_flags": ""})
            token_cursor += len(group)
    if len(used_phones) != len(provider_phones):
        raise ValueError("unmapped_phone_requires_review")
    return trial_rows, word_rows, phone_rows


def export_standalone_package(job_root: str | Path, destination: str | Path) -> Path:
    """Export only structurally accepted records, preserving Package V1 files."""
    root, target = Path(job_root), Path(destination)
    job = load_job(root)
    records = [row for row in job["recordings"]
               if row["state"] in {"AUTO_PASS", "REVIEWED"}]
    if not records:
        raise ValueError("no_structurally_accepted_recordings_to_export")
    if target.exists():
        raise FileExistsError(target)
    spec = job["task_specification"]
    run_id = job["job_id"]
    trials, words, phones = [], [], []
    identities = []
    for record in records:
        record_trials, record_words, record_phones = _rows_for_record(record, spec, run_id,
                                                                      root)
        trials.extend(record_trials)
        words.extend(record_words)
        phones.extend(record_phones)
        source = Path(record["source_path"])
        info = sf.info(source)
        identities.append({"record_key": record["record_key"],
                           "source_filename": record["recording"],
                           "task_id": spec["task_id"], "prompt_id": spec["prompt_id"],
                           "transcript_hash": spec["transcript_hash"],
                           "source_file_sha256": _hash(source),
                           "source_audio_content_sha256": audio_content_sha256(source),
                           "raw_source_file_sha256": _hash(source),
                           "raw_source_audio_content_sha256": audio_content_sha256(source),
                           "raw_source_sample_rate_hz": info.samplerate,
                           "raw_source_channels": info.channels,
                           "raw_source_sample_count": info.frames,
                           "alignment_working_audio_content_sha256": audio_content_sha256(
                               Path(record.get("corpus_audio") or source)),
                           "alignment_working_transform": record.get(
                               "alignment_working_transform", "full_recording_byte_copy"),
                           "alignment_scope_start_sec": record.get(
                               "alignment_scope", {}).get("start_sec", 0.0),
                           "alignment_scope_end_sec": record.get(
                               "alignment_scope", {}).get("end_sec", info.duration),
                           "alignment_scope_source": record.get(
                               "alignment_scope", {}).get("source", "legacy_full_recording"),
                           "alignment_scope_uncertain": bool(record.get(
                               "alignment_scope", {}).get("uncertain", False)),
                           "boundary_time_basis": "selected_source_original_time",
                           "analysis_wav_sha256": _hash(source),
                           "sample_rate": info.samplerate, "channels": info.channels,
                           "sample_count": info.frames, "alignment_status": "ALIGNED",
                           "review_status": ("HUMAN_ACCEPTED" if record["state"] == "REVIEWED"
                                             else "AUTO_ACCEPTED_STRUCTURAL"),
                           "finalization_status": "FINALIZED"})
    trial_table, word_table, phone_table = map(pd.DataFrame, (trials, words, phones))
    _validate_canonical(trial_table, word_table, phone_table, 16000)
    target.mkdir(parents=True)
    for name, table in (
        ("final_alignment_contract_trials.csv", trial_table),
        ("final_alignment_canonical_words.csv", word_table),
        ("final_alignment_canonical_phones.csv", phone_table),
        ("final_alignment_words.csv", word_table),
        ("final_alignment_phones.csv", phone_table),
        ("final_alignment_diagnostics.csv", pd.DataFrame([
            {"recording_id": row["record_key"], "status": "AUTO_PASS", "reason": ""}
            for row in records])),
        ("recording_status.csv", pd.DataFrame(identities)),
    ):
        table.to_csv(target / name, index=False)
    profile_path = Path(job["profile_path"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    final_manifest = {"alignment_status": "FROZEN",
                      "final_alignment_contract_version": FINAL_ALIGNMENT_CONTRACT_VERSION,
                      "alignment_run_id": run_id, "working_sample_rate_hz": 16000,
                      "provider": "mfa", "provider_version": job.get("mfa_environment", {}).get(
                          "version", ""), "acoustic_model": profile["acoustic_model"],
                      "dictionary": profile["dictionary"],
                      "mfa_profile_sha256": _hash(profile_path),
                      "source_final_decisions_sha256": "",
                      "source_final_intervals_sha256": "",
                      "source_audio_content_sha256_by_recording": {
                          row["record_key"]: row["source_audio_content_sha256"]
                          for row in identities}}
    (target / "final_alignment_manifest.json").write_text(
        json.dumps(final_manifest, indent=2), encoding="utf-8")
    provenance = {"producer_mode": "standalone_audio_identity_v1", "job_id": run_id,
                  "source_alignment_run_id": run_id,
                  "decoded_audio_identity_is_binding": True}
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    files = {path.relative_to(target).as_posix(): _hash(path)
             for path in target.rglob("*") if path.is_file()}
    package_manifest = {
        "package_contract_version": PACKAGE_VERSION, "package_id": uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer_application": "VSLP Alignment", "producer_version": "workbench_v1",
        "alignment_engine": "Montreal Forced Aligner",
        "alignment_engine_version": job.get("mfa_environment", {}).get("version", ""),
        "backend_name": "mfa", "backend_version": job.get("mfa_environment", {}).get(
            "version", ""), "acoustic_model": profile["acoustic_model"],
        "dictionary": profile["dictionary"], "configuration_hash": _hash(profile_path),
        "recording_count": len(records), "aligned_count": len(records),
        "finalized_count": len(records), "failed_count": len(job["recordings"]) - len(records),
        "package_file_hashes": files,
        "final_alignment_contract_version": FINAL_ALIGNMENT_CONTRACT_VERSION,
        "producer_mode": "standalone_audio_identity_v1",
        "task_specification": spec,
    }
    (target / "package_manifest.json").write_text(
        json.dumps(package_manifest, indent=2), encoding="utf-8")
    return target
