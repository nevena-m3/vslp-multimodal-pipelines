"""Portable interchange for the existing Final Alignment Contract V1.

The canonical CSVs are copied byte for byte. Package validation adds identity
and transport checks; it does not redefine feature-facing alignment semantics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd
import soundfile as sf

from .audio_identity import audio_content_sha256
from .final_contract import FINAL_ALIGNMENT_CONTRACT_VERSION
from .stage import AlignmentStore


PACKAGE_VERSION = "VSLP_ALIGNMENT_PACKAGE_V1"
REQUIRED_FILES = (
    "final_alignment_manifest.json",
    "final_alignment_contract_trials.csv",
    "final_alignment_canonical_words.csv",
    "final_alignment_canonical_phones.csv",
    "final_alignment_words.csv",
    "final_alignment_phones.csv",
    "final_alignment_diagnostics.csv",
)
FINAL_STATES = frozenset({"AUTO_ACCEPTED_STRUCTURAL", "HUMAN_ACCEPTED"})


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _available_audio_hash(path: Path) -> str:
    try:
        return audio_content_sha256(path)
    except (OSError, RuntimeError):
        return ""


def _producer_version() -> str:
    try:
        return version("vslp-multimodal-pipelines")
    except PackageNotFoundError:
        return "uninstalled-source"


def _tables(folder: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for name in REQUIRED_FILES:
        if not (folder / name).is_file():
            raise ValueError(f"missing_package_file:{name}")
    manifest = json.loads((folder / REQUIRED_FILES[0]).read_text(encoding="utf-8"))
    if (manifest.get("alignment_status") != "FROZEN" or
            manifest.get("final_alignment_contract_version") != FINAL_ALIGNMENT_CONTRACT_VERSION):
        raise ValueError("alignment_not_finalized")
    tables = [pd.read_csv(folder / name, keep_default_na=False) for name in REQUIRED_FILES[1:4]]
    return manifest, *tables


def _validate_canonical(trials: pd.DataFrame, words: pd.DataFrame,
                        phones: pd.DataFrame, rate: int) -> None:
    required = (
        {"recording_id", "trial_id", "sequence_id", "task_id", "prompt_id",
         "prompt_sha256", "start_sec", "end_sec", "alignment_review_status"},
        {"recording_id", "trial_id", "sequence_id", "word_id", "canonical_word_index",
         "word", "start_sec", "end_sec", "boundary_source", "alignment_review_status"},
        {"recording_id", "trial_id", "sequence_id", "phone_id", "phone_index_in_trial",
         "parent_word_id", "phone", "start_sec", "end_sec", "boundary_source",
         "alignment_review_status"},
    )
    if any(not columns <= set(table) for columns, table in zip(required, (trials, words, phones))):
        raise ValueError("package_schema_missing")
    if trials.empty or any(not table.alignment_review_status.isin(FINAL_STATES).all()
                           for table in (trials, words, phones)):
        raise ValueError("alignment_not_finalized")
    if (trials.duplicated(["recording_id", "trial_id"]).any() or
            words.duplicated(["recording_id", "trial_id", "word_id"]).any() or
            phones.duplicated(["recording_id", "trial_id", "phone_id"]).any()):
        raise ValueError("duplicate_canonical_identity")
    trial_map = {(str(r.recording_id), str(r.trial_id)): r for r in trials.itertuples()}
    word_map = {str(r.word_id): r for r in words.itertuples()}
    tolerance = .5 / max(1, rate)
    for trial in trials.itertuples():
        if not (np.isfinite([trial.start_sec, trial.end_sec]).all()
                and float(trial.start_sec) < float(trial.end_sec)):
            raise ValueError("invalid_trial_bounds")
    for table in (words, phones):
        for token in table.itertuples():
            trial = trial_map.get((str(token.recording_id), str(token.trial_id)))
            if trial is None or str(token.sequence_id) != str(trial.sequence_id):
                raise ValueError("invalid_trial_reference")
            start, end = float(token.start_sec), float(token.end_sec)
            if not (np.isfinite([start, end]).all() and
                    float(trial.start_sec) - tolerance <= start < end <=
                    float(trial.end_sec) + tolerance):
                raise ValueError("token_outside_trial")
    for phone in phones.itertuples():
        word = word_map.get(str(phone.parent_word_id))
        if (word is None or str(word.recording_id) != str(phone.recording_id)
                or str(word.trial_id) != str(phone.trial_id) or
                not float(word.start_sec) - tolerance <= float(phone.start_sec)
                < float(phone.end_sec) <= float(word.end_sec) + tolerance):
            raise ValueError("phone_outside_parent_word")


def export_alignment_package(project_root: str | Path, destination: str | Path) -> Path:
    """Export a *current* embedded Final Alignment without rerunning its backend."""
    from .stage import _load_embedded_final_alignment

    root, target = Path(project_root), Path(destination)
    store, issue = _load_embedded_final_alignment(root)
    if store is None:
        raise ValueError(f"embedded_alignment_unavailable:{issue}")
    if target.exists():
        raise FileExistsError(target)
    final = root / "acoustic" / "004_alignment" / "final"
    source_manifest, trials, words, phones = _tables(final)
    _validate_canonical(trials, words, phones,
                        int(source_manifest.get("working_sample_rate_hz", 16000)))
    decisions = pd.read_csv(root / "acoustic" / "003_segmentation_review" / "final" /
                            "final_segmentation_decisions.csv", keep_default_na=False)
    if decisions.duplicated("recording_id").any():
        raise ValueError("duplicate_project_record_key")
    finalized_keys = set(trials.recording_id.astype(str))
    if not finalized_keys <= set(decisions.recording_id.astype(str)):
        raise ValueError("project_record_set_mismatch")
    decisions = decisions.loc[decisions.recording_id.astype(str).isin(finalized_keys)]
    target.mkdir(parents=True)
    for path in final.rglob("*"):
        if path.is_file():
            relative = path.relative_to(final)
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / relative)
    identity_rows = []
    for row in decisions.itertuples():
        record_key = str(row.recording_id)
        subset = trials.loc[trials.recording_id.astype(str).eq(record_key)]
        source = Path(str(getattr(row, "source_file_path", "")))
        analysis = Path(str(getattr(row, "analysis_wav_path", "")))
        if not analysis.is_file():
            raise ValueError(f"source_audio_missing:{record_key}")
        with sf.SoundFile(analysis) as audio:
            sample_rate, channels, sample_count = audio.samplerate, audio.channels, len(audio)
        identity_rows.append({
            "record_key": record_key, "source_filename": str(getattr(row, "file_name", "")),
            "task_id": str(subset.task_id.iloc[0]), "prompt_id": str(subset.prompt_id.iloc[0]),
            "transcript_hash": str(subset.prompt_sha256.iloc[0]),
            "source_file_sha256": _hash(source) if source.is_file() else "",
            "source_audio_content_sha256": audio_content_sha256(analysis),
            "raw_source_audio_content_sha256": (_available_audio_hash(source)
                                                  if source.is_file() else ""),
            "analysis_wav_sha256": _hash(analysis), "sample_rate": sample_rate,
            "channels": channels, "sample_count": sample_count,
            "alignment_status": "ALIGNED", "review_status": ";".join(sorted(
                set(subset.alignment_review_status.astype(str)))),
            "finalization_status": "FINALIZED",
        })
    pd.DataFrame(identity_rows).to_csv(target / "recording_status.csv", index=False)
    provenance = {"source_alignment_run_id": source_manifest["alignment_run_id"],
                  "source_final_decisions_sha256": source_manifest["source_final_decisions_sha256"],
                  "source_final_intervals_sha256": source_manifest["source_final_intervals_sha256"],
                  "source_manifest_sha256": _hash(final / "final_alignment_manifest.json"),
                  "decoded_audio_identity_is_binding": True}
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    files = {p.relative_to(target).as_posix(): _hash(p) for p in target.rglob("*") if p.is_file()}
    manifest = {
        "package_contract_version": PACKAGE_VERSION, "package_id": uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer_application": "VSLP Alignment",
        "producer_version": _producer_version(),
        "alignment_engine": "VSLP Alignment", "alignment_engine_version": source_manifest.get(
            "algorithm_version", ""),
        "backend_name": source_manifest.get("provider", ""),
        "backend_version": source_manifest.get("provider_version", ""),
        "acoustic_model": source_manifest.get("acoustic_model", ""),
        "dictionary": source_manifest.get("dictionary", ""),
        "configuration_hash": source_manifest.get("mfa_profile_sha256", ""),
        "recording_count": len(identity_rows), "aligned_count": len(identity_rows),
        "finalized_count": len(identity_rows), "failed_count": 0,
        "package_file_hashes": files,
        "final_alignment_contract_version": FINAL_ALIGNMENT_CONTRACT_VERSION,
    }
    # The copied canonical trial table remains authoritative. Repeat its job
    # identity here so a receiving application can inspect the package quickly.
    task_rows = trials[["task_id", "prompt_id", "prompt_version", "prompt_sha256",
                        "protocol_expected_repetitions"]].drop_duplicates()
    if len(task_rows) != 1:
        raise ValueError("mixed_alignment_package_task_specification")
    task_row = task_rows.iloc[0]
    from .task_workflow import task_entry

    registered_task = task_entry(str(task_row.task_id))
    registered_prompt = next((item for item in registered_task.get("prompts", [])
                              if item["prompt_id"] == str(task_row.prompt_id)), None) if registered_task else None
    count_text = str(task_row.protocol_expected_repetitions).strip()
    count = int(count_text) if count_text else (
        registered_prompt.get("expected_repetitions") if registered_prompt else None)
    manifest["task_specification"] = {
        "task_id": str(task_row.task_id), "prompt_id": str(task_row.prompt_id),
        "prompt_version": str(task_row.prompt_version),
        "transcript_hash": str(task_row.prompt_sha256),
        "expected_repetitions": count,
    }
    job_path = root / "configs" / "alignment_job.json"
    if job_path.is_file():
        manifest["task_specification"].update(
            json.loads(job_path.read_text(encoding="utf-8")))
    (target / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def validate_alignment_package(package: str | Path, project_root: str | Path) -> dict:
    """Validate transport, canonical structure, and current project identity."""
    folder, root = Path(package), Path(project_root)
    manifest_path = folder / "package_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("package_manifest_missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"package_contract_version", "package_id", "created_at", "producer_application",
                "producer_version", "alignment_engine", "alignment_engine_version",
                "backend_name", "backend_version", "acoustic_model", "dictionary",
                "configuration_hash", "recording_count", "aligned_count", "finalized_count",
                "failed_count", "package_file_hashes", "final_alignment_contract_version"}
    if not required <= set(manifest):
        raise ValueError("package_manifest_incomplete")
    if manifest["package_contract_version"] != PACKAGE_VERSION:
        raise ValueError("unsupported_package_version")
    if manifest["final_alignment_contract_version"] != FINAL_ALIGNMENT_CONTRACT_VERSION:
        raise ValueError("unsupported_final_alignment_contract")
    files = manifest["package_file_hashes"]
    if not isinstance(files, dict) or not set(REQUIRED_FILES + (
            "recording_status.csv", "provenance.json")) <= set(files):
        raise ValueError("package_files_incomplete")
    members = list(folder.rglob("*"))
    if any(path.is_symlink() for path in members):
        raise ValueError("package_symlink_not_allowed")
    actual = {path.relative_to(folder).as_posix() for path in members if path.is_file()}
    if actual != set(files) | {"package_manifest.json"}:
        raise ValueError("package_file_inventory_mismatch")
    for name, expected in files.items():
        path = folder / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or
                not path.is_file() or path.is_symlink() or _hash(path) != expected):
            raise ValueError(f"package_file_hash_mismatch:{name}")
    source_manifest, trials, words, phones = _tables(folder)
    if source_manifest.get("alignment_status") != "FROZEN":
        raise ValueError("alignment_not_finalized")
    _validate_canonical(trials, words, phones,
                        int(source_manifest.get("working_sample_rate_hz", 16000)))
    status = pd.read_csv(folder / "recording_status.csv", keep_default_na=False)
    required_status = {"record_key", "task_id", "prompt_id", "transcript_hash",
                       "analysis_wav_sha256", "source_file_sha256",
                       "source_audio_content_sha256",
                       "finalization_status"}
    if not required_status <= set(status.columns):
        raise ValueError("package_recording_status_schema_incomplete")
    if (status.empty or status.duplicated("record_key").any() or
            len(status) != int(manifest["recording_count"])):
        raise ValueError("duplicate_or_missing_record_key")
    if not status.finalization_status.eq("FINALIZED").all():
        raise ValueError("alignment_not_finalized")
    if set(status.record_key.astype(str)) != set(trials.recording_id.astype(str)):
        raise ValueError("package_record_set_mismatch")
    if manifest.get("producer_mode") == "standalone_audio_identity_v1":
        return _validate_standalone_identity(manifest, status, trials, root)
    review = root / "acoustic" / "003_segmentation_review" / "final"
    decisions_path, intervals_path = (review / "final_segmentation_decisions.csv",
                                      review / "final_segmentation_intervals.csv")
    provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    if (_hash(decisions_path) != provenance["source_final_decisions_sha256"] or
            _hash(intervals_path) != provenance["source_final_intervals_sha256"]):
        raise ValueError("segmentation_review_mismatch")
    decisions = pd.read_csv(decisions_path, keep_default_na=False)
    if decisions.duplicated("recording_id").any():
        raise ValueError("duplicate_project_record_key")
    project = decisions.set_index("recording_id")
    project_manifest_path = root / "project_manifest.json"
    project_task = ""
    if project_manifest_path.is_file():
        from .task_workflow import _manifest_task

        task = _manifest_task(json.loads(project_manifest_path.read_text(encoding="utf-8")))
        project_task = str(task["task_id"]) if task else ""
    choices_path = root / "configs" / "alignment_choices.json"
    choices = (json.loads(choices_path.read_text(encoding="utf-8"))
               if choices_path.is_file() else {})
    per_record = []
    for row in status.itertuples():
        key = str(row.record_key)
        if key not in project.index:
            per_record.append({"record_key": key, "status": "MISSING"})
            continue
        current = project.loc[key]
        subset = trials.loc[trials.recording_id.astype(str).eq(key)]
        selected_prompt = choices.get("recordings", {}).get(key, {}).get(
            "prompt_id", "")
        if (str(row.task_id) != str(subset.task_id.iloc[0]) or
                (project_task and project_task != str(row.task_id)) or
                (selected_prompt and selected_prompt != str(row.prompt_id)) or
                str(row.prompt_id) != str(subset.prompt_id.iloc[0]) or
                str(row.transcript_hash) != str(subset.prompt_sha256.iloc[0])):
            per_record.append({"record_key": key, "status": "TASK/PROMPT MISMATCH"})
            continue
        audio = Path(str(current.get("analysis_wav_path", "")))
        source = Path(str(current.get("source_file_path", "")))
        if (not audio.is_file() or audio_content_sha256(audio) !=
                str(row.source_audio_content_sha256) or
                (str(row.source_file_sha256) and not source.is_file()) or
                (str(getattr(row, "raw_source_audio_content_sha256", "")) and
                 (not source.is_file() or audio_content_sha256(source) !=
                  str(row.raw_source_audio_content_sha256)))):
            per_record.append({"record_key": key, "status": "IDENTITY MISMATCH"})
            continue
        per_record.append({"record_key": key, "status": "MATCHED + FINALIZED"})
    for key in sorted(set(project.index.astype(str)) - set(status.record_key.astype(str))):
        if str(project.loc[key].get("final_decision", "")) != "EXCLUDE":
            per_record.append({"record_key": key, "status": "MISSING"})
    if any(row["status"] != "MATCHED + FINALIZED" for row in per_record):
        raise ValueError("package_project_identity_mismatch:" + json.dumps(per_record))
    return {"manifest": manifest, "recording_status": per_record,
            "matched": len(per_record), "finalized": len(per_record), "missing": 0,
            "stale": 0, "invalid": 0}


def _validate_standalone_identity(manifest: dict, status: pd.DataFrame,
                                  trials: pd.DataFrame, root: Path) -> dict:
    """Bind selected audio to raw lineage or an identical derived analysis WAV."""
    review = root / "acoustic" / "003_segmentation_review" / "final"
    decisions_path = review / "final_segmentation_decisions.csv"
    if not decisions_path.is_file():
        raise ValueError("project_record_index_missing")
    decisions = pd.read_csv(decisions_path, keep_default_na=False)
    if decisions.duplicated("recording_id").any():
        raise ValueError("duplicate_project_record_key")
    from .task_workflow import _manifest_task, task_entry

    project_manifest = root / "project_manifest.json"
    registered = (_manifest_task(json.loads(project_manifest.read_text(encoding="utf-8")))
                  if project_manifest.is_file() else None)
    task_spec = manifest.get("task_specification", {})
    task = task_entry(str(task_spec.get("task_id", "")))
    prompt = next((item for item in task.get("prompts", [])
                   if item["prompt_id"] == task_spec.get("prompt_id")), None) if task else None
    if (registered is None or task is None or prompt is None or
            registered["task_id"] != task["task_id"] or
            sha256(str(prompt["exact_expected_text"]).encode("utf-8")).hexdigest()
            != task_spec.get("transcript_hash") or
            str(prompt["prompt_version"]) != str(task_spec.get("prompt_version"))):
        raise ValueError("standalone_task_prompt_mismatch")
    preprocess_path = (root / "acoustic" / "001_preprocess" / "tables" /
                       "acoustic_preprocess_summary.csv")
    preprocess = (pd.read_csv(preprocess_path, keep_default_na=False).set_index("recording_id")
                  if preprocess_path.is_file() else pd.DataFrame())
    identity: dict[str, dict] = {}
    for row in decisions.itertuples():
        key = str(row.recording_id)
        source = Path(str(getattr(row, "source_file_path", "")))
        analysis = Path(str(getattr(row, "analysis_wav_path", "")))
        if not source.is_file() or not analysis.is_file():
            raise ValueError(f"standalone_project_audio_missing:{key}")
        source_file_hash = _hash(source)
        if str(getattr(row, "source_sha256", "")) not in {"", source_file_hash}:
            raise ValueError(f"standalone_project_source_changed:{key}")
        source_info, analysis_info = sf.info(source), sf.info(analysis)
        source_audio_hash = audio_content_sha256(source)
        analysis_audio_hash = audio_content_sha256(analysis)
        same_time_axis = (source_info.samplerate == analysis_info.samplerate and
                          source_info.frames == analysis_info.frames)
        if source_audio_hash != analysis_audio_hash:
            if not same_time_axis:
                raise ValueError(f"standalone_time_axis_mismatch:{key}")
            if key not in preprocess.index:
                raise ValueError(f"standalone_preprocessing_lineage_missing:{key}")
            lineage = preprocess.loc[key]
            if (isinstance(lineage, pd.DataFrame) or str(lineage.get("status")) != "ok" or
                    str(lineage.get("source_sha256")) != source_file_hash or
                    int(lineage.get("decoded_sample_rate_hz")) != source_info.samplerate or
                    int(lineage.get("analysis_sample_rate_hz")) != analysis_info.samplerate or
                    int(lineage.get("decoded_n_samples")) != source_info.frames or
                    int(lineage.get("analysis_n_samples")) != analysis_info.frames):
                raise ValueError(f"standalone_preprocessing_lineage_mismatch:{key}")
        identity[key] = {"raw_audio": source_audio_hash, "analysis_audio": analysis_audio_hash,
                         "raw_info": source_info, "analysis_info": analysis_info}
    mapping = {}
    result_rows = []
    for row in status.itertuples():
        trial_subset = trials.loc[trials.recording_id.astype(str).eq(str(row.record_key))]
        if (trial_subset.empty or str(row.task_id) != task["task_id"] or
                str(row.prompt_id) != str(prompt["prompt_id"]) or
                str(row.transcript_hash) != str(task_spec["transcript_hash"]) or
                not trial_subset.task_id.astype(str).eq(task["task_id"]).all() or
                not trial_subset.prompt_sha256.astype(str).eq(
                    str(task_spec["transcript_hash"])).all()):
            raise ValueError(f"standalone_task_prompt_mismatch:{row.record_key}")
        selected_audio = str(row.source_audio_content_sha256)
        declared_raw = str(getattr(row, "raw_source_audio_content_sha256", ""))
        if declared_raw and declared_raw != selected_audio:
            raise ValueError(f"standalone_package_identity_conflict:{row.record_key}")
        boundary_basis = str(getattr(row, "boundary_time_basis", ""))
        if boundary_basis and boundary_basis != "selected_source_original_time":
            raise ValueError(f"standalone_time_basis_unsupported:{row.record_key}")
        raw_matches = [key for key, item in identity.items()
                       if item["raw_audio"] == selected_audio]
        derived_matches = [key for key, item in identity.items()
                           if item["analysis_audio"] == selected_audio]
        matches = raw_matches if raw_matches else derived_matches
        if len(matches) != 1 or matches[0] in mapping.values():
            raise ValueError(f"standalone_audio_identity_mismatch:{row.record_key}")
        key = matches[0]
        basis = "raw_source" if raw_matches else "exact_analysis_audio"
        info = identity[key]["raw_info" if raw_matches else "analysis_info"]
        if (int(row.sample_rate) != info.samplerate or int(row.sample_count) != info.frames or
                int(row.channels) != info.channels):
            raise ValueError(f"standalone_time_axis_mismatch:{row.record_key}")
        working_hash = str(getattr(row, "alignment_working_audio_content_sha256", ""))
        transform = str(getattr(row, "alignment_working_transform", "full_recording_byte_copy"))
        if transform == "full_recording_byte_copy":
            if working_hash and working_hash != selected_audio:
                raise ValueError(f"standalone_working_audio_identity_mismatch:{row.record_key}")
        elif transform == "sample_exact_utterance_crop_v1":
            start = float(getattr(row, "alignment_scope_start_sec", -1))
            end = float(getattr(row, "alignment_scope_end_sec", -1))
            if not working_hash or not 0 <= start < end <= info.duration + 1e-7:
                raise ValueError(f"standalone_working_audio_scope_invalid:{row.record_key}")
            if (trial_subset.start_sec.astype(float).lt(start - 1e-7).any() or
                    trial_subset.end_sec.astype(float).gt(end + 1e-7).any()):
                raise ValueError(f"standalone_trial_outside_scope:{row.record_key}")
        else:
            raise ValueError(f"standalone_working_transform_unsupported:{row.record_key}")
        mapping[str(row.record_key)] = key
        result_rows.append({"record_key": key, "status": "MATCHED + FINALIZED",
                            "identity_basis": basis})
    for key in identity:
        if key not in mapping.values():
            result_rows.append({"record_key": key, "status": "MISSING"})
    return {"manifest": manifest, "recording_status": result_rows,
            "record_key_map": mapping, "matched": len(mapping),
            "finalized": len(mapping), "missing": len(identity) - len(mapping),
            "stale": 0, "invalid": 0}


def inspect_alignment_package(package: str | Path, project_root: str | Path) -> dict:
    """Give the GUI a per-record preview; only validate() authorizes import."""
    folder, root = Path(package), Path(project_root)
    try:
        return {**validate_alignment_package(folder, root), "error": ""}
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        error = str(exc)
    manifest_path = folder / "package_manifest.json"
    try:
        manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.is_file() else {})
    except (OSError, ValueError):
        manifest = {}
    if error.startswith("package_project_identity_mismatch:"):
        rows = json.loads(error.split(":", 1)[1])
        return {"manifest": manifest, "recording_status": rows,
                "matched": sum(item["status"].startswith("MATCHED") for item in rows),
                "finalized": sum(item["status"] == "MATCHED + FINALIZED" for item in rows),
                "missing": sum(item["status"] == "MISSING" for item in rows),
                "stale": sum(item["status"] in {"IDENTITY MISMATCH", "TASK/PROMPT MISMATCH"}
                             for item in rows),
                "invalid": sum(item["status"] == "INVALID PACKAGE ROW" for item in rows),
                "error": error}
    if error.startswith("standalone_"):
        # A rejected standalone package has no authorized match table. Do not
        # guess from record_key or a derived analysis WAV and display misleading
        # matched/missing counts in the GUI.
        return {"manifest": manifest, "recording_status": [], "matched": 0,
                "finalized": 0, "missing": 0, "stale": 0, "invalid": 1,
                "error": error}
    status_path = folder / "recording_status.csv"
    decisions_path = (root / "acoustic" / "003_segmentation_review" / "final" /
                      "final_segmentation_decisions.csv")
    if not status_path.is_file() or not decisions_path.is_file():
        return {"manifest": manifest, "recording_status": [], "matched": 0,
                "finalized": 0, "missing": 0, "stale": 0, "invalid": 1,
                "error": error}
    try:
        status = pd.read_csv(status_path, keep_default_na=False)
        decisions = pd.read_csv(decisions_path, keep_default_na=False)
        rows = []
        for current in decisions.itertuples():
            key = str(current.recording_id)
            matches = status.loc[status.record_key.astype(str).eq(key)]
            if matches.empty:
                state = "MISSING"
            elif len(matches) != 1:
                state = "INVALID PACKAGE ROW"
            else:
                selected = matches.iloc[0]
                audio = Path(str(getattr(current, "analysis_wav_path", "")))
                if str(selected.finalization_status) != "FINALIZED":
                    state = "MATCHED + NOT FINALIZED"
                elif (not audio.is_file() or audio_content_sha256(audio) !=
                      str(selected.source_audio_content_sha256)):
                    state = "IDENTITY MISMATCH"
                else:
                    state = "MATCHED + FINALIZED"
            rows.append({"record_key": key, "status": state})
        invalid = (len(rows) if error.startswith(("package_file_", "package_schema_",
                                                 "invalid_", "unsupported_")) else
                   sum(item["status"] == "INVALID PACKAGE ROW" for item in rows))
        return {"manifest": manifest, "recording_status": rows,
                "matched": sum(item["status"].startswith("MATCHED") for item in rows),
                "finalized": sum(item["status"] == "MATCHED + FINALIZED" for item in rows),
                "missing": sum(item["status"] == "MISSING" for item in rows),
                "stale": sum(item["status"] in {"IDENTITY MISMATCH", "TASK/PROMPT MISMATCH"}
                             for item in rows),
                "invalid": invalid, "error": error}
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return {"manifest": manifest, "recording_status": [], "matched": 0,
                "finalized": 0, "missing": 0, "stale": 0, "invalid": 1,
                "error": error}


def import_alignment_package(package: str | Path, project_root: str | Path) -> Path:
    """Install one explicitly selected package under a governed project location."""
    root = Path(project_root)
    result = validate_alignment_package(package, root)
    package_id = str(result["manifest"]["package_id"])
    if not package_id.isalnum():
        raise ValueError("invalid_package_id")
    base = root / "alignment_imports"
    destination = base / package_id
    if destination.exists():
        raise FileExistsError(destination)
    base.mkdir(parents=True, exist_ok=True)
    shutil.copytree(package, destination)
    (base / "active.json").write_text(json.dumps({
        "package_id": package_id, "package_manifest_sha256": _hash(
            destination / "package_manifest.json")}, indent=2), encoding="utf-8")
    return destination


def load_imported_alignment(project_root: str | Path) -> tuple[AlignmentStore | None, str]:
    """Load an explicitly selected imported package as the same AlignmentStore."""
    root = Path(project_root)
    active = root / "alignment_imports" / "active.json"
    if not active.is_file():
        return None, "alignment_package_not_selected"
    try:
        selection = json.loads(active.read_text(encoding="utf-8"))
        package_id = str(selection["package_id"])
        if not package_id.isalnum():
            raise ValueError("invalid_package_id")
        folder = root / "alignment_imports" / package_id
        if _hash(folder / "package_manifest.json") != selection["package_manifest_sha256"]:
            raise ValueError("package_manifest_hash_mismatch")
        validation = validate_alignment_package(folder, root)
        manifest, trials, words, phones = _tables(folder)
        provider_words = pd.read_csv(folder / "final_alignment_words.csv", keep_default_na=False)
        provider_phones = pd.read_csv(folder / "final_alignment_phones.csv", keep_default_na=False)
        diagnostics = pd.read_csv(folder / "final_alignment_diagnostics.csv", keep_default_na=False)
        mapping = validation.get("record_key_map", {})
        if mapping:
            for table in (trials, words, phones, provider_words, provider_phones, diagnostics):
                if "recording_id" in table:
                    table["recording_id"] = table.recording_id.astype(str).replace(mapping)
                for column in ("sequence_id", "word_id", "phone_id", "parent_word_id"):
                    if column in table:
                        table[column] = table[column].astype(str).map(lambda value:
                            next((f"{project_id}{value[len(source_id):]}"
                                  for source_id, project_id in mapping.items()
                                  if value.startswith(f"{source_id}:")), value))
        store = AlignmentStore(
            provider_words, provider_phones, diagnostics,
            manifest, trials, words, phones)
        return store, ""
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        return None, f"invalid_imported_alignment:{exc}"
