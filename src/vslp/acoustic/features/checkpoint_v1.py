"""Durable, per-recording acoustic execution without changing family algorithms.

The checkpoint is a commit marker for a finished family invocation. Family files
may be left by an interrupted invocation, but are never trusted without it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import traceback
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from vslp.acoustic.alignment.audio_identity import audio_content_sha256
from vslp.acoustic.features.catalog import requires_alignment_for_task
from vslp.core.feature_contract import build_feature_delivery, normalize_feature_registry
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


SCHEMA = "vslp-acoustic-recording-family-checkpoint-v1"
BASE_COLUMNS = ("recording_id", "file_name", "task_id", "task_type",
                "segmentation_run_id", "review_run_id")


@dataclass(frozen=True)
class RecordingScope:
    kept: pd.DataFrame
    intervals: pd.DataFrame
    decisions_sha256: str
    intervals_sha256: str


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _csv_identity(frame: pd.DataFrame) -> str:
    return _digest(frame.fillna("").to_dict("records"))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _checkpoint(path: Path, key: str) -> tuple[str, dict | None]:
    if not path.exists():
        return "INCOMPLETE", None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        checksum = document.pop("checksum")
        if checksum != _digest(document) or document.get("schema") != SCHEMA:
            return "CORRUPT", None
        if document.get("key") != key:
            return "STALE", None
        state = document.get("state")
        if state == "FAILED":
            return "FAILED", document
        if state != "COMPLETE" or not document.get("rows"):
            return "INCOMPLETE", None
        for artifact in document.get("artifacts", []):
            candidate = Path(artifact["path"])
            if not candidate.is_file() or sha256_file(candidate) != artifact["sha256"]:
                return "CORRUPT", None
        return "VALID COMPLETE", document
    except (OSError, ValueError, KeyError, TypeError):
        return "CORRUPT", None


def _write_checkpoint(path: Path, payload: dict) -> None:
    payload = _jsonable({**payload, "schema": SCHEMA})
    _atomic_json(path, {**payload, "checksum": _digest(payload)})


def _dependency_versions(group: str) -> dict[str, str]:
    versions = {}
    required = {"numpy", "scipy", "pandas", "soundfile"}
    if group in {"voice", "glottal", "formants", "formant_dynamics", "segmental"}:
        required.add("praat-parselmouth")
    if group in {"voice", "spectrum", "nonlinear"}:
        required.add("librosa")
    if group == "nonlinear":
        required.update({"tsfel", "PyWavelets"})
    for name in sorted(required):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def _alignment_identities(root: Path, record_ids: set[str]) -> tuple[dict[str, str], str]:
    from vslp.acoustic.alignment import load_final_alignment

    store, issue = load_final_alignment(root)
    if store is None:
        return {}, issue
    tables = []
    for field in ("trials", "words", "phones"):
        table = getattr(store, field, None)
        tables.append(table if isinstance(table, pd.DataFrame) else pd.DataFrame())
    identities = {}
    for record_id in record_ids:
        parts = [table.loc[table.recording_id.astype(str).eq(record_id)]
                 if "recording_id" in table else table.iloc[:0]
                 for table in tables]
        identities[record_id] = _digest([_csv_identity(part) for part in parts])
    return identities, ""


def _implementation_identity(group: str) -> dict[str, str]:
    base = Path(__file__).parent
    modules = {
        "voice": "family01_stage.py", "glottal": "family03_stage.py",
        "formants": "family04_stage.py", "formant_dynamics": "family05_stage.py",
        "segmental": "family06_v2_stage.py", "duration": "family07_v2_stage.py",
        "bamboo_rates": "family08_v2.py", "timing": "family09_v2_stage.py",
        "ddk": "family10_v2_stage.py", "amplitude": "family11_v2_stage.py",
        "spectrum": "family12_v2_stage.py", "nonlinear": "family13_v1_stage.py",
    }
    families = {"voice": ("family01.py", "family02.py", "family01_task.py",
                          "family02_dsp.py", "family02_research.py",
                          "family02_harmonic_difference.py", "stable_phonation.py"),
                "glottal": ("family03.py", "family03_gif.py", "family01_stage.py",
                             "family01_task.py", "stable_phonation.py"),
                "formants": ("family04.py", "family04_envelope.py",
                             "formant_service.py", "family01_task.py",
                             "stable_phonation.py"),
                "formant_dynamics": ("family05.py", "family04.py",
                                     "formant_service.py"),
                "segmental": ("family06.py", "family06_subevents.py"),
                "duration": ("family07_v2.py", "reviewed_timing.py"),
                "bamboo_rates": ("family07_v2.py", "reviewed_timing.py"),
                "timing": ("family09_v2.py", "family07_v2.py", "reviewed_timing.py"),
                "ddk": ("family10_v2.py",),
                "amplitude": ("family11_v2.py", "family07_v2.py",
                              "reviewed_timing.py"),
                "spectrum": ("family12_v2.py", "family07_v2.py",
                             "family11_v2_stage.py", "stable_phonation.py",
                             "reviewed_timing.py"),
                "nonlinear": ("family13_v1.py", "family07_v2.py",
                              "family11_v2_stage.py", "stable_phonation.py",
                              "reviewed_timing.py")}
    paths = [base / modules[group], *(base / name for name in families[group]),
             base / "mixed_dispatch.py", base / "checkpoint_v1.py"]
    if group == "bamboo_rates":
        paths.extend([base.parent / "alignment" / "task_workflow.py",
                      base.parent / "alignment" / "task_prompts.json"])
    return {str(path.relative_to(base.parent)): sha256_file(path)
            for path in paths if path.is_file()}


def _record_key(row, own: pd.DataFrame, config, group: str, ids: list[str],
                registry: pd.DataFrame, project: dict, alignment_hash: str,
                versions: dict[str, str], implementation: dict[str, str],
                audio_identity: dict[str, str], extra_inputs: dict) -> tuple[str, dict]:
    decision_fields = row._asdict()
    decision_fields.pop("Index", None)
    decision = pd.DataFrame([decision_fields])
    identity = {
        "recording_id": str(row.recording_id),
        "task": {key: project.get(key) for key in
                 ("task_id", "task_type", "task_name", "task_variant", "prompt_id",
                  "prompt_version", "expected_repetitions")},
        "source_audio_content_sha256": audio_identity["source_audio_content_sha256"],
        "analysis_audio_content_sha256": audio_identity["analysis_audio_content_sha256"],
        "review_decision_sha256": _csv_identity(decision),
        "review_intervals_sha256": _csv_identity(own),
        "alignment_identity": alignment_hash,
        "additional_input_identity": extra_inputs,
        "group": group,
        "selected_feature_ids": sorted(ids),
        "registry": registry.to_dict("records"),
        "parameters": {key: value for key, value in config.to_dict().items()
                       if key not in {"selected_features", "resume_existing_run"}},
        "implementation": implementation,
        "dependencies": versions,
    }
    provenance = {**audio_identity, **identity}
    return _digest(identity), provenance


def _context(row, task_id: str, task_type: str) -> dict:
    return {"recording_id": str(row.recording_id), "file_name": str(row.file_name),
            "task_id": task_id, "task_type": task_type,
            "segmentation_run_id": str(row.segmentation_run_id),
            "review_run_id": str(row.review_run_id)}


def _unavailable(row, feature_id: str, reason: str, lookup, task_id: str,
                 task_type: str, status: str = "unavailable") -> dict:
    version = str(lookup.loc[feature_id, "algorithm_version"])
    return {**_context(row, task_id, task_type), "feature_id": feature_id,
            "family_id": str(lookup.loc[feature_id, "family_id"]),
            "value": np.nan, "unit": str(lookup.loc[feature_id, "unit"]),
            "status": status, "failure_reason": reason, "algorithm": version,
            "algorithm_version": version,
            "parameter_set_id": str(lookup.loc[feature_id, "parameter_set_id"]),
            "provenance_ref": ""}


def _normalized_rows(group_dir: Path, checkpoint_path: Path, provenance: dict,
                     ids: list[str], lookup, task_id: str, task_type: str) -> list[dict]:
    source = group_dir / "tables" / "acoustic_feature_status_long.csv"
    output = []
    for item in pd.read_csv(source, keep_default_na=False).to_dict("records"):
        feature_id = str(item.get("feature_id") or item.get("feature") or "")
        if feature_id not in ids:
            raise ValueError(f"Executor emitted unselected output {feature_id}")
        item["feature_id"] = feature_id
        item["family_id"] = str(lookup.loc[feature_id, "family_id"])
        item["task_id"] = task_id
        item["task_type"] = task_type
        item["unit"] = str(lookup.loc[feature_id, "unit"])
        item["algorithm_version"] = str(item.get("algorithm_version") or
                                        lookup.loc[feature_id, "algorithm_version"])
        item["algorithm"] = str(item.get("algorithm") or item["algorithm_version"])
        item["parameter_set_id"] = str(item.get("parameter_set_id") or
                                       lookup.loc[feature_id, "parameter_set_id"])
        item["failure_reason"] = str(item.get("failure_reason") or item.get("reason") or "")
        # Scoped result rows use per-record review identity. Family audit files
        # retain their original whole-input provenance from the computation.
        if "final_decisions_sha256" in item:
            item["final_decisions_sha256"] = provenance["review_decision_sha256"]
        if "final_intervals_sha256" in item:
            item["final_intervals_sha256"] = provenance["review_intervals_sha256"]
        if "segmentation_review_hash" in item:
            item["segmentation_review_hash"] = (
                provenance["review_decision_sha256"] + ":" +
                provenance["review_intervals_sha256"])
        item["provenance_ref"] = str(checkpoint_path)
        native = str(item.get("native_measurement_table") or "")
        item["native_measurement_path"] = str(group_dir / "tables" / "native_measurements" / native) if native else ""
        output.append(item)
    if set(item["feature_id"] for item in output) != set(ids) or len(output) != len(ids):
        raise ValueError("executor_missing_result")
    return output


def _write_handoff(tables: Path, registry: pd.DataFrame, values: Path,
                   status: Path) -> dict[str, Path]:
    manifest = tables / "feature_export_manifest.json"
    aliases = {"feature_values_csv": tables / "feature_values.csv",
               "feature_registry_csv": tables / "feature_registry.csv",
               "feature_status_csv": tables / "feature_status.csv"}
    shutil.copyfile(values, aliases["feature_values_csv"])
    shutil.copyfile(status, aliases["feature_status_csv"])
    normalize_feature_registry(registry, "acoustic").to_csv(
        aliases["feature_registry_csv"], index=False)
    manifest.write_text(json.dumps({
        "contract": "vslp-feature-handoff", "contract_version": "1.0.0",
        "modality": "acoustic", "row_granularity": "one row per source recording and task",
        "canonical_outputs": {"feature_values": str(aliases["feature_values_csv"]),
                              "feature_registry": str(aliases["feature_registry_csv"]),
                              "feature_status": str(aliases["feature_status_csv"])},
        "legacy_sources": {"feature_values": str(values),
                           "feature_registry": str(tables / "selected_acoustic_feature_registry.csv")},
        "warning": "Identifiers and QC fields are context, not default model predictors.",
    }, indent=2), encoding="utf-8")
    return {**aliases, "feature_export_manifest_json": manifest}


def run_checkpointed_stage(decisions: Path, intervals: Path, root: Path,
                           stage: Path, tables: Path, config, registry: pd.DataFrame,
                           kept: pd.DataFrame, project: dict, task_id: str,
                           task_type: str, task_name: str, selected: list[str],
                           approved: dict, lookup, executors: dict,
                           groups: dict[str, list[str]], unsupported: list[str],
                           package_required: list[str], imported: bool,
                           progress_callback=None, cancel_event=None) -> StageResult:
    """Execute each recording/group once; assemble CSVs from verified checkpoints."""
    started = perf_counter()
    checkpoint_root = stage / "checkpoints_v1"
    timeline = pd.read_csv(intervals, keep_default_na=False)
    timeline_by_record = {str(key): own.copy() for key, own in
                          timeline.groupby(timeline.recording_id.astype(str), sort=False)}
    del timeline
    record_ids = set(kept.recording_id.astype(str))
    needs_alignment = any(requires_alignment_for_task(approved[id_], task_name)
                          for ids in groups.values() for id_ in ids)
    alignment_ids, alignment_issue = (_alignment_identities(root, record_ids)
                                     if needs_alignment else ({}, ""))
    versions = {group: _dependency_versions(group) for group in groups}
    implementation = {group: _implementation_identity(group) for group in groups}
    choices_path = root / "configs" / "alignment_choices.json"
    try:
        choices = json.loads(choices_path.read_text(encoding="utf-8")) if choices_path.is_file() else {}
    except (OSError, ValueError):
        choices = {"invalid_choices_file": True}
    choice_trials = choices.get("trials", {}) if isinstance(choices.get("trials"), dict) else {}
    choice_recordings = (choices.get("recordings", {})
                         if isinstance(choices.get("recordings"), dict) else {})
    ddk_path = root / "acoustic" / "003_segmentation_review" / "final" / "final_ddk_events.csv"
    ddk_hashes = {}
    if "ddk" in groups and ddk_path.is_file():
        events = pd.read_csv(ddk_path, keep_default_na=False)
        if "recording_id" in events:
            ddk_hashes = {str(name): _csv_identity(frame) for name, frame in
                          events.groupby(events.recording_id.astype(str), sort=False)}
        del events
    annotation_hashes = {}
    annotation_path = getattr(config, "acoustic_subevent_annotations_csv", None)
    if "segmental" in groups and annotation_path and Path(annotation_path).is_file():
        annotations = pd.read_csv(annotation_path, keep_default_na=False)
        if "recording_id" in annotations:
            annotation_hashes = {str(name): _csv_identity(frame) for name, frame in
                                 annotations.groupby(annotations.recording_id.astype(str),
                                                     sort=False)}
        del annotations
    external_paths = {"formants": ("vowel_category_manifest_path",),
                      "segmental": ("segmental_target_manifest_path",)}
    external_hashes = {}
    for group, fields in external_paths.items():
        if group in groups:
            external_hashes[group] = {
                field: sha256_file(Path(value)) if value and Path(value).is_file() else ""
                for field in fields if (value := getattr(config, field, None)) is not None}
    group_registries = {group: registry.loc[registry.feature.isin(ids)].copy()
                        for group, ids in groups.items()}
    group_alignment_needed = {
        group: any(requires_alignment_for_task(approved[id_], task_name) for id_ in ids)
        for group, ids in groups.items()}
    record_data = {}
    units = []
    counts = {"completed_previously": 0, "to_compute": 0, "stale": 0,
              "failed_retry": 0, "corrupt": 0, "incomplete": 0}

    def identity_for(record_id: str, group: str) -> tuple[str, dict]:
        row, own, audio_identity, _input_issue = record_data[record_id]
        alignment_hash = (alignment_ids.get(record_id, alignment_issue)
                          if group_alignment_needed[group] else "")
        extra_inputs = dict(external_hashes.get(group, {}))
        if group == "segmental":
            extra_inputs["manual_subevents"] = annotation_hashes.get(record_id, "missing")
        if group != "ddk":
            extra_inputs["trial_choices"] = _digest({
                "invalid_file": bool(choices.get("invalid_choices_file")),
                "trials": choice_trials.get(record_id, []),
                "recording": choice_recordings.get(record_id, {}),
            })
        else:
            extra_inputs["reviewed_ddk_events"] = ddk_hashes.get(record_id, "missing")
        return _record_key(row, own, config, group, groups[group],
                           group_registries[group], project, alignment_hash,
                           versions[group], implementation[group],
                           audio_identity, extra_inputs)

    scan_started = perf_counter()
    for row in kept.itertuples():
        record_id = str(row.recording_id)
        own = timeline_by_record.get(record_id, pd.DataFrame())
        input_issue = ""
        try:
            audio_path = Path(row.analysis_wav_path)
            source_value = getattr(row, "source_file_path", "")
            source = Path(source_value) if isinstance(source_value, str) and source_value else audio_path
            if not source.is_file():
                source = audio_path
            audio_identity = {
                "source_audio_content_sha256": audio_content_sha256(source),
                "analysis_audio_content_sha256": audio_content_sha256(audio_path),
                "source_file_sha256": sha256_file(source),
                "analysis_file_sha256": sha256_file(audio_path),
            }
        except Exception as exc:
            input_issue = f"input_audio_identity_failed:{type(exc).__name__}:{str(exc).splitlines()[0]}"
            audio_identity = {"source_audio_content_sha256": input_issue,
                              "analysis_audio_content_sha256": input_issue,
                              "source_file_sha256": "", "analysis_file_sha256": ""}
        record_data[record_id] = (row, own, audio_identity, input_issue)
        for group in groups:
            key, _provenance = identity_for(record_id, group)
            path = checkpoint_root / group / f"{_digest(record_id)[:24]}.json"
            state, _document = _checkpoint(path, key)
            counts[{"VALID COMPLETE": "completed_previously", "STALE": "stale",
                    "FAILED": "failed_retry", "CORRUPT": "corrupt",
                    "INCOMPLETE": "incomplete"}[state]] += 1
            if state != "VALID COMPLETE":
                counts["to_compute"] += 1
            units.append((record_id, group, key, path, state))
    scan_seconds = perf_counter() - scan_started
    total = len(units)
    if progress_callback:
        progress_callback(0, total, "Acoustic Features · " +
                          ", ".join(f"{name}: {value}" for name, value in counts.items()))
    compute_seconds = 0.0
    checkpoint_seconds = 0.0
    for index, (record_id, group, key, path, state) in enumerate(units, 1):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("feature_extraction_cancelled")
        if state != "VALID COMPLETE":
            row, own, _audio_identity, input_issue = record_data[record_id]
            ids = groups[group]
            subregistry = group_registries[group]
            current_key, provenance = identity_for(record_id, group)
            if current_key != key:
                raise RuntimeError(f"input_changed_during_extraction:{record_id}:{group}")
            if progress_callback:
                progress_callback(index - 1, total,
                                  f"Recording {index}/{total} · {group} · "
                                  f"elapsed {perf_counter() - started:.1f}s")
            unit_started = perf_counter()
            group_dir = checkpoint_root / "chunks" / group / _digest(record_id)[:24] / key
            scope = RecordingScope(pd.DataFrame([row._asdict()]), own,
                                   provenance["review_decision_sha256"],
                                   provenance["review_intervals_sha256"])
            extra = {}
            if group == "formants":
                from vslp.acoustic.features.formant_service import FormantService
                extra["formant_service"] = FormantService()
            try:
                if input_issue:
                    raise ValueError(input_issue)
                executor = executors[group][1]
                result = executor(decisions, root, replace(config, selected_features=ids),
                                  intervals, subregistry, execution_stage_dir=group_dir,
                                  recording_scope=scope, **extra)
                rows = _normalized_rows(group_dir, path, provenance,
                                        ids, lookup, task_id, task_type)
                family_manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
                artifact_paths = {Path(result.manifest_path),
                                  group_dir / "tables" / "acoustic_feature_status_long.csv"}
                artifact_paths.update(Path(item["path"]) for item in
                                      family_manifest.get("output_artifacts", [])
                                      if item.get("path") and Path(item["path"]).is_file())
                artifacts = [{"path": str(candidate), "sha256": sha256_file(candidate)}
                             for candidate in sorted(artifact_paths) if candidate.is_file()]
                document = {"key": key, "state": "COMPLETE", "recording_id": record_id,
                            "family_group": group, "rows": rows, "artifacts": artifacts,
                            "provenance": provenance}
            except Exception as exc:
                reason = f"family_executor_failed:{type(exc).__name__}:{str(exc).splitlines()[0]}"
                document = {"key": key, "state": "FAILED", "recording_id": record_id,
                            "family_group": group, "failure_reason": reason,
                            "traceback": traceback.format_exc(), "retry_eligible": True,
                            "rows": [_unavailable(row, id_, reason, lookup, task_id,
                                                  task_type, "error") for id_ in ids],
                            "artifacts": [], "provenance": provenance}
            compute_seconds += perf_counter() - unit_started
            write_started = perf_counter()
            _write_checkpoint(path, document)
            checkpoint_seconds += perf_counter() - write_started
        if progress_callback:
            progress_callback(index, total,
                              f"Recording {index}/{total} · {group} · "
                              f"{perf_counter() - started:.1f}s")

    # Build one record at a time; the number of selected columns is bounded by
    # the registry rather than by the number of recordings.
    assembly_started = perf_counter()
    columns = list(BASE_COLUMNS)
    column_set = set(columns)
    for _record_id, _group, key, path, _state in units:
        state, document = _checkpoint(path, key)
        if state not in {"VALID COMPLETE", "FAILED"}:
            raise ValueError(f"Checkpoint changed during assembly: {path} ({state})")
        for item in document["rows"]:
            for name in item:
                if name not in column_set:
                    columns.append(name)
                    column_set.add(name)
    for name in ("input_contract", "alignment_contract_version", "alignment_run_id"):
        if name not in column_set:
            columns.append(name)
    scalar = [id_ for id_ in selected
              if str(lookup.loc[id_, "output_granularity"]) == "recording"]
    ddk_context = ("n_ddk_events", "n_valid_sequences", "raw_analysis_duration_sec",
                   "excluded_contamination_duration_sec", "effective_ddk_duration_sec")
    wide_columns = [*BASE_COLUMNS, *scalar,
                    *(ddk_context if "F10" in set(registry.family_id.astype(str)) else ())]
    tables.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    temp_values = values_path.with_suffix(".csv.part")
    temp_status = status_path.with_suffix(".csv.part")
    success = failures = 0
    from vslp.acoustic.alignment import load_final_alignment
    alignment, _ = load_final_alignment(root) if needs_alignment else (None, "")
    alignment_provenance = alignment.manifest if alignment is not None else {}
    imported_records = set(alignment.trials.recording_id.astype(str)) if imported and alignment is not None and getattr(alignment, "trials", None) is not None else set()

    def input_contract(feature_id: str) -> str:
        if requires_alignment_for_task(approved[feature_id], task_name):
            return "FINAL_ALIGNMENT_CONTRACT_V1"
        family = approved[feature_id]["family_id"]
        return ("FINAL_DDK_EVENTS" if family == "F10" else
                "FINAL_REVIEWED_TIMING" if family in {"F07", "F08", "F09"}
                else "FINAL_REVIEWED_SEGMENTATION")

    with temp_values.open("w", newline="", encoding="utf-8") as values_stream, \
            temp_status.open("w", newline="", encoding="utf-8") as status_stream:
        value_writer = csv.DictWriter(values_stream, fieldnames=wide_columns, extrasaction="ignore")
        status_writer = csv.DictWriter(status_stream, fieldnames=columns, extrasaction="ignore")
        value_writer.writeheader()
        status_writer.writeheader()
        unit_by_record = {}
        for unit in units:
            unit_by_record.setdefault(unit[0], []).append(unit)
        for row in sorted(kept.itertuples(), key=lambda item: str(item.recording_id)):
            record_id = str(row.recording_id)
            record_rows = []
            for _record_id, _group, key, path, _state in unit_by_record.get(record_id, []):
                _state, document = _checkpoint(path, key)
                record_rows.extend(document["rows"])
            for feature_id in unsupported + package_required:
                reason = ("alignment_package_required" if feature_id in package_required
                          else "unsupported_for_current_task")
                record_rows.append(_unavailable(row, feature_id, reason, lookup, task_id, task_type))
            by_feature = {str(item["feature_id"]): item for item in record_rows}
            if len(by_feature) != len(record_rows):
                raise ValueError("Duplicate recording/feature result from family executors")
            wide = _context(row, task_id, task_type)
            for feature_id in sorted(selected):
                item = by_feature.get(feature_id)
                if item is None:
                    item = _unavailable(row, feature_id, "executor_missing_result",
                                        lookup, task_id, task_type, "error")
                aligned = requires_alignment_for_task(approved[feature_id], task_name)
                if imported and config.require_imported_alignment and aligned:
                    reason = ("alignment_package_invalid" if alignment is None else
                              "alignment_missing_for_recording" if record_id not in imported_records else "")
                    if reason:
                        item.update(value=np.nan, status="unavailable", failure_reason=reason)
                contract = input_contract(feature_id)
                item["input_contract"] = contract
                item["alignment_contract_version"] = contract if aligned else ""
                item["alignment_run_id"] = alignment_provenance.get("alignment_run_id", "") if aligned else ""
                item["value"] = pd.to_numeric(item.get("value"), errors="coerce")
                if item.get("failure_reason"):
                    failures += 1
                else:
                    success += 1
                status_writer.writerow(item)
                if feature_id in scalar:
                    wide[feature_id] = item["value"]
                if item.get("family_id") == "F10":
                    for name in ddk_context:
                        if name in item and name not in wide:
                            wide[name] = item[name]
            value_writer.writerow(wide)
        values_stream.flush()
        status_stream.flush()
        os.fsync(values_stream.fileno())
        os.fsync(status_stream.fileno())
    os.replace(temp_values, values_path)
    os.replace(temp_status, status_path)
    registry.to_csv(registry_path, index=False)
    handoff = _write_handoff(tables, registry, values_path, status_path)
    build_feature_delivery(root, "acoustic", handoff)
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version="mixed-family-checkpoint-v1",
        status="completed_with_warnings" if failures else "completed",
        input_artifacts=[ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
                         ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")],
        output_artifacts=[ArtifactRef(str(values_path), "feature_values", "text/csv"),
                          ArtifactRef(str(status_path), "feature_status", "text/csv"),
                          ArtifactRef(str(registry_path), "feature_registry", "text/csv")],
        config={"selected_feature_ids": selected, "checkpoint_schema": SCHEMA,
                "checkpoint_counts": counts, "successful_outputs": success,
                "failed_outputs": failures,
                "performance_timing_seconds": {
                    "resume_scan": round(scan_seconds, 6),
                    "extraction_compute": round(compute_seconds, 6),
                    "checkpoint_write": round(checkpoint_seconds, 6),
                    "assembly_and_writing": round(perf_counter() - assembly_started, 6),
                }},
    )
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
