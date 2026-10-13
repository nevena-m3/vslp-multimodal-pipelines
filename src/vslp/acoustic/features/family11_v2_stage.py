"""Independent, trial-first F11 executor using final reviewed timing and native audio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.alignment.reviewed_trials import load_confirmed_trials
from vslp.acoustic.features.catalog import task_key
from vslp.acoustic.features.family07_v2 import _eligible_speech
from vslp.acoustic.features.family11_v2 import PUBLIC_IDS, TASKS, VERSION, measure_trial
from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


def _bounds(frame: pd.DataFrame, role: str) -> list[tuple[float, float]]:
    return [(float(row.start_sec), float(row.end_sec)) for row in
            frame.loc[frame.segment_role.astype(str).eq(role)].itertuples()]


def _trials(root: Path, recording_id: str, task_id: str, frame: pd.DataFrame,
            decision) -> tuple[list[dict], str]:
    if task_id != "sustained_a" and (root / "configs" / "alignment_choices.json").is_file():
        return load_confirmed_trials(root, recording_id, task_id)
    speech = _bounds(frame, "speech")
    if not speech:
        return [], "no_valid_speech_intervals"
    start, end = min(a for a, _ in speech), max(b for _, b in speech)
    if task_id == "sustained_a":
        try:
            full_start, full_end = (float(getattr(decision, "full_phonation_start")),
                                    float(getattr(decision, "full_phonation_end")))
            if np.isfinite([full_start, full_end]).all() and full_end > full_start:
                start, end = full_start, full_end
        except (ValueError, TypeError, AttributeError):
            pass
    return [{"trial_id": f"{recording_id}:reviewed_001",
             "sequence_id": f"{recording_id}:reviewed_001",
             "start_sec": start, "end_sec": end,
             "trial_review_source": "FINAL_REVIEWED_SEGMENTATION"}], ""


def _trial_audio(audio: np.ndarray, sr: int, frame: pd.DataFrame, trial: dict,
                 task_id: str) -> tuple[np.ndarray, np.ndarray | None, str,
                                        list[tuple[float, float]], list[np.ndarray]]:
    left, right = float(trial["start_sec"]), float(trial["end_sec"])
    if not np.isfinite([left, right]).all() or left < 0 or right <= left or right > len(audio) / sr:
        raise ValueError("invalid_trial_bounds")
    speech = [(max(left, a), min(right, b)) for a, b in _bounds(frame, "speech")]
    speech = [(a, b) for a, b in speech if b > a]
    exclusions = [(max(left, a), min(right, b)) for a, b in _bounds(frame, "manual_exclusion")
                  if a < right and b > left]
    eligible = _eligible_speech(speech, exclusions)
    if task_id == "sustained_a":
        if exclusions:
            raise ValueError("hard_exclusion_ambiguity")
        # The accepted phonation onset/termination, not F01's stable two-second window.
        eligible = [(left, right)]
    if not eligible:
        raise ValueError("no_valid_speech_intervals")
    chunks = [audio[round(a * sr):round(b * sr)] for a, b in eligible]
    speech_audio = np.concatenate(chunks)
    connected, issue = None, ""
    if task_id == "bamboo_passage":
        span_start, span_end = eligible[0][0], eligible[-1][1]
        if any(a < span_end and b > span_start for a, b in exclusions):
            issue = "hard_exclusion_ambiguity"
        else:
            connected = audio[round(span_start * sr):round(span_end * sr)].copy()
    return speech_audio, connected, issue, eligible, chunks


def run_family11_stage(decisions_csv: str | Path, root: str | Path, config,
                       intervals_csv: str | Path | None, registry: pd.DataFrame,
                       progress_callback=None, execution_stage_dir: Path | None = None) -> StageResult:
    root = Path(root)
    final = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    if (intervals_csv is None or Path(intervals_csv).resolve() != intervals.resolve()
            or Path(decisions_csv).resolve() != decisions.resolve()):
        raise ValueError("Family 11 requires frozen reviewed segmentation")
    kept = load_final_segmentation(intervals, decisions)
    timeline = pd.read_csv(intervals, keep_default_na=False)
    project = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    task_id = str(project.get("task_id") or task_key(str(project.get("task_name", ""))) or "")
    selected = set(config.selected_features) & PUBLIC_IDS
    specs = registry.set_index("feature")
    stage = execution_stage_dir or root / "acoustic" / "005_features"
    tables = stage / "tables"
    native = tables / "native_measurements"
    values_rows, statuses, trials_audit = [], [], []
    for done, decision in enumerate(kept.itertuples(), 1):
        recording_id = str(decision.recording_id)
        own = timeline.loc[timeline.recording_id.astype(str).eq(recording_id)].copy()
        own = own.loc[own["view"].astype(str).eq("authoritative")] if "view" in own else own
        trials, trial_issue = _trials(root, recording_id, task_id, own, decision)
        base = {"recording_id": recording_id, "task_id": task_id,
                "participant_id": project.get("participant_id", ""),
                "session_id": project.get("session_id", project.get("visit_id", "")),
                "visit_id": project.get("visit_id", ""),
                "device_metadata": json.dumps(project.get("device_metadata", {}), sort_keys=True),
                "source_audio_path": str(getattr(decision, "source_file_path", "")),
                "source_audio_sha256": str(getattr(decision, "source_sha256", "")),
                "segmentation_review_hash": sha256_file(decisions) + ":" + sha256_file(intervals),
                "algorithm_version": VERSION, "input_contract": "FINAL_REVIEWED_SEGMENTATION"}
        if task_id in {"sustained_a", "sentence_s5", "bamboo_passage"} and trials:
            try:
                source = Path(base["source_audio_path"])
                if not source.is_file():
                    raise ValueError("native_source_audio_missing")
                audio, sr = sf.read(source, dtype="float64", always_2d=False)
                if audio.ndim != 1:
                    raise ValueError("native_audio_not_mono")
                for trial in trials:
                    try:
                        speech, connected, issue, eligible, chunks = _trial_audio(
                            audio, int(sr), own, trial, task_id)
                        results, audit = measure_trial(speech, int(sr), task_id,
                                                       energy_intervals=chunks,
                                                       connected_waveform=connected,
                                                       connected_issue=issue)
                        trial_reason = ""
                    except ValueError as exc:
                        results, audit = {}, {}
                        eligible, trial_reason = [], str(exc)
                    trials_audit.append({**base, "trial_id": trial["trial_id"],
                                         "sequence_id": trial["sequence_id"],
                                         "trial_start_sec": trial["start_sec"],
                                         "trial_end_sec": trial["end_sec"],
                                         "eligible_speech_intervals": json.dumps(eligible),
                                         "native_sample_rate_hz": sr, "failure_reason": trial_reason,
                                         **audit,
                                         **{feature: result[0] for feature, result in results.items()},
                                         **{f"{feature}_reason": result[1]
                                            for feature, result in results.items()}})
            except (OSError, ValueError) as exc:
                trial_issue = str(exc)
        own_trials = [row for row in trials_audit if row["recording_id"] == recording_id]
        value_row = dict(base)
        for feature in sorted(selected):
            valid = [float(row[feature]) for row in own_trials if feature in row
                     and not row.get(f"{feature}_reason") and np.isfinite(row[feature])]
            reason = ("" if valid else next((row.get(f"{feature}_reason") or row.get("failure_reason")
                                            for row in own_trials
                                            if row.get(f"{feature}_reason") or row.get("failure_reason")),
                                           trial_issue or "missing_reviewed_segmentation"))
            if task_id != TASKS[feature.split("_", 1)[0]]:
                reason = "task_not_applicable"
            value = float(np.median(valid)) if valid and not reason else np.nan
            value_row[feature] = value
            statuses.append({**base, "feature": feature, "feature_id": feature,
                             "family_id": "F11", "value": value,
                             "unit": str(specs.loc[feature, "unit"]),
                             "status": "computed" if not reason else "unavailable",
                             "failure_reason": reason, "valid_trial_count": len(valid),
                             "trial_count": len(trials)})
        values_rows.append(value_row)
        if progress_callback:
            progress_callback(done, len(kept), f"Acoustic Features — {decision.file_name}")
    native.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    audit_path = native / "family11_trial_values.csv"
    values_df, status_df = pd.DataFrame(values_rows), pd.DataFrame(statuses)
    values_df.to_csv(values_path, index=False)
    status_df.to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    pd.DataFrame(trials_audit).to_csv(audit_path, index=False)
    if execution_stage_dir is None:
        handoff = write_feature_handoff(tables, values_df, registry, status_df,
                                        "acoustic", values_path, registry_path)
        build_feature_delivery(root, "acoustic", handoff, optional_main={audit_path.name: audit_path})
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version=VERSION,
        status="completed_with_warnings" if status_df.failure_reason.astype(str).ne("").any()
        else "completed",
        input_artifacts=[ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
                         ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")],
        output_artifacts=[ArtifactRef(str(values_path), "feature_values", "text/csv"),
                          ArtifactRef(str(status_path), "feature_status", "text/csv"),
                          ArtifactRef(str(audit_path), "family11_trial_values", "text/csv")],
        config={"selected_features": sorted(selected), "input_contract": "FINAL_REVIEWED_SEGMENTATION"})
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
