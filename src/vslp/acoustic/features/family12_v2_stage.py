"""F12 reviewed-timing executor; no MFA, Alignment, or artificial STFT joins."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.alignment.reviewed_trials import load_confirmed_trials
from vslp.acoustic.features.catalog import task_key
from vslp.acoustic.features.family07_v2 import _eligible_speech, bamboo_reviewed_trial_domains
from vslp.acoustic.features.family11_v2_stage import _bounds
from vslp.acoustic.features.family12_v2 import PUBLIC_IDS, PREFIX, VERSION, measure_trial
from vslp.acoustic.features.stable_phonation import centered_stable_spans
from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


def _trial_regions(audio, sr, frame, trial, task_id, decision):
    left, right = float(trial["start_sec"]), float(trial["end_sec"])
    if not np.isfinite([left, right]).all() or left < 0 or right <= left or right > len(audio)/sr:
        raise ValueError("invalid_trial_bounds")
    if task_id == "sustained_a":
        centered = centered_stable_spans(decision, frame, len(audio)/sr, [trial])
        spans = [(a, b) for a, b, *_ in centered]
    else:
        speech = [(max(left, a), min(right, b)) for a, b in _bounds(frame, "speech")]
        speech = [(a, b) for a, b in speech if b > a]
        exclusions = _bounds(frame, "manual_exclusion")
        spans = _eligible_speech(speech, exclusions)
    if not spans:
        raise ValueError("no_valid_speech_intervals")
    chunks = [audio[round(a*sr):round(b*sr)].copy() for a, b in spans]
    return spans, chunks


def run_family12_stage(decisions_csv: str | Path, root: str | Path, config,
                       intervals_csv: str | Path | None, registry: pd.DataFrame,
                       progress_callback=None, execution_stage_dir: Path | None = None) -> StageResult:
    root = Path(root)
    final = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    if (intervals_csv is None or Path(intervals_csv).resolve() != intervals.resolve()
            or Path(decisions_csv).resolve() != decisions.resolve()):
        raise ValueError("Family 12 requires frozen reviewed segmentation")
    kept = load_final_segmentation(intervals, decisions)
    timeline = pd.read_csv(intervals, keep_default_na=False)
    project = json.loads((root/"project_manifest.json").read_text(encoding="utf-8"))
    task_id = str(project.get("task_id") or task_key(str(project.get("task_name", ""))) or "")
    selected = set(config.selected_features) & PUBLIC_IDS
    specs = registry.set_index("feature")
    stage = execution_stage_dir or root/"acoustic"/"005_features"
    tables = stage/"tables"
    native = tables/"native_measurements"
    values_rows, statuses, trial_rows = [], [], []
    review_hash = sha256_file(decisions) + ":" + sha256_file(intervals)
    for done, decision in enumerate(kept.itertuples(), 1):
        recording_id = str(decision.recording_id)
        frame = timeline.loc[timeline.recording_id.astype(str).eq(recording_id)].copy()
        if "view" in frame:
            frame = frame.loc[frame["view"].astype(str).eq("authoritative")]
        base = {"recording_id": recording_id, "task_id": task_id,
                "participant_id": project.get("participant_id", ""),
                "session_id": project.get("session_id", project.get("visit_id", "")),
                "device_metadata": json.dumps(project.get("device_metadata", {}), sort_keys=True),
                "source_audio_path": str(getattr(decision, "source_file_path", "")),
                "source_audio_sha256": str(getattr(decision, "source_sha256", "")),
                "segmentation_review_hash": review_hash,
                "input_contract": "FINAL_REVIEWED_SEGMENTATION", "algorithm_version": VERSION}
        issue = ""
        trials = []
        if task_id in PREFIX:
            if task_id == "sustained_a":
                speech = _bounds(frame, "speech")
                if speech:
                    trials = [{"trial_id": f"{recording_id}:reviewed_001",
                               "sequence_id": f"{recording_id}:reviewed_001",
                               "start_sec": min(a for a, _ in speech),
                               "end_sec": max(b for _, b in speech)}]
            elif task_id == "bamboo_passage":
                trials, issue = bamboo_reviewed_trial_domains(root, recording_id, frame)
            else:
                trials, issue = load_confirmed_trials(root, recording_id, task_id)
            if trials:
                try:
                    source = Path(base["source_audio_path"])
                    if not source.is_file():
                        raise ValueError("native_source_audio_missing")
                    audio, sr = sf.read(source, dtype="float64", always_2d=False)
                    if audio.ndim != 1:
                        raise ValueError("native_audio_not_mono")
                    for trial in trials:
                        try:
                            spans, chunks = _trial_regions(audio, int(sr), frame, trial,
                                                            task_id, decision)
                            values, audit = measure_trial(chunks, int(sr), task_id)
                            reason = ""
                        except ValueError as exc:
                            spans, values, audit, reason = [], {}, {}, str(exc)
                        trial_rows.append({**base, "trial_id": trial["trial_id"],
                                           "sequence_id": trial["sequence_id"],
                                           "prompt_id": trial.get("prompt_id", ""),
                                           "trial_review_source": trial.get("trial_review_source",
                                                                               "FINAL_REVIEWED_SEGMENTATION"),
                                           "trial_start_sec": trial["start_sec"],
                                           "trial_end_sec": trial["end_sec"],
                                           "eligible_intervals": json.dumps(spans),
                                           "failure_reason": reason, **audit,
                                           **{key: value for key, (value, _) in values.items()},
                                           **{key+"_reason": why for key, (_, why) in values.items()}})
                except (OSError, ValueError) as exc:
                    issue = str(exc)
        else:
            issue = "task_not_applicable"
        own = [row for row in trial_rows if row["recording_id"] == recording_id]
        output = dict(base)
        for feature in sorted(selected):
            task_for_feature = next((task for task, prefix in PREFIX.items()
                                     if feature.startswith(prefix+"_")), None)
            valid = [float(row[feature]) for row in own if feature in row
                     and not row.get(feature+"_reason") and np.isfinite(row[feature])]
            reason = "" if valid else next((row.get(feature+"_reason") or row.get("failure_reason")
                                             for row in own if row.get(feature+"_reason") or
                                             row.get("failure_reason")), issue or
                                            "missing_reviewed_segmentation")
            if task_for_feature != task_id:
                reason = "task_not_applicable"
            value = float(np.median(valid)) if valid and not reason else np.nan
            output[feature] = value
            statuses.append({**base, "feature": feature, "feature_id": feature,
                             "family_id": "F12", "value": value,
                             "unit": str(specs.loc[feature, "unit"]),
                             "status": "computed" if not reason else "unavailable",
                             "failure_reason": reason, "valid_trial_count": len(valid),
                             "trial_count": len(trials)})
        values_rows.append(output)
        if progress_callback:
            progress_callback(done, len(kept), f"Acoustic Features — {decision.file_name}")
    native.mkdir(parents=True, exist_ok=True)
    values_path = tables/"acoustic_features_per_file.csv"
    status_path = tables/"acoustic_feature_status_long.csv"
    registry_path = tables/"selected_acoustic_feature_registry.csv"
    audit_path = native/"family12_trial_values.csv"
    value_frame, status_frame = pd.DataFrame(values_rows), pd.DataFrame(statuses)
    value_frame.to_csv(values_path, index=False)
    status_frame.to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    pd.DataFrame(trial_rows).to_csv(audit_path, index=False)
    if execution_stage_dir is None:
        handoff = write_feature_handoff(tables, value_frame, registry, status_frame,
                                        "acoustic", values_path, registry_path)
        build_feature_delivery(root, "acoustic", handoff, optional_main={audit_path.name: audit_path})
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version=VERSION,
        status="completed_with_warnings" if status_frame.failure_reason.astype(str).ne("").any()
        else "completed",
        input_artifacts=[ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
                         ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")],
        output_artifacts=[ArtifactRef(str(values_path), "feature_values", "text/csv"),
                          ArtifactRef(str(status_path), "feature_status", "text/csv"),
                          ArtifactRef(str(audit_path), "family12_trial_values", "text/csv")],
        config={"selected_features": sorted(selected),
                "input_contract": "FINAL_REVIEWED_SEGMENTATION"})
    manifest_path = stage/"logs"/"stage_manifest.json"
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
