"""F07 selected leaves with per-leaf Final Alignment V1 prerequisites."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vslp.acoustic.segment.review import load_final_segmentation
from vslp.acoustic.features.family07_v2 import (
    ALIGNMENT_IDS, FAMILY07_IDS, S3_PROMPT_ID, TIMING_IDS, TOKEN_COLUMNS, VERSION,
    _bounds, _vowel_coverage, confirmed_trials, reviewed_timing, s3_sentence_trials,
    vowel_tokens, vowel_trial_values,
)
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


def _alignment_issue(issue: str) -> str:
    return ("missing_final_alignment" if issue in {"missing_alignment", "alignment_not_frozen"}
            else "stale_final_alignment" if issue else "")


def run_family07_stage(decisions_csv: str | Path, root: str | Path, config,
                       intervals_csv: str | Path | None, registry: pd.DataFrame,
                       progress_callback=None, execution_stage_dir: Path | None = None) -> StageResult:
    root = Path(root)
    final = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    if (intervals_csv is None or Path(intervals_csv).resolve() != intervals.resolve()
            or Path(decisions_csv).resolve() != decisions.resolve()):
        raise ValueError("Family 07 requires frozen reviewed segmentation")
    kept = load_final_segmentation(intervals, decisions)
    timeline = pd.read_csv(intervals, keep_default_na=False)
    project = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    task_id = str(project.get("task_id", ""))
    selected = set(config.selected_features) & FAMILY07_IDS
    specs = registry.set_index("feature")
    alignment_needed = bool(selected & ALIGNMENT_IDS and task_id in {
        "sentence_s3", "bamboo_passage"})
    alignment, issue = (None, "")
    if alignment_needed:
        from vslp.acoustic.alignment import load_final_alignment

        alignment, issue = load_final_alignment(root)
    stage = execution_stage_dir or root / "acoustic" / "005_features"
    tables = stage / "tables"
    native = tables / "native_measurements"
    value_rows, statuses, timing_audit, token_audit, trial_audit = [], [], [], [], []
    if progress_callback:
        progress_callback(0, len(kept), "Acoustic Features — Family 07")
    for done, record in enumerate(kept.itertuples(), 1):
        recording_id = str(record.recording_id)
        own = timeline.loc[timeline.recording_id.astype(str).eq(recording_id)].copy()
        own = own.loc[own["view"].astype(str).eq("authoritative")] if "view" in own else own
        timing, events = reviewed_timing(own, task_id)
        if not events.empty:
            events["recording_id"] = recording_id
            events["trial_id"] = ""
            events["event_role"] = events.segment_role
            speech_events = events.loc[events.segment_role.astype(str).eq("eligible_speech")]
            events["first_speech_onset_sec"] = (float(speech_events.start_sec.min())
                                                 if not speech_events.empty else np.nan)
            events["last_speech_offset_sec"] = (float(speech_events.end_sec.max())
                                                 if not speech_events.empty else np.nan)
            timing_audit.extend(events.to_dict("records"))
        applicable = ({feature for feature in selected if feature.startswith("bamboo_")
                       or feature in {"delta_v_s", "varco_v_pct", "npvi_v_pct"}}
                      if task_id == "bamboo_passage" else
                      {feature for feature in selected if feature.startswith("s3_")}
                      if task_id == "sentence_s3" else set())
        measured: dict[str, tuple[float, str]] = {}
        for feature in applicable & TIMING_IDS & set(timing):
            measured[feature] = timing[feature]
        confirmed, trial_issue = (confirmed_trials(root, recording_id, task_id, decisions, intervals)
                                  if task_id == "sentence_s3" and applicable
                                  & {"s3_sentence_duration_s", "s3_vocalic_proportion_pct"}
                                  else ([], ""))
        sentence_rows = s3_sentence_trials(confirmed, own) if confirmed else []
        by_trial = {item["trial_id"]: item for item in sentence_rows}
        if "s3_sentence_duration_s" in applicable:
            durations = [item["duration_sec"] for item in sentence_rows
                         if not item["failure_reason"]]
            measured["s3_sentence_duration_s"] = (
                (float(np.median(durations)), "") if durations else
                (np.nan, next((item["failure_reason"] for item in sentence_rows
                               if item["failure_reason"]), trial_issue or "missing_trial_bounds")))
            for item in sentence_rows:
                trial_audit.append({"recording_id": recording_id, "task_id": task_id,
                                    "feature_id": "s3_sentence_duration_s",
                                    "trial_id": item["trial_id"], "value": item["duration_sec"],
                                    "trial_start_sec": item["start_sec"],
                                    "trial_end_sec": item["end_sec"],
                                    "failure_reason": item["failure_reason"],
                                    "boundary_source": item["boundary_source"]})
        if alignment is not None and applicable & ALIGNMENT_IDS:
            own_trials = alignment.trials.loc[
                alignment.trials.recording_id.astype(str).eq(recording_id)]
            for trial in own_trials.itertuples():
                if str(trial.task_id) != task_id or (task_id == "sentence_s3" and
                        str(trial.prompt_id) != S3_PROMPT_ID):
                    continue
                tokens = vowel_tokens(alignment, recording_id, trial, _bounds(own, "speech"),
                                      _bounds(own, "manual_exclusion"))
                if not tokens.empty:
                    token_audit.extend(tokens.to_dict("records"))
                sentence = by_trial.get(str(trial.trial_id), {})
                duration = float(sentence.get("duration_sec", np.nan))
                values = vowel_trial_values(tokens, task_id, duration)
                if task_id == "sentence_s3" and sentence:
                    if (abs(float(trial.start_sec) - sentence["start_sec"]) > 1e-6 or
                            abs(float(trial.end_sec) - sentence["end_sec"]) > 1e-6):
                        values["s3_vocalic_proportion_pct"] = (np.nan, "invalid_trial_bounds")
                    elif sentence["failure_reason"]:
                        values["s3_vocalic_proportion_pct"] = (
                            np.nan, sentence["failure_reason"])
                complete, coverage_reason, expected, fraction = _vowel_coverage(tokens, task_id)
                for feature in applicable & ALIGNMENT_IDS:
                    value, reason = values.get(feature, (np.nan, "missing_expected_vowel"))
                    if feature == "s3_vocalic_proportion_pct" and not sentence:
                        value, reason = np.nan, trial_issue or "missing_trial_bounds"
                    trial_audit.append({"recording_id": recording_id,
                                        "trial_id": str(trial.trial_id),
                                        "sequence_id": str(trial.sequence_id),
                                        "task_id": task_id, "feature_id": feature,
                                        "value": value, "failure_reason": reason,
                                        "expected_vowel_count": expected,
                                        "observed_valid_vowel_count": int(tokens.valid.sum())
                                        if not tokens.empty else 0,
                                        "vowel_coverage_fraction": fraction,
                                        "coverage_status": "complete" if complete else coverage_reason,
                                        "s3_total_vocalic_duration_s": float(tokens.loc[
                                            tokens.valid, "duration_sec"].sum())
                                        if task_id == "sentence_s3" and not tokens.empty
                                        else np.nan})
            for feature in applicable & ALIGNMENT_IDS:
                rows = [item for item in trial_audit if item.get("recording_id") == recording_id
                        and item["feature_id"] == feature]
                valid = [float(item["value"]) for item in rows
                         if np.isfinite(item["value"]) and not item["failure_reason"]]
                measured[feature] = ((float(np.median(valid)), "") if valid else
                                     (np.nan, next((item["failure_reason"] for item in rows
                                                    if item["failure_reason"]),
                                                   "missing_expected_vowel")))
        base = {"recording_id": recording_id, "file_name": str(record.file_name),
                "task_id": task_id, "run_id": str(project.get("run_id", "")),
                "alignment_run_id": str(alignment.manifest.get("alignment_run_id", ""))
                if alignment else "", "algorithm_version": VERSION,
                "final_decisions_sha256": sha256_file(decisions),
                "final_intervals_sha256": sha256_file(intervals)}
        values = {**base}
        for feature in sorted(selected):
            if feature not in applicable:
                value, reason = np.nan, "task_not_applicable"
            elif feature in measured:
                value, reason = measured[feature]
            elif feature in ALIGNMENT_IDS:
                value, reason = np.nan, _alignment_issue(issue) or "missing_expected_vowel"
            else:
                value, reason = np.nan, "missing_reviewed_segmentation"
            if not np.isfinite(value) and not reason:
                reason = "nonfinite_feature"
            values[feature] = value
            statuses.append({**base, "feature": feature, "feature_id": feature,
                             "family_id": "F07", "value": value,
                             "status": "computed" if not reason else "unavailable",
                             "failure_reason": reason, "unit": str(specs.loc[feature, "unit"]),
                             "input_contract": ("FINAL_ALIGNMENT_CONTRACT_V1" if feature in ALIGNMENT_IDS
                                                else "FINAL_REVIEWED_TIMING")})
        value_rows.append(values)
        if progress_callback:
            progress_callback(done, len(kept), f"Acoustic Features — {record.file_name}")
    native.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    timing_path = native / "family07_timing_events.csv"
    token_path = native / "family07_vocalic_tokens.csv"
    trial_path = native / "family07_trials.csv"
    values_df, status_df = pd.DataFrame(value_rows), pd.DataFrame(statuses)
    values_df.to_csv(values_path, index=False)
    status_df.to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    pd.DataFrame(timing_audit).to_csv(timing_path, index=False)
    pd.DataFrame(token_audit).reindex(columns=TOKEN_COLUMNS).to_csv(token_path, index=False)
    pd.DataFrame(trial_audit).to_csv(trial_path, index=False)
    if execution_stage_dir is None:
        handoff = write_feature_handoff(tables, values_df, registry, status_df,
                                        "acoustic", values_path, registry_path)
        build_feature_delivery(root, "acoustic", handoff,
                               optional_main={path.name: path for path in
                                              (timing_path, token_path, trial_path)})
    outputs = [ArtifactRef(str(path), role, "text/csv") for path, role in
               ((values_path, "feature_values"), (status_path, "feature_status"),
                (timing_path, "family07_timing_events"),
                (token_path, "family07_vocalic_tokens"),
                (trial_path, "family07_trials"))]
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version=VERSION,
        status="completed_with_warnings" if status_df.failure_reason.astype(str).ne("").any()
        else "completed",
        input_artifacts=[ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
                         ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")],
        output_artifacts=outputs,
        config={"selected_features": sorted(selected), "algorithm_version": VERSION,
                "alignment_run_id": str(alignment.manifest.get("alignment_run_id", ""))
                if alignment else "", "final_decisions_sha256": sha256_file(decisions),
                "final_intervals_sha256": sha256_file(intervals)},
    )
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
