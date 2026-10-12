"""Family 05 execution from Final Alignment V1 and native recording audio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.alignment import load_final_alignment
from vslp.acoustic.features.family04 import profile_by_id, track_aligned_vowels
from vslp.acoustic.features.catalog import load_feature_catalog
from vslp.acoustic.features.family05 import (
    ALGORITHM_VERSION, AUDIT_COLUMNS, FAMILY05_IDS, PARAMETER_SET_ID, ROBUST_METHOD,
    TARGETS, measure_target_frames, resolve_target_phones, transition_window,
)
from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


def _selected_specs(registry: pd.DataFrame, selected: set[str]) -> dict[str, pd.Series]:
    return {feature: registry.loc[registry.feature.eq(feature)].iloc[0]
            for feature in selected}


def _alignment_failure(issue: str) -> str:
    if issue == "missing_alignment":
        return "missing_final_alignment"
    if issue.startswith("stale_") or issue.endswith("_mismatch") or issue.endswith("_changed"):
        return "stale_final_alignment"
    return issue


def run_family05_stage(decisions_csv: str | Path, root: str | Path, config,
                       intervals_csv: str | Path | None, registry: pd.DataFrame,
                       progress_callback=None, execution_stage_dir: Path | None = None) -> StageResult:
    """Measure each canonical target independently; failures remain NaN with reason."""
    root = Path(root)
    review = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = review / "final_segmentation_decisions.csv"
    intervals = review / "final_segmentation_intervals.csv"
    if (Path(decisions_csv).resolve() != decisions.resolve() or intervals_csv is None
            or Path(intervals_csv).resolve() != intervals.resolve()):
        raise ValueError("Family 05 requires authoritative frozen reviewed segmentation")
    kept = load_final_segmentation(intervals, decisions)
    timeline = pd.read_csv(intervals, keep_default_na=False)
    project = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    task_id = str(project.get("task_id", ""))
    selected = set(config.selected_features) & FAMILY05_IDS
    specs = _selected_specs(registry, selected)
    targets_by_feature = {item["feature_id"]: item for item in
                          load_feature_catalog()["outputs"] if item["feature_id"] in selected}
    profile = profile_by_id(getattr(config, "formant_profile_id", None)
                            or "family04_burg_native_5500_v1")
    applicable = {feature for feature in selected
                  if TARGETS[str(targets_by_feature[feature]["target_id"])][0] == task_id}
    alignment, alignment_issue = (load_final_alignment(root) if applicable else (None, ""))
    alignment_detail = alignment_issue
    alignment_issue = _alignment_failure(alignment_issue)
    alignment_run_id = str(alignment.manifest.get("alignment_run_id", "")) if alignment else ""
    stage = execution_stage_dir or root / "acoustic" / "005_features"
    tables = stage / "tables"
    native = tables / "native_measurements"
    status_rows: list[dict] = []
    values_rows: list[dict] = []
    trial_rows: list[dict] = []
    token_rows: list[dict] = []
    if progress_callback:
        progress_callback(0, len(kept), "Acoustic Features — Family 05")
    for done, record in enumerate(kept.itertuples(), 1):
        recording_id = str(record.recording_id)
        source_path = str(record.analysis_wav_path)
        source_hash = sha256_file(source_path)
        sample_rate = np.nan
        audio = None
        audio_issue = alignment_issue if applicable else ""
        if applicable and not audio_issue:
            try:
                audio, rate = sf.read(source_path, dtype="float32", always_2d=False)
                sample_rate = int(rate)
                if np.asarray(audio).ndim != 1:
                    raise ValueError("native_audio_not_mono")
            except (OSError, RuntimeError, ValueError) as exc:
                audio_issue = str(exc).splitlines()[0]
        own_intervals = timeline.loc[timeline.recording_id.astype(str).eq(recording_id)]
        exclusions = [(float(item.start_sec), float(item.end_sec)) for item in
                      own_intervals.loc[own_intervals.segment_role.eq("manual_exclusion")].itertuples()]
        speech = [(float(item.start_sec), float(item.end_sec)) for item in
                  own_intervals.loc[own_intervals.segment_role.eq("speech")].itertuples()]
        measured: dict[str, tuple[float, str]] = {}
        if audio is not None and alignment is not None:
            trials = alignment.trials.loc[
                alignment.trials.recording_id.astype(str).eq(recording_id)]
            for target_id in dict.fromkeys(str(targets_by_feature[feature]["target_id"])
                                           for feature in sorted(applicable)):
                for trial in trials.itertuples():
                    trial_id = str(trial.trial_id)
                    phones, target_issue = resolve_target_phones(
                        alignment, recording_id, trial_id, target_id)
                    trial_tokens: list[dict] = []
                    if not target_issue:
                        for phone in phones.itertuples():
                            start, end = float(phone.start_sec), float(phone.end_sec)
                            base = {"recording_id": recording_id, "trial_id": trial_id,
                                    "sequence_id": str(phone.sequence_id), "task_id": task_id,
                                    "task_item": str(phone.task_item), "word_id": str(phone.parent_word_id),
                                    "phone_id": str(phone.phone_id), "word": str(phone.parent_word_text),
                                    "canonical_phone": str(phone.phone_normalized), "target_id": target_id,
                                    "token_start_sec": start, "token_end_sec": end,
                                    "token_duration_sec": end - start, "native_sample_rate_hz": sample_rate,
                                    "formant_parameter_set_id": profile.profile_id,
                                    "boundary_source": str(phone.boundary_source),
                                    "alignment_review_status": str(phone.alignment_review_status),
                                    "alignment_run_id": alignment_run_id,
                                    "source_low_level_parity": "pending", "fit_method": ROBUST_METHOD,
                                    "qc_status": "FAILED", "qc_flags": ""}
                            try:
                                left, right = transition_window(start, end)
                                base.update(analysis_start_sec=left, analysis_end_sec=right,
                                            analysis_duration_sec=right - left)
                                if (not float(trial.start_sec) <= start < end <= float(trial.end_sec)
                                        or end > len(audio) / sample_rate
                                        or any(start < x_end and end > x_start
                                               for x_start, x_end in exclusions)
                                        or not any(x_start <= start < end <= x_end
                                                   for x_start, x_end in speech)):
                                    raise ValueError("target_outside_eligible_audio")
                                frame_phone = pd.DataFrame([{**phone._asdict(),
                                    "trial_start_sec": float(trial.start_sec),
                                    "trial_end_sec": float(trial.end_sec)}])
                                _, frames = track_aligned_vowels(
                                    audio, sample_rate, frame_phone, profile,
                                    recording_id=recording_id, file_name=str(record.file_name),
                                    task_id=task_id, task_type=str(project.get("task_type", "")),
                                    alignment_run_id=alignment_run_id,
                                    hard_exclusions=exclusions, analysis_window=(.20, .80))
                                base.update(measure_target_frames(frames, start=start, end=end,
                                                                  token_index=1))
                            except (ValueError, RuntimeError) as exc:
                                base["qc_flags"] = str(exc).splitlines()[0]
                            trial_tokens.append(base)
                            token_rows.append(base)
                    trial_reason = target_issue or ("" if any(
                        item["qc_status"] == "OK" for item in trial_tokens)
                        else str(trial_tokens[0].get("qc_flags", "target_phone_missing"))
                             if trial_tokens else "target_phone_missing")
                    for feature in sorted(applicable):
                        if str(targets_by_feature[feature]["target_id"]) != target_id:
                            continue
                        number = int(targets_by_feature[feature]["formant_number"])
                        column = (f"f{number}_fitted_range_hz" if feature.startswith("f")
                                  and "_range_" in feature else f"f{number}_slope_hz_s")
                        valid = [float(item[column]) for item in trial_tokens
                                 if item["qc_status"] == "OK" and np.isfinite(item[column])]
                        value = float(np.median(valid)) if valid else np.nan
                        reason = "" if valid else trial_reason
                        trial_rows.append({"recording_id": recording_id, "trial_id": trial_id,
                                           "sequence_id": str(trial.sequence_id),
                                           "task_id": task_id, "target_id": target_id,
                                           "feature_id": feature, "value": value,
                                           "failure_reason": reason,
                                           "n_valid_tokens": len(valid)})
                for feature in sorted(applicable):
                    if str(targets_by_feature[feature]["target_id"]) != target_id:
                        continue
                    candidates = [float(item["value"]) for item in trial_rows
                                  if item["recording_id"] == recording_id
                                  and item["feature_id"] == feature
                                  and np.isfinite(item["value"])]
                    reasons = [str(item["failure_reason"]) for item in trial_rows
                               if item["recording_id"] == recording_id
                               and item["feature_id"] == feature and item["failure_reason"]]
                    measured[feature] = ((float(np.median(candidates)), "") if candidates else
                                         (np.nan, reasons[0] if reasons else "target_phone_missing"))
        common = {"recording_id": recording_id, "file_name": str(record.file_name),
                  "task_id": task_id, "task_type": str(project.get("task_type", "")),
                  "source_file_path": str(record.source_file_path),
                  "analysis_wav_path": source_path, "source_sha256": source_hash,
                  "native_sample_rate_hz": sample_rate, "alignment_run_id": alignment_run_id,
                  "alignment_contract_version": "FINAL_ALIGNMENT_CONTRACT_V1" if alignment else "",
                  "algorithm_version": ALGORITHM_VERSION,
                  "parameter_set_id": PARAMETER_SET_ID, "formant_profile_id": profile.profile_id,
                  "source_low_level_parity": "pending",
                  "alignment_prerequisite_detail": alignment_detail,
                  "final_decisions_sha256": sha256_file(decisions),
                  "final_intervals_sha256": sha256_file(intervals)}
        values_rows.append({**common, **{feature: measured.get(feature, (np.nan, ""))[0]
                                        for feature in selected}})
        for feature in sorted(selected):
            value, reason = measured.get(feature, (np.nan,
                "task_not_applicable" if feature not in applicable else
                audio_issue or "target_phone_missing"))
            status_rows.append({**common, "feature": feature, "feature_id": feature,
                                "family_id": "F05", "value": value,
                                "unit": specs[feature].unit,
                                "row_granularity": "recording",
                                "status": "computed" if not reason else "unavailable",
                                "failure_reason": reason,
                                "native_measurement_table": "formant_dynamics_tokens.csv",
                                "input_contract": "FINAL_ALIGNMENT_CONTRACT_V1"})
        if progress_callback:
            progress_callback(done, len(kept), f"Acoustic Features — {record.file_name}")
    tables.mkdir(parents=True, exist_ok=True)
    native.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    token_path = native / "formant_dynamics_tokens.csv"
    trial_path = native / "formant_dynamics_trials.csv"
    values_df, status_df = pd.DataFrame(values_rows), pd.DataFrame(status_rows)
    values_df.to_csv(values_path, index=False)
    status_df.to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    pd.DataFrame(token_rows, columns=AUDIT_COLUMNS).to_csv(token_path, index=False)
    pd.DataFrame(trial_rows, columns=("recording_id", "trial_id", "sequence_id",
        "task_id", "target_id", "feature_id", "value", "failure_reason",
        "n_valid_tokens")).to_csv(trial_path, index=False)
    if execution_stage_dir is None:
        handoff = write_feature_handoff(tables, values_df, registry, status_df,
                                        "acoustic", values_path, registry_path)
        build_feature_delivery(root, "acoustic", handoff,
                               optional_main={token_path.name: token_path,
                                              trial_path.name: trial_path})
    inputs = [ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
              ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")]
    if alignment is not None:
        final = root / "acoustic" / "004_alignment" / "final"
        inputs.extend([ArtifactRef(str(final / name), name, "text/csv") for name in
                       ("final_alignment_contract_trials.csv",
                        "final_alignment_canonical_words.csv",
                        "final_alignment_canonical_phones.csv")])
    outputs = [ArtifactRef(str(path), role, "text/csv") for path, role in
               ((values_path, "feature_values"), (status_path, "feature_status"),
                (token_path, "formant_dynamics_tokens"),
                (trial_path, "formant_dynamics_trials"))]
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version=ALGORITHM_VERSION,
        status="completed_with_warnings" if any(item["failure_reason"] for item in status_rows)
        else "completed", input_artifacts=inputs, output_artifacts=outputs,
        config={"selected_features": sorted(selected), "formant_profile_id": profile.profile_id,
                "robust_method": ROBUST_METHOD, "analysis_window": [.20, .80],
                "alignment_run_id": alignment_run_id,
                "native_audio_policy": "canonical native-rate read-only"})
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
