"""F06 native acoustic subevents localized by feature-facing Final Alignment V1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.alignment import load_final_alignment
from vslp.acoustic.features.family06 import (
    F06_PARAMETER_SET_ID, F06_PUBLIC_IDS, F06_VERSION, TARGETS_V2,
    band_noise_power, first_spectral_moment_band, resolve_v1_target,
    stop_burst_tilt,
)
from vslp.acoustic.features.family06_subevents import (
    DETECTOR_VERSION, Subevent, detect_noise_interval, detect_noise_rise,
    detect_stop_burst,
)
from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


TOKEN_COLUMNS = (
    "recording_id", "trial_id", "sequence_id", "task_id", "task_item",
    "target_id", "word_id", "word", "phone_id", "canonical_phone",
    "alignment_run_id",
    "phone_start_sec", "phone_end_sec", "boundary_source", "alignment_review_status",
    "measure_type", "native_sample_rate_hz", "detector_version", "qc_status", "qc_flags",
    "value", "event_kind", "event_start_sec", "event_end_sec",
    "measurement_start_sec", "measurement_end_sec", "event_source",
    "prominence_ratio", "event_reviewer", "event_review_timestamp",
)
TRIAL_COLUMNS = ("recording_id", "trial_id", "sequence_id", "feature_id",
                 "value", "failure_reason")


def _needed_targets(feature: str) -> tuple[str, ...]:
    if feature.startswith("band_noise_"):
        suffix = feature.removeprefix("band_noise_a_").removeprefix("band_noise_b_")
        return ("see_s", "geese_s", "three_th") if suffix == "s_minus_th" else (suffix,)
    if feature == "spectral_m1_t_minus_k_hz":
        return ("spot_t", "key_k")
    if feature.startswith("fricative_noise_duration_"):
        return (("see_s", "geese_s", "three_th") if "s_minus_th" in feature else
                ("three_th",) if "_th_" in feature else ("see_s", "geese_s"))
    if feature.startswith("noise_rise_time_"):
        return (("judge_jh", "sentenced_t") if "t_minus_jh" in feature else
                ("judge_jh",) if "judge_jh" in feature else ("sentenced_t",))
    if feature == "spectral_tilt_k_minus_t_db_per_khz":
        return ("keep_k", "at_t")
    raise ValueError("unknown_family06_feature")


def _measure_type(feature: str) -> str:
    if feature.startswith("band_noise_"):
        return f"band_{feature.removeprefix('band_noise_')[0]}"
    if feature.startswith("fricative_noise_duration_"):
        return "duration"
    if feature.startswith("noise_rise_time_"):
        return "rise"
    return "m1" if "spectral_m1" in feature else "tilt"


def _event_kind(measure_type: str) -> str:
    if measure_type == "rise":
        return "NOISE_RISE_INTERVAL"
    if measure_type in {"duration", "band_a", "band_b"}:
        return "APERIODIC_NOISE_INTERVAL"
    return "STOP_BURST"


def _manual_event(annotations: pd.DataFrame, recording_id: str, trial_id: str,
                  phone_id: str, kind: str) -> Subevent | None:
    if annotations.empty:
        return None
    matches = annotations.loc[
        annotations.recording_id.astype(str).eq(recording_id)
        & annotations.trial_id.astype(str).eq(trial_id)
        & annotations.phone_id.astype(str).eq(phone_id)
        & annotations.event_kind.astype(str).eq(kind)]
    if matches.empty:
        return None
    if len(matches) != 1:
        raise ValueError("ambiguous_manual_subevent")
    item = matches.iloc[0]
    if (str(item.approved).casefold() not in {"true", "1", "yes"}
            or str(item.time_axis) != "original_recording_seconds"):
        raise ValueError("unapproved_acoustic_sub_event")
    return Subevent(kind, float(item.start_sec), float(item.end_sec),
                    "VALIDATED_MANUAL_OVERRIDE", np.nan,
                    reviewer=str(item.get("reviewer", "")),
                    review_timestamp=str(item.get("review_timestamp", "")))


def _load_annotations(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    table = pd.read_csv(path, keep_default_na=False)
    required = {"recording_id", "trial_id", "phone_id", "event_kind",
                "start_sec", "end_sec", "approved", "time_axis"}
    if not required <= set(table):
        raise ValueError("invalid_acoustic_subevent_annotations")
    return table


def _alignment_failure(issue: str) -> str:
    if issue == "missing_alignment":
        return "missing_final_alignment"
    if issue.startswith("stale_") or issue.endswith(("_mismatch", "_changed")):
        return "stale_final_alignment"
    return issue


def measure_token(audio: np.ndarray, rate: int, phone: pd.Series, trial: pd.Series,
                  measure_type: str, exclusions: list[tuple[float, float]],
                  annotations: pd.DataFrame, recording_id: str) -> dict:
    """Return a measurement or a reason; no provider waveform/timing is consulted."""
    start, end = float(phone.start_sec), float(phone.end_sec)
    if (not np.isfinite([start, end]).all() or not 0 <= float(trial.start_sec) <= start
            < end <= float(trial.end_sec) <= len(audio) / rate):
        raise ValueError("target_outside_eligible_audio")
    if any(start < right and end > left for left, right in exclusions):
        raise ValueError("target_outside_eligible_audio")
    kind = _event_kind(measure_type)
    event = _manual_event(annotations, recording_id, str(trial.trial_id),
                          str(phone.phone_id), kind)
    if event is None:
        detector = (detect_noise_rise if kind == "NOISE_RISE_INTERVAL" else
                    detect_noise_interval if kind == "APERIODIC_NOISE_INTERVAL" else
                    detect_stop_burst)
        event = detector(audio, rate, start, end)
    if (not start <= event.start_sec <= event.end_sec <= end
            or (kind != "STOP_BURST" and event.start_sec == event.end_sec)):
        raise ValueError("subevent_outside_aligned_phone")
    measurement_start = event.start_sec
    measurement_end = event.end_sec
    if measure_type in {"m1", "tilt"}:
        measurement_end = measurement_start + (.020 if measure_type == "m1" else .010)
    if measurement_end > end or measurement_end > float(trial.end_sec):
        raise ValueError("burst_window_outside_phone")
    if any(measurement_start < right and measurement_end > left
           for left, right in exclusions):
        raise ValueError("target_outside_eligible_audio")
    if measure_type.startswith("band_"):
        value = band_noise_power(audio, rate, measurement_start, measurement_end,
                                 measure_type[-1])
    elif measure_type == "duration":
        value = 1000 * (measurement_end - measurement_start)
    elif measure_type == "rise":
        value = 1000 * (measurement_end - measurement_start)
    elif measure_type == "m1":
        value = first_spectral_moment_band(audio, rate, measurement_start)
    else:
        value = stop_burst_tilt(audio, rate, measurement_start)
    if not np.isfinite(value):
        raise ValueError("invalid_acoustic_measurement")
    return {"value": float(value), "event_kind": kind,
            "event_start_sec": event.start_sec, "event_end_sec": event.end_sec,
            "measurement_start_sec": measurement_start,
            "measurement_end_sec": measurement_end, "event_source": event.source,
            "prominence_ratio": event.prominence_ratio,
            "event_reviewer": event.reviewer,
            "event_review_timestamp": event.review_timestamp}


def _feature_value(feature: str, values: dict[str, float]) -> float:
    """Combine only target summaries inside one trial; never pool phone classes."""
    needed = _needed_targets(feature)
    if any(key not in values for key in needed):
        raise ValueError("target_phone_missing")
    if feature.endswith("s_minus_th") or feature == "fricative_noise_duration_s_minus_th_ms":
        return float(np.median([values["see_s"], values["geese_s"]])
                     - values["three_th"])
    if feature == "fricative_noise_duration_s_ms":
        return float(np.median([values["see_s"], values["geese_s"]]))
    if feature == "spectral_m1_t_minus_k_hz":
        return values["spot_t"] - values["key_k"]
    if feature == "noise_rise_time_t_minus_jh_ms":
        return values["sentenced_t"] - values["judge_jh"]
    if feature == "spectral_tilt_k_minus_t_db_per_khz":
        return values["keep_k"] - values["at_t"]
    return values[needed[0]]


def run_family06_stage(decisions_csv: str | Path, root: str | Path, config,
                       intervals_csv: str | Path | None, registry: pd.DataFrame,
                       progress_callback=None, execution_stage_dir: Path | None = None,
                       recording_scope=None) -> StageResult:
    root = Path(root)
    review = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = review / "final_segmentation_decisions.csv"
    intervals = review / "final_segmentation_intervals.csv"
    if (Path(decisions_csv).resolve() != decisions.resolve() or intervals_csv is None
            or Path(intervals_csv).resolve() != intervals.resolve()):
        raise ValueError("Family 06 requires frozen authoritative reviewed segmentation")
    kept = (recording_scope.kept if recording_scope is not None else
            load_final_segmentation(intervals, decisions))
    timeline = (recording_scope.intervals if recording_scope is not None else
                pd.read_csv(intervals, keep_default_na=False))
    project = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    task_id = str(project.get("task_id", ""))
    selected = set(config.selected_features) & F06_PUBLIC_IDS
    specs = registry.set_index("feature")
    applicable = {feature for feature in selected
                  if task_id == TARGETS_V2[_needed_targets(feature)[0]][0]}
    alignment, issue = load_final_alignment(root) if applicable else (None, "")
    issue = _alignment_failure(issue)
    annotations = _load_annotations(getattr(config, "acoustic_subevent_annotations_csv", None))
    stage = execution_stage_dir or root / "acoustic" / "005_features"
    tables = stage / "tables"
    native = tables / "native_measurements"
    values_rows, statuses, tokens, trials_out, events = [], [], [], [], []
    if progress_callback:
        progress_callback(0, len(kept), "Acoustic Features — Family 06")
    for done, row in enumerate(kept.itertuples(), 1):
        recording_id = str(row.recording_id)
        own = timeline.loc[timeline.recording_id.astype(str).eq(recording_id)]
        exclusions = [(float(item.start_sec), float(item.end_sec)) for item in
                      own.loc[own.segment_role.eq("manual_exclusion")].itertuples()]
        speech = [(float(item.start_sec), float(item.end_sec)) for item in
                  own.loc[own.segment_role.eq("speech")].itertuples()]
        source = str(row.analysis_wav_path)
        audio, rate = (None, 0)
        audio_issue = issue
        if alignment is not None:
            try:
                audio, rate = sf.read(source, dtype="float32", always_2d=False)
                if audio.ndim != 1 or not np.isfinite(audio).all():
                    raise ValueError("invalid_canonical_audio")
            except (OSError, ValueError, RuntimeError) as exc:
                audio_issue = str(exc).splitlines()[0]
        measured = {}
        if alignment is not None and audio is not None:
            own_trials = alignment.trials.loc[
                alignment.trials.recording_id.astype(str).eq(recording_id)]
            for trial in own_trials.itertuples():
                trial_values: dict[tuple[str, str], float] = {}
                trial_errors: dict[str, str] = {}
                for feature in sorted(applicable):
                    kind = _measure_type(feature)
                    for target_id in _needed_targets(feature):
                        if (target_id, kind) in trial_values:
                            continue
                        phones, target_issue = resolve_v1_target(
                            alignment, recording_id, str(trial.trial_id), target_id)
                        if target_issue:
                            trial_errors[target_id] = target_issue
                            continue
                        valid = []
                        for phone in phones.itertuples():
                            audit = {"recording_id": recording_id, "trial_id": str(trial.trial_id),
                                     "sequence_id": str(phone.sequence_id), "task_id": task_id,
                                     "task_item": str(phone.task_item), "target_id": target_id,
                                     "word_id": str(phone.parent_word_id),
                                     "word": str(phone.parent_word_text),
                                     "phone_id": str(phone.phone_id),
                                     "canonical_phone": str(phone.phone_normalized),
                                     "alignment_run_id": str(alignment.manifest.get(
                                         "alignment_run_id", "")),
                                     "phone_start_sec": float(phone.start_sec),
                                     "phone_end_sec": float(phone.end_sec),
                                     "boundary_source": str(phone.boundary_source),
                                     "alignment_review_status": str(phone.alignment_review_status),
                                     "measure_type": kind, "native_sample_rate_hz": rate,
                                     "detector_version": DETECTOR_VERSION,
                                     "qc_status": "FAILED", "qc_flags": ""}
                            try:
                                if str(phone.sequence_id) != str(trial.sequence_id):
                                    raise ValueError("target_outside_eligible_audio")
                                if not any(left <= float(phone.start_sec) < float(phone.end_sec)
                                           <= right for left, right in speech):
                                    raise ValueError("target_outside_eligible_audio")
                                result = measure_token(audio, rate, pd.Series(phone._asdict()),
                                                       pd.Series(trial._asdict()), kind,
                                                       exclusions, annotations, recording_id)
                                audit.update(result, qc_status="OK")
                                valid.append(result["value"])
                                events.append({**audit, "start_sec": result["event_start_sec"],
                                               "end_sec": result["event_end_sec"]})
                            except (ValueError, RuntimeError) as exc:
                                audit["qc_flags"] = str(exc).splitlines()[0]
                                trial_errors[target_id] = audit["qc_flags"]
                            tokens.append(audit)
                        if valid:
                            trial_values[target_id, kind] = float(np.median(valid))
                for feature in sorted(applicable):
                    kind = _measure_type(feature)
                    target_values = {target: trial_values[target, kind]
                                     for target in _needed_targets(feature)
                                     if (target, kind) in trial_values}
                    try:
                        value = _feature_value(feature, target_values)
                        reason = ""
                    except ValueError:
                        value = np.nan
                        reason = next((trial_errors[target] for target in _needed_targets(feature)
                                       if target in trial_errors), "target_phone_missing")
                    trials_out.append({"recording_id": recording_id,
                                       "trial_id": str(trial.trial_id),
                                       "sequence_id": str(trial.sequence_id),
                                       "feature_id": feature, "value": value,
                                       "failure_reason": reason})
            for feature in sorted(applicable):
                own_values = [float(item["value"]) for item in trials_out
                              if item["recording_id"] == recording_id
                              and item["feature_id"] == feature
                              and np.isfinite(item["value"])]
                reasons = [str(item["failure_reason"]) for item in trials_out
                           if item["recording_id"] == recording_id
                           and item["feature_id"] == feature and item["failure_reason"]]
                measured[feature] = ((float(np.median(own_values)), "") if own_values else
                                     (np.nan, reasons[0] if reasons else "target_phone_missing"))
        base = {"recording_id": recording_id, "file_name": str(row.file_name),
                "task_id": task_id, "source_file_path": str(getattr(row, "source_file_path", "")),
                "analysis_wav_path": source,
                "source_sha256": sha256_file(source), "native_sample_rate_hz": rate,
                "alignment_run_id": str(alignment.manifest.get("alignment_run_id", ""))
                if alignment else "", "algorithm_version": F06_VERSION,
                "parameter_set_id": F06_PARAMETER_SET_ID,
                "input_contract": "FINAL_ALIGNMENT_CONTRACT_V1" if alignment else "",
                "alignment_contract_version": "FINAL_ALIGNMENT_CONTRACT_V1" if alignment else ""}
        values_rows.append({**base, **{feature: measured.get(feature, (np.nan, ""))[0]
                                        for feature in selected}})
        for feature in sorted(selected):
            value, reason = measured.get(feature, (np.nan,
                "task_not_applicable" if feature not in applicable else
                audio_issue or "target_phone_missing"))
            statuses.append({**base, "feature": feature, "feature_id": feature,
                             "family_id": "F06", "value": value,
                             "unit": str(specs.loc[feature, "unit"]),
                             "status": "computed" if not reason else "unavailable",
                             "failure_reason": reason})
        if progress_callback:
            progress_callback(done, len(kept), f"Acoustic Features — {row.file_name}")
    native.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    token_path = native / "family06_tokens.csv"
    trial_path = native / "family06_trials.csv"
    event_path = native / "family06_subevents.csv"
    values_df, statuses_df = pd.DataFrame(values_rows), pd.DataFrame(statuses)
    values_df.to_csv(values_path, index=False)
    statuses_df.to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    pd.DataFrame(tokens).reindex(columns=TOKEN_COLUMNS).to_csv(token_path, index=False)
    pd.DataFrame(trials_out).reindex(columns=TRIAL_COLUMNS).to_csv(trial_path, index=False)
    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df = events_df.drop_duplicates(
            subset=["recording_id", "trial_id", "phone_id", "event_kind", "event_source"])
    events_df.reindex(columns=(*TOKEN_COLUMNS, "start_sec", "end_sec")).to_csv(
        event_path, index=False)
    if execution_stage_dir is None:
        handoff = write_feature_handoff(tables, values_df, registry, statuses_df,
                                        "acoustic", values_path, registry_path)
        build_feature_delivery(root, "acoustic", handoff,
                               optional_main={path.name: path for path in
                                              (token_path, trial_path, event_path)})
    outputs = [ArtifactRef(str(path), role, "text/csv") for path, role in
               ((values_path, "feature_values"), (status_path, "feature_status"),
                (token_path, "family06_tokens"), (trial_path, "family06_trials"),
                (event_path, "family06_subevents"))]
    inputs = [ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
              ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")]
    if alignment is not None:
        final = root / "acoustic" / "004_alignment" / "final"
        inputs.extend(ArtifactRef(str(final / name), name, "text/csv") for name in
                      ("final_alignment_contract_trials.csv",
                       "final_alignment_canonical_words.csv",
                       "final_alignment_canonical_phones.csv"))
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version=F06_VERSION,
        status="completed_with_warnings" if any(item["failure_reason"] for item in statuses)
        else "completed",
        input_artifacts=inputs,
        output_artifacts=outputs,
        config={"selected_features": sorted(selected), "parameter_set_id": F06_PARAMETER_SET_ID,
                "detector_version": DETECTOR_VERSION,
                "band_a_hz": [720, 10000, 4200, 10000],
                "band_b_hz": [720, 6500, 720, 2150],
                "m1_band_hz": [500, 5000], "m1_window_ms": 20,
                "tilt_band_hz": [1500, 5000], "tilt_window_ms": 10,
                "alignment_run_id": str(alignment.manifest.get("alignment_run_id", ""))
                if alignment else "", "native_audio_policy": "native rate; no upsampling",
                "manual_annotations_sha256": sha256_file(config.acoustic_subevent_annotations_csv)
                if getattr(config, "acoustic_subevent_annotations_csv", None) else ""})
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
