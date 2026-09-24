"""Family 06 execution from frozen Alignment and validated acoustic annotations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.alignment import load_final_alignment
from vslp.acoustic.features.family06 import (
    ALGORITHM_VERSION, AUDIT_COLUMNS, IMPLEMENTED_IDS, PARAMETER_SET_ID,
    first_spectral_moment, load_target_manifest, load_validated_subevents,
    matched_m1_contrast, stop_burst_tilt, wideband_noise_energy,
)
from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


def _aligned_target(store, recording_id: str, target: dict):
    """Match exact frozen word and phone identity plus explicit token indices."""
    words = store.get_word_tokens(recording_id)
    phones = store.get_phone_tokens(recording_id)
    word = words.loc[words.word_index.eq(int(target["word_index"]))
                  & words.word.astype(str).eq(str(target["word"]))]
    phone = phones.loc[phones.word_index.eq(int(target["word_index"]))
                       & phones.phone_index.eq(int(target["phone_index"]))
                       & phones.phone_normalized.astype(str).eq(str(target["phone"]).upper())]
    if len(word) != 1 or len(phone) != 1:
        raise ValueError("missing_approved_aligned_target")
    if ("alignment_status" in word and str(word.iloc[0].alignment_status) != "ALIGNED") or (
            "alignment_status" in phone and str(phone.iloc[0].alignment_status) != "ALIGNED"):
        raise ValueError("invalid_aligned_target_status")
    return phone.iloc[0]


def _measure_target(target: dict, annotation: pd.Series, phone: pd.Series,
                    audio: np.ndarray, sample_rate: int,
                    analysis_start: float, analysis_end: float,
                    exclusions: pd.DataFrame) -> tuple[float, float, float]:
    feature = target["feature_id"]
    if feature == "wideband_noise_energy_0_10khz":
        if sample_rate / 2 <= 10000:
            raise ValueError("insufficient_bandwidth")
        if not {"noise_start_sec", "noise_end_sec"} <= set(annotation.index):
            raise ValueError("missing_validated_noise_interval")
        start, end = float(annotation.noise_start_sec), float(annotation.noise_end_sec)
        if not float(phone.start_sec) <= start < end <= float(phone.end_sec):
            raise ValueError("noise_interval_not_within_aligned_phone")
    else:
        if "burst_start_sec" not in annotation.index or annotation.burst_start_sec == "":
            raise ValueError("missing_validated_burst_start")
        start = float(annotation.burst_start_sec)
        if not float(phone.start_sec) <= start <= float(phone.end_sec):
            raise ValueError("burst_not_within_aligned_phone")
        duration = .020 if feature == "m1_t_minus_k_hz" else .010
        end = start + duration
    if not analysis_start <= start < end <= analysis_end:
        raise ValueError("sub_event_outside_reviewed_analysis")
    if not exclusions.empty and ((exclusions.start_sec.astype(float) < end)
                                 & (exclusions.end_sec.astype(float) > start)).any():
        raise ValueError("sub_event_overlaps_manual_exclusion")
    if feature == "wideband_noise_energy_0_10khz":
        value = wideband_noise_energy(audio, sample_rate, start, end)
    elif feature == "m1_t_minus_k_hz":
        value = first_spectral_moment(audio, sample_rate, start)
    else:
        value = stop_burst_tilt(audio, sample_rate, start)
    return float(value), start, end


def run_family06_stage(decisions_csv: str | Path, root: str | Path, config,
                       intervals_csv: str | Path | None, registry: pd.DataFrame,
                       progress_callback=None, execution_stage_dir: Path | None = None) -> StageResult:
    root = Path(root)
    final = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    if (Path(decisions_csv).resolve() != decisions.resolve() or intervals_csv is None
            or Path(intervals_csv).resolve() != intervals.resolve()):
        raise ValueError("Family 06 requires frozen authoritative reviewed segmentation")
    kept = load_final_segmentation(intervals, decisions)
    timeline = pd.read_csv(intervals, keep_default_na=False)
    project = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    task_id = str(project.get("task_id", ""))
    selected = set(config.selected_features) & IMPLEMENTED_IDS
    alignment, alignment_issue = load_final_alignment(root)
    targets, target_issue = load_target_manifest(
        getattr(config, "segmental_target_manifest_path", None), task_id)
    subevents, subevent_issue = load_validated_subevents(
        getattr(config, "acoustic_subevent_annotations_csv", None))
    stage = execution_stage_dir or root / "acoustic" / "005_features"
    tables = stage / "tables"
    values, statuses, audits = [], [], []
    if progress_callback:
        progress_callback(0, len(kept), "Acoustic Features — segmental contrasts")
    for done, row in enumerate(kept.itertuples(), 1):
        record_id = str(row.recording_id)
        record_timeline = timeline.loc[timeline.recording_id.astype(str).eq(record_id)]
        exclusions = record_timeline.loc[record_timeline.segment_role.eq("manual_exclusion")]
        base = {"recording_id": record_id, "file_name": str(row.file_name),
                "task_id": task_id, "task_type": str(project.get("task_type", "")),
                "segmentation_run_id": str(row.segmentation_run_id),
                "review_run_id": str(row.review_run_id),
                "alignment_run_id": alignment.manifest["alignment_run_id"] if alignment else "",
                "alignment_provider": alignment.manifest.get("provider", "") if alignment else "",
                "source_sha256": str(row.source_sha256),
                "analysis_wav_path": str(row.analysis_wav_path),
                "algorithm": ALGORITHM_VERSION, "algorithm_version": ALGORITHM_VERSION,
                "parameter_set_id": PARAMETER_SET_ID,
                "final_decisions_sha256": sha256_file(decisions),
                "final_intervals_sha256": sha256_file(intervals)}
        feature_results = {}
        audio = None
        sample_rate = 0
        for feature_id in sorted(selected):
            feature_targets = [item for item in targets if item["feature_id"] == feature_id]
            reason = alignment_issue or target_issue
            if not reason and not feature_targets:
                reason = "missing_target_definition"
            if not reason:
                reason = subevent_issue
            token_rows = []
            if not reason:
                try:
                    if audio is None:
                        audio, sample_rate = sf.read(row.analysis_wav_path, dtype="float32",
                                                     always_2d=False)
                        if audio.ndim != 1 or not np.isfinite(audio).all():
                            raise ValueError("invalid_canonical_audio")
                    for target in feature_targets:
                        annotation = subevents.loc[
                            subevents.recording_id.astype(str).eq(record_id)
                            & subevents.target_id.astype(str).eq(str(target["target_id"]))]
                        if len(annotation) != 1:
                            raise ValueError("missing_acoustic_sub_event")
                        annotation = annotation.iloc[0]
                        if (int(annotation.word_index) != int(target["word_index"])
                                or int(annotation.phone_index) != int(target["phone_index"])):
                            raise ValueError("acoustic_sub_event_target_mismatch")
                        phone = _aligned_target(alignment, record_id, target)
                        value, start, end = _measure_target(
                            target, annotation, phone, audio, int(sample_rate),
                            float(row.analysis_start_sec), float(row.analysis_end_sec), exclusions)
                        spec = registry.loc[registry.feature.eq(feature_id)].iloc[0]
                        audit = {"recording_id": record_id, "file_name": row.file_name,
                                 "task_id": task_id, "alignment_run_id": base["alignment_run_id"],
                                 "target_id": target["target_id"], "pair_id": target.get("pair_id", ""),
                                 "role": target.get("role", ""),
                                 "word_index": target["word_index"],
                                 "phone_index": target["phone_index"], "phone": target["phone"],
                                 "phone_start_sec": phone.start_sec, "phone_end_sec": phone.end_sec,
                                 "measurement_start_sec": start, "measurement_end_sec": end,
                                 "measurement_boundary_source": "validated_external_annotation",
                                 "contrast_partner": target.get("contrast_partner", ""),
                                 "feature_id": feature_id, "token_value": value, "unit": spec.unit,
                                 "validity": "valid", "failure_reason": "",
                                 "parameter_set_id": PARAMETER_SET_ID,
                                 "algorithm_version": ALGORITHM_VERSION,
                                 "annotation_source": annotation.annotation_source,
                                 "reviewer": annotation.reviewer}
                        token_rows.append(audit)
                except (OSError, ValueError, RuntimeError) as exc:
                    reason = str(exc).splitlines()[0]
            if not reason:
                if feature_id == "m1_t_minus_k_hz":
                    value, reason = matched_m1_contrast(pd.DataFrame(token_rows))
                else:
                    value = float(np.mean([item["token_value"] for item in token_rows]))
            else:
                value = np.nan
            if reason:
                value = np.nan
                audits.append({**{field: "" for field in AUDIT_COLUMNS},
                               "recording_id": record_id, "file_name": row.file_name,
                               "task_id": task_id, "alignment_run_id": base["alignment_run_id"],
                               "feature_id": feature_id, "validity": "invalid",
                               "failure_reason": reason, "parameter_set_id": PARAMETER_SET_ID,
                               "algorithm_version": ALGORITHM_VERSION})
            audits.extend(token_rows)
            feature_results[feature_id] = (value, reason)
        values.append({**base, **{key: value for key, (value, _) in feature_results.items()}})
        for feature_id, (value, reason) in feature_results.items():
            spec = registry.loc[registry.feature.eq(feature_id)].iloc[0]
            statuses.append({**base, "feature": feature_id, "feature_id": feature_id,
                             "family_id": "F06", "value": value, "unit": spec.unit,
                             "status": "unavailable" if reason else "computed",
                             "failure_reason": reason})
        if progress_callback:
            progress_callback(done, len(kept), f"Acoustic Features — {row.file_name}")
    tables.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    audit_path = tables / "native_measurements" / "family06_segmental_targets.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values).to_csv(values_path, index=False)
    pd.DataFrame(statuses).to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    pd.DataFrame(audits).reindex(columns=AUDIT_COLUMNS).to_csv(audit_path, index=False)
    if execution_stage_dir is None:
        handoff = write_feature_handoff(tables, pd.DataFrame(values), registry,
                                        pd.DataFrame(statuses), "acoustic", values_path, registry_path)
        build_feature_delivery(root, "acoustic", handoff)
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version=ALGORITHM_VERSION,
        status="completed_with_warnings" if any(item["failure_reason"] for item in statuses)
        else "completed",
        input_artifacts=[ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
                         ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")],
        output_artifacts=[ArtifactRef(str(values_path), "feature_values", "text/csv"),
                          ArtifactRef(str(status_path), "feature_status", "text/csv"),
                          ArtifactRef(str(audit_path), "segmental_target_audit", "text/csv")],
        config={"selected_features": sorted(selected), "parameter_set_id": PARAMETER_SET_ID,
                "target_manifest_sha256": (sha256_file(config.segmental_target_manifest_path)
                                           if getattr(config, "segmental_target_manifest_path", None)
                                           and Path(config.segmental_target_manifest_path).is_file() else ""),
                "subevent_annotations_sha256": (sha256_file(config.acoustic_subevent_annotations_csv)
                                                if getattr(config, "acoustic_subevent_annotations_csv", None)
                                                and Path(config.acoustic_subevent_annotations_csv).is_file() else ""),
                "alignment_run_id": alignment.manifest["alignment_run_id"] if alignment else "",
                "native_sample_rate_policy": "canonical native rate; no upsampling",
                "fft_size_minimum": 2048, "m1_window_ms": 20, "tilt_window_ms": 10,
                "tilt_band_hz": [1500, 5000], "wideband_hz": [0, 10000],
                "power_convention": "Hann one-sided digital power / (FFT length * mean Hann-square)"})
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
