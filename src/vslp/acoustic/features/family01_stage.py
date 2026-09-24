"""Reviewed native-rate Family 01 extraction and track audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.features.family01 import FAMILY01_IDS, calculate_f0_features, praat_f0_track
from vslp.acoustic.features.family02 import FAMILY02_IDS, calculate_voice_quality
from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


def _region_tracks(row, intervals: pd.DataFrame, audio: np.ndarray, sr: int):
    sustained = str(row.segmentation_method) == "sustained_phonation"
    if sustained:
        start, end = float(row.stable_region_start), float(row.stable_region_end)
        if not 0 <= start < end <= len(audio) / sr:
            raise ValueError("stable_phonation_region_unavailable")
        spans = [(start, end)]
        region = "final_reviewed_stable_phonation"
    else:
        speech = intervals.loc[intervals.segment_role.eq("speech")].sort_values("start_sec")
        spans = [(float(item.start_sec), float(item.end_sec)) for item in speech.itertuples()]
        region = "final_reviewed_patient_speech"
    if not spans:
        raise ValueError("final_reviewed_speech_region_unavailable")
    tracks = []
    for start, end in spans:
        segment = audio[round(start * sr):round(end * sr)]
        if segment.size < int(sr * .1):
            continue
        tracks.append(praat_f0_track(segment, sr, analysis_start_sec=start,
                                     analysis_region=region))
    if not tracks:
        raise ValueError("analysis_region_too_short")
    from vslp.acoustic.features.family01 import F0Track
    track = F0Track(np.concatenate([t.time_sec for t in tracks]),
                    np.concatenate([t.f0_hz for t in tracks]),
                    np.concatenate([t.voiced_valid for t in tracks]), sr,
                    spans[0][0], spans[-1][1], region)
    return track, sustained


def run_family01_stage(decisions_csv: str | Path, root: str | Path, config,
                       intervals_csv: str | Path | None, registry: pd.DataFrame,
                       progress_callback=None, execution_stage_dir: Path | None = None) -> StageResult:
    root = Path(root)
    final = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    if (Path(decisions_csv).resolve() != decisions.resolve() or intervals_csv is None
            or Path(intervals_csv).resolve() != intervals.resolve()):
        raise ValueError("Family 01 requires authoritative frozen reviewed segmentation")
    kept = load_final_segmentation(intervals, decisions)
    all_intervals = pd.read_csv(intervals, keep_default_na=False)
    run = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    selected = set(config.selected_features) & (FAMILY01_IDS | FAMILY02_IDS)
    selected01 = selected & FAMILY01_IDS
    selected02 = selected & FAMILY02_IDS
    stage = execution_stage_dir or root / "acoustic" / "005_features"
    values, statuses, track_rows, pulse_rows = [], [], [], []
    if progress_callback:
        progress_callback(0, len(kept), "Acoustic Features")
    for index, row in enumerate(kept.itertuples(), 1):
        record_id = str(row.recording_id)
        timeline = all_intervals.loc[all_intervals.recording_id.astype(str).eq(record_id)]
        result = {feature_id: (np.nan, "analysis_unavailable") for feature_id in selected}
        track = None
        used_region = "unavailable"
        used_start = np.nan
        used_end = np.nan
        cycle_audit = {"n_cycles": 0, "n_pulses": 0, "hnr_valid_frames": 0}
        try:
            audio, sr = sf.read(row.analysis_wav_path, dtype="float32", always_2d=False)
            if audio.ndim != 1:
                raise ValueError("canonical_audio_not_mono")
            if selected01:
                try:
                    track, sustained = _region_tracks(row, timeline, audio, int(sr))
                    used_region, used_start, used_end = (track.analysis_region,
                                                         track.analysis_start_sec,
                                                         track.analysis_end_sec)
                    result.update({k: v for k, v in calculate_f0_features(track, sustained=sustained).items()
                                   if k in selected01})
                    track_rows.extend({
                        "recording_id": record_id, "file_name": row.file_name,
                        "time_sec": float(t), "f0_hz": float(hz), "voiced_valid": bool(valid),
                        "analysis_region": track.analysis_region, "algorithm_version": track.algorithm_version,
                        "parameter_set_id": track.parameter_set_id, "sample_rate_hz": int(sr),
                    } for t, hz, valid in zip(track.time_sec, track.f0_hz, track.voiced_valid, strict=True))
                except (ValueError, RuntimeError) as exc:
                    result.update({k: (np.nan, str(exc).splitlines()[0]) for k in selected01})
            if selected02:
                try:
                    if str(row.segmentation_method) != "sustained_phonation":
                        raise ValueError("stable_phonation_region_unavailable")
                    start, end = float(row.stable_region_start), float(row.stable_region_end)
                    if not 0 <= start < end <= len(audio) / sr:
                        raise ValueError("stable_phonation_region_unavailable")
                    region_audio = audio[round(start * sr):round(end * sr)]
                    used_region, used_start, used_end = (
                        "final_reviewed_stable_phonation", start, end)
                    voice_values, cycle_audit = calculate_voice_quality(region_audio, int(sr))
                    result.update({k: v for k, v in voice_values.items() if k in selected02})
                    times = cycle_audit.get("pulse_time_sec", np.array([]))
                    periods = cycle_audit.get("period_sec", np.array([]))
                    pulse_rows.extend({"recording_id": record_id, "file_name": row.file_name,
                                       "pulse_index": i + 1, "time_sec": float(start + t),
                                       "following_period_sec": float(periods[i]) if i < len(periods) else np.nan,
                                       "analysis_region": "final_reviewed_stable_phonation",
                                       "algorithm_version": "family02-praat-1.0.0",
                                       "parameter_set_id": "family02_stable_vowel_praat_v1"}
                                      for i, t in enumerate(times))
                except (ValueError, RuntimeError) as exc:
                    result.update({k: (np.nan, str(exc)) for k in selected02})
        except (OSError, ValueError, RuntimeError) as exc:
            reason = str(exc).splitlines()[0]
            result = {feature_id: (np.nan, reason) for feature_id in selected}
        base = {
            "recording_id": record_id, "file_name": row.file_name,
            "task_id": run.get("task_id", ""), "task_type": run.get("task_type", ""),
            "task_name": run["task_name"], "run_id": run["run_id"],
            "source_sha256": row.source_sha256, "source_file_path": row.source_file_path,
            "analysis_wav_path": row.analysis_wav_path,
            "segmentation_run_id": row.segmentation_run_id, "review_run_id": row.review_run_id,
            "analysis_region": used_region,
            "analysis_start_sec": used_start,
            "analysis_end_sec": used_end,
            "n_valid_frames": int(track.voiced_valid.sum()) if track else 0,
            "tracking_yield": track.tracking_yield if track else np.nan,
            "n_cycles": cycle_audit["n_cycles"],
            "n_pulses": cycle_audit["n_pulses"],
            "hnr_valid_frames": cycle_audit["hnr_valid_frames"],
            "final_decisions_sha256": sha256_file(decisions),
            "final_intervals_sha256": sha256_file(intervals),
        }
        values.append({**base, **{feature_id: result[feature_id][0] for feature_id in selected}})
        for feature_id in sorted(selected):
            value, reason = result[feature_id]
            spec = registry.loc[registry.feature.eq(feature_id)].iloc[0]
            statuses.append({**base, "feature": feature_id, "feature_id": feature_id,
                             "family_id": spec.family_id, "value": value, "unit": spec.unit,
                             "algorithm": spec.algorithm_version, "algorithm_version": spec.algorithm_version,
                             "parameter_set_id": spec.parameter_set_id,
                             "status": "unavailable" if reason else "computed",
                             "failure_reason": reason})
        if progress_callback:
            progress_callback(index, len(kept), f"Acoustic Features — {row.file_name}")
    tables = stage / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    track_path = tables / "native_measurements" / "f0_tracks.csv" if selected01 else None
    pulse_path = tables / "native_measurements" / "voice_pulses.csv" if selected02 else None
    pd.DataFrame(values).to_csv(values_path, index=False)
    pd.DataFrame(statuses).to_csv(status_path, index=False)
    if track_path:
        track_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(track_rows).to_csv(track_path, index=False)
    if pulse_path:
        pulse_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pulse_rows).to_csv(pulse_path, index=False)
    registry.to_csv(registry_path, index=False)
    if execution_stage_dir is None:
        handoff = write_feature_handoff(tables, pd.DataFrame(values), registry,
                                        pd.DataFrame(statuses), "acoustic", values_path, registry_path)
        build_feature_delivery(root, "acoustic", handoff)
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version="reviewed-voice-1.0.0",
        status="completed_with_warnings" if any(s["failure_reason"] for s in statuses) else "completed",
        input_artifacts=[ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
                         ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")],
        output_artifacts=[ArtifactRef(str(values_path), "feature_values", "text/csv"),
                          ArtifactRef(str(status_path), "feature_status", "text/csv"),
                          *([ArtifactRef(str(track_path), "shared_f0_track", "text/csv")] if track_path else []),
                          *([ArtifactRef(str(pulse_path), "praat_pulses", "text/csv")] if pulse_path else [])],
        config={"selected_features": sorted(selected),
                "parameter_set_ids": sorted({s["parameter_set_id"] for s in statuses}),
                "source_audio_policy": "canonical native rate; read only"})
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
