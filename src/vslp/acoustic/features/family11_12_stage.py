"""Reviewed-audio execution for Families 11 and 12."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.features.family11 import FAMILY11_IDS, calculate_amplitude_features
from vslp.acoustic.features.family12 import FAMILY12_IDS, calculate_spectral_features
from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


def _bool(value) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}


def _analysis_audio(row, intervals: pd.DataFrame, audio: np.ndarray, sr: int):
    if str(row.segmentation_method) == "sustained_phonation":
        start, end = float(row.stable_region_start), float(row.stable_region_end)
        if not 0 <= start < end <= len(audio) / sr:
            raise ValueError("stable_phonation_region_unavailable")
        return audio[round(start*sr):round(end*sr)].copy(), start, end, "final_reviewed_stable_phonation", 1
    speech = intervals.loc[intervals.segment_role.eq("speech")].sort_values("start_sec")
    spans = [(float(item.start_sec), float(item.end_sec)) for item in speech.itertuples()
             if float(item.end_sec) > float(item.start_sec)]
    chunks = [audio[round(start*sr):round(end*sr)] for start, end in spans]
    chunks = [chunk for chunk in chunks if chunk.size]
    if not chunks:
        raise ValueError("final_reviewed_speech_region_unavailable")
    return np.concatenate(chunks), spans[0][0], spans[-1][1], "final_reviewed_patient_speech", len(chunks)


def run_family11_12_stage(decisions_csv: str | Path, root: str | Path, config,
                          intervals_csv: str | Path | None, registry: pd.DataFrame,
                          progress_callback=None, execution_stage_dir: Path | None = None) -> StageResult:
    root = Path(root)
    final = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    if (Path(decisions_csv).resolve() != decisions.resolve() or intervals_csv is None
            or Path(intervals_csv).resolve() != intervals.resolve()):
        raise ValueError("Families 11/12 require authoritative frozen reviewed segmentation")
    kept = load_final_segmentation(intervals, decisions)
    all_intervals = pd.read_csv(intervals, keep_default_na=False)
    run = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    selected = set(config.selected_features) & (FAMILY11_IDS | FAMILY12_IDS)
    selected11, selected12 = selected & FAMILY11_IDS, selected & FAMILY12_IDS
    preprocess_path = root / "acoustic" / "001_preprocess" / "tables" / "acoustic_preprocess_summary.csv"
    preprocess = pd.read_csv(preprocess_path, keep_default_na=False) if preprocess_path.is_file() else pd.DataFrame()
    values, statuses, audits = [], [], []
    matrix_artifacts = []
    if progress_callback:
        progress_callback(0, len(kept), "Acoustic Features")
    for index, row in enumerate(kept.itertuples(), 1):
        recording_id = str(row.recording_id)
        timeline = all_intervals.loc[all_intervals.recording_id.astype(str).eq(recording_id)]
        result = {feature_id: (np.nan, "analysis_unavailable") for feature_id in selected}
        prep = preprocess.loc[preprocess.recording_id.astype(str).eq(recording_id)] if not preprocess.empty else pd.DataFrame()
        normalized = _bool(prep.iloc[-1].get("amplitude_normalization", False)) if not prep.empty else False
        dc_removed = _bool(prep.iloc[-1].get("remove_dc_offset", False)) if not prep.empty else False
        metadata = {"native_sample_rate_hz": np.nan, "working_sample_rate_hz": np.nan,
                    "n_frames": 0, "n_samples": 0, "clipping_fraction": np.nan,
                    "analysis_start_sec": np.nan, "analysis_end_sec": np.nan,
                    "analysis_region": "unavailable", "n_source_spans": 0,
                    "mfcc_matrix_path": ""}
        try:
            canonical, sr = sf.read(row.analysis_wav_path, dtype="float32", always_2d=False)
            if canonical.ndim != 1:
                raise ValueError("canonical_audio_not_mono")
            analysis, start, end, region, span_count = _analysis_audio(row, timeline, canonical, int(sr))
            metadata.update({"native_sample_rate_hz": int(sr), "analysis_start_sec": start,
                             "analysis_end_sec": end, "analysis_region": region,
                             "n_source_spans": span_count, "n_samples": len(analysis)})
            if selected11:
                try:
                    family11, audit11 = calculate_amplitude_features(
                        analysis, int(sr), amplitude_normalized=normalized)
                    result.update({key: value for key, value in family11.items() if key in selected11})
                    metadata["clipping_fraction"] = audit11["clipping_fraction"]
                except (ValueError, RuntimeError) as exc:
                    result.update({key: (np.nan, str(exc).splitlines()[0]) for key in selected11})
            if selected12:
                try:
                    family12, audit12 = calculate_spectral_features(analysis, int(sr))
                    result.update({key: value for key, value in family12.items() if key in selected12})
                    metadata.update({"working_sample_rate_hz": audit12.working_sample_rate_hz,
                                     "n_frames": audit12.n_frames})
                    if audit12.mfcc_matrix.size and any(key.startswith("mfcc") for key in selected12):
                        folder = (execution_stage_dir or root / "acoustic" / "005_features") / "tables" / "native_measurements" / "mfcc_matrices"
                        folder.mkdir(parents=True, exist_ok=True)
                        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(str(row.file_name)).stem)
                        matrix_path = folder / f"{stem}__{recording_id[:8]}__mfcc.npz"
                        np.savez_compressed(matrix_path, mfcc=audit12.mfcc_matrix,
                                            working_sample_rate_hz=audit12.working_sample_rate_hz,
                                            frame_hop_sec=0.010)
                        metadata["mfcc_matrix_path"] = str(matrix_path)
                        matrix_artifacts.append(matrix_path)
                except (ValueError, RuntimeError) as exc:
                    result.update({key: (np.nan, str(exc).splitlines()[0]) for key in selected12})
        except (OSError, ValueError, RuntimeError) as exc:
            reason = str(exc).splitlines()[0]
            result = {feature_id: (np.nan, reason) for feature_id in selected}
        base = {"recording_id": recording_id, "file_name": row.file_name,
                "task_id": run.get("task_id", ""), "task_type": run.get("task_type", ""),
                "task_name": run["task_name"], "run_id": run["run_id"],
                "source_sha256": row.source_sha256, "source_file_path": row.source_file_path,
                "analysis_wav_path": row.analysis_wav_path,
                "segmentation_run_id": row.segmentation_run_id, "review_run_id": row.review_run_id,
                "dc_offset_removal_applied": dc_removed,
                "amplitude_normalization_applied": normalized,
                "final_decisions_sha256": sha256_file(decisions),
                "final_intervals_sha256": sha256_file(intervals), **metadata}
        values.append({**base, **{feature_id: result[feature_id][0] for feature_id in selected}})
        for feature_id in sorted(selected):
            value, reason = result[feature_id]
            spec = registry.loc[registry.feature.eq(feature_id)].iloc[0]
            statuses.append({**base, "feature": feature_id, "feature_id": feature_id,
                             "family_id": spec.family_id, "value": value, "unit": spec.unit,
                             "algorithm": spec.algorithm_version,
                             "algorithm_version": spec.algorithm_version,
                             "parameter_set_id": spec.parameter_set_id,
                             "status": "unavailable" if reason else "computed",
                             "failure_reason": reason})
        audits.append(base)
        if progress_callback:
            progress_callback(index, len(kept), f"Acoustic Features — {row.file_name}")
    stage = execution_stage_dir or root / "acoustic" / "005_features"
    tables = stage / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    audit_path = tables / "native_measurements" / "family11_12_waveform_spectral_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    values_df, statuses_df = pd.DataFrame(values), pd.DataFrame(statuses)
    values_df.to_csv(values_path, index=False)
    statuses_df.to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    pd.DataFrame(audits).to_csv(audit_path, index=False)
    if execution_stage_dir is None:
        handoff = write_feature_handoff(tables, values_df, registry, statuses_df,
                                        "acoustic", values_path, registry_path)
        build_feature_delivery(root, "acoustic", handoff)
    artifacts = [ArtifactRef(str(values_path), "feature_values", "text/csv"),
                 ArtifactRef(str(status_path), "feature_status", "text/csv"),
                 ArtifactRef(str(audit_path), "waveform_spectral_audit", "text/csv")]
    artifacts.extend(ArtifactRef(str(path), "mfcc_frame_matrix", "application/npz")
                     for path in matrix_artifacts)
    manifest = StageManifest(stage_name="acoustic_features",
        stage_version="family11-12-1.0.0",
        status="completed_with_warnings" if any(row["failure_reason"] for row in statuses) else "completed",
        input_artifacts=[ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
                         ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")],
        output_artifacts=artifacts,
        config={"selected_features": sorted(selected),
                "source_audio_policy": "canonical native rate read-only; private 48 kHz spectral copy",
                "preprocessing_provenance_source": str(preprocess_path) if preprocess_path.is_file() else "ingest"})
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
