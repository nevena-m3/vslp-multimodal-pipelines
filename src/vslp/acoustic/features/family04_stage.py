"""Family 04 native token/vowel outputs and scalar /i,a,u/ features."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf

from vslp.acoustic.alignment import load_final_alignment
from vslp.acoustic.features.family04 import (
    ALGORITHM_VERSION, DERIVED_IDS, FAMILY04_IDS, FRAME_COLUMNS, TOKEN_COLUMNS,
    TOKEN_IDS, VOWEL_COLUMNS, VOWEL_IDS, aggregate_vowel_centroids,
    derive_iau_features, load_vowel_category_mapping, profile_by_id,
    track_aligned_vowels,
)
from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


def run_family04_stage(decisions_csv: str | Path, root: str | Path, config,
                       intervals_csv: str | Path | None, registry: pd.DataFrame,
                       progress_callback=None, execution_stage_dir: Path | None = None,
                       formant_service=None) -> StageResult:
    """Consume only frozen reviewed segmentation and frozen Alignment tokens."""
    root = Path(root)
    review = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = review / "final_segmentation_decisions.csv"
    intervals = review / "final_segmentation_intervals.csv"
    if (Path(decisions_csv).resolve() != decisions.resolve() or intervals_csv is None
            or Path(intervals_csv).resolve() != intervals.resolve()):
        raise ValueError("Family 04 requires authoritative frozen reviewed segmentation")
    kept = load_final_segmentation(intervals, decisions)
    selected = set(config.selected_features) & FAMILY04_IDS
    run = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    task_id = str(run.get("task_id", ""))
    profile = profile_by_id(getattr(config, "formant_profile_id", None)
                            or "family04_burg_native_5500_v1")
    alignment, alignment_issue = load_final_alignment(root)
    mapping, mapping_issue, mapping_hash = load_vowel_category_mapping(
        getattr(config, "vowel_category_manifest_path", None), task_id)
    alignment_run_id = alignment.manifest["alignment_run_id"] if alignment else ""
    stage = execution_stage_dir or root / "acoustic" / "005_features"
    tables = stage / "tables"
    native = tables / "native_measurements"
    values_rows: list[dict] = []
    status_rows: list[dict] = []
    token_tables: list[pd.DataFrame] = []
    vowel_tables: list[pd.DataFrame] = []
    frame_paths: list[Path] = []
    if progress_callback:
        progress_callback(0, len(kept), "Acoustic Features — Family 04")
    for done, row in enumerate(kept.itertuples(), 1):
        record_id = str(row.recording_id)
        token_table = pd.DataFrame(columns=TOKEN_COLUMNS)
        vowel_table = pd.DataFrame(columns=VOWEL_COLUMNS)
        frame_table = pd.DataFrame(columns=FRAME_COLUMNS)
        audio_issue = alignment_issue
        sample_rate = np.nan
        if not audio_issue:
            phone_tokens = alignment.get_vowel_tokens(record_id)
            if phone_tokens.empty:
                audio_issue = "missing_aligned_vowel_tokens"
            else:
                try:
                    audio, rate = sf.read(row.analysis_wav_path, dtype="float32",
                                          always_2d=False)
                    sample_rate = int(rate)
                    def compute_track():
                        return track_aligned_vowels(
                            audio, int(rate), phone_tokens, profile,
                            recording_id=record_id, file_name=str(row.file_name),
                            task_id=task_id, task_type=str(run.get("task_type", "")),
                            alignment_run_id=alignment_run_id, category_mapping=mapping)

                    token_table, frame_table = (
                        formant_service.get_or_compute(record_id, alignment_run_id,
                                                       profile.profile_id, compute_track)
                        if formant_service is not None else compute_track())
                    vowel_table = aggregate_vowel_centroids(token_table)
                except (OSError, ValueError, RuntimeError) as exc:
                    audio_issue = str(exc).splitlines()[0]
        if not token_table.empty:
            token_tables.append(token_table)
        if not vowel_table.empty:
            vowel_tables.append(vowel_table)
        frame_path = ""
        if not frame_table.empty:
            frames_dir = native / "formant_frame_tracks"
            frames_dir.mkdir(parents=True, exist_ok=True)
            stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(str(row.file_name)).stem)
            frame_file = frames_dir / f"{stem}__{record_id[:8]}__{profile.profile_id}.npz"
            np.savez_compressed(frame_file,
                                **{field: frame_table[field].to_numpy() for field in FRAME_COLUMNS},
                                native_sample_rate_hz=sample_rate,
                                alignment_run_id=alignment_run_id,
                                formant_profile_id=profile.profile_id)
            frame_paths.append(frame_file)
            frame_path = str(frame_file)
        derived = derive_iau_features(vowel_table, mapping is not None)
        base = {
            "recording_id": record_id, "file_name": row.file_name,
            "task_id": task_id, "task_type": run.get("task_type", ""),
            "task_name": run.get("task_name", ""), "run_id": run.get("run_id", ""),
            "source_sha256": row.source_sha256,
            "source_file_path": row.source_file_path,
            "analysis_wav_path": row.analysis_wav_path,
            "segmentation_run_id": row.segmentation_run_id,
            "review_run_id": row.review_run_id,
            "alignment_run_id": alignment_run_id,
            "alignment_provider": alignment.manifest.get("provider", "") if alignment else "",
            "alignment_words_sha256": alignment.manifest.get("final_words_sha256", "") if alignment else "",
            "alignment_phones_sha256": alignment.manifest.get("final_phones_sha256", "") if alignment else "",
            "vowel_category_manifest_sha256": mapping_hash,
            "formant_profile_id": profile.profile_id,
            "algorithm_version": ALGORITHM_VERSION,
            "praat_version": parselmouth.PRAAT_VERSION,
            "parselmouth_version": parselmouth.__version__,
            "native_sample_rate_hz": sample_rate,
            "n_aligned_vowel_tokens": len(token_table),
            "n_valid_formant_tokens": int(token_table.tracking_status.eq("OK").sum()),
            "n_valid_formant_frames": int(token_table.n_valid_frames.sum()) if not token_table.empty else 0,
            "formant_frame_track_path": frame_path,
            "final_decisions_sha256": sha256_file(decisions),
            "final_intervals_sha256": sha256_file(intervals),
        }
        scalar_values = {feature: derived[feature][0] if not audio_issue else np.nan
                         for feature in selected & DERIVED_IDS}
        values_rows.append({**base, **scalar_values})
        for feature in sorted(selected):
            spec = registry.loc[registry.feature.eq(feature)].iloc[0]
            if audio_issue:
                value, reason, state = np.nan, audio_issue, "unavailable"
            elif feature in TOKEN_IDS:
                count = int(token_table.tracking_status.eq("OK").sum())
                value = np.nan
                reason = "" if count else "insufficient_valid_formant_tokens"
                state = "computed_native" if count else "unavailable"
            elif feature in VOWEL_IDS:
                count = int(vowel_table.n_valid_tokens.gt(0).sum()) if not vowel_table.empty else 0
                reason = mapping_issue or ("" if count else "missing_required_vowel")
                value = np.nan
                state = "computed_native" if not reason else "unavailable"
            else:
                value, reason = derived[feature]
                state = "computed" if not reason else "unavailable"
            status_rows.append({**base, "feature": feature, "feature_id": feature,
                                "family_id": "F04", "value": value,
                                "unit": spec.unit, "row_granularity": spec.output_granularity,
                                "status": state, "failure_reason": reason,
                                "native_measurement_table": (
                                    "formant_token_measurements.csv" if feature in TOKEN_IDS else
                                    "formant_vowel_measurements.csv" if feature in VOWEL_IDS else ""),
                                "n_valid_tokens": base["n_valid_formant_tokens"]})
        if progress_callback:
            progress_callback(done, len(kept), f"Acoustic Features — {row.file_name}")
    tables.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    values_df, status_df = pd.DataFrame(values_rows), pd.DataFrame(status_rows)
    values_df.to_csv(values_path, index=False)
    status_df.to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    optional_main = {}
    token_path = native / "formant_token_measurements.csv"
    vowel_path = native / "formant_vowel_measurements.csv"
    if token_tables:
        token_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(token_tables, ignore_index=True).reindex(columns=TOKEN_COLUMNS).to_csv(token_path, index=False)
        optional_main[token_path.name] = token_path
    if vowel_tables:
        vowel_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(vowel_tables, ignore_index=True).reindex(columns=VOWEL_COLUMNS).to_csv(vowel_path, index=False)
        optional_main[vowel_path.name] = vowel_path
    if execution_stage_dir is None:
        handoff = write_feature_handoff(tables, values_df, registry, status_df,
                                        "acoustic", values_path, registry_path)
        build_feature_delivery(root, "acoustic", handoff, optional_main=optional_main)
    outputs = [ArtifactRef(str(values_path), "feature_values", "text/csv"),
               ArtifactRef(str(status_path), "feature_status", "text/csv")]
    if token_path.is_file():
        outputs.append(ArtifactRef(str(token_path), "formant_token_measurements", "text/csv"))
    if vowel_path.is_file():
        outputs.append(ArtifactRef(str(vowel_path), "formant_vowel_measurements", "text/csv"))
    outputs.extend(ArtifactRef(str(path), "formant_frame_track", "application/npz")
                   for path in frame_paths)
    inputs = [ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
              ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")]
    if alignment is not None:
        inputs.extend([ArtifactRef(alignment.manifest["final_words_path"],
                                   "final_alignment_words", "text/csv"),
                       ArtifactRef(alignment.manifest["final_phones_path"],
                                   "final_alignment_phones", "text/csv")])
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version=ALGORITHM_VERSION,
        status="completed_with_warnings" if any(item["failure_reason"] for item in status_rows)
        else "completed", input_artifacts=inputs, output_artifacts=outputs,
        config={"selected_features": sorted(selected),
                "formant_profile_id": profile.profile_id,
                "native_rate_policy": "canonical native-rate read-only",
                "vowel_category_manifest_sha256": mapping_hash,
                "alignment_run_id": alignment_run_id,
                "token_and_vowel_ids_are_native_long_form": True})
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
