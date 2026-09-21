"""Family 08: source-defined speaking and articulation rates.

Only frozen reviewed intervals and a versioned prompt-count manifest are read.
No acoustic waveform or segmentation algorithm is run here.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult

FAMILY08_IDS = frozenset({
    "speaking_rate_syll_s",
    "speaking_rate_words_min",
    "articulation_rate_syll_s",
})
ALGORITHM_VERSION = "family08-rate-1.0.0"
PARAMETER_SET_ID = "family08_bamboo_reviewed_v1"
MIN_PAUSE_SEC = 0.300
MANUAL_EXCLUSION_POLICY = "subtract"


def load_prompt_counts(path: str | Path | None, task_name: str) -> tuple[dict[str, Any] | None, str]:
    """Require an explicit fixed-prompt count and version; never guess text counts."""
    if not path or not Path(path).is_file():
        return None, "prompt_manifest_missing"
    try:
        item = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "prompt_manifest_unreadable"
    if item.get("schema_version") != "1" or item.get("task_id") != "bamboo_passage":
        return None, "prompt_manifest_schema_or_task_mismatch"
    if task_name != "Bamboo Passage":
        return None, "task_not_supported_by_family08_specification"
    if not item.get("prompt_version") or not item.get("count_source"):
        return None, "prompt_version_or_count_source_missing"
    if item.get("applies_to_all_recordings") is not True:
        return None, "prompt_applicability_not_confirmed"
    for key in ("syllable_count", "word_count"):
        value = item.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            return None, f"{key}_must_be_positive_integer"
    return item, ""


def compute_family08_timing(intervals: pd.DataFrame, *,
                            manual_exclusion_policy: str = MANUAL_EXCLUSION_POLICY) -> tuple[dict[str, float], str]:
    """Measure reviewed task time and >=300 ms internal pauses at exact boundaries."""
    required = {"start_sec", "end_sec", "segment_role", "view"}
    if not required <= set(intervals.columns):
        return {}, "final_interval_schema_missing"
    timeline = intervals.loc[intervals["view"].eq("authoritative")].copy()
    speech = timeline.loc[timeline["segment_role"].eq("speech")]
    if speech.empty:
        return {}, "no_final_patient_speech"
    start = float(pd.to_numeric(speech["start_sec"], errors="coerce").min())
    end = float(pd.to_numeric(speech["end_sec"], errors="coerce").max())
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return {}, "invalid_utterance_boundaries"
    elapsed = end - start
    internal = timeline.loc[timeline["segment_role"].eq("internal_nonspeech")].copy()
    pause_durations = []
    for row in internal.itertuples():
        duration = max(0.0, min(end, float(row.end_sec)) - max(start, float(row.start_sec)))
        if duration >= MIN_PAUSE_SEC - 1e-9:
            pause_durations.append(duration)
    pause_time = float(sum(pause_durations))
    excluded = timeline.loc[timeline["segment_role"].eq("manual_exclusion")].copy()
    excluded_time = sum(max(0.0, min(end, float(row.end_sec)) - max(start, float(row.start_sec)))
                        for row in excluded.itertuples())
    if excluded_time > 0:
        if manual_exclusion_policy == "nan":
            return {}, "internal_manual_exclusion_requires_denominator_policy"
        if manual_exclusion_policy == "subtract":
            elapsed -= excluded_time
        elif manual_exclusion_policy != "include":
            return {}, "invalid_manual_exclusion_policy"
    speech_time = elapsed - pause_time
    if elapsed <= 0 or speech_time <= 0:
        return {}, "nonpositive_elapsed_or_speech_time"
    return {
        "utterance_start_sec": start,
        "utterance_end_sec": end,
        "elapsed_task_sec": elapsed,
        "internal_pauses_ge_300ms_sec": pause_time,
        "manual_exclusion_sec": float(excluded_time),
        "articulation_speech_sec": speech_time,
    }, ""


def calculate_family08(timing: dict[str, float], prompt: dict[str, Any] | None,
                       selected: set[str]) -> dict[str, tuple[float, str]]:
    """Return a value and explicit failure reason for every requested exact ID."""
    results = {}
    for feature_id in sorted(selected):
        if not timing:
            results[feature_id] = (np.nan, "timing_unavailable")
            continue
        if prompt is None:
            results[feature_id] = (np.nan, "prompt_manifest_missing_or_invalid")
            continue
        count_key = "word_count" if feature_id == "speaking_rate_words_min" else "syllable_count"
        count = prompt.get(count_key)
        if count is None:
            results[feature_id] = (np.nan, f"{count_key}_missing")
            continue
        divisor = (timing["articulation_speech_sec"] if feature_id == "articulation_rate_syll_s"
                   else timing["elapsed_task_sec"])
        scale = 60.0 if feature_id == "speaking_rate_words_min" else 1.0
        results[feature_id] = (scale * float(count) / divisor, "")
    return results


def run_family08_stage(segmentation_summary_csv: str | Path, output_root: str | Path,
                       config: Any, final_intervals_csv: str | Path | None,
                       registry: pd.DataFrame, progress_callback=None) -> StageResult:
    root = Path(output_root)
    final_dir = root / "acoustic" / "003_segmentation_review" / "final"
    decisions_path = final_dir / "final_segmentation_decisions.csv"
    intervals_path = final_dir / "final_segmentation_intervals.csv"
    if (Path(segmentation_summary_csv).resolve() != decisions_path.resolve()
            or final_intervals_csv is None
            or Path(final_intervals_csv).resolve() != intervals_path.resolve()
            or not decisions_path.is_file() or not intervals_path.is_file()):
        raise ValueError("Family 08 requires frozen authoritative reviewed segmentation")
    run = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    task_name = str(run.get("task_name", ""))
    stage = root / "acoustic" / "005_features"
    prompt_snapshot = None
    if config.prompt_manifest_path and Path(config.prompt_manifest_path).is_file():
        prompt_snapshot = stage / "configs" / "prompt_count_manifest.json"
        prompt_snapshot.parent.mkdir(parents=True, exist_ok=True)
        if Path(config.prompt_manifest_path).resolve() != prompt_snapshot.resolve():
            shutil.copy2(config.prompt_manifest_path, prompt_snapshot)
    prompt, prompt_issue = load_prompt_counts(prompt_snapshot, task_name)
    prompt_hash = sha256_file(prompt_snapshot) if prompt_snapshot else ""
    kept = load_final_segmentation(intervals_path, decisions_path)
    if kept.empty:
        raise ValueError("No kept recordings are available in frozen reviewed segmentation")
    all_intervals = pd.read_csv(intervals_path, keep_default_na=False)
    decisions_hash = sha256_file(decisions_path)
    intervals_hash = sha256_file(intervals_path)
    selected = set(config.selected_features)
    catalog = {row.feature: row for row in registry.itertuples()}
    values_rows = []
    status_rows = []
    if progress_callback:
        progress_callback(0, len(kept), "Acoustic Features")
    for index, row in enumerate(kept.itertuples(), start=1):
        record_id = str(row.recording_id)
        timeline = all_intervals.loc[all_intervals.recording_id.astype(str).eq(record_id)]
        timing, timing_issue = compute_family08_timing(timeline)
        results = calculate_family08(timing, prompt, selected)
        value_row = {
            "recording_id": record_id, "file_name": row.file_name,
            "project_name": run["project_name"], "task_name": task_name,
            "run_id": run["run_id"], "source_sha256": row.source_sha256,
            "source_file_path": row.source_file_path,
            "segmentation_run_id": row.segmentation_run_id,
            "review_run_id": row.review_run_id,
            "boundary_source": row.boundary_source,
            "prompt_version": prompt.get("prompt_version", "") if prompt else "",
            "count_source": prompt.get("count_source", "") if prompt else "",
            "prompt_manifest_sha256": prompt_hash,
            "algorithm_version": ALGORITHM_VERSION,
            "parameter_set_id": PARAMETER_SET_ID,
            "analysis_region": "first_to_last_final_patient_speech",
            "pause_min_sec": MIN_PAUSE_SEC,
            "manual_exclusion_policy": MANUAL_EXCLUSION_POLICY,
            "utterance_start_sec": timing.get("utterance_start_sec", np.nan),
            "utterance_end_sec": timing.get("utterance_end_sec", np.nan),
            "elapsed_task_sec": timing.get("elapsed_task_sec", np.nan),
            "internal_pauses_ge_300ms_sec": timing.get("internal_pauses_ge_300ms_sec", np.nan),
            "articulation_speech_sec": timing.get("articulation_speech_sec", np.nan),
            "final_decisions_sha256": decisions_hash,
            "final_intervals_sha256": intervals_hash,
            **{feature_id: results[feature_id][0] for feature_id in sorted(selected)},
        }
        values_rows.append(value_row)
        for feature_id in sorted(selected):
            value, reason = results[feature_id]
            reason = timing_issue or prompt_issue or reason
            if reason and np.isfinite(value):
                value = np.nan
                value_row[feature_id] = np.nan
            qc_flag = ("above_broad_plausibility_range"
                       if np.isfinite(value) and feature_id != "speaking_rate_words_min" and value > 10
                       else "")
            status_rows.append({
                "recording_id": record_id, "file_name": row.file_name,
                "task_name": task_name, "feature": feature_id,
                "status": ("unavailable" if reason else
                           "computed_with_warning" if qc_flag else "computed"),
                "reason": reason, "value": value,
                "unit": catalog[feature_id].unit,
                "qc_range_text": "≥0 words/min" if feature_id == "speaking_rate_words_min" else "0–10 syllables/s",
                "qc_flag": qc_flag,
                "algorithm_version": ALGORITHM_VERSION,
                "parameter_set_id": PARAMETER_SET_ID,
                "analysis_region": "first_to_last_final_patient_speech",
                "pause_min_sec": MIN_PAUSE_SEC,
                "manual_exclusion_policy": MANUAL_EXCLUSION_POLICY,
                "prompt_version": value_row["prompt_version"],
                "count_source": value_row["count_source"],
                "prompt_manifest_sha256": prompt_hash,
                "segmentation_run_id": row.segmentation_run_id,
                "review_run_id": row.review_run_id,
                "boundary_source": row.boundary_source,
                "final_decisions_sha256": decisions_hash,
                "final_intervals_sha256": intervals_hash,
            })
        if progress_callback:
            progress_callback(index, len(kept), f"Acoustic Features — {row.file_name}")
    tables = stage / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    pd.DataFrame(values_rows).to_csv(values_path, index=False)
    pd.DataFrame(status_rows).to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    handoff = write_feature_handoff(
        tables, pd.DataFrame(values_rows), registry, pd.DataFrame(status_rows),
        "acoustic", values_path, registry_path)
    build_feature_delivery(root, "acoustic", handoff)
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version=ALGORITHM_VERSION,
        status="completed_with_warnings" if any(
            row["reason"] or row["qc_flag"] for row in status_rows) else "completed",
        input_artifacts=[
            ArtifactRef(str(decisions_path), "final_segmentation_decisions", "text/csv"),
            ArtifactRef(str(intervals_path), "final_segmentation_intervals", "text/csv"),
            *([ArtifactRef(str(config.prompt_manifest_path), "prompt_count_manifest", "application/json")]
              if config.prompt_manifest_path and Path(config.prompt_manifest_path).is_file() else []),
        ],
        output_artifacts=[
            ArtifactRef(str(values_path), "feature_values", "text/csv"),
            ArtifactRef(str(status_path), "feature_status", "text/csv"),
            ArtifactRef(str(registry_path), "feature_registry", "text/csv"),
            *([ArtifactRef(str(prompt_snapshot), "prompt_count_snapshot", "application/json")]
              if prompt_snapshot else []),
        ],
        config={"selected_features": sorted(selected), "algorithm_version": ALGORITHM_VERSION,
                "parameter_set_id": PARAMETER_SET_ID, "pause_min_sec": MIN_PAUSE_SEC,
                "manual_exclusion_policy": MANUAL_EXCLUSION_POLICY,
                "prompt_manifest_path": config.prompt_manifest_path,
                "prompt_manifest_sha256": prompt_hash,
                "prompt_manifest_snapshot": str(prompt_snapshot) if prompt_snapshot else "",
                "final_decisions_sha256": decisions_hash,
                "final_intervals_sha256": intervals_hash},
    )
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
