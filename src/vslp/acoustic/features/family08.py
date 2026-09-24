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
from vslp.acoustic.features.catalog import task_key
from vslp.acoustic.features.family09 import (
    ALGORITHM_VERSION as FAMILY09_VERSION, FAMILY09_IDS, calculate_family09,
)
from vslp.acoustic.features.family07 import (
    ALIGNMENT_IDS, ALGORITHM_VERSION as FAMILY07_VERSION, EVENT_COLUMNS as FAMILY07_EVENT_COLUMNS,
    FAMILY07_IDS, calculate_family07, derive_duration_events,
)
from vslp.acoustic.features.reviewed_timing import (
    EVENT_COLUMNS, MIN_INTERNAL_PAUSE_SEC, derive_reviewed_timing,
)
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
MIN_PAUSE_SEC = MIN_INTERNAL_PAUSE_SEC
MANUAL_EXCLUSION_POLICY = "subtract"


def load_prompt_counts(path: str | Path | None, task_name: str,
                       task_id: str | None = None) -> tuple[dict[str, Any] | None, str]:
    """Require an explicit fixed-prompt count and version; never guess text counts."""
    if not path or not Path(path).is_file():
        return None, "prompt_manifest_missing"
    try:
        item = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "prompt_manifest_unreadable"
    expected_task_id = task_id or task_key(task_name)
    if (item.get("schema_version") != "1" or not expected_task_id
            or item.get("task_id") != expected_task_id):
        return None, "prompt_manifest_schema_or_task_mismatch"
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
    _events, summary, issue = derive_reviewed_timing(intervals)
    if issue:
        return {}, issue
    if manual_exclusion_policy != MANUAL_EXCLUSION_POLICY:
        return {}, "unsupported_manual_exclusion_policy"
    return summary, ""


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


def run_reviewed_timing_stage(segmentation_summary_csv: str | Path, output_root: str | Path,
                              config: Any, final_intervals_csv: str | Path | None,
                              registry: pd.DataFrame, progress_callback=None,
                              execution_stage_dir: Path | None = None) -> StageResult:
    root = Path(output_root)
    final_dir = root / "acoustic" / "003_segmentation_review" / "final"
    decisions_path = final_dir / "final_segmentation_decisions.csv"
    intervals_path = final_dir / "final_segmentation_intervals.csv"
    if (Path(segmentation_summary_csv).resolve() != decisions_path.resolve()
            or final_intervals_csv is None
            or Path(final_intervals_csv).resolve() != intervals_path.resolve()
            or not decisions_path.is_file() or not intervals_path.is_file()):
        raise ValueError("Families 07/08/09 require frozen authoritative reviewed segmentation")
    run = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    task_name = str(run.get("task_name", ""))
    stage = execution_stage_dir or root / "acoustic" / "005_features"
    selected = set(config.selected_features)
    selected07 = selected & FAMILY07_IDS
    selected08, selected09 = selected & FAMILY08_IDS, selected & FAMILY09_IDS
    prompt_snapshot = None
    if selected08 and config.prompt_manifest_path and Path(config.prompt_manifest_path).is_file():
        prompt_snapshot = stage / "configs" / "prompt_count_manifest.json"
        prompt_snapshot.parent.mkdir(parents=True, exist_ok=True)
        if Path(config.prompt_manifest_path).resolve() != prompt_snapshot.resolve():
            shutil.copy2(config.prompt_manifest_path, prompt_snapshot)
    current_task_id = run.get("task_id") or task_key(task_name) or ""
    prompt, prompt_issue = load_prompt_counts(
        prompt_snapshot, task_name, current_task_id) if selected08 else (None, "")
    prompt_hash = sha256_file(prompt_snapshot) if prompt_snapshot else ""
    alignment_store = None
    alignment_table, alignment_issue, alignment_hash = pd.DataFrame(), "", ""
    if selected07 & ALIGNMENT_IDS:
        from vslp.acoustic.alignment import load_final_alignment

        alignment_store, alignment_issue = load_final_alignment(root)
        if alignment_store is not None:
            alignment_table = alignment_store.family07_rows()
            alignment_hash = alignment_store.manifest["final_words_sha256"]
    kept = load_final_segmentation(intervals_path, decisions_path)
    if kept.empty:
        raise ValueError("No kept recordings are available in frozen reviewed segmentation")
    all_intervals = pd.read_csv(intervals_path, keep_default_na=False)
    decisions_hash = sha256_file(decisions_path)
    intervals_hash = sha256_file(intervals_path)
    catalog = {row.feature: row for row in registry.itertuples()}
    values_rows = []
    status_rows = []
    event_tables = []
    duration_event_tables = []
    if progress_callback:
        progress_callback(0, len(kept), "Acoustic Features")
    for index, row in enumerate(kept.itertuples(), start=1):
        record_id = str(row.recording_id)
        timeline = all_intervals.loc[all_intervals.recording_id.astype(str).eq(record_id)]
        events, timing, timing_issue = derive_reviewed_timing(
            timeline, recording_id=record_id, file_name=str(row.file_name),
            recording_duration_sec=float(row.duration_sec),
            parameter_set_id="family09_bamboo_reviewed_v1")
        if selected09:
            event_tables.append(events)
        record_alignment = (alignment_table.loc[
            alignment_table.recording_id.astype(str).eq(record_id)].copy()
            if not alignment_table.empty else None)
        duration_events = derive_duration_events(
            record_id, str(row.file_name), events, record_alignment)
        family07_results, family07_support = calculate_family07(
            duration_events, timing, record_alignment, selected07)
        if selected07:
            duration_event_tables.append(duration_events)
        results = {
            **family07_results,
            **calculate_family08(timing, prompt, selected08),
            **calculate_family09(events, timing, selected09),
        }
        value_row = {
            "recording_id": record_id, "file_name": row.file_name,
            "project_name": run["project_name"], "task_name": task_name,
            "task_id": current_task_id,
            "task_type": run.get("task_type", ""),
            "run_id": run["run_id"], "source_sha256": row.source_sha256,
            "source_file_path": row.source_file_path,
            "segmentation_run_id": row.segmentation_run_id,
            "review_run_id": row.review_run_id,
            "alignment_run_id": (alignment_store.manifest["alignment_run_id"]
                                 if alignment_store is not None else ""),
            "alignment_words_sha256": alignment_hash,
            "alignment_phones_sha256": (alignment_store.manifest["final_phones_sha256"]
                                        if alignment_store is not None else ""),
            "boundary_source": row.boundary_source,
            "prompt_version": prompt.get("prompt_version", "") if prompt else "",
            "count_source": prompt.get("count_source", "") if prompt else "",
            "prompt_manifest_sha256": prompt_hash,
            "algorithm_version": (ALGORITHM_VERSION if not selected09 else
                                  FAMILY09_VERSION if not selected08 else "family08+family09"),
            "parameter_set_id": (PARAMETER_SET_ID if not selected09 else
                                 "family09_bamboo_reviewed_v1" if not selected08 else "mixed"),
            "analysis_region": "first_to_last_final_patient_speech",
            "pause_min_sec": MIN_PAUSE_SEC,
            "manual_exclusion_policy": MANUAL_EXCLUSION_POLICY,
            "raw_task_duration_sec": timing.get("raw_task_duration_sec", np.nan),
            "analysis_start_sec": timing.get("analysis_start_sec", np.nan),
            "analysis_end_sec": timing.get("analysis_end_sec", np.nan),
            "analysis_window_duration_sec": timing.get("analysis_window_duration_sec", np.nan),
            "excluded_contamination_duration_sec": timing.get("excluded_contamination_duration_sec", np.nan),
            "n_pause_events": timing.get("n_pause_events", np.nan),
            "n_phrase_events": timing.get("n_phrase_events", np.nan),
            "utterance_start_sec": timing.get("utterance_start_sec", np.nan),
            "utterance_end_sec": timing.get("utterance_end_sec", np.nan),
            "elapsed_task_sec": timing.get("elapsed_task_sec", np.nan),
            "internal_pauses_ge_300ms_sec": timing.get("internal_pauses_ge_300ms_sec", np.nan),
            "articulation_speech_sec": timing.get("articulation_speech_sec", np.nan),
            "n_duration_events": family07_support["n_events"],
            "n_expected_words": family07_support["n_expected_words"],
            "n_aligned_words": family07_support["n_aligned_words"],
            "alignment_coverage": family07_support["alignment_coverage"],
            "n_valid_vowels": family07_support["n_valid_vowels"],
            "n_adjacent_vowel_pairs": family07_support["n_adjacent_vowel_pairs"],
            "task_boundary_method": family07_support["task_boundary_method"],
            "final_decisions_sha256": decisions_hash,
            "final_intervals_sha256": intervals_hash,
            **{feature_id: results[feature_id][0] for feature_id in sorted(selected)},
        }
        values_rows.append(value_row)
        for feature_id in sorted(selected):
            value, reason = results[feature_id]
            reason = (timing_issue or
                      (alignment_issue if feature_id in ALIGNMENT_IDS else "") or
                      (prompt_issue if feature_id in selected08 else "") or reason)
            numeric_value = isinstance(value, (int, float, np.integer, np.floating))
            if reason and (not numeric_value or np.isfinite(value)):
                value = np.nan
                value_row[feature_id] = np.nan
            qc_flag = ("above_broad_plausibility_range"
                       if numeric_value and np.isfinite(value)
                       and feature_id in {"speaking_rate_syll_s", "articulation_rate_syll_s"}
                       and value > 10
                       else "")
            status_rows.append({
                "recording_id": record_id, "file_name": row.file_name,
                "task_name": task_name, "task_id": value_row["task_id"],
                "feature": feature_id, "feature_id": feature_id,
                "family_id": catalog[feature_id].family_id,
                "status": ("unavailable" if reason else
                           "computed_with_warning" if qc_flag else "computed"),
                "reason": reason, "failure_reason": reason, "value": value,
                "unit": catalog[feature_id].unit,
                "qc_range_text": catalog[feature_id].qc_range_text,
                "qc_flag": qc_flag,
                "algorithm_version": catalog[feature_id].algorithm_version,
                "parameter_set_id": catalog[feature_id].parameter_set_id,
                "analysis_region": catalog[feature_id].analysis_region,
                "pause_min_sec": MIN_PAUSE_SEC,
                "sample_sd_ddof": 1 if feature_id in selected09 else "",
                "manual_exclusion_policy": MANUAL_EXCLUSION_POLICY,
                "analysis_start_sec": value_row["analysis_start_sec"],
                "analysis_end_sec": value_row["analysis_end_sec"],
                "excluded_contamination_duration_sec": value_row["excluded_contamination_duration_sec"],
                "n_pause_events": value_row["n_pause_events"],
                "n_phrase_events": value_row["n_phrase_events"],
                "n_events": value_row["n_duration_events"],
                "n_expected_words": value_row["n_expected_words"],
                "n_aligned_words": value_row["n_aligned_words"],
                "alignment_coverage": value_row["alignment_coverage"],
                "n_valid_vowels": value_row["n_valid_vowels"],
                "n_adjacent_vowel_pairs": value_row["n_adjacent_vowel_pairs"],
                "task_boundary_method": value_row["task_boundary_method"],
                "alignment_table_sha256": alignment_hash,
                "alignment_run_id": value_row["alignment_run_id"],
                "alignment_phones_sha256": value_row["alignment_phones_sha256"],
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
    events_path = None
    if selected09:
        events_path = tables / "native_measurements" / "pause_phrase_events.csv"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(event_tables, ignore_index=True).reindex(columns=EVENT_COLUMNS).to_csv(
            events_path, index=False)
    duration_events_path = None
    if selected07:
        duration_events_path = tables / "native_measurements" / "family07_duration_events.csv"
        duration_events_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(duration_event_tables, ignore_index=True).reindex(
            columns=FAMILY07_EVENT_COLUMNS).to_csv(duration_events_path, index=False)
    if execution_stage_dir is None:
        handoff = write_feature_handoff(
            tables, pd.DataFrame(values_rows), registry, pd.DataFrame(status_rows),
            "acoustic", values_path, registry_path)
        build_feature_delivery(root, "acoustic", handoff)
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version="reviewed-timing-1.0.0",
        status="completed_with_warnings" if any(
            row["reason"] or row["qc_flag"] for row in status_rows) else "completed",
        input_artifacts=[
            ArtifactRef(str(decisions_path), "final_segmentation_decisions", "text/csv"),
            ArtifactRef(str(intervals_path), "final_segmentation_intervals", "text/csv"),
            *([ArtifactRef(str(config.prompt_manifest_path), "prompt_count_manifest", "application/json")]
              if config.prompt_manifest_path and Path(config.prompt_manifest_path).is_file() else []),
            *([ArtifactRef(alignment_store.manifest["final_words_path"],
                           "final_alignment_words", "text/csv"),
               ArtifactRef(alignment_store.manifest["final_phones_path"],
                           "final_alignment_phones", "text/csv")]
              if alignment_store is not None else []),
        ],
        output_artifacts=[
            ArtifactRef(str(values_path), "feature_values", "text/csv"),
            ArtifactRef(str(status_path), "feature_status", "text/csv"),
            ArtifactRef(str(registry_path), "feature_registry", "text/csv"),
            *([ArtifactRef(str(events_path), "pause_phrase_events", "text/csv")]
              if events_path else []),
            *([ArtifactRef(str(duration_events_path), "family07_duration_events", "text/csv")]
              if duration_events_path else []),
            *([ArtifactRef(str(prompt_snapshot), "prompt_count_snapshot", "application/json")]
              if prompt_snapshot else []),
        ],
        config={"selected_features": sorted(selected),
                "algorithm_versions": {
                    **({"F07": FAMILY07_VERSION} if selected07 else {}),
                    **({"F08": ALGORITHM_VERSION} if selected08 else {}),
                    **({"F09": FAMILY09_VERSION} if selected09 else {})},
                "parameter_set_ids": {
                    **({"F07": "family07_reviewed_timing_v1/family07_alignment_tokens_v1"}
                       if selected07 else {}),
                    **({"F08": PARAMETER_SET_ID} if selected08 else {}),
                    **({"F09": "family09_bamboo_reviewed_v1"} if selected09 else {})},
                "pause_min_sec": MIN_PAUSE_SEC, "sample_sd_ddof": 1,
                "manual_exclusion_policy": MANUAL_EXCLUSION_POLICY,
                "prompt_manifest_path": config.prompt_manifest_path,
                "prompt_manifest_sha256": prompt_hash,
                "prompt_manifest_snapshot": str(prompt_snapshot) if prompt_snapshot else "",
                "alignment_run_id": (alignment_store.manifest["alignment_run_id"]
                                     if alignment_store is not None else ""),
                "alignment_table_sha256": alignment_hash,
                "final_decisions_sha256": decisions_hash,
                "final_intervals_sha256": intervals_hash},
    )
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
