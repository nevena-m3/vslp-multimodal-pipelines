"""Family 10 measurements from frozen reviewed DDK events only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult

FAMILY10_IDS = frozenset({"ddk_rate_syll_s", "ddk_cycle_mad_s"})
ALGORITHM_VERSION = "family10-reviewed-ddk-1.0.0"
PARAMETER_SET_ID = "family10_ddk_reviewed_v1"
MIN_VALID_EVENTS = 5
EVENT_COLUMNS = (
    "recording_id", "file_name", "event_index", "sequence_id",
    "event_start_sec", "event_end_sec", "event_time_sec",
    "next_event_interval_sec", "cycle_interval_index", "boundary_source",
    "reviewer", "excluded_from_feature", "exclusion_reason",
    "event_kind", "automatic_source_interval_id",
)


def derive_ddk_feature_events(
    intervals: pd.DataFrame, *, recording_id: str = "", file_name: str = "",
    analysis_start_sec: float, analysis_end_sec: float,
) -> tuple[pd.DataFrame, dict[str, float], str]:
    """Read final events; split trains at exclusions and retain an audit row per event."""
    empty = pd.DataFrame(columns=EVENT_COLUMNS)
    if not np.isfinite([analysis_start_sec, analysis_end_sec]).all() or not (
            0 <= analysis_start_sec < analysis_end_sec):
        return empty, {}, "invalid_ddk_duration"
    required = {"view", "segment_role", "start_sec", "end_sec", "boundary_source"}
    if not required <= set(intervals):
        return empty, {}, "missing_final_ddk_segmentation"
    timeline = intervals.loc[intervals["view"].eq("authoritative")].copy()
    if timeline.empty:
        return empty, {}, "missing_final_ddk_segmentation"
    timeline["start_sec"] = pd.to_numeric(timeline.start_sec, errors="coerce")
    timeline["end_sec"] = pd.to_numeric(timeline.end_sec, errors="coerce")
    if (timeline[["start_sec", "end_sec"]].isna().any().any()
            or (timeline.end_sec <= timeline.start_sec).any()):
        return empty, {}, "invalid_final_ddk_event_boundaries"
    timeline = timeline.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)
    if (timeline.start_sec.to_numpy()[1:] < timeline.end_sec.to_numpy()[:-1] - 1e-7).any():
        return empty, {}, "overlapping_final_ddk_intervals"
    speech = timeline.loc[timeline.segment_role.eq("speech")]
    exclusions = timeline.loc[timeline.segment_role.eq("manual_exclusion")]
    excluded_duration = 0.0
    for row in exclusions.itertuples():
        excluded_duration += max(0.0, min(float(row.end_sec), analysis_end_sec) -
                                 max(float(row.start_sec), analysis_start_sec))
    raw_duration = analysis_end_sec - analysis_start_sec
    effective_duration = raw_duration - excluded_duration
    if effective_duration <= 0:
        return empty, {}, "invalid_ddk_duration"

    # A reviewed exclusion can split one original event into two remnants.
    # Neither remnant is treated as a new syllable event.
    fragmented: set[str] = set()
    if "automatic_source_interval_id" in speech:
        for source_id, group in speech.groupby("automatic_source_interval_id", dropna=False):
            if pd.isna(source_id) or str(source_id).strip() in {"", "nan"} or len(group) < 2:
                continue
            ordered = group.sort_values("start_sec")
            for left, right in zip(ordered.iloc[:-1].itertuples(),
                                   ordered.iloc[1:].itertuples(), strict=True):
                if any(abs(float(ex.start_sec) - float(left.end_sec)) <= 1e-6
                       and abs(float(ex.end_sec) - float(right.start_sec)) <= 1e-6
                       for ex in exclusions.itertuples()):
                    fragmented.add(str(source_id))
    rows: list[dict[str, Any]] = []
    sequence_id = 1
    pending_break = False
    for item in timeline.itertuples():
        role = str(item.segment_role)
        if role == "manual_exclusion":
            rows.append({
                "recording_id": recording_id, "file_name": file_name,
                "event_index": "", "sequence_id": sequence_id,
                "event_start_sec": float(item.start_sec), "event_end_sec": float(item.end_sec),
                "event_time_sec": np.nan, "next_event_interval_sec": np.nan,
                "cycle_interval_index": np.nan, "boundary_source": str(item.boundary_source),
                "reviewer": str(getattr(item, "reviewer", "")),
                "excluded_from_feature": True,
                "exclusion_reason": str(getattr(item, "manual_exclusion_reason", "")),
                "event_kind": "manual_exclusion", "automatic_source_interval_id": "",
            })
            pending_break = True
        elif role == "speech":
            if pending_break:
                sequence_id += 1
                pending_break = False
            source_id = str(getattr(item, "automatic_source_interval_id", ""))
            excluded = source_id in fragmented
            rows.append({
                "recording_id": recording_id, "file_name": file_name,
                "event_index": len([r for r in rows if r["event_kind"] == "ddk_event"]) + 1,
                "sequence_id": sequence_id,
                "event_start_sec": float(item.start_sec), "event_end_sec": float(item.end_sec),
                "event_time_sec": (float(item.start_sec) + float(item.end_sec)) / 2,
                "next_event_interval_sec": np.nan, "cycle_interval_index": np.nan,
                "boundary_source": str(item.boundary_source),
                "reviewer": str(getattr(item, "reviewer", "")),
                "excluded_from_feature": excluded,
                "exclusion_reason": "event_fragmented_by_exclusion" if excluded else "",
                "event_kind": "ddk_event", "automatic_source_interval_id": source_id,
            })
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    valid = events.loc[events.event_kind.eq("ddk_event") &
                       ~events.excluded_from_feature.astype(bool)]
    cycle_count = 0
    for _, group in valid.groupby("sequence_id", sort=True):
        indices = group.index.tolist()
        for left, right in zip(indices[:-1], indices[1:], strict=True):
            cycle_count += 1
            events.at[left, "next_event_interval_sec"] = (
                float(events.at[right, "event_time_sec"]) -
                float(events.at[left, "event_time_sec"]))
            events.at[left, "cycle_interval_index"] = cycle_count
    summary = {
        "analysis_start_sec": analysis_start_sec,
        "analysis_end_sec": analysis_end_sec,
        "raw_analysis_duration_sec": raw_duration,
        "excluded_contamination_duration_sec": excluded_duration,
        "effective_ddk_duration_sec": effective_duration,
        "n_ddk_events": int(len(valid)),
        "n_valid_sequences": int(valid.sequence_id.nunique()),
        "n_cycle_intervals": cycle_count,
    }
    return events, summary, ""


def calculate_family10(
    events: pd.DataFrame, summary: dict[str, float], selected: set[str],
) -> dict[str, tuple[float, str]]:
    """Calculate source-defined DDK rate and adjacent-interval MAD in seconds."""
    if not summary:
        return {feature_id: (np.nan, "missing_final_ddk_segmentation")
                for feature_id in selected}
    valid = events.loc[events.event_kind.eq("ddk_event") &
                       ~events.excluded_from_feature.astype(bool)]
    n = len(valid)
    result: dict[str, tuple[float, str]] = {}
    for feature_id in selected:
        if n < MIN_VALID_EVENTS:
            result[feature_id] = (np.nan, "insufficient_ddk_events")
        elif feature_id == "ddk_rate_syll_s":
            duration = float(summary["effective_ddk_duration_sec"])
            result[feature_id] = (n / duration, "") if duration > 0 else (
                np.nan, "invalid_ddk_duration")
        elif feature_id == "ddk_cycle_mad_s":
            differences = []
            for _, group in valid.groupby("sequence_id", sort=True):
                timestamps = group.event_time_sec.to_numpy(dtype=float)
                if len(timestamps) >= 3:
                    differences.extend(np.abs(np.diff(np.diff(timestamps))).tolist())
            result[feature_id] = (float(np.mean(differences)), "") if differences else (
                np.nan, "insufficient_cycle_intervals")
        else:
            raise ValueError(f"Unregistered Family 10 output: {feature_id}")
    return result


def run_family10_stage(
    segmentation_summary_csv: str | Path, output_root: str | Path, config: Any,
    final_intervals_csv: str | Path | None, registry: pd.DataFrame, progress_callback=None,
    execution_stage_dir: Path | None = None,
) -> StageResult:
    """Write Family 10 values, event audit, status, and handoff from frozen review."""
    root = Path(output_root)
    final = root / "acoustic" / "003_segmentation_review" / "final"
    decisions_path = final / "final_segmentation_decisions.csv"
    intervals_path = final / "final_segmentation_intervals.csv"
    if (Path(segmentation_summary_csv).resolve() != decisions_path.resolve()
            or final_intervals_csv is None
            or Path(final_intervals_csv).resolve() != intervals_path.resolve()
            or not decisions_path.is_file() or not intervals_path.is_file()):
        raise ValueError("Family 10 requires frozen authoritative reviewed DDK segmentation")
    run = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    kept = load_final_segmentation(intervals_path, decisions_path)
    if kept.empty:
        raise ValueError("No kept DDK recordings in frozen reviewed segmentation")
    all_intervals = pd.read_csv(intervals_path, keep_default_na=False)
    decisions_hash, intervals_hash = sha256_file(decisions_path), sha256_file(intervals_path)
    catalog = {row.feature: row for row in registry.itertuples()}
    selected = set(config.selected_features)
    values_rows, status_rows, event_tables = [], [], []
    if progress_callback:
        progress_callback(0, len(kept), "Acoustic Features — DDK")
    for index, row in enumerate(kept.itertuples(), start=1):
        record_id = str(row.recording_id)
        record_timeline = all_intervals.loc[all_intervals.recording_id.astype(str).eq(record_id)]
        if str(row.segmentation_method) == "ddk_energy":
            events, summary, issue = derive_ddk_feature_events(
                record_timeline, recording_id=record_id, file_name=str(row.file_name),
                analysis_start_sec=float(row.analysis_start_sec),
                analysis_end_sec=float(row.analysis_end_sec))
        else:
            events, summary, issue = (pd.DataFrame(columns=EVENT_COLUMNS), {},
                                      "missing_final_ddk_segmentation")
        event_tables.append(events)
        results = calculate_family10(events, summary, selected)
        value_row = {
            "recording_id": record_id, "file_name": row.file_name,
            "project_name": run["project_name"], "task_name": run["task_name"],
            "task_id": "ddk", "run_id": run["run_id"],
            "source_sha256": row.source_sha256,
            "source_file_path": row.source_file_path,
            "segmentation_run_id": row.segmentation_run_id,
            "review_run_id": row.review_run_id,
            "boundary_source": row.boundary_source,
            "algorithm_version": ALGORITHM_VERSION,
            "parameter_set_id": PARAMETER_SET_ID,
            "analysis_region": "reviewed_ddk_analysis_window",
            **{key: summary.get(key, np.nan) for key in (
                "analysis_start_sec", "analysis_end_sec", "raw_analysis_duration_sec",
                "excluded_contamination_duration_sec", "effective_ddk_duration_sec",
                "n_ddk_events", "n_valid_sequences", "n_cycle_intervals")},
            "final_decisions_sha256": decisions_hash,
            "final_intervals_sha256": intervals_hash,
        }
        for feature_id in sorted(selected):
            value, reason = results[feature_id]
            reason = issue or reason
            if reason:
                value = np.nan
            value_row[feature_id] = value
            status_rows.append({
                "recording_id": record_id, "file_name": row.file_name,
                "task_name": run["task_name"], "task_id": "ddk",
                "feature": feature_id, "feature_id": feature_id,
                "family_id": "F10", "status": "unavailable" if reason else "computed",
                "reason": reason, "failure_reason": reason, "value": value,
                "unit": catalog[feature_id].unit,
                "qc_range_text": catalog[feature_id].qc_range_text,
                "algorithm_version": ALGORITHM_VERSION,
                "parameter_set_id": PARAMETER_SET_ID,
                "analysis_region": "reviewed_ddk_analysis_window",
                **{key: value_row[key] for key in (
                    "analysis_start_sec", "analysis_end_sec", "raw_analysis_duration_sec",
                    "excluded_contamination_duration_sec", "effective_ddk_duration_sec",
                    "n_ddk_events", "n_valid_sequences", "n_cycle_intervals",
                    "segmentation_run_id", "review_run_id", "boundary_source",
                    "final_decisions_sha256", "final_intervals_sha256")},
            })
        values_rows.append(value_row)
        if progress_callback:
            progress_callback(index, len(kept), f"Acoustic Features — {row.file_name}")
    stage = execution_stage_dir or root / "acoustic" / "005_features"
    tables = stage / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    events_path = tables / "native_measurements" / "ddk_feature_events.csv"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values_rows).to_csv(values_path, index=False)
    pd.DataFrame(status_rows).to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    pd.concat(event_tables, ignore_index=True).reindex(columns=EVENT_COLUMNS).to_csv(
        events_path, index=False)
    if execution_stage_dir is None:
        handoff = write_feature_handoff(
            tables, pd.DataFrame(values_rows), registry, pd.DataFrame(status_rows),
            "acoustic", values_path, registry_path)
        build_feature_delivery(root, "acoustic", handoff)
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version=ALGORITHM_VERSION,
        status="completed_with_warnings" if any(row["reason"] for row in status_rows)
        else "completed",
        input_artifacts=[
            ArtifactRef(str(decisions_path), "final_segmentation_decisions", "text/csv"),
            ArtifactRef(str(intervals_path), "final_segmentation_intervals", "text/csv"),
        ],
        output_artifacts=[
            ArtifactRef(str(values_path), "feature_values", "text/csv"),
            ArtifactRef(str(status_path), "feature_status", "text/csv"),
            ArtifactRef(str(registry_path), "feature_registry", "text/csv"),
            ArtifactRef(str(events_path), "ddk_feature_events", "text/csv"),
        ],
        config={"selected_features": sorted(selected), "algorithm_version": ALGORITHM_VERSION,
                "parameter_set_id": PARAMETER_SET_ID,
                "event_source": "final_reviewed_ddk_events",
                "event_landmark": "final_reviewed_event_midpoint",
                "rate_denominator": "analysis_window_minus_manual_contamination",
                "sequence_break_on_manual_exclusion": True,
                "minimum_valid_events": MIN_VALID_EVENTS,
                "temporal_variability_definition": "adjacent_interval_mad_unnormalized_s",
                "final_decisions_sha256": decisions_hash,
                "final_intervals_sha256": intervals_hash},
    )
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
