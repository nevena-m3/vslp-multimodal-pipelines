"""Persistent manual segmentation review and immutable final boundary freeze.

Adapted from quality_framework_features segmentation_review.py and segmentation.py.
No study metadata or metadata-based exclusion gates are used here.
"""

from __future__ import annotations

import os
import json
import hashlib
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from vslp.acoustic.segment.pipeline import _frames, _plot, _segments
from vslp.acoustic.segment.pipeline import segmentation_runs
from vslp.acoustic.preprocess.stage import safe_stem
from vslp.acoustic.segment.silero_reference import Interval, erode_intervals, internal_nonspeech
from vslp.acoustic.segment.stage import _read_canonical_audio
from vslp.core.provenance import python_environment, sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


REVIEW_COLUMNS = [
    "recording_id", "file_name", "task_name", "segmentation_method",
    "segmentation_run_id", "review_run_id", "segmentation_parameters", "automatic_status",
    "automatic_flags", "review_required", "duration_sec", "analysis_wav_path",
    "automatic_segments_path", "automatic_frames_path", "automatic_boundaries_path",
    "automatic_plot_path", "source_sha256", "project_name", "run_id",
    "run_created_at_local", "run_created_at_utc", "source_file_path",
    "full_phonation_start", "full_phonation_end", "stable_region_start", "stable_region_end",
    "n_events", "n_syllables", "ddk_sequence_start_sec", "ddk_sequence_end_sec",
    "ddk_sequence_duration_sec", "ddk_rate_hz", "ctv_sec", "ddk_timing_source",
    "cycle_mean_sec", "cycle_sd_sec",
    "review_status", "final_decision", "final_decision_source", "recording_exclusion_reason",
    "boundary_source", "manual_override_applied", "reviewer",
    "review_date", "review_notes", "manual_segments_path", "final_segments_path",
    "frame_csv_path", "segments_csv_path", "plot_path", "analysis_start_sec",
    "analysis_end_sec", "manual_exclusions_path", "manual_exclusion_applied",
    "reviewed_plot_path", "final_intervals_path", "last_modified_utc",
]
OVERRIDE_COLUMNS = ["recording_id", "file_name", "segmentation_run_id",
                    "segment_index", "start_sec", "end_sec", "duration_sec",
                    "boundary_source", "reviewer", "review_date", "review_notes", "notes"]
EXCLUSION_COLUMNS = ["recording_id", "file_name", "segmentation_run_id",
                     "exclusion_index", "start_sec", "end_sec", "duration_sec",
                     "exclusion_reason", "reason",
                     "reviewer", "review_date", "notes"]
WINDOW_COLUMNS = ["segmentation_run_id", "recording_id", "file_name",
                  "analysis_start_sec", "analysis_end_sec", "reviewer",
                  "review_date", "notes", "last_modified_utc"]
AUDIT_COLUMNS = ["event_id", "timestamp_utc", "segmentation_run_id",
                 "recording_id", "file_name", "reviewer", "action",
                 "old_value_json", "new_value_json", "notes"]
INTERVAL_COLUMNS = ["recording_id", "file_name", "task_name", "segmentation_method",
                    "segmentation_run_id", "review_run_id", "segmentation_parameters", "view",
                    "interval_index", "start_sec", "end_sec", "duration_sec", "boundary_source",
                    "reviewer", "review_date", "review_notes", "segment_type", "segment_role",
                    "analysis_start_sec", "analysis_end_sec", "manual_exclusion_applied",
                    "manual_exclusion_reason", "automatic_source_interval_id"]

EXCLUSION_REASONS = {"Other speaker", "Cough / throat clear", "Non-task speech",
                     "Recording artifact", "Other"}
RECORDING_EXCLUSION_REASONS = {"Preprocess not accepted", "Unusable audio",
                               "Segmentation excluded", "Other"}


def select_segmentation_run(output_root: str | Path, segmentation_run_id: str) -> Path:
    """Bind all review reads/writes to one explicitly selected immutable run."""
    matches = [entry for entry in segmentation_runs(output_root)
               if entry["segmentation_run_id"] == segmentation_run_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown segmentation run: {segmentation_run_id}")
    summary = Path(matches[0]["summary_path"])
    if not summary.is_file():
        raise FileNotFoundError(summary)
    registry_path = Path(output_root) / "acoustic" / "003_segmentation_review" / "logs" / "review_registry.json"
    registry = ({"review_runs": []} if not registry_path.is_file() else
                json.loads(registry_path.read_text(encoding="utf-8")))
    registry.update({"selected_segmentation_run_id": segmentation_run_id,
                     "selected_summary_path": str(summary),
                     "selected_summary_sha256": sha256_file(summary)})
    _atomic_json(registry, registry_path)
    return summary


def parse_manual_intervals_text(text: str) -> list[tuple[float, float]]:
    """Port the reference's one-start,end-pair-per-line parser."""
    intervals: list[tuple[float, float]] = []
    for line_number, raw_line in enumerate(str(text).splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pieces = [piece.strip() for piece in line.replace("\t", ",").split(",")]
        if len(pieces) != 2:
            raise ValueError(f"Line {line_number} must contain exactly start_sec,end_sec: {raw_line!r}")
        try:
            start, end = map(float, pieces)
        except ValueError as exc:
            raise ValueError(f"Line {line_number} contains a non-numeric boundary: {raw_line!r}") from exc
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError(f"Line {line_number} contains a non-finite boundary.")
        intervals.append((start, end))
    return intervals


def validate_manual_interval_list(intervals: list[tuple[float, float]], *,
                                  duration_sec: float) -> list[tuple[float, float]]:
    """Port the reference's finite, ordered, non-overlap and duration checks."""
    if not intervals:
        raise ValueError("KEEP_MANUAL requires at least one interval.")
    ordered = sorted((float(start), float(end)) for start, end in intervals)
    for index, (start, end) in enumerate(ordered):
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Invalid manual interval {index}: ({start}, {end})")
        if end > float(duration_sec) + 1e-6:
            raise ValueError(f"Manual interval {index} ends beyond the {duration_sec:.6f}s recording.")
        if index and start < ordered[index - 1][1] - 1e-9:
            raise ValueError(f"Manual intervals {index - 1} and {index} overlap.")
    return ordered


def _paths(output_root: str | Path) -> dict[str, Path]:
    base = Path(output_root) / "acoustic" / "003_segmentation_review"
    registry_path = base / "logs" / "review_registry.json"
    legacy_selection = base / "selected_segmentation_run.json"
    if not registry_path.is_file() and legacy_selection.is_file():
        selected = json.loads(legacy_selection.read_text(encoding="utf-8"))
        _atomic_json({"review_runs": [],
                      "selected_segmentation_run_id": selected.get("segmentation_run_id", "default"),
                      "selected_summary_path": selected.get("summary_path", ""),
                      "selected_summary_sha256": selected.get("summary_sha256", "")}, registry_path)
        legacy_selection.unlink()
    registry = (json.loads(registry_path.read_text(encoding="utf-8"))
                if registry_path.is_file() else {})
    run_id = registry.get("selected_segmentation_run_id", "default")
    review_run_id = f"review_{run_id}"
    stage = base / "runs" / review_run_id
    legacy_run = base / "runs" / run_id
    if legacy_run.is_dir() and not stage.exists():
        stage.parent.mkdir(parents=True, exist_ok=True)
        os.replace(legacy_run, stage)
    tables = stage / "tables"
    final = base / "final"
    for old_name, new_name in {
        "segmentation_review_queue.csv": "review_queue.csv",
        "segmentation_review_decisions.csv": "review_decisions.csv",
        "manual_segmentation_overrides.csv": "manual_overrides.csv",
        "manual_exclusion_intervals.csv": "exclusion_intervals.csv",
        "reviewed_exclusion_intervals.csv": "exclusion_intervals.csv",
        "review_audit_log.csv": "audit_log.csv",
    }.items():
        old_path, new_path = tables / old_name, tables / new_name
        if old_path.is_file() and not new_path.exists():
            os.replace(old_path, new_path)
    for name in ("final_segmentation_decisions.csv", "final_segmentation_intervals.csv"):
        old_path, new_path = tables / name, final / name
        if old_path.is_file() and not new_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old_path, new_path)
    legacy_tables = base / "tables"
    if legacy_tables.is_dir():
        mapping = {
            "segmentation_review_queue.csv": tables / "review_queue.csv",
            "segmentation_review_decisions.csv": tables / "review_decisions.csv",
            "manual_segmentation_overrides.csv": tables / "manual_overrides.csv",
            "manual_exclusion_intervals.csv": tables / "exclusion_intervals.csv",
            "reviewed_exclusion_intervals.csv": tables / "exclusion_intervals.csv",
            "analysis_windows.csv": tables / "analysis_windows.csv",
            "review_audit_log.csv": tables / "audit_log.csv",
            "final_segmentation_decisions.csv": final / "final_segmentation_decisions.csv",
            "final_segmentation_intervals.csv": final / "final_segmentation_intervals.csv",
        }
        archive = stage / "logs" / "legacy_root_tables"
        for source in legacy_tables.rglob("*"):
            if not source.is_file():
                continue
            destination = mapping.get(source.name, archive / source.relative_to(legacy_tables))
            if destination.exists():
                destination = archive / source.relative_to(legacy_tables)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
        for directory in sorted(legacy_tables.rglob("*"), reverse=True):
            if directory.is_dir():
                directory.rmdir()
        legacy_tables.rmdir()
        legacy_decisions = final / "final_segmentation_decisions.csv"
        legacy_intervals = final / "final_segmentation_intervals.csv"
        if legacy_decisions.is_file() and legacy_intervals.is_file() and not (final / "final_manifest.json").exists():
            frame = pd.read_csv(legacy_decisions, dtype=str, keep_default_na=False)
            reviewers = frame["reviewer"] if "reviewer" in frame else pd.Series("", index=frame.index)
            counts = {"n_total": len(frame),
                      "n_keep_auto": int(frame.final_decision.eq("KEEP_AUTO").sum()),
                      "n_keep_manual": int(frame.final_decision.eq("KEEP_MANUAL").sum()),
                      "n_excluded": int(frame.final_decision.eq("EXCLUDE").sum()),
                      "n_failed": int(frame.final_decision.eq("FAILED").sum()),
                      "n_manually_reviewed": int(reviewers.ne("").sum()),
                      "n_automatic_accepted_untouched": int((frame.automatic_status.eq("ACCEPTED") & reviewers.eq("")).sum())}
            _atomic_csv(pd.DataFrame([counts]), final / "final_segmentation_summary.csv")
            _atomic_json({"source_segmentation_run_id": run_id, "review_run_id": review_run_id,
                          "freeze_timestamp_utc": "legacy_migration", **counts,
                          "decisions_sha256": sha256_file(legacy_decisions),
                          "intervals_sha256": sha256_file(legacy_intervals)}, final / "final_manifest.json")
    legacy_decisions = final / "final_segmentation_decisions.csv"
    legacy_intervals = final / "final_segmentation_intervals.csv"
    if legacy_decisions.is_file() and legacy_intervals.is_file() and not (final / "final_manifest.json").exists():
        frame = pd.read_csv(legacy_decisions, dtype=str, keep_default_na=False)
        reviewers = frame["reviewer"] if "reviewer" in frame else pd.Series("", index=frame.index)
        counts = {"n_total": len(frame), "n_keep_auto": int(frame.final_decision.eq("KEEP_AUTO").sum()),
                  "n_keep_manual": int(frame.final_decision.eq("KEEP_MANUAL").sum()),
                  "n_excluded": int(frame.final_decision.eq("EXCLUDE").sum()),
                  "n_failed": int(frame.final_decision.eq("FAILED").sum()),
                  "n_manually_reviewed": int(reviewers.ne("").sum()),
                  "n_automatic_accepted_untouched": int((frame.automatic_status.eq("ACCEPTED") & reviewers.eq("")).sum())}
        _atomic_csv(pd.DataFrame([counts]), final / "final_segmentation_summary.csv")
        _atomic_json({"source_segmentation_run_id": run_id, "review_run_id": review_run_id,
                      "freeze_timestamp_utc": "legacy_migration", **counts,
                      "decisions_sha256": sha256_file(legacy_decisions),
                      "intervals_sha256": sha256_file(legacy_intervals)}, final / "final_manifest.json")
    return {"stage": stage, "review_run_id": review_run_id,
            "registry": base / "logs" / "review_registry.json",
            "stage_manifest": base / "logs" / "stage_manifest.json",
            "queue": tables / "review_queue.csv",
            "decisions": tables / "review_decisions.csv",
            "overrides": tables / "manual_overrides.csv",
            "exclusions": tables / "exclusion_intervals.csv",
            "entries": tables / "review_entries",
            "events": stage / "logs" / "review_events",
            "windows": tables / "analysis_windows.csv",
            "audit": tables / "audit_log.csv",
            "final_decisions": final / "final_segmentation_decisions.csv",
            "final_intervals": final / "final_segmentation_intervals.csv",
            "final_summary": final / "final_segmentation_summary.csv",
            "final_manifest": final / "final_manifest.json",
            "manifest": stage / "logs" / "stage_manifest.json"}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _review_entries(output_root: str | Path) -> list[dict[str, Any]]:
    paths = _paths(output_root)
    legacy = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(paths["entries"].glob("*.json"))]
    events = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(paths["events"].glob("*.json"))]
    return [*legacy, *events]


def _replay_entries(frame: pd.DataFrame, entries: list[dict[str, Any]], key: str) -> pd.DataFrame:
    for entry in entries:
        recording_id = str(entry["recording_id"])
        if key == "decision":
            selected = frame.recording_id.eq(recording_id)
            if int(selected.sum()) != 1:
                raise ValueError(f"Review journal has unknown recording ID: {recording_id}")
            for column, value in entry[key].items():
                frame.loc[selected, column] = str(value)
        else:
            frame = frame.loc[~frame.recording_id.eq(recording_id)].copy()
            addition = pd.DataFrame(entry[key], columns=OVERRIDE_COLUMNS if key == "overrides" else EXCLUSION_COLUMNS)
            frame = pd.concat([frame, addition], ignore_index=True)
    return frame


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False).reindex(columns=columns, fill_value="")


def _boolean(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _pending(decisions: pd.DataFrame) -> pd.DataFrame:
    required = decisions["review_required"].map(_boolean)
    unresolved = decisions.final_decision.isin(["", "PENDING"])
    return decisions.loc[required & unresolved]


def initialize_segmentation_review(segmentation_summary_csv: str | Path,
                                   output_root: str | Path) -> StageResult:
    """Create a persistent queue once; refuse stale automatic segmentation inputs."""
    source = Path(segmentation_summary_csv)
    if not source.is_file():
        raise FileNotFoundError(source)
    paths = _paths(output_root)
    if paths["queue"].exists():
        existing = pd.read_csv(paths["queue"], dtype=str, keep_default_na=False)
        if "automatic_summary_sha256" not in existing or not existing["automatic_summary_sha256"].eq(sha256_file(source)).all():
            raise ValueError("Automatic segmentation changed after review initialization; use a new run.")
        return StageResult(status="completed" if paths["final_manifest"].exists() else "completed_with_warnings",
            manifest_path=paths["manifest"], summary_table=paths["queue"], report_path=None)
    summary = pd.read_csv(source, dtype=str, keep_default_na=False)
    selected_run_id = "" if paths["review_run_id"] == "review_default" else paths["review_run_id"].removeprefix("review_")
    if summary.recording_id.duplicated().any():
        raise ValueError("Automatic segmentation has duplicate recording IDs")
    rows = []
    source_hash = sha256_file(source)
    for _, auto in summary.iterrows():
        status = str(auto.get("automatic_status", "FAILED")).upper()
        if status not in {"ACCEPTED", "REVIEW", "EXCLUDED", "FAILED"}:
            raise ValueError(f"Invalid automatic status: {status}")
        row = {"recording_id": auto.get("recording_id", ""), "file_name": auto.get("file_name", ""),
               "task_name": auto.get("task_name", ""),
               "segmentation_method": auto.get("segmentation_method", auto.get("method", "")),
               "segmentation_run_id": auto.get("segmentation_run_id", "") or selected_run_id,
               "review_run_id": paths["review_run_id"],
               "segmentation_parameters": auto.get("parameter_profile", ""),
               "automatic_status": status, "automatic_flags": auto.get("flags", ""),
               "review_required": str(status in {"REVIEW", "EXCLUDED", "FAILED"} or _boolean(auto.get("review_required", False))),
               "duration_sec": auto.get("duration_sec", ""),
               "analysis_wav_path": auto.get("analysis_wav_path", ""),
               "automatic_segments_path": auto.get("segments_csv_path", ""),
               "automatic_frames_path": auto.get("frame_csv_path", ""),
               "automatic_boundaries_path": auto.get("boundaries_csv_path", ""),
               "automatic_plot_path": auto.get("plot_png_path", ""),
               "source_sha256": auto.get("source_sha256", ""),
               "project_name": auto.get("project_name", ""), "run_id": auto.get("run_id", ""),
               "run_created_at_local": auto.get("run_created_at_local", ""),
               "run_created_at_utc": auto.get("run_created_at_utc", ""),
               "source_file_path": auto.get("source_file_path", ""),
               "full_phonation_start": auto.get("full_phonation_start", ""),
               "full_phonation_end": auto.get("full_phonation_end", ""),
               "stable_region_start": auto.get("stable_region_start", ""),
               "stable_region_end": auto.get("stable_region_end", ""),
               "n_events": auto.get("n_events", ""),
               "n_syllables": auto.get("n_syllables", ""),
               "ddk_sequence_start_sec": auto.get("ddk_sequence_start_sec", ""),
               "ddk_sequence_end_sec": auto.get("ddk_sequence_end_sec", ""),
               "ddk_sequence_duration_sec": auto.get("ddk_sequence_duration_sec", ""),
               "ddk_rate_hz": auto.get("ddk_rate_hz", ""),
               "ctv_sec": auto.get("ctv_sec", ""), "ddk_timing_source": "automatic_energy_peak",
               "cycle_mean_sec": auto.get("cycle_mean_sec", ""), "cycle_sd_sec": auto.get("cycle_sd_sec", ""),
               "review_status": "UNREVIEWED",
               "final_decision": ("FAILED" if status == "FAILED" else
                                  "KEEP_AUTO" if status == "ACCEPTED" and not _boolean(auto.get("review_required", False)) else "PENDING"),
               "final_decision_source": "AUTOMATIC" if status == "ACCEPTED" and not _boolean(auto.get("review_required", False)) else "",
               "recording_exclusion_reason": "",
               "boundary_source": "AUTO" if status == "ACCEPTED" and not _boolean(auto.get("review_required", False)) else "NONE",
               "manual_override_applied": "False", "reviewer": "", "review_date": "",
               "review_notes": "", "manual_segments_path": "", "final_segments_path": "",
               "frame_csv_path": "", "segments_csv_path": "", "plot_path": "",
               "analysis_start_sec": "0", "analysis_end_sec": auto.get("duration_sec", ""),
               "manual_exclusions_path": str(paths["exclusions"]),
               "manual_exclusion_applied": "False", "reviewed_plot_path": "",
               "final_intervals_path": "", "last_modified_utc": ""}
        rows.append(row)
    queue = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    queue["automatic_summary_sha256"] = source_hash
    decisions = queue[REVIEW_COLUMNS].copy()
    _atomic_csv(queue, paths["queue"])
    _atomic_csv(decisions, paths["decisions"])
    _atomic_csv(pd.DataFrame(columns=OVERRIDE_COLUMNS), paths["overrides"])
    _atomic_csv(pd.DataFrame(columns=EXCLUSION_COLUMNS), paths["exclusions"])
    manifest = StageManifest(stage_name="acoustic_segmentation_manual_review", stage_version="1.0.0",
        status="completed_with_warnings" if not _pending(decisions).empty else "completed",
        input_artifacts=[ArtifactRef(path=str(source), role="automatic_segmentation", media_type="text/csv", sha256=source_hash)],
        output_artifacts=[ArtifactRef(path=str(paths["queue"]), role="review_queue", media_type="text/csv"),
                          ArtifactRef(path=str(paths["decisions"]), role="review_decisions", media_type="text/csv"),
                          ArtifactRef(path=str(paths["overrides"]), role="manual_boundaries", media_type="text/csv"),
                          ArtifactRef(path=str(paths["exclusions"]), role="reviewed_exclusions", media_type="text/csv")],
        config={"automatic_summary_sha256": source_hash}, environment={"python": python_environment()},
        notes=["Automatic segmentation artifacts are immutable; review decisions are separate."])
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json(paths["manifest"])
    registry = ({"review_runs": []} if not paths["registry"].is_file() else
                json.loads(paths["registry"].read_text(encoding="utf-8")))
    if not any(item["review_run_id"] == paths["review_run_id"] for item in registry["review_runs"]):
        registry["review_runs"].append({"review_run_id": paths["review_run_id"],
            "segmentation_run_id": selected_run_id, "automatic_summary_path": str(source),
            "automatic_summary_sha256": source_hash, "review_path": str(paths["stage"])})
    _atomic_json(registry, paths["registry"])
    _atomic_json({"stage": "segmentation_manual_review", "registry_path": str(paths["registry"]),
                  "review_runs": len(registry["review_runs"])}, paths["stage_manifest"])
    return StageResult(status=manifest.status, manifest_path=paths["manifest"],
                       summary_table=paths["queue"], report_path=None)


def load_review_state(output_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = _paths(output_root)
    if not paths["decisions"].is_file():
        raise FileNotFoundError("Initialize Segmentation manual review first")
    entries = _review_entries(output_root)
    decisions = _replay_entries(_read_csv(paths["decisions"], REVIEW_COLUMNS), entries, "decision")
    overrides = _replay_entries(_read_csv(paths["overrides"], OVERRIDE_COLUMNS), entries, "overrides")
    return decisions, overrides


def load_review_exclusions(output_root: str | Path) -> pd.DataFrame:
    path = _paths(output_root)["exclusions"]
    frame = _read_csv(path, EXCLUSION_COLUMNS) if path.exists() else pd.DataFrame(columns=EXCLUSION_COLUMNS)
    return _replay_entries(frame, _review_entries(output_root), "exclusions")


def validate_analysis_window(start: float, end: float, duration: float) -> tuple[float, float]:
    values = (float(start), float(end), float(duration))
    if not all(np.isfinite(value) for value in values) or not 0 <= values[0] < values[1] <= values[2] + 1e-6:
        raise ValueError("Analysis window must satisfy 0 ≤ start < end ≤ recording duration")
    return values[0], min(values[1], values[2])


def validate_review_exclusions(exclusions: list[dict[str, Any]], *, duration_sec: float) -> list[dict[str, Any]]:
    checked = []
    for item in exclusions:
        start, end = validate_analysis_window(float(item["start_sec"]), float(item["end_sec"]), duration_sec)
        reason = str(item.get("exclusion_reason", "")).strip()
        if reason not in EXCLUSION_REASONS:
            raise ValueError(f"Choose a valid exclusion reason for {start:.3f}–{end:.3f}s")
        checked.append({"start_sec": start, "end_sec": end, "exclusion_reason": reason,
                        "notes": str(item.get("notes", "")).strip()})
    checked.sort(key=lambda item: item["start_sec"])
    if any(right["start_sec"] < left["end_sec"] - 1e-9 for left, right in zip(checked, checked[1:])):
        raise ValueError("Manual exclusion intervals overlap")
    return checked


def _reviewed_plot(paths: dict[str, Path], row: pd.Series, decision: str,
                   source: str, intervals: list[tuple[float, float]],
                   exclusions: list[dict[str, Any]], start: float | None,
                   end: float | None, reviewer: str) -> Path | None:
    """Render a live review artifact without touching automatic plots."""
    wav = Path(str(row.analysis_wav_path)) if row.analysis_wav_path else None
    if wav is None or not wav.is_file():
        return None
    try:
        x, sr = _read_canonical_audio(wav)
    except Exception:
        if decision == "EXCLUDE":
            return None
        raise
    duration = len(x) / sr
    automatic = (_intervals_from_segments(row.automatic_segments_path)
                 if row.automatic_segments_path and Path(row.automatic_segments_path).is_file() else [])
    selected = [Interval(a, b) for a, b in intervals] if decision != "EXCLUDE" else []
    if selected and start is not None and end is not None:
        try:
            resolved, _ = resolve_reviewed_intervals(selected, duration_sec=duration,
                analysis_start_sec=start, analysis_end_sec=end, exclusions=exclusions,
                boundary_source=source, automatic_intervals=automatic)
            selected = [Interval(item["start_sec"], item["end_sec"]) for item in resolved]
        except ValueError:
            pass  # A pending edit can remain unresolved until its decision is saved.
    base = f"{safe_stem(row.file_name)}__{str(row.recording_id)[:8]}"
    plot = paths["stage"] / "plots" / "reviewed" / f"{base}__reviewed.png"
    plot.parent.mkdir(parents=True, exist_ok=True)
    temporary = plot.with_name(f".{plot.stem}.{uuid4().hex}.png")
    try:
        _plot(temporary, x, sr, row.segmentation_method, selected, decision,
              [str(row.automatic_flags)] if row.automatic_flags else [],
              automatic_intervals=automatic if source != "AUTO" else None,
              analysis_window=(start, end) if start is not None and end is not None else None,
              excluded_intervals=[(float(item["start_sec"]), float(item["end_sec"])) for item in exclusions],
              file_name=str(row.file_name),
              parameter_profile=f"{row.segmentation_parameters} | run={row.segmentation_run_id} | "
                                f"source={source} | reviewer={reviewer or 'pending'}")
        os.replace(temporary, plot)
    finally:
        temporary.unlink(missing_ok=True)
    return plot


def _materialize_review_tables(output_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild working CSV views from committed per-action JSON events."""
    paths = _paths(output_root)
    decisions, overrides = load_review_state(output_root)
    exclusions = load_review_exclusions(output_root)
    if decisions.duplicated(["segmentation_run_id", "recording_id"]).any():
        raise ValueError("Duplicate current-state review decision")
    _atomic_csv(decisions[REVIEW_COLUMNS], paths["decisions"])
    _atomic_csv(overrides[OVERRIDE_COLUMNS], paths["overrides"])
    _atomic_csv(exclusions[EXCLUSION_COLUMNS], paths["exclusions"])
    reviewed = decisions.loc[decisions.last_modified_utc.ne("")]
    windows = pd.DataFrame([{
        "segmentation_run_id": row.segmentation_run_id, "recording_id": row.recording_id,
        "file_name": row.file_name, "analysis_start_sec": row.analysis_start_sec,
        "analysis_end_sec": row.analysis_end_sec, "reviewer": row.reviewer,
        "review_date": row.review_date, "notes": row.review_notes,
        "last_modified_utc": row.last_modified_utc}
        for row in reviewed.itertuples()], columns=WINDOW_COLUMNS)
    _atomic_csv(windows, paths["windows"])
    audit = pd.DataFrame([entry["audit"] for entry in _review_entries(output_root)
                          if "audit" in entry], columns=AUDIT_COLUMNS)
    if len(audit):
        _atomic_csv(audit, paths["audit"])
    verified = _read_csv(paths["decisions"], REVIEW_COLUMNS)
    if list(verified.columns) != REVIEW_COLUMNS or verified.duplicated(
            ["segmentation_run_id", "recording_id"]).any():
        raise RuntimeError("Review decision projection failed verification")
    return decisions, overrides


def save_segmentation_review_entry(output_root: str | Path, recording_id: str,
                                   final_decision: str, reviewer: str,
                                   review_notes: str = "", manual_intervals_text: str = "",
                                   review_date: str | None = None, *,
                                   analysis_start_sec: float | None = None,
                                   analysis_end_sec: float | None = None,
                                   exclusion_intervals: list[dict[str, Any]] | None = None,
                                   action: str | None = None,
                                   recording_exclusion_reason: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Commit a review action, then verify its working CSV projections."""
    paths = _paths(output_root)
    if paths["final_manifest"].exists():
        raise FileExistsError("Final segmentation is frozen; create a new run for revisions")
    decisions, overrides = load_review_state(output_root)
    selected = decisions.recording_id.eq(str(recording_id))
    if int(selected.sum()) != 1:
        raise ValueError(f"Expected one recording_id={recording_id}; found {int(selected.sum())}")
    row = decisions.loc[selected].iloc[0]
    old_value = {key: str(row.get(key, "")) for key in REVIEW_COLUMNS}
    decision = final_decision.strip().upper()
    if decision not in {"PENDING", "KEEP_AUTO", "KEEP_MANUAL", "EXCLUDE"}:
        raise ValueError("Decision must be PENDING, KEEP_AUTO, KEEP_MANUAL, or EXCLUDE")
    if row.automatic_status == "FAILED" and decision not in {"PENDING", "EXCLUDE"}:
        raise ValueError("Computational FAILED recordings cannot receive intervals")
    reviewer, notes = reviewer.strip(), review_notes.strip()
    if decision != "PENDING" and not reviewer:
        raise ValueError("Reviewer name is required for a saved decision")
    if decision in {"KEEP_MANUAL", "EXCLUDE"} and not notes:
        raise ValueError("Manual changes and exclusions require review notes")
    when = review_date or date.today().isoformat()
    date.fromisoformat(when)
    now = datetime.now(timezone.utc).isoformat()
    try:
        duration = float(row.duration_sec)
        if not np.isfinite(duration) or duration <= 0:
            duration = None
    except (TypeError, ValueError):
        duration = None
    if decision == "EXCLUDE":
        start, end, exclusions = 0.0, duration, []
    else:
        if duration is None:
            if decision != "PENDING":
                raise ValueError("Audio duration unavailable for boundary decision")
            start, end, exclusions = None, None, []
        else:
            start, end = validate_analysis_window(
                0.0 if analysis_start_sec is None else analysis_start_sec,
                duration if analysis_end_sec is None else analysis_end_sec, duration)
            exclusions = validate_review_exclusions(exclusion_intervals or [], duration_sec=duration)
            if any(item["start_sec"] < start or item["end_sec"] > end for item in exclusions):
                raise ValueError("Manual exclusion intervals must lie within the analysis window")
            if decision != "PENDING" and (start > 0 or end < duration or exclusions) and not notes:
                raise ValueError("Analysis trimming and exclusions require review notes")
    if decision == "EXCLUDE":
        reason = recording_exclusion_reason.strip()
        if not reason:
            reason = ("Preprocess not accepted" if "preprocess_not_accepted" in str(row.automatic_flags)
                      else "Unusable audio" if row.automatic_status == "FAILED"
                      else "Segmentation excluded" if row.automatic_status == "EXCLUDED" else "Other")
        if reason not in RECORDING_EXCLUSION_REASONS:
            raise ValueError("Choose a valid recording exclusion reason")
    else:
        reason = ""
    current_overrides = overrides.loc[overrides.recording_id.eq(str(recording_id))].copy()
    manual_action = decision == "KEEP_MANUAL" or action in {
        "ADD_INTERVAL", "DELETE_INTERVAL", "EDIT_BOUNDARIES", "RESET_TO_AUTO"}
    if manual_action:
        if duration is None:
            raise ValueError("Audio duration unavailable for manual intervals")
        intervals = validate_manual_interval_list(parse_manual_intervals_text(manual_intervals_text),
                                                  duration_sec=duration)
        if action == "RESET_TO_AUTO" or decision == "KEEP_AUTO":
            current_overrides = current_overrides.iloc[0:0].copy()
        else:
            current_overrides = pd.DataFrame([{
                "recording_id": recording_id, "file_name": row.file_name,
                "segmentation_run_id": row.segmentation_run_id, "segment_index": index,
                "start_sec": left, "end_sec": right, "duration_sec": right - left,
                "boundary_source": "MANUAL", "reviewer": reviewer,
                "review_date": when, "review_notes": notes, "notes": notes}
                for index, (left, right) in enumerate(intervals)], columns=OVERRIDE_COLUMNS)
    elif decision == "KEEP_AUTO":
        current_overrides = current_overrides.iloc[0:0].copy()
    if decision == "KEEP_AUTO":
        auto_path = Path(row.automatic_segments_path)
        if not auto_path.is_file() or pd.read_csv(auto_path).segment_type.eq("speech").sum() == 0:
            raise ValueError("Automatic boundaries have no speech support; edit manually or exclude")
    if decision == "KEEP_MANUAL" and current_overrides.empty:
        raise ValueError("KEEP_MANUAL requires at least one interval")
    if decision == "EXCLUDE":
        source = "NONE"
    elif decision == "KEEP_MANUAL" or not current_overrides.empty:
        source = "MANUAL"
    else:
        source = "AUTO_MODIFIED" if duration is not None and (start > 0 or end < duration or exclusions) else "AUTO"
    decisions.loc[selected, "final_decision"] = decision
    decisions.loc[selected, "review_status"] = "SAVED" if decision != "PENDING" else "DRAFT"
    decisions.loc[selected, "final_decision_source"] = (
        "MANUAL_CONFIRMATION" if decision == "EXCLUDE" else
        "MANUAL_REVIEW" if decision != "PENDING" else "")
    decisions.loc[selected, "recording_exclusion_reason"] = reason
    decisions.loc[selected, "boundary_source"] = source
    decisions.loc[selected, "manual_override_applied"] = str(not current_overrides.empty)
    decisions.loc[selected, "reviewer"] = reviewer
    decisions.loc[selected, "review_date"] = when if reviewer else ""
    decisions.loc[selected, "review_notes"] = notes
    decisions.loc[selected, "analysis_start_sec"] = "" if start is None else str(start)
    decisions.loc[selected, "analysis_end_sec"] = "" if end is None else str(end)
    decisions.loc[selected, "manual_exclusion_applied"] = str(bool(exclusions))
    decisions.loc[selected, "last_modified_utc"] = now
    current_exclusions = pd.DataFrame([{
        "recording_id": recording_id, "file_name": row.file_name,
        "segmentation_run_id": row.segmentation_run_id, "exclusion_index": index,
        "start_sec": item["start_sec"], "end_sec": item["end_sec"],
        "duration_sec": item["end_sec"] - item["start_sec"],
        "exclusion_reason": item["exclusion_reason"], "reason": item["exclusion_reason"],
        "reviewer": reviewer, "review_date": when, "notes": item["notes"]}
        for index, item in enumerate(exclusions)], columns=EXCLUSION_COLUMNS)
    chosen_intervals = ([(float(r.start_sec), float(r.end_sec)) for r in current_overrides.itertuples()]
                        if not current_overrides.empty else
                        [(p.start_sec, p.end_sec) for p in _intervals_from_segments(row.automatic_segments_path)]
                        if row.automatic_segments_path and Path(row.automatic_segments_path).is_file() else [])
    plot = _reviewed_plot(paths, row, decision, source, chosen_intervals, exclusions, start, end, reviewer)
    decisions.loc[selected, "reviewed_plot_path"] = str(plot) if plot else ""
    new_value = {key: str(decisions.loc[selected, key].iloc[0]) for key in REVIEW_COLUMNS}
    event_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex
    audit = {"event_id": event_id, "timestamp_utc": now,
             "segmentation_run_id": str(row.segmentation_run_id),
             "recording_id": str(recording_id), "file_name": str(row.file_name),
             "reviewer": reviewer,
             "action": action or {"KEEP_AUTO": "KEEP_AUTO", "KEEP_MANUAL": "SAVE_MANUAL",
                                   "EXCLUDE": "EXCLUDE_RECORDING", "PENDING": "EDIT_BOUNDARIES"}[decision],
             "old_value_json": json.dumps(old_value), "new_value_json": json.dumps(new_value),
             "notes": notes}
    event = {"recording_id": str(recording_id),
             "segmentation_run_id": str(row.segmentation_run_id),
             "decision": decisions.loc[selected, REVIEW_COLUMNS].iloc[0].to_dict(),
             "overrides": current_overrides[OVERRIDE_COLUMNS].to_dict("records"),
             "exclusions": current_exclusions[EXCLUSION_COLUMNS].to_dict("records"),
             "audit": audit}
    _atomic_json(event, paths["events"] / f"{event_id}.json")
    current, manual = _materialize_review_tables(output_root)
    verified = _read_csv(paths["decisions"], REVIEW_COLUMNS)
    saved = verified.loc[verified.recording_id.eq(str(recording_id))]
    if len(saved) != 1 or saved.iloc[0].last_modified_utc != now:
        raise RuntimeError("Saved review was not found in the working decision table")
    return current, manual


def _intervals_from_segments(path: str) -> list[Interval]:
    segments = pd.read_csv(path)
    return [Interval(float(row.start_sec), float(row.end_sec)) for row in
            segments.loc[segments.segment_type.eq("speech")].itertuples()]


def preview_manual_segmentation(output_root: str | Path, recording_id: str,
                                manual_intervals_text: str) -> Path:
    decisions, _ = load_review_state(output_root)
    selected = decisions.loc[decisions.recording_id.eq(str(recording_id))]
    if len(selected) != 1:
        raise ValueError(f"Unknown recording ID: {recording_id}")
    row = selected.iloc[0]
    if row.automatic_status == "FAILED":
        raise ValueError("Computational FAILED recordings cannot be edited")
    x, sr = _read_canonical_audio(Path(row.analysis_wav_path))
    manual = [Interval(a, b) for a, b in validate_manual_interval_list(
        parse_manual_intervals_text(manual_intervals_text), duration_sec=len(x) / sr)]
    automatic = _intervals_from_segments(row.automatic_segments_path) if Path(row.automatic_segments_path).is_file() else []
    base = str(recording_id).replace("/", "_").replace("\\", "_")
    path = _paths(output_root)["stage"] / "plots" / "reviewed" / f"{base}__manual_preview.png"
    views = _final_views(manual, len(x) / sr)
    frames = _frames(x, sr, {**views, "raw_speech": manual}, 30)
    _plot(path, x, sr, row.segmentation_method, manual, "MANUAL PREVIEW", [],
          trace=_review_trace(frames), automatic_intervals=automatic,
          trace_label="RMS support")
    return path


def _final_views(intervals: list[Interval], duration: float,
                 timeline: pd.DataFrame | None = None) -> dict[str, list[Interval]]:
    pauses = (internal_nonspeech(intervals, duration) if timeline is None else [
        Interval(float(row.start_sec), float(row.end_sec)) for row in
        timeline.loc[timeline.segment_role.eq("internal_nonspeech")].itertuples()])
    return {"primary_speech": intervals, "strict_speech": erode_intervals(intervals, .05),
            "strict_internal_nonspeech": erode_intervals(pauses, .2)}


def _review_trace(frames: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve the support panel on reviewed speech plots."""
    times = frames.mid_sec.to_numpy(dtype=float)
    rms = frames.rms.to_numpy(dtype=float)
    return times, rms, np.zeros_like(rms)


def resolve_reviewed_intervals(
    source_intervals: list[Interval], *, duration_sec: float,
    analysis_start_sec: float, analysis_end_sec: float,
    exclusions: list[dict[str, Any]] | None = None,
    boundary_source: str = "AUTO", automatic_intervals: list[Interval] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Resolve exact speech intervals, then partition the analysis timeline by role."""
    start, end = validate_analysis_window(analysis_start_sec, analysis_end_sec, duration_sec)
    checked = validate_review_exclusions(exclusions or [], duration_sec=duration_sec)
    ordered = validate_manual_interval_list(
        [(item.start_sec, item.end_sec) for item in source_intervals], duration_sec=duration_sec)
    speech: list[dict[str, Any]] = []
    for source_id, (left, right) in enumerate(ordered):
        left, right = max(left, start), min(right, end)
        if right <= left:
            continue
        pieces = [(left, right)]
        for excluded in checked:
            following = []
            for a, b in pieces:
                cut_a, cut_b = excluded["start_sec"], excluded["end_sec"]
                if cut_b <= a or cut_a >= b:
                    following.append((a, b))
                else:
                    if cut_a > a:
                        following.append((a, cut_a))
                    if cut_b < b:
                        following.append((cut_b, b))
            pieces = following
        for a, b in pieces:
            auto_id: int | str = source_id if boundary_source.startswith("AUTO") else ""
            if automatic_intervals is not None and boundary_source == "MANUAL":
                overlap = [max(0.0, min(b, auto.end_sec) - max(a, auto.start_sec))
                           for auto in automatic_intervals]
                if overlap and max(overlap) > 0:
                    auto_id = int(np.argmax(overlap))
            speech.append({"start_sec": a, "end_sec": b, "automatic_source_interval_id": auto_id})
    speech.sort(key=lambda item: item["start_sec"])
    if not speech:
        raise ValueError("Review corrections leave no patient-speech interval; exclude the recording")

    cuts = sorted({0.0, float(duration_sec), start, end,
                   *(value for item in speech for value in (item["start_sec"], item["end_sec"])),
                   *(value for item in checked for value in (max(start, min(end, item["start_sec"])),
                                                               max(start, min(end, item["end_sec"]))))})
    rows = []
    for a, b in zip(cuts, cuts[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2
        active = next((item for item in speech if item["start_sec"] <= mid < item["end_sec"]), None)
        contaminated = next((item for item in checked if item["start_sec"] <= mid < item["end_sec"]), None)
        if mid < start or mid >= end:
            kind, role, reason, auto_id = "outside_analysis", "outside_analysis_window", "", ""
        elif contaminated:
            kind, role, reason, auto_id = "excluded", "manual_exclusion", contaminated["exclusion_reason"], ""
        elif active:
            kind, role, reason, auto_id = "speech", "speech", "", active["automatic_source_interval_id"]
        elif mid < speech[0]["start_sec"]:
            kind, role, reason, auto_id = "nonspeech", "leading_nonspeech", "", ""
        elif mid >= speech[-1]["end_sec"]:
            kind, role, reason, auto_id = "nonspeech", "trailing_nonspeech", "", ""
        else:
            kind, role, reason, auto_id = "nonspeech", "internal_nonspeech", "", ""
        if rows and rows[-1]["segment_type"] == kind and rows[-1]["segment_role"] == role and rows[-1]["manual_exclusion_reason"] == reason and rows[-1]["automatic_source_interval_id"] == auto_id:
            rows[-1]["end_sec"] = b
            rows[-1]["duration_sec"] = b - rows[-1]["start_sec"]
        else:
            rows.append({"segment_type": kind, "segment_role": role, "start_sec": a,
                         "end_sec": b, "duration_sec": b - a,
                         "manual_exclusion_reason": reason,
                         "automatic_source_interval_id": auto_id})
    return speech, pd.DataFrame(rows)


def finalize_segmentation_review(output_root: str | Path) -> StageResult:
    """Freeze final decisions, intervals, and reviewed artifacts exactly once."""
    paths = _paths(output_root)
    if paths["final_manifest"].exists():
        raise FileExistsError("Final segmentation already exists and is immutable")
    decisions, overrides = load_review_state(output_root)
    saved_exclusions = load_review_exclusions(output_root)
    if not _pending(decisions).empty:
        raise ValueError(f"{len(_pending(decisions))} required segmentation reviews remain pending")
    persisted = _read_csv(paths["decisions"], REVIEW_COLUMNS)
    if not persisted.fillna("").equals(decisions[REVIEW_COLUMNS].fillna("")):
        raise RuntimeError("Working review table differs from committed review actions")
    interval_rows: list[dict[str, Any]] = []
    final_rows = []
    for _, row in decisions.iterrows():
        record = row.to_dict()
        record["review_run_id"] = paths["review_run_id"]
        record["final_intervals_path"] = str(paths["final_intervals"])
        decision = row.final_decision
        if row.automatic_status == "FAILED":
            record["final_decision"] = "FAILED"
            record["boundary_source"] = "NONE"
            final_rows.append(record)
            continue
        if decision == "EXCLUDE":
            record["boundary_source"] = "NONE"
            final_rows.append(record)
            continue
        if decision not in {"KEEP_AUTO", "KEEP_MANUAL"}:
            raise ValueError(f"Unresolved decision for {row.file_name}")
        if _boolean(row.review_required) and not row.reviewer:
            raise ValueError(f"Reviewer identity missing for {row.file_name}")
        x, sr = _read_canonical_audio(Path(row.analysis_wav_path))
        duration = len(x) / sr
        if decision == "KEEP_MANUAL":
            group = overrides.loc[overrides.recording_id.eq(row.recording_id)].sort_values("segment_index")
            intervals = validate_manual_interval_list([(float(r.start_sec), float(r.end_sec)) for r in group.itertuples()],
                                                       duration_sec=duration)
            if not intervals or group.reviewer.ne(row.reviewer).any():
                raise ValueError(f"Manual interval provenance mismatch for {row.file_name}")
            primary = [Interval(start, end) for start, end in intervals]
            source = "MANUAL"
            if row.segmentation_method == "sustained_phonation":
                record["full_phonation_start"] = primary[0].start_sec
                record["full_phonation_end"] = primary[-1].end_sec
                record["stable_region_start"] = ""
                record["stable_region_end"] = ""
        else:
            primary = _intervals_from_segments(row.automatic_segments_path)
            if not primary:
                raise ValueError(f"KEEP_AUTO has no speech intervals for {row.file_name}")
            source = "AUTO"
        analysis_start, analysis_end = validate_analysis_window(
            float(row.analysis_start_sec or 0), float(row.analysis_end_sec or duration), duration)
        excluded_rows = saved_exclusions.loc[saved_exclusions.recording_id.eq(row.recording_id)]
        exclusion_items = excluded_rows[["start_sec", "end_sec", "exclusion_reason", "notes"]].to_dict("records")
        automatic = _intervals_from_segments(row.automatic_segments_path) if Path(row.automatic_segments_path).is_file() else []
        source = row.boundary_source or source
        resolved, timeline = resolve_reviewed_intervals(
            primary, duration_sec=duration, analysis_start_sec=analysis_start,
            analysis_end_sec=analysis_end, exclusions=exclusion_items,
            boundary_source=source, automatic_intervals=automatic)
        primary = [Interval(item["start_sec"], item["end_sec"]) for item in resolved]
        if row.segmentation_method == "ddk_energy" and primary:
            if source != "AUTO" or exclusion_items or analysis_start > 0 or analysis_end < duration:
                centers = np.array([(item.start_sec + item.end_sec) / 2 for item in primary])
                cycles = np.diff(centers)
                sequence_duration = primary[-1].end_sec - primary[0].start_sec
                record.update({"n_events": len(primary), "n_syllables": len(primary),
                    "ddk_sequence_start_sec": primary[0].start_sec,
                    "ddk_sequence_end_sec": primary[-1].end_sec,
                    "ddk_sequence_duration_sec": sequence_duration,
                    "ddk_rate_hz": len(primary) / sequence_duration if sequence_duration > 0 else "",
                    "ctv_sec": np.mean(np.abs(np.diff(cycles))) if len(cycles) > 1 else "",
                    "cycle_mean_sec": np.mean(cycles) if len(cycles) else "",
                    "cycle_sd_sec": np.std(cycles) if len(cycles) else "",
                    "ddk_timing_source": "reviewed_interval_midpoint"})
        if row.segmentation_method == "sustained_phonation":
            record["full_phonation_start"] = primary[0].start_sec
            record["full_phonation_end"] = primary[-1].end_sec
            stable_start = record.get("stable_region_start", "")
            stable_end = record.get("stable_region_end", "")
            if not stable_start or not stable_end or not any(
                interval.start_sec <= float(stable_start) < float(stable_end) <= interval.end_sec
                for interval in primary
            ):
                record["stable_region_start"] = ""
                record["stable_region_end"] = ""
        views = _final_views(primary, duration, timeline)
        record["analysis_start_sec"] = analysis_start
        record["analysis_end_sec"] = analysis_end
        for index, segment in timeline.iterrows():
            interval_rows.append({"recording_id": row.recording_id, "file_name": row.file_name,
                "task_name": row.task_name, "segmentation_method": row.segmentation_method,
                "segmentation_run_id": row.segmentation_run_id,
                "review_run_id": paths["review_run_id"],
                "segmentation_parameters": row.segmentation_parameters,
                "view": "authoritative", "interval_index": index, "start_sec": segment.start_sec,
                "end_sec": segment.end_sec, "duration_sec": segment.duration_sec,
                "boundary_source": source, "reviewer": row.reviewer,
                "review_date": row.review_date, "review_notes": row.review_notes,
                "segment_type": segment.segment_type, "segment_role": segment.segment_role,
                "analysis_start_sec": analysis_start, "analysis_end_sec": analysis_end,
                "manual_exclusion_applied": segment.segment_role == "manual_exclusion",
                "manual_exclusion_reason": segment.manual_exclusion_reason,
                "automatic_source_interval_id": segment.automatic_source_interval_id})
        base = f"{safe_stem(row.file_name)}__{str(row.recording_id)[:8]}"
        segments_path = paths["final_decisions"].parent / "segments" / f"{base}__final_segments.csv"
        frames_path = paths["final_decisions"].parent / "frames" / f"{base}__final_frames.csv"
        segments_path.parent.mkdir(parents=True, exist_ok=True)
        frames_path.parent.mkdir(parents=True, exist_ok=True)
        timeline.to_csv(segments_path, index=False)
        frames = _frames(x, sr, {**views, "raw_speech": primary}, 30)
        frames.to_csv(frames_path, index=False)
        record["final_segments_path"] = str(segments_path)
        record["segments_csv_path"] = str(segments_path)
        record["frame_csv_path"] = str(frames_path)
        record["manual_segments_path"] = str(segments_path) if source == "MANUAL" else ""
        record["boundary_source"] = source
        record["manual_override_applied"] = str(source == "MANUAL")
        plot_path = paths["final_decisions"].parent / "plots" / f"{base}__final.png"
        if row.reviewed_plot_path and Path(row.reviewed_plot_path).is_file():
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row.reviewed_plot_path, plot_path)
        if not plot_path.is_file():
            _plot(plot_path, x, sr, row.segmentation_method, primary,
                  decision, [row.automatic_flags] if row.automatic_flags else [],
                  trace=_review_trace(frames),
                  automatic_intervals=automatic if source != "AUTO" else None,
                  trace_label="RMS support", analysis_window=(analysis_start, analysis_end),
                  excluded_intervals=[(float(item["start_sec"]), float(item["end_sec"])) for item in exclusion_items],
                  file_name=str(row.file_name),
                  parameter_profile=f"{row.segmentation_parameters} | run={row.segmentation_run_id} | reviewer={row.reviewer}")
        record["plot_path"] = str(plot_path)
        record["reviewed_plot_path"] = str(plot_path)
        final_rows.append(record)
    frozen = pd.DataFrame(final_rows, columns=[*REVIEW_COLUMNS, "segments_csv_path"])
    intervals = pd.DataFrame(interval_rows, columns=INTERVAL_COLUMNS)
    _atomic_csv(intervals, paths["final_intervals"])
    # The decisions file is the freeze marker. Write it last so an interrupted
    # materialization cannot appear finalized to downstream stages.
    _atomic_csv(frozen, paths["final_decisions"])
    counts = {"n_total": len(frozen),
              "n_keep_auto": int(frozen.final_decision.eq("KEEP_AUTO").sum()),
              "n_keep_manual": int(frozen.final_decision.eq("KEEP_MANUAL").sum()),
              "n_excluded": int(frozen.final_decision.eq("EXCLUDE").sum()),
              "n_failed": int(frozen.final_decision.eq("FAILED").sum()),
              "n_manually_reviewed": int(frozen.reviewer.ne("").sum()),
              "n_automatic_accepted_untouched": int((frozen.automatic_status.eq("ACCEPTED") & frozen.reviewer.eq("")).sum())}
    _atomic_csv(pd.DataFrame([counts]), paths["final_summary"])
    _atomic_json({"source_segmentation_run_id": str(frozen.segmentation_run_id.iloc[0]) if len(frozen) else "",
                  "review_run_id": paths["review_run_id"],
                  "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "automatic_segmentation_method": sorted(frozen.segmentation_method.unique().tolist()),
                  "automatic_parameters": sorted(frozen.segmentation_parameters.unique().tolist()),
                  **counts, "decisions_sha256": sha256_file(paths["final_decisions"]),
                  "intervals_sha256": sha256_file(paths["final_intervals"])}, paths["final_manifest"])
    _atomic_json({"stage": "segmentation_manual_review", "status": "completed",
                  "review_run_id": paths["review_run_id"],
                  "final_manifest_path": str(paths["final_manifest"]),
                  "registry_path": str(paths["registry"])}, paths["stage_manifest"])
    return StageResult(status="completed", manifest_path=paths["final_manifest"],
                       summary_table=paths["final_decisions"], report_path=None)


def load_final_segmentation(final_intervals_csv: str | Path,
                            final_decisions_csv: str | Path) -> pd.DataFrame:
    """Downstream contract: validate frozen intervals before returning kept rows."""
    intervals = pd.read_csv(final_intervals_csv)
    decisions = pd.read_csv(final_decisions_csv, keep_default_na=False)
    if decisions.recording_id.duplicated().any():
        raise ValueError("Duplicate frozen segmentation decision IDs")
    kept = decisions.loc[decisions.final_decision.isin(["KEEP_AUTO", "KEEP_MANUAL"])].copy()
    for row in kept.itertuples():
        support = intervals.loc[intervals.recording_id.astype(str).eq(str(row.recording_id))
                                & intervals.segment_role.eq("speech")]
        if support.empty or not support.boundary_source.eq(row.boundary_source).all():
            raise ValueError(f"Frozen interval support or provenance missing for {row.recording_id}")
        if not Path(row.final_segments_path).is_file() or not Path(row.frame_csv_path).is_file():
            raise FileNotFoundError(f"Reviewed segments/frames missing for {row.recording_id}")
        segment_intervals = _intervals_from_segments(row.final_segments_path)
        table_intervals = [(float(item.start_sec), float(item.end_sec)) for item in support.sort_values("interval_index").itertuples()]
        if len(segment_intervals) != len(table_intervals) or any(
            abs(segment.start_sec - start) > 1e-6 or abs(segment.end_sec - end) > 1e-6
            for segment, (start, end) in zip(segment_intervals, table_intervals, strict=True)
        ):
            raise ValueError(f"Frozen interval and segment boundaries disagree for {row.recording_id}")
    kept["segments_csv_path"] = kept.final_segments_path
    kept["status"] = "ok"
    return kept
