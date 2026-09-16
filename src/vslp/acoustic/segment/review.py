"""Persistent manual segmentation review and immutable final boundary freeze.

Adapted from quality_framework_features segmentation_review.py and segmentation.py.
No study metadata or metadata-based exclusion gates are used here.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from vslp.acoustic.segment.pipeline import _frames, _plot, _segments
from vslp.acoustic.segment.silero_reference import Interval, erode_intervals, internal_nonspeech
from vslp.acoustic.segment.stage import _read_canonical_audio
from vslp.core.provenance import python_environment, sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


REVIEW_COLUMNS = [
    "recording_id", "file_name", "task_name", "segmentation_method", "automatic_status",
    "automatic_flags", "review_required", "duration_sec", "analysis_wav_path",
    "automatic_segments_path", "automatic_frames_path", "automatic_boundaries_path",
    "automatic_plot_path", "source_sha256", "project_name", "run_id",
    "run_created_at_local", "run_created_at_utc", "source_file_path",
    "full_phonation_start", "full_phonation_end", "stable_region_start", "stable_region_end",
    "n_events", "ddk_rate_hz", "cycle_mean_sec", "cycle_sd_sec",
    "final_decision", "boundary_source", "manual_override_applied", "reviewer",
    "review_date", "review_notes", "manual_segments_path", "final_segments_path",
    "frame_csv_path", "segments_csv_path", "plot_path",
]
OVERRIDE_COLUMNS = ["recording_id", "file_name", "segment_index", "start_sec", "end_sec",
                    "reviewer", "review_date", "review_notes"]
INTERVAL_COLUMNS = ["recording_id", "file_name", "task_name", "segmentation_method", "view",
                    "interval_index", "start_sec", "end_sec", "duration_sec", "boundary_source",
                    "reviewer", "review_date", "review_notes"]


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
    stage = Path(output_root) / "acoustic" / "003_segmentation_review"
    tables = stage / "tables"
    return {"stage": stage, "queue": tables / "segmentation_review_queue.csv",
            "decisions": tables / "segmentation_review_decisions.csv",
            "overrides": tables / "manual_segmentation_overrides.csv",
            "final_decisions": tables / "final_segmentation_decisions.csv",
            "final_intervals": tables / "final_segmentation_intervals.csv",
            "manifest": stage / "logs" / "stage_manifest.json"}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False).reindex(columns=columns, fill_value="")


def _boolean(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _pending(decisions: pd.DataFrame) -> pd.DataFrame:
    required = decisions["review_required"].map(_boolean)
    unresolved = decisions.final_decision.eq("")
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
        return StageResult(status="completed" if paths["final_decisions"].exists() else "completed_with_warnings",
            manifest_path=paths["manifest"], summary_table=paths["queue"], report_path=None)
    summary = pd.read_csv(source, dtype=str, keep_default_na=False)
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
               "automatic_status": status, "automatic_flags": auto.get("flags", ""),
               "review_required": str(status in {"REVIEW", "EXCLUDED"} or _boolean(auto.get("review_required", False))),
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
               "n_events": auto.get("n_events", ""), "ddk_rate_hz": auto.get("ddk_rate_hz", ""),
               "cycle_mean_sec": auto.get("cycle_mean_sec", ""), "cycle_sd_sec": auto.get("cycle_sd_sec", ""),
               "final_decision": "FAILED" if status == "FAILED" else ("KEEP_AUTO" if status == "ACCEPTED" and not _boolean(auto.get("review_required", False)) else ""),
               "boundary_source": "NONE" if status == "FAILED" else ("AUTO" if status == "ACCEPTED" and not _boolean(auto.get("review_required", False)) else ""),
               "manual_override_applied": "False", "reviewer": "", "review_date": "",
               "review_notes": "", "manual_segments_path": "", "final_segments_path": "",
               "frame_csv_path": "", "segments_csv_path": "", "plot_path": ""}
        rows.append(row)
    queue = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    queue["automatic_summary_sha256"] = source_hash
    decisions = queue[REVIEW_COLUMNS].copy()
    _atomic_csv(queue, paths["queue"])
    _atomic_csv(decisions, paths["decisions"])
    _atomic_csv(pd.DataFrame(columns=OVERRIDE_COLUMNS), paths["overrides"])
    manifest = StageManifest(stage_name="acoustic_segmentation_manual_review", stage_version="1.0.0",
        status="completed_with_warnings" if not _pending(decisions).empty else "completed",
        input_artifacts=[ArtifactRef(path=str(source), role="automatic_segmentation", media_type="text/csv", sha256=source_hash)],
        output_artifacts=[ArtifactRef(path=str(paths["queue"]), role="review_queue", media_type="text/csv"),
                          ArtifactRef(path=str(paths["decisions"]), role="review_decisions", media_type="text/csv")],
        config={"automatic_summary_sha256": source_hash}, environment={"python": python_environment()},
        notes=["Automatic segmentation artifacts are immutable; review decisions are separate."])
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json(paths["manifest"])
    return StageResult(status=manifest.status, manifest_path=paths["manifest"],
                       summary_table=paths["queue"], report_path=None)


def load_review_state(output_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = _paths(output_root)
    if not paths["decisions"].is_file():
        raise FileNotFoundError("Initialize Segmentation manual review first")
    return _read_csv(paths["decisions"], REVIEW_COLUMNS), _read_csv(paths["overrides"], OVERRIDE_COLUMNS)


def save_segmentation_review_entry(output_root: str | Path, recording_id: str,
                                   final_decision: str, reviewer: str,
                                   review_notes: str = "", manual_intervals_text: str = "",
                                   review_date: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save one decision and replace that recording's manual intervals atomically."""
    paths = _paths(output_root)
    if paths["final_decisions"].exists():
        raise FileExistsError("Final segmentation is frozen; create a new run for revisions")
    decisions, overrides = load_review_state(output_root)
    selected = decisions.recording_id.eq(str(recording_id))
    if int(selected.sum()) != 1:
        raise ValueError(f"Expected one recording_id={recording_id}; found {int(selected.sum())}")
    row = decisions.loc[selected].iloc[0]
    if row.automatic_status == "FAILED":
        raise ValueError("Computational FAILED recordings cannot be manually rescued")
    decision = final_decision.strip().upper()
    if decision not in {"KEEP_AUTO", "KEEP_MANUAL", "EXCLUDE"}:
        raise ValueError("Decision must be KEEP_AUTO, KEEP_MANUAL, or EXCLUDE")
    reviewer = reviewer.strip()
    notes = review_notes.strip()
    if not reviewer:
        raise ValueError("Reviewer name is required for a saved review")
    if decision in {"KEEP_MANUAL", "EXCLUDE"} and not notes:
        raise ValueError("Manual changes and exclusions require review notes")
    when = review_date or date.today().isoformat()
    date.fromisoformat(when)
    new_overrides = overrides.loc[~overrides.recording_id.eq(str(recording_id))].copy()
    if decision == "KEEP_MANUAL":
        intervals = validate_manual_interval_list(parse_manual_intervals_text(manual_intervals_text),
                                                   duration_sec=float(row.duration_sec))
        addition = pd.DataFrame([{"recording_id": recording_id, "file_name": row.file_name,
            "segment_index": index, "start_sec": start, "end_sec": end,
            "reviewer": reviewer, "review_date": when, "review_notes": notes}
            for index, (start, end) in enumerate(intervals)], columns=OVERRIDE_COLUMNS)
        new_overrides = pd.concat([new_overrides, addition], ignore_index=True)
    elif decision == "KEEP_AUTO":
        auto_path = Path(row.automatic_segments_path)
        if not auto_path.is_file() or pd.read_csv(auto_path).segment_type.eq("speech").sum() == 0:
            raise ValueError("Automatic boundaries have no speech support; edit manually or exclude")
    decisions.loc[selected, "final_decision"] = decision
    decisions.loc[selected, "boundary_source"] = {"KEEP_AUTO": "AUTO", "KEEP_MANUAL": "MANUAL", "EXCLUDE": "NONE"}[decision]
    decisions.loc[selected, "manual_override_applied"] = str(decision == "KEEP_MANUAL")
    decisions.loc[selected, "reviewer"] = reviewer
    decisions.loc[selected, "review_date"] = when
    decisions.loc[selected, "review_notes"] = notes
    _atomic_csv(new_overrides[OVERRIDE_COLUMNS], paths["overrides"])
    _atomic_csv(decisions[REVIEW_COLUMNS], paths["decisions"])
    return decisions, new_overrides


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


def _final_views(intervals: list[Interval], duration: float) -> dict[str, list[Interval]]:
    return {"primary_speech": intervals, "strict_speech": erode_intervals(intervals, .05),
            "strict_internal_nonspeech": erode_intervals(internal_nonspeech(intervals, duration), .2)}


def _review_trace(frames: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve the support panel on reviewed speech plots."""
    times = frames.mid_sec.to_numpy(dtype=float)
    rms = frames.rms.to_numpy(dtype=float)
    return times, rms, np.zeros_like(rms)


def finalize_segmentation_review(output_root: str | Path) -> StageResult:
    """Freeze final decisions, intervals, and reviewed artifacts exactly once."""
    paths = _paths(output_root)
    if paths["final_decisions"].exists() or paths["final_intervals"].exists():
        raise FileExistsError("Final segmentation already exists and is immutable")
    decisions, overrides = load_review_state(output_root)
    if not _pending(decisions).empty:
        raise ValueError(f"{len(_pending(decisions))} required segmentation reviews remain pending")
    interval_rows: list[dict[str, Any]] = []
    final_rows = []
    stage = paths["stage"]
    for _, row in decisions.iterrows():
        record = row.to_dict()
        decision = row.final_decision
        if decision == "FAILED":
            if row.automatic_status != "FAILED":
                raise ValueError("Only computational failures may have FAILED final state")
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
            elif row.segmentation_method == "ddk_energy":
                centers = np.array([(item.start_sec + item.end_sec) / 2 for item in primary])
                cycles = np.diff(centers)
                record["n_events"] = len(primary)
                record["ddk_rate_hz"] = 1 / np.mean(cycles) if len(cycles) else ""
                record["cycle_mean_sec"] = np.mean(cycles) if len(cycles) else ""
                record["cycle_sd_sec"] = np.std(cycles) if len(cycles) else ""
        else:
            primary = _intervals_from_segments(row.automatic_segments_path)
            if not primary:
                raise ValueError(f"KEEP_AUTO has no speech intervals for {row.file_name}")
            source = "AUTO"
        views = _final_views(primary, duration)
        for view, intervals in views.items():
            for index, item in enumerate(intervals):
                interval_rows.append({"recording_id": row.recording_id, "file_name": row.file_name,
                    "task_name": row.task_name, "segmentation_method": row.segmentation_method,
                    "view": view, "interval_index": index, "start_sec": item.start_sec,
                    "end_sec": item.end_sec, "duration_sec": item.duration_sec,
                    "boundary_source": source, "reviewer": row.reviewer,
                    "review_date": row.review_date, "review_notes": row.review_notes})
        base = str(row.recording_id).replace("/", "_").replace("\\", "_")
        segments_path = stage / "tables" / "segments" / f"{base}__final_segments.csv"
        frames_path = stage / "tables" / "frames" / f"{base}__final_frames.csv"
        segments_path.parent.mkdir(parents=True, exist_ok=True)
        frames_path.parent.mkdir(parents=True, exist_ok=True)
        _segments(primary, duration).to_csv(segments_path, index=False)
        frames = _frames(x, sr, {**views, "raw_speech": primary}, 30)
        frames.to_csv(frames_path, index=False)
        record["final_segments_path"] = str(segments_path)
        record["segments_csv_path"] = str(segments_path)
        record["frame_csv_path"] = str(frames_path)
        record["manual_segments_path"] = str(segments_path) if source == "MANUAL" else ""
        record["boundary_source"] = source
        record["manual_override_applied"] = str(source == "MANUAL")
        plot_path = stage / "plots" / "reviewed" / f"{base}__reviewed.png"
        automatic = _intervals_from_segments(row.automatic_segments_path) if Path(row.automatic_segments_path).is_file() else []
        _plot(plot_path, x, sr, row.segmentation_method, primary,
              decision, [row.automatic_flags] if row.automatic_flags else [],
              trace=_review_trace(frames),
              automatic_intervals=automatic if source == "MANUAL" else None,
              trace_label="RMS support")
        record["plot_path"] = str(plot_path)
        final_rows.append(record)
    frozen = pd.DataFrame(final_rows, columns=[*REVIEW_COLUMNS, "segments_csv_path"])
    intervals = pd.DataFrame(interval_rows, columns=INTERVAL_COLUMNS)
    _atomic_csv(intervals, paths["final_intervals"])
    # The decisions file is the freeze marker. Write it last so an interrupted
    # materialization cannot appear finalized to downstream stages.
    _atomic_csv(frozen, paths["final_decisions"])
    source = paths["decisions"]
    manifest = StageManifest(stage_name="acoustic_segmentation_manual_review", stage_version="1.0.0",
        status="completed", input_artifacts=[ArtifactRef(path=str(source), role="review_decisions", media_type="text/csv", sha256=sha256_file(source))],
        output_artifacts=[ArtifactRef(path=str(paths["final_decisions"]), role="final_segmentation_decisions", media_type="text/csv"),
                          ArtifactRef(path=str(paths["final_intervals"]), role="final_segmentation_intervals", media_type="text/csv")],
        config={"frozen_at_utc": datetime.now(timezone.utc).isoformat()},
        environment={"python": python_environment()}, notes=["Automatic segmentation remains immutable."])
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json(paths["manifest"])
    return StageResult(status="completed", manifest_path=paths["manifest"],
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
                                & intervals.view.eq("primary_speech")]
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
