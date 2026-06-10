"""Artifact inspection helpers for the kinematics GUI.

The inspector is intentionally read-only. It inventories pipeline outputs so the GUI
can show what exists, what is missing, and which tables are safe to preview.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    label: str
    folder: str
    required_outputs: tuple[str, ...]
    optional_outputs: tuple[str, ...] = ()


KINEMATICS_STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(
        "000_ingest",
        "Setup / Ingest",
        "000_ingest",
        ("tables/video_ingest_manifest.csv",),
        ("tables/video_ingest_summary.csv", "tables/video_ingest_manifest.json", "logs/video_ingest.log"),
    ),
    StageSpec(
        "001_metadata",
        "Metadata",
        "001_metadata",
        (),
        ("tables/metadata_linkage.csv", "tables/metadata_linkage.json"),
    ),
    StageSpec(
        "002_landmarks",
        "Face landmarks",
        "002_landmarks",
        ("tables/landmarks_manifest.csv",),
        ("tables/landmark_extraction_plan.csv", "tables/landmark_config.json", "tables/landmarks_manifest.json"),
    ),
    StageSpec(
        "003_selection",
        "Landmark selection",
        "003_selection",
        ("tables/selected_landmarks.json",),
        ("tables/selected_landmark_summary.csv", "tables/selected_landmark_requirements.csv"),
    ),
    StageSpec(
        "004_normalization",
        "Normalization",
        "004_normalization",
        ("tables/normalized_landmarks_manifest.csv",),
        ("tables/normalization_summary.csv", "tables/normalization_config.json"),
    ),
    StageSpec(
        "005_video_qc",
        "Video / landmark QC",
        "005_video_qc",
        ("tables/landmark_video_qc_summary.csv",),
        ("tables/landmark_video_qc_summary.json", "tables/video_qc_framework_placeholder.csv", "tables/video_qc_framework_placeholder.json"),
    ),
    StageSpec(
        "006_features",
        "Feature computation",
        "006_features",
        ("tables/kinematic_features.csv",),
        ("tables/kinematic_feature_framework.csv", "tables/kinematic_feature_implementation_audit.csv"),
    ),
    StageSpec(
        "007_aggregation",
        "Temporal aggregation",
        "007_aggregation",
        ("tables/kinematic_aggregated_features.csv",),
        ("tables/aggregation_guide.csv", "tables/aggregation_guide.json"),
    ),
    StageSpec(
        "008_inspector",
        "Inspector",
        "008_inspector",
        (),
        ("tables/artifact_inventory.csv", "tables/stage_status.csv"),
    ),
    StageSpec(
        "009_reports",
        "Reports & outputs",
        "009_reports",
        (),
        ("kinematics_gui_workflow_outline_report.html", "kinematics_pipeline_summary_report.html", "report_manifest.json"),
    ),
)


_PREVIEW_EXTENSIONS = {".csv", ".json", ".html", ".txt", ".log", ".md"}
_TABLE_EXTENSIONS = {".csv"}


def _kin_root(output_root: Path) -> Path:
    output_root = Path(output_root).expanduser().resolve()
    return output_root if output_root.name == "kinematics" else output_root / "kinematics"


def _iso_mtime(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return ""


def _csv_shape(path: Path) -> tuple[int | None, int | None]:
    try:
        df = pd.read_csv(path, nrows=5000)
        rows = len(df)
        # Count all rows cheaply if file might be longer than nrows.
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            total_lines = sum(1 for _ in handle)
        rows = max(0, total_lines - 1)
        return rows, len(df.columns)
    except Exception:
        return None, None


def stage_status_dataframe(output_root: Path) -> pd.DataFrame:
    """Return one row per kinematics stage with required/optional output state."""
    root = _kin_root(output_root)
    rows: list[dict[str, object]] = []
    for spec in KINEMATICS_STAGE_SPECS:
        stage_dir = root / spec.folder
        required = [stage_dir / rel for rel in spec.required_outputs]
        optional = [stage_dir / rel for rel in spec.optional_outputs]
        required_existing = [p for p in required if p.exists()]
        optional_existing = [p for p in optional if p.exists()]
        if spec.required_outputs:
            if len(required_existing) == len(required):
                status = "complete"
            elif required_existing:
                status = "partial"
            else:
                status = "missing"
        else:
            status = "available" if stage_dir.exists() or optional_existing else "not_run"
        rows.append({
            "stage_id": spec.stage_id,
            "stage": spec.label,
            "status": status,
            "stage_folder": str(stage_dir),
            "required_outputs": "; ".join(spec.required_outputs),
            "required_present": len(required_existing),
            "required_expected": len(required),
            "optional_present": len(optional_existing),
            "missing_required": "; ".join(str(p) for p in required if not p.exists()),
            "latest_modified_utc": max([_iso_mtime(p) for p in required_existing + optional_existing] or [""]),
        })
    return pd.DataFrame(rows)


def artifact_inventory_dataframe(output_root: Path) -> pd.DataFrame:
    """Return a readable inventory of kinematics output artifacts."""
    root = _kin_root(output_root)
    rows: list[dict[str, object]] = []
    if not root.exists():
        return pd.DataFrame(columns=[
            "stage_id", "stage", "artifact_type", "filename", "relative_path", "size_kb",
            "modified_utc", "previewable", "rows", "columns",
        ])
    stage_lookup = {spec.folder: spec for spec in KINEMATICS_STAGE_SPECS}
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True):
        rel = path.relative_to(root)
        folder = rel.parts[0] if rel.parts else ""
        spec = stage_lookup.get(folder)
        suffix = path.suffix.lower()
        rows_count: int | None = None
        cols_count: int | None = None
        if suffix in _TABLE_EXTENSIONS:
            rows_count, cols_count = _csv_shape(path)
        if suffix in {".csv"}:
            artifact_type = "table"
        elif suffix in {".json"}:
            artifact_type = "config/manifest"
        elif suffix in {".html"}:
            artifact_type = "report"
        elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            artifact_type = "figure"
        elif suffix in {".log", ".txt", ".md"}:
            artifact_type = "text/log"
        else:
            artifact_type = "artifact"
        rows.append({
            "stage_id": folder,
            "stage": spec.label if spec else folder,
            "artifact_type": artifact_type,
            "filename": path.name,
            "relative_path": str(rel),
            "absolute_path": str(path),
            "size_kb": round(path.stat().st_size / 1024.0, 2),
            "modified_utc": _iso_mtime(path),
            "previewable": suffix in _PREVIEW_EXTENSIONS,
            "rows": rows_count,
            "columns": cols_count,
        })
    return pd.DataFrame(rows)


def write_inspector_inventory(output_root: Path) -> dict[str, str]:
    """Write stage status and artifact inventory tables for the GUI/report."""
    root = _kin_root(output_root)
    tables = root / "008_inspector" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    stage_df = stage_status_dataframe(output_root)
    artifact_df = artifact_inventory_dataframe(output_root)
    stage_csv = tables / "stage_status.csv"
    artifact_csv = tables / "artifact_inventory.csv"
    manifest_json = tables / "inspector_manifest.json"
    stage_df.to_csv(stage_csv, index=False)
    artifact_df.to_csv(artifact_csv, index=False)
    payload = {
        "schema": "vslp_kinematics_inspector_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage_status_csv": str(stage_csv),
        "artifact_inventory_csv": str(artifact_csv),
        "n_stages": int(len(stage_df)),
        "n_artifacts": int(len(artifact_df)),
        "note": "Read-only artifact inventory. This does not validate clinical or scientific correctness by itself.",
    }
    manifest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "stage_status_csv": str(stage_csv),
        "artifact_inventory_csv": str(artifact_csv),
        "inspector_manifest_json": str(manifest_json),
        "n_stages": str(len(stage_df)),
        "n_artifacts": str(len(artifact_df)),
    }
