"""Temporal aggregation for VSLP kinematic feature time series.

This module collapses the frame-level kinematic time-series outputs from the
Feature stage into auditable per-video scalar tables. It intentionally keeps the
feature-computation table intact and writes a separate aggregation layer so the
collapse policy is explicit and reproducible.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .schemas import AGGREGATION_PROFILES


@dataclass(frozen=True)
class TemporalAggregationConfig:
    profile: str = "robust_default"
    include_raw_signals: bool = False
    include_velocity_signals: bool = True
    min_valid_fraction: float = 0.50
    min_detected_fraction: float = 0.60
    movement_kinds: tuple[str, ...] = ("open", "close", "whole")
    overwrite: bool = True


@dataclass(frozen=True)
class AggregationGuideRow:
    section: str
    item: str
    meaning: str
    output_effect: str
    recommended_use: str


AGGREGATION_GUIDE_ROWS: tuple[AggregationGuideRow, ...] = (
    AggregationGuideRow(
        "Why aggregate?",
        "Frame-level signal -> per-video scalar table",
        "A kinematic signal can have one value per frame. Aggregation summarizes that distribution so each video can be joined with metadata, QC, acoustic features, and ML/statistical models.",
        "Writes one row per video while preserving the original time-series CSV for audit.",
        "Use scalar tables for cohort modeling/export; inspect time series when a video is REVIEW/FAIL or a feature is biologically surprising.",
    ),
    AggregationGuideRow(
        "Information retention",
        "Do not rely on mean alone",
        "Mean collapses the entire trajectory into one average level and can hide bursts, peaks, pauses, dropouts, and asymmetric movement.",
        "Robust profiles export median, IQR, p05, p95, and p95-p05 range in addition to optional dense summaries.",
        "Use median/IQR/p95 as default biomarkers; treat mean as descriptive, not primary.",
    ),
    AggregationGuideRow(
        "Statistic",
        "median",
        "Typical value of the signal across valid frames, less sensitive to tracking spikes than mean.",
        "<signal>_median",
        "Default central tendency for amplitude, speed, acceleration, and ratio signals.",
    ),
    AggregationGuideRow(
        "Statistic",
        "IQR",
        "Middle-spread of the signal distribution; reflects variability without being dominated by extremes.",
        "<signal>_iqr",
        "Useful for movement variability, consistency, and possible incoordination.",
    ),
    AggregationGuideRow(
        "Statistic",
        "p05 / p95",
        "Lower and upper robust tails of the signal distribution. p95 often approximates near-peak movement without using a single noisy maximum.",
        "<signal>_p05 and <signal>_p95",
        "Use p95/range for near-peak opening, speed, or spread; avoid raw max unless visually audited.",
    ),
    AggregationGuideRow(
        "Statistic",
        "p95 - p05 range",
        "Robust dynamic range across the analyzed frames.",
        "<signal>_range_p05_p95",
        "Better default than max-min for noisy markerless video.",
    ),
    AggregationGuideRow(
        "Statistic",
        "valid fraction",
        "Fraction of frames where that signal had a finite value.",
        "<signal>_valid_fraction",
        "Do not trust scalar features with low valid fraction; review the video/QC first.",
    ),
    AggregationGuideRow(
        "Profile",
        "robust_default",
        "Summarizes each selected signal over the whole video using robust distributional statistics.",
        "One row per video; no movement-level rows required.",
        "Default export for most early ALS/PD exploratory analyses.",
    ),
    AggregationGuideRow(
        "Profile",
        "movement_segmented",
        "Uses movement_id/movement_kind when available to summarize individual movement events before summarizing across events.",
        "Adds movement_level_features.csv and movement-derived summary columns.",
        "Use for repeated open/close, DDK, smile, pucker, or other event-like tasks.",
    ),
    AggregationGuideRow(
        "Profile",
        "full_timeseries_summary",
        "Summarizes the full trajectory without assuming repeated movement events.",
        "One row per video using whole-video distribution summaries.",
        "Use for passage/conversation or continuous tasks where movement segmentation is not meaningful.",
    ),
    AggregationGuideRow(
        "Profile",
        "clinically_sensitive",
        "Exports robust summaries plus more dense tail/mean/sd descriptors. More informative but more QC-sensitive.",
        "Larger scalar table; movement-level summaries enabled.",
        "Use after good QC, not as the first-pass ML feature set.",
    ),
    AggregationGuideRow(
        "Profile",
        "exploratory_dense",
        "Large discovery-oriented reduction set.",
        "Many scalar columns; higher multiple-comparison/overfitting risk.",
        "Use for research discovery only, then down-select and validate.",
    ),
    AggregationGuideRow(
        "Setting",
        "include raw unsmoothed signals",
        "Includes raw/less-processed signals if the feature time-series table contains columns ending in _raw.",
        "Adds more columns if raw signals exist.",
        "Keep off by default; turn on for debugging or method comparison.",
    ),
    AggregationGuideRow(
        "Setting",
        "include velocity-derived signals",
        "Includes columns ending in _velocity. Velocity can be clinically meaningful but is more sensitive to frame rate and tracking noise.",
        "Controls whether velocity signals are summarized.",
        "Keep on for oral-motor speed analysis after QC.",
    ),
    AggregationGuideRow(
        "Setting",
        "minimum valid fraction",
        "Minimum finite-frame fraction required for each signal before flagging it.",
        "Adds low_valid_fraction:<signal> to aggregation_qc_flags when violated.",
        "Use 0.50 for exploratory review; use 0.80-0.90 for stricter export.",
    ),
    AggregationGuideRow(
        "Setting",
        "minimum detected-face fraction",
        "Minimum fraction of frames with a detected face before flagging the video-level aggregation.",
        "Adds low_face_detection to aggregation_qc_flags when violated.",
        "Keep aligned with Video QC; strict analyses should use >=0.90.",
    ),
)


def aggregation_guide_dataframe() -> pd.DataFrame:
    """Return a reviewer-facing guide for aggregation statistics and settings."""
    return pd.DataFrame([asdict(row) for row in AGGREGATION_GUIDE_ROWS])


def write_aggregation_guide(output_root: Path | str) -> dict:
    """Write the aggregation guide next to aggregation outputs."""
    out_root = _aggregation_dir(output_root)
    tables = out_root / "tables"
    guide = aggregation_guide_dataframe()
    csv_path = tables / "aggregation_guide.csv"
    json_path = tables / "aggregation_guide.json"
    guide.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"rows": guide.to_dict(orient="records")}, indent=2), encoding="utf-8")
    return {"aggregation_guide_csv": csv_path, "aggregation_guide_json": json_path, "n_rows": int(len(guide))}


def _aggregation_dir(output_root: Path | str) -> Path:
    out = Path(output_root).expanduser().resolve() / "kinematics" / "007_aggregation"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    return out


def _features_dir(output_root: Path | str) -> Path:
    return Path(output_root).expanduser().resolve() / "kinematics" / "006_features"


def _finite_fraction(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.isfinite(arr).mean()) if arr.size else math.nan


def _iqr(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if np.isfinite(arr).sum() == 0:
        return math.nan
    return float(np.nanquantile(arr, 0.75) - np.nanquantile(arr, 0.25))


def _safe_summary(values: np.ndarray, prefix: str, dense: bool = False) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    out: dict[str, float] = {f"{prefix}_valid_fraction": _finite_fraction(arr)}
    if np.isfinite(arr).sum() == 0:
        basic = ["median", "iqr", "p05", "p95", "range_p05_p95"]
        extra = ["mean", "sd", "min", "max", "p01", "p25", "p75", "p99"] if dense else []
        for name in basic + extra:
            out[f"{prefix}_{name}"] = math.nan
        return out
    p05 = float(np.nanquantile(arr, 0.05))
    p95 = float(np.nanquantile(arr, 0.95))
    out.update({
        f"{prefix}_median": float(np.nanmedian(arr)),
        f"{prefix}_iqr": _iqr(arr),
        f"{prefix}_p05": p05,
        f"{prefix}_p95": p95,
        f"{prefix}_range_p05_p95": p95 - p05,
    })
    if dense:
        out.update({
            f"{prefix}_mean": float(np.nanmean(arr)),
            f"{prefix}_sd": float(np.nanstd(arr)),
            f"{prefix}_min": float(np.nanmin(arr)),
            f"{prefix}_max": float(np.nanmax(arr)),
            f"{prefix}_p01": float(np.nanquantile(arr, 0.01)),
            f"{prefix}_p25": float(np.nanquantile(arr, 0.25)),
            f"{prefix}_p75": float(np.nanquantile(arr, 0.75)),
            f"{prefix}_p99": float(np.nanquantile(arr, 0.99)),
        })
    return out


def _signal_columns(ts: pd.DataFrame, cfg: TemporalAggregationConfig) -> list[str]:
    blocked = {"frame", "timestamp_ms", "time_s", "face_detected", "movement_id", "movement_kind"}
    cols: list[str] = []
    for col in ts.columns:
        if col in blocked:
            continue
        if col.endswith("_raw") and not cfg.include_raw_signals:
            continue
        if col.endswith("_velocity") and not cfg.include_velocity_signals:
            continue
        if pd.api.types.is_numeric_dtype(ts[col]):
            cols.append(col)
    return cols


def _time_span_seconds(ts: pd.DataFrame) -> float | None:
    if "time_s" not in ts.columns or len(ts) < 2:
        return None
    t = pd.to_numeric(ts["time_s"], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(t).sum() < 2:
        return None
    return float(np.nanmax(t) - np.nanmin(t))


def _movement_rows(video_id: str, ts: pd.DataFrame, signal_cols: Iterable[str], dense: bool) -> list[dict]:
    if "movement_id" not in ts.columns:
        return []
    rows: list[dict] = []
    mids = [int(v) for v in pd.unique(ts["movement_id"]) if pd.notna(v) and int(v) >= 0]
    for mid in sorted(mids):
        seg = ts.loc[ts["movement_id"].astype(int) == mid].copy()
        if seg.empty:
            continue
        kind = str(seg["movement_kind"].iloc[0]) if "movement_kind" in seg.columns else ""
        row: dict[str, object] = {
            "video_id": video_id,
            "movement_id": mid,
            "movement_kind": kind,
            "n_frames": int(len(seg)),
            "start_frame": int(pd.to_numeric(seg.get("frame", pd.Series([0])).iloc[0], errors="coerce") or 0),
            "end_frame": int(pd.to_numeric(seg.get("frame", pd.Series([len(seg) - 1])).iloc[-1], errors="coerce") or 0),
            "duration_s": _time_span_seconds(seg),
        }
        for col in signal_cols:
            row.update(_safe_summary(pd.to_numeric(seg[col], errors="coerce").to_numpy(dtype=float), col, dense=dense))
        rows.append(row)
    return rows


def aggregate_timeseries_file(input_csv: Path | str, cfg: TemporalAggregationConfig, video_id: str | None = None) -> tuple[dict, list[dict]]:
    path = Path(input_csv).expanduser().resolve()
    ts = pd.read_csv(path)
    if video_id is None:
        name = path.name
        video_id = name.replace("-kinematic-timeseries.csv", "").replace("_kinematic_timeseries.csv", "")
    dense = cfg.profile in {"clinically_sensitive", "exploratory_dense"}
    movement_profile = cfg.profile in {"movement_segmented", "clinically_sensitive", "exploratory_dense"}
    signal_cols = _signal_columns(ts, cfg)
    detected = pd.Series([True] * len(ts))
    if "face_detected" in ts.columns:
        detected = ts["face_detected"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    row: dict[str, object] = {
        "video_id": video_id,
        "input_timeseries_csv": str(path),
        "aggregation_profile": cfg.profile,
        "n_frames": int(len(ts)),
        "n_signals_aggregated": int(len(signal_cols)),
        "face_detected_fraction": float(detected.mean()) if len(detected) else math.nan,
        "duration_s": _time_span_seconds(ts),
    }
    flags: list[str] = []
    if len(ts) == 0:
        flags.append("empty_timeseries")
    if row["face_detected_fraction"] is not None and np.isfinite(float(row["face_detected_fraction"])) and float(row["face_detected_fraction"]) < cfg.min_detected_fraction:
        flags.append("low_face_detection")
    if not signal_cols:
        flags.append("no_numeric_kinematic_signals")

    for col in signal_cols:
        values = pd.to_numeric(ts[col], errors="coerce").to_numpy(dtype=float)
        if _finite_fraction(values) < cfg.min_valid_fraction:
            flags.append(f"low_valid_fraction:{col}")
        row.update(_safe_summary(values, col, dense=dense))

    movement_level = _movement_rows(video_id, ts, signal_cols, dense=dense) if movement_profile else []
    if movement_level:
        mv = pd.DataFrame(movement_level)
        row["n_movements"] = int(len(mv))
        row["n_open_movements"] = int((mv.get("movement_kind", pd.Series(dtype=str)) == "open").sum())
        row["n_close_movements"] = int((mv.get("movement_kind", pd.Series(dtype=str)) == "close").sum())
        for col in signal_cols:
            med_col = f"{col}_median"
            range_col = f"{col}_range_p05_p95"
            if med_col in mv.columns:
                row.update(_safe_summary(pd.to_numeric(mv[med_col], errors="coerce").to_numpy(dtype=float), f"movement_{col}_median", dense=False))
            if range_col in mv.columns:
                row.update(_safe_summary(pd.to_numeric(mv[range_col], errors="coerce").to_numpy(dtype=float), f"movement_{col}_range", dense=False))
    else:
        row["n_movements"] = int(ts["movement_id"].nunique()) if "movement_id" in ts.columns else 0

    row["aggregation_qc_flags"] = ";".join(dict.fromkeys(flags))
    row["status"] = "ok" if not flags else "qc_flagged"
    return row, movement_level


def run_temporal_aggregation(
    output_root: Path | str,
    cfg: TemporalAggregationConfig | None = None,
    features_manifest_csv: Path | str | None = None,
) -> dict:
    cfg = cfg or TemporalAggregationConfig()
    if cfg.profile not in AGGREGATION_PROFILES:
        raise ValueError(f"Unknown aggregation profile: {cfg.profile}")
    root = Path(output_root).expanduser().resolve()
    features_root = _features_dir(root)
    features_csv = Path(features_manifest_csv).expanduser().resolve() if features_manifest_csv else features_root / "tables" / "kinematic_features.csv"
    if not features_csv.exists():
        raise FileNotFoundError(f"Kinematic feature table not found: {features_csv}. Run Features first.")
    feature_df = pd.read_csv(features_csv)
    out_root = _aggregation_dir(root)
    tables = out_root / "tables"
    rows: list[dict] = []
    movement_rows: list[dict] = []
    for _, rec in feature_df.iterrows():
        video_id = str(rec.get("video_id", ""))
        ts_path = str(rec.get("output_timeseries_csv", "") or "")
        if not ts_path:
            rows.append({"video_id": video_id, "status": "error", "aggregation_qc_flags": "missing_timeseries_path"})
            continue
        try:
            row, mv = aggregate_timeseries_file(ts_path, cfg, video_id=video_id)
            row["feature_status"] = str(rec.get("status", ""))
            row["feature_qc_flags"] = str(rec.get("feature_qc_flags", ""))
            rows.append(row)
            movement_rows.extend(mv)
        except Exception as exc:  # noqa: BLE001
            rows.append({"video_id": video_id, "input_timeseries_csv": ts_path, "status": "error", "aggregation_qc_flags": repr(exc)})
    agg_df = pd.DataFrame(rows)
    agg_csv = tables / "kinematic_aggregated_features.csv"
    agg_df.to_csv(agg_csv, index=False)
    movement_csv = tables / "movement_level_features.csv"
    if movement_rows:
        pd.DataFrame(movement_rows).to_csv(movement_csv, index=False)
    else:
        pd.DataFrame(columns=["video_id", "movement_id", "movement_kind", "n_frames", "duration_s"]).to_csv(movement_csv, index=False)
    config_json = tables / "aggregation_config.json"
    config_payload = asdict(cfg) | {"profile_description": AGGREGATION_PROFILES.get(cfg.profile, "")}
    config_json.write_text(json.dumps(config_payload, indent=2, default=str), encoding="utf-8")
    guide_outputs = write_aggregation_guide(root)
    counts = agg_df.get("status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not agg_df.empty else {}
    manifest = {
        "n_videos": int(len(agg_df)),
        "n_ok": int((agg_df.get("status") == "ok").sum()) if not agg_df.empty else 0,
        "n_qc_flagged": int((agg_df.get("status") == "qc_flagged").sum()) if not agg_df.empty else 0,
        "n_error": int((agg_df.get("status") == "error").sum()) if not agg_df.empty else 0,
        "status_counts": {str(k): int(v) for k, v in counts.items()},
        "input_features_csv": str(features_csv),
        "outputs": {
            "aggregated_features_csv": str(agg_csv),
            "movement_level_features_csv": str(movement_csv),
            "aggregation_config_json": str(config_json),
            "aggregation_guide_csv": str(guide_outputs["aggregation_guide_csv"]),
            "aggregation_guide_json": str(guide_outputs["aggregation_guide_json"]),
        },
        "config": config_payload,
        "note": "Temporal aggregation collapses frame-level kinematic time series. QC flags are retained for analyst review and are not automatic exclusions.",
    }
    manifest_json = tables / "aggregation_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {
        "aggregated_features_csv": agg_csv,
        "movement_level_features_csv": movement_csv,
        "config_json": config_json,
        "manifest_json": manifest_json,
        **guide_outputs,
        **{k: manifest[k] for k in ["n_videos", "n_ok", "n_qc_flagged", "n_error", "status_counts"]},
    }


__all__ = [
    "TemporalAggregationConfig",
    "AggregationGuideRow",
    "AGGREGATION_GUIDE_ROWS",
    "aggregation_guide_dataframe",
    "write_aggregation_guide",
    "aggregate_timeseries_file",
    "run_temporal_aggregation",
]
