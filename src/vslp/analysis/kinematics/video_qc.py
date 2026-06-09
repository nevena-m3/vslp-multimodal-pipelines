"""Landmark/video quality-control summaries for the VSLP kinematics GUI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .normalization import load_selected_landmarks
from .schemas import LANDMARK_PRESETS


@dataclass(frozen=True)
class VideoQCConfig:
    selected_landmarks: tuple[int, ...] = LANDMARK_PRESETS["ALS oral-motor core 15"]
    pass_face_fraction: float = 0.90
    review_face_fraction: float = 0.75
    pass_selected_valid_fraction: float = 0.90
    review_selected_valid_fraction: float = 0.75
    max_gap_review_fraction: float = 0.10
    max_gap_fail_fraction: float = 0.25
    max_gap_review_seconds: float = 0.50
    max_gap_fail_seconds: float = 2.00
    jitter_review_p95: float = 0.080
    jitter_fail_p95: float = 0.140
    max_out_of_range_xy_fraction: float = 0.01


def _runs_of_false(mask: np.ndarray) -> tuple[int, int]:
    """Return number of no-face runs and maximum consecutive missing run."""
    if mask.size == 0:
        return 0, 0
    missing = ~mask.astype(bool)
    n_runs = 0
    max_run = 0
    cur = 0
    for val in missing:
        if val:
            cur += 1
            if cur == 1:
                n_runs += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return int(n_runs), int(max_run)


def _missing_runs(mask: np.ndarray, timestamps_ms: np.ndarray | None = None) -> list[dict[str, float | int]]:
    missing = ~mask.astype(bool)
    runs: list[dict[str, float | int]] = []
    start: int | None = None
    for i, is_missing in enumerate(missing):
        if is_missing and start is None:
            start = i
        if start is not None and (not is_missing or i == len(missing) - 1):
            end = i - 1 if not is_missing else i
            row: dict[str, float | int] = {"start_frame": int(start), "end_frame": int(end), "n_frames": int(end - start + 1)}
            if timestamps_ms is not None and len(timestamps_ms) == len(mask):
                t0 = timestamps_ms[start]
                t1 = timestamps_ms[end]
                if np.isfinite(t0) and np.isfinite(t1):
                    row["start_ms"] = float(t0)
                    row["end_ms"] = float(t1)
                    row["duration_s"] = float(max(0.0, (t1 - t0) / 1000.0))
            runs.append(row)
            start = None
    return runs


def _safe_bool_series(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.fillna(False).to_numpy(dtype=bool)
    text = series.astype(str).str.lower().str.strip()
    return text.isin(["true", "1", "yes", "y"]).to_numpy(dtype=bool)


def _timestamp_metrics(df: pd.DataFrame) -> dict[str, float | bool | None]:
    if "timestamp_ms" not in df.columns:
        return {"timestamp_monotonic": None, "median_frame_dt_ms": None, "fps_estimated_from_timestamps": None}
    ts = pd.to_numeric(df["timestamp_ms"], errors="coerce").to_numpy(dtype=float)
    valid = ts[np.isfinite(ts)]
    if valid.size < 2:
        return {"timestamp_monotonic": None, "median_frame_dt_ms": None, "fps_estimated_from_timestamps": None}
    diffs = np.diff(valid)
    positive = diffs[diffs > 0]
    median_dt = float(np.nanmedian(positive)) if positive.size else np.nan
    fps = float(1000.0 / median_dt) if np.isfinite(median_dt) and median_dt > 0 else np.nan
    return {
        "timestamp_monotonic": bool(np.all(diffs >= 0)),
        "median_frame_dt_ms": median_dt,
        "fps_estimated_from_timestamps": fps,
    }


def _landmark_validity(df: pd.DataFrame, selected: tuple[int, ...], detected: np.ndarray) -> dict[str, float | int | str | None]:
    usable = [idx for idx in selected if all(c in df.columns for c in (f"{idx}_x", f"{idx}_y", f"{idx}_z"))]
    missing = [idx for idx in selected if idx not in usable]
    if not usable or len(df) == 0:
        return {
            "n_selected_landmarks": int(len(selected)),
            "n_selected_landmarks_present": int(len(usable)),
            "missing_selected_landmarks": ",".join(map(str, missing)),
            "selected_landmark_valid_rate": None,
            "selected_complete_frame_fraction": None,
            "out_of_range_xy_fraction": None,
        }
    availability = []
    xy_out_flags = []
    for idx in usable:
        cols = [f"{idx}_x", f"{idx}_y", f"{idx}_z"]
        values = df[cols].apply(pd.to_numeric, errors="coerce")
        valid = detected & values.notna().all(axis=1).to_numpy(dtype=bool)
        availability.append(valid)
        xy = df[[f"{idx}_x", f"{idx}_y"]].apply(pd.to_numeric, errors="coerce")
        xy_out = ((xy < -0.05) | (xy > 1.05)).any(axis=1).to_numpy(dtype=bool) & detected
        xy_out_flags.append(xy_out)
    avail_matrix = np.vstack(availability).T
    out_matrix = np.vstack(xy_out_flags).T if xy_out_flags else np.zeros((len(df), 0), dtype=bool)
    denom = max(int(detected.sum()) * len(usable), 1)
    valid_rate = float(avail_matrix[detected].sum() / denom) if detected.any() else 0.0
    complete_fraction = float(avail_matrix.all(axis=1).mean()) if len(df) else None
    out_fraction = float(out_matrix[detected].sum() / denom) if detected.any() else 0.0
    return {
        "n_selected_landmarks": int(len(selected)),
        "n_selected_landmarks_present": int(len(usable)),
        "missing_selected_landmarks": ",".join(map(str, missing)),
        "selected_landmark_valid_rate": valid_rate,
        "selected_complete_frame_fraction": complete_fraction,
        "out_of_range_xy_fraction": out_fraction,
    }


def _landmark_jitter(df: pd.DataFrame, selected: tuple[int, ...], detected: np.ndarray) -> dict[str, float | int | None]:
    usable = [idx for idx in selected if f"{idx}_x" in df.columns and f"{idx}_y" in df.columns]
    if not usable or detected.sum() < 3:
        return {"n_qc_landmarks": len(usable), "median_frame_displacement": None, "p95_frame_displacement": None, "large_jump_fraction": None}

    work = df.loc[detected, [c for idx in usable for c in (f"{idx}_x", f"{idx}_y")]].apply(pd.to_numeric, errors="coerce")
    disps: list[float] = []
    for idx in usable:
        xy = work[[f"{idx}_x", f"{idx}_y"]].to_numpy(dtype=float)
        valid = np.isfinite(xy).all(axis=1)
        if valid.sum() < 3:
            continue
        xy = xy[valid]
        step = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
        disps.extend([float(v) for v in step if np.isfinite(v)])
    if not disps:
        return {"n_qc_landmarks": len(usable), "median_frame_displacement": None, "p95_frame_displacement": None, "large_jump_fraction": None}
    arr = np.asarray(disps, dtype=float)
    return {
        "n_qc_landmarks": len(usable),
        "median_frame_displacement": float(np.nanmedian(arr)),
        "p95_frame_displacement": float(np.nanpercentile(arr, 95)),
        "large_jump_fraction": float(np.mean(arr > 0.08)),
    }


def _normalization_lookup(root: Path) -> dict[str, dict]:
    path = root / "kinematics" / "004_normalization" / "tables" / "normalized_landmarks_manifest.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return {}
    rows = {}
    for _, rec in df.iterrows():
        vid = str(rec.get("video_id", ""))
        if vid:
            rows[vid] = dict(rec)
    return rows


def _classify(row: dict, cfg: VideoQCConfig) -> tuple[str, str]:
    reasons: list[str] = []
    face_frac = row.get("face_detected_fraction")
    max_gap = int(row.get("max_no_face_gap_frames") or 0)
    n_frames = int(row.get("n_frames") or 0)
    gap_frac = float(max_gap / n_frames) if n_frames else 1.0
    gap_s = row.get("max_no_face_gap_s")
    selected_rate = row.get("selected_landmark_valid_rate")
    jitter_p95 = row.get("p95_frame_displacement")
    out_xy = row.get("out_of_range_xy_fraction")
    norm_status = str(row.get("normalization_status") or "")
    norm_flags = str(row.get("normalization_qc_flags") or "")

    status = "pass"
    if face_frac is None or not np.isfinite(face_frac):
        return "fail", "No valid detected-face fraction."
    if face_frac < cfg.review_face_fraction:
        status = "fail"
        reasons.append(f"face detection below review threshold ({face_frac:.1%})")
    elif face_frac < cfg.pass_face_fraction:
        status = "review"
        reasons.append(f"face detection below pass threshold ({face_frac:.1%})")

    if selected_rate is None or not np.isfinite(selected_rate):
        status = "fail"
        reasons.append("selected landmark validity unavailable")
    elif selected_rate < cfg.review_selected_valid_fraction:
        status = "fail"
        reasons.append(f"selected landmark validity below review threshold ({selected_rate:.1%})")
    elif selected_rate < cfg.pass_selected_valid_fraction and status != "fail":
        status = "review"
        reasons.append(f"selected landmark validity below pass threshold ({selected_rate:.1%})")

    if n_frames == 0:
        status = "fail"
        reasons.append("zero decoded frames")
    elif gap_frac >= cfg.max_gap_fail_fraction:
        status = "fail"
        reasons.append(f"long no-face gap ({max_gap} frames; {gap_frac:.1%} of video)")
    elif gap_frac >= cfg.max_gap_review_fraction and status != "fail":
        status = "review"
        reasons.append(f"long no-face gap ({max_gap} frames; {gap_frac:.1%} of video)")
    if gap_s is not None and np.isfinite(gap_s):
        if gap_s >= cfg.max_gap_fail_seconds:
            status = "fail"
            reasons.append(f"long no-face gap ({gap_s:.2f} s)")
        elif gap_s >= cfg.max_gap_review_seconds and status != "fail":
            status = "review"
            reasons.append(f"long no-face gap ({gap_s:.2f} s)")

    if jitter_p95 is not None and np.isfinite(jitter_p95):
        if jitter_p95 >= cfg.jitter_fail_p95:
            status = "fail"
            reasons.append(f"very high landmark frame-to-frame displacement p95={jitter_p95:.3f}")
        elif jitter_p95 >= cfg.jitter_review_p95 and status != "fail":
            status = "review"
            reasons.append(f"high landmark frame-to-frame displacement p95={jitter_p95:.3f}")

    if out_xy is not None and np.isfinite(out_xy) and out_xy > cfg.max_out_of_range_xy_fraction:
        status = "fail"
        reasons.append(f"out-of-range selected x/y coordinate burden ({out_xy:.2%})")
    if row.get("timestamp_monotonic") is False:
        status = "fail"
        reasons.append("non-monotonic timestamps")
    if norm_status == "error":
        status = "fail"
        reasons.append("normalization errored")
    elif norm_status == "qc_flagged" and status != "fail":
        status = "review"
        reasons.append(f"normalization flagged: {norm_flags or 'see normalization manifest'}")

    return status, "; ".join(reasons) if reasons else "No major automated QC concerns detected."


def run_video_qc(output_root: Path | str, landmarks_manifest_csv: Path | str | None = None, cfg: VideoQCConfig | None = None) -> dict:
    """Summarize landmark extraction quality into per-video QC rows.

    This is intentionally conservative. It flags videos for review/failure but does
    not delete or exclude anything automatically.
    """
    root = Path(output_root).expanduser().resolve()
    if cfg is None:
        selected, _preset = load_selected_landmarks(root)
        cfg = VideoQCConfig(selected_landmarks=selected)
    if landmarks_manifest_csv is None:
        landmarks_manifest_csv = root / "kinematics" / "002_landmarks" / "tables" / "landmarks_manifest.csv"
    manifest_path = Path(landmarks_manifest_csv).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Landmark manifest not found: {manifest_path}")

    out_dir = root / "kinematics" / "005_video_qc"
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    norm_rows = _normalization_lookup(root)
    rows: list[dict] = []
    gap_rows: list[dict] = []
    for _, rec in manifest.iterrows():
        video_id = str(rec.get("video_id", ""))
        lmks = str(rec.get("output_csv", "") or "")
        source_path = str(rec.get("source_path", "") or "")
        row = {
            "video_id": video_id,
            "source_path": source_path,
            "landmark_csv": lmks,
            "landmark_status": str(rec.get("status", "")),
            "n_frames": int(pd.to_numeric(rec.get("n_frames", 0), errors="coerce") or 0),
            "n_faces_detected": int(pd.to_numeric(rec.get("n_faces_detected", 0), errors="coerce") or 0),
            "face_detected_fraction": float(pd.to_numeric(rec.get("face_detected_fraction", np.nan), errors="coerce")),
            "n_no_face_runs": 0,
            "max_no_face_gap_frames": 0,
            "max_no_face_gap_fraction": None,
            "max_no_face_gap_s": None,
            "timestamp_monotonic": None,
            "median_frame_dt_ms": None,
            "fps_estimated_from_timestamps": None,
            "n_qc_landmarks": 0,
            "median_frame_displacement": None,
            "p95_frame_displacement": None,
            "large_jump_fraction": None,
            "normalization_status": "not_run",
            "normalization_qc_flags": "",
            "normalization_scale_cv": None,
            "normalization_scale_max_jump_fraction": None,
        }
        path = Path(lmks).expanduser() if lmks else None
        if path and path.exists():
            try:
                df = pd.read_csv(path)
                if "face_detected" in df.columns and not df.empty:
                    detected = _safe_bool_series(df["face_detected"])
                    timestamp_info = _timestamp_metrics(df)
                    row.update(timestamp_info)
                    timestamps = pd.to_numeric(df.get("timestamp_ms"), errors="coerce").to_numpy(dtype=float) if "timestamp_ms" in df.columns else None
                    n_runs, max_gap = _runs_of_false(detected)
                    gaps = _missing_runs(detected, timestamps)
                    for gap in gaps:
                        gap_rows.append({"video_id": video_id, **gap})
                    durations = [float(g["duration_s"]) for g in gaps if "duration_s" in g]
                    median_dt = row.get("median_frame_dt_ms")
                    fallback_gap_s = float(max_gap * median_dt / 1000.0) if median_dt is not None and np.isfinite(median_dt) else None
                    row["n_frames"] = int(len(df))
                    row["n_faces_detected"] = int(detected.sum())
                    row["face_detected_fraction"] = float(detected.mean()) if len(detected) else np.nan
                    row["n_no_face_runs"] = n_runs
                    row["max_no_face_gap_frames"] = max_gap
                    row["max_no_face_gap_fraction"] = float(max_gap / len(df)) if len(df) else None
                    row["max_no_face_gap_s"] = max(durations) if durations else fallback_gap_s
                    row.update(_landmark_validity(df, cfg.selected_landmarks, detected))
                    row.update(_landmark_jitter(df, cfg.selected_landmarks, detected))
            except Exception as exc:  # noqa: BLE001
                row["qc_read_error"] = repr(exc)
        else:
            row["qc_read_error"] = "landmark_csv_missing"
        if video_id in norm_rows:
            norm = norm_rows[video_id]
            row["normalization_status"] = str(norm.get("status", ""))
            row["normalization_qc_flags"] = str(norm.get("qc_flags", "") or "")
            row["normalization_scale_cv"] = pd.to_numeric(norm.get("scale_value_cv"), errors="coerce")
            row["normalization_scale_max_jump_fraction"] = pd.to_numeric(norm.get("scale_frame_to_frame_max_jump_fraction"), errors="coerce")
        qc_status, rationale = _classify(row, cfg)
        row["qc_status"] = qc_status
        row["qc_rationale"] = rationale
        rows.append(row)

    qc_df = pd.DataFrame(rows)
    summary_csv = tables / "landmark_video_qc_summary.csv"
    canonical_summary_csv = tables / "video_qc_summary.csv"
    qc_df.to_csv(summary_csv, index=False)
    qc_df.to_csv(canonical_summary_csv, index=False)
    gaps_csv = tables / "video_qc_long_gaps.csv"
    pd.DataFrame(gap_rows).to_csv(gaps_csv, index=False)
    status_counts = qc_df["qc_status"].value_counts(dropna=False).to_dict() if not qc_df.empty else {}
    payload = {
        "summary": {
            "n_videos": int(len(qc_df)),
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "mean_face_detected_fraction": float(pd.to_numeric(qc_df.get("face_detected_fraction"), errors="coerce").mean()) if not qc_df.empty else None,
            "mean_selected_landmark_valid_rate": float(pd.to_numeric(qc_df.get("selected_landmark_valid_rate"), errors="coerce").mean()) if not qc_df.empty else None,
        },
        "config": asdict(cfg),
        "input_landmarks_manifest": str(manifest_path),
        "outputs": {"summary_csv": str(summary_csv), "canonical_summary_csv": str(canonical_summary_csv), "long_gaps_csv": str(gaps_csv)},
        "note": "Automated QC flags acquisition/landmark risk for analyst review. It does not automatically exclude videos.",
    }
    summary_json = tables / "landmark_video_qc_summary.json"
    canonical_summary_json = tables / "video_qc_summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    canonical_summary_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return {"summary_csv": summary_csv, "summary_json": summary_json, "long_gaps_csv": gaps_csv, **payload["summary"]}
