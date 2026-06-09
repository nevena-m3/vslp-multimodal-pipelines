"""Landmark/video quality-control summaries for the VSLP kinematics GUI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import LANDMARK_PRESETS


@dataclass(frozen=True)
class VideoQCConfig:
    selected_landmarks: tuple[int, ...] = LANDMARK_PRESETS["ALS oral-motor core 15"]
    pass_face_fraction: float = 0.90
    review_face_fraction: float = 0.75
    max_gap_review_fraction: float = 0.10
    max_gap_fail_fraction: float = 0.25
    jitter_review_p95: float = 0.080
    jitter_fail_p95: float = 0.140


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


def _safe_bool_series(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    text = series.astype(str).str.lower().str.strip()
    return text.isin(["true", "1", "yes", "y"]).to_numpy(dtype=bool)


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


def _classify(row: dict, cfg: VideoQCConfig) -> tuple[str, str]:
    reasons: list[str] = []
    face_frac = row.get("face_detected_fraction")
    max_gap = int(row.get("max_no_face_gap_frames") or 0)
    n_frames = int(row.get("n_frames") or 0)
    gap_frac = float(max_gap / n_frames) if n_frames else 1.0
    jitter_p95 = row.get("p95_frame_displacement")

    status = "pass"
    if face_frac is None or not np.isfinite(face_frac):
        return "fail", "No valid detected-face fraction."
    if face_frac < cfg.review_face_fraction:
        status = "fail"
        reasons.append(f"face detection below review threshold ({face_frac:.1%})")
    elif face_frac < cfg.pass_face_fraction:
        status = "review"
        reasons.append(f"face detection below pass threshold ({face_frac:.1%})")

    if n_frames == 0:
        status = "fail"
        reasons.append("zero decoded frames")
    elif gap_frac >= cfg.max_gap_fail_fraction:
        status = "fail"
        reasons.append(f"long no-face gap ({max_gap} frames; {gap_frac:.1%} of video)")
    elif gap_frac >= cfg.max_gap_review_fraction and status != "fail":
        status = "review"
        reasons.append(f"long no-face gap ({max_gap} frames; {gap_frac:.1%} of video)")

    if jitter_p95 is not None and np.isfinite(jitter_p95):
        if jitter_p95 >= cfg.jitter_fail_p95:
            status = "fail"
            reasons.append(f"very high landmark frame-to-frame displacement p95={jitter_p95:.3f}")
        elif jitter_p95 >= cfg.jitter_review_p95 and status != "fail":
            status = "review"
            reasons.append(f"high landmark frame-to-frame displacement p95={jitter_p95:.3f}")

    return status, "; ".join(reasons) if reasons else "No major automated QC concerns detected."


def run_video_qc(output_root: Path | str, landmarks_manifest_csv: Path | str | None = None, cfg: VideoQCConfig | None = None) -> dict:
    """Summarize landmark extraction quality into per-video QC rows.

    This is intentionally conservative. It flags videos for review/failure but does
    not delete or exclude anything automatically.
    """
    cfg = cfg or VideoQCConfig()
    root = Path(output_root).expanduser().resolve()
    if landmarks_manifest_csv is None:
        landmarks_manifest_csv = root / "kinematics" / "002_landmarks" / "tables" / "landmarks_manifest.csv"
    manifest_path = Path(landmarks_manifest_csv).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Landmark manifest not found: {manifest_path}")

    out_dir = root / "kinematics" / "005_video_qc"
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    rows: list[dict] = []
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
            "n_qc_landmarks": 0,
            "median_frame_displacement": None,
            "p95_frame_displacement": None,
            "large_jump_fraction": None,
        }
        path = Path(lmks).expanduser() if lmks else None
        if path and path.exists():
            try:
                df = pd.read_csv(path)
                if "face_detected" in df.columns and not df.empty:
                    detected = _safe_bool_series(df["face_detected"])
                    n_runs, max_gap = _runs_of_false(detected)
                    row["n_frames"] = int(len(df))
                    row["n_faces_detected"] = int(detected.sum())
                    row["face_detected_fraction"] = float(detected.mean()) if len(detected) else np.nan
                    row["n_no_face_runs"] = n_runs
                    row["max_no_face_gap_frames"] = max_gap
                    row["max_no_face_gap_fraction"] = float(max_gap / len(df)) if len(df) else None
                    row.update(_landmark_jitter(df, cfg.selected_landmarks, detected))
            except Exception as exc:  # noqa: BLE001
                row["qc_read_error"] = repr(exc)
        else:
            row["qc_read_error"] = "landmark_csv_missing"
        qc_status, rationale = _classify(row, cfg)
        row["qc_status"] = qc_status
        row["qc_rationale"] = rationale
        rows.append(row)

    qc_df = pd.DataFrame(rows)
    summary_csv = tables / "landmark_video_qc_summary.csv"
    qc_df.to_csv(summary_csv, index=False)
    status_counts = qc_df["qc_status"].value_counts(dropna=False).to_dict() if not qc_df.empty else {}
    payload = {
        "summary": {
            "n_videos": int(len(qc_df)),
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "mean_face_detected_fraction": float(pd.to_numeric(qc_df.get("face_detected_fraction"), errors="coerce").mean()) if not qc_df.empty else None,
        },
        "config": asdict(cfg),
        "input_landmarks_manifest": str(manifest_path),
        "outputs": {"summary_csv": str(summary_csv)},
        "note": "Automated QC flags acquisition/landmark risk for analyst review. It does not automatically exclude videos.",
    }
    summary_json = tables / "landmark_video_qc_summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return {"summary_csv": summary_csv, "summary_json": summary_json, **payload["summary"]}
