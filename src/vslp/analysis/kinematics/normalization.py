"""Computational normalization for MediaPipe kinematic landmarks.

The normalization stage converts MediaPipe's per-frame normalized image-space
coordinates into analyst-ready coordinates expressed relative to a stable
anatomical scale. It keeps the full landmark extraction outputs unchanged and
writes a separate normalized layer for downstream QC and feature computation.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .schemas import LANDMARK_PRESETS, NORMALIZATION_METHODS


@dataclass(frozen=True)
class NormalizationConfig:
    method: str = "intercanthal_distance"
    selected_landmarks: tuple[int, ...] = LANDMARK_PRESETS["ALS oral-motor core 15"]
    selected_preset: str = "ALS oral-motor core 15"
    center_landmark: int = 1
    min_face_detected_fraction: float = 0.60
    min_scale_valid_fraction: float = 0.60
    overwrite: bool = True
    notes: str = ""


_METHOD_ANCHORS: dict[str, tuple[int, ...]] = {
    # Inner canthi / eye corners in the MediaPipe 478-point topology.
    "intercanthal_distance": (133, 362),
    # Common outer-eye proxy. This is not a true interpupillary distance, but it
    # is often stable and easy to audit visually.
    "interpupillary_or_outer_eye": (33, 263),
    # Vertical face-height proxy. Landmark 10 is upper face/forehead region;
    # 152 is lower chin. This is more pose-sensitive than eye-width scaling.
    "face_height_nose_chin": (10, 152),
}


def _normalization_dir(output_root: Path | str) -> Path:
    out = Path(output_root).expanduser().resolve() / "kinematics" / "004_normalization" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _landmarks_tables_dir(output_root: Path | str) -> Path:
    return Path(output_root).expanduser().resolve() / "kinematics" / "002_landmarks" / "tables"


def _selection_json_path(output_root: Path | str) -> Path:
    return Path(output_root).expanduser().resolve() / "kinematics" / "003_selection" / "tables" / "selected_landmarks.json"


def _coord_columns(idx: int) -> tuple[str, str, str]:
    return f"{idx}_x", f"{idx}_y", f"{idx}_z"


def _present_columns(df: pd.DataFrame, idx: int) -> bool:
    return all(col in df.columns for col in _coord_columns(idx))


def _distance(df: pd.DataFrame, a: int, b: int) -> pd.Series:
    ax, ay, az = _coord_columns(a)
    bx, by, bz = _coord_columns(b)
    return np.sqrt((df[ax] - df[bx]) ** 2 + (df[ay] - df[by]) ** 2 + (df[az] - df[bz]) ** 2)


def load_selected_landmarks(output_root: Path | str, fallback_preset: str = "ALS oral-motor core 15") -> tuple[tuple[int, ...], str]:
    """Load visual workstation selection, falling back to the default preset."""
    path = _selection_json_path(output_root)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = tuple(int(v) for v in payload.get("selected_landmarks", []))
        if values:
            return values, str(payload.get("preset") or "custom_visual_selection")
    return tuple(LANDMARK_PRESETS[fallback_preset]), fallback_preset


def _resolve_selected_landmarks(output_root: Path | str, selected_landmarks: Iterable[int] | None, preset: str) -> tuple[tuple[int, ...], str]:
    if selected_landmarks:
        vals = tuple(dict.fromkeys(int(v) for v in selected_landmarks))
        return vals, preset or "custom"
    return load_selected_landmarks(output_root, fallback_preset=preset or "ALS oral-motor core 15")


def write_normalization_config(output_root: Path | str, method: str, *, notes: str = "") -> Path:
    """Write a traceable normalization config without computing trajectories."""
    if method not in NORMALIZATION_METHODS:
        raise KeyError(method)
    selected, preset = load_selected_landmarks(output_root)
    out = _normalization_dir(output_root)
    path = out / "normalization_config.json"
    payload = {
        "method": method,
        "interpretation": NORMALIZATION_METHODS[method],
        "selected_landmarks": list(selected),
        "selected_preset": preset,
        "notes": notes,
        "output_stage": "004_normalization",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _scale_series(df: pd.DataFrame, method: str) -> tuple[pd.Series, str, str]:
    face_mask = df.get("face_detected", pd.Series([True] * len(df))).astype(bool)
    if method == "raw_normalized_coordinates":
        return pd.Series([1.0] * len(df), index=df.index), "raw_unit_scale", "ok"
    if method == "face_bbox_width":
        x_cols = [c for c in df.columns if c.endswith("_x")]
        if not x_cols:
            return pd.Series([np.nan] * len(df), index=df.index), "face_bbox_width", "missing_x_columns"
        values = df.loc[:, x_cols]
        scale = values.max(axis=1, skipna=True) - values.min(axis=1, skipna=True)
        scale = scale.where(face_mask)
        return scale, "face_bbox_width", "ok"
    if method == "procrustes_head_stabilized":
        # Computational placeholder: use intercanthal scaling now and explicitly
        # mark that rigid Procrustes rotation has not yet been applied.
        method = "intercanthal_distance"
        status_suffix = "procrustes_not_yet_applied"
    else:
        status_suffix = "ok"
    anchors = _METHOD_ANCHORS.get(method)
    if not anchors:
        return pd.Series([np.nan] * len(df), index=df.index), method, "unknown_method"
    a, b = anchors
    if not (_present_columns(df, a) and _present_columns(df, b)):
        return pd.Series([np.nan] * len(df), index=df.index), f"landmark_distance_{a}_{b}", "missing_anchor_columns"
    scale = _distance(df, a, b).where(face_mask)
    return scale, f"landmark_distance_{a}_{b}", status_suffix


def _center_frame(df: pd.DataFrame, method: str, center_landmark: int) -> pd.DataFrame:
    if _present_columns(df, center_landmark):
        cx, cy, cz = _coord_columns(center_landmark)
        return pd.DataFrame({"center_x": df[cx], "center_y": df[cy], "center_z": df[cz]})
    # Fallback to the midpoint of the default eye anchors if nose center is not
    # available. If anchors are also unavailable, use zero so output columns are
    # still present but QC will flag the problem via scale status.
    anchors = _METHOD_ANCHORS.get(method, _METHOD_ANCHORS["intercanthal_distance"])
    a, b = anchors if len(anchors) == 2 else _METHOD_ANCHORS["intercanthal_distance"]
    if _present_columns(df, a) and _present_columns(df, b):
        ax, ay, az = _coord_columns(a)
        bx, by, bz = _coord_columns(b)
        return pd.DataFrame({"center_x": (df[ax] + df[bx]) / 2.0, "center_y": (df[ay] + df[by]) / 2.0, "center_z": (df[az] + df[bz]) / 2.0})
    return pd.DataFrame({"center_x": 0.0, "center_y": 0.0, "center_z": 0.0}, index=df.index)


def normalize_landmark_file(input_csv: Path | str, output_csv: Path | str, cfg: NormalizationConfig) -> dict:
    """Normalize one landmark CSV and return manifest/QC fields."""
    input_csv = Path(input_csv).expanduser().resolve()
    output_csv = Path(output_csv).expanduser().resolve()
    if output_csv.exists() and not cfg.overwrite:
        return {"status": "skipped_existing", "input_csv": str(input_csv), "output_csv": str(output_csv)}
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Landmark CSV is empty: {input_csv}")
    if "face_detected" not in df.columns:
        df["face_detected"] = True
    face_detected = df["face_detected"].astype(bool)
    scale_frame, scale_source, scale_status = _scale_series(df, cfg.method)
    valid_scale = np.isfinite(scale_frame.to_numpy(dtype=float)) & (scale_frame.to_numpy(dtype=float) > 0)
    robust_scale = float(np.nanmedian(scale_frame[valid_scale])) if valid_scale.any() else math.nan
    robust_scale_ok = bool(np.isfinite(robust_scale) and robust_scale > 0)
    center = _center_frame(df, cfg.method, cfg.center_landmark)
    rows = pd.DataFrame({
        "frame": df.get("frame", pd.Series(range(len(df)))),
        "timestamp_ms": df.get("timestamp_ms", pd.Series([np.nan] * len(df))),
        "face_detected": face_detected,
        "normalization_method": cfg.method,
        "scale_source": scale_source,
        "scale_status": scale_status if robust_scale_ok else "invalid_scale",
        "scale_value_video_median": robust_scale,
        "scale_value_frame": scale_frame,
        "scale_valid_frame": valid_scale,
        "center_landmark": cfg.center_landmark,
        "center_x": center["center_x"],
        "center_y": center["center_y"],
        "center_z": center["center_z"],
    })
    missing_selected: list[int] = []
    for idx in cfg.selected_landmarks:
        if not _present_columns(df, idx):
            missing_selected.append(idx)
            continue
        x, y, z = _coord_columns(idx)
        denom = robust_scale if robust_scale_ok else np.nan
        rows[f"{idx}_x_norm"] = (df[x] - center["center_x"]) / denom
        rows[f"{idx}_y_norm"] = (df[y] - center["center_y"]) / denom
        rows[f"{idx}_z_norm"] = (df[z] - center["center_z"]) / denom
        rows[f"{idx}_available"] = face_detected & df[[x, y, z]].notna().all(axis=1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_csv.with_suffix(".part.csv")
    rows.to_csv(tmp, index=False)
    tmp.replace(output_csv)
    n_frames = int(len(df))
    n_face = int(face_detected.sum())
    face_fraction = float(n_face / n_frames) if n_frames else math.nan
    scale_valid_fraction = float(valid_scale.sum() / n_frames) if n_frames else math.nan
    selected_available_cols = [f"{idx}_available" for idx in cfg.selected_landmarks if f"{idx}_available" in rows.columns]
    if selected_available_cols:
        selected_frame_fraction = float(rows[selected_available_cols].all(axis=1).mean())
        mean_selected_availability = float(rows[selected_available_cols].mean(axis=0).mean())
    else:
        selected_frame_fraction = math.nan
        mean_selected_availability = math.nan
    qc_flags = []
    if not robust_scale_ok:
        qc_flags.append("invalid_scale")
    if face_fraction < cfg.min_face_detected_fraction:
        qc_flags.append("low_face_detected_fraction")
    if scale_valid_fraction < cfg.min_scale_valid_fraction:
        qc_flags.append("low_scale_valid_fraction")
    if missing_selected:
        qc_flags.append("missing_selected_landmark_columns")
    return {
        "status": "ok" if not qc_flags else "qc_flagged",
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "n_frames": n_frames,
        "n_face_detected": n_face,
        "face_detected_fraction": face_fraction,
        "scale_source": scale_source,
        "scale_status": scale_status,
        "scale_value_video_median": robust_scale,
        "scale_valid_fraction": scale_valid_fraction,
        "n_selected_landmarks": int(len(cfg.selected_landmarks)),
        "missing_selected_landmarks": ",".join(map(str, missing_selected)),
        "selected_complete_frame_fraction": selected_frame_fraction,
        "mean_selected_landmark_availability": mean_selected_availability,
        "qc_flags": ";".join(qc_flags),
    }


def run_normalization(output_root: Path | str, cfg: NormalizationConfig) -> dict:
    """Run computational normalization for all successful landmark outputs."""
    if cfg.method not in NORMALIZATION_METHODS:
        raise KeyError(cfg.method)
    output_root = Path(output_root).expanduser().resolve()
    landmark_tables = _landmarks_tables_dir(output_root)
    manifest_csv = landmark_tables / "landmarks_manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Landmark manifest not found: {manifest_csv}. Run MediaPipe landmark extraction first.")
    manifest = pd.read_csv(manifest_csv)
    out_dir = _normalization_dir(output_root)
    rows: list[dict] = []
    for _, rec in manifest.iterrows():
        status = str(rec.get("status", ""))
        if status not in {"ok", "skipped_existing"}:
            continue
        input_csv = Path(str(rec.get("output_csv", "")))
        if not input_csv.exists():
            input_csv = landmark_tables / f"{rec.get('video_id')}-lmks.csv"
        video_id = str(rec.get("video_id") or input_csv.stem.replace("-lmks", ""))
        out_csv = out_dir / f"{video_id}-norm-lmks.csv"
        try:
            row = normalize_landmark_file(input_csv, out_csv, cfg)
            row.update({
                "video_id": video_id,
                "source_path": rec.get("source_path", ""),
                "task_guess": rec.get("task_guess", ""),
            })
        except Exception as exc:  # noqa: BLE001
            row = {
                "video_id": video_id,
                "source_path": rec.get("source_path", ""),
                "task_guess": rec.get("task_guess", ""),
                "input_csv": str(input_csv),
                "output_csv": "",
                "status": "error",
                "error": repr(exc),
            }
        rows.append(row)
    out_manifest = out_dir / "normalized_landmarks_manifest.csv"
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_manifest, index=False)
    config_path = out_dir / "normalization_config.json"
    payload = {
        "config": asdict(cfg),
        "method_interpretation": NORMALIZATION_METHODS[cfg.method],
        "manifest_csv": str(out_manifest),
        "n_videos": int(len(out_df)),
        "n_ok": int((out_df.get("status") == "ok").sum()) if not out_df.empty else 0,
        "n_qc_flagged": int((out_df.get("status") == "qc_flagged").sum()) if not out_df.empty else 0,
        "n_error": int((out_df.get("status") == "error").sum()) if not out_df.empty else 0,
        "mean_face_detected_fraction": float(pd.to_numeric(out_df.get("face_detected_fraction"), errors="coerce").mean()) if not out_df.empty else math.nan,
        "mean_scale_valid_fraction": float(pd.to_numeric(out_df.get("scale_valid_fraction"), errors="coerce").mean()) if not out_df.empty else math.nan,
        "rows": rows,
    }
    config_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return {
        "manifest_csv": out_manifest,
        "config_json": config_path,
        "n_videos": payload["n_videos"],
        "n_ok": payload["n_ok"],
        "n_qc_flagged": payload["n_qc_flagged"],
        "n_error": payload["n_error"],
        "mean_face_detected_fraction": payload["mean_face_detected_fraction"],
        "mean_scale_valid_fraction": payload["mean_scale_valid_fraction"],
    }


def run_normalization_from_selection(
    output_root: Path | str,
    method: str,
    *,
    selected_landmarks: Iterable[int] | None = None,
    preset: str = "ALS oral-motor core 15",
    center_landmark: int = 1,
    overwrite: bool = True,
    notes: str = "",
) -> dict:
    """Convenience wrapper used by the GUI."""
    selected, selected_preset = _resolve_selected_landmarks(output_root, selected_landmarks, preset)
    cfg = NormalizationConfig(
        method=method,
        selected_landmarks=selected,
        selected_preset=selected_preset,
        center_landmark=int(center_landmark),
        overwrite=bool(overwrite),
        notes=notes,
    )
    return run_normalization(output_root, cfg)
