"""Landmark extraction planning and optional MediaPipe execution hooks."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pandas as pd

from .schemas import LANDMARK_PRESETS

@dataclass(frozen=True)
class LandmarkRunConfig:
    manifest_csv: Path
    output_root: Path
    model_path: Path | None = None
    selected_preset: str = "ALS oral-motor core 15"
    selected_indices: tuple[int, ...] = LANDMARK_PRESETS["ALS oral-motor core 15"]
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    run_extraction: bool = False


def write_landmark_plan(cfg: LandmarkRunConfig) -> dict[str, Path | int | str]:
    out = Path(cfg.output_root) / "kinematics" / "002_landmarks" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(cfg.manifest_csv)
    plan = manifest[[c for c in ["video_id", "source_path", "relative_path", "fps", "duration_sec", "status"] if c in manifest.columns]].copy()
    plan["landmark_preset"] = cfg.selected_preset
    plan["selected_landmarks"] = ",".join(map(str, cfg.selected_indices))
    plan["mediapipe_model_path"] = "" if cfg.model_path is None else str(cfg.model_path)
    plan["landmark_status"] = "planned"
    plan_csv = out / "landmark_extraction_plan.csv"
    config_json = out / "landmark_config.json"
    plan.to_csv(plan_csv, index=False)
    config_json.write_text(json.dumps({**asdict(cfg), "manifest_csv": str(cfg.manifest_csv), "output_root": str(cfg.output_root), "model_path": str(cfg.model_path) if cfg.model_path else None}, indent=2, default=str), encoding="utf-8")
    return {"plan_csv": plan_csv, "config_json": config_json, "n_videos": len(plan), "status": "planned"}


def mediapipe_capability_note() -> str:
    return (
        "MediaPipe Face Landmarker returns face landmarks and optional blendshapes/"
        "facial transformation matrices. The current VSLP scaffold treats per-frame "
        "face_detected fraction, missing-frame burden, tracking gaps, and configured "
        "detection/presence/tracking thresholds as landmark-quality evidence; it does "
        "not assume a per-landmark confidence score is available."
    )
