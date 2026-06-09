"""MediaPipe landmark planning and execution for the kinematics GUI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .mediapipe_runtime import (
    DEFAULT_EXPECTED_LANDMARKS,
    FACE_LANDMARKER_MODEL_URL,
    download_face_landmarker_model,
    mediapipe_environment_status,
    run_landmark_extraction_from_manifest,
)
from .schemas import LANDMARK_PRESETS


@dataclass(frozen=True)
class LandmarkRunConfig:
    model_path: str | Path = "models/face_landmarker.task"
    selected_preset: str = "ALS oral-motor core 15"
    selected_landmarks: tuple[int, ...] = LANDMARK_PRESETS["ALS oral-motor core 15"]
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    normalization_method: str = "intercanthal_distance"
    n_landmarks: int = DEFAULT_EXPECTED_LANDMARKS
    overwrite: bool = False
    # Debug/test limiter; keep None for real analysis.
    max_frames: int | None = None


def _landmark_tables_dir(output_root: Path | str) -> Path:
    out = Path(output_root).expanduser().resolve() / "kinematics" / "002_landmarks" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_landmark_plan(output_root: Path | str, cfg: LandmarkRunConfig, manifest_csv: Path | str | None = None) -> Path:
    """Write a reproducible MediaPipe extraction plan without running extraction."""
    out = _landmark_tables_dir(output_root)
    plan_csv = out / "landmark_extraction_plan.csv"
    config_json = out / "landmark_config.json"
    rows: list[dict]
    if manifest_csv and Path(manifest_csv).exists():
        manifest = pd.read_csv(manifest_csv)
        keep_cols = [c for c in ["video_id", "source_path", "relative_path", "extension", "codec_name", "fps", "duration_sec", "n_frames_estimated", "status", "warning"] if c in manifest.columns]
        rows = manifest[keep_cols].to_dict(orient="records")
    else:
        rows = []
    for row in rows:
        row["landmark_preset"] = cfg.selected_preset
        row["selected_landmarks"] = ",".join(map(str, cfg.selected_landmarks))
        row["mediapipe_model_path"] = str(Path(cfg.model_path).expanduser())
        row["landmark_status"] = "planned"
    pd.DataFrame(rows).to_csv(plan_csv, index=False)
    payload = {
        **asdict(cfg),
        "model_path": str(Path(cfg.model_path).expanduser()),
        "output_root": str(Path(output_root).expanduser()),
        "manifest_csv": str(manifest_csv) if manifest_csv else None,
        "model_download_url": FACE_LANDMARKER_MODEL_URL,
        "environment": asdict(mediapipe_environment_status()),
        "note": mediapipe_capability_note(),
    }
    config_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return plan_csv


def download_default_model(output_root: Path | str, cfg: LandmarkRunConfig | None = None) -> Path:
    """Download FaceLandmarker .task model to the configured path.

    Relative model paths are resolved relative to the repository/current working
    directory by Python's normal Path rules. For GUI use we recommend the default
    ``models/face_landmarker.task`` path in the project repository.
    """
    cfg = cfg or LandmarkRunConfig()
    return download_face_landmarker_model(Path(cfg.model_path), overwrite=False)


def run_mediapipe_landmarks(
    output_root: Path | str,
    cfg: LandmarkRunConfig,
    manifest_csv: Path | str,
) -> dict:
    """Run real MediaPipe Face Landmarker extraction from an ingest manifest."""
    return run_landmark_extraction_from_manifest(
        manifest_csv,
        output_root,
        model_path=cfg.model_path,
        detection_conf=cfg.min_face_detection_confidence,
        presence_conf=cfg.min_face_presence_confidence,
        tracking_conf=cfg.min_tracking_confidence,
        selected_landmarks=cfg.selected_landmarks,
        selected_preset=cfg.selected_preset,
        n_landmarks=cfg.n_landmarks,
        overwrite=cfg.overwrite,
        max_frames=cfg.max_frames,
    )


def mediapipe_capability_note() -> str:
    return (
        "MediaPipe Face Landmarker is used in VIDEO mode to write one landmark row per video frame. "
        "The result exposes landmarks and optional blendshapes/transformation matrices; VSLP currently uses "
        "the landmark coordinates and a per-frame face_detected flag. MediaPipe does not provide a simple "
        "per-landmark confidence interval in the CSV output, so landmark quality is summarized through derived "
        "QC indicators: detected-frame fraction, dropped-frame count, long gaps, tracking stability, jitter, and "
        "configured detection/presence/tracking thresholds. Frames with no detected face are preserved as NaN "
        "rather than deleted, making tracking gaps auditable."
    )
