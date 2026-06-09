"""MediaPipe Face Landmarker execution for the VSLP kinematics GUI.

This module is import-light: cv2 and mediapipe are imported only inside runtime
functions so the GUI can open on systems where the optional video stack has not
been installed yet.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_EXPECTED_LANDMARKS = 478


@dataclass(frozen=True)
class MediaPipeEnvironmentStatus:
    opencv_available: bool
    mediapipe_available: bool
    opencv_version: str | None = None
    mediapipe_version: str | None = None
    message: str = ""


def mediapipe_environment_status() -> MediaPipeEnvironmentStatus:
    """Return import/version status for optional runtime dependencies."""
    cv2_ok = False
    mp_ok = False
    cv2_version = None
    mp_version = None
    messages: list[str] = []
    try:
        import cv2  # type: ignore

        cv2_ok = True
        cv2_version = getattr(cv2, "__version__", None)
    except Exception as exc:  # noqa: BLE001
        messages.append(f"opencv-python unavailable: {exc}")
    try:
        import mediapipe as mp  # type: ignore

        mp_ok = True
        mp_version = getattr(mp, "__version__", None)
    except Exception as exc:  # noqa: BLE001
        messages.append(f"mediapipe unavailable: {exc}")
    if cv2_ok and mp_ok:
        messages.append("MediaPipe runtime dependencies are available.")
    return MediaPipeEnvironmentStatus(cv2_ok, mp_ok, cv2_version, mp_version, " | ".join(messages))


def install_mediapipe_runtime(*, upgrade: bool = False, timeout_sec: int = 900) -> dict:
    """Install the runtime packages required for real FaceLandmarker extraction.

    This uses the active Python interpreter (``sys.executable``), so in the GUI it
    installs into the same virtual environment that launched VSLP. It is intended
    for workstation setup, not for every run. A GUI restart is recommended after a
    fresh install, although the current process can usually import the packages
    immediately.
    """
    import subprocess
    import sys

    packages = ["opencv-python", "mediapipe"]
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    if upgrade:
        cmd.insert(5, "--upgrade")
    started = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    status = mediapipe_environment_status()
    return {
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-8000:],
        "elapsed_sec": round(time.time() - started, 3),
        "environment": asdict(status),
        "ok": bool(proc.returncode == 0 and status.opencv_available and status.mediapipe_available),
    }


def ensure_mediapipe_runtime_ready(model_path: Path | str, *, auto_download_model: bool = True) -> dict:
    """Preflight check for real MediaPipe extraction.

    Raises a RuntimeError with actionable instructions if opencv-python,
    mediapipe, or the FaceLandmarker model bundle is missing. This prevents the
    GUI from silently producing an all-error manifest when the runtime is not set
    up.
    """
    status = mediapipe_environment_status()
    missing = []
    if not status.opencv_available:
        missing.append("opencv-python")
    if not status.mediapipe_available:
        missing.append("mediapipe")
    if missing:
        raise RuntimeError(
            "MediaPipe runtime is not installed in this Python environment. "
            f"Missing: {', '.join(missing)}. Use the Landmarks tab button "
            "'Install / Verify MediaPipe Runtime', or run: python -m pip install opencv-python mediapipe"
        )
    model = Path(model_path).expanduser().resolve()
    if not model.exists():
        if auto_download_model:
            download_face_landmarker_model(model, overwrite=False)
        else:
            raise FileNotFoundError(
                f"FaceLandmarker model not found: {model}. Use 'Download Default Model' first."
            )
    return {"environment": asdict(mediapipe_environment_status()), "model_path": str(model), "model_exists": model.exists()}


def download_face_landmarker_model(model_path: Path | str, *, overwrite: bool = False) -> Path:
    """Download Google's Face Landmarker .task bundle to ``model_path``.

    The URL matches the MediaPipe public model asset path used in Google's Tasks
    examples. The function is intentionally small and transparent so users can
    also download the model manually if internet access is unavailable.
    """
    path = Path(model_path).expanduser().resolve()
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    with urllib.request.urlopen(FACE_LANDMARKER_MODEL_URL, timeout=60) as response:  # noqa: S310
        data = response.read()
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def _timestamp_ms(frame_idx: int, fps: float) -> int:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    return int(round(frame_idx * 1000.0 / fps))


def _landmark_columns(n_landmarks: int) -> list[str]:
    cols = ["frame", "timestamp_ms", "face_detected"]
    for idx in range(n_landmarks):
        cols.extend([f"{idx}_x", f"{idx}_y", f"{idx}_z"])
    return cols


def _row_from_result(result, frame_idx: int, timestamp_ms: int, n_landmarks: int) -> tuple[list[float | int | bool], bool, int]:
    faces = getattr(result, "face_landmarks", None) or []
    if not faces:
        return [frame_idx, timestamp_ms, False, *([np.nan] * n_landmarks * 3)], False, n_landmarks
    face = faces[0]
    actual = len(face)
    # Most recent Face Landmarker tasks commonly return 478 points. The GUI keeps
    # a stable rectangular output even if a model/version differs by padding or
    # truncating and recording the actual count in the manifest.
    vals: list[float] = []
    use_n = min(actual, n_landmarks)
    for lm in face[:use_n]:
        vals.extend([float(lm.x), float(lm.y), float(lm.z)])
    if use_n < n_landmarks:
        vals.extend([np.nan] * (n_landmarks - use_n) * 3)
    return [frame_idx, timestamp_ms, True, *vals], True, actual


def _make_detector(model_path: Path, *, detection_conf: float, presence_conf: float, tracking_conf: float):
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks import python  # type: ignore
    from mediapipe.tasks.python import vision  # type: ignore

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=float(detection_conf),
        min_face_presence_confidence=float(presence_conf),
        min_tracking_confidence=float(tracking_conf),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def extract_landmarks_from_video(
    video_path: Path | str,
    output_csv: Path | str,
    *,
    model_path: Path | str,
    detection_conf: float = 0.5,
    presence_conf: float = 0.5,
    tracking_conf: float = 0.5,
    n_landmarks: int = DEFAULT_EXPECTED_LANDMARKS,
    overwrite: bool = False,
    max_frames: int | None = None,
) -> dict:
    """Extract one per-frame MediaPipe landmark CSV from one video."""
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
    import cv2  # type: ignore

    video_path = Path(video_path).expanduser().resolve()
    output_csv = Path(output_csv).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    if output_csv.exists() and not overwrite:
        return {
            "source_path": str(video_path),
            "output_csv": str(output_csv),
            "status": "skipped_existing",
            "n_frames": 0,
            "n_faces_detected": 0,
            "n_dropped": 0,
            "fps": 0.0,
            "face_detected_fraction": np.nan,
            "error": "",
        }
    if not model_path.is_file():
        raise FileNotFoundError(f"MediaPipe FaceLandmarker model not found: {model_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"OpenCV reported invalid fps={fps!r} for {video_path}")

    rows: list[list[float | int | bool]] = []
    n_detected = 0
    actual_counts: list[int] = []
    started = time.time()
    mp, detector = _make_detector(
        model_path,
        detection_conf=detection_conf,
        presence_conf=presence_conf,
        tracking_conf=tracking_conf,
    )
    try:
        frame_idx = 0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts = _timestamp_ms(frame_idx, fps)
            result = detector.detect_for_video(mp_image, ts)
            row, present, actual = _row_from_result(result, frame_idx, ts, n_landmarks)
            rows.append(row)
            n_detected += int(present)
            if present:
                actual_counts.append(actual)
            frame_idx += 1
    finally:
        cap.release()
        detector.close()

    df = pd.DataFrame(rows, columns=_landmark_columns(n_landmarks))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_csv.with_suffix(".part.csv")
    df.to_csv(tmp, index=False)
    os.replace(tmp, output_csv)
    n_frames = len(df)
    n_dropped = n_frames - n_detected
    return {
        "source_path": str(video_path),
        "output_csv": str(output_csv),
        "status": "ok",
        "n_frames": int(n_frames),
        "n_faces_detected": int(n_detected),
        "n_dropped": int(n_dropped),
        "fps": float(fps),
        "face_detected_fraction": float(n_detected / n_frames) if n_frames else np.nan,
        "expected_landmarks_written": int(n_landmarks),
        "actual_landmark_count_min": int(min(actual_counts)) if actual_counts else None,
        "actual_landmark_count_max": int(max(actual_counts)) if actual_counts else None,
        "elapsed_sec": round(time.time() - started, 3),
        "error": "",
    }


def run_landmark_extraction_from_manifest(
    manifest_csv: Path | str,
    output_root: Path | str,
    *,
    model_path: Path | str,
    detection_conf: float = 0.5,
    presence_conf: float = 0.5,
    tracking_conf: float = 0.5,
    selected_landmarks: Iterable[int] = (),
    selected_preset: str = "",
    n_landmarks: int = DEFAULT_EXPECTED_LANDMARKS,
    overwrite: bool = False,
    max_frames: int | None = None,
) -> dict:
    """Run MediaPipe extraction for every video in the ingest manifest."""
    manifest_csv = Path(manifest_csv).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    tables = output_root / "kinematics" / "002_landmarks" / "tables"
    logs = output_root / "kinematics" / "002_landmarks" / "logs"
    tables.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_csv)
    rows: list[dict] = []
    for _, rec in manifest.iterrows():
        source_path = Path(str(rec.get("source_path", ""))).expanduser()
        video_id = str(rec.get("video_id") or source_path.stem)
        out_csv = tables / f"{video_id}-lmks.csv"
        try:
            row = extract_landmarks_from_video(
                source_path,
                out_csv,
                model_path=model_path,
                detection_conf=detection_conf,
                presence_conf=presence_conf,
                tracking_conf=tracking_conf,
                n_landmarks=n_landmarks,
                overwrite=overwrite,
                max_frames=max_frames,
            )
            row.update({
                "video_id": video_id,
                "relative_path": rec.get("relative_path", ""),
                "task_guess": rec.get("task_guess", ""),
            })
        except Exception as exc:  # noqa: BLE001
            row = {
                "video_id": video_id,
                "source_path": str(source_path),
                "relative_path": rec.get("relative_path", ""),
                "task_guess": rec.get("task_guess", ""),
                "output_csv": "",
                "status": "error",
                "n_frames": 0,
                "n_faces_detected": 0,
                "n_dropped": 0,
                "fps": np.nan,
                "face_detected_fraction": np.nan,
                "error": repr(exc),
            }
        rows.append(row)

    out_df = pd.DataFrame(rows)
    manifest_out = tables / "landmarks_manifest.csv"
    out_df.to_csv(manifest_out, index=False)
    config = {
        "manifest_csv": str(manifest_csv),
        "output_root": str(output_root),
        "model_path": str(Path(model_path).expanduser().resolve()),
        "min_face_detection_confidence": detection_conf,
        "min_face_presence_confidence": presence_conf,
        "min_tracking_confidence": tracking_conf,
        "selected_preset": selected_preset,
        "selected_landmarks": list(selected_landmarks),
        "n_landmarks": n_landmarks,
        "overwrite": overwrite,
        "max_frames": max_frames,
        "environment": asdict(mediapipe_environment_status()),
    }
    summary = {
        "n_videos": int(len(out_df)),
        "n_ok": int((out_df["status"] == "ok").sum()) if not out_df.empty else 0,
        "n_error": int((out_df["status"] == "error").sum()) if not out_df.empty else 0,
        "n_skipped_existing": int((out_df["status"] == "skipped_existing").sum()) if not out_df.empty else 0,
        "mean_face_detected_fraction": float(pd.to_numeric(out_df.get("face_detected_fraction"), errors="coerce").mean()) if not out_df.empty else np.nan,
    }
    manifest_json = tables / "landmarks_manifest.json"
    manifest_json.write_text(json.dumps({"summary": summary, "config": config, "rows": rows}, indent=2, default=str), encoding="utf-8")
    return {
        "manifest_csv": manifest_out,
        "manifest_json": manifest_json,
        "n_videos": summary["n_videos"],
        "n_ok": summary["n_ok"],
        "n_error": summary["n_error"],
        "n_skipped_existing": summary["n_skipped_existing"],
        "mean_face_detected_fraction": summary["mean_face_detected_fraction"],
    }
