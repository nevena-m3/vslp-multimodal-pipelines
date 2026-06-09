from pathlib import Path

import pandas as pd

from vslp.analysis.kinematics.landmarks import LandmarkRunConfig, write_landmark_plan
from vslp.analysis.kinematics.mediapipe_runtime import _landmark_columns, mediapipe_environment_status


def test_landmark_columns_are_frame_plus_xyz_triplets():
    cols = _landmark_columns(3)
    assert cols[:3] == ["frame", "timestamp_ms", "face_detected"]
    assert cols[3:] == ["0_x", "0_y", "0_z", "1_x", "1_y", "1_z", "2_x", "2_y", "2_z"]


def test_mediapipe_environment_status_is_import_light():
    status = mediapipe_environment_status()
    assert isinstance(status.opencv_available, bool)
    assert isinstance(status.mediapipe_available, bool)
    assert isinstance(status.message, str)


def test_write_landmark_plan_uses_ingest_manifest(tmp_path: Path):
    manifest = tmp_path / "ingest.csv"
    pd.DataFrame([
        {"video_id": "clip1", "source_path": "C:/x/clip1.mp4", "relative_path": "clip1.mp4", "fps": 30.0, "duration_sec": 2.0, "status": "pass"}
    ]).to_csv(manifest, index=False)
    cfg = LandmarkRunConfig(model_path="models/face_landmarker.task", selected_landmarks=(13, 14))
    plan = write_landmark_plan(tmp_path, cfg, manifest_csv=manifest)
    assert plan.exists()
    df = pd.read_csv(plan)
    assert df.loc[0, "video_id"] == "clip1"
    assert df.loc[0, "selected_landmarks"] == "13,14"
