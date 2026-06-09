from vslp.analysis.kinematics import LandmarkRunConfig, mediapipe_capability_note
from vslp.analysis.kinematics.mediapipe_runtime import mediapipe_environment_status


def test_mediapipe_bootstrap_surfaces_status():
    status = mediapipe_environment_status()
    assert hasattr(status, "opencv_available")
    assert hasattr(status, "mediapipe_available")


def test_landmark_config_has_auto_model_defaults():
    cfg = LandmarkRunConfig()
    assert str(cfg.model_path).endswith("face_landmarker.task")
    assert cfg.n_landmarks >= 468
    assert "landmark" in mediapipe_capability_note().lower()
