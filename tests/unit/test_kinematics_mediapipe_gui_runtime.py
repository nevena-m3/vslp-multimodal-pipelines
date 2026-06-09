from pathlib import Path


def test_kinematics_gui_exposes_runtime_install_button():
    app_py = Path(__file__).parents[2] / "src" / "vslp" / "gui" / "kinematics" / "app.py"
    text = app_py.read_text(encoding="utf-8")
    assert "Install / Verify MediaPipe Runtime" in text
    assert "auto_prepare_mediapipe_check" in text
    assert "python -m pip install opencv-python mediapipe" in text
