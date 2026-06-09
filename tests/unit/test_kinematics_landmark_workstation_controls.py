from pathlib import Path


def test_landmark_workstation_zoom_and_display_controls_present():
    app = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "Overlay display" in app
    assert "Auto-zoom to detected face" in app
    assert "Zoom Face" in app
    assert "set_landmark_display_mode" in app
    assert "_landmark_video_changed" in app
    assert "wheelEvent" in app
    assert "zoom_factor" in app


def test_landmark_frame_navigation_controls_present():
    app_py = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "_landmark_frame_choice_changed" in app_py
    assert "_landmark_spin_changed" in app_py
    assert "Frame review is intentionally explicit" in app_py
    assert "_schedule_landmark_frame_reload" in app_py
    assert "#07111F" in app_py
