from pathlib import Path


def test_visual_landmark_selector_strings_present():
    app = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "LandmarkMeshCanvas" in app
    assert "Load Selected Frame" in app
    assert "Save Overlay Preview PNG" in app
    assert "selected_landmark_video_overlay_preview.png" in app
    assert "Real video frame + Google MediaPipe overlay" in app
