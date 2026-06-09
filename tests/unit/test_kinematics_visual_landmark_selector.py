from pathlib import Path


def test_visual_landmark_selector_strings_present():
    app = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "LandmarkMeshCanvas" in app
    assert "Load Mesh From Extracted Landmarks" in app
    assert "Save Mesh Preview PNG" in app
    assert "selected_landmark_mesh_preview.png" in app
    assert "MediaPipe face-mesh selector" in app
