from pathlib import Path


def test_landmark_overlay_uses_tunable_review_grade_rendering():
    app = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "Review-grade rendering" in app
    assert "overlay_style_combo" in app
    assert "Subtle" in app
    assert "Balanced" in app
    assert "High contrast" in app
    assert "show_selected_labels" in app


def test_landmark_frame_navigation_uses_scrubber_not_step_buttons():
    app = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "QAbstractSpinBox.NoButtons" in app
    assert "Load Selected Frame" in app
    assert "Use Representative Detected Frame" in app
    assert "use_representative_landmark_frame" in app
    assert "_set_landmark_frame_value" in app
    assert 'QPushButton("Prev")' not in app
    assert 'QPushButton("Next")' not in app
    assert 'QPushButton("-10")' not in app
    assert 'QPushButton("+10")' not in app
