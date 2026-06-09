from pathlib import Path


def test_landmark_overlay_uses_quiet_tunable_review_grade_rendering():
    app = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "Review-grade rendering" in app
    assert "overlay_style_combo" in app
    assert "Subtle" in app
    assert "Balanced" in app
    assert "High contrast" in app
    assert "landmark_display_combo" in app
    assert "All faint" in app
    assert "Selected + anchors" in app


def test_landmark_frame_navigation_uses_bookmarks_not_step_buttons_or_scrubber():
    app = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "QAbstractSpinBox.NoButtons" in app
    assert "Load Frame" in app
    assert "Load middle detected frame" in app
    assert "Detected-frame bookmark" in app
    assert "_frame_choices_from_landmarks" in app
    assert "_populate_landmark_frame_choices" in app
    assert "use_representative_landmark_frame" in app
    assert "_set_landmark_frame_value" in app
    assert 'QPushButton("Prev")' not in app
    assert 'QPushButton("Next")' not in app
    assert 'QPushButton("-10")' not in app
    assert 'QPushButton("+10")' not in app
    assert "QSlider" not in app
