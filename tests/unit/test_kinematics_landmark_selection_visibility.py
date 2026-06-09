from pathlib import Path


def test_landmark_overlay_uses_high_contrast_rendering():
    app = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "High-contrast landmark rendering" in app
    assert "selected_point_radius" in app
    assert "#22C55E" in app
    assert "QColor(0, 0, 0, 235)" in app
    assert "bright rims" in app


def test_landmark_frame_navigation_uses_explicit_buttons_not_spin_arrows():
    app = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "QAbstractSpinBox.NoButtons" in app
    assert "_step_landmark_frame" in app
    assert "QPushButton(\"Prev\")" in app
    assert "QPushButton(\"Next\")" in app
    assert "QPushButton(\"-10\")" in app
    assert "QPushButton(\"+10\")" in app
