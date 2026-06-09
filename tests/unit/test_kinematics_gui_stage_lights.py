from pathlib import Path


def test_sidebar_stage_status_lights_are_declared() -> None:
    source = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "self.stage_status_dots" in source
    assert "StageStatusDot" in source
    assert "#22C55E" in source
    assert "Complete" in source


def test_landmark_extraction_dashboard_cards_are_declared() -> None:
    source = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "Landmark extraction readiness" in source
    assert "self.landmark_metric_labels" in source
    assert "_update_landmark_dashboard_from_manifest" in source
    assert "mean detected-frame fraction" in source
