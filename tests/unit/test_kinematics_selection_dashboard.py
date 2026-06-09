from pathlib import Path

from vslp.analysis.kinematics.selection import analyze_landmark_selection, write_selected_landmarks


def test_selection_diagnostics_complete_for_auditable_oral_set(tmp_path):
    selected = [13, 14, 61, 291, 78, 308, 17, 152, 0, 133, 362, 33, 263]
    diagnostics = analyze_landmark_selection(selected)
    assert diagnostics.status == "Complete"
    assert diagnostics.n_selected == len(selected)
    assert "mouth/lips" in diagnostics.region_counts
    assert not diagnostics.warning_flags


def test_selection_diagnostics_review_when_required_anchors_missing():
    diagnostics = analyze_landmark_selection([13, 14, 61, 291])
    assert diagnostics.status == "Review"
    assert "missing_required_landmarks" in diagnostics.warning_flags
    anchor = next(r for r in diagnostics.requirement_results if r.requirement_id == "normalization_intercanthal")
    assert anchor.status == "FAIL"
    assert anchor.missing_required == (133, 362)


def test_write_selected_landmarks_outputs_json_summary_and_requirements(tmp_path):
    outputs = write_selected_landmarks(
        tmp_path,
        [13, 14, 61, 291, 78, 308, 17, 152, 0, 133, 362, 33, 263],
        preset="unit-test",
        app_version="test-version",
    )
    assert outputs["selected_json"].exists()
    assert outputs["summary_csv"].exists()
    assert outputs["requirements_csv"].exists()
    text = outputs["selected_json"].read_text(encoding="utf-8")
    assert "selection_status" in text
    assert "test-version" in text


def test_selection_dashboard_strings_present_in_gui():
    source = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "Selection readiness and scientific coverage" in source
    assert "self.selection_metric_labels" in source
    assert "selection_requirement_table" in source
    assert "write_selected_landmarks" in source
