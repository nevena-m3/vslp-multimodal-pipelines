from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = (ROOT / "src" / "vslp" / "gui" / "features" / "app.py").read_text(encoding="utf-8")
PLOTS = (ROOT / "src" / "vslp" / "analysis" / "features" / "plots.py").read_text(encoding="utf-8")


def test_version_updated_to_v121():
    assert 'APP_VERSION = "v0.121.0"' in APP


def test_longitudinal_adds_single_feature_family_change_plot():
    block = APP[APP.index('def _longitudinal_page'):APP.index('def _active_analysis_table')]
    assert 'Repeated-record cohort summary' in block
    assert 'Selected subject-task visit timeline' in block
    assert 'Selected subject-task feature-family change' in block
    assert 'longitudinal_feature_family_trajectory' in block


def test_feature_family_controls_are_present_but_scoped():
    block = APP[APP.index('def _longitudinal_page'):APP.index('def _active_analysis_table')]
    assert 'Feature family:' in block
    assert 'longitudinal_family_combo' in APP
    assert 'All mapped features' in APP


def test_feature_change_requires_same_task_and_repeated_subject():
    method = APP[APP.index('def _longitudinal_feature_family_change_table'):APP.index('def update_longitudinal_dashboard')]
    assert 'Select one task first' in method
    assert 'same task' in method or 'same-task' in method
    assert 'Selected subject has fewer than two recordings for the selected task' in method
    assert 'standardized change from baseline' in method or 'change_from_baseline' in method


def test_plot_function_and_generation_are_wired():
    assert 'plot_longitudinal_feature_family_trajectory' in APP
    assert 'def plot_longitudinal_feature_family_trajectory' in PLOTS
    assert 'longitudinal_feature_family_trajectory.png' in APP
    assert 'Feature-family change' in APP
