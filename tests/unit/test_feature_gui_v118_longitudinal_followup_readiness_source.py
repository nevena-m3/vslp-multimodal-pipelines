from pathlib import Path

APP = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
PLOTS = Path("src/vslp/analysis/features/plots.py").read_text(encoding="utf-8")


def test_version_updated_to_v118():
    assert 'APP_VERSION = "v0.118.0"' in APP


def test_longitudinal_menu_stays_single_safe_view():
    block = APP[APP.index('def _longitudinal_page'):APP.index('def _active_analysis_table')]
    assert 'Follow-up depth and readiness' in block
    assert 'Feature-family trajectory' not in block
    assert 'Manual QC trajectory' not in block
    assert 'Feature + manual QC alignment' not in block


def test_longitudinal_caption_is_followup_focused():
    assert 'elapsed days' in APP or 'follow-up spacing' in APP
    assert 'feature-change and manual-QC trajectory plots will be added after this view is accepted' in APP


def test_plot_readiness_is_clinician_followup_map():
    assert 'def plot_longitudinal_readiness' in PLOTS
    assert 'Longitudinal follow-up depth' in PLOTS
    assert 'Days from first to last dated visit' in PLOTS
    assert 'date missing |' in PLOTS


def test_stable_preview_still_present():
    assert 'setScaledContents(False)' in APP
    assert 'Qt.KeepAspectRatio' in APP
