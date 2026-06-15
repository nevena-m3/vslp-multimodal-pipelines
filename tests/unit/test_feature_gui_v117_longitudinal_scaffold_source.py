from pathlib import Path

APP = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
PLOTS = Path('src/vslp/analysis/features/plots.py').read_text(encoding='utf-8')


def test_version_updated_to_v117():
    assert 'APP_VERSION = "v0.117.0"' in APP


def test_longitudinal_menu_exposes_only_readiness_view():
    block = APP[APP.index('def _longitudinal_page'):APP.index('def _active_analysis_table')]
    assert 'Longitudinal readiness' in block
    assert 'Subject repeats' not in block
    assert 'Session / visit structure' not in block
    assert 'Iteration coverage' not in block
    assert 'Visit-date coverage' not in block


def test_longitudinal_has_task_scope_and_stable_preview():
    block = APP[APP.index('def _longitudinal_page'):APP.index('def _active_analysis_table')]
    assert 'longitudinal_task_combo' in block
    assert 'setScaledContents(False)' in block
    assert 'Feature family' not in block


def test_longitudinal_generation_only_writes_readiness_plot():
    block = APP[APP.index('def generate_longitudinal_plots'):APP.index('def preview_longitudinal_plot')]
    assert 'plot_longitudinal_readiness' in block
    assert 'longitudinal_readiness' in block
    assert 'long_session_matrix' not in block
    assert 'long_iteration_counts' not in block
    assert 'long_date_timeline' not in block


def test_readiness_plot_function_exists():
    assert 'def plot_longitudinal_readiness' in PLOTS
    assert 'subject-task' in PLOTS
