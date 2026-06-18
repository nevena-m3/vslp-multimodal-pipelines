from pathlib import Path

APP = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')


def _block(start: str, end: str) -> str:
    return APP[APP.index(start):APP.index(end, APP.index(start))]


def test_version_updated_to_v115():
    assert 'APP_VERSION = "v0.115.0"' in APP


def test_longitudinal_menu_is_scaffold_only():
    block = _block('def _longitudinal_page', 'def _screening_page')
    assert 'Longitudinal readiness' in block
    assert 'Feature-family trajectories: next patch' in block
    assert 'Feature trajectories: next patch' in block
    assert 'Feature-family trajectory' not in block
    assert 'Manual QC trajectory' not in block
    assert 'Feature + manual QC alignment' not in block


def test_longitudinal_has_stable_preview_layout():
    block = _block('def _longitudinal_page', 'def _screening_page')
    assert 'QSplitter(Qt.Horizontal)' in block
    assert 'setScaledContents(False)' in block
    assert 'setMaximumHeight(72)' in block
    assert 'setMaximumHeight(96)' in block
    assert 'contentsRect().size()' in block


def test_longitudinal_task_scope_and_readiness_tables_exist():
    assert 'def _longitudinal_scoped_df' in APP
    assert 'def _longitudinal_readiness_tables' in APP
    assert 'ready_for_longitudinal_review' in APP
    assert 'elapsed_days' in APP
    assert 'days_since_previous' in APP


def test_longitudinal_uses_single_safe_plot_key():
    block = _block('def generate_longitudinal_plots', 'def preview_longitudinal_plot')
    assert 'long_readiness' in block
    assert 'plot_longitudinal_subject_records' in block
    assert 'long_session_matrix' not in block
    assert 'long_iteration_counts' not in block
    assert 'long_date_timeline' not in block
