from pathlib import Path

APP = Path("src/vslp/gui/features/app.py")


def _app():
    return APP.read_text(encoding="utf-8")


def test_v102_version_and_scope_output_helpers_exist():
    src = _app()
    assert 'APP_VERSION = "v0.108.0"' in (read(APP) if 'read' in globals() else APP.read_text(encoding='utf-8'))
    assert 'def _missingness_scope_outputs' in src
    assert 'def _update_missingness_snapshot' in src
    assert 'def _update_missingness_tables' in src


def test_v102_missingness_tables_are_filled_from_current_scope_outputs():
    src = _app()
    body = src[src.index('def generate_missingness_scope_plots'):src.index('def regenerate_missingness_scope_plots')]
    assert 'self.current_missingness_outputs = outputs' in body
    assert 'self._update_missingness_snapshot(outputs, scope_label)' in body
    assert 'self._update_missingness_tables(outputs)' in body
    assert 'missingness_scope_summary' in src


def test_v102_update_dashboard_rebuilds_scoped_outputs_not_stale_full_outputs():
    src = _app()
    body = src[src.index('def update_missingness_dashboard'):src.index('def _missingness_plot_caption_text')]
    assert 'scoped_outputs' in body
    assert 'self._missingness_scope_outputs()' in body
    assert 'self._update_missingness_snapshot(scoped_outputs, scope_label)' in body
    assert 'self._update_missingness_tables(scoped_outputs)' in body


def test_v102_availability_snapshot_has_defined_purpose():
    src = _app()
    assert 'Current-scope missingness metrics. Updates with Task focus and Context/Value.' in src
    assert 'Scope' in src
    assert 'High-missing rows' in src
