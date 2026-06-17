from pathlib import Path

APP = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
PLOTS = Path('src/vslp/analysis/features/plots.py').read_text(encoding='utf-8')


def test_version_updated_to_v120():
    assert 'APP_VERSION = "v0.120.0"' in APP


def test_longitudinal_adds_exactly_one_next_plot():
    block = APP[APP.index('def _longitudinal_page'):APP.index('def _active_analysis_table')]
    assert 'Repeated-record cohort summary' in block
    assert 'Selected subject-task visit timeline' in block
    assert 'Feature-family trajectory' not in block
    assert 'Manual QC trajectory' not in block


def test_subject_focus_is_repeated_only_for_longitudinal():
    block = APP[APP.index('def _refresh_longitudinal_subject_combo'):APP.index('def _longitudinal_scope_df')]
    assert 'n_records' in block
    assert '>= 2' in block
    assert 'Select repeated subject' in block


def test_visit_records_table_uses_selected_subject_and_temporal_fields():
    block = APP[APP.index('def _longitudinal_visit_records_table'):APP.index('def update_longitudinal_dashboard')]
    assert '_longitudinal_selected_subject' in block
    assert 'days_since_first' in block
    assert 'interval_from_previous_days' in block
    assert 'session_or_visit' in block
    assert 'iteration' in block


def test_visit_timeline_plot_function_present_and_stable():
    assert 'plot_longitudinal_visit_timeline' in APP
    block = PLOTS[PLOTS.index('def plot_longitudinal_visit_timeline'):PLOTS.index('def plot_longitudinal_session_matrix')]
    assert 'Selected subject-task visit timeline' in block
    assert 'Days since first dated recording' in block
    assert 'Recording order' in block
    assert 'Verify this timeline before interpreting feature trajectories' in block
