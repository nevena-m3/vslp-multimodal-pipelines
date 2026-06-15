from pathlib import Path

APP = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
PLOTS = Path('src/vslp/analysis/features/plots.py').read_text(encoding='utf-8')


def test_version_updated_to_v119():
    assert 'APP_VERSION = "v0.119.0"' in APP


def test_longitudinal_keeps_one_safe_repeated_cohort_plot():
    block = APP[APP.index('def _longitudinal_page'):APP.index('def _active_analysis_table')]
    assert 'Repeated-record cohort summary' in block
    assert 'Longitudinal readiness' not in block
    assert 'Feature-family trajectory' not in block
    assert 'Manual QC trajectory' not in block


def test_longitudinal_snapshot_uses_repeated_units_and_robust_stats():
    block = APP[APP.index('def update_longitudinal_dashboard'):APP.index('def generate_longitudinal_plots')]
    assert 'repeated_df' in block
    assert 'Repeated subjects' in block
    assert 'Median / mean records' in block
    assert 'Median / mean span' in block
    assert 'n_records"' in block and '>= 2' in block


def test_longitudinal_tables_are_repeated_only():
    block = APP[APP.index('def update_longitudinal_dashboard'):APP.index('def generate_longitudinal_plots')]
    assert 'repeated_for_tables' in block
    assert 'No subject-task unit has at least two recordings' in block
    assert 'Repeated subject-task units' in APP


def test_repeated_cohort_plot_is_aggregate_not_per_subject_topn():
    block = PLOTS[PLOTS.index('def plot_longitudinal_readiness'):PLOTS.index('def plot_longitudinal_session_matrix')]
    assert 'Repeated-record cohort summary' in block
    assert 'Recording depth' in block
    assert 'Follow-up span distribution' in block
    assert 'Repeated coverage by task' in block
    assert 'subject-task units with at least two recordings' in block
