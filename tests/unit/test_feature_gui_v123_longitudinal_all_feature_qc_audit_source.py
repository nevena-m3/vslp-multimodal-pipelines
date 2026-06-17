from pathlib import Path

APP = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
PLOTS = Path('src/vslp/analysis/features/plots.py').read_text(encoding='utf-8')


def test_version_updated_to_v123():
    assert 'APP_VERSION = "v0.123.0"' in APP


def test_longitudinal_adds_qc_change_audit_plot():
    assert 'Selected subject-task QC change audit' in APP
    assert 'longitudinal_qc_change_audit' in APP
    assert 'plot_longitudinal_qc_change_audit' in APP
    assert 'def plot_longitudinal_qc_change_audit' in PLOTS


def test_feature_change_plot_no_top_feature_cap():
    block = PLOTS[PLOTS.index('def plot_longitudinal_feature_family_trajectory'):PLOTS.index('def plot_longitudinal_qc_change_audit')]
    assert 'All selected-family features are included' in block
    assert '.head(8)' not in block
    assert 'pivot_table' in block
    assert 'All {n_features} selected-family features are shown' in block


def test_direction_is_registry_only_and_not_inferred():
    assert 'def _interpret_longitudinal_change_direction' in APP
    assert 'Do not infer direction from feature names' in APP
    assert 'clinical direction not specified' in APP
    assert 'better' in APP and 'worse' in APP


def test_qc_change_table_handles_manual_and_automated_sources():
    block = APP[APP.index('def _longitudinal_qc_change_table'):APP.index('def update_longitudinal_dashboard')]
    assert '_manual_qc_dataframe' in block
    assert 'Manual QC' in block
    assert 'Automated QC' in block
    assert 'qc_family_from_name' in block
    assert 'len(q) == len(df)' in block


def test_show_still_regenerates_selected_longitudinal_plot():
    block = APP[APP.index('def preview_longitudinal_plot'):APP.index('def open_current_longitudinal_plot')]
    assert 'self.generate_longitudinal_plots()' in block
    assert 'longitudinal_qc_change_audit' in block
