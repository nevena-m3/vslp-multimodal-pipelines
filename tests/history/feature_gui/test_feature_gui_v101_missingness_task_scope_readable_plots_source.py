from pathlib import Path

APP = Path('src/vslp/gui/features/app.py')
PLOTS = Path('src/vslp/analysis/features/plots.py')


def _app():
    return APP.read_text(encoding='utf-8')


def _plots():
    return PLOTS.read_text(encoding='utf-8')


def test_v101_version_and_missingness_plot_labels_are_readable():
    src = _app()
    assert 'APP_VERSION = "v0.108.0"' in (read(APP) if 'read' in globals() else APP.read_text(encoding='utf-8'))
    assert 'Feature availability summary' in src
    assert 'Co-missingness pair summary' in src
    assert 'self.missing_plot_combo.addItem("Feature availability heatmap"' not in src
    assert 'self.missing_plot_combo.addItem("Co-missingness clusters"' not in src


def test_v101_missingness_task_focus_uses_metadata_first_context_column():
    src = _app()
    assert 'def _missingness_task_column' in src
    helper = src[src.index('def _missingness_task_column'):src.index('def _missingness_context_series')]
    assert 'metadata__task' in helper
    assert 'parsed_task' in helper
    assert 'Filename-derived' in helper or 'Filename' in helper
    refresh = src[src.index('def _refresh_missingness_task_combo'):src.index('def _missingness_scope_table')]
    assert '_missingness_task_column(df)' in refresh
    scope = src[src.index('def _missingness_scope_table'):src.index('def _missingness_feature_cols')]
    assert '_missingness_task_column(scoped)' in scope


def test_v101_missingness_scope_regenerates_new_bar_summaries_not_heatmaps():
    src = _app()
    body = src[src.index('def generate_missingness_scope_plots'):src.index('def regenerate_missingness_scope_plots')]
    assert 'plot_feature_availability_bars' in body
    assert 'plot_comissing_pair_bars' in body
    assert 'missingness_comissing_pairs' in src
    assert 'plot_feature_availability_heatmap(df, feature_cols' not in body
    assert 'plot_comissing_heatmap(df, feature_cols' not in body
    assert 'feature_availability_heatmap' in body  # backward-compatible alias only
    assert 'missingness_comissing_heatmap' in body


def test_v101_new_plot_helpers_are_bar_based():
    plots = _plots()
    assert 'def plot_feature_availability_bars' in plots
    assert 'def plot_comissing_pair_bars' in plots
    availability = plots[plots.index('def plot_feature_availability_bars'):plots.index('def plot_comissing_pair_bars')]
    pair = plots[plots.index('def plot_comissing_pair_bars'):plots.index('def plot_distribution_grid')]
    assert 'ax.barh' in availability
    assert 'imshow' not in availability
    assert 'ax.barh' in pair
    assert 'imshow' not in pair
