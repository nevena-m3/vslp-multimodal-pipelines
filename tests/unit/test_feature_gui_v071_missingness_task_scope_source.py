from pathlib import Path


def test_feature_gui_v071_missingness_has_local_task_scope():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.77.0"' in text
    assert 'self.missing_task_combo' in text
    assert 'def _refresh_missingness_task_combo' in text
    assert 'def _missingness_scope_table' in text
    assert 'def generate_missingness_scope_plots' in text
    assert 'def regenerate_missingness_scope_plots' in text
    assert 'regen.clicked.connect(self.regenerate_missingness_scope_plots)' in text
    assert 'Current scope:' in text


def test_feature_gui_v071_plot_preview_uses_stable_display_helper():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'def _display_plot_image' in text
    assert 'label.clear()' in text
    assert 'self._display_plot_image(self.missing_plot_preview, path)' in text
    assert 'pix.scaled(self.missing_plot_preview.size()' not in text


def test_feature_gui_v071_missingness_heatmaps_use_all_features_when_feasible():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'plot_feature_availability_heatmap(df, feature_cols' in text
    assert 'max_features=None, max_rows=None' in text
    assert 'plot_comissing_heatmap(df, feature_cols' in text
    assert 'max_features=None)' in text
    plots = Path("src/vslp/analysis/features/plots.py").read_text(encoding="utf-8")
    assert 'def plot_comissing_heatmap(feature_df: pd.DataFrame, feature_cols: Sequence[str], path: Path, max_features: int | None = None)' in plots
    assert 'cols = miss_fr.index.tolist() if max_features is None' in plots
