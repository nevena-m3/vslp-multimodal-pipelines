from pathlib import Path


def test_feature_gui_v066_distributions_matches_overview_missingness_layout():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.75.0"' in text
    block = text[text.index('def _distributions_page'):text.index('def update_distribution_dashboard')]
    assert 'Distribution snapshot' in block
    assert 'Detailed distribution tables' in block
    assert 'Open current plot' in block
    assert 'dist_split = QHBoxLayout()' in block
    assert 'side_panel.setFixedWidth(300)' in block
    assert 'self.dist_plot_preview.setMinimumHeight(520)' in block
    assert 'Review status' in block
    assert 'Selected feature diagnostic' in block
    assert 'does not duplicate Missingness, QC Integration, Relationships, Screening, or ML Export' in block


def test_feature_gui_v066_distributions_snapshot_is_vertical_stack():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    block = text[text.index('def update_distribution_dashboard'):text.index('def _selected_distribution_feature')]
    assert 'self.dist_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx, 0)' in block
    assert '("Features", n_features, "numeric predictors")' in block
    assert '("Zero variance", zero_n, "not useful for ML")' in block
