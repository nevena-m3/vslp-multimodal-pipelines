from pathlib import Path


def test_feature_gui_v067_qc_matches_overview_style_layout():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.77.0"' in text
    block = text[text.index('def _qc_page'):text.index('def update_qc_dashboard')]
    assert 'QC snapshot' in block
    assert 'Detailed QC integration tables' in block
    assert 'Open current plot' in block
    assert 'qc_split = QHBoxLayout()' in block
    assert 'side_panel.setFixedWidth(300)' in block
    assert 'self.qc_plot_preview.setMinimumHeight(520)' in block
    assert 'Feature x QC-family heatmap' in block
    assert 'Selected feature x selected QC' in block
    assert 'does not duplicate Missingness, Distributions, Relationships, Screening, or ML Export' in block


def test_feature_gui_v067_qc_snapshot_is_vertical_stack():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    block = text[text.index('def update_qc_dashboard'):text.index('def _selected_qc_feature')]
    assert 'self.qc_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx, 0)' in block
    assert '("QC table", "yes" if str(metric_value("qc_table_loaded", False)).lower() == "true" else "no", "artifact context")' in block
    assert '("Monitor pairs", metric_value("feature_qc_pairs_abs_rho_ge_0_30", 0), "abs rho >= .30")' in block
    assert '("Review pairs", metric_value("feature_qc_pairs_abs_rho_ge_0_50", 0), "abs rho >= .50")' in block
