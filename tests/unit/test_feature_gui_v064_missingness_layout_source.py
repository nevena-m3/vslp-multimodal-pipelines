from pathlib import Path


def test_feature_gui_v064_missingness_matches_overview_layout():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.69.0"' in text
    block = text[text.index('def _missingness_page'):text.index('def update_missingness_dashboard')]
    assert 'Availability snapshot' in block
    assert 'Detailed missingness tables' in block
    assert 'Open current plot' in block
    assert 'missing_split = QHBoxLayout()' in block
    assert 'side_panel.setFixedWidth(300)' in block
    assert 'Feature missingness burden' in block
    assert 'Recording missingness burden' in block


def test_feature_gui_v064_missingness_is_not_redundant_with_other_menus():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    block = text[text.index('def _missingness_page'):text.index('def update_missingness_dashboard')]
    assert 'Top missing features' not in text[text.index('def _overview_page'):text.index('def _metric_tile')]
    assert 'Distribution shape' not in block[block.index('self.missing_plot_combo'):block.index('plot_header.addWidget(self.missing_plot_combo')]
    assert 'does not duplicate distribution, QC, relationship, screening, or ML-export plots' in block
