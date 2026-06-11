from pathlib import Path


def test_feature_gui_v081_overview_visual_polish_app_source():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.84.0"' in text
    assert 'Overview is the orientation layer' in text
    assert 'border-left:5px solid' in text
    assert 'side_panel.setFixedWidth(340)' in text
    assert 'self.overview_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx // 2, idx % 2)' in text
    assert 'QFrame:hover' in text
    assert 'font-size:25px' in text


def test_feature_gui_v081_overview_design_plot_is_professional_status_board():
    plots = Path("src/vslp/analysis/features/plots.py").read_text(encoding="utf-8")
    assert 'Professional dataset-context status board for Overview' in plots
    assert 'Observed values' in plots
    assert 'pill-style status marker' in plots
    assert 'context fields detected' in plots
    assert 'Missing clinical fields are acceptable when metadata is absent' in plots


def test_feature_gui_v081_does_not_change_context_or_analysis_logic():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'def _overview_context_detection_table' in text
    assert 'def _missingness_scope_table' in text
    assert '_build_global_context_bar' not in text
    assert 'warnings.catch_warnings' not in text
    assert 'derive_clinical_context' not in text
