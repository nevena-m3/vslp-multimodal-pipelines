from pathlib import Path

APP = Path('src/vslp/gui/features/app.py')


def _reliability_page_source():
    text = APP.read_text(encoding='utf-8')
    start = text.index('    def _reliability_page(self) -> QWidget:')
    end = text.index('    def _selected_reliability_source(self) -> str:', start)
    return text[start:end]


def test_v129_version_is_set():
    assert 'APP_VERSION = "v0.129.0"' in APP.read_text(encoding='utf-8')


def test_reliability_controls_are_above_plot_and_not_in_lower_details_card():
    src = _reliability_page_source()
    assert 'control_panel = QFrame()' in src
    assert 'self.reliability_source_combo = QComboBox()' in src
    assert 'self.reliability_task_combo = QComboBox()' in src
    assert 'self.reliability_family_combo = QComboBox()' in src
    assert 'self.reliability_feature_combo = QComboBox()' in src
    assert src.index('control_panel = QFrame()') < src.index('plot_panel = QFrame()')
    assert src.index('card.layout.addWidget(control_panel)') < src.index('card.layout.addLayout(reliability_split)')


def test_reliability_details_are_side_panel_not_bottom_card():
    src = _reliability_page_source()
    assert 'side_panel = QFrame()' in src
    assert 'Reliability details' in src
    assert 'self.reliability_metric_grid = QGridLayout()' in src
    assert src.index('plot_panel = QFrame()') < src.index('side_panel = QFrame()') < src.index('card.layout.addLayout(reliability_split)')
    assert 'Detailed reliability tables' in src
    assert src.index('card.layout.addLayout(reliability_split)') < src.index('Detailed reliability tables')


def test_reliability_gallery_is_inline_like_other_menus():
    src = _reliability_page_source()
    assert '_add_standard_plot_gallery' not in src
    assert 'self.reliability_plot_combo = QComboBox()' in src
    assert 'show_btn.clicked.connect(lambda: self.preview_reliability_plot(self.reliability_plot_combo.currentData()))' in src
    assert 'regen.clicked.connect(lambda: self.update_reliability_dashboard(getattr(self, "outputs", {})))' in src
    assert 'self.reliability_plot_preview = QLabel' in src
    assert 'self.reliability_interpretation_label = QLabel' in src
