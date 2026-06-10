from pathlib import Path


def test_feature_gui_v058_version_and_run_log_source():
    source = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
    assert 'APP_VERSION = "v0.59.0"' in source
    assert 'Run Log · Feature Analysis GUI {APP_VERSION}' in source
    assert 'self.run_log = QTextEdit()' in source
    assert 'self.run_progress = QProgressBar()' in source
    assert 'def log_error(self, context: str, exc: Exception)' in source


def test_feature_gui_v058_scrolling_and_table_visibility_source():
    source = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
    assert 'sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)' in source
    assert 'sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)' in source
    assert 'widget.setMinimumWidth(980)' in source
    assert 'table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)' in source
    assert 'table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)' in source
