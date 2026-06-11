from pathlib import Path


def test_feature_gui_v058_version_and_run_log_source():
    source = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
    assert 'APP_VERSION = "v0.74.0"' in source
    assert 'Run Log' in source
    assert 'run_progress' in source


def test_feature_gui_v058_scroll_and_table_visibility_source():
    source = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
    assert 'ScrollPerPixel' in source
    assert 'setAlternatingRowColors(True)' in source
