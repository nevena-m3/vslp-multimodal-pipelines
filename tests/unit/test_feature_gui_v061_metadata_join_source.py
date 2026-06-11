from pathlib import Path


def test_feature_gui_v061_metadata_alias_and_file_join_source():
    text = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
    assert 'APP_VERSION = "v0.75.0"' in text
    assert '_canonical_metadata_name' in text
    assert 'raw_media_file_name' in text
    assert 'subjectid' in text
    assert 'task_name' in text
    assert '_match_file_basename' in text
    assert 'metadata_key_join' in text


def test_feature_gui_v061_design_plot_is_compact_and_empty_aware():
    text = Path('src/vslp/analysis/features/plots.py').read_text(encoding='utf-8')
    assert 'Dataset design context' in text
    assert 'empty column' in text
    assert 'Available means a detected column has at least one non-missing value' in text
    assert 'large card tiles' in text
