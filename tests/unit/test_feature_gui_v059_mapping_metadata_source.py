from pathlib import Path


def test_feature_gui_v059_version_mapping_confirmation_and_no_wheel_combo():
    source = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
    assert 'APP_VERSION = "v0.65.0"' in source
    assert 'NoWheelComboBox' in source
    assert 'Confirm modified column mapping' in source
    assert 'accepted_column_mapping.csv' in source


def test_feature_gui_v059_metadata_context_source():
    source = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
    assert 'Metadata context strategy' in source
    assert '_merge_metadata_context' in source
    assert 'metadata_key_join' in source
