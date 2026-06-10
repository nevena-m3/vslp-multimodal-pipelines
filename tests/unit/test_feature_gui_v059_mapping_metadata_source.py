from pathlib import Path


def test_feature_gui_v059_version_mapping_confirmation_and_no_wheel_combo():
    source = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.59.0"' in source
    assert 'class NoWheelComboBox(QComboBox)' in source
    assert 'def wheelEvent(self, event)' in source
    assert 'Confirm modified column mapping' in source
    assert 'accepted_column_mapping.csv' in source


def test_feature_gui_v059_metadata_context_and_plot_first_layout():
    source = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'def _merge_metadata_context' in source
    assert 'metadata_key_join:' in source
    assert 'metadata_row_order_join:same_row_count' in source
    assert 'metadata_context' in source
    assert 'Detailed tables' in source
