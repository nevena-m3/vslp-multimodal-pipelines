from pathlib import Path

APP = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')


def test_metadata_collision_rename_helper_exists():
    assert 'def _metadata_collision_rename_map(' in APP
    assert 'metadata__diagnosis__dup' in APP or 'self._unique_column_name(base, used)' in APP
    assert 'target name unique across the feature frame' in APP


def test_metadata_key_join_uses_safe_collision_map_before_merge():
    assert 'rename_map, duplicate_canonical_cols = self._metadata_collision_rename_map(' in APP
    assert 'right = self._ensure_unique_columns(right, "Metadata table merge slice")' in APP
    assert 'merged = self._ensure_unique_columns(merged, "Analysis table after metadata merge")' in APP


def test_metadata_row_order_join_uses_safe_collision_map():
    assert 'helper_renames: dict[str, str] = {}' in APP
    assert 'collision_map, duplicate_canonical_cols = self._metadata_collision_rename_map(' in APP
    assert 'Metadata row-order slice' in APP
    assert 'Analysis table after row-order metadata merge' in APP


def test_duplicate_metadata_fill_handles_dataframe_selection():
    assert 'if isinstance(fill_values, pd.DataFrame):' in APP
    assert 'fill_values = fill_values.iloc[:, 0]' in APP
