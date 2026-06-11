from pathlib import Path

APP = Path('src/vslp/gui/features/app.py')


def _src():
    return APP.read_text(encoding='utf-8')


def test_version_is_v098():
    assert 'APP_VERSION = "v0.98.0"' in _src()


def test_feature_mapping_collects_by_source_index_not_raw_role_list():
    s = _src()
    block = s[s.index('    def collect_mapping_from_table'):s.index('    def set_selected_role')]
    assert 'source_index' in block
    assert 'source_column' in block
    assert 'syncing roles by source index/column' in block
    assert 'df["role"] = roles' not in block
    assert 'roles.append' not in block


def test_mapping_table_widgets_store_source_metadata():
    s = _src()
    block = s[s.index('    def refresh_mapping_table'):s.index('    def collect_mapping_from_table')]
    assert 'enumerate(df.index.tolist())' in block
    assert 'combo.setProperty("source_index"' in block
    assert 'combo.setProperty("source_column"' in block


def test_overview_emergency_uses_safe_collectible_mapping_but_collect_is_guarded():
    s = _src()
    assert 'mapping = self.collect_mapping_from_table()' in s
    assert 'table has {table_rows} visible rows but mapping has {len(df)} rows' in s
