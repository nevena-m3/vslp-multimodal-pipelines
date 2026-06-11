from pathlib import Path

APP = Path('src/vslp/gui/features/app.py')


def _src():
    return APP.read_text(encoding='utf-8')


def test_v099_version_and_selected_filename_helper():
    src = _src()
    assert 'APP_VERSION = "v0.99.0"' in src
    assert 'def _selected_filename_source_column' in src
    assert 'return selected' in src
    assert 'Apply, which made the button appear to do nothing' in src


def test_apply_filename_context_uses_selected_column_directly():
    src = _src()
    body = src[src.index('def apply_filename_context_from_metadata_mapping'):src.index('def _style_metadata_role_combo')]
    assert 'selected_col = self._selected_filename_source_column(self.feature_df)' in body
    assert '_parse_filename_context_frame(base_df, source_col=selected_col)' in body
    assert 'Filename-derived context applied. Parsed rows:' in body


def test_filename_parser_preserves_multitoken_task_spans():
    src = _src()
    parser = src[src.index('def _parse_filename_context_by_template'):src.index('def _parse_filename_context_frame')]
    assert 'lo, hi = sorted([task_start_idx, task_end_idx])' in parser
    assert 'task = "_".join(tokens[lo:hi + 1])' in parser
    assert 'self._filename_tokens_from_stem' in src
    assert 'Final token:' in src


def test_no_metadata_merge_uses_filename_fallback_only():
    src = _src()
    merge = src[src.index('def _merge_metadata_context'):src.index('feature_base = self._standardize_feature_match_helpers(feature_df)', src.index('def _merge_metadata_context') + 1)]
    assert 'if self.meta_df is None or self.meta_df.empty:' in merge
    assert '_fill_empty_context_from_filename(feature_base)' in src
