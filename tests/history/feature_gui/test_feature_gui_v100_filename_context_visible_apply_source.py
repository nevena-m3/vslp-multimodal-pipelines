from pathlib import Path

APP = Path('src/vslp/gui/features/app.py')
PLOTS = Path('src/vslp/analysis/features/plots.py')


def _src():
    return APP.read_text(encoding='utf-8')


def test_v100_version_and_apply_has_visible_side_effects():
    src = _src()
    assert 'APP_VERSION = "v0.108.0"' in (read(APP) if 'read' in globals() else APP.read_text(encoding='utf-8'))
    body = src[src.index('def apply_filename_context_from_metadata_mapping'):src.index('def _style_metadata_role_combo')]
    assert 'Applying filename context from Metadata Mapping fallback using selected column' in body
    assert 'self.filename_context_applied = True' in body
    assert 'filename_context_parse_preview.csv' in body
    assert 'Downstream menus now use these parsed context fields' in body


def test_v100_task_span_defaults_to_final_token_and_preserves_text():
    src = _src()
    assert 'def _filename_task_from_span' in src
    helper = src[src.index('def _filename_task_from_span'):src.index('def _parse_filename_context_by_template')]
    assert 'end_idx = len(tokens) - 1' in helper
    assert '"_".join(span)' in helper
    parser = src[src.index('def _parse_filename_context_by_template'):src.index('def _parse_filename_context_frame')]
    assert 'task = self._filename_task_from_span(tokens, task_start_idx, task_end_idx)' in parser
    assert 'str(task).strip()' in parser
    assert '.upper()' not in parser


def test_v100_selected_filename_column_remains_explicit_source():
    src = _src()
    body = src[src.index('def apply_filename_context_from_metadata_mapping'):src.index('def _style_metadata_role_combo')]
    assert 'selected_col = self._selected_filename_source_column(self.feature_df)' in body
    assert '_parse_filename_context_frame(base_df, source_col=selected_col)' in body


def test_v100_overview_task_labels_not_truncated_in_plotting_source():
    plots = PLOTS.read_text(encoding='utf-8')
    assert 'labels = [str(x) for x in sub["level"]]' in plots
    assert 'ax.set_yticklabels([str(x) for x in df["task"]]' in plots
    assert 'fig.subplots_adjust(left=0.17, right=0.98)' in plots
    assert 'fig.subplots_adjust(left=0.26, right=0.98)' in plots
