from pathlib import Path


def test_feature_gui_v079_filename_column_dropdown_shows_all_columns():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.85.0"' in text
    assert 'Filename column' in text
    assert 'Return all primary-table columns as user-selectable filename sources.' in text
    assert 'for _score, c in scored' in text
    assert 'self.filename_source_combo.addItem(str(c))' in text
    assert 'Filename column' in text


def test_feature_gui_v079_example_uses_user_selected_column_directly():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'Pull one example directly from the user-selected filename column.' in text
    assert 'selected and selected != "Auto-detect" and selected in source_df.columns' in text
    assert 'source_col = selected' in text
    assert 'source_col = self._file_source_column(source_df)' in text


def test_feature_gui_v079_keeps_multitoken_task_template():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'Task start' in text
    assert 'Task end' in text
    assert 'task = "_".join(tokens[lo:hi + 1])' in text
    assert 'parsed_duration' in text
