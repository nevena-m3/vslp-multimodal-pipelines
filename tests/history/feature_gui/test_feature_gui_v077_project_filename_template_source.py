from pathlib import Path


def test_feature_gui_v077_project_has_filename_template_controls():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.86.0"' in text
    assert 'self.filename_example_label' in text
    assert 'self.filename_token_combos' in text
    assert 'Subject ID' in text
    assert 'Protocol ID' in text
    assert 'Iteration' in text
    assert 'Recording date' in text
    assert 'Task code' in text
    assert 'Task name' in text
    assert 'Inspect filename example' in text


def test_feature_gui_v077_filename_template_parser_backend():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'def _filename_template_mapping' in text
    assert 'def _filename_template_is_active' in text
    assert 'def refresh_filename_template_ui' in text
    assert 'def _parse_filename_context_by_template' in text
    assert 'parsed_task_code' in text
    assert 'parsed_by_user_template' in text
    assert 'fill_col("task_code", "parsed_task_code")' in text


def test_feature_gui_v077_preserves_safety_boundary():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert '_build_global_context_bar' not in text
    assert 'Apply context + regenerate' not in text
    assert 'warnings.catch_warnings' not in text
    assert 'derive_clinical_context' not in text
    assert 'filename_context_parse_df' in text
