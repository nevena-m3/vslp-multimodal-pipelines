from pathlib import Path


def test_v087_filename_template_regex_fixed_and_apply_does_not_reset_template():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.' in text
    assert 're.match(r"Token\\s+(\\d+)\\b", val)' in text
    assert 'Do not refresh token controls here; the user-selected template is the source of truth.' in text
    apply_block = text[text.index('def apply_filename_context_from_metadata_mapping'):text.index('def refresh_metadata_mapping_table')]
    assert 'self.refresh_filename_template_ui()' not in apply_block


def test_v087_no_metadata_hides_metadata_role_buttons_and_has_role_filters():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'self.metadata_quick_role_widget.setVisible(False)' in text
    assert 'self.update_metadata_mapping_mode_visibility()' in text
    for label in ["Recording identity", "Core clinical", "Demographics", "Media / device", "Manual metadata QC", "Administrative"]:
        assert label in text


def test_v087_parsed_count_fallback_uses_actual_context_columns():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'context_cols = [c for c in ["parsed_subject_id", "parsed_task", "parsed_recording_date"]' in text
    assert 'parsed_df[context_cols].notna().any(axis=1).sum()' in text
