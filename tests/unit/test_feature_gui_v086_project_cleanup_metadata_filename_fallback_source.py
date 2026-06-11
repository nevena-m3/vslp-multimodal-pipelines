from pathlib import Path


def test_feature_gui_v086_project_no_filename_context_section():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.86.0"' in text
    project = text[text.index('def _project_page'):text.index('def _mapping_page')]
    assert 'Filename context' not in project
    assert 'filename_source_combo' not in project
    assert 'filename_token_combos' not in project
    assert 'Source tables' in project
    assert 'Analysis settings' in project


def test_feature_gui_v086_filepicker_labels_are_borderless():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    block = text[text.index('class FilePicker'):text.index('class NoWheelComboBox')]
    assert 'background:transparent; border:none; padding:0px' in block


def test_feature_gui_v086_filename_context_lives_in_metadata_mapping_when_no_metadata():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    md = text[text.index('def _metadata_mapping_page'):text.index('def refresh_metadata_mapping_table')]
    assert 'Filename-derived metadata fallback' in md
    assert 'self.filename_source_combo' in md
    assert 'self.filename_token_combos' in md
    assert 'Apply filename context' in md
    assert 'def update_metadata_mapping_mode_visibility' in text
    assert 'def apply_filename_context_from_metadata_mapping' in text
    assert 'setVisible(not has_metadata)' in text
    assert 'self.metadata_quick_role_widget.setVisible(has_metadata)' in text


def test_feature_gui_v086_metadata_filters_are_analysis_groups():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    for label in [
        'Recording identity', 'Core clinical', 'Priority clinical scores',
        'Demographics', 'Disease history / covariates', 'Media / device',
        'Manual metadata QC', 'Administrative'
    ]:
        assert label in text


def test_feature_gui_v086_no_metadata_merge_applies_filename_context():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    block = text[text.index('def _merge_metadata_context'):text.index('feature_base = self._standardize_feature_match_helpers(feature_df)', text.index('def _merge_metadata_context')) + 2000]
    assert 'self.meta_df is None or self.meta_df.empty' in block
    assert 'feature_base, parsed_context = self._fill_empty_context_from_filename(feature_base)' in block
    assert 'feature_table_only:filename_context_fallback_' in block


def test_feature_gui_v086_safe_boundaries_preserved():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert '_build_global_context_bar' not in text
    assert 'warnings.catch_warnings' not in text
    assert 'derive_clinical_context' not in text
    assert 'def _merge_metadata_context' in text
