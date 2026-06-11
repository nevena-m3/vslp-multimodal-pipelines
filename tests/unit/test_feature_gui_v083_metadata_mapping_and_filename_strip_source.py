from pathlib import Path


def test_feature_gui_v083_filename_tokens_strip_extensions():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.84.0"' in text
    assert 'Strip one or more trailing file extensions' in text
    assert 'PUFF.wav' not in text[text.index('def _filename_tokens_from_stem'):text.index('def _filename_template_mapping')]
    assert 'while re.search(r"\\.[A-Za-z0-9]{1,8}$", raw)' in text


def test_feature_gui_v083_metadata_mapping_menu_is_registered():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert '"metadata_mapping"' in text
    assert 'Metadata Mapping' in text
    assert 'self._metadata_mapping_page()' in text
    assert 'def _metadata_mapping_page' in text
    assert 'accepted_metadata_mapping.csv' in text


def test_feature_gui_v083_metadata_mapping_roles_cover_clinical_demographic_fields():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    for label in [
        'Diagnosis', 'Sex / gender', 'ALSFRS total', 'ALSFRS bulbar',
        'ALSBDI', 'Severity bin', 'Subject ID', 'Task name', 'Recording date'
    ]:
        assert label in text
    assert 'def _apply_metadata_mapping_to_table' in text
    assert 'out = self._apply_metadata_mapping_to_table(meta_df.copy())' not in text
    assert 'out = meta_df.copy()' in text


def test_feature_gui_v083_processing_boundaries_preserved():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert '_build_global_context_bar' not in text
    assert 'warnings.catch_warnings' not in text
    assert 'derive_clinical_context' not in text
    assert 'def _merge_metadata_context' in text
