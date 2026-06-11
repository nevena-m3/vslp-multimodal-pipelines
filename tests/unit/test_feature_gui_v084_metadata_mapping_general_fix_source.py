from pathlib import Path


def test_feature_gui_v084_filename_fill_is_dtype_safe():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.85.0"' in text
    assert 'Use object dtype during partial fills' in text
    assert 'safe_existing = out[canonical].astype("object")' in text
    assert 'values = parsed[parsed_col].reindex(out.index)' in text


def test_feature_gui_v084_metadata_mapping_no_recursion():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    block = text[text.index('def _apply_metadata_mapping_to_table'):text.index('def _standardize_metadata_table')]
    assert 'out = meta_df.copy()' in block
    assert 'self._apply_metadata_mapping_to_table(meta_df.copy())' not in block


def test_feature_gui_v084_metadata_roles_are_general_and_meaningful():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    for label in [
        'Primary target', 'Secondary target', 'Target 1', 'Target 2', 'Target 3',
        'Outcome 1', 'Outcome 2', 'Severity score', 'Clinical score 1',
        'Functional score', 'Bulbar score', 'Demographic covariate',
        'Clinical covariate', 'Disease group', 'Group label'
    ]:
        assert label in text
    assert 'ALSFRS bulbar score' in text[text.index('def _metadata_mapping_roles'):text.index('def _metadata_role_to_canonical')]


def test_feature_gui_v084_preserves_project_and_metadata_pages():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'def _metadata_mapping_page' in text
    assert '"metadata_mapping"' in text
    assert 'Filename column' in text
    assert '_build_global_context_bar' not in text
    assert 'warnings.catch_warnings' not in text
    assert 'derive_clinical_context' not in text
