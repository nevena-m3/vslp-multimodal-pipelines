from pathlib import Path


def test_feature_gui_v085_metadata_roles_are_grouped_and_priority_specific():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.' in text
    roles = text[text.index('def _metadata_mapping_roles'):text.index('def _metadata_role_to_canonical')]
    for label in [
        '-- File / recording identity --',
        '-- Core clinical grouping --',
        '-- Priority clinical scores --',
        '-- Manual metadata QC / validity flags --',
        'ALSFRS bulbar score',
        'ALSBDI total score',
        'Diagnosis',
        'Recording date',
        'Sex / gender',
        'Manual audio QC flag',
        'Manual video QC flag',
    ]:
        assert label in roles


def test_feature_gui_v085_metadata_mapping_table_is_fullscreen_with_examples():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'self.metadata_mapping_table.setMinimumHeight(620)' in text
    assert 'self.metadata_mapping_filter_combo' in text
    assert 'Examples' in text
    assert 'def _metadata_example_values' in text
    assert 'def _metadata_role_group' in text
    assert 'Manual metadata QC flags are yes/no acquisition observations' in text


def test_feature_gui_v085_specific_metadata_example_aliases():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    for alias in [
        'alsfrs_bulbar_subscore',
        'alsbdi_total_score',
        'diagnosis',
        'recording_date',
        'sex',
        'background_noise',
        'poor_audio_quality',
        'frozen_video',
        'task_completed_as_instructed',
        'needs_parsing',
    ]:
        assert alias in text
    assert '"alsfrs_bulbar": "ALSFRS bulbar score"' in text
    assert '"alsbdi": "ALSBDI total score"' in text


def test_feature_gui_v085_preserves_safe_boundaries():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert '_build_global_context_bar' not in text
    assert 'warnings.catch_warnings' not in text
    assert 'derive_clinical_context' not in text
    assert 'def _merge_metadata_context' in text
