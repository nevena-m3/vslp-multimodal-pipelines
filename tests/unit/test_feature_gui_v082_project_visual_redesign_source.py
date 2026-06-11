from pathlib import Path


def test_feature_gui_v082_project_visual_redesign_source():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.85.0"' in text
    assert 'Project setup' in text
    assert 'def _project_step_label' in text
    assert 'def _project_section_title' in text
    assert 'Filename context' in text
    assert 'self.filename_source_combo' in text
    assert 'QLabel#ExampleBox' in text
    assert 'Task start' in text
    assert 'Task end' in text


def test_feature_gui_v082_project_layout_is_less_text_heavy():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    project = text[text.index('def _project_page'):text.index('def _mapping_page')]
    assert 'policy_note' not in project
    assert 'Filename context inference' not in project
    assert 'Optional context extraction from a filename-like column' not in project
    assert 'Source tables' in project
    assert 'Analysis settings' in project
    assert 'Run messages will appear here.' in project


def test_feature_gui_v082_does_not_change_processing_logic():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'def _filename_source_candidates' in text
    assert 'def refresh_filename_source_combo' in text
    assert 'def _parse_filename_context_by_template' in text
    assert 'def _merge_metadata_context' in text
    assert '_build_global_context_bar' not in text
    assert 'warnings.catch_warnings' not in text
    assert 'derive_clinical_context' not in text
