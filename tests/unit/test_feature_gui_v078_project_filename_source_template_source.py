from pathlib import Path


def test_feature_gui_v078_project_filename_source_column_controls():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.85.0"' in text
    assert 'self.filename_source_combo' in text
    assert 'Context source:' in text
    assert 'Filename column:' in text
    assert 'Selected filename column + metadata fallback' in text
    assert 'Auto-detect filename column + metadata fallback' in text
    assert 'Metadata only (no filename parsing)' in text
    assert 'def _filename_source_candidates' in text
    assert 'def refresh_filename_source_combo' in text


def test_feature_gui_v078_project_output_before_filename_inference():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    project = text[text.index('def _project_page'):text.index('def _mapping_page')]
    assert project.index('Output folder:') < project.index('Filename context inference')


def test_feature_gui_v078_multitoken_task_and_duration_template():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert '("duration", "Duration")' in text
    assert '("task", "Task name start")' in text
    assert '("task_end", "Task name end")' in text
    assert 'parsed_duration' in text
    assert 'fill_col("duration", "parsed_duration")' in text
    assert 'task = "_".join(tokens[lo:hi + 1])' in text


def test_feature_gui_v078_metadata_remains_clinical_source():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'Metadata is still loaded and used for clinical variables' in text
    assert '_merge_metadata_context' in text
    assert 'diagnosis' in text.lower()
