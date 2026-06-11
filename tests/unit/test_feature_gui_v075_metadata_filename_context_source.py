from pathlib import Path


def test_feature_gui_v075_project_has_filename_context_controls():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.85.0"' in text
    assert 'Filename context inference' in text
    assert 'self.filename_parser_combo' in text
    assert 'Auto fallback: metadata first, then filename' in text
    assert 'Filename only when metadata is absent' in text
    assert 'Off' in text


def test_feature_gui_v075_metadata_join_uses_best_actual_match_not_first_existing_key():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'Select the best actual metadata join' in text
    assert 'metadata_key_join:' in text
    assert 'matched_' in text
    assert 'dedup_file_metadata' in text
    assert 'n_matches <= 0' in text
    assert '_match_file_stem' in text
    assert '_match_file_basename' in text
    assert 'drop_duplicates(subset=left_keys, keep="first")' in text


def test_feature_gui_v075_filename_parser_fills_task_subject_iteration_date_safely():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'def _parse_filename_context_frame' in text
    assert 'SUBJECT_PROTOCOL_ITERATION_YYYYMMDD_RECORD_TASK' in text
    assert 'parsed_subject_id' in text
    assert 'parsed_iteration' in text
    assert 'parsed_recording_date' in text
    assert 'parsed_task' in text
    assert 'fill_col("task", "parsed_task")' in text
    assert 'Last textual run' in text
    assert 'Metadata values always take priority' in text or 'metadata values always take priority' in text.lower()


def test_feature_gui_v075_outputs_filename_context_parse_table():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert '"filename_context_parse"' in text
    assert 'filename_parser_mode' in text
    assert 'Filename context parser mode:' in text
    assert 'filename_context_fallback_' in text
