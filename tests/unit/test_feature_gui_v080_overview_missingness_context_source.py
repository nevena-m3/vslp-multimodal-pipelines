from pathlib import Path


def test_feature_gui_v080_overview_uses_robust_context_detection():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.84.0"' in text
    assert 'def _context_aliases' in text
    assert 'def _first_context_column' in text
    assert 'def _overview_context_detection_table' in text
    assert 'robust_design = self._overview_context_detection_table(active_df)' in text
    assert 'self._fill_table(self.overview_design_table, robust_design)' in text
    assert '"Diagnosis"' in text
    assert '"Severity score"' in text
    assert '"Task code"' in text


def test_feature_gui_v080_metadata_join_uses_filename_context_before_matching():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'feature_base, pre_parsed_context = self._fill_empty_context_from_filename(feature_base)' in text
    assert '["subject_id", "protocol_id", "iteration", "recording_date", "task_code"]' in text
    assert 'parsed_by_user_template' in text
    assert 'filename_context_fallback' in text


def test_feature_gui_v080_missingness_has_task_and_context_scope():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'self.missing_task_combo' in text
    assert 'self.missing_context_combo' in text
    assert 'self.missing_context_value_combo' in text
    assert 'def _missingness_context_series' in text
    assert 'def _refresh_missingness_context_controls' in text
    assert 'def _refresh_missingness_context_value_combo' in text
    assert 'context = all/unavailable' in text
    assert '__local_missingness_context__' in text


def test_feature_gui_v080_audit_aliases_are_broad():
    audit = Path("src/vslp/analysis/features/audit.py").read_text(encoding="utf-8")
    assert 'normalize_name(n)' in audit
    assert 'metadata__Diagnosis' in audit
    assert 'ALSFRS total score' in audit
    assert 'metadata__Task Name' in audit
    assert 'Clinical Visit ID' in audit
