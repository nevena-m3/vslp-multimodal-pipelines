from pathlib import Path

APP = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")


def test_project_menu_does_not_own_filename_fallback_controls():
    start = APP.index("    def _project_page")
    end = APP.index("    def _mapping_page", start)
    block = APP[start:end]
    assert "filename_source_combo" not in block
    assert "filename_token_combos" not in block
    assert "Apply filename context" not in block
    assert "Filename context" not in block
    assert "Primary feature table" in block
    assert "QC table" in block
    assert "Metadata table" in block
    assert "Output folder" in block


def test_metadata_mapping_has_clean_no_metadata_filename_mode():
    assert "self.metadata_toolbar_widget = QWidget()" in APP
    assert "self.filename_metadata_fallback_frame" in APP
    assert "Filename-derived metadata fallback" in APP
    assert "self.filename_fallback_status_label" in APP
    assert "self.metadata_toolbar_widget.setVisible(has_metadata)" in APP
    assert "self.metadata_mapping_table.setVisible(has_metadata)" in APP
    assert "self.metadata_quick_role_widget.setVisible(has_metadata)" in APP
    assert "self.metadata_mapping_summary_label.setVisible(has_metadata)" in APP


def test_apply_filename_context_bypasses_metadata_merge_and_writes_analysis_df():
    start = APP.index("    def apply_filename_context_from_metadata_mapping")
    end = APP.index("    def refresh_metadata_mapping_table", start)
    block = APP[start:end]
    assert "self._merge_metadata_context" not in block
    assert "self._parse_filename_context_frame(base_df)" in block
    for col in ["subject_id", "protocol_id", "iteration", "duration", "recording_date", "task_code", "task"]:
        assert f'"{col}"' in block
    assert "self.analysis_df = out" in block
    assert "Status counts" in block
    assert "Non-empty values" in block
    assert "selected_col" in block


def test_filename_parser_strips_extensions_and_supports_task_span():
    assert "while re.search" in APP
    assert "raw = re.sub" in APP
    assert "task_start_idx" in APP
    assert "task_end_idx" in APP
    assert '"_".join(tokens[lo:hi + 1])' in APP
    assert "parsed_by_user_template" in APP
