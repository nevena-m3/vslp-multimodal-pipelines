from pathlib import Path


def test_feature_gui_v070_global_context_bar_exists():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.72.0"' in text
    assert 'def _build_global_context_bar' in text
    assert 'self.context_bar = self._build_global_context_bar()' in text
    assert 'Analysis context' in text
    assert 'Task:' in text
    assert 'Group:' in text
    assert 'Value:' in text
    assert 'Apply context + regenerate' in text
    assert 'Clear context' in text


def test_feature_gui_v070_context_filters_drive_all_outputs():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'def _context_filtered_df' in text
    assert 'def _context_label' in text
    assert 'active_df = self._context_filtered_df()' in text
    assert 'analysis_context' in text
    assert 'Plots/tables regenerated for context' in text
    assert 'Context rejected because it has no rows' in text


def test_feature_gui_v070_context_options_are_metadata_aware():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'def _refresh_global_context_options' in text
    assert 'def _refresh_global_group_values' in text
    assert '"diagnosis"' in text
    assert '"severity_score"' in text
    assert '"session_id"' in text
    assert '"visit_id"' in text
    assert '"iteration"' in text
