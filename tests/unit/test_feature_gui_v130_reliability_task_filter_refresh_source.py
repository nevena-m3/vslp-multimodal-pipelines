from pathlib import Path

APP = Path('src/vslp/gui/features/app.py')


def test_v130_version_and_task_filter_handlers_present():
    text = APP.read_text(encoding='utf-8')
    assert 'APP_VERSION = "v0.130.0"' in text
    assert 'self.reliability_task_combo.currentIndexChanged.connect(self._on_reliability_task_changed)' in text
    assert 'def _on_reliability_task_changed' in text
    assert 'self._refresh_reliability_scope()' in text


def test_reliability_task_uses_item_data_not_display_label_only():
    text = APP.read_text(encoding='utf-8')
    section = text[text.index('def _selected_reliability_task'):text.index('def _selected_reliability_family')]
    assert 'currentData()' in section
    assert 'return str(data).strip()' in section
    assert 'return str(text).strip()' in section


def test_task_combo_populates_exact_task_data_and_preserves_selection():
    text = APP.read_text(encoding='utf-8')
    assert 'self.reliability_task_combo.addItem("All tasks", "All tasks")' in text
    assert 'self.reliability_task_combo.addItem(str(task_value), str(task_value))' in text
    assert 'findData(current_task)' in text


def test_scoped_design_reports_applied_task_filter_and_filtered_rows():
    text = APP.read_text(encoding='utf-8')
    assert '"task_filter_applied"' in text
    assert 'Rows/recordings included after the current task/source filter.' in text
    assert 'current task/source scope' in text


def test_control_changes_preview_current_plot_without_forcing_selected_feature_plot():
    text = APP.read_text(encoding='utf-8')
    assert 'def _preview_current_reliability_plot' in text
    assert 'def _on_reliability_family_changed' in text
    assert 'def _on_reliability_feature_changed' in text
    assert 'leave the currently selected summary plot visible' in text
