from pathlib import Path


def test_feature_gui_v068_relationships_visual_parity():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.73.0"' in text
    block = text[text.index('def _relationships_page'):text.index('def update_relationships_dashboard')]
    assert 'Relationship snapshot' in block
    assert 'Detailed relationship tables' in block
    assert 'Open current plot' in block
    assert 'rel_split = QHBoxLayout()' in block
    assert 'side_panel.setFixedWidth(300)' in block
    assert 'self.relationship_plot_preview.setMinimumHeight(520)' in block
    assert 'Task focus:' in block
    assert 'does not duplicate Task Review, Outcome Screening, QC Integration, Missingness, or ML Export' in block


def test_feature_gui_v068_adds_task_and_longitudinal_pages():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert '"task_review"' in text
    assert '"longitudinal"' in text
    assert 'def _task_review_page' in text
    assert 'def _longitudinal_page' in text
    assert 'def update_task_review_dashboard' in text
    assert 'def update_longitudinal_dashboard' in text
    assert 'Task Review' in text
    assert 'Longitudinal / Iterations' in text


def test_feature_gui_v068_task_longitudinal_detection_helpers():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'def _task_col' in text
    assert 'def _subject_col' in text
    assert 'def _session_col' in text
    assert 'def _iteration_col' in text
    assert 'def _date_col' in text
    assert 'def _refresh_focus_combos' in text
    assert 'self.update_task_review_dashboard(outputs)' in text
    assert 'self.update_longitudinal_dashboard(outputs)' in text
