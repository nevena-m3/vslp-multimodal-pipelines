from pathlib import Path


def test_feature_gui_v069_task_longitudinal_plots_are_real_plot_previews():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.71.0"' in text
    assert 'plot_task_counts' in text
    assert 'plot_task_subject_matrix' in text
    assert 'plot_task_label_context' in text
    assert 'plot_task_feature_support' in text
    assert 'plot_longitudinal_subject_records' in text
    assert 'plot_longitudinal_session_matrix' in text
    assert 'plot_longitudinal_iteration_counts' in text
    assert 'plot_longitudinal_date_timeline' in text
    assert 'def generate_task_review_plots' in text
    assert 'def preview_task_review_plot' in text
    assert 'def open_current_task_review_plot' in text
    assert 'def generate_longitudinal_plots' in text
    assert 'def preview_longitudinal_plot' in text
    assert 'def open_current_longitudinal_plot' in text


def test_feature_gui_v069_task_and_longitudinal_have_open_current_plot_buttons():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    task_block = text[text.index('def _task_review_page'):text.index('def _longitudinal_page')]
    long_block = text[text.index('def _longitudinal_page'):text.index('def _active_analysis_table')]
    assert 'Open current plot' in task_block
    assert 'Open current plot' in long_block
    assert 'self.task_review_preview.setAlignment(Qt.AlignCenter)' in task_block
    assert 'self.longitudinal_preview.setAlignment(Qt.AlignCenter)' in long_block


def test_feature_availability_heatmap_prefers_full_dataset_when_feasible():
    text = Path("src/vslp/analysis/features/plots.py").read_text(encoding="utf-8")
    assert 'max_features: int | None = None' in text
    assert 'max_rows: int | None = None' in text
    assert 'Prefer full-dataset display when feasible' in text
