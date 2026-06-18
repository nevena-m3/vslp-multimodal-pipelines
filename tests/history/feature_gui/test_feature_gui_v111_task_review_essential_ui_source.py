from pathlib import Path

APP = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
PLOTS = Path('src/vslp/analysis/features/plots.py').read_text(encoding='utf-8')


def test_version_updated_to_v111():
    assert 'APP_VERSION = "v0.111.0"' in APP


def test_task_review_exposes_only_essential_plot_names():
    assert 'Task readiness summary' in APP
    assert 'Clinical balance by task' in APP
    assert 'Subject coverage by task' in APP
    assert 'Feature completeness by task' in APP
    assert 'Task-specific feature profile", "task_feature_profile"' not in APP
    assert 'Legacy coverage matrix' not in APP


def test_task_review_deprecates_noisy_legacy_paths():
    assert 'The older task feature profile and subject-by-task heatmap are intentionally not exposed in v0.111' in APP
    assert 'self.plot_paths["task_feature_profile"] = self.plot_paths.get("task_feature_completeness", "")' in APP
    assert 'self.plot_paths["task_subject_matrix"] = self.plot_paths.get("task_subject_coverage", "")' in APP


def test_task_review_pixmap_preview_is_size_stable():
    assert 'setScaledContents(False)' in APP
    assert 'setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)' in APP
    assert 'contentsRect().size()' in APP
    assert 'self.task_review_preview.clear()' in APP


def test_task_review_copy_is_essential_not_plot_maximizing():
    assert 'intentionally keeps only the essential plots' in APP
    assert 'four practical questions' in APP
