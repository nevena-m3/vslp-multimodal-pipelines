from pathlib import Path


def test_feature_gui_v070_safe_local_context_no_global_or_analysis_mutation():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.84.0"' in text
    assert '_build_global_context_bar' not in text
    assert 'Apply context + regenerate' not in text
    assert 'derive_clinical_context' not in text
    assert 'clinical_context.py' not in text
    assert 'warnings.catch_warnings' not in text
    assert 'import warnings' not in text


def test_feature_gui_v070_task_review_local_context_controls():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'def _safe_context_candidates' in text
    assert 'def _clinical_context_series' in text
    assert 'def _refresh_clinical_context_controls' in text
    assert 'self.clinical_context_combo' in text
    assert 'self.clinical_value_combo' in text
    assert 'Task x clinical context' in text
    assert 'Task counts within clinical value' in text
    assert 'Context detection' in text
    assert 'Severity presets' in text


def test_feature_gui_v070_bulbar_bins_are_local_and_validity_aware():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'def _bulbar_severity_local' in text
    assert 'invalid_above_12' in text
    assert 'near_normal_11_12' in text
    assert 'mild_9_10' in text
    assert 'moderate_6_8' in text
    assert 'severe_0_5' in text


def test_feature_gui_v070_task_clinical_plot_functions_available():
    text = Path("src/vslp/analysis/features/plots.py").read_text(encoding="utf-8")
    assert 'def plot_task_clinical_context' in text
    assert 'def plot_task_clinical_filtered_counts' in text
    assert 'clinical_series is computed locally by the GUI' in text
