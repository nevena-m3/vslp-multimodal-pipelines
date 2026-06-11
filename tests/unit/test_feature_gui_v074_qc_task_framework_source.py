from pathlib import Path


def test_feature_gui_v074_qc_has_local_task_and_framework_controls():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.76.0"' in text
    assert 'self.qc_task_combo' in text
    assert 'self.qc_framework_combo' in text
    assert 'Auto / all QC' in text
    assert 'Acoustic QC' in text
    assert 'Kinematic QC' in text
    assert 'def _qc_scope_tables' in text
    assert 'def generate_qc_scope_plots' in text
    assert 'def regenerate_qc_scope_plots' in text
    assert 'regen.clicked.connect(self.regenerate_qc_scope_plots)' in text


def test_feature_gui_v074_qc_scope_is_local_and_safe():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert '_build_global_context_bar' not in text
    assert 'Apply context + regenerate' not in text
    assert 'derive_clinical_context' not in text
    assert 'warnings.catch_warnings' not in text
    assert 'base analysis outputs are not modified' in text.lower() or 'Base analysis outputs are not modified' in text
    assert 'plot_qc_artifact_model' in text
    assert 'self._display_plot_image(self.qc_plot_preview, path)' in text


def test_feature_gui_v074_qc_framework_plot_function_exists():
    plots = Path("src/vslp/analysis/features/plots.py").read_text(encoding="utf-8")
    assert 'def plot_qc_framework_selection' in plots
    assert 'QC framework selection' in plots
    assert 'Acoustic/Kinematic modes use conservative column-name heuristics' in plots
