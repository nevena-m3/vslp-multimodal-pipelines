from pathlib import Path


def test_feature_gui_v076_restores_qc_framework_model_plot():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    plots = Path("src/vslp/analysis/features/plots.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.76.0"' in text
    assert 'plot_qc_framework_selection' not in text
    assert 'plot_qc_artifact_model(plots_dir / f"qc_artifact_model_{safe_scope}.png", self._qc_framework_mode())' in text
    assert 'def plot_qc_artifact_model(path: Path, framework: str = "Auto / all QC")' in plots
    assert 'Additive\\ninterference' in plots
    assert 'Gain / level\\ndynamics' in plots
    assert 'Reverberation\\n/ echo' in plots
    assert 'Channel / device\\n/ platform' in plots
    assert 'Nonlinear\\ndistortion' in plots
    assert 'Temporal\\ndiscontinuities' in plots
    assert 'Current framework: Acoustic QC' in plots
    assert 'Current framework: Kinematic QC' in plots


def test_feature_gui_v076_qc_feature_metric_selectors_are_active():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'def _refresh_qc_selector_combos' in text
    assert 'self.qc_feature_combo.currentIndexChanged.connect' in text
    assert 'self.qc_metric_combo.currentIndexChanged.connect' in text
    assert 'self._refresh_qc_selector_combos()' in text
    assert 'QMessageBox.warning(self, "Selected QC scatter failed", str(exc))' in text


def test_feature_gui_v076_keeps_safe_qc_scope_boundary():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert '_build_global_context_bar' not in text
    assert 'Apply context + regenerate' not in text
    assert 'warnings.catch_warnings' not in text
    assert 'derive_clinical_context' not in text
    assert 'def _qc_scope_tables' in text
    assert 'def generate_qc_scope_plots' in text
