from pathlib import Path


def test_feature_gui_v072_distributions_has_local_task_and_clinical_scope():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.77.0"' in text
    assert 'self.dist_task_combo' in text
    assert 'self.dist_context_combo' in text
    assert 'self.dist_context_value_combo' in text
    assert 'def _refresh_dist_task_combo' in text
    assert 'def _dist_context_series' in text
    assert 'def _distribution_scope_table' in text
    assert 'def generate_distribution_scope_plots' in text
    assert 'def regenerate_distribution_scope_plots' in text
    assert 'regen.clicked.connect(self.regenerate_distribution_scope_plots)' in text


def test_feature_gui_v072_distributions_scope_is_local_not_global():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert '_build_global_context_bar' not in text
    assert 'Apply context + regenerate' not in text
    assert 'derive_clinical_context' not in text
    assert 'warnings.catch_warnings' not in text
    assert 'active_df, scope_label = self._distribution_scope_table()' in text
    assert 'self._display_plot_image(self.dist_plot_preview, path)' in text


def test_feature_gui_v072_distributions_has_bulbar_severity_option():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'ALSFRS bulbar severity' in text
    assert '_bulbar_severity_local' in text
    assert '__local_clinical_context__' in text
    assert 'Clinical context' in text
