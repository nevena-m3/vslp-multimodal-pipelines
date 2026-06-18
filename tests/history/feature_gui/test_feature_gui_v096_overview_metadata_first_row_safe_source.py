from pathlib import Path

APP = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
PLOTS = Path('src/vslp/analysis/features/plots.py').read_text(encoding='utf-8')


def test_overview_uses_metadata_first_safe_source_frame():
    assert 'def _overview_context_source_frame' in APP
    assert 'def _overview_series_for_role' in APP
    assert 'Metadata-loaded policy: metadata-derived canonical fields are preferred.' in APP
    assert 'metadata_loaded = getattr(self, "meta_df", None) is not None and not self.meta_df.empty' in APP
    assert 'preferred.extend([f"metadata__{canonical}", canonical])' in APP


def test_overview_no_longer_calls_distribution_audit_for_overview_only_outputs():
    block = APP[APP.index('def _build_overview_outputs'):APP.index('def regenerate_overview_plots')]
    assert 'feature_distribution_summary(active_df' not in block
    assert 'overview_feature_quality_landscape(dist' not in block
    assert 'self._overview_quality_table(active_df, feature_cols)' in block
    assert 'self._overview_feature_family_table(active_df, feature_cols)' in block


def test_overview_plots_are_preaggregated_not_raw_analysis_frame():
    block = APP[APP.index('def _generate_overview_plots'):APP.index('def preview_selected_overview_plot')]
    assert 'plot_overview_design_from_tables' in block
    assert 'plot_subject_task_summary_bars' in block
    assert 'plot_overview_dataset_structure(active_df' not in block
    assert 'plot_subject_task_matrix(active_df' not in block


def test_overview_dashboard_tables_use_outputs_not_active_context_recalculation():
    block = APP[APP.index('def update_overview_dashboard'):APP.index('def _refresh_overview_task_focus_from_outputs')]
    assert 'outputs.get("dataset_design_overview"' in block
    assert 'outputs.get("overview_subject_task_coverage"' in block
    assert '_overview_context_detection_table(active_for_context)' not in block
    assert '_overview_subject_task_summary(active_for_context)' not in block


def test_new_plot_functions_exist():
    assert 'def plot_overview_design_from_tables' in PLOTS
    assert 'def plot_subject_task_summary_bars' in PLOTS
    assert 'pre-aggregated' in PLOTS
