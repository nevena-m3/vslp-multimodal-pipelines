from pathlib import Path

APP = Path('src/vslp/gui/features/app.py')
PLOTS = Path('src/vslp/analysis/features/plots.py')


def read(path):
    return path.read_text(encoding='utf-8')


def test_version_and_distribution_menu_professional_plots():
    s = read(APP)
    assert 'APP_VERSION = "v0.108.0"' in (read(APP) if 'read' in globals() else APP.read_text(encoding='utf-8'))
    assert 'Shape and tail audit", "distribution_shape_story"' in s
    assert 'Range and outlier triage", "distribution_outlier_range_story"' in s
    assert 'Recording outlier burden", "row_outlier_story"' in s
    dist_section = s[s.index('    def _distributions_page'):s.index('    def _refresh_dist_task_combo')]
    assert 'Selected feature by group' not in dist_section
    dist_section = s[s.index('    def _distributions_page'):s.index('    def _refresh_dist_task_combo')]
    assert 'Clinical context:' not in dist_section
    assert 'Grouping for selected-feature plots comes from Clinical context above' not in dist_section


def test_distribution_scope_outputs_drive_tables_snapshot_and_plots():
    s = read(APP)
    assert 'def _distribution_scope_outputs' in s
    assert 'def _update_distribution_snapshot' in s
    assert 'def _update_distribution_tables' in s
    assert 'self._fill_table(self.dist_summary_table, outputs.get("feature_distribution_summary"' in s
    assert 'self._fill_table(self.dist_review_table, outputs.get("distribution_review_summary"' in s
    assert 'self._fill_table(self.dist_row_burden_table, outputs.get("row_outlier_burden_summary"' in s
    assert 'self._update_distribution_snapshot(outputs, scope_label)' in s
    assert 'self._update_distribution_tables(outputs)' in s


def test_distribution_plots_are_current_scope_and_backward_compatible():
    s = read(APP)
    assert 'self.plot_paths["distribution_shape_story"]' in s
    assert 'plot_distribution_shape_story(shape' in s
    assert 'self.plot_paths["distribution_outlier_range_story"]' in s
    assert 'plot_distribution_outlier_range_story(expected, outliers, review' in s
    assert 'self.plot_paths["row_outlier_story"]' in s
    assert 'plot_row_outlier_story(row_burden' in s
    assert 'self.plot_paths["distribution_shape_summary"] = self.plot_paths["distribution_shape_story"]' in s
    assert 'self.plot_paths["expected_range_flags"] = self.plot_paths["distribution_outlier_range_story"]' in s
    assert 'self.plot_paths["row_outlier_burden"] = self.plot_paths["row_outlier_story"]' in s


def test_new_plot_functions_exist():
    s = read(PLOTS)
    assert 'def plot_distribution_shape_story' in s
    assert 'def plot_distribution_outlier_range_story' in s
    assert 'def plot_row_outlier_story' in s
    assert 'Distribution shape and tail audit' in s
    assert 'Range and outlier triage' in s
    assert 'Recording outlier burden' in s
