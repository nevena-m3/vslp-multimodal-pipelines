from pathlib import Path

APP = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
PLOTS = Path('src/vslp/analysis/features/plots.py').read_text(encoding='utf-8')


def test_overview_plot_menu_is_dataset_structure_focused():
    assert '"Design context"' in APP
    assert '"Role mapping summary"' in APP
    assert '"Feature-family coverage"' in APP
    assert '"Feature-quality landscape"' in APP
    assert '"Subject x task coverage"' in APP
    assert '"Readiness scorecard"' not in APP


def test_overview_has_task_focus_for_family_and_quality():
    assert 'self.overview_task_combo = QComboBox()' in APP
    assert 'self.overview_task_combo.addItem("All tasks")' in APP
    assert 'def _generate_task_scoped_overview_plot(self, label: str) -> str:' in APP
    assert 'self._overview_feature_family_table(scoped, feature_cols)' in APP or 'self._overview_feature_family_table(active_df, feature_cols)' in APP
    assert 'self._overview_quality_table(scoped, feature_cols)' in APP or 'self._overview_quality_table(active_df, feature_cols)' in APP


def test_new_overview_structure_and_role_plots_exist():
    assert 'def plot_overview_dataset_structure(' in PLOTS
    assert 'Repeated-analysis subjects' in PLOTS
    assert 'def plot_role_mapping_summary(' in PLOTS
    assert 'Accepted role mapping summary' in PLOTS


def test_subject_task_coverage_is_not_heatmap():
    assert 'def plot_subject_task_coverage_bars(' in PLOTS
    subject_task = PLOTS[PLOTS.index('def plot_subject_task_coverage_bars('):PLOTS.index('def plot_overview_readiness_scorecard')]
    assert 'imshow' not in subject_task
    assert 'records' in subject_task and 'subjects' in subject_task and 'repeated_subjects' in subject_task
    assert 'return plot_subject_task_coverage_bars(df, path)' in PLOTS


def test_overview_tables_match_new_design():
    assert 'self.overview_subject_task_table = QTableWidget(0, 0)' in APP
    assert 'tabs.addTab(self.overview_subject_task_table, "Subject x task")' in APP
    assert 'self._fill_table(self.overview_subject_task_table, outputs.get("overview_subject_task_coverage", pd.DataFrame()))' in APP
