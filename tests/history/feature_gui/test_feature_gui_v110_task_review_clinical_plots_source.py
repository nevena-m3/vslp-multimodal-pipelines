from pathlib import Path

APP = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
PLOTS = Path('src/vslp/analysis/features/plots.py').read_text(encoding='utf-8')


def test_version_and_task_plot_labels_are_clinician_facing():
    assert 'APP_VERSION = "v0.110.0"' in APP
    assert 'Task readiness overview' in APP
    assert 'Task clinical balance' in APP
    assert 'Subject-task coverage' in APP
    assert 'Task feature completeness' in APP
    assert 'Task-specific feature profile' in APP
    assert 'Legacy coverage matrix' in APP


def test_task_review_uses_mapped_numeric_feature_columns():
    assert 'def _task_review_feature_cols' in APP
    assert 'str.contains("Feature", case=False' in APP
    assert 'pd.api.types.is_numeric_dtype' in APP
    assert 'return clean[:200]' in APP


def test_task_review_builds_readiness_table_and_snapshot():
    assert 'def _task_readiness_table' in APP
    for col in ['n_rows', 'n_subjects', 'mean_feature_missingness', 'smallest_clinical_group_n', 'readiness_status']:
        assert col in APP
    assert 'well_supported' in APP
    assert 'usable_with_caution' in APP
    assert 'review_before_modeling' in APP


def test_new_task_plots_exist_and_are_generated():
    for fn in [
        'plot_task_readiness_dashboard',
        'plot_task_clinical_balance_bars',
        'plot_task_subject_coverage_summary',
        'plot_task_feature_profile',
    ]:
        assert f'def {fn}' in PLOTS
        assert fn in APP
    for key in [
        'task_readiness_overview',
        'task_clinical_balance',
        'task_subject_coverage',
        'task_feature_completeness',
        'task_feature_profile',
    ]:
        assert key in APP


def test_task_plots_have_interpretation_guardrails():
    assert 'not a clinical outcome test' in PLOTS
    assert 'not performance' in PLOTS
    assert 'Readiness is descriptive' in PLOTS
    assert 'task comparisons may be confounded' in APP or 'clinically balanced' in APP
