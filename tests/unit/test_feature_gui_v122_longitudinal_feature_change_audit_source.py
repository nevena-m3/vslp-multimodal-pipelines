from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = (ROOT / 'src' / 'vslp' / 'gui' / 'features' / 'app.py').read_text(encoding='utf-8')
PLOTS = (ROOT / 'src' / 'vslp' / 'analysis' / 'features' / 'plots.py').read_text(encoding='utf-8')


def test_version_updated_to_v122():
    assert 'APP_VERSION = "v0.122.0"' in APP


def test_feature_change_plot_label_is_audit_not_family_composite():
    assert 'Selected subject-task feature change audit' in APP
    assert 'Selected subject-task feature-family change' not in APP
    assert 'Feature change audit' in APP


def test_same_task_filter_is_preserved_for_feature_change():
    block = APP[APP.index('def _longitudinal_feature_family_change_table'):APP.index('def update_longitudinal_dashboard')]
    assert 'selected_task = self._longitudinal_selected_task()' in block
    assert 'task_df = df.loc[df[task_col].astype(str).eq(str(selected_task))].copy()' in block
    assert 'g = task_df.loc[task_df[subj_col].astype(str).eq(str(selected_subject))].copy()' in block


def test_feature_change_table_keeps_features_separate_and_direction_uncertain():
    block = APP[APP.index('def _longitudinal_feature_family_change_table'):APP.index('def update_longitudinal_dashboard')]
    assert 'for feat in usable:' in block
    assert 'standardized_change_from_baseline' in block
    assert 'clinical_direction_from_registry' in block
    assert 'numeric direction only; clinical meaning not inferred' in block
    assert 'family_mean_standardized_change_from_baseline' not in block


def test_show_regenerates_current_longitudinal_plot_without_manual_regenerate_first():
    block = APP[APP.index('def preview_longitudinal_plot'):APP.index('def open_current_longitudinal_plot')]
    assert 'self.generate_longitudinal_plots()' in block
    assert 'Always regenerate the selected longitudinal plot' in block


def test_plot_function_shows_top_individual_features_not_composite_score():
    assert 'def plot_longitudinal_feature_family_trajectory' in PLOTS
    block = PLOTS[PLOTS.index('def plot_longitudinal_feature_family_trajectory'):PLOTS.index('def plot_longitudinal_session_matrix')]
    assert 'Selected subject-task feature change audit' in block
    assert 'top_features' in block
    assert 'for feat in top_features' in block
    assert 'family_mean_standardized_change_from_baseline' not in block
    assert 'Direction: positive/negative is numeric change only' in block
