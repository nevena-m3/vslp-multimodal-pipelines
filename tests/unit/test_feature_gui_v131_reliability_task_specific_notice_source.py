from pathlib import Path

APP = Path("src/vslp/gui/features/app.py")
PLOTS = Path("src/vslp/analysis/features/plots.py")


def test_v131_version_and_notice_plot_imported():
    app = APP.read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.131.0"' in app
    assert 'plot_reliability_scope_notice' in app


def test_task_specific_reliability_plots_generate_diagnostic_notices():
    app = APP.read_text(encoding="utf-8")
    section = app[app.index('def _generate_reliability_plot'):app.index('def preview_reliability_plot')]
    assert 'evaluable_icc <= 0' in section
    assert 'has_mdc = False' in section
    assert 'has_family =' in section
    assert 'plot_reliability_scope_notice' in section


def test_selected_feature_plot_has_fallback_notice_instead_of_none():
    app = APP.read_text(encoding="utf-8")
    section = app[app.index('elif key in {"selected_feature_same_task_reliability"'):app.index('else:', app.index('elif key in {"selected_feature_same_task_reliability"'))]
    assert 'if not feature and rep is not None' in section
    assert 'plot_reliability_scope_notice' in section


def test_notice_plot_explains_sparse_task_scope():
    plots = PLOTS.read_text(encoding="utf-8")
    assert 'def plot_reliability_scope_notice' in plots
    assert 'Sparse tasks should be reported as design limitations' in plots
    assert 'Repeated units' in plots


def test_reliability_message_mentions_task_specific_diagnostics():
    app = APP.read_text(encoding="utf-8")
    assert 'task-specific diagnostic notices' in app
