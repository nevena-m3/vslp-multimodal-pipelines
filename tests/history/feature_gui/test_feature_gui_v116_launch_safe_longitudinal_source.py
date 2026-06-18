from pathlib import Path

APP = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")


def test_version_updated_to_v116_or_later():
    assert 'APP_VERSION = "v0.116.0"' in APP or 'APP_VERSION = "v0.117.0"' in APP


def test_task_review_runtime_methods_exist_for_launch():
    for name in [
        "def update_task_review_dashboard",
        "def preview_task_review_plot",
        "def open_current_task_review_plot",
    ]:
        assert name in APP


def test_task_review_page_connects_to_existing_methods():
    block = APP[APP.index("def _task_review_page"):APP.index("def _longitudinal_page")]
    assert "self.open_current_task_review_plot" in block
    assert "def open_current_task_review_plot" in APP


def test_longitudinal_page_runtime_methods_exist():
    for name in [
        "def _longitudinal_page",
        "def update_longitudinal_dashboard",
        "def preview_longitudinal_plot",
        "def open_current_longitudinal_plot",
    ]:
        assert name in APP


def test_no_v112_v113_v114_longitudinal_redesign_markers():
    # v0.116/v0.117 intentionally avoid the broken v112-v114 longitudinal redesign.
    assert "Feature + manual QC alignment" not in APP
    assert "Feature-family trajectory" not in APP
    assert "Manual QC trajectory" not in APP
