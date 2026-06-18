from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "vslp" / "gui" / "features" / "app.py"


def text():
    return APP.read_text(encoding="utf-8")


def test_safe_dashboard_update_helper_exists_and_logs_traceback():
    s = text()
    assert "def _safe_dashboard_update" in s
    assert "def _safe_refresh_analysis_dashboards" in s
    assert "traceback.format_exc()" in s
    assert "dashboard update skipped" in s


def test_run_analysis_uses_isolated_dashboard_refresh():
    s = text()
    run_block = s[s.index("def run_analysis"):s.index("def populate_output_tables")]
    assert "self._safe_refresh_analysis_dashboards(outputs)" in run_block
    assert "self.update_missingness_dashboard(outputs)" not in run_block
    assert "self.update_distribution_dashboard(outputs)" not in run_block
    assert "self.update_task_review_dashboard(outputs)" not in run_block


def test_regenerate_overview_uses_isolated_dashboard_refresh_and_error_logging():
    s = text()
    block = s[s.index("def regenerate_overview_plots"):s.index("def preview_plot")]
    assert "self._safe_refresh_analysis_dashboards(outputs)" in block
    assert "self.update_qc_dashboard(outputs)" not in block
    assert "self.log_error(\"Could not generate overview plots\", exc)" in block


def test_overview_is_refreshed_before_downstream_menus():
    s = text()
    block = s[s.index("dashboard_updates = ["):s.index("failed = []")]
    assert block.index('("Overview", self.update_overview_dashboard)') < block.index('("Missingness", self.update_missingness_dashboard)')
    assert block.index('("Overview", self.update_overview_dashboard)') < block.index('("QC Integration", self.update_qc_dashboard)')
