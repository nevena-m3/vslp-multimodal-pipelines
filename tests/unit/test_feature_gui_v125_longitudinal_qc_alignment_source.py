from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "vslp" / "gui" / "features" / "app.py"
PLOTS = ROOT / "src" / "vslp" / "analysis" / "features" / "plots.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_v125_version_and_qc_family_selector_added():
    app = _read(APP)
    assert 'APP_VERSION = "v0.125.0"' in app
    assert "self.longitudinal_qc_family_combo" in app
    assert "QC family:" in app
    assert "All QC families" in app
    assert "_refresh_longitudinal_qc_family_combo" in app
    assert "_longitudinal_selected_qc_family" in app


def test_v125_automated_qc_aligns_to_analysis_rows_safely():
    app = _read(APP)
    block = app[app.index("def _align_automated_qc_to_analysis"):app.index("def _refresh_longitudinal_qc_family_combo")]
    assert "Prefer explicit shared record/file keys" in block
    assert "file_name" in block
    assert "record_key" in block
    assert "Path(s).name.lower().strip()" in block
    assert "fall back to row order" in block
    assert "q.index = df.index" in block


def test_v125_automated_qc_uses_all_metrics_in_selected_qc_family():
    app = _read(APP)
    block = app[app.index("# Automated QC: aligned uploaded QC table"):app.index("out = pd.DataFrame(rows)", app.index("# Automated QC: aligned uploaded QC table"))]
    assert "_align_automated_qc_to_analysis" in block
    assert "_longitudinal_qc_numeric_columns" in block
    assert "qc_family_from_name(c)" in block
    assert "selected_qc_family" in block
    assert "numeric_metrics.append" in block
    assert "standardized_change_from_baseline" in block
    assert "same plotting semantics as acoustic features" in block


def test_v125_manual_qc_remains_yes_no_and_family_filtered():
    app = _read(APP)
    block = app[app.index("# Manual QC: accepted metadata mapping roles only"):app.index("# Automated QC: aligned uploaded QC table")]
    assert "selected_qc_family" in block
    assert "manual_qc_status" in block
    assert '"Yes" if float(val) > 0 else "No"' in block
    assert "manual QC flag present" in block


def test_v125_plots_remain_line_plots_without_heatmaps_for_longitudinal_qc():
    plots = _read(PLOTS)
    block = plots[plots.index("def _plot_longitudinal_qc_lines"):plots.index("def plot_longitudinal_qc_change_audit")]
    assert "ax.plot(" in block
    assert "drawstyle=\"steps-post\"" in block
    assert "ax.imshow" not in block
    assert "Standardized automated QC change from own baseline" in block
