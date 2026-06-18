from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "vslp" / "gui" / "features" / "app.py"
PLOTS = ROOT / "src" / "vslp" / "analysis" / "features" / "plots.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_v124_version_and_plot_menu_are_line_qc_separated():
    app = _read(APP)
    assert 'APP_VERSION = "v0.124.0"' in app
    assert 'Selected subject-task feature change", "longitudinal_feature_family_trajectory"' in app
    assert 'Selected subject-task manual QC change", "longitudinal_manual_qc_change"' in app
    assert 'Selected subject-task automated QC change", "longitudinal_automated_qc_change"' in app
    assert 'Selected subject-task QC change audit", "longitudinal_qc_change_audit"' not in app


def test_v124_feature_change_plot_uses_all_feature_lines_not_heatmap():
    plots = _read(PLOTS)
    start = plots.index("def plot_longitudinal_feature_family_trajectory")
    end = plots.index("def _plot_longitudinal_qc_lines", start)
    block = plots[start:end]
    assert "Every selected-family feature is plotted as a separate line" in block
    assert "No family-average score is computed" in block
    assert "ax.plot(" in block
    assert "ax.imshow" not in block
    assert "no composite family score" in block.lower()


def test_v124_manual_and_automated_qc_are_separate_line_plots():
    plots = _read(PLOTS)
    assert "def plot_longitudinal_manual_qc_change" in plots
    assert "def plot_longitudinal_automated_qc_change" in plots
    assert 'source="Manual QC"' in plots
    assert 'source="Automated QC"' in plots
    assert 'ax.set_yticklabels(["No", "Yes"]' in plots
    assert 'drawstyle="steps-post"' in plots
    assert "Standardized automated QC change from own baseline" in plots


def test_v124_tables_are_split_for_manual_and_automated_qc():
    app = _read(APP)
    assert "self.long_manual_qc_change_table" in app
    assert "self.long_auto_qc_change_table" in app
    assert "Manual QC change" in app
    assert "Automated QC change" in app
    assert "_longitudinal_manual_qc_change_table" in app
    assert "_longitudinal_automated_qc_change_table" in app


def test_v124_bottom_text_overlap_margins_are_increased():
    plots = _read(PLOTS)
    assert "bottom=0.25" in plots
    assert "bbox_to_anchor=(1.01, 0.5)" in plots
    assert "labelpad=10" in plots
