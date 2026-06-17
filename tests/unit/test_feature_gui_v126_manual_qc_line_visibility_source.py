from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "vslp" / "gui" / "features" / "app.py"
PLOTS = ROOT / "src" / "vslp" / "analysis" / "features" / "plots.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_v126_version_is_set():
    assert 'APP_VERSION = "v0.126.0"' in _read(APP)


def test_v126_manual_qc_uses_direct_lines_not_step_heatmap():
    plots = _read(PLOTS)
    block = plots[plots.index("def _plot_longitudinal_qc_lines"):plots.index("def plot_longitudinal_qc_change_audit")]
    assert "ax.plot(" in block
    assert "drawstyle=\"steps-post\"" not in block
    assert "ax.imshow" not in block
    assert "Manual QC flags are categorical" in block


def test_v126_manual_qc_offsets_overlapping_yes_no_traces():
    plots = _read(PLOTS)
    start = plots.index("manual_offsets = {}")
    end = plots.index("else:\n        ax.axhline", start)
    block = plots[start:end]
    assert "np.linspace" in block
    assert "small within-band offset" in block
    assert "raw_y + offset" in block
    assert "The table preserves exact Yes/No values" in block


def test_v126_manual_qc_keeps_exact_yes_no_labels():
    plots = _read(PLOTS)
    start = plots.index("if is_manual:", plots.index("ax.set_title(title"))
    end = plots.index("else:\n        ax.axhline", start)
    block = plots[start:end]
    assert 'ax.set_yticklabels(["No", "Yes"]' in block
    assert '"Yes" if raw_val > 0 else "No"' in plots
    assert "Small within-band offsets prevent overlapping flags" in block


def test_v126_automated_qc_still_uses_numeric_feature_style_lines():
    plots = _read(PLOTS)
    block = plots[plots.index("def _plot_longitudinal_qc_lines"):plots.index("def plot_longitudinal_qc_change_audit")]
    assert "Standardized automated QC change from own baseline" in block
    assert "Automated QC metrics are numeric" in block
