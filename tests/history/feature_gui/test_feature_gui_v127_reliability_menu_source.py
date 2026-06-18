from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "vslp" / "gui" / "features" / "app.py"
PLOTS = ROOT / "src" / "vslp" / "analysis" / "features" / "plots.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_version_and_reliability_imports_present():
    text = read(APP)
    assert 'APP_VERSION = "v0.127.0"' in text
    assert "plot_reliability_design_support" in text
    assert "plot_reliability_measurement_error_landscape" in text
    assert "plot_selected_feature_same_task_reliability" in text


def test_reliability_controls_are_task_and_family_scoped():
    text = read(APP)
    assert "Task focus:" in text
    assert "Feature family:" in text
    assert "same subject" not in text.lower() or "same-task" in text.lower()
    assert "same-task reliability scope" in text
    assert "All tasks" in text
    assert "All families" in text


def test_reliability_plots_are_clinically_meaningful():
    text = read(APP)
    for label in [
        "Design support for reliability",
        "Feature ICC ranking",
        "Detectable-change / error landscape",
        "Reliability by feature family",
        "Selected feature same-task trajectory",
    ]:
        assert label in text
    assert "MDC95" in text
    assert "sem_within_subject" in text
    assert "n_repeated_subject_task_units" in text


def test_reliability_plot_functions_exist():
    text = read(PLOTS)
    assert "def plot_reliability_design_support" in text
    assert "def plot_reliability_measurement_error_landscape" in text
    assert "def plot_selected_feature_same_task_reliability" in text
    assert "MDC95 / total SD" in text
    assert "Repeated same-task recording order" in text


def test_reliability_preview_generates_on_demand():
    text = read(APP)
    assert "def _generate_reliability_plot" in text
    assert "self._display_plot_image(self.reliability_plot_preview, path)" in text
    assert "self.regenerate_overview_plots()" not in text[text.index("    def preview_reliability_plot"):text.index("    def update_reliability_interpretation")]
