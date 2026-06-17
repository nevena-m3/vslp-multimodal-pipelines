import pandas as pd

from vslp.gui.features.app import FeatureAnalysisGUI


class _Combo:
    def __init__(self, text: str, data: str | None = None):
        self._text = text
        self._data = data if data is not None else text

    def count(self) -> int:
        return 1

    def currentData(self) -> str:
        return self._data

    def currentText(self) -> str:
        return self._text


def _gui() -> FeatureAnalysisGUI:
    return FeatureAnalysisGUI.__new__(FeatureAnalysisGUI)


def _screening_outputs() -> dict[str, pd.DataFrame]:
    cont = pd.DataFrame(
        [
            {"feature": "speech_rate", "outcome_variable": "alsfrs", "abs_effect": 0.42, "n_pairwise": 24},
            {"feature": "jitter_local", "outcome_variable": "alsfrs", "abs_effect": 0.18, "n_pairwise": 22},
        ]
    )
    cat = pd.DataFrame(
        [
            {"feature": "speech_rate", "group_variable": "diagnosis", "abs_effect": 0.35, "n_pairwise": 28},
        ]
    )
    return {
        "screening_summary": pd.DataFrame(
            [
                {"metric": "screening_variables_detected", "value": 2, "interpretation": "test"},
                {"metric": "continuous_feature_outcome_tests", "value": 2, "interpretation": "test"},
                {"metric": "categorical_group_feature_tests", "value": 1, "interpretation": "test"},
            ]
        ),
        "screening_variable_catalog": pd.DataFrame({"variable": ["alsfrs", "diagnosis"]}),
        "screening_continuous_outcome_associations": cont,
        "screening_categorical_group_associations": cat,
        "screening_group_balance": pd.DataFrame({"group_variable": ["diagnosis"], "level": ["ALS"], "n": [14]}),
    }


def test_v134_screening_ui_uses_workstation_layout_not_standard_gallery():
    text = open("src/vslp/gui/features/app.py", encoding="utf-8").read()
    section = text[text.index("def _screening_page"):text.index("def _selected_screening_task")]
    preview = text[text.index("def preview_screening_plot"):text.index("def update_screening_interpretation")]

    assert "self.screening_task_combo" in section
    assert "screening_split = QHBoxLayout()" in section
    assert "Screening details" in section
    assert "self.screening_priority_table" in section
    assert "_add_standard_plot_gallery" not in section
    assert "self.regenerate_overview_plots()" not in preview


def test_v134_screening_priority_effects_combines_continuous_and_group_tables():
    gui = _gui()
    outputs = _screening_outputs()

    priority = gui._screening_priority_effects(
        outputs["screening_continuous_outcome_associations"],
        outputs["screening_categorical_group_associations"],
    )

    assert priority.iloc[0]["feature"] == "speech_rate"
    assert priority.iloc[0]["variable"] == "alsfrs"
    assert set(priority["screening_type"]) == {"continuous_outcome", "categorical_group"}


def test_v134_screening_plot_generates_from_screening_outputs(tmp_path):
    gui = _gui()
    gui.output_dir = tmp_path / "feature_analysis"
    gui.outputs = _screening_outputs()
    gui.screening_task_combo = _Combo("All tasks (triage only)", "All tasks")
    gui.plot_paths = {}

    path = gui._generate_screening_plot("screening_effect_ranking")

    assert path is not None
    assert (tmp_path / "feature_analysis" / "plots").exists()
    assert "screening_effect_ranking" in gui.plot_paths
    assert gui.plot_paths["screening_effect_ranking"] == path
