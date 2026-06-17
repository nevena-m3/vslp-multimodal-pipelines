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


class _Search:
    def __init__(self, text: str = ""):
        self._text = text

    def text(self) -> str:
        return self._text


def _gui() -> FeatureAnalysisGUI:
    return FeatureAnalysisGUI.__new__(FeatureAnalysisGUI)


def _recommendations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": "speech_rate",
                "family_or_subsystem": "respiratory_timing",
                "readiness_recommendation": "recommended",
                "readiness_score": 91,
                "primary_reasons": "no major review flags detected",
                "recommended_action": "eligible for downstream ML export after standard leakage-safe preprocessing",
                "ml_export_default": True,
            },
            {
                "feature": "jitter_local",
                "family_or_subsystem": "phonatory_voice_quality",
                "readiness_recommendation": "review_before_use",
                "readiness_score": 48,
                "primary_reasons": "strong QC association; variable repeatability",
                "recommended_action": "do not interpret without QC sensitivity/covariate analysis",
                "ml_export_default": False,
            },
            {
                "feature": "dead_feature",
                "family_or_subsystem": "audit",
                "readiness_recommendation": "exclude_or_recompute",
                "readiness_score": 12,
                "primary_reasons": "zero variance; extreme missingness",
                "recommended_action": "exclude by default unless kept for audit only",
                "ml_export_default": False,
            },
        ]
    )


def test_v133_recommendation_action_plan_explains_each_decision():
    gui = _gui()
    plan = gui._recommendation_action_plan(_recommendations())

    assert set(plan["decision"]) == {
        "recommended",
        "recommended_with_caution",
        "review_before_use",
        "exclude_or_recompute",
        "exclude_by_default",
    }
    review = plan.loc[plan["decision"].eq("review_before_use")].iloc[0]
    assert review["n_features"] == 1
    assert "strong QC association" in review["top_current_reasons"]
    assert review["default_export_policy"] == "hold_pending_review"


def test_v133_priority_review_routes_features_to_source_menus():
    gui = _gui()
    priority = gui._recommendation_priority_review(_recommendations())

    assert priority.iloc[0]["feature"] == "dead_feature"
    assert priority.loc[priority["feature"].eq("dead_feature"), "next_screen"].iloc[0] == "Distributions / Outliers"
    assert priority.loc[priority["feature"].eq("jitter_local"), "next_screen"].iloc[0] == "QC Integration"


def test_v133_handoff_summary_counts_export_sets():
    gui = _gui()
    handoff = gui._recommendation_handoff_summary(_recommendations())
    values = dict(zip(handoff["handoff_item"], handoff["value"]))

    assert values["default_export_features"] == 1
    assert values["manual_review_queue"] == 1
    assert values["hold_or_recompute"] == 1


def test_v133_recommendation_ui_has_decision_filters_and_tables():
    text = open("src/vslp/gui/features/app.py", encoding="utf-8").read()

    assert 'APP_VERSION = "v0.133.0"' in text
    assert "self.recommendation_decision_combo" in text
    assert "self.recommendation_family_combo" in text
    assert "self.recommendation_search_edit" in text
    assert "self.recommendation_action_table" in text
    assert "self.recommendation_priority_table" in text
    assert "ML / export handoff" in text
    assert "self.regenerate_overview_plots()" not in text[text.index("def preview_recommendation_plot"):text.index("def update_recommendation_interpretation")]


def test_v133_recommendation_plot_generates_from_outputs_without_overview(tmp_path):
    gui = _gui()
    gui.output_dir = tmp_path / "feature_analysis"
    gui.outputs = {
        "feature_recommendations": _recommendations(),
        "ml_export_manifest": _recommendations()[["feature", "readiness_recommendation"]].copy(),
    }
    gui.recommendation_decision_combo = _Combo("All decisions", "__all__")
    gui.recommendation_family_combo = _Combo("All families", "All families")
    gui.recommendation_search_edit = _Search("")
    gui.plot_paths = {}

    path = gui._generate_recommendation_plot("recommendation_counts")

    assert path is not None
    assert (tmp_path / "feature_analysis" / "plots").exists()
    assert "recommendation_counts" in gui.plot_paths
    assert gui.plot_paths["recommendation_counts"] == path
    assert pd.notna(path)
