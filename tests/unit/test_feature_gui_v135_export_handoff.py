import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageStat

from vslp.gui.features.app import (
    FeatureAnalysisGUI,
    QMessageBox,
    ROLE_COVARIATE,
    ROLE_FEATURE,
    ROLE_IDENTIFIER,
    ROLE_QC,
    ROLE_TARGET,
)
from vslp.analysis.features.plots import plot_ml_export_manifest_summary


class _Combo:
    def __init__(self, text: str):
        self._text = text

    def count(self) -> int:
        return 1

    def currentText(self) -> str:
        return self._text


class _Picker:
    path = ""


def _gui() -> FeatureAnalysisGUI:
    return FeatureAnalysisGUI.__new__(FeatureAnalysisGUI)


def _recommendations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": "speech_rate",
                "family_or_subsystem": "timing",
                "readiness_recommendation": "recommended",
                "readiness_score": 90,
                "primary_reasons": "no major review flags detected",
                "recommended_action": "include",
                "ml_export_default": True,
            },
            {
                "feature": "jitter_local",
                "family_or_subsystem": "voice",
                "readiness_recommendation": "review_before_use",
                "readiness_score": 45,
                "primary_reasons": "strong QC association",
                "recommended_action": "review",
                "ml_export_default": False,
            },
        ]
    )


def test_v135_menu_order_places_export_before_advanced_builder():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")

    assert text.index('(\"export\", \"o  Export / Report\")') < text.index(
        '(\"ml_export\", \"o  Advanced ML Export Builder\")'
    )
    assert text.index('\"recommendations\", \"export\", \"ml_export\"') > 0
    assert '(\"Export / Report\", self.update_export_dashboard)' in text


def test_v135_analysis_navigation_is_passive_and_logs_once():
    gui = _gui()
    gui.analysis_ready = False
    gui.outputs = {}
    gui.analysis_navigation_notice_logged = False
    messages = []
    gui.log = messages.append

    assert gui._prompt_run_analysis_before_menu("missing") is True
    assert gui._prompt_run_analysis_before_menu("recommendations") is True
    assert len(messages) == 1


def test_v135_context_acceptance_prompt_is_only_shown_once(monkeypatch):
    gui = _gui()
    gui.analysis_ready = False
    gui.outputs = {}
    gui.analysis_prompt_responded = False
    gui.log = lambda _message: None
    shown = []
    gui.show_page = shown.append
    calls = []

    def question(*_args, **_kwargs):
        calls.append(True)
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "question", question)
    gui._prompt_run_analysis_after_context_accept("Metadata mapping")
    gui._prompt_run_analysis_after_context_accept("Metadata mapping")

    assert len(calls) == 1
    assert shown == ["overview", "overview"]


def test_v135_export_page_has_task_workstation_and_package_status_table():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    section = text[text.index("def _export_page"):text.index("def _export_profile_key")]

    assert "self.export_task_combo" in section
    assert "export_split = QHBoxLayout()" in section
    assert "details.setFixedWidth(330)" in section
    assert "self.export_package_table" in section
    assert "Package contents / status" in section
    assert "All tasks is retained for audit" in section
    assert "card.layout.addLayout(self.export_metric_grid)" not in section


def test_v135_task_scoped_export_writes_complete_handoff_package(tmp_path):
    gui = _gui()
    gui.feature_df = pd.DataFrame(
        {
            "subject_id": ["P01", "P02", "P01"],
            "task": ["READ", "READ", "DDK"],
            "outcome": [10.0, 20.0, 10.0],
            "age": [60, 70, 60],
            "snr": [25.0, 22.0, 30.0],
            "speech_rate": [1.1, 1.4, 3.0],
            "jitter_local": [0.1, 0.2, 0.9],
        }
    )
    gui.analysis_df = gui.feature_df.copy()
    gui.qc_df = None
    gui.meta_df = None
    gui.registry_df = None
    gui.outputs = {
        "feature_recommendations": _recommendations(),
        "feature_recommendation_summary": pd.DataFrame([{"metric": "features_reviewed", "value": 2}]),
    }
    gui.output_dir = tmp_path / "feature_analysis"
    gui.output_edit = type("Edit", (), {"text": lambda self: str(tmp_path)})()
    gui.export_task_combo = _Combo("READ")
    gui.export_profile_combo = _Combo("Recommended + caution (default ML starting set)")
    gui.feature_picker = _Picker()
    gui.qc_picker = _Picker()
    gui.meta_picker = _Picker()
    gui.registry_picker = _Picker()
    gui.mapping_df = pd.DataFrame(
        {
            "column": ["subject_id", "task", "outcome", "age", "snr", "speech_rate", "jitter_local"],
            "role": [ROLE_IDENTIFIER, ROLE_COVARIATE, ROLE_TARGET, ROLE_COVARIATE, ROLE_QC, ROLE_FEATURE, ROLE_FEATURE],
        }
    )
    gui.collect_mapping_from_table = lambda: gui.mapping_df
    gui._export_scoped_outputs = lambda _outputs: gui.outputs
    gui._export_scoped_frame = lambda task: gui.analysis_df.loc[gui.analysis_df["task"].eq(task)].copy()
    gui.write_report = lambda path, *_args, **_kwargs: Path(path).write_text("<html>report</html>", encoding="utf-8")

    export_dir, paths = gui._make_export_package_tables()

    expected = {
        "ml_ready_feature_matrix.csv",
        "ml_target_table.csv",
        "ml_covariate_table.csv",
        "ml_qc_covariate_table_from_feature_table.csv",
        "feature_export_manifest.csv",
        "feature_recommendation_summary.csv",
        "export_profile_summary.csv",
        "feature_recommendation_legend.csv",
        "README.md",
        "export_config.json",
        "vslp_feature_analysis_export_report.html",
    }
    assert expected.issubset({p.name for p in export_dir.iterdir()})
    assert paths["export_zip"].exists()
    matrix = pd.read_csv(export_dir / "ml_ready_feature_matrix.csv")
    manifest = pd.read_csv(export_dir / "feature_export_manifest.csv")
    config = json.loads((export_dir / "export_config.json").read_text(encoding="utf-8"))
    assert len(matrix) == 2
    assert set(matrix["task"]) == {"READ"}
    assert "speech_rate" in matrix.columns
    assert "jitter_local" not in matrix.columns
    assert set(manifest["task_scope"]) == {"READ"}
    assert config["task_specific"] is True
    assert config["task_n_rows"] == 2


def test_v135_advanced_builder_keeps_ml_preprocessing_out_of_scope():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    section = text[text.index("def _ml_export_builder_page"):text.index("def pick_ml_export_output_folder")]

    assert "Advanced ML Export Builder" in section
    assert "does not split subjects, impute, scale, transform, select predictors, or train a model" in section
    assert "Safe early fusion requires a shared unique key" in section
    assert "self.ml_export_package_table" in section
    assert "acoustic_ml_ready.csv" in section
    assert "multimodal_early_fusion_ml_ready.csv" in section
    assert "ml_export_manifest.json" in section


def test_v135_export_plot_renders_current_final_include_manifest(tmp_path):
    manifest = _recommendations().copy()
    manifest["final_include"] = [True, False]
    path = plot_ml_export_manifest_summary(manifest, tmp_path / "export_manifest.png")

    assert path.exists()
    image = Image.open(path).convert("RGB")
    assert image.width >= 700
    assert ImageStat.Stat(image).var[0] > 20


def test_v135_export_preview_is_cached_and_defaults_to_a_specific_task():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    refresh = text[text.index("def _refresh_export_task_combo"):text.index("def _selected_export_task")]
    plot = text[text.index("def _generate_export_plot"):text.index("def _export_interpretation_html")]

    assert "if not initialized and tasks" in refresh
    assert "combo.setCurrentText(tasks[0])" in refresh
    assert "_export_plot_token" in plot
    assert "and path.exists()" in plot
