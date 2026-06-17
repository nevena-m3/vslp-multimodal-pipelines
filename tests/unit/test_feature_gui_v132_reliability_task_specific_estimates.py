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


def _gui_with_task(task: str = "BAMBOO") -> FeatureAnalysisGUI:
    gui = FeatureAnalysisGUI.__new__(FeatureAnalysisGUI)
    df = pd.DataFrame(
        {
            "subject_id": ["S1", "S1", "S2", "S2", "S3", "S3"],
            "session_id": ["V1", "V2", "V1", "V2", "V1", "V2"],
            "task": ["BAMBOO", "BAMBOO", "BAMBOO", "BAMBOO", "DDK", "DDK"],
            "stable_feature": [1.0, 1.1, 3.0, 3.1, 8.0, 8.4],
            "sparse_feature": [pd.NA, pd.NA, 5.0, 5.1, pd.NA, pd.NA],
        }
    )
    gui.feature_df = df
    gui.analysis_df = df
    gui.meta_df = pd.DataFrame()
    gui.qc_df = pd.DataFrame()
    gui.registry_df = pd.DataFrame()
    gui.mapping_df = pd.DataFrame(
        [
            {"column": "subject_id", "role": "Identifier"},
            {"column": "session_id", "role": "Time / visit"},
            {"column": "task", "role": "Task"},
            {"column": "stable_feature", "role": "Feature"},
            {"column": "sparse_feature", "role": "Feature"},
        ]
    )
    gui.outputs = {}
    gui.reliability_source_combo = _Combo("Acoustic features", "features")
    gui.reliability_task_combo = _Combo(task, task)
    gui.reliability_family_combo = _Combo("All families", "All families")
    return gui


def _design_value(design: pd.DataFrame, metric: str):
    row = design.loc[design["metric"].astype(str).eq(metric)]
    assert not row.empty
    return row["value"].iloc[0]


def test_v132_task_specific_reliability_estimates_with_two_repeated_units():
    gui = _gui_with_task("BAMBOO")
    scoped = gui._build_scoped_reliability_outputs()
    design = scoped["design"]
    repeatability = scoped["repeatability"]

    assert _design_value(design, "task_scope") == "BAMBOO"
    assert _design_value(design, "rows") == 4
    assert _design_value(design, "repeated_subject_task_units") == 2

    stable = repeatability.loc[repeatability["feature"].eq("stable_feature")].iloc[0]
    assert pd.notna(stable["icc1_proxy"])
    assert stable["n_repeated_subject_task_units"] == 2
    assert "Preliminary estimate" in stable["interpretation"]


def test_v132_selected_feature_records_are_task_specific():
    gui = _gui_with_task("BAMBOO")
    records = gui._selected_feature_reliability_records("stable_feature")

    assert len(records) == 4
    assert set(records["task"].astype(str)) == {"BAMBOO"}
    assert set(records["subject"].astype(str)) == {"S1", "S2"}
    assert records.groupby("subject")["record_order"].max().to_dict() == {"S1": 2, "S2": 2}

