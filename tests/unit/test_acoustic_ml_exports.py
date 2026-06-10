from pathlib import Path

import pandas as pd

from vslp.acoustic.aggregate.stage import acoustic_feature_manifest_dataframe, write_acoustic_ml_ready_exports


def test_acoustic_feature_manifest_marks_registry_features_as_ml_candidates():
    manifest = acoustic_feature_manifest_dataframe(["subject_id", "task", "speech_rate", "diagnosis", "qc_status"])
    by_col = manifest.set_index("column_name")
    assert by_col.loc["speech_rate", "column_role"] == "canonical_feature"
    assert bool(by_col.loc["speech_rate", "include_in_ml_default"])
    assert by_col.loc["subject_id", "column_role"] == "metadata"
    assert not bool(by_col.loc["subject_id", "include_in_ml_default"])
    assert by_col.loc["diagnosis", "column_role"] == "label_or_outcome"
    assert by_col.loc["qc_status", "column_role"] == "qc_metric"


def test_write_acoustic_ml_ready_exports_creates_predictor_table(tmp_path: Path):
    df = pd.DataFrame(
        {
            "subject_id": ["S1", "S2"],
            "session_id": ["V1", "V1"],
            "task": ["passage", "passage"],
            "diagnosis": ["ALS", "Control"],
            "speech_rate": [120.0, 180.0],
            "localJitter": [0.8, 0.3],
            "source_file_path": ["a.wav", "b.wav"],
        }
    )
    paths = write_acoustic_ml_ready_exports(df, tmp_path)
    ml_ready = pd.read_csv(paths["ml_ready_csv"])
    manifest = pd.read_csv(paths["feature_manifest_csv"])
    assert "subject_id" in ml_ready.columns
    assert "speech_rate" in ml_ready.columns
    assert "localJitter" in ml_ready.columns
    assert "diagnosis" not in ml_ready.columns
    assert "source_file_path" not in ml_ready.columns
    assert set(["column_name", "column_role", "include_in_ml_default"]).issubset(manifest.columns)
