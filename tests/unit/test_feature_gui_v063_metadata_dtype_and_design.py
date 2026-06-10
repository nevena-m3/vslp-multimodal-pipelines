from pathlib import Path

import pandas as pd

from vslp.analysis.features.audit import design_overview
from vslp.analysis.features.column_mapping import classify_columns
from vslp.analysis.features.plots import plot_dataset_design_tiles


def test_v063_source_version_and_metadata_fill_cast():
    source = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.63.0"' in source
    assert 'merged[canonical] = merged[canonical].astype("object")' in source
    assert 'fill_values = merged.loc[empty_mask, metadata_col].astype("object")' in source


def test_design_overview_marks_empty_columns_not_detected_metadata():
    df = pd.DataFrame({
        "subject_id": [float("nan"), float("nan")],
        "session_id": [float("nan"), float("nan")],
        "diagnosis": [float("nan"), float("nan")],
        "feature_a": [1.0, 2.0],
    })
    mapping = classify_columns(df, table_kind="feature")
    design = design_overview(df, mapping)
    subject = design.loc[design["variable_type"] == "subject"].iloc[0]
    diagnosis = design.loc[design["variable_type"] == "diagnosis"].iloc[0]
    assert subject["column"] == "subject_id"
    assert subject["status"] == "empty_column"
    assert int(subject["n_unique"]) == 0
    assert diagnosis["status"] == "empty_column"


def test_design_context_plot_accepts_empty_column_status(tmp_path):
    design = pd.DataFrame({
        "variable_type": ["subject", "diagnosis", "task"],
        "column": ["subject_id", "diagnosis", "task"],
        "status": ["empty_column", "detected", "missing"],
        "n_unique": [0, 2, 0],
        "n_missing": [10, 0, 10],
        "top_values": ["", "ALS: 5; Control: 5", ""],
    })
    out = plot_dataset_design_tiles(design, tmp_path / "design.png")
    assert out.exists()
