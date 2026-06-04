from pathlib import Path

import pandas as pd

from vslp.acoustic.metadata.stage import infer_metadata_column_mapping, _normalize_task_value


def test_metadata_column_mapping_detects_example_columns():
    df = pd.DataFrame({
        "Raw Media File name": ["S1.wav"],
        "SubjectID": ["S1"],
        "Protocol ID": [270],
        "Iteration": [1],
        "Recording date": ["2026-01-01"],
        "Task Name": ["Bamboo passage"],
        "Clinical Visit ID": [1],
        "Diagnosis": ["ALS"],
        "Date of Diagnosis": ["2025-01-01"],
        "ALSFRS total score": [41],
        "ALSFRS bulbar subscore": [10],
    })
    mapping, _ = infer_metadata_column_mapping(df)
    assert mapping["file_name"] == "Raw Media File name"
    assert mapping["subject_id"] == "SubjectID"
    assert mapping["protocol_id"] == "Protocol ID"
    assert mapping["iteration"] == "Iteration"
    assert mapping["recording_date"] == "Recording date"
    assert mapping["task"] == "Task Name"
    assert mapping["session_id"] == "Clinical Visit ID"
    assert mapping["diagnosis"] == "Diagnosis"
    assert mapping["severity_score"] == "ALSFRS total score"
    assert mapping["alsfrs_bulbar"] == "ALSFRS bulbar subscore"


def test_task_normalization():
    assert _normalize_task_value("Bamboo passage") == "bamboo"
    assert _normalize_task_value("VC3G") == "vc3g"
