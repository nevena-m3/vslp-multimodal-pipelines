"""Feature result severity and completeness remain independent of alignment."""

from __future__ import annotations

import pandas as pd
import pytest

from vslp.gui.acoustic_app.feature_results import result_summary
from vslp.acoustic.features.catalog import project_task_key


def test_result_summary_distinguishes_expected_unavailable_from_error(tmp_path):
    path = tmp_path / "acoustic_feature_status_long.csv"
    pd.DataFrame([
        {"recording_id": "r1", "feature_id": "rate", "value": 97,
         "status": "computed", "failure_reason": ""},
        {"recording_id": "r1", "feature_id": "formant", "value": "",
         "status": "unavailable", "failure_reason": "alignment_package_required"},
        {"recording_id": "r1", "feature_id": "ddk", "value": "",
         "status": "unavailable", "failure_reason": "unsupported_for_current_task"},
        {"recording_id": "r1", "feature_id": "broken", "value": "",
         "status": "unavailable", "failure_reason": "family_executor_failed:RuntimeError:test"},
    ]).to_csv(path, index=False)
    summary, rows = result_summary(path)
    assert (summary["selected_features"], summary["recordings"]) == (4, 1)
    assert (summary["computed"], summary["unavailable"], summary["error"]) == (1, 2, 1)
    assert rows.set_index("feature_id").loc["formant", "severity"] == "UNAVAILABLE"
    assert rows.set_index("feature_id").loc["broken", "severity"] == "ERROR"
    streamed, no_rows = result_summary(path, include_rows=False)
    assert streamed == summary
    assert no_rows.empty


def test_result_summary_rejects_silently_duplicated_leaf(tmp_path):
    path = tmp_path / "status.csv"
    rows = [{"recording_id": "r1", "feature_id": "rate", "value": 1,
             "status": "computed", "failure_reason": ""}] * 2
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate_feature_status_row"):
        result_summary(path)


def test_current_bamboo_project_label_routes_to_frozen_passage():
    assert project_task_key({"task_name": "Bamboo", "task_id": "Bamboo",
                             "task_type": "Passage / connected speech",
                             "task_family": "other"}) == "bamboo_passage"


def test_package_errors_have_plain_language_message():
    from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow

    assert "source recording" in AcousticPipelineWindow._package_failure_message(
        "standalone_audio_identity_mismatch:record_000001")
    assert "time axis" in AcousticPipelineWindow._package_failure_message(
        "standalone_time_axis_mismatch:record_000001")
    assert "task or transcript" in AcousticPipelineWindow._package_failure_message(
        "standalone_task_prompt_mismatch:record_000001")
