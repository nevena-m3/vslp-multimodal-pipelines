"""Manual-reference reports are descriptive and require explicit token keys."""

from __future__ import annotations

import pandas as pd
import pytest

from vslp.acoustic.alignment.validation import compare_boundaries, compare_feature_sensitivity


def test_manual_reference_boundary_errors():
    mfa = pd.DataFrame([{"recording_id": "r", "word_index": 1,
                         "start_sec": 1.1, "end_sec": 1.5, "duration_sec": .4}])
    manual = pd.DataFrame([{"recording_id": "r", "word_index": 1,
                            "start_sec": 1, "end_sec": 1.6, "duration_sec": .6}])
    report = compare_boundaries(mfa, manual, token_type="word", tolerances_sec=(.15,))
    assert report.set_index("metric").loc["onset_absolute_error_sec", "median"] == pytest.approx(.1)
    assert report.set_index("metric").loc["duration_absolute_error_sec", "median"] == pytest.approx(.2)
    assert report.set_index("metric").loc["onset_absolute_error_sec", "within_0.15"] == 1


def test_feature_sensitivity_pairs_only_identical_feature_ids():
    mfa = pd.DataFrame([{"recording_id": "r", "feature_id": "mean_word_duration_s", "value": .41}])
    manual = pd.DataFrame([{"recording_id": "r", "feature_id": "mean_word_duration_s", "value": .50}])
    result = compare_feature_sensitivity(mfa, manual)
    assert result.iloc[0]["median"] == pytest.approx(.09)
