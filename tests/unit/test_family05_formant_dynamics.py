"""Family 05 generic calculations over frozen Family 04 formant frames."""

import json

import numpy as np
import pandas as pd
import pytest

from vslp.acoustic.features.catalog import load_feature_catalog
from vslp.acoustic.features.family04 import FRAME_COLUMNS
from vslp.acoustic.features.family05 import (
    FEATURE_ID_TEMPLATES, aggregate_equivalent_targets,
    load_formant_target_manifest, load_shared_formant_frames,
    range_from_fitted_trajectory, regression_formant_slope,
    select_target_tokens,
)


def _frames():
    times = np.arange(0.0, 0.101, 0.01)
    return pd.DataFrame({
        "token_index": np.zeros(len(times), dtype=int),
        "time_sec": times,
        "f1_hz": 300 + 1000 * times,
        "f2_hz": 1500 - 500 * times,
        "f3_hz": np.full(len(times), 2500.),
        "valid_ordered": np.ones(len(times), dtype=bool),
        "in_middle_50_percent": np.ones(len(times), dtype=bool),
    })


def test_exact_templates_without_public_leaves():
    assert FEATURE_ID_TEMPLATES == (
        "f1_slope_<target>_hz_s", "f2_slope_<target>_hz_s",
        "f1_range_<diphthong>_hz", "f2_range_<diphthong>_hz",
    )
    catalog = load_feature_catalog()
    assert not [row for row in catalog["outputs"] if row["family_id"] == "F05"]
    assert {row["construct_id"] for row in catalog["constructs"] if row["family_id"] == "F05"} == {"C033", "C034"}


def test_ols_slopes_and_frame_requirements():
    frames = _frames()
    f1, reason, count = regression_formant_slope(frames, token_index=0,
        token_start_sec=0, token_end_sec=.1, formant_number=1)
    assert f1 == pytest.approx(1000)
    assert reason == "" and count >= 5
    f2, reason, _ = regression_formant_slope(frames, token_index=0,
        token_start_sec=0, token_end_sec=.1, formant_number=2)
    assert f2 == pytest.approx(-500) and reason == ""
    value, reason, _ = regression_formant_slope(frames, token_index=0,
        token_start_sec=0, token_end_sec=.03, formant_number=1)
    assert np.isnan(value) and reason == "transition_too_short"
    value, reason, _ = regression_formant_slope(frames.iloc[:3], token_index=0,
        token_start_sec=0, token_end_sec=.1, formant_number=1)
    assert np.isnan(value) and reason == "insufficient_valid_trajectory_frames"


def test_fitted_range_and_target_aggregation():
    assert range_from_fitted_trajectory(np.array([100, 120, 140, 160, 180])) == (80, "")
    assert np.isnan(range_from_fitted_trajectory(np.array([1, 2]))[0])
    assert aggregate_equivalent_targets([100, 300, 200]) == (200, "")
    assert aggregate_equivalent_targets([])[1] == "no_valid_target_tokens"


def test_manifest_requires_explicit_labels_and_ids(tmp_path):
    path = tmp_path / "targets.json"
    data = {"manifest_version": "1", "task_id": "bamboo", "targets": [{
        "target_id": "approved_transition", "target_type": "transition",
        "expected_word": "EXACT", "expected_phone": "",
        "alignment_selection_rule": "exact_word", "window_start_fraction": .2,
        "window_end_fraction": .8, "aggregation": "median_across_equivalent_tokens",
        "f1_feature_id": "f1_slope_source_target_hz_s",
        "f2_feature_id": "f2_slope_source_target_hz_s", "fitted_trajectory_source": "",
    }]}
    path.write_text(json.dumps(data), encoding="utf-8")
    (target,) = load_formant_target_manifest(path, "bamboo")
    assert target.f1_feature_id == "f1_slope_source_target_hz_s"
    tokens = pd.DataFrame({"word_index": [1, 2], "phone_normalized": ["AY", "AY"]})
    words = pd.DataFrame({"word_index": [1, 2], "word": ["EXACT", "OTHER"]})
    assert len(select_target_tokens(target, tokens, words)) == 1
    with pytest.raises(ValueError, match="mismatch"):
        load_formant_target_manifest(path, "wstg")
    data["targets"][0]["f1_feature_id"] = "f1_slope_<target>_hz_s"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="feature_ids_not_frozen"):
        load_formant_target_manifest(path, "bamboo")


def test_shared_frame_npz_provenance(tmp_path):
    path = tmp_path / "track.npz"
    base = _frames()
    arrays = {key: base[key].to_numpy() for key in FRAME_COLUMNS}
    np.savez(path, **arrays, formant_profile_id="profile", alignment_run_id="alignment")
    loaded = load_shared_formant_frames(path, profile_id="profile", alignment_run_id="alignment")
    assert loaded.equals(base[list(FRAME_COLUMNS)])
    with pytest.raises(ValueError, match="provenance"):
        load_shared_formant_frames(path, profile_id="wrong", alignment_run_id="alignment")
