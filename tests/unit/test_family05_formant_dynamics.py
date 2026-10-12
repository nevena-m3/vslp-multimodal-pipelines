"""Family 05 generic calculations over frozen Family 04 formant frames."""

import json

import numpy as np
import pandas as pd
import pytest

from vslp.acoustic.features.catalog import load_feature_catalog, task_fit
from vslp.acoustic.alignment.task_workflow import task_entry
from vslp.acoustic.features.family04 import FRAME_COLUMNS, profile_by_id, track_aligned_vowels
from vslp.acoustic.features.family05 import (
    FAMILY05_IDS, FEATURE_ID_TEMPLATES, TARGETS, aggregate_equivalent_targets,
    load_formant_target_manifest, load_shared_formant_frames,
    measure_target_frames, range_from_fitted_trajectory, regression_formant_slope,
    resolve_target_phones, robust_bisquare_fit, transition_window,
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
        "b1_hz": np.full(len(times), 90.),
        "b2_hz": np.full(len(times), 120.),
        "b3_hz": np.full(len(times), 150.),
        "valid_ordered": np.ones(len(times), dtype=bool),
        "in_middle_50_percent": np.ones(len(times), dtype=bool),
    })


def test_exact_templates_and_eight_public_leaves():
    assert FEATURE_ID_TEMPLATES == (
        "f1_slope_<target>_hz_s", "f2_slope_<target>_hz_s",
        "f1_range_<diphthong>_hz", "f2_range_<diphthong>_hz",
    )
    catalog = load_feature_catalog()
    outputs = [row for row in catalog["outputs"] if row["family_id"] == "F05"]
    assert len(outputs) == 8
    assert {row["feature_id"] for row in outputs} == FAMILY05_IDS
    assert all(row["selectable"] for row in outputs)
    assert {row["task_id"] for row in outputs} == {"sentence_s2", "sentence_s5"}
    assert {row["construct_id"] for row in catalog["constructs"] if row["family_id"] == "F05"} == {"C033", "C034"}
    assert task_entry("sentence_s2")["prompts"][0]["exact_expected_text"] == "Buy Bobby a puppy."
    s2_prompt = task_entry("sentence_s2")["prompts"][0]
    assert s2_prompt["expected_repetitions"] == 3
    assert s2_prompt["current_protocol_expected_repetitions"] == 3
    assert s2_prompt["historical_jafari_protocol_repetitions"] == 10
    assert s2_prompt["historical_protocol_is_not_current_task_contract"] is True
    assert task_fit(next(row for row in outputs if row["feature_id"] ==
                         "f1_range_buy_ay_hz"), "S2") == "Recommended"
    assert task_fit(next(row for row in outputs if row["feature_id"] ==
                         "f1_range_buy_ay_hz"), "S5") == "Not specified"


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


def test_central_window_minimum_and_signed_robust_fit():
    assert transition_window(10, 10.1) == pytest.approx((10.02, 10.08))
    with pytest.raises(ValueError, match="target_too_short"):
        transition_window(0, .039)
    frames = _frames()
    frames.loc[5, "f1_hz"] = 1100  # gross isolated LPC excursion
    result = measure_target_frames(frames, start=0, end=.1, token_index=0)
    assert result["qc_status"] == "OK"
    assert result["n_valid_frames"] >= 5
    assert result["f1_slope_hz_s"] == pytest.approx(1000, abs=200)
    assert result["f2_slope_hz_s"] == pytest.approx(-500, abs=1)
    assert result["f2_fitted_range_hz"] == pytest.approx(30, abs=.01)
    assert result["f1_fitted_range_hz"] == pytest.approx(
        abs(result["f1_slope_hz_s"]) * .06)
    frames = frames.loc[frames.time_sec < .04]
    failed = measure_target_frames(frames, start=0, end=.1, token_index=0)
    assert np.isnan(failed["f1_slope_hz_s"])
    assert failed["qc_flags"] == "insufficient_formant_frames"
    invalid = _frames()
    invalid["f2_hz"] = 100  # shared F04 physiological range gate
    failed_track = measure_target_frames(invalid, start=0, end=.1, token_index=0)
    assert "formant_track_invalid" in failed_track["qc_flags"]
    assert np.isnan(failed_track["f1_slope_hz_s"])
    flat = robust_bisquare_fit(np.linspace(0, .1, 8), np.full(8, 500.))
    assert flat["slope"] == pytest.approx(0, abs=.001)
    assert abs(flat["slope"]) * .06 == pytest.approx(0, abs=.001)


def test_final_alignment_target_parent_identity():
    class Store:
        trials = pd.DataFrame([{"recording_id": "r", "trial_id": "t",
                                "task_id": "sentence_s2",
                                "prompt_id": "sentence_s2_buy_bobby_a_puppy"}])

        def words_for_trial(self, *_):
            return pd.DataFrame([{"word_id": "w1", "canonical_word_text": "buy"},
                                 {"word_id": "w2", "canonical_word_text": "guy"}])

        def phones_for_trial(self, *_):
            return pd.DataFrame([{"parent_word_id": "w1", "parent_word_text": "buy",
                                  "phone_normalized": "AY"},
                                 {"parent_word_id": "w2", "parent_word_text": "guy",
                                  "phone_normalized": "AY"}])

    assert TARGETS["buy_ay"] == ("sentence_s2", "buy", "AY")
    assert TARGETS["away_ey"] == ("sentence_s5", "away", "EY")
    assert TARGETS["ago_ow"] == ("sentence_s5", "ago", "OW")
    selected, issue = resolve_target_phones(Store(), "r", "t", "buy_ay")
    assert issue == "" and len(selected) == 1
    assert selected.iloc[0].parent_word_text == "buy"
    wrong = Store()
    wrong.trials = wrong.trials.copy()
    wrong.trials.loc[0, "task_id"] = "sentence_s5"
    assert resolve_target_phones(wrong, "r", "t", "buy_ay")[1] == "wrong_task"

    class WrongParent(Store):
        def phones_for_trial(self, *_):
            return pd.DataFrame([{"parent_word_id": "w2", "parent_word_text": "guy",
                                  "phone_normalized": "AY"}])

    assert resolve_target_phones(WrongParent(), "r", "t", "buy_ay")[1] == "wrong_parent_word"


def test_f05_reuses_f04_burg_with_full_support_and_20_80_window():
    class Formant:
        def xs(self):
            return np.arange(.10, .201, .005)

        def get_value_at_time(self, number, time):
            return (400., 1800., 2900.)[number - 1]

    tokens = pd.DataFrame([{"start_sec": .10, "end_sec": .20,
                            "trial_start_sec": .10, "trial_end_sec": .20,
                            "phone_raw": "AY1", "phone_normalized": "AY"}])
    _, frames = track_aligned_vowels(
        np.zeros(16000), 16000, tokens, profile_by_id(), recording_id="r",
        file_name="native.wav", task_id="sentence_s2", task_type="sentence",
        alignment_run_id="a", formant_object=Formant(), analysis_window=(.2, .8))
    assert frames.time_sec.min() == pytest.approx(.12)
    assert .175 - 1e-9 <= frames.time_sec.max() <= .18 + 1e-9
    assert frames.in_middle_50_percent.sum() < len(frames)
    _, excluded = track_aligned_vowels(
        np.zeros(16000), 16000, tokens, profile_by_id(), recording_id="r",
        file_name="native.wav", task_id="sentence_s2", task_type="sentence",
        alignment_run_id="a", formant_object=Formant(), analysis_window=(.2, .8),
        hard_exclusions=[(.14, .16)])
    assert all(not (time - .0125 < .16 and time + .0125 > .14)
               for time in excluded.time_sec)
