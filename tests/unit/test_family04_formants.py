"""Exact Family 04 profiles, token rules, and hand-calculated centroids."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vslp.acoustic.features.catalog import feature_availability, load_feature_catalog
from vslp.acoustic.features.family04 import (
    DEFAULT_PROFILE_ID, DERIVED_IDS, FAMILY04_IDS, SENSITIVITY_PROFILE_ID,
    aggregate_vowel_centroids, derive_iau_features, load_vowel_category_mapping,
    profile_by_id, summarize_formant_token, track_aligned_vowels, vowel_dispersion,
    vowel_distance,
)


def test_registry_exact_ids_evidence_and_templates():
    catalog = load_feature_catalog()
    family = [item for item in catalog["constructs"] if item["family_id"] == "F04"]
    assert len(family) == 12
    outputs = {item["feature_id"]: item for item in catalog["outputs"] if item["family_id"] == "F04"}
    assert FAMILY04_IDS <= set(outputs)
    assert "vowel_distance_<x>_<y>_hz" not in outputs
    assert "vowel_dispersion_<variant>_hz" not in outputs
    assert all(outputs[key]["evidence_level"] for key in outputs)
    assert outputs["f2_token_hz"]["evidence_study_count"] == 9
    assert outputs["vsa3_iau_hz2"]["evidence_level"] == "MODERATE"
    assert all(feature_availability(outputs[key])[0] == "Implemented" for key in FAMILY04_IDS)
    assert feature_availability(outputs["vowel_envelope_distance_ai_source"])[0] == "Not implemented"
    assert outputs["f1_token_hz"]["output_granularity"] == "token"
    assert outputs["f1_vowel_median_hz"]["output_granularity"] == "vowel"


def test_frozen_profiles_and_sample_rate_gate():
    default = profile_by_id()
    sensitivity = profile_by_id(SENSITIVITY_PROFILE_ID)
    assert default.profile_id == DEFAULT_PROFILE_ID
    assert (default.window_length_sec, default.time_step_sec, default.max_formants,
            default.pre_emphasis_from_hz, default.ceiling_hz) == (.025, .005, 5, 50, 5500)
    assert sensitivity.ceiling_hz == 5000
    with pytest.raises(ValueError, match="unknown_formant_profile"):
        profile_by_id("sex_inferred")
    with pytest.raises(ValueError, match="incompatible_native_sample_rate"):
        track_aligned_vowels(np.zeros(8000), 8000, pd.DataFrame(), default,
                             recording_id="r", file_name="x.wav", task_id="bamboo",
                             task_type="", alignment_run_id="a")


def test_middle_half_minimum_frames_order_and_range_flags():
    profile = profile_by_id()
    times = np.arange(0, .101, .005)
    frames = pd.DataFrame({"time_sec": times, "f1_hz": 500., "f2_hz": 1500.,
                           "f3_hz": 2700., "valid_ordered": True})
    frames.loc[frames.time_sec.eq(.05), "f2_hz"] = 4000.
    frames.loc[frames.time_sec.eq(.05), "valid_ordered"] = True
    frames.loc[frames.time_sec.eq(.04), "valid_ordered"] = False
    measured = summarize_formant_token(frames, 0, .1, profile)
    assert measured["measurement_start_sec"] == pytest.approx(.025)
    assert measured["measurement_end_sec"] == pytest.approx(.075)
    assert measured["n_valid_frames"] >= 3
    assert measured["f2_token_hz"] == 1500
    assert "f2_outside_plausibility_range" in measured["review_flags"]
    assert "nonordered_formant_frames" in measured["review_flags"]
    assert summarize_formant_token(frames, 0, .049, profile)["tracking_status"] == "TOKEN_TOO_SHORT"
    sparse = frames.loc[frames.time_sec.isin([.045, .05])]
    assert summarize_formant_token(sparse, 0, .1, profile)["tracking_status"] == "INSUFFICIENT_VALID_FRAMES"


def test_shared_tracker_uses_aligned_token_and_preserves_waveform():
    class FrozenFormant:
        def xs(self):
            return np.arange(0, .3, .005)

        def get_value_at_time(self, number, time):
            return {1: 400, 2: 1800, 3: 2900}[number]

    audio = np.zeros(16000, dtype="float32")
    copy = audio.copy()
    tokens = pd.DataFrame([{"start_sec": .1, "end_sec": .2, "word_index": 1,
                            "phone_index": 1, "phone_raw": "IY1",
                            "phone_normalized": "IY"}])
    measurements, frames = track_aligned_vowels(
        audio, 16000, tokens, profile_by_id(), recording_id="r", file_name="x.wav",
        task_id="bamboo", task_type="Passage / connected speech", alignment_run_id="a",
        category_mapping={"IY": "i"}, formant_object=FrozenFormant())
    assert np.array_equal(audio, copy)
    assert measurements.iloc[0].f1_token_hz == 400
    assert measurements.iloc[0].vowel_category == "i"
    assert measurements.iloc[0].n_valid_frames >= 3
    assert frames.in_middle_50_percent.sum() == measurements.iloc[0].n_valid_frames
    assert not hasattr(profile_by_id(), "sex")


def test_native_rate_parselmouth_burg_synthetic_behavior():
    rate = 16000
    times = np.arange(rate) / rate
    audio = sum(weight * np.sin(2 * np.pi * frequency * times)
                for weight, frequency in ((.2, 500), (.15, 1500), (.1, 2700)))
    phones = pd.DataFrame([{"start_sec": .2, "end_sec": .7, "word_index": 1,
                            "phone_index": 1, "phone_raw": "IY1", "phone_normalized": "IY"}])
    tokens, frames = track_aligned_vowels(
        audio, rate, phones, profile_by_id(), recording_id="r", file_name="x.wav",
        task_id="bamboo", task_type="", alignment_run_id="a")
    assert tokens.tracking_status.iloc[0] == "OK"
    assert tokens.n_valid_frames.iloc[0] >= 3
    assert frames.time_sec.min() >= .2 and frames.time_sec.max() <= .7
    assert tokens.f1_token_hz.iloc[0] < tokens.f2_token_hz.iloc[0] < tokens.f3_token_hz.iloc[0]


def test_mapping_is_explicit_versioned_and_rejects_overlap(tmp_path):
    assert load_vowel_category_mapping(None, "bamboo")[1] == "missing_vowel_category_mapping"
    path = tmp_path / "vowels.json"
    value = {"manifest_version": "1", "task_id": "bamboo", "phone_set": "ARPABET_CMU_39",
             "categories": [{"canonical_vowel": "i", "accepted_phone_labels": ["IY"]},
                            {"canonical_vowel": "a", "accepted_phone_labels": ["AA"]},
                            {"canonical_vowel": "u", "accepted_phone_labels": ["UW"]}]}
    path.write_text(json.dumps(value), encoding="utf-8")
    mapping, reason, digest = load_vowel_category_mapping(path, "bamboo")
    assert reason == "" and len(digest) == 64
    assert mapping == {"IY": "i", "AA": "a", "UW": "u"}
    assert load_vowel_category_mapping(path, "wstg")[1] == "vowel_mapping_manifest_mismatch"
    value["categories"][1]["accepted_phone_labels"] = ["IY"]
    path.write_text(json.dumps(value), encoding="utf-8")
    assert load_vowel_category_mapping(path, "bamboo")[1] == "vowel_mapping_phone_invalid_or_duplicated"


def test_vowel_aggregation_and_all_exact_iau_formulas():
    tokens = pd.DataFrame([{"recording_id": "r", "file_name": "x.wav", "task_id": "bamboo",
                            "alignment_run_id": "a", "formant_profile_id": DEFAULT_PROFILE_ID,
                            "vowel_category": category, "tracking_status": "OK",
                            "f1_token_hz": f1, "f2_token_hz": f2, "f3_token_hz": f3}
                           for category, f1, f2, f3 in (("i", 300, 2300, 3100),
                                                         ("a", 800, 1200, 2900),
                                                         ("u", 350, 900, 2800))])
    vowels = aggregate_vowel_centroids(tokens)
    assert set(vowels.target_vowel) == {"i", "a", "u"}
    assert vowels.loc[vowels.target_vowel.eq("i"), "f2_vowel_median_hz"].iloc[0] == 2300
    assert vowels.review_flags.eq("low_n_vowel_tokens").all()
    results = derive_iau_features(vowels, True)
    assert set(results) == DERIVED_IDS
    assert results["vsa3_iau_hz2"] == (322500, "")
    assert results["fri_iau"][0] == pytest.approx(4950 / 900)
    assert results["sfri_iau"][0] == pytest.approx(3500 / 900)
    assert results["vai_iau"][0] == pytest.approx(3100 / 2750)
    assert results["fcr_iau"][0] == pytest.approx(2750 / 3100)
    assert results["f2_distance_i_a_hz"] == (1100, "")
    missing = derive_iau_features(vowels.loc[vowels.target_vowel.ne("u")], True)
    assert missing["vsa3_iau_hz2"][1] == "missing_required_vowel"
    assert missing["f2_distance_i_a_hz"] == (1100, "")
    assert derive_iau_features(vowels, False)["vsa3_iau_hz2"][1] == "missing_vowel_category_mapping"
    assert vowel_distance((0, 0), (3, 4)) == 5
    assert vowel_dispersion([(0, 0), (3, 4)]) == 5
