"""Frozen F04 task, token, centroid, and source-lineage contracts."""

import numpy as np
import pandas as pd
import pytest

from vslp.acoustic.features.catalog import (
    load_feature_catalog, requires_alignment_for_task, task_fit,
)
from vslp.acoustic.features.family04 import (
    DISPERSION_IDS, FORMANT_BOUNDS, S3_TARGETS, aggregate_vowel_centroids,
    derive_iau_features, derive_s3_features, derive_trial_features, load_vowel_category_mapping,
    profile_by_id, summarize_formant_token, track_sustained_midpoints,
)
from vslp.acoustic.features.family04_envelope import (
    ENVELOPE_GRID_HZ, ENVELOPE_PARAMETER_SET_ID, ENVELOPE_SOURCE_PARITY,
    envelope_l1_distance,
)
from vslp.acoustic.features.family01_task import task_context
from vslp.acoustic.features.stable_phonation import centered_stable_spans


def test_task_registry_recommendations_and_alignment_exemption():
    outputs = {row["feature_id"]: row for row in load_feature_catalog()["outputs"]}
    expected = {
        "f2_median_hz": {"Sustained /a/", "S5", "Bamboo Passage"},
        "f1_median_hz": {"S5", "Bamboo Passage"},
        "f3_median_hz": {"WSTG"},
        "vsa3_iau_hz2": {"Bamboo Passage"},
        "fri_iau": {"Bamboo Passage"},
        "sfri_iau": {"Bamboo Passage"},
        "vai_iau": {"Bamboo Passage"},
        "fcr_iau": {"Bamboo Passage"},
        "vowel_distance_ai_hz": {"S3"},
        "f2_distance_i_a_hz": {"S3"},
        "vowel_envelope_distance_ai_source": {"S3"},
    }
    for feature, names in expected.items():
        assert {name for name in ("Sustained /a/", "S3", "S5", "Bamboo Passage", "WSTG")
                if task_fit(outputs[feature], name) == "Recommended"} == names
    assert not requires_alignment_for_task(outputs["f2_median_hz"], "Sustained /a/")
    assert requires_alignment_for_task(outputs["f2_median_hz"], "Bamboo Passage")
    assert task_context({"task_id": "sentence_s3"})["stimulus_text"] == \
        "The blue spot is on the key again"
    assert task_context({"task_id": "sentence_s5"})["stimulus_text"] == \
        "We were away a year ago"
    assert len({o["construct_id"] for o in outputs.values() if o["family_id"] == "F04"}) == 12
    assert sum(o["selectable"] for o in outputs.values() if o["family_id"] == "F04") == 21


def test_exact_corner_inventory_and_stress_stripping():
    mapping, reason, version = load_vowel_category_mapping(None, "bamboo_passage")
    assert (mapping, reason, version) == ({"AA": "a", "IY": "i", "UW": "u"},
                                          "", "f04_iau_arpabet_v1")
    assert all(label not in mapping for label in ("AH", "AE", "IH", "UH"))
    assert S3_TARGETS == {"a": ("spot", "AA"), "i": ("key", "IY"),
                          "u": ("blue", "UW")}


def test_trial_specific_centroids_and_all_eight_dispersion_outputs():
    rows = []
    for trial, offset in (("one", 0.), ("two", 100.)):
        for vowel, f1, f2 in (("a", 800., 1200.), ("i", 300., 2300.),
                              ("u", 350., 900.)):
            rows.append({"trial_id": trial, "vowel_category": vowel,
                         "tracking_status": "OK", "f1_token_hz": f1 + offset,
                         "f2_token_hz": f2 + offset, "f3_token_hz": 3000.,
                         "recording_id": "r", "task_id": "bamboo_passage"})
    centroids = aggregate_vowel_centroids(pd.DataFrame(rows))
    assert len(centroids) == 6
    assert centroids.review_flags.eq("low_n_vowel_tokens").all()
    first = derive_iau_features(centroids.loc[centroids.trial_id.eq("one")], True)
    second = derive_iau_features(centroids.loc[centroids.trial_id.eq("two")], True)
    assert first["vsa3_iau_hz2"] == second["vsa3_iau_hz2"]
    assert DISPERSION_IDS <= set(first)
    assert first["vowel_dispersion_ai_hz"][0] == pytest.approx(np.hypot(500, 1100))
    assert first["vowel_dispersion_mean_3corner_hz"][0] == pytest.approx(np.mean([
        first[f"vowel_dispersion_{pair}_hz"][0] for pair in ("ai", "au", "iu")]))
    assert first["vowel_dispersion_global_mean_hz"][0] == pytest.approx(np.mean([
        first[f"vowel_dispersion_{v}_to_global_hz"][0] for v in "aiu"]))
    missing = derive_iau_features(centroids.loc[(centroids.trial_id.eq("one"))
                                                       & centroids.target_vowel.ne("u")], True)
    assert set(reason for _, reason in missing.values()) == {"missing_required_vowel"}


def test_s3_lexical_targets_ignore_other_matching_vowels():
    tokens = pd.DataFrame([
        {"word": word, "phone_normalized": phone, "tracking_status": "OK",
         "f1_token_hz": f1, "f2_token_hz": f2}
        for word, phone, f1, f2 in (("blue", "UW", 350, 900),
                                    ("spot", "AA", 800, 1200),
                                    ("key", "IY", 300, 2300),
                                    ("again", "AA", 900, 1000),
                                    ("spot", "AE", 700, 1800))])
    values = derive_s3_features(tokens)
    assert values["vowel_distance_ai_hz"][0] == pytest.approx(np.hypot(500, 1100))
    assert values["vowel_distance_au_hz"][0] == pytest.approx(np.hypot(450, 300))
    assert values["vowel_distance_iu_hz"][0] == pytest.approx(np.hypot(50, 1400))
    assert values["f2_distance_i_a_hz"] == (1100, "")
    assert "vowel_distance" not in values


def test_s3_duplicate_canonical_target_blocks_only_dependent_outputs():
    tokens = pd.DataFrame([
        {"word": word, "phone_normalized": phone, "tracking_status": "OK",
         "f1_token_hz": f1, "f2_token_hz": f2}
        for word, phone, f1, f2 in (("blue", "UW", 350., 900.),
                                    ("spot", "AA", 800., 1200.),
                                    ("key", "IY", 300., 2300.),
                                    ("key", "IY", 310., 2250.),
                                    ("again", "IY", 400., 1700.))])
    values = derive_s3_features(tokens)
    assert values["vowel_distance_ai_hz"][1] == "AMBIGUOUS_CANONICAL_TARGET"
    assert values["vowel_distance_iu_hz"][1] == "AMBIGUOUS_CANONICAL_TARGET"
    assert values["f2_distance_i_a_hz"][1] == "AMBIGUOUS_CANONICAL_TARGET"
    assert np.isfinite(values["vowel_distance_au_hz"][0])


def test_s5_and_wstg_static_formants_use_all_valid_vowels_within_trial():
    tokens = pd.DataFrame([
        {"word": "we", "phone_normalized": "IY", "tracking_status": "OK",
         "f1_token_hz": 300., "f2_token_hz": 2200., "f3_token_hz": 3000.},
        {"word": "away", "phone_normalized": "EY", "tracking_status": "OK",
         "f1_token_hz": 500., "f2_token_hz": 1800., "f3_token_hz": 3200.},
        {"word": "year", "phone_normalized": "IH", "tracking_status": "TOKEN_TOO_SHORT",
         "f1_token_hz": np.nan, "f2_token_hz": np.nan, "f3_token_hz": np.nan},
    ])
    for task in ("sentence_s5", "wstg"):
        values = derive_trial_features(tokens, pd.DataFrame(), task, True)
        assert values["f1_median_hz"] == (400., "")
        assert values["f2_median_hz"] == (2000., "")
        assert values["f3_median_hz"] == (3100., "")


def test_midpoint_uses_one_burg_evaluation_per_trial(monkeypatch):
    calls = []

    class Formant:
        def get_value_at_time(self, index, time):
            calls.append((index, time))
            return (500, 1500, 3000)[index - 1]

        def get_bandwidth_at_time(self, index, time):
            return (80, 120, 150)[index - 1]

    class Sound:
        def __init__(self, *_args, **_kwargs):
            pass

        def to_formant_burg(self, **settings):
            assert settings == dict(time_step=.005, max_number_of_formants=5,
                                    maximum_formant=5500, window_length=.025,
                                    pre_emphasis_from=50.)
            return Formant()

    monkeypatch.setattr("vslp.acoustic.features.family04.parselmouth.Sound", Sound)
    tokens = track_sustained_midpoints(np.zeros(5 * 44100), 44100,
                                       [("t1", .5, 2.5), ("t2", 3., 4.)],
                                       profile_by_id(), recording_id="r",
                                       file_name="x.wav", task_id="sustained_a")
    assert calls == [(i, time) for time in (1.5, 3.5) for i in (1, 2, 3)]
    assert tokens.f2_token_hz.tolist() == [1500, 1500]
    assert tokens.b3_hz.tolist() == [150, 150]


def test_neutral_sustained_region_is_centered_and_exclusion_free():
    from types import SimpleNamespace

    intervals = pd.DataFrame([
        {"segment_role": "speech", "start_sec": 0., "end_sec": 4.},
        {"segment_role": "manual_exclusion", "start_sec": 2.8, "end_sec": 3.2},
    ])
    row = SimpleNamespace(stable_region_start=.2, stable_region_end=3.8)
    spans = centered_stable_spans(row, intervals, 4., [
        {"trial_id": "t1", "start_sec": 0., "end_sec": 4.}])
    assert spans == [(.5, 2.5, "t1", 0., 4.)]


def test_qc_rejects_out_of_range_without_clipping_and_preserves_bandwidth():
    frames = pd.DataFrame({"time_sec": np.arange(.025, .08, .005),
                           "f1_hz": 500., "f2_hz": 1500., "f3_hz": 3000.,
                           "b1_hz": 80., "b2_hz": 120., "b3_hz": 160.,
                           "valid_ordered": True})
    frames.loc[frames.index[2], "f2_hz"] = 4000.
    result = summarize_formant_token(frames, 0., .1, profile_by_id())
    assert result["f2_token_hz"] == 1500.
    assert result["b3_hz"] == 160.
    assert "out_of_plausibility_range" in result["review_flags"]
    assert FORMANT_BOUNDS["f2"] == (500., 3500.)


def test_envelope_l1_definition_and_source_status():
    flat = np.zeros_like(ENVELOPE_GRID_HZ)
    perturbed = flat.copy()
    perturbed[10:20] = 2.
    assert envelope_l1_distance(flat, flat) == 0.
    assert envelope_l1_distance(flat, perturbed) == pytest.approx(20 / 256)
    assert ENVELOPE_PARAMETER_SET_ID == "family04_s3_allpole_period_window_v1"
    assert ENVELOPE_SOURCE_PARITY == "pending"
