"""Deterministic F06 target, spectral, landmark and aggregation contracts."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from vslp.acoustic.features.catalog import load_feature_catalog, task_fit
from vslp.acoustic.features.family06 import (
    F06_PUBLIC_IDS, band_noise_power, first_spectral_moment_band, resolve_v1_target,
)
from vslp.acoustic.features.family06_subevents import (
    DETECTOR_VERSION, detect_noise_interval, detect_noise_rise, detect_stop_burst,
)
from vslp.acoustic.features.family06_v2_stage import (
    _feature_value, _manual_event, _needed_targets, measure_token,
)


def _store(task: str, prompt: str, word: str, phone: str, *, parent: str | None = None):
    words = pd.DataFrame([{"recording_id": "r", "trial_id": "t", "word_id": "w1",
                           "canonical_word_text": word}])
    phones = pd.DataFrame([{"recording_id": "r", "trial_id": "t", "phone_id": "p1",
                            "parent_word_id": "w1", "parent_word_text": parent or word,
                            "phone_normalized": phone,
                            "phone_index_in_word": 4 if word == "sentenced" else 1,
                            "start_sec": .2, "end_sec": .4}])
    if word == "sentenced":
        prefix = phones.iloc[[0]].copy()
        phones = pd.concat([
            *[prefix.assign(phone_id=f"pre{i}", phone_normalized=label,
                            phone_index_in_word=i)
              for i, label in enumerate(("S", "EH", "N"), 1)],
            phones], ignore_index=True)
    class Store:
        trials = pd.DataFrame([{"recording_id": "r", "trial_id": "t",
                                "task_id": task, "prompt_id": prompt}])

        def words_for_trial(self, _r, _t):
            return words

        def phones_for_trial(self, _r, _t):
            return phones

    return Store()


@pytest.mark.parametrize(("target", "task", "prompt", "word", "phone"), [
    ("see_s", "wstg", "wstg_we_see_three_geese", "see", "S"),
    ("geese_s", "wstg", "wstg_we_see_three_geese", "geese", "S"),
    ("three_th", "wstg", "wstg_we_see_three_geese", "three", "TH"),
    ("spot_t", "sentence_s3", "sentence_s3_blue_spot_key", "spot", "T"),
    ("key_k", "sentence_s3", "sentence_s3_blue_spot_key", "key", "K"),
    ("judge_jh", "sentence_s7", "sentence_s7_municipal_judge", "judge", "JH"),
    ("sentenced_t", "sentence_s7", "sentence_s7_municipal_judge", "sentenced", "T"),
    ("keep_k", "sentence_s9", "sentence_s9_peter_will_keep", "keep", "K"),
    ("at_t", "sentence_s9", "sentence_s9_peter_will_keep", "at", "T"),
])
def test_exact_final_alignment_v1_target(target, task, prompt, word, phone):
    rows, issue = resolve_v1_target(_store(task, prompt, word, phone), "r", "t", target)
    assert issue == "" and rows.phone_id.tolist() == ["p1"]
    assert resolve_v1_target(_store(task, prompt, "other", phone), "r", "t", target)[1] == "target_word_missing"
    assert resolve_v1_target(_store(task, prompt, word, "AA"), "r", "t", target)[1] == "target_phone_missing"
    assert resolve_v1_target(_store(task, prompt, word, phone, parent="other"),
                             "r", "t", target)[1] == "wrong_parent_word"
    assert resolve_v1_target(_store(task, "wrong_prompt", word, phone),
                             "r", "t", target)[1] == "wrong_prompt"


def test_registry_public_ids_match_processing_routes():
    assert len(F06_PUBLIC_IDS) == 16
    for feature in F06_PUBLIC_IDS:
        assert _needed_targets(feature)
    assert _needed_targets("spectral_tilt_k_minus_t_db_per_khz") == ("keep_k", "at_t")
    assert _needed_targets("spectral_m1_t_minus_k_hz") == ("spot_t", "key_k")
    assert _needed_targets("band_noise_a_s_minus_th") == ("see_s", "geese_s", "three_th")


def test_s7_s9_registered_scientific_texts_and_no_s2_substitution():
    prompts = json.loads((Path(__file__).parents[2] / "src" / "vslp" / "acoustic" /
                          "alignment" / "task_prompts.json").read_text(encoding="utf-8"))
    by_task = {item["task_id"]: item["prompts"][0] for item in prompts["tasks"]
               if item["prompts"]}
    assert by_task["sentence_s7"]["exact_expected_text"] == (
        "The municipal judge sentenced the criminal.")
    assert by_task["sentence_s9"]["exact_expected_text"] == "Peter will keep at the peak."
    assert by_task["sentence_s2"]["exact_expected_text"] == "Buy Bobby a puppy."


def test_recommendation_matrix_does_not_substitute_tasks():
    outputs = {item["feature_id"]: item for item in load_feature_catalog()["outputs"]
               if item["family_id"] == "F06"}
    for feature, recommended in (
        ("band_noise_a_see_s", "WSTG"),
        ("spectral_m1_t_minus_k_hz", "S3"),
        ("fricative_noise_duration_th_ms", "WSTG"),
        ("noise_rise_time_judge_jh_ms", "S7"),
        ("spectral_tilt_k_minus_t_db_per_khz", "S9"),
    ):
        assert task_fit(outputs[feature], recommended) == "Recommended"
        assert task_fit(outputs[feature], "S2") == "Not specified"
    assert not outputs["wideband_noise_energy_0_10khz"]["selectable"]


def test_sentenced_first_t_is_selected_and_second_t_never_substitutes():
    store = _store("sentence_s7", "sentence_s7_municipal_judge", "sentenced", "T")
    original = store.phones_for_trial

    def duplicate(recording_id, trial_id):
        frame = original(recording_id, trial_id)
        return pd.concat([frame, frame.loc[frame.phone_id.eq("p1")].assign(
            phone_id="p2", phone_index_in_word=8)],
                         ignore_index=True)

    store.phones_for_trial = duplicate
    rows, issue = resolve_v1_target(store, "r", "t", "sentenced_t")
    assert issue == "" and rows.phone_id.tolist() == ["p1"]
    store.phones_for_trial = lambda r, t: duplicate(r, t).loc[
        lambda frame: frame.phone_id.ne("p1")].copy()
    rows, issue = resolve_v1_target(store, "r", "t", "sentenced_t")
    assert rows.empty and issue == "target_phone_missing"


def test_band_formulas_and_native_bandwidth():
    rate = 48000
    time = np.arange(rate) / rate
    low = .1 * np.sin(2 * np.pi * 1000 * time)
    mid = .1 * np.sin(2 * np.pi * 3000 * time)
    high = .1 * np.sin(2 * np.pi * 8000 * time)
    assert band_noise_power(mid, rate, .2, .3, "a") > 0
    assert band_noise_power(mid, rate, .2, .3, "b") > 0
    assert band_noise_power(high, rate, .2, .3, "a") < band_noise_power(mid, rate, .2, .3, "a")
    assert band_noise_power(low, rate, .2, .3, "b") < band_noise_power(mid, rate, .2, .3, "b")
    assert band_noise_power(mid * 2, rate, .2, .3, "a") == pytest.approx(
        4 * band_noise_power(mid, rate, .2, .3, "a"))
    with pytest.raises(ValueError, match="sample_rate_insufficient"):
        band_noise_power(np.zeros(16000), 16000, .2, .3, "a")
    assert band_noise_power(np.sin(2 * np.pi * 3000 * np.arange(16000) / 16000),
                            16000, .2, .3, "b") > 0


def test_m1_common_band_and_signed_contrasts():
    rate = 48000
    time = np.arange(rate) / rate
    low = np.sin(2 * np.pi * 2000 * time)
    high = np.sin(2 * np.pi * 4000 * time)
    assert first_spectral_moment_band(high, rate, .2) > first_spectral_moment_band(low, rate, .2)
    assert _feature_value("spectral_m1_t_minus_k_hz", {"spot_t": 4000, "key_k": 2000}) == 2000
    assert _feature_value("spectral_tilt_k_minus_t_db_per_khz",
                          {"keep_k": -3, "at_t": -8}) == 5
    assert _feature_value("fricative_noise_duration_s_ms",
                          {"see_s": 100, "geese_s": 140}) == 120
    assert _feature_value("fricative_noise_duration_s_minus_th_ms",
                          {"see_s": 100, "geese_s": 140, "three_th": 80}) == 40
    assert _feature_value("noise_rise_time_t_minus_jh_ms",
                          {"sentenced_t": 20, "judge_jh": 35}) == -15


def test_noise_landmark_and_manual_priority():
    rate = 48000
    rng = np.random.default_rng(42)
    audio = np.zeros(rate)
    audio[round(.25 * rate):round(.35 * rate)] = rng.normal(0, .15, round(.1 * rate))
    noise = detect_noise_interval(audio, rate, .2, .4)
    assert noise.kind == "APERIODIC_NOISE_INTERVAL"
    assert .2 <= noise.start_sec < noise.end_sec <= .4
    assert noise.source == "AUTOMATIC_VSLP_DETECTOR"
    assert DETECTOR_VERSION
    annotation = pd.DataFrame([{"recording_id": "r", "trial_id": "t", "phone_id": "p",
                                "event_kind": "APERIODIC_NOISE_INTERVAL", "start_sec": .26,
                                "end_sec": .34, "approved": True,
                                "time_axis": "original_recording_seconds"}])
    event = _manual_event(annotation, "r", "t", "p", "APERIODIC_NOISE_INTERVAL")
    assert event.source == "VALIDATED_MANUAL_OVERRIDE"
    phone = pd.Series({"start_sec": .2, "end_sec": .4, "phone_id": "p"})
    trial = pd.Series({"start_sec": .1, "end_sec": .5, "trial_id": "t"})
    measured = measure_token(audio, rate, phone, trial, "duration", [], annotation, "r")
    assert measured["value"] == pytest.approx(80.)
    assert measured["event_source"] == "VALIDATED_MANUAL_OVERRIDE"
    with pytest.raises(ValueError, match="target_outside_eligible_audio"):
        measure_token(audio, rate, phone, trial, "duration", [(.3, .32)], annotation, "r")
    with pytest.raises(ValueError, match="target_outside_eligible_audio"):
        measure_token(audio, rate, phone, trial, "duration", [(.21, .22)], annotation, "r")


def test_burst_detector_requires_prominence_and_window_containment():
    rate = 48000
    audio = np.zeros(rate)
    with pytest.raises(ValueError, match="burst_not_detected"):
        detect_stop_burst(audio, rate, .2, .4)
    annotation = pd.DataFrame([{"recording_id": "r", "trial_id": "t", "phone_id": "p",
                                "event_kind": "STOP_BURST", "start_sec": .395,
                                "end_sec": .395, "approved": True,
                                "time_axis": "original_recording_seconds"}])
    phone = pd.Series({"start_sec": .2, "end_sec": .4, "phone_id": "p"})
    trial = pd.Series({"start_sec": .1, "end_sec": .5, "trial_id": "t"})
    with pytest.raises(ValueError, match="burst_window_outside_phone"):
        measure_token(audio, rate, phone, trial, "m1", [], annotation, "r")
    rng = np.random.default_rng(1)
    left, right = round(.27 * rate), round(.29 * rate)
    audio[left:right] = rng.normal(0, .2, right - left)
    event = detect_stop_burst(audio, rate, .2, .4)
    assert event.kind == "STOP_BURST"
    assert .26 <= event.start_sec <= .29


def test_noise_rise_uses_persistent_ten_to_ninety_percent_crossings():
    rate = 48000
    rng = np.random.default_rng(2)
    audio = np.zeros(rate)
    left, right = round(.2 * rate), round(.35 * rate)
    time = np.arange(right - left) / rate
    ramp = np.clip((time - .03) / .06, 0, 1)
    audio[left:right] = ramp * rng.normal(0, .15, right - left)
    event = detect_noise_rise(audio, rate, .2, .4)
    assert event.kind == "NOISE_RISE_INTERVAL"
    assert .23 < event.start_sec < event.end_sec < .35
    assert .02 < event.end_sec - event.start_sec < .09


def test_rise_failure_is_explicit_for_silence():
    with pytest.raises(ValueError, match="rise_not_detected"):
        detect_noise_rise(np.zeros(48000), 48000, .2, .4)
