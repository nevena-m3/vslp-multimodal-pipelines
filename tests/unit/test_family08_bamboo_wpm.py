"""Frozen nominal Bamboo WPM uses the F07 timing authority, not Alignment."""

import json

import numpy as np
import pandas as pd

from vslp.acoustic.alignment.reviewed_trials import confirmed_trial_rows
from vslp.acoustic.features.catalog import load_feature_catalog, task_fit
from vslp.acoustic.features.family07_v2 import reviewed_timing
from vslp.acoustic.features.family08_v2 import (
    FAMILY08_IDS, _trial_timeline, bamboo_trial_rates, frozen_bamboo_prompt, nominal_rates,
)


def _timing(total=60.0, speech=45.0):
    return {"bamboo_total_duration_s": (total, ""),
            "bamboo_speech_duration_s": (speech, "")}


def test_frozen_prompt_and_registry():
    prompt = frozen_bamboo_prompt()
    assert prompt["scientific_word_count"] == 97
    assert prompt["expected_repetitions"] == 1
    catalog = load_feature_catalog()
    outputs = {item["feature_id"]: item for item in catalog["outputs"]
               if item["family_id"] == "F08"}
    assert set(outputs) == FAMILY08_IDS
    assert all(item["unit"] == "words/min" for item in outputs.values())
    assert all(task_fit(item, "Bamboo Passage") == "Recommended" for item in outputs.values())
    assert all(task_fit(item, "WSTG") != "Recommended" for item in outputs.values())
    assert not {"speaking_rate_syll_s", "speaking_rate_words_min",
                "articulation_rate_syll_s"} & set(outputs)


def test_exact_formulas_pause_effect_and_no_pause_limit():
    rates = nominal_rates(_timing())
    assert rates["bamboo_speaking_rate_wpm"] == (97.0, "")
    assert rates["bamboo_articulation_rate_wpm"] == (60 * 97 / 45, "")
    longer = nominal_rates(_timing(total=75))
    assert longer["bamboo_speaking_rate_wpm"][0] < 97
    assert longer["bamboo_articulation_rate_wpm"] == rates["bamboo_articulation_rate_wpm"]
    equal = nominal_rates(_timing(total=45))
    assert equal["bamboo_speaking_rate_wpm"] == equal["bamboo_articulation_rate_wpm"]


def test_leaf_failures_are_isolated_and_explicit():
    missing_speech = nominal_rates({"bamboo_total_duration_s": (60, "")})
    assert missing_speech["bamboo_speaking_rate_wpm"] == (97, "")
    assert np.isnan(missing_speech["bamboo_articulation_rate_wpm"][0])
    assert missing_speech["bamboo_articulation_rate_wpm"][1] == "missing_bamboo_speech_duration"
    missing_total = nominal_rates({"bamboo_speech_duration_s": (45, "")})
    assert missing_total["bamboo_articulation_rate_wpm"][1] == ""
    assert missing_total["bamboo_speaking_rate_wpm"][1] == "missing_bamboo_total_duration"
    inconsistent = nominal_rates(_timing(total=30, speech=45))
    assert all(reason == "speech_duration_exceeds_total"
               for _value, reason in inconsistent.values())
    assert all(np.isnan(value) for value, _reason in inconsistent.values())
    assert nominal_rates(_timing(total=0))["bamboo_speaking_rate_wpm"][1] == "total_duration_nonpositive"


def test_f07_denominators_and_trial_clipping_keep_pauses():
    intervals = pd.DataFrame([
        {"segment_role": "speech", "start_sec": 0, "end_sec": 20},
        {"segment_role": "internal_nonspeech", "start_sec": 20, "end_sec": 35},
        {"segment_role": "speech", "start_sec": 35, "end_sec": 60},
    ])
    first = _trial_timeline(intervals, 0, 30)
    second = _trial_timeline(intervals, 30, 60)
    first_timing, _ = reviewed_timing(first, "bamboo_passage")
    second_timing, _ = reviewed_timing(second, "bamboo_passage")
    assert first_timing["bamboo_total_duration_s"][0] == 20
    assert second_timing["bamboo_total_duration_s"][0] == 25
    assert first_timing["bamboo_speech_duration_s"][0] == 20
    assert second_timing["bamboo_speech_duration_s"][0] == 25
    separate = [nominal_rates(timing)["bamboo_speaking_rate_wpm"][0]
                for timing in (first_timing, second_timing)]
    assert np.median(separate) != 60 * 97 / 60
    complete, _ = reviewed_timing(intervals, "bamboo_passage")
    assert complete["bamboo_total_duration_s"][0] == 60
    assert complete["bamboo_speech_duration_s"][0] == 45


def test_two_confirmed_bamboo_trials_are_independent(tmp_path):
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    (final / "final_segmentation_decisions.csv").write_text("recording_id,final_decision\nr1,KEEP\n")
    (final / "final_segmentation_intervals.csv").write_text("recording_id,segment_role\nr1,speech\n")
    trials = confirmed_trial_rows(tmp_path, recording_id="r1", task_id="bamboo_passage",
                                  prompt_id="bamboo_passage_v1",
                                  bounds=[(0, 60), (100, 175)])
    config = tmp_path / "configs"
    config.mkdir()
    (config / "alignment_choices.json").write_text(
        json.dumps({"trials": {"r1": trials}}), encoding="utf-8")
    intervals = pd.DataFrame([
        {"segment_role": "speech", "start_sec": 0, "end_sec": 45},
        {"segment_role": "internal_nonspeech", "start_sec": 45, "end_sec": 60},
        {"segment_role": "speech", "start_sec": 55, "end_sec": 60},
        {"segment_role": "speech", "start_sec": 100, "end_sec": 150},
        {"segment_role": "internal_nonspeech", "start_sec": 150, "end_sec": 175},
        {"segment_role": "speech", "start_sec": 170, "end_sec": 175},
    ])
    rows = bamboo_trial_rates(tmp_path, "r1", intervals)
    assert [row["trial_id"] for row in rows] == [trial["trial_id"] for trial in trials]
    assert [row["bamboo_total_duration_s"] for row in rows] == [60, 75]
    assert [row["bamboo_speech_duration_s"] for row in rows] == [50, 55]
    assert np.isclose(np.median([row["bamboo_speaking_rate_wpm"] for row in rows]),
                      np.median([60 * 97 / 60, 60 * 97 / 75]))
