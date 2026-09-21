"""Exact Family 08 formulas, frozen pause rule and prompt provenance."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vslp.acoustic.features.catalog import load_feature_catalog, task_fit
from vslp.acoustic.features.family08 import (
    ALGORITHM_VERSION, FAMILY08_IDS, MIN_PAUSE_SEC, PARAMETER_SET_ID,
    calculate_family08, compute_family08_timing, load_prompt_counts,
)


def _timeline(*, excluded=False):
    rows = [
        ("speech", 1.0, 2.0),
        ("internal_nonspeech", 2.0, 2.4),
        ("speech", 2.4, 2.8 if excluded else 3.4),
    ]
    if excluded:
        rows.extend([("manual_exclusion", 2.8, 3.0), ("speech", 3.0, 3.4)])
    rows.extend([
        ("internal_nonspeech", 3.4, 3.55),
        ("speech", 3.55, 4.55),
    ])
    return pd.DataFrame([{"view": "authoritative", "segment_role": role,
                          "start_sec": start, "end_sec": end}
                         for role, start, end in rows])


def test_exact_ids_evidence_units_and_family_task_scope():
    catalog = load_feature_catalog()
    outputs = {item["feature_id"]: item for item in catalog["outputs"]}
    assert set(outputs) == FAMILY08_IDS
    assert all(item["construct_id"] in {"C048", "C049"} for item in outputs.values())
    assert all(item["evidence_level"] == "STRONG" for item in outputs.values())
    assert outputs["speaking_rate_syll_s"]["evidence_study_count"] == 10
    assert outputs["articulation_rate_syll_s"]["evidence_study_count"] == 4
    assert outputs["speaking_rate_words_min"]["unit"] == "words/min"
    assert outputs["articulation_rate_syll_s"]["unit"] == "syllables/s"
    assert all(task_fit(item, "Bamboo Passage") == "Recommended" for item in outputs.values())
    assert all(task_fit(item, "WSTG") == "Not specified" for item in outputs.values())
    assert ALGORITHM_VERSION == "family08-rate-1.0.0"
    assert PARAMETER_SET_ID == "family08_bamboo_reviewed_v1"
    assert MIN_PAUSE_SEC == 0.3


def test_speaking_includes_pause_articulation_removes_only_ge_300ms():
    timing, issue = compute_family08_timing(_timeline())
    assert issue == ""
    assert np.isclose(timing["elapsed_task_sec"], 3.55)
    assert np.isclose(timing["internal_pauses_ge_300ms_sec"], .4)
    assert np.isclose(timing["articulation_speech_sec"], 3.15)
    prompt = {"syllable_count": 12, "word_count": 6}
    values = calculate_family08(timing, prompt, set(FAMILY08_IDS))
    assert np.isclose(values["speaking_rate_syll_s"][0], 12 / 3.55)
    assert np.isclose(values["speaking_rate_words_min"][0], 60 * 6 / 3.55)
    assert np.isclose(values["articulation_rate_syll_s"][0], 12 / 3.15)
    assert all(reason == "" for _, reason in values.values())


def test_manual_contamination_subtracts_from_both_denominators():
    timing, issue = compute_family08_timing(_timeline(excluded=True))
    assert issue == ""
    assert np.isclose(timing["manual_exclusion_sec"], .2)
    assert np.isclose(timing["elapsed_task_sec"], 3.35)
    assert np.isclose(timing["articulation_speech_sec"], 2.95)
    assert np.isclose(timing["internal_pauses_ge_300ms_sec"], .4)


def test_missing_inputs_are_nan_with_reasons():
    empty = pd.DataFrame(columns=("view", "segment_role", "start_sec", "end_sec"))
    timing, issue = compute_family08_timing(empty)
    assert timing == {} and issue == "no_final_patient_speech"
    values = calculate_family08({}, {"syllable_count": 12}, set(FAMILY08_IDS))
    assert all(np.isnan(value) and reason == "timing_unavailable"
               for value, reason in values.values())
    timing, _ = compute_family08_timing(_timeline())
    values = calculate_family08(timing, {"syllable_count": 12}, set(FAMILY08_IDS))
    assert np.isnan(values["speaking_rate_words_min"][0])
    assert values["speaking_rate_words_min"][1] == "word_count_missing"
    assert np.isfinite(values["speaking_rate_syll_s"][0])


def test_prompt_manifest_must_be_versioned_and_explicit(tmp_path):
    path = tmp_path / "prompt.json"
    prompt = {"schema_version": "1", "task_id": "bamboo_passage",
              "prompt_version": "Bamboo-v1", "count_source": "frozen_prompt_count",
              "syllable_count": 12, "word_count": 6,
              "applies_to_all_recordings": True}
    path.write_text(json.dumps(prompt), encoding="utf-8")
    loaded, issue = load_prompt_counts(path, "Bamboo Passage")
    assert loaded == prompt and issue == ""
    assert load_prompt_counts(path, "WSTG")[1] == "task_not_supported_by_family08_specification"
    prompt["applies_to_all_recordings"] = False
    path.write_text(json.dumps(prompt), encoding="utf-8")
    assert load_prompt_counts(path, "Bamboo Passage")[1] == "prompt_applicability_not_confirmed"
    assert load_prompt_counts(tmp_path / "missing.json", "Bamboo Passage")[1] == "prompt_manifest_missing"


def test_frozen_synthetic_reference_rates():
    fixture = json.loads((Path(__file__).parents[1] / "fixtures" /
                          "family08_rate_reference.json").read_text(encoding="utf-8"))
    timing, issue = compute_family08_timing(pd.DataFrame(fixture["authoritative_intervals"]))
    assert issue == ""
    actual = calculate_family08(timing, fixture["prompt"], set(FAMILY08_IDS))
    for feature_id, expected in fixture["expected"].items():
        assert np.isclose(actual[feature_id][0], expected)
        assert actual[feature_id][1] == ""
