"""Task identity, recommendation and original-time F01 region regression tests."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from vslp.acoustic.features.catalog import load_feature_catalog, task_fit
from vslp.acoustic.features.family01 import FAMILY01_IDS, F0Track, calculate_f0_features
from vslp.acoustic.features.family01_stage import (
    _approved_sentence_intonation, _f0_qc, _region_tracks,
)
from vslp.acoustic.features.family01_task import task_context
from vslp.acoustic.features.stage import FeatureExtractionConfig, _select_registry
from vslp.acoustic.run_setup import initialize_acoustic_run


def _audio(sr=16000):
    time = np.arange(sr * 2) / sr
    return .2 * np.sin(2 * np.pi * 180 * time)


def test_recommendation_is_separate_from_executability():
    catalog = load_feature_catalog()
    outputs = {item["feature_id"]: item for item in catalog["outputs"]}
    sustained = "Sustained /a/"
    sentence = "S5"
    for feature_id in FAMILY01_IDS:
        assert outputs[feature_id]["executable_for_task"] is True
        assert task_fit(outputs[feature_id], "DDK", "DDK") != "Recommended"
    for feature_id in FAMILY01_IDS - {"pfr_maxmin_st", "intonation_f0_sd_st",
                                      "intonation_f0_iqr_st"}:
        assert task_fit(outputs[feature_id], sentence,
                        "Sentence / controlled speech") == "Recommended"
        assert task_fit(outputs[feature_id], "Bamboo Passage",
                        "Passage / connected speech") == "Recommended"
    assert task_fit(outputs["pfr_maxmin_st"], sentence,
                    "Sentence / controlled speech") != "Recommended"
    for alias, primitive in (("intonation_f0_sd_st", "f0_sd_st"),
                             ("intonation_f0_iqr_st", "f0_iqr_st")):
        assert outputs[alias]["deprecated_alias_of"] == primitive
        assert task_fit(outputs[alias], sentence,
                        "Sentence / controlled speech") != "Recommended"
    for feature_id in ("f0_mean_hz", "f0_median_hz", "f0_sd_st", "f0_iqr_st",
                       "f0_range_st"):
        assert task_fit(outputs[feature_id], sustained, "Sustained phonation") == "Recommended"
    for feature_id in ("pfr_maxmin_st", "observed_f0_range_p05_p95_st",
                       "intonation_f0_sd_st", "intonation_f0_iqr_st"):
        assert task_fit(outputs[feature_id], sustained, "Sustained phonation") != "Recommended"
    assert task_fit(outputs["f0_mean_hz"], "WSTG",
                    "Sentence / controlled speech") != "Recommended"
    assert outputs["observed_f0_range_p05_p95_st"]["executable_for_task"]
    assert "preferred" in outputs["observed_f0_range_p05_p95_st"]["human_name"].lower()
    assert not outputs["intonation_factor_score_if_frozen"]["selectable"]
    assert not outputs["intonation_factor_score_if_frozen"]["executable_for_task"]
    assert task_fit(outputs["intonation_factor_score_if_frozen"], "S5",
                    "Sentence / controlled speech") != "Recommended"
    assert "not an elicited physiologic" in outputs["pfr_maxmin_st"]["interpretation_note"]
    snapshot = _select_registry(FeatureExtractionConfig(selected_features=["pfr_maxmin_st"]))
    assert bool(snapshot.executable_for_task.iloc[0])
    assert '"sustained_vowel": false' in snapshot.recommended_for_task.iloc[0]
    constructs = {item["construct_id"]: item for item in catalog["constructs"]}
    assert task_fit(constructs["C001"], sustained, "Sustained phonation") == "Recommended"
    assert task_fit(constructs["C003"], sustained, "Sustained phonation") != "Recommended"
    assert task_fit(constructs["C004"], sentence,
                    "Sentence / controlled speech") == "Recommended"
    assert constructs["C004"]["qc_range_text"] == "—"
    assert constructs["C004"]["unit"] == "source-specific factor score"
    assert "not an unweighted" in constructs["C004"]["factor_limitation"]


def test_explicit_task_identity_and_s5_text():
    s5 = task_context({"task_id": "sentence_s5", "task_name": "S5",
                       "task_type": "Sentence / controlled speech"})
    assert (s5["task_family"], s5["task_item"], s5["stimulus_text"]) == (
        "sentence", "S5", "We were away a year ago")
    vowel = task_context({"task_id": "sustained_i", "task_name": "Sustained /i/",
                          "task_type": "Sustained phonation"})
    assert (vowel["task_family"], vowel["task_item"]) == ("sustained_vowel", "i")
    ddk = task_context({"task_id": "ddk", "task_name": "DDK", "task_type": "DDK",
                        "task_item": "pataka"})
    assert (ddk["task_family"], ddk["task_item"]) == ("ddk", "pataka")


def test_setup_persists_selected_s5_identity_for_feature_stage(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    run = initialize_acoustic_run(project_name="synthetic", task_name="S5",
                                  task_type="Sentence / controlled speech",
                                  input_folder=source, output_parent=tmp_path / "output",
                                  setup_values={})
    manifest = json.loads((run.root / "project_manifest.json").read_text(encoding="utf-8"))
    assert (manifest["task_family"], manifest["task_item"]) == ("sentence", "S5")
    assert task_context(manifest)["stimulus_text"] == "We were away a year ago"


def test_sustained_stable_region_intersects_speech_exclusion_and_trials():
    sr = 16000
    intervals = pd.DataFrame([
        {"segment_role": "speech", "start_sec": .2, "end_sec": 1.8},
        {"segment_role": "manual_exclusion", "start_sec": .9, "end_sec": 1.1},
    ])
    trials = [{"trial_id": "one", "start_sec": .2, "end_sec": 1.0},
              {"trial_id": "two", "start_sec": 1.0, "end_sec": 1.8}]
    row = SimpleNamespace(stable_region_start=.4, stable_region_end=1.6)
    track, sustained, spans = _region_tracks(row, intervals, _audio(sr), sr,
                                              "sustained_vowel", trials)
    assert sustained and [identity for _, identity, *_ in spans] == ["one", "two"]
    assert track.time_sec.min() >= .4 and track.time_sec.max() <= 1.6
    assert not np.any((track.time_sec >= .9) & (track.time_sec < 1.1))
    assert abs(calculate_f0_features(track, sustained=True)["f0_mean_hz"][0] - 180) < 2


def test_bamboo_sentence_and_ddk_use_separate_reviewed_voiced_spans():
    sr = 16000
    intervals = pd.DataFrame([
        {"segment_role": "speech", "start_sec": .1, "end_sec": .7},
        {"segment_role": "speech", "start_sec": 1.0, "end_sec": 1.9},
        {"segment_role": "manual_exclusion", "start_sec": 1.3, "end_sec": 1.5},
    ])
    row = SimpleNamespace(stable_region_start=np.nan, stable_region_end=np.nan)
    for family, region in (("bamboo_passage", "reviewed_passage_voiced_frames"),
                           ("sentence", "reviewed_sentence_voiced_frames"),
                           ("ddk", "reviewed_ddk_voiced_frames")):
        track, sustained, spans = _region_tracks(row, intervals, _audio(sr), sr,
                                                  family, [])
        assert not sustained and track.analysis_region == region
        assert len(spans) == 3
        assert not np.any((track.time_sec >= .7) & (track.time_sec < 1.0))
        assert not np.any((track.time_sec >= 1.3) & (track.time_sec < 1.5))
        assert calculate_f0_features(track, sustained=False)["f0_mean_hz"][1] == ""


def test_technical_octave_flags_do_not_repair_source_f0():
    hz = np.array([100., 200., 100., 100.])
    track = F0Track(np.arange(4) * .005, hz.copy(), np.ones(4, dtype=bool),
                    16000, 0., .02, "reviewed_sentence_voiced_frames")
    flags = _f0_qc([(track, "S5", 0., .02)])
    assert flags["possible_octave_jump"]
    assert flags["possible_pitch_halving"] and flags["possible_pitch_doubling"]
    assert np.array_equal(track.f0_hz, hz)


def test_centered_two_second_sustained_region_and_shorter_trial():
    sr = 16000
    audio = np.tile(_audio(sr), 3)
    intervals = pd.DataFrame([
        {"segment_role": "speech", "start_sec": 0., "end_sec": 6.},
    ])
    row = SimpleNamespace(stable_region_start=.5, stable_region_end=5.5)
    _, _, spans = _region_tracks(row, intervals, audio, sr, "sustained_vowel", [])
    assert len(spans) == 1
    assert np.isclose(spans[0][0].analysis_start_sec, 2.)
    assert np.isclose(spans[0][0].analysis_end_sec, 4.)
    trials = [{"trial_id": "short", "start_sec": .5, "end_sec": 1.5,
               "stable_region_start": .6, "stable_region_end": 1.4},
              {"trial_id": "long", "start_sec": 2., "end_sec": 5.5,
               "stable_region_start": 2.2, "stable_region_end": 5.2}]
    _, _, spans = _region_tracks(row, intervals, audio, sr, "sustained_vowel", trials)
    by_id = {item[1]: item[0] for item in spans}
    assert np.isclose(by_id["short"].analysis_start_sec, .6)
    assert np.isclose(by_id["short"].analysis_end_sec, 1.4)
    assert np.isclose(by_id["long"].analysis_start_sec, 2.7)
    assert np.isclose(by_id["long"].analysis_end_sec, 4.7)


def test_bamboo_intonation_uses_only_explicit_approved_sentence_intervals():
    times = np.arange(80) * .005
    hz = np.r_[np.tile([100., 150.], 20), np.tile([120., 180.], 20)]
    track = F0Track(times, hz, np.ones(80, dtype=bool), 16000, 0., .4,
                    "reviewed_passage_voiced_frames")
    assert _approved_sentence_intonation(track, pd.DataFrame()) is None
    annotated = pd.DataFrame([
        {"sentence_id": "one", "sentence_review_status": "APPROVED",
         "segment_role": "speech", "start_sec": 0., "end_sec": .2},
        {"sentence_id": "two", "sentence_review_status": "APPROVED",
         "segment_role": "speech", "start_sec": .2, "end_sec": .4},
    ])
    rows = []
    result = _approved_sentence_intonation(track, annotated, rows)
    assert len(rows) == 2
    assert all(item["construct_role"] == "intonation_component" for item in rows)
    assert [item["sentence_id"] for item in rows] == ["one", "two"]
    first = calculate_f0_features(F0Track(times[:40], hz[:40],
                                  np.ones(40, dtype=bool), 16000, 0., .2, "sentence"),
                                  sustained=False)
    second = calculate_f0_features(F0Track(times[40:], hz[40:],
                                   np.ones(40, dtype=bool), 16000, .2, .4, "sentence"),
                                   sustained=False)
    for key in ("f0_sd_st", "f0_iqr_st"):
        assert np.isclose(result[key], np.median([first[key][0], second[key][0]]))
    annotated.loc[0, "sentence_review_status"] = "UNREVIEWED"
    assert _approved_sentence_intonation(track, annotated) is None
