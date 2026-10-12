"""Deterministic nonclinical F07 scientific contract tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vslp.acoustic.features.catalog import load_feature_catalog
from vslp.acoustic.features.family07_v2 import (
    ALIGNMENT_IDS, FAMILY07_IDS, MIN_BAMBOO_VOWELS, S3_EXPECTED_WORD_VOWELS,
    TIMING_IDS, _vowel_coverage, confirmed_trials, reviewed_timing,
    s3_sentence_trials, vowel_tokens, vowel_trial_values,
)


def _intervals():
    return pd.DataFrame([
        {"segment_role": "leading_nonspeech", "start_sec": 0., "end_sec": 1.},
        {"segment_role": "speech", "start_sec": 1., "end_sec": 2.},
        {"segment_role": "internal_nonspeech", "start_sec": 2., "end_sec": 2.5},
        {"segment_role": "speech", "start_sec": 2.5, "end_sec": 3.5},
        {"segment_role": "trailing_nonspeech", "start_sec": 3.5, "end_sec": 4.},
    ])


def _tokens(durations, *, sequence="s1"):
    rows = []
    for index, duration in enumerate(durations, 1):
        rows.append({"phone_index_in_trial": index, "sequence_id": sequence,
                     "word": S3_EXPECTED_WORD_VOWELS[(index - 1) % 9][0],
                     "phone_normalized": S3_EXPECTED_WORD_VOWELS[(index - 1) % 9][1],
                     "duration_sec": duration, "valid": True})
    return pd.DataFrame(rows)


def test_catalog_f07_exact_11_leaf_inventory_and_mixed_prerequisites():
    family = [o for o in load_feature_catalog()["outputs"] if o["family_id"] == "F07"]
    assert len(family) == 12
    assert {o["feature_id"] for o in family if o["selectable"]} == FAMILY07_IDS
    assert len(FAMILY07_IDS) == 11
    assert {o["feature_id"] for o in family if not o["selectable"]} == {
        "mean_word_duration_s"}
    assert {o["feature_id"] for o in family if o["prerequisites"]["requires_alignment"]
            and o["selectable"]} == ALIGNMENT_IDS
    assert FAMILY07_IDS - ALIGNMENT_IDS == TIMING_IDS


def test_bamboo_reviewed_speech_sum_and_effective_envelope_are_distinct():
    result, audit = reviewed_timing(_intervals(), "bamboo_passage")
    assert result["bamboo_speech_duration_s"] == (2., "")
    assert result["bamboo_total_duration_s"] == (2.5, "")
    assert len(audit) == 7
    frame = _intervals()
    frame.loc[1, "end_sec"] = 2.1
    frame.loc[2, "segment_role"] = "manual_exclusion"
    result, pieces = reviewed_timing(frame, "bamboo_passage")
    assert result["bamboo_speech_duration_s"] == (2., "")
    assert result["bamboo_total_duration_s"] == (2.5, "")
    assert pieces.loc[pieces.segment_role.eq("eligible_speech"),
                      "duration_sec"].sum() == pytest.approx(2.)
    edge_exclusion = pd.concat([_intervals(), pd.DataFrame([{
        "segment_role": "manual_exclusion", "start_sec": 1., "end_sec": 1.5}])],
        ignore_index=True)
    edge_result, _ = reviewed_timing(edge_exclusion, "bamboo_passage")
    assert edge_result["bamboo_speech_duration_s"] == (1.5, "")
    assert edge_result["bamboo_total_duration_s"] == (2., "")
    frame.loc[1, "end_sec"] = 2.6
    assert reviewed_timing(frame, "bamboo_passage")[0][
        "bamboo_speech_duration_s"][1] == "invalid_speech_interval"
    empty, _ = reviewed_timing(frame.loc[frame.segment_role.ne("speech")], "bamboo_passage")
    assert empty["bamboo_speech_duration_s"][1] == "no_valid_speech_intervals"


def test_s3_confirmed_current_trial_is_independent_of_final_alignment(tmp_path):
    from vslp.acoustic.alignment.reviewed_trials import confirmed_trial_rows

    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions, intervals = (final / "final_segmentation_decisions.csv",
                            final / "final_segmentation_intervals.csv")
    decisions.write_bytes(b"reviewed decisions")
    intervals.write_bytes(b"reviewed intervals")
    config = tmp_path / "configs"
    config.mkdir()
    rows = confirmed_trial_rows(tmp_path, recording_id="r", task_id="sentence_s3",
                                prompt_id="sentence_s3_blue_spot_key", bounds=[(.5, 2.5)],
                                trial_ids=["t"])
    choices = {"trials": {"r": rows}}
    (config / "alignment_choices.json").write_text(json.dumps(choices), encoding="utf-8")
    trials, issue = confirmed_trials(tmp_path, "r", "sentence_s3", decisions, intervals)
    assert issue == "" and len(trials) == 1
    measured = s3_sentence_trials(trials, _intervals())
    assert measured[0]["duration_sec"] == 2.
    excluded = pd.DataFrame([{"segment_role": "manual_exclusion", "start_sec": 1.,
                              "end_sec": 1.1}])
    assert s3_sentence_trials(trials, excluded)[0]["failure_reason"] == (
        "sentence_contains_hard_exclusion")
    decisions.write_bytes(b"changed")
    assert confirmed_trials(tmp_path, "r", "sentence_s3", decisions, intervals)[1] == (
        "stale_trial_confirmation")


def test_s3_all_nine_vowels_summary_and_sum_based_proportion():
    d = np.array([.05, .06, .07, .08, .09, .10, .11, .12, .12])
    assert d.sum() == pytest.approx(.8)
    tokens = _tokens(d)
    values = vowel_trial_values(tokens, "sentence_s3", 2.)
    assert values["s3_vocalic_duration_mean_s"][0] == pytest.approx(d.mean())
    assert values["s3_vocalic_duration_median_s"][0] == pytest.approx(np.median(d))
    assert values["s3_vocalic_duration_sd_s"][0] == pytest.approx(np.std(d, ddof=1))
    assert values["s3_vocalic_duration_iqr_s"][0] == pytest.approx(
        np.quantile(d, .75) - np.quantile(d, .25))
    assert values["s3_vocalic_proportion_pct"] == pytest.approx((40., ""))
    assert vowel_trial_values(tokens.iloc[:-1], "sentence_s3", 2.)[
        "s3_vocalic_duration_mean_s"][1] == "incomplete_vowel_coverage"
    wrong_word = tokens.copy()
    wrong_word.loc[6, "word"] = "another"
    assert vowel_trial_values(wrong_word, "sentence_s3", 2.)[
        "s3_vocalic_duration_mean_s"][1] == "incomplete_vowel_coverage"
    assert vowel_trial_values(tokens, "sentence_s3", 0.)[
        "s3_vocalic_proportion_pct"][1] == "sentence_duration_nonpositive"
    assert vowel_trial_values(tokens, "sentence_s3", .5)[
        "s3_vocalic_proportion_pct"][1] == "vocalic_duration_exceeds_sentence_duration"


def test_bamboo_trial_first_variability_and_sequence_breaks():
    assert MIN_BAMBOO_VOWELS == 30
    first = _tokens([.1, .2] * 15)
    second = _tokens([.2] * 30)
    a = vowel_trial_values(first, "bamboo_passage")
    b = vowel_trial_values(second, "bamboo_passage")
    assert a["delta_v_s"][0] == pytest.approx(np.std(first.duration_sec, ddof=1))
    assert a["varco_v_pct"][0] == pytest.approx(
        100 * np.std(first.duration_sec, ddof=1) / np.mean(first.duration_sec))
    assert a["npvi_v_pct"][0] == pytest.approx(100 * (2 / 3))
    assert b["npvi_v_pct"][0] == 0.
    assert np.median([a["delta_v_s"][0], b["delta_v_s"][0]]) != pytest.approx(
        np.std(pd.concat([first.duration_sec, second.duration_sec]), ddof=1))
    separate = first.copy()
    separate.loc[15:, "sequence_id"] = "s2"
    assert vowel_trial_values(separate, "bamboo_passage")["npvi_v_pct"][0] == pytest.approx(
        100 * (2 / 3))
    assert _vowel_coverage(first.iloc[:-1], "bamboo_passage")[1] == (
        "insufficient_bamboo_vowels")
    interrupted = _tokens([.1, .2] * 16)
    interrupted.loc[15, "valid"] = False
    interrupted.loc[15, "duration_sec"] = 1.
    result = vowel_trial_values(interrupted, "bamboo_passage")
    assert result["npvi_v_pct"][0] == pytest.approx(100 * (2 / 3))


def test_final_alignment_vowels_stress_and_boundary_failures_are_audited():
    class Store:
        @staticmethod
        def phones_for_trial(_recording_id, _trial_id):
            return pd.DataFrame([
                {"phone_id": "good", "phone_raw": "AA1", "phone_normalized": "AA",
                 "phone_index_in_trial": 1, "sequence_id": "s", "parent_word_id": "w",
                 "parent_word_text": "spot", "start_sec": .2, "end_sec": .24,
                 "alignment_review_status": "HUMAN_ACCEPTED",
                 "boundary_source": "REVIEWED_MANUAL_CORRECTION"},
                {"phone_id": "short", "phone_raw": "IY1", "phone_normalized": "IY",
                 "phone_index_in_trial": 2, "sequence_id": "s", "parent_word_id": "w2",
                 "parent_word_text": "key", "start_sec": .3, "end_sec": .32,
                 "alignment_review_status": "AUTO_ACCEPTED_STRUCTURAL",
                 "boundary_source": "MFA"},
                {"phone_id": "excluded", "phone_raw": "UW0", "phone_normalized": "UW",
                 "phone_index_in_trial": 3, "sequence_id": "s", "parent_word_id": "w3",
                 "parent_word_text": "blue", "start_sec": .6, "end_sec": .7,
                 "alignment_review_status": "HUMAN_ACCEPTED", "boundary_source": "MFA"},
            ])

    trial = pd.Series({"trial_id": "t", "task_id": "sentence_s3", "start_sec": .1,
                       "end_sec": .9})
    audit = vowel_tokens(Store(), "r", trial, [(.1, .9)], [(.65, .75)])
    assert audit.phone_normalized.tolist() == ["AA", "IY", "UW"]
    assert audit.rejection_reason.tolist() == ["", "vowel_too_short",
                                               "vowel_overlaps_exclusion"]
    assert bool(audit.iloc[0].short_vowel_review_flag)
    assert audit.iloc[0].boundary_source == "REVIEWED_MANUAL_CORRECTION"
