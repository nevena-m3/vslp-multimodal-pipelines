"""Hand-calculated Family 07 timing and alignment behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vslp.acoustic.features.family07 import (
    FAMILY07_IDS, calculate_family07, derive_duration_events,
)
from vslp.acoustic.features.catalog import load_feature_catalog


def _review_events():
    return pd.DataFrame([
        {"event_type": "pause", "event_index": 1, "start_sec": 2.0, "end_sec": 2.4,
         "duration_sec": .4, "source_role": "internal_nonspeech", "boundary_source": "AUTO"},
        {"event_type": "excluded_contamination", "event_index": 1, "start_sec": 3.0,
         "end_sec": 3.5, "duration_sec": .5, "source_role": "manual_exclusion",
         "boundary_source": "MANUAL"},
    ])


def _timing():
    return {"articulation_speech_sec": 5.6, "elapsed_task_sec": 6.0,
            "analysis_start_sec": 1.0, "analysis_end_sec": 8.0,
            "analysis_window_duration_sec": 7.0,
            "excluded_contamination_duration_sec": .5}


def test_exact_ids_and_reviewed_timing_values():
    assert FAMILY07_IDS == {"mean_word_duration_s", "speech_time_s",
                            "task_elapsed_duration_s", "delta_v_s",
                            "varco_v_pct", "npvi_v_pct"}
    catalog = {item["feature_id"]: item for item in load_feature_catalog()["outputs"]}
    assert {feature_id: catalog[feature_id]["unit"] for feature_id in FAMILY07_IDS} == {
        "mean_word_duration_s": "s", "speech_time_s": "s",
        "task_elapsed_duration_s": "s", "delta_v_s": "s",
        "varco_v_pct": "%", "npvi_v_pct": "%",
    }
    assert all(catalog[feature_id]["evidence_level"] for feature_id in FAMILY07_IDS)
    values, _ = calculate_family07(pd.DataFrame(), _timing(), None,
                                   {"speech_time_s", "task_elapsed_duration_s"})
    assert values["speech_time_s"] == (5.6, "")
    assert values["task_elapsed_duration_s"] == (6.0, "")


def test_word_alignment_coverage_and_missing_alignment(tmp_path):
    alignment = pd.DataFrame([
        {"recording_id": "r1", "token_type": "word", "label": f"w{i}",
         "start_sec": i * .5, "end_sec": i * .5 + .25, "expected_count": 5,
         "alignment_source": "test"} for i in range(5)
    ])
    events = derive_duration_events("r1", "x.wav", _review_events(), alignment)
    values, support = calculate_family07(events, _timing(), alignment,
                                         {"mean_word_duration_s"})
    assert values["mean_word_duration_s"] == (.25, "")
    assert support["alignment_coverage"] == 1.0
    values, _ = calculate_family07(pd.DataFrame(columns=events.columns), _timing(), None,
                                   {"mean_word_duration_s"})
    assert np.isnan(values["mean_word_duration_s"][0])
    assert values["mean_word_duration_s"][1] == "missing_alignment"
    loaded = alignment.iloc[:3].copy()
    events = derive_duration_events("r1", "x.wav", _review_events(), loaded)
    values, support = calculate_family07(events, _timing(), loaded,
                                         {"mean_word_duration_s"})
    assert support["alignment_coverage"] == .6
    assert values["mean_word_duration_s"][1] == "low_alignment_coverage"
    values, _ = calculate_family07(events, _timing(), loaded, {"delta_v_s"})
    assert values["delta_v_s"][1] == "missing_phone_alignment"


def test_vowel_variability_ddof_and_contamination_sequence_break():
    alignment = pd.DataFrame([
        {"recording_id": "r1", "token_type": "vowel", "label": "a",
         "start_sec": s, "end_sec": s + d, "alignment_source": "test"}
        for s, d in [(1.0, .1), (1.4, .2), (2.5, .3), (3.2, .2), (3.6, .5)]
    ])
    events = derive_duration_events("r1", "x.wav", _review_events(), alignment)
    # The token overlapping 3.0-3.5 is excluded; the exclusion also separates sequences.
    assert not events.loc[events.exclusion_reason.eq("manual_contamination_overlap")].empty
    selected = {"delta_v_s", "varco_v_pct", "npvi_v_pct"}
    values, support = calculate_family07(events, _timing(), alignment, selected)
    valid = np.array([.1, .2, .3, .5])
    assert np.isclose(values["delta_v_s"][0], np.std(valid, ddof=1))
    assert np.isclose(values["varco_v_pct"][0], 100 * np.std(valid, ddof=1) / valid.mean())
    # Pairs are .1->.2 and .2->.3; .3->.5 is not bridged across contamination.
    assert support["n_adjacent_vowel_pairs"] == 2
    expected = 100 * np.mean([.1 / .15, .1 / .25])
    assert np.isclose(values["npvi_v_pct"][0], expected)


def test_short_vowels_and_insufficient_variability_are_nan():
    alignment = pd.DataFrame([{"recording_id": "r1", "token_type": "vowel",
                               "label": "a", "start_sec": 1.0, "end_sec": 1.02}])
    events = derive_duration_events("r1", "x.wav", _review_events(), alignment)
    values, _ = calculate_family07(events, _timing(), alignment,
                                   {"delta_v_s", "varco_v_pct", "npvi_v_pct"})
    assert all(np.isnan(value) and reason == "insufficient_vowel_tokens"
               for value, reason in values.values())
    review_alignment = pd.DataFrame([{"recording_id": "r1", "token_type": "vowel",
                                      "label": "a", "start_sec": 1.0, "end_sec": 1.04}])
    review_events = derive_duration_events("r1", "x.wav", _review_events(), review_alignment)
    assert review_events.loc[review_events.event_type.eq("vowel"), "review_flag"].iloc[0] == (
        "vowel_duration_below_50ms_review")
