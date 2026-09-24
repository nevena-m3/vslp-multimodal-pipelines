"""Frozen Family 09 pause/phrase formulas and authoritative event semantics."""

import json

import numpy as np
import pandas as pd

from vslp.acoustic.features.catalog import feature_availability, load_feature_catalog, task_fit, task_indicator
from vslp.acoustic.features.family08 import compute_family08_timing
from vslp.acoustic.features.family09 import (
    FAMILY09_IDS, PARAMETER_SET_ID, calculate_family09,
)
from vslp.acoustic.features.reviewed_timing import derive_reviewed_timing


def _timeline(*, gaps=(.3, .6), exclusion=None, leading=.5, trailing=.5):
    rows = []
    cursor = 0.0
    if leading:
        rows.append(("leading_nonspeech", cursor, cursor + leading))
        cursor += leading
    for index in range(len(gaps) + 1):
        rows.append(("speech", cursor, cursor + 1.0))
        cursor += 1.0
        if index < len(gaps):
            gap = gaps[index]
            rows.append(("internal_nonspeech", cursor, cursor + gap))
            cursor += gap
    if trailing:
        rows.append(("trailing_nonspeech", cursor, cursor + trailing))
        cursor += trailing
    if exclusion:
        start, end, reason = exclusion
        updated = []
        for role, a, b in rows:
            if a < start < end < b:
                updated.extend([(role, a, start), ("manual_exclusion", start, end),
                                (role, end, b)])
            else:
                updated.append((role, a, b))
        rows = updated
    return pd.DataFrame([{
        "view": "authoritative", "segment_role": role,
        "start_sec": a, "end_sec": b, "analysis_start_sec": leading,
        "analysis_end_sec": cursor - trailing,
        "manual_exclusion_reason": exclusion[2] if role == "manual_exclusion" else "",
    } for role, a, b in rows])


def _values(timeline):
    events, timing, issue = derive_reviewed_timing(timeline, recording_id="r", file_name="r.wav")
    assert issue == ""
    return events, timing, calculate_family09(events, timing, set(FAMILY09_IDS))


def test_exact_ids_evidence_task_scope_and_frozen_profile():
    catalog = load_feature_catalog()
    leaves = {item["feature_id"]: item for item in catalog["outputs"] if item["family_id"] == "F09"}
    assert set(leaves) == FAMILY09_IDS
    assert {item["construct_id"] for item in leaves.values()} == {
        f"C{number:03d}" for number in range(50, 58)}
    assert len(catalog["families"]) == 13 and len(catalog["constructs"]) == 79
    assert leaves["pause_count"]["unit"] == "count"
    assert leaves["percent_pause_ge300ms"]["unit"] == "%"
    assert leaves["cv_pause_duration"]["evidence_level"] == "NOT ESTABLISHED"
    assert leaves["total_pause_duration_s"]["evidence_study_count"] == 3
    assert all(task_fit(item, "Bamboo Passage") == "Recommended" for item in leaves.values())
    assert all(task_fit(item, "WSTG") == "Not specified" for item in leaves.values())
    assert leaves["pause_pattern_factor_if_frozen"]["selectable"] is False
    assert task_indicator(leaves["pause_pattern_factor_if_frozen"], "Bamboo Passage")[0] == "GREEN"
    assert feature_availability(leaves["pause_pattern_factor_if_frozen"])[0] == "Not implemented"
    assert PARAMETER_SET_ID == "family09_bamboo_reviewed_v1"
    assert leaves["pause_count"]["parameter_profile"]["active_parameters"] == {
        "minimum_internal_pause_ms": 300, "sample_sd_ddof": 1,
        "boundary_source": "frozen reviewed segmentation"}


def test_exact_300ms_and_hand_calculated_pause_phrase_values():
    events, timing, values = _values(_timeline(gaps=(.3, .6)))
    assert events.loc[events.event_type.eq("pause"), "duration_sec"].round(6).tolist() == [.3, .6]
    assert np.isclose(values["total_pause_duration_s"][0], .9)
    assert values["pause_count"] == (2, "")
    assert np.isclose(values["mean_pause_duration_s"][0], .45)
    assert np.isclose(values["percent_pause_ge300ms"][0], 100 * .9 / 3.9)
    expected_cv = np.std([.3, .6], ddof=1) / .45
    assert np.isclose(values["cv_pause_duration"][0], expected_cv)
    assert np.isclose(values["cv_pause_duration_pct"][0], 100 * expected_cv)
    assert np.isclose(values["mean_phrase_duration_s"][0], 1.0)
    assert np.isclose(values["cv_phrase_duration"][0], 0.0)
    assert values["cv_phrase_duration"][1] == ""
    assert timing["n_phrase_events"] == 3
    assert np.isclose(events.loc[events.event_type.eq("pause"), "duration_sec"].sum(),
                      values["total_pause_duration_s"][0])
    family08, issue = compute_family08_timing(_timeline(gaps=(.3, .6)))
    assert issue == "" and family08 == timing


def test_below_threshold_and_no_pause_zero_versus_nan():
    events, timing, values = _values(_timeline(gaps=(.299,)))
    assert events.event_type.eq("internal_gap").sum() == 1
    assert timing["n_pause_events"] == 0
    assert values["total_pause_duration_s"] == (0.0, "")
    assert values["pause_count"] == (0, "")
    assert values["percent_pause_ge300ms"] == (0.0, "")
    assert np.isnan(values["mean_pause_duration_s"][0])
    assert values["mean_pause_duration_s"][1] == "no_pause"
    assert values["cv_pause_duration"][1] == "no_pause"


def test_one_pause_cv_undefined_and_exclusions_not_pause():
    events, timing, values = _values(_timeline(gaps=(.4,), exclusion=(2.1, 2.3, "Cough / throat clear")))
    assert timing["n_pause_events"] == 1
    assert np.isclose(timing["excluded_contamination_duration_sec"], .2)
    assert np.isclose(values["total_pause_duration_s"][0], .4)
    assert values["cv_pause_duration"][1] == "insufficient_pause_events"
    assert events.event_type.eq("excluded_contamination").sum() == 1
    assert events.loc[events.event_type.eq("excluded_contamination"), "qualifies_as_pause"].eq(False).all()
    components = calculate_family09(events, timing, {"pause_pattern_components"})
    assert components["pause_pattern_components"][1] == "insufficient_pause_events_for_composite_components"


def test_other_speaker_or_cough_splits_phrase_without_creating_pause():
    for reason in ("Other speaker", "Cough / throat clear"):
        events, timing, values = _values(_timeline(gaps=(), exclusion=(.9, 1.1, reason)))
        assert timing["n_pause_events"] == 0
        assert timing["n_phrase_events"] == 2
        assert values["cv_phrase_duration"][1] == "phrases_split_only_by_contamination"
        assert not events.loc[events.event_type.eq("excluded_contamination"), "qualifies_as_pause"].any()


def test_research_components_are_explicit_and_factor_requires_frozen_transform():
    _events, _timing, values = _values(_timeline())
    components = json.loads(values["pause_pattern_components"][0])
    assert set(components) == {"mean_pause_duration_s", "sd_pause_duration_s",
                               "cv_pause_duration", "percent_pause_ge300ms"}
    assert np.isnan(values["pause_pattern_factor_if_frozen"][0])
    assert values["pause_pattern_factor_if_frozen"][1] == "frozen_source_transform_unavailable"


def test_missing_frozen_timeline_has_explicit_failure():
    empty = pd.DataFrame(columns=["view", "segment_role", "start_sec", "end_sec"])
    events, timing, issue = derive_reviewed_timing(empty)
    assert events.empty and timing == {} and issue == "no_final_patient_speech"
    values = calculate_family09(events, timing, {"pause_count"})
    assert np.isnan(values["pause_count"][0]) and values["pause_count"][1] == "reviewed_timing_unavailable"
