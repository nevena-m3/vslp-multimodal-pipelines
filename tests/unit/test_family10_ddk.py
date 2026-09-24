"""Family 10 calculations use frozen reviewed DDK events only."""

import numpy as np
import pandas as pd

from vslp.acoustic.features.catalog import load_feature_catalog, task_fit, task_indicator
from vslp.acoustic.features.family10 import (
    ALGORITHM_VERSION, EVENT_COLUMNS, FAMILY10_IDS, MIN_VALID_EVENTS,
    PARAMETER_SET_ID, calculate_family10, derive_ddk_feature_events,
)


def _timeline(times, *, width=.06, exclusion=None, source="MANUAL"):
    rows = []
    for index, center in enumerate(times):
        rows.append({"view": "authoritative", "segment_role": "speech",
                     "start_sec": center - width / 2, "end_sec": center + width / 2,
                     "boundary_source": source, "reviewer": "R",
                     "automatic_source_interval_id": index})
    if exclusion:
        a, b, reason = exclusion
        rows.append({"view": "authoritative", "segment_role": "manual_exclusion",
                     "start_sec": a, "end_sec": b, "boundary_source": source,
                     "reviewer": "R", "manual_exclusion_reason": reason,
                     "automatic_source_interval_id": ""})
    return pd.DataFrame(rows)


def _measure(times, *, window=(0, 1), exclusion=None):
    events, summary, issue = derive_ddk_feature_events(
        _timeline(times, exclusion=exclusion), recording_id="r", file_name="r.wav",
        analysis_start_sec=window[0], analysis_end_sec=window[1])
    assert issue == ""
    return events, summary, calculate_family10(events, summary, set(FAMILY10_IDS))


def test_exact_ids_registry_units_evidence_and_explicit_task():
    leaves = {item["feature_id"]: item for item in load_feature_catalog()["outputs"]
              if item["family_id"] == "F10"}
    assert set(leaves) == FAMILY10_IDS
    assert {item["construct_id"] for item in leaves.values()} == {"C058", "C059"}
    assert leaves["ddk_rate_syll_s"]["unit"] == "syllables/s"
    assert leaves["ddk_cycle_mad_s"]["unit"] == "s"
    assert leaves["ddk_rate_syll_s"]["evidence_level"] == "MODERATE"
    assert leaves["ddk_cycle_mad_s"]["evidence_level"] == "LIMITED"
    assert all(task_fit(item, "DDK") == "Recommended" for item in leaves.values())
    assert all(task_fit(item, "DDK /ta/") == "Not specified" for item in leaves.values())
    assert task_indicator(leaves["ddk_rate_syll_s"], "DDK")[0] == "GREEN"
    assert task_indicator(leaves["ddk_rate_syll_s"], "WSTG")[0] == "GRAY"
    assert ALGORITHM_VERSION == "family10-reviewed-ddk-1.0.0"
    assert PARAMETER_SET_ID == "family10_ddk_reviewed_v1"
    assert MIN_VALID_EVENTS == 5


def test_regular_five_hz_and_slower_trains_with_window_denominator():
    events, summary, values = _measure([.1, .3, .5, .7, .9])
    assert values["ddk_rate_syll_s"] == (5.0, "")
    assert np.isclose(values["ddk_cycle_mad_s"][0], 0)
    assert summary["n_ddk_events"] == 5 and summary["n_cycle_intervals"] == 4
    assert events.loc[events.event_kind.eq("ddk_event"), "next_event_interval_sec"].dropna().round(6).tolist() == [.2] * 4
    _, summary, values = _measure([.25, .75, 1.25, 1.75, 2.25], window=(0, 2.5))
    assert values["ddk_rate_syll_s"] == (2.0, "")
    assert summary["effective_ddk_duration_sec"] == 2.5


def test_irregular_train_uses_adjacent_interval_mad_in_seconds():
    _events, _summary, values = _measure([.1, .3, .55, .75, 1.1], window=(0, 1.2))
    assert np.isclose(values["ddk_cycle_mad_s"][0], (.05 + .05 + .15) / 3)
    assert values["ddk_cycle_mad_s"][1] == ""


def test_manual_delete_add_and_move_change_count_or_timing():
    _events, _summary, baseline = _measure([.1, .3, .5, .7, .9])
    _events, _summary, deleted = _measure([.1, .3, .5, .9])
    assert np.isnan(deleted["ddk_rate_syll_s"][0])
    assert deleted["ddk_rate_syll_s"][1] == "insufficient_ddk_events"
    _events, _summary, added = _measure([.1, .2, .3, .5, .7, .9])
    assert added["ddk_rate_syll_s"] == (6.0, "")
    _events, _summary, moved = _measure([.1, .3, .55, .7, .9])
    assert moved["ddk_cycle_mad_s"][0] > baseline["ddk_cycle_mad_s"][0]


def test_review_window_and_cough_subtract_duration_and_split_sequences():
    times = [.1, .3, .5, .7, .9]
    events, summary, values = _measure(
        times, exclusion=(.57, .63, "Cough / throat clear"))
    assert summary["raw_analysis_duration_sec"] == 1
    assert np.isclose(summary["excluded_contamination_duration_sec"], .06)
    assert np.isclose(summary["effective_ddk_duration_sec"], .94)
    assert np.isclose(values["ddk_rate_syll_s"][0], 5 / .94)
    assert summary["n_valid_sequences"] == 2
    assert summary["n_cycle_intervals"] == 3
    assert events.loc[events.event_kind.eq("manual_exclusion"), "exclusion_reason"].tolist() == ["Cough / throat clear"]
    assert events.loc[events.event_kind.eq("ddk_event") &
                      events.event_time_sec.eq(.5), "next_event_interval_sec"].isna().all()
    _, summary, trimmed = _measure(times, window=(.05, .95))
    assert np.isclose(summary["effective_ddk_duration_sec"], .9)
    assert np.isclose(trimmed["ddk_rate_syll_s"][0], 5 / .9)


def test_other_speaker_break_and_no_cross_contamination_cycle():
    events, summary, values = _measure(
        [.1, .25, .4, .7, .85], exclusion=(.48, .6, "Other speaker"))
    assert summary["n_valid_sequences"] == 2
    assert summary["n_cycle_intervals"] == 3
    assert np.isclose(values["ddk_cycle_mad_s"][0], 0)
    assert not events.next_event_interval_sec.dropna().ge(.3).any()


def test_insufficient_cycle_intervals_and_invalid_or_missing_final():
    two_sequences = _timeline([.1, .3, .5, .7, .9], exclusion=(.6, .65, "Other speaker"))
    events, summary, issue = derive_ddk_feature_events(
        two_sequences, analysis_start_sec=0, analysis_end_sec=1)
    assert issue == ""
    assert calculate_family10(events, summary, {"ddk_cycle_mad_s"})["ddk_cycle_mad_s"][1] == ""
    three_sequences = _timeline([.1, .25, .4, .55, .7, .85])
    for start, end in ((.31, .34), (.61, .64)):
        three_sequences.loc[len(three_sequences)] = {
            "view": "authoritative", "segment_role": "manual_exclusion",
            "start_sec": start, "end_sec": end, "boundary_source": "MANUAL",
            "reviewer": "R", "manual_exclusion_reason": "Other speaker",
            "automatic_source_interval_id": ""}
    events, summary, issue = derive_ddk_feature_events(
        three_sequences, analysis_start_sec=0, analysis_end_sec=1)
    assert issue == ""
    assert summary["n_ddk_events"] == 6
    assert calculate_family10(events, summary, {"ddk_cycle_mad_s"})["ddk_cycle_mad_s"][1] == "insufficient_cycle_intervals"
    empty = pd.DataFrame(columns=EVENT_COLUMNS)
    assert np.isnan(calculate_family10(empty, {}, {"ddk_rate_syll_s"})["ddk_rate_syll_s"][0])
    assert derive_ddk_feature_events(_timeline([.1]), analysis_start_sec=1,
                                    analysis_end_sec=1)[2] == "invalid_ddk_duration"


def test_event_audit_columns_and_manual_provenance():
    events, summary, _values = _measure([.1, .3, .5, .7, .9])
    assert tuple(events.columns) == EVENT_COLUMNS
    assert events.boundary_source.eq("MANUAL").all()
    assert events.reviewer.eq("R").all()
    assert events.recording_id.eq("r").all()
    assert summary["n_valid_sequences"] == 1


def test_contamination_splitting_single_source_event_does_not_create_two_cycles():
    timeline = _timeline([.1, .3, .5, .7, .9])
    timeline = timeline.loc[~timeline.start_sec.between(.47, .53)].copy()
    fragments = pd.DataFrame([
        {"view": "authoritative", "segment_role": "speech", "start_sec": .47,
         "end_sec": .49, "boundary_source": "AUTO_MODIFIED", "reviewer": "R",
         "automatic_source_interval_id": 2},
        {"view": "authoritative", "segment_role": "manual_exclusion", "start_sec": .49,
         "end_sec": .51, "boundary_source": "AUTO_MODIFIED", "reviewer": "R",
         "manual_exclusion_reason": "Cough / throat clear",
         "automatic_source_interval_id": ""},
        {"view": "authoritative", "segment_role": "speech", "start_sec": .51,
         "end_sec": .53, "boundary_source": "AUTO_MODIFIED", "reviewer": "R",
         "automatic_source_interval_id": 2},
    ])
    timeline = pd.concat([timeline, fragments], ignore_index=True)
    events, summary, issue = derive_ddk_feature_events(
        timeline, analysis_start_sec=0, analysis_end_sec=1)
    assert issue == ""
    assert summary["n_ddk_events"] == 4
    assert events.loc[events.exclusion_reason.eq("event_fragmented_by_exclusion")].shape[0] == 2
    assert calculate_family10(events, summary, {"ddk_rate_syll_s"})["ddk_rate_syll_s"][1] == "insufficient_ddk_events"
