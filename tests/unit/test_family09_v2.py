"""Frozen Bamboo F09 pause and phrase event definitions."""

import numpy as np
import pandas as pd
import pytest

from vslp.acoustic.alignment.reviewed_trials import confirmed_trial_rows
from vslp.acoustic.features.catalog import load_feature_catalog, task_fit
from vslp.acoustic.features.family09_v2 import (
    PAUSE_THRESHOLD_SEC, PUBLIC_IDS, reviewed_pause_events,
)
from vslp.acoustic.features.family09_v2_stage import _clip, _trial_domains


def _intervals(gaps, *, speech_sec=1.0):
    rows = []
    start = 1.0
    for gap in gaps:
        rows.append({"view": "authoritative", "segment_role": "speech",
                     "start_sec": start, "end_sec": start + speech_sec})
        start += speech_sec
        rows.append({"view": "authoritative", "segment_role": "internal_nonspeech",
                     "start_sec": start, "end_sec": start + gap})
        start += gap
    rows.append({"view": "authoritative", "segment_role": "speech",
                 "start_sec": start, "end_sec": start + speech_sec})
    return pd.DataFrame(rows)


def _result(gaps):
    return reviewed_pause_events(_intervals(gaps), "r1", "trial_001")


@pytest.mark.parametrize("gap,qualifies", [(.100, False), (.250, False),
                                            (.299, False), (.300, True),
                                            (.301, True), (.600, True)])
def test_exact_threshold_and_phrase_split(gap, qualifies):
    events, values, support = _result([gap])
    assert PAUSE_THRESHOLD_SEC == .300
    assert len(events.loc[events.event_type.eq("PAUSE")]) == int(qualifies)
    assert support["bamboo_phrase_count"] == (2 if qualifies else 1)
    assert values["bamboo_pause_count"] == (float(int(qualifies)), "")


def test_short_gap_retained_within_phrase_and_zero_pause_policy():
    events, values, support = _result([.090, .420])
    phrases = events.loc[events.event_type.eq("PHRASE")]
    assert len(phrases) == 2
    assert np.isclose(phrases.iloc[0].duration_sec, 2.09)
    assert phrases.iloc[0].internal_subthreshold_gap_count == 1
    assert support["bamboo_total_pause_duration_s"] == pytest.approx(.420)
    _, zero, support_zero = _result([.090])
    assert zero["bamboo_pause_count"] == (0.0, "")
    assert zero["bamboo_percent_pause_pct"] == (0.0, "")
    assert support_zero["bamboo_phrase_count"] == 1
    assert zero["bamboo_mean_phrase_duration_s"][1] == ""
    assert zero["bamboo_mean_pause_duration_s"][1] == "no_qualifying_pause"
    assert zero["bamboo_pause_duration_median_s"][1] == "no_qualifying_pause"
    assert zero["bamboo_pause_duration_cv"][1] == "insufficient_pause_events_for_cv"
    assert zero["bamboo_phrase_duration_cv"][1] == "insufficient_phrases_for_cv"


def test_mean_median_and_sample_cv_are_distinct():
    _, values, support = _result([.3, .4, 2.0])
    assert values["bamboo_pause_duration_median_s"][0] == pytest.approx(.4)
    assert values["bamboo_mean_pause_duration_s"][0] == pytest.approx(.9)
    assert values["bamboo_pause_duration_cv"][0] == pytest.approx(
        np.std([.3, .4, 2.0], ddof=1) / .9)
    phrases = [1., 1., 1., 1.]
    assert values["bamboo_phrase_duration_cv"][0] == pytest.approx(
        np.std(phrases, ddof=1) / np.mean(phrases))
    assert support["bamboo_total_pause_duration_s"] == pytest.approx(2.7)


def test_percent_pause_uses_f07_total_duration():
    events, values, support = _result([1., .5])
    assert len(events.loc[events.event_type.eq("PAUSE")]) == 2
    assert support["bamboo_total_duration_s"] == pytest.approx(4.5)
    assert values["bamboo_percent_pause_pct"][0] == pytest.approx(100 * 1.5 / 4.5)


def test_hard_exclusion_is_not_physiological_pause():
    intervals = _intervals([.5])
    intervals.loc[1, "segment_role"] = "manual_exclusion"
    events, values, _ = reviewed_pause_events(intervals, "r1", "trial_001")
    assert events.empty
    assert all(np.isnan(value) and reason == "hard_exclusion_ambiguity"
               for value, reason in values.values())


def test_registry_seven_bamboo_leaves_and_deferred_composite():
    outputs = [item for item in load_feature_catalog()["outputs"] if item["family_id"] == "F09"]
    selectable = {item["feature_id"] for item in outputs if item["selectable"]}
    assert selectable == PUBLIC_IDS
    assert len(selectable) == 7
    assert all(task_fit(item, "Bamboo Passage") == "Recommended"
               for item in outputs if item["selectable"])
    assert all(task_fit(item, "WSTG") != "Recommended"
               for item in outputs if item["selectable"])
    assert all(not item["prerequisites"]["requires_alignment"] for item in outputs)
    deferred = next(item for item in outputs if item["construct_id"] == "C057")
    assert not deferred["selectable"]
    assert deferred["blocked_reason"] == "exact_composite_definition_not_frozen"


def test_confirmed_trials_keep_cv_and_events_separate(tmp_path):
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    (final / "final_segmentation_decisions.csv").write_text("recording_id,final_decision\nr1,KEEP\n")
    (final / "final_segmentation_intervals.csv").write_text("recording_id,segment_role\nr1,speech\n")
    choices = tmp_path / "configs"
    choices.mkdir()
    trials = confirmed_trial_rows(tmp_path, recording_id="r1", task_id="bamboo_passage",
                                  prompt_id="bamboo_passage_v1",
                                  bounds=[(0, 4), (10, 14)])
    import json

    (choices / "alignment_choices.json").write_text(
        json.dumps({"trials": {"r1": trials}}), encoding="utf-8")
    first = _intervals([.3, .6])
    second = _intervals([.4, .5])
    second[["start_sec", "end_sec"]] += 10
    own = pd.concat([first, second], ignore_index=True)
    domains, issue = _trial_domains(tmp_path, "r1", own)
    assert issue == "" and len(domains) == 2
    values = []
    for trial in domains:
        events, result, _ = reviewed_pause_events(
            _clip(own, trial["start_sec"], trial["end_sec"]), "r1", trial["trial_id"])
        assert set(events.trial_id) == {trial["trial_id"]}
        values.append(result["bamboo_pause_duration_cv"][0])
    assert np.isclose(np.median(values), np.median([
        np.std([.3, .6], ddof=1) / .45,
        np.std([.4, .5], ddof=1) / .45]))
