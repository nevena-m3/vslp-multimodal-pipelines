"""Synthetic, nonclinical checks for the frozen F10 reviewed-event consumer."""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vslp.acoustic.features.catalog import load_feature_catalog, task_fit
from vslp.acoustic.features.family10_v2 import (
    PUBLIC_IDS, RATE_IDS, TARGETS, VARIABILITY_IDS, measure_trial,
)


def _events(target="pa", n=10, *, centers=None, start=0.0, end=2.0,
            trial_id="trial_1", sequence_id="seq_1"):
    if centers is None:
        centers = np.linspace(start + .05, end - .05, n)
    width = .1
    rows = []
    for index, center in enumerate(centers, 1):
        rows.append({"recording_id": "r", "trial_id": trial_id,
                     "sequence_id": sequence_id, "task": "ddk",
                     "ddk_target": target, "ddk_mode": TARGETS[target],
                     "event_id": f"e{index}", "event_index": index,
                     "start_sec": float(center - width / 2),
                     "center_sec": float(center), "end_sec": float(center + width / 2),
                     "valid": True, "review_status": "HUMAN_ACCEPTED",
                     "rejection_reason": "", "sequence_label": "pa-ta-ka",
                     "sequence_complete": True})
    return pd.DataFrame(rows)


def test_eight_task_specific_selectable_leaves_and_recommendations():
    outputs = [x for x in load_feature_catalog()["outputs"] if x["family_id"] == "F10"]
    assert len(outputs) == 8
    assert {x["feature_id"] for x in outputs} == PUBLIC_IDS
    assert all(x["selectable"] for x in outputs)
    assert {x["construct_id"] for x in outputs} == {"C058", "C059"}
    for output in outputs:
        assert task_fit(output, f"/{output['target']}/") == "Recommended"
        assert task_fit(output, "Bamboo Passage") == "Not specified"
        assert output["prerequisites"]["requires_ddk_events"]
        assert not output["prerequisites"]["requires_alignment"]
    assert not {"ddk_rate_syll_s", "ddk_cycle_mad_s"} & PUBLIC_IDS


def test_amr_onset_to_offset_rate_first_ten_and_audit():
    events = _events(centers=np.linspace(.05, 1.95, 10))
    result, audit, summary = measure_trial(events, "pa")
    assert result["rate"] == pytest.approx((5., ""))
    assert result["variability"][0] == pytest.approx(0., abs=1e-12)
    assert summary["n_events_used"] == 10
    assert summary["syllable_equivalent_count"] == 10
    assert audit.loc[audit.selected, "selected_rank"].tolist() == list(range(1, 11))
    assert audit.loc[audit.selected, "event_id"].tolist() == [f"e{i}" for i in range(1, 11)]
    extended = pd.concat([events, _events(n=5, centers=np.arange(2.1, 3.1, .2))],
                         ignore_index=True)
    extended.loc[10:, "event_id"] = [f"late{i}" for i in range(5)]
    extended.loc[10:, "event_index"] = range(11, 16)
    again, audit, summary = measure_trial(extended, "pa")
    assert again == result
    assert summary["n_detected_events"] == 15
    assert audit.selected.sum() == 10


def test_nine_events_unavailable_and_rejected_events_do_not_fill_quota():
    result, _, summary = measure_trial(_events("ta", n=9), "ta")
    assert math.isnan(result["rate"][0])
    assert result["rate"][1] == "insufficient_valid_ddk_events"
    assert result["variability"][1] == "insufficient_valid_ddk_events"
    assert summary["n_valid_events"] == 9
    events = _events()
    events.loc[0, "review_status"] = "PROPOSED"
    result, _, _ = measure_trial(events, "pa")
    assert result["rate"][1] == "insufficient_valid_ddk_events"


def test_pataka_complete_sequence_rate_and_incomplete_skip():
    complete = _events("pataka", centers=np.linspace(.05, 4.95, 10), end=5.0)
    candidate = complete.iloc[[0]].copy()
    candidate["event_id"] = "incomplete"
    candidate["event_index"] = 0
    candidate["center_sec"] = .01
    candidate["start_sec"] = 0
    candidate["end_sec"] = .02
    candidate["sequence_complete"] = False
    result, audit, summary = measure_trial(pd.concat([candidate, complete], ignore_index=True),
                                            "pataka")
    assert result["rate"][0] == pytest.approx(6.)
    assert summary["n_complete_valid_sequences"] == 10
    assert summary["n_sequences_used"] == 10
    assert summary["syllable_equivalent_count"] == 30
    assert not audit.loc[audit.event_id.eq("incomplete"), "selected"].iloc[0]
    assert audit.loc[audit.selected, "selected_rank"].tolist() == list(range(1, 11))
    result, _, _ = measure_trial(complete.iloc[:9], "pataka")
    assert result["rate"][1] == "insufficient_valid_ddk_events"


def test_temporal_variability_is_adjacent_interval_mad_not_regular_positive_score():
    intervals = np.array([.20, .25, .20, .30, .20, .25, .20, .30, .20])
    centers = np.r_[.05, .05 + np.cumsum(intervals)]
    result, _, _ = measure_trial(_events("ka", centers=centers, end=2.5), "ka")
    assert result["variability"][0] == pytest.approx(
        np.mean(np.abs(np.diff(intervals))))
    assert result["variability"][0] > 0


def test_input_order_sorted_by_center_and_ties_rejected():
    events = _events()
    shuffled = events.sample(frac=1, random_state=9)
    result, audit, _ = measure_trial(shuffled, "pa")
    assert result["rate"][1] == ""
    assert audit.loc[audit.selected].sort_values("selected_rank").event_id.tolist() == [
        f"e{i}" for i in range(1, 11)]
    events.loc[1, ["start_sec", "center_sec", "end_sec"]] = events.loc[0, ["start_sec", "center_sec", "end_sec"]]
    result, _, _ = measure_trial(events, "pa")
    assert result["rate"][1] == "invalid_ddk_event_order"


def test_no_cross_sequence_or_cross_trial_pooling():
    events = _events()
    events.loc[5:, "sequence_id"] = "seq_2"
    result, _, _ = measure_trial(events, "pa")
    assert result["rate"][1] == "invalid_ddk_event_order"
    first = _events(trial_id="trial_1", end=2.0)
    second = _events(trial_id="trial_2", start=10.0, end=14.0)
    a, _, _ = measure_trial(first, "pa")
    b, _, _ = measure_trial(second, "pa")
    assert np.median([a["rate"][0], b["rate"][0]]) == pytest.approx(3.75)


def test_wrong_mode_and_invalid_bounds_fail_explicitly():
    events = _events()
    events.loc[:, "ddk_mode"] = "SMR"
    result, _, _ = measure_trial(events, "pa")
    assert result["rate"][1] == "insufficient_valid_ddk_events"
    events = _events()
    events.loc[0, "start_sec"] = 2.0
    result, _, _ = measure_trial(events, "pa")
    assert result["rate"][1] == "invalid_ddk_event_bounds"


def test_missing_event_artifact_is_explicit(tmp_path):
    from vslp.acoustic.features.family10_v2 import load_reviewed_ddk_events

    rows, issue = load_reviewed_ddk_events(tmp_path / "final_ddk_events.csv")
    assert rows.empty and issue == "missing_reviewed_ddk_events"


def test_stage_uses_reviewed_table_and_trial_median_without_detector(tmp_path, monkeypatch):
    from vslp.acoustic.features.family10_v2_stage import run_family10_stage

    root = tmp_path
    final = root / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    decisions.write_text("recording_id,final_decision\nr,KEEP_AUTO\n", encoding="utf-8")
    intervals.write_text("recording_id,segment_role\nr,speech\n", encoding="utf-8")
    (root / "project_manifest.json").write_text(
        '{"task_name":"DDK","task_id":"ddk"}', encoding="utf-8")
    events = pd.concat([_events(end=2.0), _events(start=10, end=14,
                                                    trial_id="trial_2")], ignore_index=True)
    events.to_csv(final / "final_ddk_events.csv", index=False)
    kept = pd.DataFrame([{"recording_id": "r", "file_name": "synthetic.wav"}])
    monkeypatch.setattr("vslp.acoustic.features.family10_v2_stage.load_final_segmentation",
                        lambda *_: kept)
    class Config:
        selected_features = [RATE_IDS["pa"], VARIABILITY_IDS["pa"]]

    registry = pd.DataFrame([{"feature": feature, "unit": unit}
                             for feature, unit in ((RATE_IDS["pa"], "syllables/s"),
                                                   (VARIABILITY_IDS["pa"], "s"))])
    result = run_family10_stage(decisions, root, Config(), intervals, registry,
                                execution_stage_dir=root / "out")
    values = pd.read_csv(result.summary_table)
    assert values[RATE_IDS["pa"]].iloc[0] == pytest.approx(3.75)
    audit = pd.read_csv(root / "out" / "tables" / "native_measurements" /
                        "family10_selected_events.csv")
    assert audit.selected.sum() == 20
    trials = pd.read_csv(root / "out" / "tables" / "native_measurements" /
                         "family10_trial_values.csv")
    assert set(trials.trial_id) == {"trial_1", "trial_2"}


def test_missing_stage_events_produce_nan_reason(tmp_path, monkeypatch):
    from vslp.acoustic.features.family10_v2_stage import run_family10_stage

    root = tmp_path
    final = root / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    decisions.write_text("recording_id,final_decision\nr,KEEP_AUTO\n", encoding="utf-8")
    intervals.write_text("recording_id,segment_role\nr,speech\n", encoding="utf-8")
    (root / "project_manifest.json").write_text(
        '{"task_name":"DDK","task_id":"ddk"}', encoding="utf-8")
    monkeypatch.setattr("vslp.acoustic.features.family10_v2_stage.load_final_segmentation",
                        lambda *_: pd.DataFrame([{"recording_id": "r", "file_name": "r.wav"}]))
    class Config:
        selected_features = [RATE_IDS["pa"]]

    result = run_family10_stage(decisions, root, Config(), intervals,
                                pd.DataFrame([{"feature": RATE_IDS["pa"], "unit": "syllables/s"}]),
                                execution_stage_dir=root / "out")
    status = pd.read_csv(result.summary_table.parent / "acoustic_feature_status_long.csv")
    assert status.failure_reason.iloc[0] == "missing_reviewed_ddk_events"
    assert math.isnan(status.value.iloc[0])


def test_catalog_prerequisite_names_reviewed_events_not_alignment():
    from vslp.acoustic.features.catalog import prerequisite_issues

    output = next(x for x in load_feature_catalog()["outputs"]
                  if x["feature_id"] == RATE_IDS["pa"])
    missing = prerequisite_issues(output, task_name="DDK", final_decisions=None,
                                  final_intervals=None)
    assert missing == ["Final reviewed segmentation is unavailable",
                       "Reviewed DDK events required"]
    ready = prerequisite_issues(output, task_name="DDK", final_decisions=None,
                                final_intervals=None,
                                reviewed_ddk_events_available=True)
    assert ready == ["Final reviewed segmentation is unavailable"]


def test_task_targets_are_never_pooled():
    first = _events("pa")
    other = _events("ta", end=4.0)
    together = pd.concat([first, other], ignore_index=True)
    pa, pa_audit, _ = measure_trial(together, "pa")
    ta, ta_audit, _ = measure_trial(together, "ta")
    assert pa["rate"][0] == pytest.approx(5.0)
    assert ta["rate"][0] == pytest.approx(2.5)
    assert set(pa_audit.ddk_target) == {"pa"}
    assert set(ta_audit.ddk_target) == {"ta"}
