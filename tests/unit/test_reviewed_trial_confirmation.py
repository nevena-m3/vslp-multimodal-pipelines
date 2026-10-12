"""Contract tests for human trial confirmation without invoking MFA."""

from __future__ import annotations

import json

from vslp.acoustic.alignment.reviewed_trials import (
    CONTRACT_VERSION, confirmed_trial_rows, load_confirmed_trials,
    segmentation_review_hash,
)


def _project(tmp_path):
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    (final / "final_segmentation_decisions.csv").write_text("recording_id,final_decision\nr,KEEP_MANUAL\n")
    (final / "final_segmentation_intervals.csv").write_text("recording_id,start_sec,end_sec\nr,1,2\n")
    return tmp_path


def _write(root, rows):
    config = root / "configs"
    config.mkdir(exist_ok=True)
    (config / "alignment_choices.json").write_text(json.dumps({"trials": {"r": rows}}))


def test_confirmation_round_trip_and_bounds_unchanged(tmp_path):
    root = _project(tmp_path)
    bounds = [(1.25, 2.75), (3.0, 4.0)]
    rows = confirmed_trial_rows(root, recording_id="r", task_id="sentence_s3",
                                prompt_id="sentence_s3_blue_spot_key", bounds=bounds)
    assert [(row["start_sec"], row["end_sec"]) for row in rows] == bounds
    assert all(row["trial_status"] == "READY_FOR_ALIGNMENT" for row in rows)
    assert all(row["trial_review_source"] == "HUMAN" for row in rows)
    assert all(row["segmentation_review_hash"] == segmentation_review_hash(root) for row in rows)
    assert all(row["reviewed_trial_contract_version"] == CONTRACT_VERSION for row in rows)
    assert len({row["sequence_id"] for row in rows}) == 2
    _write(root, rows)
    loaded, issue = load_confirmed_trials(root, "r", "sentence_s3")
    assert issue == "" and [(row["start_sec"], row["end_sec"]) for row in loaded] == bounds


def test_stale_missing_and_legacy_provenance_are_rejected(tmp_path):
    root = _project(tmp_path)
    row = confirmed_trial_rows(root, recording_id="r", task_id="sentence_s3",
                               prompt_id="sentence_s3_blue_spot_key", bounds=[(1., 2.)])[0]
    for field, expected in (("segmentation_review_hash", "legacy_unverified_trial_confirmation"),
                            ("trial_review_source", "legacy_unverified_trial_confirmation"),
                            ("trial_status", "legacy_unverified_trial_confirmation")):
        old = dict(row)
        old.pop(field)
        _write(root, [old])
        assert load_confirmed_trials(root, "r", "sentence_s3")[1] == expected
    _write(root, [{"trial_id": "trial_001", "start_sec": 1., "end_sec": 2.}])
    assert load_confirmed_trials(root, "r", "sentence_s3")[1] == "legacy_unverified_trial_confirmation"
    _write(root, [row])
    final = root / "acoustic" / "003_segmentation_review" / "final"
    (final / "final_segmentation_intervals.csv").write_text("changed\n")
    assert load_confirmed_trials(root, "r", "sentence_s3")[1] == "stale_trial_confirmation"


def test_wrong_source_or_status_is_rejected(tmp_path):
    root = _project(tmp_path)
    row = confirmed_trial_rows(root, recording_id="r", task_id="sentence_s3",
                               prompt_id="sentence_s3_blue_spot_key", bounds=[(1., 2.)])[0]
    for field, value, expected in (("trial_review_source", "AUTO", "trial_not_human_confirmed"),
                                   ("trial_status", "STALE", "trial_not_ready_for_alignment")):
        _write(root, [{**row, field: value}])
        assert load_confirmed_trials(root, "r", "sentence_s3")[1] == expected
