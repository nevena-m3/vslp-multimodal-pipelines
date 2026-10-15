"""Nonclinical trial proposals; VAD regions never certify linguistic attempts."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from vslp.acoustic.alignment.task_workflow import project_alignment_context
from vslp.acoustic.alignment.trials import (
    normalized_trial_state, structurally_proposed_trials, trial_manifest_for_alignment,
    trial_proposal_review_queue, validate_trial_choices,
)


def test_legacy_human_confirmation_maps_to_ready_with_human_provenance():
    for old in ("HUMAN_CONFIRMED", "HUMAN_ACCEPTED"):
        trial = normalized_trial_state({"review_status": old, "trial_id": "trial_001"})
        assert trial["trial_status"] == "READY_FOR_ALIGNMENT"
        assert trial["trial_review_source"] == "HUMAN"
        assert "alignment_review_status" not in trial


def test_ambiguous_legacy_auto_acceptance_requires_trial_review():
    trial = normalized_trial_state({"review_status": "AUTO_ACCEPTED_STRUCTURAL"})
    assert trial["trial_status"] == "STALE"
    assert trial["trial_review_source"] == "LEGACY_AMBIGUOUS"


def test_pre_mfa_trial_cannot_carry_post_mfa_acceptance():
    with pytest.raises(ValueError, match="INVALID_PRE_MFA_ALIGNMENT_STATUS"):
        normalized_trial_state({"trial_status": "READY_FOR_ALIGNMENT",
                                "alignment_review_status": "AUTO_ACCEPTED_STRUCTURAL"})


def _review(tmp_path, task_id, speech, exclusions=(), attempt_ids=None):
    root = tmp_path / "synthetic_project"
    root.mkdir()
    (root / "project_manifest.json").write_text(json.dumps({"task_id": task_id}))
    final = root / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    pd.DataFrame([{"recording_id": "r1", "file_name": "synthetic.wav",
                   "final_decision": "KEEP_MANUAL", "analysis_start_sec": 0,
                   "analysis_end_sec": 20}]).to_csv(
                       final / "final_segmentation_decisions.csv", index=False)
    rows = [{"recording_id": "r1", "segment_role": role,
             "start_sec": start, "end_sec": end}
            for role, regions in (("speech", speech),
                                  ("manual_exclusion", exclusions))
            for start, end in regions]
    if attempt_ids is not None:
        for row, attempt_id in zip(rows, attempt_ids):
            row["linguistic_attempt_id"] = attempt_id
    pd.DataFrame(rows).to_csv(
                      final / "final_segmentation_intervals.csv", index=False)
    return root


@pytest.mark.parametrize("count", [2, 3, 4])
def test_wstg_preserves_observed_regions_without_forcing_three(tmp_path, count):
    speech = [(float(i * 3 + 1), float(i * 3 + 2)) for i in range(count)]
    root = _review(tmp_path, "wstg", speech)
    context = project_alignment_context(root)
    choices = structurally_proposed_trials(root, context)
    trials = choices["trials"]["r1"]
    assert [(item["start_sec"], item["end_sec"]) for item in trials] == speech
    assert len(trials) == count
    assert all(item["trial_status"] == "READY_FOR_ALIGNMENT" for item in trials)
    assert all("alignment_review_status" not in item for item in trials)
    assert ("REPETITION_COUNT_MISMATCH" in trials[0]["proposal_flags"]) == (count != 3)
    assert all(item["task_id"] == "wstg" and
               item["prompt_id"] == "wstg_we_see_three_geese" and
               item["prompt_version"] == "v1" for item in trials)
    context["choices"] = choices
    path = trial_manifest_for_alignment(root, context)
    manifest_trials = json.loads(path.read_text())["trials"]
    assert len(manifest_trials) == count
    assert all(item["protocol_expected_repetitions"] == 3 and
               item["observed_candidate_count"] == count and
               item["protocol_count_match"] == (count == 3) for item in manifest_trials)
    assert all(("REPETITION_COUNT_MISMATCH" in item["flags"]) == (count != 3)
               for item in manifest_trials)
    assert all(item["trial_review_source"] in {"AUTO", "HUMAN"} for item in trials)
    assert all("do not establish linguistic completeness" in item["structural_limit"]
               for item in trials)


def test_count_match_does_not_certify_fragmented_sentence_attempts(tmp_path):
    root = _review(tmp_path, "wstg", [(1, 2), (2.3, 3), (5, 6)])
    trials = structurally_proposed_trials(root, project_alignment_context(root))["trials"]["r1"]
    assert len(trials) == 3
    assert all(item["trial_status"] == "READY_FOR_ALIGNMENT" and
               item["trial_review_source"] == "AUTO" and
               "alignment_review_status" not in item for item in trials)
    assert all("do not establish linguistic completeness" in item["structural_limit"]
               for item in trials)


def test_bamboo_internal_pauses_propose_one_structurally_accepted_passage(tmp_path):
    root = _review(tmp_path, "bamboo_passage", [(1, 3), (4, 6), (8, 12)])
    context = project_alignment_context(root)
    trials = structurally_proposed_trials(root, context)["trials"]["r1"]
    assert len(trials) == 1
    assert (trials[0]["start_sec"], trials[0]["end_sec"]) == (1, 12)
    assert trials[0]["proposal_flags"] == []
    assert trials[0]["trial_status"] == "READY_FOR_ALIGNMENT"
    assert trials[0]["prompt_id"] == "bamboo_passage_v1"


def test_bamboo_restart_is_not_silently_labeled_complete(tmp_path):
    root = _review(tmp_path, "bamboo_passage", [(1, 3), (7, 9)],
                   attempt_ids=["attempt_1", "attempt_2"])
    context = project_alignment_context(root)
    trials = structurally_proposed_trials(root, context)["trials"]["r1"]
    assert len(trials) == 1  # Envelope is a visual candidate, not a certified production.
    assert trials[0]["trial_status"] == "NEEDS_TRIAL_REVIEW"
    assert "MULTIPLE_EXPLICIT_ATTEMPTS" in trials[0]["proposal_flags"]
    context["choices"]["trials"] = {"r1": trials}
    with pytest.raises(ValueError, match="TRIAL_CONFIRMATION_REQUIRED"):
        trial_manifest_for_alignment(root, context)


def test_wstg_paused_attempt_can_be_human_merged_without_fabrication(tmp_path):
    root = _review(tmp_path, "wstg", [(1, 2), (2.3, 3)])
    context = project_alignment_context(root)
    choices = structurally_proposed_trials(root, context)
    assert len(choices["trials"]["r1"]) == 2
    choices["trials"]["r1"] = [{"trial_id": "trial_001", "start_sec": 1,
        "end_sec": 3, "prompt_id": "wstg_we_see_three_geese",
        "review_status": "HUMAN_ACCEPTED", "proposal_source": "HUMAN_CONFIRMED"}]
    context["choices"] = choices
    trials = json.loads(trial_manifest_for_alignment(root, context).read_text())["trials"]
    assert len(trials) == 1
    assert (trials[0]["start_sec"], trials[0]["end_sec"]) == (1, 3)
    assert trials[0]["trial_id"] == "trial_001"


def test_bamboo_human_can_split_restart_into_two_actual_attempts(tmp_path):
    root = _review(tmp_path, "bamboo_passage", [(1, 3), (7, 9)])
    context = project_alignment_context(root)
    choices = structurally_proposed_trials(root, context)
    assert len(choices["trials"]["r1"]) == 1
    choices["trials"]["r1"] = [
        {"trial_id": f"trial_{i:03d}", "start_sec": start, "end_sec": end,
         "prompt_id": "bamboo_passage_v1", "review_status": "HUMAN_ACCEPTED",
         "proposal_source": "HUMAN_CONFIRMED"}
        for i, (start, end) in enumerate([(1, 3), (7, 9)], 1)]
    context["choices"] = choices
    trials = json.loads(trial_manifest_for_alignment(root, context).read_text())["trials"]
    assert len(trials) == 2
    assert [item["trial_id"] for item in trials] == ["trial_001", "trial_002"]


def test_passage_exclusion_is_visible_and_cannot_enter_alignment(tmp_path):
    root = _review(tmp_path, "bamboo_passage", [(1, 3), (5, 8)], [(3.5, 4)])
    context = project_alignment_context(root)
    choices = structurally_proposed_trials(root, context)
    assert "TRIAL_CROSSES_EXCLUSION" in choices["trials"]["r1"][0]["proposal_flags"]
    context["choices"] = choices
    with pytest.raises(ValueError, match="TRIAL_CROSSES_EXCLUSION"):
        validate_trial_choices(root, context)


@pytest.mark.parametrize("task_id", ["sustained_a", "ddk"])
def test_nonlinguistic_tasks_receive_no_mfa_trial_proposal(tmp_path, task_id):
    root = _review(tmp_path, task_id, [(1, 3)])
    context = project_alignment_context(root)
    assert not structurally_proposed_trials(root, context).get("trials", {})


def test_trial_review_queue_excludes_structural_auto_passes(tmp_path):
    root = _review(tmp_path, "wstg", [(1, 2), (4, 5), (7, 8)])
    context = project_alignment_context(root)
    context["choices"] = structurally_proposed_trials(root, context)
    assert trial_proposal_review_queue(context) == []
    context["choices"]["trials"]["r1"][1].update(
        trial_status="NEEDS_TRIAL_REVIEW", trial_review_reason=["RESTART_STRUCTURE_AMBIGUOUS"])
    queue = trial_proposal_review_queue(context)
    assert [(item["trial_id"], item["trial_review_reason"]) for item in queue] == [
        ("trial_002", ["RESTART_STRUCTURE_AMBIGUOUS"])]
    context["choices"]["trials"]["r1"][1]["trial_status"] = "OMITTED"
    assert trial_proposal_review_queue(context) == []


def test_auto_accepted_trial_becomes_stale_when_review_changes(tmp_path):
    from vslp.acoustic.alignment.task_workflow import save_project_choices

    root = _review(tmp_path, "bamboo_passage", [(1, 3)])
    context = project_alignment_context(root)
    choices = structurally_proposed_trials(root, context)
    assert choices["trials"]["r1"][0]["trial_status"] == "READY_FOR_ALIGNMENT"
    save_project_choices(root, choices)
    intervals = (root / "acoustic" / "003_segmentation_review" / "final" /
                 "final_segmentation_intervals.csv")
    table = pd.read_csv(intervals)
    table.loc[0, "end_sec"] = 4
    table.to_csv(intervals, index=False)
    reopened = project_alignment_context(root)
    reopened["choices"] = structurally_proposed_trials(root, reopened)
    assert reopened["choices"]["trials"]["r1"][0]["trial_status"] == "STALE"
    assert trial_proposal_review_queue(reopened)[0]["trial_review_reason"] == [
        "UPSTREAM_REVIEW_OR_PROMPT_CHANGED"]
    with pytest.raises(ValueError, match="STALE_TRIAL_PROPOSAL"):
        trial_manifest_for_alignment(root, reopened)


def test_prompt_version_conflict_invalidates_structural_auto_acceptance(tmp_path):
    root = _review(tmp_path, "bamboo_passage", [(1, 3)])
    context = project_alignment_context(root)
    choices = structurally_proposed_trials(root, context)
    choices["trials"]["r1"][0]["prompt_version"] = "stale_version"
    context["choices"] = choices
    structurally_proposed_trials(root, context)
    assert choices["trials"]["r1"][0]["trial_status"] == "STALE"
    assert trial_proposal_review_queue(context)[0]["trial_status"] == "STALE"


def test_bamboo_gui_shows_one_candidate_then_saves_human_confirmation(
        tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    import soundfile as sf
    from PySide6.QtWidgets import QApplication
    from vslp.acoustic.alignment.task_workflow import load_project_choices
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget

    root = _review(tmp_path, "bamboo_passage", [(1, 3), (4, 6), (8, 12)])
    audio = root / "synthetic.wav"
    sf.write(audio, np.zeros(20 * 16000, dtype=np.float32), 16000)
    decisions = root / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_decisions.csv"
    table = pd.read_csv(decisions)
    table["analysis_wav_path"] = str(audio)
    table.to_csv(decisions, index=False)
    app = QApplication.instance() or QApplication([])
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    assert app is not None
    assert widget.recordings.count() == 1
    assert len(widget.proposal_regions) == 1
    assert widget._project_trials_confirmed()
    assert load_project_choices(root)["trials"]["r1"][0][
        "trial_status"] == "READY_FOR_ALIGNMENT"
    assert widget._apply_proposal_bounds([(1, 11)])
    assert not widget._project_trials_confirmed()
    assert load_project_choices(root)["trials"]["r1"][0]["trial_status"] == "STALE"
    widget._confirm_proposals()
    assert widget._project_trials_confirmed()
    saved = load_project_choices(root)["trials"]["r1"][0]
    assert saved["trial_status"] == "READY_FOR_ALIGNMENT"
    assert saved["trial_review_source"] == "HUMAN"
    assert saved["trial_id"] == "trial_001"
    assert saved["prompt_version"] == "v1"
    widget.close()
    reopened = AlignmentWidget(lambda: root)
    reopened.refresh()
    assert reopened._project_trials_confirmed()
    assert reopened.trials.currentData()["trial_id"] == "trial_001"
    reopened.close()
