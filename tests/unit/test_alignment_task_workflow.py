"""Explicit task, prompt, and speaker resolution without filename inference."""

from __future__ import annotations

import json
import hashlib

import pandas as pd
import pytest

from vslp.acoustic.alignment.stage import PromptManifest
from vslp.acoustic.alignment.task_workflow import (
    build_task_alignment_config, load_project_choices, project_alignment_context,
    resolve_prompt, save_project_choices, task_entry,
)


def _project(tmp_path, task="wstg", metadata_speaker="speaker_A"):
    root = tmp_path / "project"
    root.mkdir()
    (root / "project_manifest.json").write_text(json.dumps({
        "task_id": task, "task_name": (
            "WSTG" if task == "wstg" else "Bamboo Passage" if task == "bamboo_passage" else task)}),
        encoding="utf-8")
    review = root / "acoustic" / "003_segmentation_review" / "final"
    review.mkdir(parents=True)
    pd.DataFrame([{"recording_id": "r1", "file_name": "opaque.wav",
                   "final_decision": "KEEP_AUTO"}]).to_csv(
                       review / "final_segmentation_decisions.csv", index=False)
    if metadata_speaker:
        index = root / "acoustic" / "000_metadata" / "tables"
        index.mkdir(parents=True)
        pd.DataFrame([{"recording_id": "r1", "subject_id": metadata_speaker,
                       "metadata_source": "metadata_csv",
                       "subject_id_source": "metadata_csv"}]).to_csv(
                           index / "project_file_index.csv", index=False)
    return root


def test_wstg_prompt_and_explicit_speaker_are_resolved(tmp_path):
    root = _project(tmp_path)
    context = project_alignment_context(root)
    prompt, issue = resolve_prompt(context)
    assert issue == ""
    assert prompt["exact_expected_text"] == "We see three geese."
    assert prompt["prompt_id"] == "wstg_we_see_three_geese"
    assert context["records"][0]["speaker_id"] == "speaker_A"
    config = build_task_alignment_config(root)
    loaded = PromptManifest.load(config.prompt_manifest_path)
    assert loaded.transcript == "We see three geese."
    assert config.recording_prompt_manifest_paths["r1"] == config.prompt_manifest_path
    assert config.speaker_manifest_path.endswith("speakers.json")


def test_no_filename_speaker_or_prompt_inference(tmp_path):
    root = _project(tmp_path, metadata_speaker="")
    context = project_alignment_context(root)
    assert context["records"][0]["speaker_id"] == ""
    with pytest.raises(ValueError, match="SPEAKER_ID_REQUIRED"):
        build_task_alignment_config(root)


def test_bamboo_canonical_prompt_resolves_and_survives_project_reopen(tmp_path):
    root = _project(tmp_path, task="bamboo_passage")
    expected = ("Bamboo walls are getting to be very popular. They are strong, easy to use, "
        "and good-looking. They provide a good background and can create a look of a Japanese "
        "garden. Bamboo is one of the largest and most rapidly growing grasses all over the "
        "world. Many varieties of bamboo are grown in Asia, although it is also grown in "
        "America. Last year we bought a new home and have been working on the flower garden. "
        "In a few more days, we will be done with the bamboo wall in our garden. We have "
        "really enjoyed the project.")
    prompt, issue = resolve_prompt(project_alignment_context(root))
    assert issue == "" and prompt["exact_expected_text"] == expected
    assert prompt["prompt_id"] == "bamboo_passage_v1"
    assert prompt["prompt_version"] == "v1"
    assert hashlib.sha256(expected.encode("utf-8")).hexdigest() == (
        "a132fa58fe945d86812dec3cce3936f6f586a24527105f3bc9009f18efdb5f67")
    config = build_task_alignment_config(root)
    assert PromptManifest.load(config.prompt_manifest_path).transcript == expected
    assert PromptManifest.load(config.prompt_manifest_path).prompt_version == "v1"
    choices = load_project_choices(root)
    choices["recordings"] = {"r1": {"prompt_id": "bamboo_passage_v1"}}
    save_project_choices(root, choices)
    assert resolve_prompt(project_alignment_context(root))[0]["prompt_id"] == "bamboo_passage_v1"
    from PySide6.QtWidgets import QApplication
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget
    QApplication.instance() or QApplication([])
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    assert widget.task_label.text() == "Bamboo Passage"
    assert widget.prompt_display.text() == expected
    assert not widget.prompt._alignment_row.isVisible()
    widget.close()


def test_nonlinguistic_tasks_are_exempt():
    assert task_entry("sustained_a")["alignment_applicable"] is False
    assert task_entry("ddk")["alignment_applicable"] is False


def test_recording_choice_and_reviewed_transcript_persist(tmp_path):
    root = _project(tmp_path)
    choices = load_project_choices(root)
    choices["reviewer_id"] = "reviewer_01"
    choices["recordings"] = {"r1": {"prompt_id": "wstg_we_see_three_geese",
        "speaker_id": "speaker_B", "alignment_transcript": "We see geese.",
        "transcript_reason": "omission"}}
    save_project_choices(root, choices)
    assert project_alignment_context(root)["records"][0]["speaker_id"] == "speaker_B"
    config = build_task_alignment_config(root)
    override = json.loads(open(config.transcript_overrides_path, encoding="utf-8").read())
    assert override["overrides"][0]["alignment_transcript"] == "We see geese."
    assert override["overrides"][0]["reason"] == "omission"


def test_wstg_alignment_gui_displays_prompt_without_manifest_browser(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    root = _project(tmp_path)
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    assert widget.task_label.text() == "WSTG"
    assert widget.prompt_display.text() == "We see three geese."
    assert widget.assignments.item(0, 2).text() == "speaker_A"
    assert not widget.prompt._alignment_row.isVisible()
    assert not widget.speakers._alignment_row.isVisible()
    calls = []
    widget.run_task_requested.connect(lambda root, profile: calls.append((root, profile)))
    widget.show_preflight_result({"status": "READY", "environment": {
        "status": "AVAILABLE", "version": "3.3.4", "conda_environment": "vslp-mfa-334",
        "resources": {"acoustic": {"identity": "english_us_arpa"},
                      "dictionary": {"identity": "english_us_arpa"}}}})
    widget.run_button.click()
    assert len(calls) == 1
    assert calls[0][0] == str(root)
    widget.close()
    assert app is not None


def test_multiple_prompt_task_requires_explicit_recording_selection(tmp_path, monkeypatch):
    from vslp.acoustic.alignment import task_workflow
    root = _project(tmp_path)
    review = (root / "acoustic" / "003_segmentation_review" / "final" /
              "final_segmentation_decisions.csv")
    decisions = pd.read_csv(review)
    second = decisions.iloc[0].copy()
    second["recording_id"] = "r2"
    second["file_name"] = "another_opaque.wav"
    pd.concat([decisions, pd.DataFrame([second])], ignore_index=True).to_csv(review, index=False)
    registry = task_workflow.task_registry()
    registry["tasks"][0]["prompts"].append({"prompt_id": "explicit_second",
        "prompt_version": "v2", "exact_expected_text": "A second supplied sentence.",
        "language": "en"})
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setattr(task_workflow, "REGISTRY_PATH", registry_path)
    with pytest.raises(ValueError, match="PROMPT_SELECTION_REQUIRED"):
        build_task_alignment_config(root)
    choices = load_project_choices(root)
    choices["recordings"] = {"r1": {"prompt_id": "wstg_we_see_three_geese",
                                      "speaker_id": "speaker_A"},
                            "r2": {"prompt_id": "explicit_second",
                                   "speaker_id": "speaker_A"}}
    save_project_choices(root, choices)
    config = build_task_alignment_config(root)
    assert PromptManifest.load(config.recording_prompt_manifest_paths["r1"]).prompt_version == "v1"
    assert PromptManifest.load(config.recording_prompt_manifest_paths["r2"]).prompt_version == "v2"


@pytest.mark.parametrize(("task_id", "selected", "destination"), [
    ("bamboo_passage", ["f1_token_hz"], "Alignment"),
    ("bamboo_passage", ["pause_count"], "Acoustic Features"),
    ("ddk", ["ddk_rate_syll_s"], "Acoustic Features"),
    ("sustained_a", ["f0_mean_hz"], "Acoustic Features"),
])
def test_review_continuation_routes_only_required_linguistic_alignment(
        tmp_path, monkeypatch, task_id, selected, destination):
    from PySide6.QtWidgets import QApplication
    from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow

    QApplication.instance() or QApplication([])
    root = _project(tmp_path, task=task_id)
    intervals = (root / "acoustic" / "003_segmentation_review" / "final" /
                 "final_segmentation_intervals.csv")
    intervals.write_text("recording_id,view,segment_role,start_sec,end_sec\n"
                         "r1,authoritative,speech,0,1\n", encoding="utf-8")
    window = AcousticPipelineWindow()
    window._run_root = root
    monkeypatch.setattr(window, "_require_project_initialized", lambda: True)
    monkeypatch.setattr(window, "_selected_feature_names", lambda: selected)
    monkeypatch.setattr(window, "_refresh_feature_count_label", lambda: None)
    assert window.run_all_btn.text() == "Run to Manual Review"
    assert window.review_widget.continue_button.text() == "Continue after Review"
    window.run_after_review()
    assert window.tabs.tabText(window.tabs.currentIndex()) == destination
    window.close()
