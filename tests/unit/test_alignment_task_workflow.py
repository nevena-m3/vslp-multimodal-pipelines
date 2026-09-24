"""Explicit task, prompt, and speaker resolution without filename inference."""

from __future__ import annotations

import json

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
        "task_id": task, "task_name": "WSTG" if task == "wstg" else task}), encoding="utf-8")
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


def test_bamboo_requires_source_text_and_nonlinguistic_tasks_are_exempt(tmp_path):
    root = _project(tmp_path, task="bamboo_passage")
    assert resolve_prompt(project_alignment_context(root))[1] == "MISSING_CANONICAL_PROMPT"
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
