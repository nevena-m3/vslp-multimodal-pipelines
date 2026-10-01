"""Explicit task, prompt, and speaker resolution without filename inference."""

from __future__ import annotations

import json
import hashlib

import pandas as pd
import pytest

from vslp.acoustic.alignment.stage import PromptManifest
from vslp.acoustic.alignment.task_workflow import (
    build_task_alignment_config, check_task_preflight, load_project_choices,
    project_alignment_context, resolve_prompt, run_task_alignment,
    save_project_choices, task_entry,
)


_QT_APP = None


def _keep_qt_app():
    """Retain one QApplication through module teardown on Windows."""
    global _QT_APP
    from PySide6.QtWidgets import QApplication
    _QT_APP = QApplication.instance() or QApplication([])
    return _QT_APP


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
                   "analysis_start_sec": 0, "analysis_end_sec": 1,
                   "final_decision": "KEEP_AUTO"}]).to_csv(
                       review / "final_segmentation_decisions.csv", index=False)
    pd.DataFrame([{"recording_id": "r1", "segment_role": "speech",
                   "start_sec": .1, "end_sec": .9}]).to_csv(
                       review / "final_segmentation_intervals.csv", index=False)
    save_project_choices(root, {"schema_version": "1", "recordings": {},
        "trials": {"r1": [{"trial_id": "trial_001", "start_sec": .1, "end_sec": .9}]}})
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


@pytest.mark.parametrize("task,name,prompt_id,expected_repetitions", [
    ("wstg", "WSTG", "wstg_we_see_three_geese", 3),
    ("bamboo_passage", "Bamboo Passage", "bamboo_passage_v1", 1),
])
def test_existing_project_resolves_registered_setup_task_when_legacy_id_is_unknown(
        tmp_path, task, name, prompt_id, expected_repetitions):
    """An older Setup slug must not display TASK_NOT_REGISTERED on reopen."""
    root = _project(tmp_path, task=task)
    manifest_path = root / "project_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task_id"] = "legacy_setup_slug"
    manifest["task_name"] = name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    context = project_alignment_context(root)
    prompt, issue = resolve_prompt(context)
    assert issue == ""
    assert context["task_id"] == task
    assert prompt["prompt_id"] == prompt_id
    assert prompt.get("expected_repetitions") == expected_repetitions
    assert project_alignment_context(root)["task_id"] == task
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget
    _keep_qt_app()
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    assert widget.prompt_display.text() == prompt["exact_expected_text"]
    assert "TASK_NOT_REGISTERED" not in widget.task_brief.text()
    if expected_repetitions is not None:
        assert f"Expected repetitions: {expected_repetitions}" in widget.trial_count.text()
    widget.close()


def test_unknown_setup_task_does_not_guess_from_recording_filename(tmp_path):
    root = _project(tmp_path)
    manifest_path = root / "project_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(task_id="unknown", task_name="Other task")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    context = project_alignment_context(root)
    assert resolve_prompt(context)[1] == "TASK_NOT_REGISTERED"


def test_no_filename_speaker_or_prompt_inference(tmp_path):
    root = _project(tmp_path, metadata_speaker="")
    context = project_alignment_context(root)
    key = context["records"][0]["speaker_id"]
    assert key.startswith("technical_")
    assert context["records"][0]["speaker_id_source"] == "technical_recording_key"
    assert project_alignment_context(root)["records"][0]["speaker_id"] == key
    assert build_task_alignment_config(root).speaker_manifest_path


@pytest.mark.parametrize("task,expected", [
    ("wstg", "We see three geese."),
    ("bamboo_passage", "Bamboo walls are getting to be very popular."),
])
def test_task_run_hands_registered_prompt_to_alignment(tmp_path, monkeypatch, task, expected):
    from vslp.acoustic.alignment import task_workflow

    root = _project(tmp_path, task=task)
    handed_off = []
    monkeypatch.setattr(task_workflow, "run_acoustic_alignment",
                        lambda output_root, config, progress_callback=None:
                        handed_off.append((output_root, config)))
    run_task_alignment(root)
    assert handed_off[0][0] == root
    config = handed_off[0][1]
    assert config.prompt_manifest_path
    assert PromptManifest.load(config.prompt_manifest_path).transcript.startswith(expected)


@pytest.mark.parametrize("task", ["wstg", "bamboo_passage"])
def test_validated_import_uses_registered_prompt_without_file_browsing(tmp_path, task):
    from PySide6.QtWidgets import QApplication
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget

    QApplication.instance() or QApplication([])
    root = _project(tmp_path, task=task, metadata_speaker="")
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    emitted = []
    widget.run_requested.connect(emitted.append)
    widget.source.setCurrentIndex(widget.source.findData("external"))
    widget._run()
    assert len(emitted) == 1
    assert PromptManifest.load(emitted[0].prompt_manifest_path).task_id == task
    assert emitted[0].recording_prompt_manifest_paths["r1"] == emitted[0].prompt_manifest_path
    widget.close()


def test_preflight_requires_confirmed_trials_not_manual_speaker_ids(tmp_path, monkeypatch):
    from vslp.acoustic.alignment import task_workflow

    root = _project(tmp_path, metadata_speaker="")
    review = root / "acoustic" / "003_segmentation_review" / "final"
    rows = [{"recording_id": f"r{i}", "file_name": f"opaque_{i}.wav",
             "analysis_start_sec": 0, "analysis_end_sec": 1,
             "final_decision": "KEEP_AUTO", "analysis_wav_path": str(root / "audio.wav")}
            for i in range(48)]
    pd.DataFrame(rows).to_csv(review / "final_segmentation_decisions.csv", index=False)
    pd.DataFrame([{"recording_id": f"r{i}", "segment_role": "speech",
                   "start_sec": .1, "end_sec": .9} for i in range(48)]).to_csv(
                       review / "final_segmentation_intervals.csv", index=False)
    (root / "audio.wav").write_bytes(b"synthetic fixture placeholder")
    monkeypatch.setattr(task_workflow.MfaProvider, "inspect_environment",
                        lambda *_args: pytest.fail("MFA should not be checked before trial review"))
    result = check_task_preflight(root)
    assert result["issue"].startswith("TRIAL_BOUNDARIES_REQUIRED")
    choices = load_project_choices(root)
    choices["trials"] = {f"r{i}": [{"trial_id": "trial_001", "start_sec": .1,
                                    "end_sec": .9}] for i in range(48)}
    save_project_choices(root, choices)
    def available(provider, _profile):
        provider.resolved_dictionary = str(root / "dictionary.dict")
        return {"status": "AVAILABLE", "version": "3.3.4"}
    monkeypatch.setattr(task_workflow.MfaProvider, "inspect_environment", available)
    monkeypatch.setattr(task_workflow, "dictionary_oovs", lambda *_args: [])
    ready = check_task_preflight(root)
    assert ready["status"] == "READY"
    assert ready["prompt"] == "We see three geese."
    assert len(ready["repetition_gaps"]) == 48
    assert ready["repetition_gaps"]["r0"] == {"expected": 3, "confirmed": 1}


def test_48_recording_technical_speakers_persist_without_entry(tmp_path):
    from PySide6.QtWidgets import QApplication
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget

    app = _keep_qt_app()
    root = _project(tmp_path, metadata_speaker="")
    review = root / "acoustic" / "003_segmentation_review" / "final"
    pd.DataFrame([{"recording_id": f"r{i}", "file_name": f"opaque_{i}.wav",
                   "analysis_start_sec": 0, "analysis_end_sec": 1,
                   "final_decision": "KEEP_AUTO"} for i in range(48)]).to_csv(
                       review / "final_segmentation_decisions.csv", index=False)
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    assert widget.assignments.rowCount() == 48
    assert "Available for all 48" in widget.speaker_status.text()
    assert all(widget.assignments.item(i, 2).text().startswith("technical_")
               for i in range(48))
    app.processEvents()
    first = widget.assignments.item(0, 2).text()
    second = widget.assignments.item(1, 2).text()
    assert first != second
    widget.close()
    widget.deleteLater()
    app.processEvents()
    reopened = AlignmentWidget(lambda: root)
    reopened.refresh()
    assert reopened.assignments.item(0, 2).text() == first
    assert reopened.assignments.item(1, 2).text() == second
    assert reopened.prompt_display.text() == "We see three geese."
    reopened.close()
    reopened.deleteLater()
    app.processEvents()
    assert app is not None


def test_run_log_clears_old_prompt_error_for_new_alignment_operation(tmp_path):
    from PySide6.QtWidgets import QApplication
    from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow

    QApplication.instance() or QApplication([])
    window = AcousticPipelineWindow()
    window.append_log("ERROR in alignment: MISSING_PROMPT")
    window._on_worker_started("alignment_preflight")
    assert "MISSING_PROMPT" not in window.log_box.toPlainText()
    assert "alignment_preflight" in window.log_box.toPlainText()
    alignment_tab = next(index for index in range(window.tabs.count())
                         if window.tabs.tabText(index) == "Alignment")
    window.tabs.setCurrentIndex(alignment_tab)
    assert window.log_box.isHidden()
    window.close()


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


@pytest.mark.parametrize("task", ["sustained_a", "ddk"])
def test_nonlinguistic_trials_can_be_confirmed_without_mfa(tmp_path, task):
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget
    root = _project(tmp_path, task=task, metadata_speaker="")
    choices = load_project_choices(root)
    choices["trials"] = {}
    save_project_choices(root, choices)
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    for start, end in ((.1, .4), (.5, .9)):
        widget.trial_start.setValue(start)
        widget.trial_end.setValue(end)
        widget._add_trial()
    assert widget.trials.count() == 2
    assert "Linguistic Alignment is not required" in widget.readiness.text()
    assert not widget.run_button.isEnabled()
    assert len(load_project_choices(root)["trials"]["r1"]) == 2
    widget.close()


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
    choices["trials"]["r2"] = [{"trial_id": "trial_001", "start_sec": .1,
                                  "end_sec": .9, "prompt_id": "explicit_second"}]
    save_project_choices(root, choices)
    config = build_task_alignment_config(root)
    assert PromptManifest.load(config.recording_prompt_manifest_paths["r1"]).prompt_version == "v1"
    assert PromptManifest.load(config.recording_prompt_manifest_paths["r2"]).prompt_version == "v2"
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget
    _keep_qt_app()
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    assert not widget.record_stimulus.isHidden()
    widget.recordings.setCurrentIndex(1)
    assert widget.record_stimulus.currentData() == "explicit_second"
    assert widget.prompt_display.text() == "A second supplied sentence."
    widget.close()


@pytest.mark.parametrize(("task_id", "selected", "destination"), [
    ("bamboo_passage", ["f1_token_hz"], "Acoustic Features"),
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


def test_feature_first_run_without_alignment_dispatches_directly(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow

    QApplication.instance() or QApplication([])
    root = _project(tmp_path, task="bamboo_passage")
    (root / "acoustic" / "003_segmentation_review" / "final" /
     "final_segmentation_intervals.csv").write_text(
         "recording_id,view,segment_role,start_sec,end_sec\nr1,authoritative,speech,0,1\n",
         encoding="utf-8")
    window = AcousticPipelineWindow()
    window._run_root = root
    monkeypatch.setattr(window, "_require_paths", lambda: (root, root))
    monkeypatch.setattr(window, "_require_project_initialized", lambda: True)
    monkeypatch.setattr(window, "_output_root", lambda: root)
    monkeypatch.setattr(window, "_selected_feature_names", lambda: ["pause_count"])
    dispatched = []
    monkeypatch.setattr(window, "_run_worker", lambda name, fn, kwargs:
                        dispatched.append((name, kwargs["config"].selected_features)))
    window.run_features()
    assert dispatched == [("features", ["pause_count"])]
    window.close()


@pytest.mark.parametrize("choice,expected", [
    ("RUN REQUIRED ALIGNMENT", "alignment"),
    ("RUN NON-ALIGNMENT FEATURES ONLY", "non_alignment"),
])
def test_mixed_feature_selection_prompts_for_alignment_only_when_needed(
        tmp_path, monkeypatch, choice, expected):
    from PySide6.QtWidgets import QApplication, QMessageBox
    from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow

    QApplication.instance() or QApplication([])
    root = _project(tmp_path, task="bamboo_passage")
    (root / "acoustic" / "003_segmentation_review" / "final" /
     "final_segmentation_intervals.csv").write_text(
         "recording_id,view,segment_role,start_sec,end_sec\nr1,authoritative,speech,0,1\n",
         encoding="utf-8")
    window = AcousticPipelineWindow()
    window._run_root = root
    monkeypatch.setattr(window, "_require_paths", lambda: (root, root))
    monkeypatch.setattr(window, "_output_root", lambda: root)
    monkeypatch.setattr(window, "_require_project_initialized", lambda: True)
    monkeypatch.setattr(window, "_selected_feature_names", lambda:
                        ["pause_count", "f1_vowel_median_hz"])
    dispatched = []
    monkeypatch.setattr(window, "_run_worker", lambda name, fn, kwargs:
                        dispatched.append(kwargs["config"].selected_features))

    def choose(dialog):
        next(button for button in dialog.buttons() if button.text() == choice).click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", choose)
    window.run_features()
    if expected == "alignment":
        assert dispatched == []
        assert window.tabs.currentWidget() is window.alignment_widget
        assert window._return_to_features_after_alignment
    else:
        assert dispatched == [["pause_count"]]
    window.close()


def test_return_from_required_alignment_keeps_feature_selection(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow

    QApplication.instance() or QApplication([])
    root = _project(tmp_path, task="bamboo_passage")
    window = AcousticPipelineWindow()
    window._run_root = root
    monkeypatch.setattr(window, "_selected_feature_names", lambda:
                        ["pause_count", "f1_vowel_median_hz"])
    window._return_to_features_after_alignment = True
    window.alignment_widget._feature_return_pending = True
    window.tabs.setCurrentWidget(window.alignment_widget)
    window.alignment_widget.return_to_features_requested.emit()
    assert window.tabs.tabText(window.tabs.currentIndex()) == "Acoustic Features"
    assert window._selected_feature_names() == ["pause_count", "f1_vowel_median_hz"]
    assert not window._return_to_features_after_alignment
    window.close()
