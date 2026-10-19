"""The two launchable windows have distinct jobs and share canonical prompts."""

from __future__ import annotations

import json
import os

from PySide6.QtWidgets import QApplication, QFileDialog

from vslp.acoustic.alignment.task_workflow import project_alignment_context, task_entry
from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow
from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget
from vslp.gui.alignment_app.main_window import AlignmentWindow, alignment_tasks


def _app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def test_independent_windows_and_canonical_task_setup():
    _app()
    acoustic = AcousticPipelineWindow()
    alignment = AlignmentWindow()
    assert acoustic.windowTitle() == "VSLP Acoustic Features"
    assert alignment.windowTitle() == "VSLP Alignment"
    assert "Alignment" not in [acoustic.tabs.tabText(i) for i in range(acoustic.tabs.count())]
    assert "Alignment Review" in [alignment.tabs.tabText(i) for i in range(alignment.tabs.count())]
    assert acoustic.alignment_widget is None
    assert not hasattr(alignment, "feature_tree")
    assert alignment._selected_task()["task_id"] == "bamboo_passage"
    assert alignment.repetitions.value() == 1
    assert alignment.transcript.toPlainText() == task_entry("bamboo_passage")["prompts"][0][
        "exact_expected_text"]
    alignment.task_type.setCurrentIndex(1)
    s2 = alignment.variant.findData("sentence_s2")
    alignment.variant.setCurrentIndex(s2)
    assert alignment.repetitions.value() == 3
    assert alignment.transcript.toPlainText() == task_entry("sentence_s2")["prompts"][0][
        "exact_expected_text"]
    assert {task["task_id"] for task in alignment_tasks()} >= {
        "bamboo_passage", "sentence_s2", "sentence_s5", "sentence_s7"}
    acoustic.close()
    alignment.close()


def test_job_repetitions_are_explicit_without_changing_registry(tmp_path):
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    (root / "project_manifest.json").write_text(json.dumps({
        "task_id": "sentence_s2", "task_name": "S2",
        "task_type": "Sentence / controlled speech"}), encoding="utf-8")
    prompt = task_entry("sentence_s2")["prompts"][0]
    (root / "configs" / "alignment_job.json").write_text(json.dumps({
        "task_id": "sentence_s2", "prompt_id": prompt["prompt_id"],
        "expected_repetitions": 4}), encoding="utf-8")
    context = project_alignment_context(root)
    assert context["expected_repetitions"] == 4
    assert task_entry("sentence_s2")["prompts"][0]["expected_repetitions"] == 3


def test_review_run_change_does_not_repopulate_recordings_during_clear(tmp_path, monkeypatch):
    _app()
    review = AlignmentWidget(lambda: tmp_path)
    monkeypatch.setattr(review, "_show_step", lambda: None)
    review.recordings.blockSignals(True)
    review.recordings.addItem("old recording", "old")
    review.recordings.blockSignals(False)
    selection_events = []
    review.recordings.currentIndexChanged.connect(selection_events.append)
    review._run_changed()
    assert review.recordings.count() == 0
    assert selection_events == []


def test_reopened_acoustic_project_displays_imported_package_status(tmp_path, monkeypatch):
    _app()
    window = AcousticPipelineWindow()
    window._run_root = tmp_path
    active = tmp_path / "alignment_imports" / "active.json"
    active.parent.mkdir()
    active.write_text(json.dumps({"package_id": "abc123"}), encoding="utf-8")
    monkeypatch.setattr("vslp.gui.acoustic_app.main_window.inspect_alignment_package",
                        lambda *_: {"manifest": {"package_contract_version":
                            "VSLP_ALIGNMENT_PACKAGE_V1", "recording_count": 4},
                            "matched": 4, "finalized": 4, "missing": 0,
                            "stale": 0, "invalid": 0, "error": ""})
    window._refresh_alignment_package_status()
    assert "Matched 4/4" in window.feature_package_status.text()
    assert "Finalized 4" in window.feature_package_status.text()
    window.close()


def test_alignment_setup_inputs_and_export_route(tmp_path, monkeypatch):
    _app()
    window = AlignmentWindow()
    wav = tmp_path / "sample.wav"
    wav.touch()
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        lambda *_args: ([str(wav)], "Audio"))
    window._add_files()
    assert window._input_files == [wav]
    assert window.input_label.text() == "1 recording(s) selected"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *_args: str(tmp_path))
    window._add_folder()
    assert window._input_files == []
    assert window.input_label.text() == str(tmp_path)
    window._root = tmp_path
    dispatched = []
    monkeypatch.setattr(window, "_run", lambda *args: dispatched.append(args))
    window._export()
    assert dispatched[0][0] == "export"
    assert dispatched[0][2]["project_root"] == tmp_path
    assert dispatched[0][2]["destination"].parent == tmp_path
    window.close()
