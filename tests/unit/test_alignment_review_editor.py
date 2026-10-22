"""Minimum interaction acceptance for the standalone Alignment review editor."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from PySide6.QtCore import QPointF, Qt, QUrl
from PySide6.QtWidgets import QApplication, QFileDialog

from vslp.acoustic.alignment.task_workflow import load_project_choices, task_entry
from vslp.acoustic.alignment.trial_review import set_trial_review, trial_review_table
from vslp.gui.alignment_app.inputs import prepare_alignment_inputs
from vslp.gui.alignment_app.main_window import (
    AlignmentWindow, _local_directory, validate_export_folder,
)
from vslp.gui.alignment_app.review_editor import AlignmentReviewEditor


def _app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def test_alignment_preparation_uses_full_recording_scope_without_silero(tmp_path, monkeypatch):
    wav = tmp_path / "sample.wav"
    wav.touch()
    summary = tmp_path / "preprocess.csv"
    pd.DataFrame([{"recording_id": "one", "file_name": "sample.wav", "status": "ok",
                   "analysis_wav_path": str(wav), "analysis_duration_sec": 2.5,
                   "source_sha256": "abc"}]).to_csv(summary, index=False)
    monkeypatch.setattr("vslp.gui.alignment_app.inputs.run_acoustic_ingest",
                        lambda *_args, **_kwargs: SimpleNamespace(summary_table=summary))
    monkeypatch.setattr("vslp.gui.alignment_app.inputs.run_acoustic_preprocess",
                        lambda *_args, **_kwargs: SimpleNamespace(summary_table=summary))
    result = prepare_alignment_inputs(tmp_path, tmp_path)
    assert result["recording_count"] == 1
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    decisions = pd.read_csv(final / "final_segmentation_decisions.csv")
    intervals = pd.read_csv(final / "final_segmentation_intervals.csv")
    assert decisions.iloc[0].segmentation_method == "alignment_full_recording_scope"
    assert intervals.iloc[0].boundary_source == "ALIGNMENT_SCOPE_UNREVIEWED"
    assert intervals.iloc[0].end_sec == 2.5
    assert load_project_choices(tmp_path)["manual_trial_review_required"] is True


def test_review_selection_zoom_and_fit(tmp_path):
    _app()
    editor = AlignmentReviewEditor(lambda: tmp_path)
    editor._recording_duration = 10.0
    editor.plot.setXRange(0, 10, padding=0)
    editor._set_selection(2.0, 1.0)
    assert editor._selected_interval == (1.0, 2.0)
    assert len(editor._selection_overlays) == 3
    editor._zoom(0.5)
    assert editor.plot.viewRange()[0][1] - editor.plot.viewRange()[0][0] < 10
    editor._fit_full_recording()
    assert editor.plot.viewRange()[0] == [0.0, 10.0]
    assert len(editor._editor_shortcuts) == 9
    editor.close()


def test_waveform_click_moves_shared_playhead(tmp_path):
    _app()
    editor = AlignmentReviewEditor(lambda: tmp_path)
    editor.show()
    QApplication.processEvents()
    editor.plot.setXRange(0, 10, padding=0)
    scene_point = editor.plot.getViewBox().mapViewToScene(QPointF(3.0, 0.0))

    class Click:
        def scenePos(self):  # noqa: N802 - pyqtgraph event shape
            return scene_point

        def modifiers(self):
            return Qt.NoModifier

        def double(self):
            return False

    editor._token_clicked(Click())
    assert abs(editor.play_cursor.value() - 3.0) < .01
    assert all(abs(cursor.value() - 3.0) < .01 for cursor in editor._tier_cursors)
    editor.close()


def test_previous_and_next_flagged_navigation(tmp_path, monkeypatch):
    _app()
    editor = AlignmentReviewEditor(lambda: tmp_path)
    monkeypatch.setattr(editor, "_apply_queue", lambda: None)
    monkeypatch.setattr(editor, "_recording_changed", lambda: None)
    editor.recordings.addItem("First", "one")
    editor.recordings.addItem("Second", "two")
    editor.recordings.addItem("Third", "three")
    editor.queue.setCurrentText("Needs Review")
    editor._step_flagged(1)
    assert editor.recordings.currentData() == "two"
    editor._step_flagged(1)
    assert editor.recordings.currentData() == "three"
    editor._step_flagged(-1)
    assert editor.recordings.currentData() == "two"
    editor.close()


def test_issue_and_note_persist_in_existing_review_table(tmp_path, monkeypatch):
    _app()
    base = tmp_path / "acoustic" / "004_alignment" / "runs" / "run1" / "tables"
    base.mkdir(parents=True)
    pd.DataFrame([{"recording_id": "one", "trial_id": "trial_001",
                   "review_status": "NEEDS_REVIEW"}]).to_csv(
                       base / "alignment_trial_diagnostics.csv", index=False)
    editor = AlignmentReviewEditor(lambda: tmp_path)
    editor.runs.addItem("run1", "run1")
    editor.recordings.addItem("one", "one")
    editor.trials.addItem("Trial 1", {"trial_id": "trial_001"})
    monkeypatch.setattr("vslp.gui.alignment_app.review_editor.set_trial_review",
                        lambda *_args: None)
    monkeypatch.setattr(editor, "_trial_changed", lambda: None)
    monkeypatch.setattr(editor, "_apply_queue", lambda: None)
    editor._record_issue("Word boundary", "Move left edge")
    saved = pd.read_csv(base / "alignment_trial_review.csv", keep_default_na=False)
    assert saved.iloc[0].gui_issue_category == "Word boundary"
    assert saved.iloc[0].gui_issue_note == "Move left edge"
    assert saved.iloc[0].gui_reviewer_action == "FLAGGED"
    editor.close()


def test_review_status_persists_without_losing_issue_note(tmp_path):
    base = tmp_path / "acoustic" / "004_alignment" / "runs" / "run1" / "tables"
    base.mkdir(parents=True)
    pd.DataFrame([{"recording_id": "one", "trial_id": "trial_001"}]).to_csv(
        base / "alignment_trial_diagnostics.csv", index=False)
    pd.DataFrame([{"recording_id": "one", "canonical_sample_rate_hz": 48000}]).to_csv(
        base / "alignment_diagnostics.csv", index=False)
    pd.DataFrame([{"recording_id": "one", "word_index": 1,
                   "start_sec": 0.1, "end_sec": 0.9}]).to_csv(
                       base / "alignment_words.csv", index=False)
    pd.DataFrame([{"recording_id": "one", "phone_index": 1, "word_index": 1,
                   "start_sec": 0.2, "end_sec": 0.8}]).to_csv(
                       base / "alignment_phones.csv", index=False)
    pd.DataFrame([{"recording_id": "one", "trial_id": "trial_001",
                   "token_type": "word", "token_index": 1},
                  {"recording_id": "one", "trial_id": "trial_001",
                   "token_type": "phone", "token_index": 1}]).to_csv(
                       base / "alignment_trial_tokens.csv", index=False)
    pd.DataFrame([{"recording_id": "one", "trial_id": "trial_001",
                   "review_status": "NEEDS_REVIEW", "gui_issue_category": "Word boundary",
                   "gui_issue_note": "Check onset"}]).to_csv(
                       base / "alignment_trial_review.csv", index=False)
    set_trial_review(tmp_path, "run1", "one", "trial_001", "ACCEPTED")
    row = trial_review_table(tmp_path, "run1").iloc[0]
    assert row.review_status == "ACCEPTED"
    assert row.alignment_review_status == "HUMAN_ACCEPTED"
    assert row.gui_issue_category == "Word boundary"
    assert row.gui_issue_note == "Check onset"


def test_sentence_selector_and_repetitions():
    _app()
    window = AlignmentWindow()
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        "Setup", "Align", "Review", "Export"]
    window.task_type.setCurrentIndex(1)
    window.variant.setCurrentIndex(window.variant.findData("sentence_s2"))
    assert window.variant.isHidden() is False
    assert window.repetitions.value() == 3
    assert window.transcript.toPlainText() == task_entry("sentence_s2")["prompts"][0][
        "exact_expected_text"]
    window.repetitions.setValue(4)
    assert window.repetitions.value() == 4
    window.close()


def test_stale_unc_folder_is_never_probed_or_used(tmp_path, monkeypatch):
    _app()
    monkeypatch.setattr("vslp.gui.alignment_app.main_window.QSettings", lambda *_args:
                        SimpleNamespace(value=lambda *_args: r"\\rssfiles\home\slp\musikicn\Desktop",
                                        setValue=lambda *_args: None))
    window = AlignmentWindow()
    window._last_local_folder = r"\\rssfiles\home\slp\musikicn\Desktop"
    window._root = tmp_path
    original_is_dir = type(tmp_path).is_dir

    def guarded_is_dir(path):
        assert not str(path).startswith("\\\\")
        return original_is_dir(path)

    monkeypatch.setattr(type(tmp_path), "is_dir", guarded_is_dir)
    assert _local_directory(window._last_local_folder) is None
    assert window._dialog_start() == str(tmp_path)
    captured = []

    def choose(*args):
        captured.append(args)
        return str(tmp_path)

    monkeypatch.setattr("vslp.gui.alignment_app.main_window.QFileDialog.getExistingDirectory",
                        choose)
    assert window._choose_folder("Choose") == str(tmp_path)
    assert captured[0][2] == str(tmp_path)
    assert captured[0][3] == QFileDialog.DontUseNativeDialog
    window.close()


def test_project_metadata_worker_and_lazy_review(tmp_path, monkeypatch):
    _app()
    window = AlignmentWindow()
    events = []
    monkeypatch.setattr(window.review, "refresh", lambda: events.append("review_loaded"))

    def snapshot(root):
        time.sleep(.05)
        return {"root": root, "manifest": {"task_id": "bamboo_passage"},
                "context": {"records": [], "choices": {}, "task_name": "Bamboo Passage",
                            "expected_repetitions": 1},
                "finalized": set(), "job": {}}

    monkeypatch.setattr("vslp.gui.alignment_app.main_window.load_project_snapshot", snapshot)
    window.open_project(tmp_path)
    assert window.progress_label.text() == "Loading project..."
    assert events == []
    deadline = time.monotonic() + 3
    while window._thread is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.005)
    assert window._thread is None
    assert window.batch_table.rowCount() == 0
    assert events == []
    monkeypatch.setattr("vslp.gui.alignment_app.main_window.preload_review_data",
                        lambda root: {"run_id": "", "final_alignment": (None, "missing")})
    window._review_alignment()
    deadline = time.monotonic() + 3
    while not events and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.005)
    assert events == ["review_loaded"]
    window.close()


def test_technical_log_remains_present_when_collapsed():
    _app()
    window = AlignmentWindow()
    window.show()
    QApplication.processEvents()
    window._log_message("Project opened", "INFO")
    window._log_message("Project warning", "WARNING")
    assert "INFO  Project opened" in window.log.toPlainText()
    assert "WARNING  Project warning" in window.log.toPlainText()
    assert window.technical.isVisible()
    window._toggle_log()
    assert window.technical.isVisible()
    assert not window.log.isVisible()
    window._toggle_log()
    assert window.log.isVisible()
    window.close()


def test_worker_error_records_detail_in_log_and_short_status(monkeypatch):
    _app()
    window = AlignmentWindow()
    summaries = []
    monkeypatch.setattr("vslp.gui.alignment_app.main_window.QMessageBox.critical",
                        lambda _parent, _title, message: summaries.append(message))
    window._failed("project_load", "Folder unavailable\n\nTraceback details")
    assert summaries == ["Folder unavailable"]
    assert "ERROR  project_load: Folder unavailable" in window.log.toPlainText()
    assert "Traceback details" in window.log.toPlainText()
    assert window.statusBar().currentMessage() == "project_load failed"
    window.close()


def test_export_folder_write_probe_and_fallback(tmp_path, monkeypatch):
    _app()
    created = validate_export_folder(tmp_path / "new" / "AlignmentExports")
    assert created.is_dir()
    assert not list(created.glob(".vslp_export_check_*"))
    window = AlignmentWindow()
    monkeypatch.setattr(window, "_export_candidates",
                        lambda: (Path(r"\\rssfiles\home\slp\Desktop"), tmp_path / "safe"))
    assert window._safe_export_default() == tmp_path / "safe"
    assert (tmp_path / "safe").is_dir()
    monkeypatch.setattr(window, "_export_candidates",
                        lambda: (tmp_path / "Desktop" / "project" / "exports",
                                 tmp_path / "safe"))
    assert window._safe_export_default() == tmp_path / "safe"
    assert not (tmp_path / "Desktop").exists()
    window.close()


def test_denied_export_folder_never_starts_package_worker(tmp_path, monkeypatch):
    _app()
    window = AlignmentWindow()
    window._root = tmp_path
    window.export_parent.setText(str(tmp_path / "denied"))
    monkeypatch.setattr("vslp.gui.alignment_app.main_window.validate_export_folder",
                        lambda _path: (_ for _ in ()).throw(PermissionError("denied")))
    warnings = []
    monkeypatch.setattr("vslp.gui.alignment_app.main_window.QMessageBox.warning",
                        lambda *_args: warnings.append(True))
    prompted = []
    monkeypatch.setattr(window, "_choose_export_folder", lambda: prompted.append(True))
    dispatched = []
    monkeypatch.setattr(window, "_run", lambda *_args: dispatched.append(True))
    window._export()
    assert warnings == [True]
    assert prompted == [True]
    assert dispatched == []
    assert "Export path validation failed" in window.log.toPlainText()
    window.close()


def test_playback_speed_controls_and_current_intervals(tmp_path, monkeypatch):
    _app()
    editor = AlignmentReviewEditor(lambda: tmp_path)
    assert editor.speed_buttons[1.0].isChecked()
    assert not editor.speed_buttons[0.75].isEnabled()
    wav = tmp_path / "sample.wav"
    wav.touch()
    editor.player.setSource(QUrl.fromLocalFile(str(wav)))
    editor._sync_playback_controls()
    available = editor.player.pitchCompensationAvailability().name != "Unavailable"
    assert editor.speed_buttons[0.5].isEnabled() is available
    assert editor.speed_buttons[0.75].isEnabled() is available
    if available:
        editor.speed_buttons[0.75].click()
        assert editor.player.playbackRate() == 0.75
        assert editor.player.pitchCompensation()
    else:
        assert "Pitch-preserving" in editor.speed_buttons[0.75].toolTip()

    class Player:
        def __init__(self):
            self.position = None
            self.play_count = 0
            self.rate = None

        def setPosition(self, value):  # noqa: N802 - QMediaPlayer shape
            self.position = value

        def setPlaybackRate(self, value):  # noqa: N802 - QMediaPlayer shape
            self.rate = value

        def play(self):
            self.play_count += 1

    fake = Player()
    monkeypatch.setattr(editor, "_ensure_player", lambda: fake)
    editor.trials.addItem("Trial 1", {"start_sec": 1.22, "end_sec": 62.65})
    editor._play_trial()
    assert (fake.position, editor._play_end_ms, fake.play_count) == (1220, 62650, 1)
    editor.trials.addItem("Trial 2", {"start_sec": 65.0, "end_sec": 90.0})
    editor.trials.setCurrentIndex(1)
    editor._play_trial()
    assert (fake.position, editor._play_end_ms, fake.play_count) == (65000, 90000, 2)
    editor.sentence_selector.addItem("Sentence 1", (3.0, 7.5))
    editor._play_sentence()
    assert (fake.position, editor._play_end_ms, fake.play_count) == (3000, 7500, 3)
    editor._set_selection(4.0, 5.0)
    editor._play_selection()
    assert (fake.position, editor._play_end_ms, fake.play_count) == (4000, 5000, 4)
    if available:
        editor._set_speed(0.5)
        assert fake.rate == 0.5
    editor.close()
