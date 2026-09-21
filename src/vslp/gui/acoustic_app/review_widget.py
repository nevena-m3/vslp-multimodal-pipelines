"""Interactive, persistent segmentation review workstation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout, QWidget)

from vslp.acoustic.segment.review import (EXCLUSION_REASONS, RECORDING_EXCLUSION_REASONS,
    finalize_segmentation_review,
    initialize_segmentation_review, load_review_exclusions, load_review_state,
    parse_manual_intervals_text, save_segmentation_review_entry, select_segmentation_run,
    validate_manual_interval_list)
from vslp.acoustic.segment.pipeline import segmentation_runs
from vslp.gui.acoustic_app.waveform_editor import WaveformEditor, playback_bounds


def _speech_intervals(path: str) -> list[tuple[float, float]]:
    if not path or not Path(path).is_file():
        return []
    table = pd.read_csv(path)
    return [(float(row.start_sec), float(row.end_sec)) for row in
            table.loc[table.segment_type.eq("speech")].itertuples()]


def _clock(seconds: float) -> str:
    minutes, remainder = divmod(max(0.0, float(seconds)), 60)
    return f"{int(minutes):02d}:{remainder:05.2f}"


class SegmentationReviewWidget(QWidget):
    finalized = Signal(object)
    continue_requested = Signal()

    def __init__(self, output_root: Callable[[], Path], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_root = output_root
        self._decisions = pd.DataFrame()
        self._overrides = pd.DataFrame()
        self._exclusions = pd.DataFrame()
        self._current_id = ""
        self._current_run_id = ""
        self._dirty = False
        self._loading = False
        self._in_apply = False
        self._selection_stop_ms: int | None = None
        self._undo: list[tuple[list[tuple[float, float]], float, float, list[dict]]] = []
        self._player = QMediaPlayer(self)
        self._player.setAudioOutput(QAudioOutput(self))
        self._player.positionChanged.connect(self._position_changed)
        self._selection_timer = QTimer(self)
        self._selection_timer.setInterval(20)
        self._selection_timer.timeout.connect(lambda: self._position_changed(self._player.position()))
        self._player.playbackStateChanged.connect(
            lambda state: self.play_button.setText("Pause" if state == QMediaPlayer.PlayingState else "Play"))
        outer = QVBoxLayout(self)
        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("Segmentation run to review:"))
        self.run_combo = QComboBox()
        self.run_combo.currentIndexChanged.connect(self._select_run)
        run_row.addWidget(self.run_combo, 1)
        outer.addLayout(run_row)
        top = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Pending review", "Review required", "Excluded automatically",
            "Accepted automatically", "Manually reviewed", "Kept automatic after review",
            "Kept manual", "Excluded", "All", "All recordings"])
        self.filter_combo.currentIndexChanged.connect(self._filter_rows)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Find recording or flag")
        self.search_edit.textChanged.connect(self._filter_rows)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        self.progress_label = QLabel("No review loaded")
        for widget in (self.filter_combo, self.search_edit, refresh, self.progress_label):
            top.addWidget(widget)
        outer.addLayout(top)
        splitter = QSplitter(Qt.Horizontal)
        self.recording_list = QListWidget()
        self.recording_list.currentItemChanged.connect(self._show_selected)
        splitter.addWidget(self.recording_list)
        detail = QWidget()
        right = QVBoxLayout(detail)
        self.identity_label, self.status_label, self.flags_label = QLabel("Select a recording"), QLabel(), QLabel()
        for label in (self.identity_label, self.status_label, self.flags_label):
            right.addWidget(label)
        self.availability_label = QLabel()
        right.addWidget(self.availability_label)
        self.saved_state_label = QLabel("Saved")
        right.addWidget(self.saved_state_label)
        self.editor = WaveformEditor()
        self.editor.seek_requested.connect(self._seek)
        self.editor.intervals_changed.connect(self._boundary_changed)
        self.editor.edit_started.connect(self._snapshot)
        right.addWidget(self.editor, 1)
        role_key = QLabel(
            '<span style="color:#27814f">■</span> Speech  '
            '<span style="color:#91b4c9">■</span> Leading  '
            '<span style="color:#e8ac91">■</span> Internal pause  '
            '<span style="color:#b19fc9">■</span> Trailing  '
            '<span style="color:#848d97">■</span> Outside analysis  '
            '<span style="color:#d28064">■</span> Excluded  '
            '<span style="color:#27814f">│</span> Auto  '
            '<span style="color:#7735a4">┊</span> Edited')
        right.addWidget(role_key)
        transport = QHBoxLayout()
        self.back_button = QPushButton("◀ 5s")
        self.back_button.clicked.connect(lambda: self._seek(max(0, self.editor.cursor_sec - 5)))
        pan_left = QPushButton("◀ View")
        pan_left.clicked.connect(lambda: self.editor.pan_time(-0.75))
        pan_right = QPushButton("View ▶")
        pan_right.clicked.connect(lambda: self.editor.pan_time(0.75))
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._play_pause)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop)
        self.selection_play_button = QPushButton("Play selection")
        self.selection_play_button.clicked.connect(self._play_selection)
        self.time_label = QLabel("00:00.00 / 00:00.00")
        for widget in (pan_left, pan_right, self.back_button, self.play_button, self.stop_button,
                       self.selection_play_button, self.time_label):
            transport.addWidget(widget)
        transport.addStretch(1)
        right.addLayout(transport)
        self.edit_button = QPushButton("Edit boundaries")
        self.edit_button.clicked.connect(self._enter_edit_mode)
        right.addWidget(self.edit_button)
        self.edit_controls = QWidget()
        edits = QVBoxLayout(self.edit_controls)
        edits.setContentsMargins(0, 0, 0, 0)
        row1 = QHBoxLayout()
        self.start_button = QPushButton("Set analysis start")
        self.start_button.clicked.connect(lambda: self._apply(lambda: self.editor.set_analysis_window(
            self.editor.cursor_sec, self.editor.analysis_end_sec), "SET_ANALYSIS_START"))
        self.end_button = QPushButton("Set analysis end")
        self.end_button.clicked.connect(lambda: self._apply(lambda: self.editor.set_analysis_window(
            self.editor.analysis_start_sec, self.editor.cursor_sec), "SET_ANALYSIS_END"))
        self.full_window_button = QPushButton("Full recording")
        self.full_window_button.clicked.connect(lambda: self._apply(lambda: self.editor.set_analysis_window(
            0, self.editor.duration_sec), "RESET_ANALYSIS_WINDOW"))
        self.add_button = QPushButton("Add speech interval")
        self.add_button.clicked.connect(self._add_interval)
        self.delete_button = QPushButton("Delete selected speech")
        self.delete_button.clicked.connect(lambda: self._apply(self.editor.delete_selected, "DELETE_INTERVAL"))
        for widget in (self.start_button, self.end_button, self.full_window_button,
                       self.add_button, self.delete_button):
            row1.addWidget(widget)
        edits.addLayout(row1)
        row2 = QHBoxLayout()
        self.reason_combo = QComboBox()
        self.reason_combo.addItems(sorted(EXCLUSION_REASONS))
        self.exclude_interval_button = QPushButton("Exclude selected segment / range")
        self.exclude_interval_button.clicked.connect(self._exclude_selection)
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self._undo_edit)
        self.reset_button = QPushButton("Reset to automatic")
        self.reset_button.clicked.connect(self._reset_automatic)
        self.zoom_button = QPushButton("Zoom to selection")
        self.zoom_button.clicked.connect(self.editor.zoom_to_selection)
        for widget in (self.reason_combo, self.exclude_interval_button, self.undo_button,
                       self.reset_button, self.zoom_button):
            row2.addWidget(widget)
        edits.addLayout(row2)
        self.manual_edit = QPlainTextEdit()
        self.manual_edit.setPlaceholderText("Advanced: one start_sec,end_sec interval per line")
        self.manual_edit.setMaximumHeight(70)
        self.manual_edit.textChanged.connect(self._advanced_changed)
        edits.addWidget(self.manual_edit)
        self.edit_controls.hide()
        right.addWidget(self.edit_controls)
        self.reviewer_edit = QLineEdit()
        self.reviewer_edit.setPlaceholderText("Reviewer name")
        self.reviewer_edit.textChanged.connect(self._mark_dirty)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Review notes, including corrections")
        self.notes_edit.textChanged.connect(self._mark_dirty)
        self.notes_edit.setMaximumHeight(60)
        right.addWidget(self.reviewer_edit)
        right.addWidget(self.notes_edit)
        self.recording_reason_combo = QComboBox()
        self.recording_reason_combo.addItems(sorted(RECORDING_EXCLUSION_REASONS))
        self.recording_reason_combo.setToolTip("Reason for excluding the whole recording")
        right.addWidget(self.recording_reason_combo)
        actions = QHBoxLayout()
        self.keep_button = QPushButton("Keep automatic")
        self.keep_button.clicked.connect(lambda: self._save("KEEP_AUTO"))
        self.save_manual_button = QPushButton("Save manual")
        self.save_manual_button.clicked.connect(lambda: self._save("KEEP_MANUAL"))
        self.exclude_button = QPushButton("Exclude recording")
        self.exclude_button.clicked.connect(lambda: self._save("EXCLUDE"))
        for button in (self.keep_button, self.save_manual_button, self.exclude_button):
            actions.addWidget(button)
        right.addLayout(actions)
        navigation = QHBoxLayout()
        previous = QPushButton("Previous")
        previous.clicked.connect(lambda: self._navigate(-1))
        next_button = QPushButton("Next")
        next_button.clicked.connect(lambda: self._navigate(1))
        self.finalize_button = QPushButton("Freeze final segmentation")
        self.finalize_button.clicked.connect(self._finalize)
        self.continue_button = QPushButton("Continue to QC & Features")
        self.continue_button.clicked.connect(self.continue_requested.emit)
        self.continue_button.setEnabled(False)
        for button in (previous, next_button, self.finalize_button, self.continue_button):
            navigation.addWidget(button)
        right.addLayout(navigation)
        splitter.addWidget(detail)
        splitter.setStretchFactor(1, 4)
        outer.addWidget(splitter, 1)
        space = QShortcut(QKeySequence(Qt.Key_Space), self)
        space.setContext(Qt.WidgetWithChildrenShortcut)
        space.activated.connect(self._space_pressed)
        escape = QShortcut(QKeySequence(Qt.Key_Escape), self)
        escape.setContext(Qt.WidgetWithChildrenShortcut)
        escape.activated.connect(self.editor.clear_selection)

    def refresh(self) -> None:
        try:
            root = self._output_root()
            runs = segmentation_runs(root)
            self.run_combo.blockSignals(True)
            self.run_combo.clear()
            if len(runs) > 1:
                self.run_combo.addItem("Choose a segmentation run…", "")
            for entry in runs:
                params = entry["parameters"]
                if entry["method"] == "silero_vad":
                    profile = (f"threshold={float(params.get('threshold', .5)):.2f}, "
                               f"min speech={params.get('min_speech_duration_ms', '?')}, "
                               f"min silence={params.get('min_silence_duration_ms', '?')}")
                elif entry["method"] == "ddk_energy":
                    ddk = params.get("ddk", {})
                    profile = (f"LPF={ddk.get('envelope_lowpass_hz', '?')} Hz, "
                               f"energy={ddk.get('frame_ms', '?')} ms, "
                               f"threshold={ddk.get('threshold_window_ms', '?')} ms")
                else:
                    profile = "sustained phonation"
                label = (f"{entry['created_at_utc']} · {entry['method']} · {entry['task_name']} · "
                         f"{profile} · N={entry['n_recordings']}")
                self.run_combo.addItem(label, entry["segmentation_run_id"])
            self.run_combo.blockSignals(False)
            if len(runs) == 1 and not self._current_run_id:
                self._current_run_id = runs[0]["segmentation_run_id"]
            if len(runs) > 1 and not self._current_run_id:
                self.recording_list.clear()
                self.identity_label.setText("Choose a segmentation run to review")
                return
            if self._current_run_id:
                self.run_combo.blockSignals(True)
                self.run_combo.setCurrentIndex(self.run_combo.findData(self._current_run_id))
                self.run_combo.blockSignals(False)
                source = select_segmentation_run(root, self._current_run_id)
                review_stage = root / "acoustic" / "003_segmentation_review" / "runs" / f"review_{self._current_run_id}"
            else:
                source = root / "acoustic" / "002_segmentation" / "tables" / "acoustic_segmentation_summary.csv"
                review_stage = root / "acoustic" / "003_segmentation_review" / "runs" / "review_default"
            initialize_segmentation_review(source, root)
            self._decisions, self._overrides = load_review_state(root)
            self._exclusions = load_review_exclusions(root)
            frozen_path = root / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_decisions.csv"
            frozen = frozen_path.exists()
            if frozen:
                candidate = pd.read_csv(frozen_path, dtype=str, keep_default_na=False)
                if not candidate.empty and candidate.review_run_id.eq(review_stage.name).all():
                    self._decisions = candidate
                else:
                    frozen = False
            self.finalize_button.setEnabled(not frozen)
            self.continue_button.setEnabled(frozen)
            required = self._decisions.review_required.astype(str).str.lower().isin(["true", "1", "yes"])
            pending = required & self._decisions.final_decision.isin(["", "PENDING"])
            self.progress_label.setText(f"Required: {int(required.sum() - pending.sum())}/{int(required.sum())}")
            self._filter_rows()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Manual review", str(exc))

    def _select_run(self) -> None:
        if self._dirty and not self._persist_draft("SAVE_DRAFT"):
            return
        run_id = self.run_combo.currentData()
        if run_id:
            self._current_run_id = str(run_id)
            self._current_id = ""
            self.refresh()

    def _filter_rows(self) -> None:
        if self._dirty and not self._persist_draft("SAVE_DRAFT"):
            return
        if self._decisions.empty:
            return
        chosen, frame = self._current_id, self._decisions.copy()
        state = self.filter_combo.currentText()
        required = frame.review_required.astype(str).str.lower().isin(["true", "1", "yes"])
        if state == "Pending review":
            frame = frame.loc[required & frame.final_decision.isin(["", "PENDING"])]
        elif state == "Review required":
            frame = frame.loc[required]
        elif state == "Excluded automatically":
            frame = frame.loc[frame.automatic_status.eq("EXCLUDED")]
        elif state == "Accepted automatically":
            frame = frame.loc[frame.automatic_status.eq("ACCEPTED") & frame.reviewer.eq("")]
        elif state == "Manually reviewed":
            frame = frame.loc[frame.reviewer.ne("")]
        elif state == "Kept automatic after review":
            frame = frame.loc[frame.final_decision.eq("KEEP_AUTO") & frame.reviewer.ne("")]
        elif state == "Kept manual":
            frame = frame.loc[frame.final_decision.eq("KEEP_MANUAL")]
        elif state == "Excluded":
            frame = frame.loc[frame.final_decision.eq("EXCLUDE")]
        query = self.search_edit.text().strip().casefold()
        if query:
            text = frame[["recording_id", "file_name", "automatic_flags"]].fillna("").agg(" ".join, axis=1).str.casefold()
            frame = frame.loc[text.str.contains(query, regex=False)]
        self.recording_list.blockSignals(True)
        self.recording_list.clear()
        for row in frame.itertuples():
            item = QListWidgetItem(f"[{row.automatic_status}] {row.file_name}")
            item.setData(Qt.UserRole, str(row.recording_id))
            self.recording_list.addItem(item)
        self.recording_list.blockSignals(False)
        target = next((i for i in range(self.recording_list.count())
                       if self.recording_list.item(i).data(Qt.UserRole) == chosen), 0)
        if self.recording_list.count():
            self.recording_list.setCurrentRow(target)
            self._show_selected(self.recording_list.currentItem())
        else:
            self._current_id = ""
            self.identity_label.setText("No recordings in this filter")
            self.editor.hide()

    def _show_selected(self, item, _previous=None) -> None:
        if item is None:
            return
        if self._dirty and _previous is not None and item is not _previous:
            choice = QMessageBox.question(self, "Unsaved review", "Save changes before leaving this recording?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if choice == QMessageBox.Cancel or (choice == QMessageBox.Save and not self._persist_draft("SAVE_DRAFT")):
                self.recording_list.blockSignals(True)
                self.recording_list.setCurrentItem(_previous)
                self.recording_list.blockSignals(False)
                return
            self._set_dirty(False)
        self._loading = True
        self._stop()
        self._current_id = str(item.data(Qt.UserRole))
        row = self._decisions.loc[self._decisions.recording_id.eq(self._current_id)].iloc[0]
        self.identity_label.setText(f"{row.file_name}  ·  {row.recording_id}")
        self.status_label.setText(f"{row.segmentation_method}  |  Automatic: {row.automatic_status}  |  Final: {row.final_decision or 'Pending'}")
        self.flags_label.setText(f"Flags: {row.automatic_flags or 'None'}")
        self.reviewer_edit.setText(row.reviewer)
        self.notes_edit.setPlainText(row.review_notes)
        if row.recording_exclusion_reason:
            self.recording_reason_combo.setCurrentText(row.recording_exclusion_reason)
        elif "preprocess_not_accepted" in row.automatic_flags:
            self.recording_reason_combo.setCurrentText("Preprocess not accepted")
        elif row.automatic_status == "FAILED":
            self.recording_reason_combo.setCurrentText("Unusable audio")
        saved = self._overrides.loc[self._overrides.recording_id.eq(self._current_id)]
        automatic = _speech_intervals(row.automatic_segments_path)
        reviewed = ([(float(r.start_sec), float(r.end_sec)) for r in saved.sort_values("segment_index").itertuples()]
                    if not saved.empty else automatic)
        excluded = self._exclusions.loc[self._exclusions.recording_id.eq(self._current_id)]
        excluded_items = [{"start_sec": float(r.start_sec), "end_sec": float(r.end_sec),
                           "exclusion_reason": r.exclusion_reason, "notes": r.notes}
                          for r in excluded.itertuples()]
        wav = Path(row.analysis_wav_path) if row.analysis_wav_path else Path("__missing__")
        available = wav.is_file() and row.automatic_status != "FAILED"
        self.editor.setVisible(available)
        if available:
            try:
                raw_frame_path = row.frame_csv_path or row.automatic_frames_path
                frame_path = Path(raw_frame_path) if raw_frame_path else None
                duration = float(row.duration_sec)
                start, end = float(row.analysis_start_sec or 0), float(row.analysis_end_sec or duration)
                self.editor.load_recording(wav, frame_path, automatic, reviewed, (start, end), excluded_items)
                self._player.setSource(QUrl.fromLocalFile(str(wav.resolve())))
                self.time_label.setText(f"00:00.00 / {_clock(self.editor.duration_sec)}")
            except Exception:
                available = False
                self.editor.hide()
        self.availability_label.setText("" if available else
            "Audio unavailable for boundary review. Recording-level disposition is still available.")
        self._undo.clear()
        self.edit_controls.hide()
        frozen = not self.finalize_button.isEnabled()
        for button in (self.keep_button, self.edit_button, self.save_manual_button):
            button.setEnabled(available and not frozen)
        self.exclude_button.setEnabled(not frozen)
        for button in (self.play_button, self.stop_button, self.selection_play_button, self.back_button):
            button.setEnabled(available)
        self.reviewer_edit.setReadOnly(frozen)
        self.notes_edit.setReadOnly(frozen)
        self.recording_reason_combo.setEnabled(not frozen)
        self._sync_advanced_text()
        self._loading = False
        self._set_dirty(False)

    def _seek(self, seconds: float) -> None:
        if self._current_id:
            self.editor.set_cursor(seconds)
            self._player.setPosition(round(self.editor.cursor_sec * 1000))
            self.time_label.setText(f"{_clock(self.editor.cursor_sec)} / {_clock(self.editor.duration_sec)}")

    def _position_changed(self, milliseconds: int) -> None:
        if self._selection_stop_ms is not None and milliseconds >= self._selection_stop_ms:
            self._player.pause()
            self._selection_stop_ms = None
            self._selection_timer.stop()
        self.editor.set_cursor(milliseconds / 1000)
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self.editor.follow_cursor()
        self.time_label.setText(f"{_clock(self.editor.cursor_sec)} / {_clock(self.editor.duration_sec)}")

    def _play_pause(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
            return
        self._selection_stop_ms = None
        self._selection_timer.stop()
        start, _ = playback_bounds(self.editor.cursor_sec, self.editor.duration_sec)
        self._player.setPosition(start)
        self._player.play()

    def _play_selection(self) -> None:
        if not self.editor.selection:
            QMessageBox.information(self, "Play selection", "Drag a time range on the waveform first.")
            return
        start, end = playback_bounds(self.editor.cursor_sec, self.editor.duration_sec, self.editor.selection)
        self._selection_stop_ms = end
        self._selection_timer.start()
        self._player.setPosition(start)
        self._player.play()

    def _stop(self) -> None:
        self._selection_stop_ms = None
        self._selection_timer.stop()
        self._player.stop()

    def _snapshot(self, intervals=None) -> None:
        self._undo.append((intervals if intervals is not None else self.editor.intervals(), self.editor.analysis_start_sec,
                           self.editor.analysis_end_sec, [dict(item) for item in self.editor.exclusions]))

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        self.saved_state_label.setText("Unsaved changes" if dirty else "Saved")

    def _mark_dirty(self, *_args) -> None:
        if not self._loading and self._current_id:
            self._set_dirty(True)

    def _boundary_changed(self) -> None:
        self._sync_advanced_text()
        self._mark_dirty()
        if not self._loading and not self._in_apply:
            self._persist_draft("EDIT_BOUNDARIES")

    def _persist_draft(self, action: str) -> bool:
        if not self._current_id:
            return False
        try:
            save_segmentation_review_entry(
                self._output_root(), self._current_id, "PENDING",
                self.reviewer_edit.text(), self.notes_edit.toPlainText(),
                self.manual_edit.toPlainText() if self.editor.isVisible() else "",
                analysis_start_sec=self.editor.analysis_start_sec if self.editor.isVisible() else None,
                analysis_end_sec=self.editor.analysis_end_sec if self.editor.isVisible() else None,
                exclusion_intervals=self.editor.exclusions if self.editor.isVisible() else [], action=action)
            self._set_dirty(False)
            return True
        except Exception as exc:  # noqa: BLE001
            self._set_dirty(True)
            QMessageBox.warning(self, "Review not saved", str(exc))
            return False

    def _apply(self, action, action_name: str = "EDIT_BOUNDARIES") -> None:
        self._snapshot()
        self._in_apply = True
        try:
            action()
        except ValueError as exc:
            self._undo.pop()
            QMessageBox.warning(self, "Review edit", str(exc))
            return
        finally:
            self._in_apply = False
        self._sync_advanced_text()
        self._mark_dirty()
        self._persist_draft(action_name)

    def _enter_edit_mode(self) -> None:
        self.editor.set_editing(True)
        self.edit_controls.show()

    def _add_interval(self) -> None:
        if self.editor.selection:
            self._apply(lambda: self.editor.add_interval(*self.editor.selection), "ADD_INTERVAL")
        else:
            QMessageBox.information(self, "Add speech", "Drag a time range on the waveform first.")

    def _exclude_selection(self) -> None:
        target = self.editor.selection or self.editor.selected_interval()
        if not target:
            QMessageBox.information(self, "Exclude interval", "Click a detected segment or drag a time range first.")
            return
        start, end = target
        self._apply(lambda: self.editor.set_exclusions([*self.editor.exclusions, {
            "start_sec": start, "end_sec": end, "exclusion_reason": self.reason_combo.currentText(),
            "notes": self.notes_edit.toPlainText().strip()}]), "EXCLUDE_INTERVAL")

    def _undo_edit(self) -> None:
        if self._undo:
            intervals, start, end, exclusions = self._undo.pop()
            self.editor.set_intervals(intervals)
            self.editor.set_analysis_window(start, end)
            self.editor.set_exclusions(exclusions)
            self._sync_advanced_text()
            self._persist_draft("UNDO")

    def _reset_automatic(self) -> None:
        def reset():
            self.editor.set_intervals(self.editor.automatic_intervals)
            self.editor.set_analysis_window(0, self.editor.duration_sec)
            self.editor.set_exclusions([])
        self._apply(reset, "RESET_TO_AUTO")

    def _sync_advanced_text(self) -> None:
        self.manual_edit.blockSignals(True)
        self.manual_edit.setPlainText("\n".join(f"{a:.6f},{b:.6f}" for a, b in self.editor.intervals()))
        self.manual_edit.blockSignals(False)

    def _advanced_changed(self) -> None:
        if self.edit_controls.isVisible():
            try:
                intervals = validate_manual_interval_list(
                    parse_manual_intervals_text(self.manual_edit.toPlainText()),
                    duration_sec=self.editor.duration_sec)
                self.editor.set_intervals(intervals)
                self._mark_dirty()
            except ValueError:
                pass  # Incomplete text is checked again on Save.

    def _save(self, decision: str) -> None:
        if not self._current_id:
            return
        try:
            if decision == "KEEP_MANUAL":
                intervals = validate_manual_interval_list(
                    parse_manual_intervals_text(self.manual_edit.toPlainText()),
                    duration_sec=self.editor.duration_sec)
                self.editor.set_intervals(intervals)
            if decision == "EXCLUDE":
                start, end, exclusions = 0.0, None, []
            else:
                start, end = self.editor.analysis_start_sec, self.editor.analysis_end_sec
                exclusions = self.editor.exclusions
            save_segmentation_review_entry(self._output_root(), self._current_id, decision,
                self.reviewer_edit.text(), self.notes_edit.toPlainText(), self.manual_edit.toPlainText(),
                analysis_start_sec=start, analysis_end_sec=end, exclusion_intervals=exclusions,
                recording_exclusion_reason=self.recording_reason_combo.currentText())
            self._set_dirty(False)
            self.refresh()
        except Exception as exc:  # noqa: BLE001
            self._set_dirty(True)
            QMessageBox.warning(self, "Review not saved", str(exc))

    def _navigate(self, offset: int) -> None:
        if self._dirty:
            choice = QMessageBox.question(self, "Unsaved review", "Save changes before leaving this recording?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if choice == QMessageBox.Cancel:
                return
            if choice == QMessageBox.Save and not self._persist_draft("SAVE_DRAFT"):
                return
            if choice == QMessageBox.Discard:
                self._set_dirty(False)
        index = self.recording_list.currentRow() + offset
        if 0 <= index < self.recording_list.count():
            self.recording_list.setCurrentRow(index)

    def _finalize(self) -> None:
        try:
            result = finalize_segmentation_review(self._output_root())
            self.refresh()
            self.finalized.emit(result)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Review incomplete", str(exc))

    def keyPressEvent(self, event) -> None:
        super().keyPressEvent(event)

    def _space_pressed(self) -> None:
        if not any(widget.hasFocus() for widget in (self.manual_edit, self.notes_edit,
                                                     self.reviewer_edit, self.search_edit)):
            self._play_pause()
