"""Researcher-facing review controls around the existing alignment editor."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pyqtgraph as pg
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout,
)

from vslp.acoustic.alignment.corrections import load_corrections
from vslp.acoustic.alignment.trial_review import set_trial_review, trial_review_table
from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget


ISSUE_CATEGORIES = (
    "Word boundary", "Phone boundary", "Missing repetition", "Extra repetition",
    "False start", "Transcript mismatch", "Background / overlapping speech",
    "Noise / artifact", "Other",
)


class AlignmentReviewEditor(AlignmentWidget):
    """Keep the existing alignment renderer while simplifying its visible workflow."""

    def __init__(self, root_provider: Callable[[], Path]) -> None:
        self._editor_ready = False
        super().__init__(root_provider)
        # The standalone Align screen owns preflight and execution.  Automatic
        # preflight from the legacy embedded widget can open a modal while a
        # researcher is simply moving through the review queue.
        self._preflight_timer.stop()
        self._preflight_timer.timeout.disconnect()
        self._selection_start: float | None = None
        self._selected_interval: tuple[float, float] | None = None
        self._selection_overlays: list[pg.LinearRegionItem] = []
        self._playback_speed = 1.0
        self._tier_cursors = []
        for track in (self.word_plot, self.phone_plot):
            line = pg.InfiniteLine(angle=90, movable=False,
                                   pen=pg.mkPen("#ECA35D", width=2))
            track.addItem(line)
            self._tier_cursors.append(line)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.setMinimumHeight(250)
        self.plot.setMaximumHeight(360)
        self._build_editor_controls()
        self._editor_ready = True
        self._show_step()

    def _build_editor_controls(self) -> None:
        layout = self.layout()
        self.queue = QComboBox()
        for name in ("All", "Needs Review", "Corrected", "Failed", "Reviewed"):
            self.queue.addItem(name)
        self.queue.currentIndexChanged.connect(self._apply_queue)
        self.previous_flagged = QPushButton("← Previous flagged")
        self.next_flagged = QPushButton("Next flagged →")
        self.previous_flagged.clicked.connect(lambda: self._step_flagged(-1))
        self.next_flagged.clicked.connect(lambda: self._step_flagged(1))
        self.editor_heading = QLabel("Select a recording to review")
        self.editor_heading.setObjectName("AppTitleLabel")
        queue_row = QHBoxLayout()
        for item in (self.previous_flagged, self.editor_heading, self.next_flagged,
                     QLabel("Queue"), self.queue):
            queue_row.addWidget(item)
        layout.insertLayout(0, queue_row)

        self.play_button.setText("▶ Play / Pause")
        self.play_trial_button = QPushButton("▶ Play trial")
        self.play_trial_button.clicked.connect(self._play_trial)
        self.play_sentence_button = QPushButton("▶ Play sentence")
        self.play_sentence_button.clicked.connect(self._play_sentence)
        self.play_visible_button.setText("▶ Play selection")
        self.play_visible_button.clicked.disconnect()
        self.play_visible_button.clicked.connect(self._play_selection)
        self.zoom_out = QPushButton("−")
        self.zoom_in = QPushButton("+")
        self.zoom_fit = QPushButton("Fit")
        self.zoom_out.clicked.connect(lambda: self._zoom(1.5))
        self.zoom_in.clicked.connect(lambda: self._zoom(1 / 1.5))
        self.zoom_fit.clicked.connect(self._fit_full_recording)
        playback = QGroupBox("Playback")
        playback_layout = QVBoxLayout(playback)
        self.active_trial_label = QLabel("Select a trial")
        playback_layout.addWidget(self.active_trial_label)
        strip = QHBoxLayout()
        for item in (self.play_button, self.play_trial_button, self.play_sentence_button,
                     self.play_visible_button):
            strip.addWidget(item)
        strip.addStretch()
        playback_layout.addLayout(strip)
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed"))
        self.speed_group = QButtonGroup(self)
        self.speed_group.setExclusive(True)
        self.speed_buttons = {}
        player = self._ensure_player()
        pitch_available = (hasattr(player, "pitchCompensationAvailability") and
                           player.pitchCompensationAvailability() !=
                           QMediaPlayer.PitchCompensationAvailability.Unavailable)
        if pitch_available:
            player.setPitchCompensation(True)
        self._pitch_slow_available = pitch_available
        for rate in (0.5, 0.75, 1.0):
            button = QPushButton(f"{rate:g}×")
            button.setCheckable(True)
            button.setStyleSheet(
                "QPushButton:checked { background-color: #0E7664; "
                "border: 2px solid #68E0C5; font-weight: 800; }")
            button.setChecked(rate == 1.0)
            button.setEnabled(rate == 1.0 or pitch_available)
            if rate != 1.0 and not pitch_available:
                button.setToolTip("Pitch-preserving slow playback not yet available")
            button.clicked.connect(lambda _checked=False, value=rate: self._set_speed(value))
            self.speed_group.addButton(button)
            self.speed_buttons[rate] = button
            speed_row.addWidget(button)
        speed_row.addSpacing(20)
        for item in (QLabel("Zoom"), self.zoom_out, self.zoom_fit, self.zoom_in):
            speed_row.addWidget(item)
        speed_row.addStretch()
        playback_layout.addLayout(speed_row)
        layout.insertWidget(2, playback)
        self._sync_playback_controls()

        self.accept_button.setText("✓ Accept / Mark reviewed")
        self.flag_button = QPushButton("⚑ Flag issue")
        self.flag_button.clicked.connect(self._flag_issue)
        self.exclude_button = QPushButton("Exclude")
        self.exclude_button.clicked.connect(lambda: self._record_issue("Excluded", ""))
        self.finalize_button = QPushButton("Finalize reviewed alignment")
        self.finalize_button.clicked.connect(
            lambda: self.freeze_requested.emit(self.runs.currentData() or ""))
        primary_actions = QHBoxLayout()
        for item in (self.accept_button, self.flag_button, self.exclude_button):
            primary_actions.addWidget(item)
        layout.insertLayout(3, primary_actions)
        secondary_actions = QHBoxLayout()
        for item in (self.edit_alignment_button, self.finalize_button):
            secondary_actions.addWidget(item)
        layout.insertLayout(4, secondary_actions)

        shortcuts = {
            "Space": self._play_pause, "Shift+Space": self._play_selection,
            "A": lambda: self._set_review("ACCEPTED"), "F": self._flag_issue,
            "N": lambda: self._step_flagged(1), "P": lambda: self._step_flagged(-1),
            "+": lambda: self._zoom(1 / 1.5), "-": lambda: self._zoom(1.5),
            "0": self._fit_full_recording,
        }
        self._editor_shortcuts = []
        for key, callback in shortcuts.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._editor_shortcuts.append(shortcut)

    def _show_step(self) -> None:
        super()._show_step()
        if not getattr(self, "_editor_ready", False):
            return
        for widget in (
            self.steps, self.action_hint, self.primary, self.next_action,
            self.task_brief, self.readiness, self.normal_summary, self.summary,
            self.review_header,
            self.keep_needs_review_button, self.needs_review_button,
            self.previous_issue, self.next_issue, self.view_auto_passed,
            self.play_token_button, self.edit_trial_button, self.reset_token_button,
            self.reset_trial_button, self.show_technical_button,
        ):
            widget.hide()
        planning = not bool(self.runs.currentData())
        self.trial_count.setVisible(planning)
        self.proposal_strip.setVisible(planning and bool(self.proposal_regions))
        self.trial_actions.setVisible(planning)
        self.primary.setVisible(planning)
        self.primary.setText("Confirm repetitions")
        self.accept_button.setVisible(bool(self.runs.currentData()))
        self.flag_button.setVisible(bool(self.runs.currentData()))
        self.exclude_button.setVisible(bool(self.runs.currentData()))
        self.edit_alignment_button.setVisible(bool(self.runs.currentData()))
        self.finalize_button.setVisible(bool(self.runs.currentData()))
        self.finalize_button.setEnabled(self.freeze_button.isEnabled())
        self.play_sentence_button.setVisible(self.sentence_selector.count() > 0)
        self._update_active_trial_label()
        self._sync_playback_controls()
        self._update_editor_heading()

    def _sync_playback_controls(self) -> None:
        if not getattr(self, "speed_buttons", None):
            return
        has_audio = bool(self.player is not None and not self.player.source().isEmpty())
        has_trial = bool(self.trials.currentData())
        self.play_button.setEnabled(has_audio and has_trial)
        self.play_trial_button.setEnabled(has_audio and has_trial)
        self.play_visible_button.setEnabled(has_audio)
        self.play_sentence_button.setEnabled(has_audio and self.sentence_selector.count() > 0)
        for rate, button in self.speed_buttons.items():
            button.setEnabled(has_audio and (rate == 1.0 or self._pitch_slow_available))

    def _update_active_trial_label(self) -> None:
        if not getattr(self, "_editor_ready", False):
            return
        trial = self.trials.currentData() or {}
        if not trial:
            self.active_trial_label.setText("Select a trial")
            return
        number = self.trials.currentIndex() + 1
        bounds = (f"{float(trial['start_sec']):.2f}–"
                  f"{float(trial['end_sec']):.2f} s")
        task = self.task_label.text()
        description = (f" · {task} — {self.prompt_display.text()}"
                       if task and "Bamboo" not in task else "")
        self.active_trial_label.setText(f"Trial {number} · {bounds}{description}")

    def _trial_changed(self) -> None:
        super()._trial_changed()
        self._update_active_trial_label()

    def _set_speed(self, rate: float) -> None:
        button = self.speed_buttons[rate]
        if not button.isEnabled():
            return
        self._playback_speed = rate
        player = self._ensure_player()
        player.setPlaybackRate(rate)

    def _ensure_player(self) -> QMediaPlayer:
        player = super()._ensure_player()
        player.setPlaybackRate(getattr(self, "_playback_speed", 1.0))
        return player

    def _play_bounds(self, start: float, end: float) -> None:
        if end <= start:
            return
        player = self._ensure_player()
        player.setPosition(round(start * 1000))
        self._play_end_ms = round(end * 1000)
        player.play()
        self.play_button.setText("Pause")

    def _play_trial(self) -> None:
        trial = self.trials.currentData() or {}
        if trial:
            self._play_bounds(float(trial["start_sec"]), float(trial["end_sec"]))

    def _play_sentence(self) -> None:
        bounds = self.sentence_selector.currentData()
        if bounds:
            self._play_bounds(float(bounds[0]), float(bounds[1]))

    def _update_editor_heading(self) -> None:
        if not getattr(self, "_editor_ready", False):
            return
        trial = self.trials.currentData() or {}
        current = self.recordings.currentData()
        status = self._display_status(str(current), str(trial.get("trial_id", "")))
        self.editor_heading.setText(
            f"Recording {self.recordings.currentIndex() + 1} / {self.recordings.count()}"
            f"  ·  {self.task_label.text()}  ·  Trial {self.trials.currentIndex() + 1}"
            f"  ·  {status}")

    def _display_status(self, recording_id: str, trial_id: str,
                        table=None, corrections=None) -> str:
        run_id = self.runs.currentData()
        if not run_id or not trial_id:
            return "NEEDS REVIEW"
        if table is None:
            table = trial_review_table(self.root_provider(), str(run_id))
        matched = table.loc[table.recording_id.astype(str).eq(recording_id)
                            & table.trial_id.astype(str).eq(trial_id)]
        if matched.empty:
            return "FAILED"
        row = matched.iloc[0]
        if (str(row.get("gui_issue_category", "")) == "Excluded"
                and str(row.review_status) != "ACCEPTED"):
            return "EXCLUDED"
        if str(row.review_status) == "ACCEPTED":
            if corrections is None:
                corrections = load_corrections(self.root_provider(), str(run_id))
            corrected = corrections.loc[
                corrections.recording_id.astype(str).eq(recording_id)
                & corrections.trial_id.astype(str).eq(trial_id)]
            return "CORRECTED" if not corrected.empty else "REVIEWED"
        if str(row.review_status) == "AUTO_ACCEPTED_STRUCTURAL":
            return "AUTO OK"
        return "NEEDS REVIEW"

    def _apply_queue(self) -> None:
        if not getattr(self, "_editor_ready", False):
            return
        wanted = self.queue.currentText()
        previous = self.recordings.currentData()
        context = getattr(self, "_resolved_context", None) or {}
        records = context.get("records", [])
        run_id = self.runs.currentData()
        table = trial_review_table(self.root_provider(), str(run_id)) if run_id else None
        corrections = load_corrections(self.root_provider(), str(run_id)) if run_id else None
        self.recordings.blockSignals(True)
        self.recordings.clear()
        for record in records:
            identity = str(record["recording_id"])
            trials = (context.get("choices", {}).get("trials", {}).get(identity, []))
            states = {self._display_status(identity, str(t.get("trial_id", "")),
                                           table, corrections)
                      for t in trials} or {"NEEDS REVIEW"}
            if wanted == "All" or (wanted == "Needs Review" and "NEEDS REVIEW" in states) or (
                    wanted == "Corrected" and "CORRECTED" in states) or (
                    wanted == "Failed" and "FAILED" in states) or (
                    wanted == "Reviewed" and states & {"REVIEWED", "AUTO OK"}):
                self.recordings.addItem(record["file_name"], identity)
        index = self.recordings.findData(previous)
        if self.recordings.count():
            self.recordings.setCurrentIndex(max(index, 0))
        self.recordings.blockSignals(False)
        self._recording_changed()

    def refresh(self) -> None:
        # The embedded widget refreshes once while populating runs and again
        # while populating recordings. Defer rendering until the queue is ready.
        self._building_review_queue = True
        try:
            super().refresh()
        finally:
            self._building_review_queue = False
        if getattr(self, "_editor_ready", False):
            self._apply_queue()

    def _step_flagged(self, direction: int) -> None:
        if self.queue.currentText() != "Needs Review":
            self.queue.setCurrentText("Needs Review")
        next_index = self.recordings.currentIndex() + direction
        if 0 <= next_index < self.recordings.count():
            self.recordings.setCurrentIndex(next_index)

    def _flag_issue(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Flag alignment issue")
        form = QFormLayout(dialog)
        category = QComboBox()
        category.addItems(ISSUE_CATEGORIES)
        note = QLineEdit()
        note.setPlaceholderText("Optional note")
        form.addRow("Issue", category)
        form.addRow("Note", note)
        confirm = QPushButton("Save issue")
        confirm.clicked.connect(dialog.accept)
        form.addRow(confirm)
        if dialog.exec() == QDialog.Accepted:
            self._record_issue(category.currentText(), note.text().strip())

    def _record_issue(self, category: str, note: str) -> None:
        trial = self.trials.currentData() or {}
        run_id = self.runs.currentData()
        recording_id = self.recordings.currentData()
        trial_id = str(trial.get("trial_id", ""))
        if not (run_id and recording_id and trial_id):
            return
        set_trial_review(self.root_provider(), str(run_id), str(recording_id), trial_id,
                         "NEEDS_REVIEW")
        table = trial_review_table(self.root_provider(), str(run_id))
        mask = (table.recording_id.astype(str).eq(str(recording_id))
                & table.trial_id.astype(str).eq(trial_id))
        for column, value in (("gui_issue_category", category),
                              ("gui_issue_note", note), ("gui_reviewer_action", "FLAGGED")):
            if column not in table:
                table[column] = ""
            table.loc[mask, column] = value
        path = (self.root_provider() / "acoustic" / "004_alignment" / "runs" /
                str(run_id) / "tables" / "alignment_trial_review.csv")
        table.to_csv(path, index=False)
        self._review_cache = None
        self._trial_changed()
        self._apply_queue()

    def _set_review(self, status: str) -> None:
        super()._set_review(status)
        if getattr(self, "_editor_ready", False):
            self._apply_queue()

    def _save_token_region(self, tier: str, index: int, trial_id: str,
                           region: pg.LinearRegionItem) -> None:
        self._standalone_review_payload = None
        super()._save_token_region(tier, index, trial_id, region)

    def _reset_selected_token(self) -> None:
        self._standalone_review_payload = None
        super()._reset_selected_token()

    def _reset_current_trial(self) -> None:
        self._standalone_review_payload = None
        super()._reset_current_trial()

    def _zoom(self, factor: float) -> None:
        left, right = self.plot.viewRange()[0]
        center = (left + right) / 2
        width = max(0.01, (right - left) * factor)
        self.plot.setXRange(center - width / 2, center + width / 2, padding=0)

    def _play_selection(self) -> None:
        if self._selected_interval:
            start, end = self._selected_interval
            self._play_bounds(start, end)
        elif self._selected_token:
            self._play_token()

    def _set_selection(self, start: float, end: float) -> None:
        self._selected_interval = tuple(sorted((start, end)))
        for track, overlay in zip((self.plot, self.word_plot, self.phone_plot),
                                  self._selection_overlays):
            track.removeItem(overlay)
        self._selection_overlays.clear()
        for track in (self.plot, self.word_plot, self.phone_plot):
            overlay = pg.LinearRegionItem(self._selected_interval,
                brush=pg.mkBrush("#5A91D944"), movable=False)
            overlay.setAcceptedMouseButtons(Qt.NoButton)
            track.addItem(overlay)
            self._selection_overlays.append(overlay)

    def _token_clicked(self, event, tier: str | None = None) -> None:
        super()._token_clicked(event, tier)
        plot = self.word_plot if tier == "Word" else self.phone_plot if tier == "Phone" else self.plot
        if plot.sceneBoundingRect().contains(event.scenePos()):
            position = plot.plotItem.vb.mapSceneToView(event.scenePos()).x()
            self.play_cursor.setPos(position)
            for line in self._tier_cursors:
                line.setPos(position)
            if event.double() and tier and self._selected_token:
                self.plot.setXRange(*self._selected_token, padding=.12)

    def _playback_position(self, position_ms: int) -> None:
        super()._playback_position(position_ms)
        for line in getattr(self, "_tier_cursors", []):
            line.setPos(position_ms / 1000)

    def _recording_changed(self) -> None:
        if getattr(self, "_building_review_queue", False):
            return
        self._selected_interval = None
        self._selection_overlays = []
        super()._recording_changed()
        for track, line in zip((self.word_plot, self.phone_plot),
                               getattr(self, "_tier_cursors", [])):
            track.addItem(line)
        self._update_editor_heading()
        self._sync_playback_controls()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        if (getattr(self, "_editor_ready", False) and watched is self.plot.viewport()
                and not self._editing_alignment and not self._add_mode):
            if event.type() in {QEvent.MouseButtonPress, QEvent.MouseMove,
                                QEvent.MouseButtonRelease}:
                if event.type() != QEvent.MouseMove and event.button() != Qt.LeftButton:
                    return super().eventFilter(watched, event)
                scene = self.plot.mapToScene(event.position().toPoint())
                if not self.plot.plotItem.vb.sceneBoundingRect().contains(scene):
                    return super().eventFilter(watched, event)
                position = self.plot.plotItem.vb.mapSceneToView(scene).x()
                if event.type() == QEvent.MouseButtonPress:
                    self._selection_start = position
                    return True
                if event.type() == QEvent.MouseButtonRelease and self._selection_start is not None:
                    start = self._selection_start
                    self._selection_start = None
                    if abs(position - start) >= .01:
                        self._set_selection(start, position)
                    else:
                        self.play_cursor.setPos(position)
                        for line in self._tier_cursors:
                            line.setPos(position)
                    return True
                if event.type() == QEvent.MouseMove and self._selection_start is not None:
                    return True
        return super().eventFilter(watched, event)
