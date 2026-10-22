"""Lazy, linked waveform/word/phone review for standalone Alignment jobs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)
import soundfile as sf

from vslp.acoustic.alignment.workbench import qc_job, set_alignment_scope
from vslp.acoustic.alignment.workbench import load_job
from vslp.acoustic.alignment.workbench_review import (
    load_corrections, reviewed_tokens, save_corrections,
)


def load_recording_review(job_root: Path, record: dict) -> dict:
    """Decode only the active recording on a worker; return a compact plot."""
    source = Path(record["source_path"])
    with sf.SoundFile(source) as stream:
        audio = stream.read(dtype="float32", always_2d=True).mean(axis=1)
        rate = stream.samplerate
    stride = max(1, len(audio) // 6000)
    display = audio[::stride]
    times = np.arange(len(display), dtype=np.float32) * stride / rate
    words, phones = reviewed_tokens(job_root, record["record_key"])
    spec = load_job(job_root)["task_specification"]
    expected = spec["alignment_transcript"].split()
    actual = [str(label).upper() for label in words.label]
    trials = []
    if actual == expected:
        per_trial = len(spec["normalized_transcript"].split())
        for index in range(spec["expected_repetitions"]):
            group = words.iloc[index * per_trial:(index + 1) * per_trial]
            trials.append({"label": f"Trial {index + 1}",
                           "start_sec": float(group.start_sec.iloc[0]),
                           "end_sec": float(group.end_sec.iloc[-1])})
    elif spec["expected_repetitions"] == 1:
        trials.append({"label": "Trial 1 · structure needs review",
                       "start_sec": 0.0, "end_sec": len(audio) / rate})
    return {"source": str(source), "duration": len(audio) / rate,
            "times": times, "waveform": display, "words": words.to_dict("records"),
            "phones": phones.to_dict("records"),
            "trials": trials,
            "alignment_scope": record.get("alignment_scope", {
                "start_sec": 0.0, "end_sec": len(audio) / rate,
                "source": "legacy_full_recording"}),
            "corrections": load_corrections(job_root, record["record_key"])}


class WorkbenchReview(QWidget):
    changed = Signal()
    realign_requested = Signal(str, str, str)

    def __init__(self) -> None:
        super().__init__()
        self.job_root: Path | None = None
        self.record: dict | None = None
        self.payload: dict | None = None
        self._selected_word: int | None = None
        self._selected_phone: int | None = None
        self._play_end_ms: int | None = None
        self._speed = 1.0
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self._track_labels: list[tuple[pg.TextItem, float, int]] = []
        self._drag_lines: list[pg.InfiniteLine] = []
        self.player = QMediaPlayer(self)
        self.output = QAudioOutput(self)
        self.player.setAudioOutput(self.output)
        self.player.positionChanged.connect(self._position_changed)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        self.header = QLabel("Select a recording to review")
        self.header.setObjectName("SectionTitleLabel")
        self.header.setWordWrap(True)
        layout.addWidget(self.header)
        self.scope_label = QLabel("Alignment scope: —")
        self.scope_label.setWordWrap(True)
        layout.addWidget(self.scope_label)
        scope_row = QHBoxLayout()
        self.scope_start = QDoubleSpinBox()
        self.scope_end = QDoubleSpinBox()
        for control in (self.scope_start, self.scope_end):
            control.setDecimals(3)
            control.setRange(0, 86400)
            control.setSuffix(" s")
        scope_row.addWidget(QLabel("Start"))
        scope_row.addWidget(self.scope_start)
        scope_row.addWidget(QLabel("End"))
        scope_row.addWidget(self.scope_end)
        for label, handler in (("Play scope", self._play_scope),
                               ("Set from selection", self._scope_from_selection),
                               ("Edit scope", self._save_scope)):
            button = QPushButton(label)
            button.clicked.connect(handler)
            scope_row.addWidget(button)
        layout.addLayout(scope_row)
        self.word_plot = pg.PlotWidget(title="WORDS")
        self.phone_plot = pg.PlotWidget(title="PHONES")
        self.wave_plot = pg.PlotWidget(title="WAVEFORM")
        self.word_plot.setXLink(self.wave_plot)
        self.phone_plot.setXLink(self.wave_plot)
        self.wave_plot.getPlotItem().vb.sigXRangeChanged.connect(self._update_label_visibility)
        self.wave_plot.setMinimumHeight(135)
        self.word_plot.setMinimumHeight(80)
        self.phone_plot.setMinimumHeight(80)
        self.wave_plot.setMaximumHeight(165)
        self.word_plot.setMaximumHeight(90)
        self.phone_plot.setMaximumHeight(90)
        for plot in (self.wave_plot, self.word_plot, self.phone_plot):
            plot.showGrid(x=True, y=False, alpha=.16)
            plot.getPlotItem().setClipToView(True)
            plot.setMouseEnabled(x=True, y=False)
        layout.addWidget(self.wave_plot)
        layout.addWidget(self.word_plot)
        layout.addWidget(self.phone_plot)
        self.selection = pg.LinearRegionItem(values=(0, 0), movable=True,
                                              brush=(20, 160, 180, 45))
        self.wave_plot.addItem(self.selection)
        self.track_selections = []
        for plot in (self.word_plot, self.phone_plot):
            region = pg.LinearRegionItem(values=(0, 0), movable=False,
                                         brush=(20, 160, 180, 35))
            plot.addItem(region)
            self.track_selections.append(region)
        self.selection.sigRegionChanged.connect(self._selection_changed)
        self.playheads = [pg.InfiniteLine(angle=90, movable=False,
                                         pen=pg.mkPen("#e6ad42")) for _ in range(3)]
        for plot, line in zip((self.wave_plot, self.word_plot, self.phone_plot),
                              self.playheads):
            plot.addItem(line)
        self.word_plot.scene().sigMouseClicked.connect(lambda event: self._track_click(event, "word"))
        self.phone_plot.scene().sigMouseClicked.connect(lambda event: self._track_click(event, "phone"))
        playback = QHBoxLayout()
        self.trial_selector = QComboBox()
        playback.addWidget(self.trial_selector)
        for title, handler in (("Play / Pause", self._play_pause),
                               ("Play trial", self._play_trial),
                               ("Play selection", self._play_selection)):
            button = QPushButton(title)
            button.clicked.connect(handler)
            playback.addWidget(button)
        playback.addWidget(QLabel("Speed"))
        self.speed_buttons = {}
        available = self.player.pitchCompensationAvailability()
        pitch = available == QMediaPlayer.PitchCompensationAvailability.Available
        if pitch:
            self.player.setPitchCompensation(True)
        for rate in (.5, .75, 1.0):
            button = QPushButton(f"{rate:g}×")
            button.setCheckable(True)
            button.setChecked(rate == 1.0)
            if rate < 1 and not pitch:
                button.setEnabled(False)
                button.setToolTip("Pitch-preserving slow playback not yet available")
            button.clicked.connect(lambda _, selected=rate: self._set_speed(selected))
            self.speed_buttons[rate] = button
            playback.addWidget(button)
        playback.addStretch()
        layout.addLayout(playback)
        navigation = QHBoxLayout()
        for label, handler in (("−", self._zoom_out), ("Fit", self._fit),
                               ("+", self._zoom_in)):
            button = QPushButton(label)
            button.clicked.connect(handler)
            navigation.addWidget(button)
        navigation.addWidget(QLabel("Mouse wheel zooms · drag to pan · select a word/phone for edits"))
        navigation.addStretch()
        layout.addLayout(navigation)
        editor = QFormLayout()
        self.reviewer = QLineEdit()
        self.reviewer.setPlaceholderText("Reviewer name required for corrections")
        editor.addRow("Reviewer", self.reviewer)
        self.issue = QComboBox()
        self.issue.addItems(["Word boundary", "Phone boundary", "Missing repetition",
                             "Extra repetition", "False start", "Transcript mismatch",
                             "Background speech", "Noise/artifact", "Other"])
        editor.addRow("Issue", self.issue)
        self.note = QPlainTextEdit()
        self.note.setMaximumHeight(60)
        editor.addRow("Note", self.note)
        self.canonical_transcript = QPlainTextEdit()
        self.canonical_transcript.setReadOnly(True)
        self.canonical_transcript.setMaximumHeight(75)
        editor.addRow("Canonical task transcript", self.canonical_transcript)
        self.spoken_transcript = QPlainTextEdit()
        self.spoken_transcript.setMaximumHeight(75)
        self.spoken_transcript.setToolTip(
            "Edit the words actually spoken, then re-run MFA for this recording. "
            "Lexical edits invalidate the old word and phone boundaries.")
        editor.addRow("Reviewed spoken transcript", self.spoken_transcript)
        self.realign_button = QPushButton("Re-align reviewed spoken transcript")
        self.realign_button.clicked.connect(self._request_realign)
        editor.addRow("", self.realign_button)
        bounds = QHBoxLayout()
        self.bound_start = QDoubleSpinBox()
        self.bound_end = QDoubleSpinBox()
        for control in (self.bound_start, self.bound_end):
            control.setDecimals(4)
            control.setRange(0, 86400)
            control.setSuffix(" s")
        bounds.addWidget(QLabel("Start"))
        bounds.addWidget(self.bound_start)
        bounds.addWidget(QLabel("End"))
        bounds.addWidget(self.bound_end)
        self.save_boundary = QPushButton("Save boundary")
        self.save_boundary.clicked.connect(self._save_boundary)
        bounds.addWidget(self.save_boundary)
        editor.addRow("Correction", bounds)
        layout.addLayout(editor)
        actions = QHBoxLayout()
        for label, callback in (("Accept reviewed", self._accept), ("Flag issue", self._flag),
                                ("Exclude", self._exclude), ("Undo", self._undo_change),
                                ("Redo", self._redo_change)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            actions.addWidget(button)
        layout.addLayout(actions)

    def set_record(self, job_root: Path, record: dict, payload: dict) -> None:
        self.job_root, self.record, self.payload = job_root, record, payload
        self.player.setSource(QUrl.fromLocalFile(payload["source"]))
        self.header.setText(f"{record['recording']} · {record['state']} · "
                            f"{record.get('aligned_word_count', 0)} aligned words")
        scope = payload["alignment_scope"]
        self.scope_start.setValue(float(scope["start_sec"]))
        self.scope_end.setValue(float(scope["end_sec"]))
        source_label = ("automatic proposal" if scope["source"] ==
                        "alignment_energy_proposal_v1" else scope["source"])
        self.scope_label.setText(f"Alignment scope: {float(scope['start_sec']):.2f}–"
                                 f"{float(scope['end_sec']):.2f} s · {source_label}")
        self.scope_label.setToolTip(f"Alignment working-audio scope source: {scope['source']}")
        edits = payload["corrections"]
        spec = load_job(job_root)["task_specification"]
        self.canonical_transcript.setPlainText(spec["canonical_transcript"])
        self.spoken_transcript.setPlainText(
            edits.get("reviewed_spoken_transcript") or spec["alignment_transcript"])
        self.trial_selector.clear()
        for trial in payload["trials"]:
            self.trial_selector.addItem(trial["label"], trial)
        self.reviewer.setText(edits.get("reviewer", ""))
        self.issue.setCurrentText(edits.get("issue", "Word boundary") or "Word boundary")
        self.note.setPlainText(edits.get("note", ""))
        self._selected_word = self._selected_phone = None
        self._render()
        self._fit()

    def _request_realign(self) -> None:
        if not self.record or not self.reviewer.text().strip():
            QMessageBox.warning(self, "Reviewer required", "Enter a reviewer name first.")
            return
        spoken = self.spoken_transcript.toPlainText().strip()
        if not spoken:
            QMessageBox.warning(self, "Transcript required", "Enter the words actually spoken.")
            return
        self.realign_requested.emit(self.record["record_key"], spoken,
                                    self.reviewer.text().strip())

    def _render(self) -> None:
        if not self.payload:
            return
        self._track_labels.clear()
        self.wave_plot.clear()
        self.wave_plot.plot(self.payload["times"], self.payload["waveform"],
                            pen=pg.mkPen("#88c7e9", width=1))
        scope = self.payload["alignment_scope"]
        duration = float(self.payload["duration"])
        for start, end in ((0, float(scope["start_sec"])),
                           (float(scope["end_sec"]), duration)):
            if end > start:
                shade = pg.LinearRegionItem(values=(start, end), movable=False,
                                            brush=(80, 80, 80, 100), pen=None)
                self.wave_plot.addItem(shade)
        self.wave_plot.addItem(self.selection)
        self.wave_plot.addItem(self.playheads[0])
        self._drag_lines.clear()
        for plot, key, selected in ((self.word_plot, "words", self._selected_word),
                                    (self.phone_plot, "phones", self._selected_phone)):
            plot.clear()
            track_index = 0 if key == "words" else 1
            plot.addItem(self.track_selections[track_index])
            plot.addItem(self.playheads[track_index + 1])
            for item in self.payload[key]:
                start, end = float(item["start_sec"]), float(item["end_sec"])
                index = int(item["provider_index"])
                color = "#dba447" if index == selected else (
                    "#257895" if key == "words" else "#3f866d")
                plot.addItem(pg.BarGraphItem(x0=[start], x1=[end], y0=[0], y1=[1],
                                             brush=pg.mkBrush(color), pen=pg.mkPen("#081626")))
                label = pg.TextItem(str(item["label"]), color="white", anchor=(.5, .5))
                label.setPos((start + end) / 2, .5)
                plot.addItem(label)
                self._track_labels.append((label, end - start, len(str(item["label"]))))
                if index == selected:
                    for side, position in (("start", start), ("end", end)):
                        handle = pg.InfiniteLine(pos=position, angle=90, movable=True,
                                                 pen=pg.mkPen("#f5c14b", width=3))
                        handle.sigPositionChangeFinished.connect(
                            lambda line, edge=side: self._drag_boundary_finished(edge, line))
                        plot.addItem(handle)
                        self._drag_lines.append(handle)
            plot.setYRange(-.1, 1.1, padding=0)
        self._update_label_visibility()

    def _update_label_visibility(self, *_args) -> None:
        start, end = self.wave_plot.getPlotItem().vb.viewRange()[0]
        seconds_per_pixel = max(1e-9, (end - start) / max(1, self.word_plot.width()))
        for label, duration, letters in self._track_labels:
            label.setVisible(duration / seconds_per_pixel >= max(30, letters * 7))

    def _track_click(self, event, kind: str) -> None:
        if not self.payload:
            return
        plot = self.word_plot if kind == "word" else self.phone_plot
        position = plot.getPlotItem().vb.mapSceneToView(event.scenePos())
        t = position.x()
        self.player.setPosition(round(max(0, t) * 1000))
        target = next((item for item in self.payload[kind + "s"]
                       if float(item["start_sec"]) <= t <= float(item["end_sec"])), None)
        if target is None:
            return
        if kind == "word":
            self._selected_word = int(target["provider_index"])
            self._selected_phone = None
        else:
            self._selected_phone = int(target["provider_index"])
            self._selected_word = None
        self.bound_start.setValue(float(target["start_sec"]))
        self.bound_end.setValue(float(target["end_sec"]))
        if event.double():
            width = max(.1, float(target["end_sec"]) - float(target["start_sec"]))
            self.wave_plot.setXRange(max(0, float(target["start_sec"]) - width),
                                     float(target["end_sec"]) + width, padding=0)
        self._render()

    def _drag_boundary_finished(self, edge: str, line: pg.InfiniteLine) -> None:
        value = float(line.value())
        (self.bound_start if edge == "start" else self.bound_end).setValue(value)
        self._save_boundary()

    def _fit(self) -> None:
        if self.payload:
            self.wave_plot.setXRange(0, self.payload["duration"], padding=0)

    def _zoom_in(self) -> None:
        self.wave_plot.getPlotItem().vb.scaleBy((.7, 1))

    def _zoom_out(self) -> None:
        self.wave_plot.getPlotItem().vb.scaleBy((1.4, 1))

    def _set_speed(self, speed: float) -> None:
        self._speed = speed
        self.player.setPlaybackRate(speed)
        for rate, button in self.speed_buttons.items():
            button.setChecked(rate == speed)

    def _play_bounds(self, start: float, end: float) -> None:
        if not self.payload or not 0 <= start < end <= self.payload["duration"] + .001:
            return
        self._play_end_ms = round(end * 1000)
        self.player.setPosition(round(start * 1000))
        self.player.setPlaybackRate(self._speed)
        self.player.play()

    def _play_pause(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        elif self.payload:
            self._play_end_ms = None
            self.player.play()

    def _play_trial(self) -> None:
        trial = self.trial_selector.currentData()
        if trial:
            self._play_bounds(float(trial["start_sec"]), float(trial["end_sec"]))

    def _play_scope(self) -> None:
        if self.payload:
            scope = self.payload["alignment_scope"]
            self._play_bounds(float(scope["start_sec"]), float(scope["end_sec"]))

    def _scope_from_selection(self) -> None:
        start, end = sorted(self.selection.getRegion())
        self.scope_start.setValue(start)
        self.scope_end.setValue(end)

    def _save_scope(self) -> None:
        if not self.job_root or not self.record:
            return
        if not self.reviewer.text().strip():
            QMessageBox.warning(self, "Reviewer required", "Enter a reviewer name first.")
            return
        try:
            set_alignment_scope(self.job_root, self.record["record_key"],
                                self.scope_start.value(), self.scope_end.value(),
                                self.reviewer.text().strip())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Alignment scope", str(exc))
            return
        QMessageBox.information(self, "Alignment scope saved",
                                "Revalidate the corpus and rerun MFA for this recording. "
                                "The previous boundaries are no longer current.")
        self.payload = None
        for plot in (self.wave_plot, self.word_plot, self.phone_plot):
            plot.clear()
        self.header.setText("Alignment scope changed · revalidate and rerun MFA")
        self.changed.emit()

    def _play_selection(self) -> None:
        start, end = sorted(self.selection.getRegion())
        self._play_bounds(start, end)

    def _position_changed(self, position: int) -> None:
        for line in self.playheads:
            line.setPos(position / 1000)
        if self._play_end_ms is not None and position >= self._play_end_ms:
            self.player.pause()
            self._play_end_ms = None

    def _selection_changed(self, *_args) -> None:
        bounds = self.selection.getRegion()
        for region in self.track_selections:
            region.setRegion(bounds)

    def _persist(self, changes: dict, reason: str) -> None:
        if not self.job_root or not self.record:
            return
        if not self.reviewer.text().strip():
            QMessageBox.warning(self, "Reviewer required", "Enter a reviewer name first.")
            return
        previous = load_corrections(self.job_root, self.record["record_key"])
        self._undo.append(previous)
        self._redo.clear()
        save_corrections(self.job_root, self.record["record_key"], changes,
                         self.reviewer.text(), reason)
        qc_job(self.job_root)
        self.changed.emit()

    def _save_boundary(self) -> None:
        if not self.job_root or not self.record or not self.payload:
            return
        kind = "word" if self._selected_word is not None else "phone"
        index = self._selected_word if kind == "word" else self._selected_phone
        if index is None or self.bound_start.value() >= self.bound_end.value():
            QMessageBox.warning(self, "Boundary", "Select a word or phone and valid bounds.")
            return
        items = sorted(self.payload[kind + "s"], key=lambda item: float(item["start_sec"]))
        position = next((at for at, item in enumerate(items)
                         if int(item["provider_index"]) == index), None)
        if position is None:
            return
        start, end = self.bound_start.value(), self.bound_end.value()
        if (position > 0 and start < float(items[position - 1]["end_sec"]) - 1e-7 or
                position + 1 < len(items) and end > float(items[position + 1]["start_sec"]) + 1e-7 or
                end > float(self.payload["duration"]) + 1e-7):
            QMessageBox.warning(self, "Boundary", "Correction would cross another interval.")
            return
        if kind == "phone":
            current = items[position]
            parent = next((word for word in self.payload["words"]
                           if float(word["start_sec"]) - 1e-7 <= float(current["start_sec"])
                           < float(current["end_sec"]) <= float(word["end_sec"]) + 1e-7), None)
            if parent is None or start < float(parent["start_sec"]) - 1e-7 or end > float(
                    parent["end_sec"]) + 1e-7:
                QMessageBox.warning(self, "Boundary", "Phone must remain within its word.")
                return
        else:
            old = items[position]
            children = [phone for phone in self.payload["phones"]
                        if float(old["start_sec"]) - 1e-7 <= float(phone["start_sec"])
                        < float(phone["end_sec"]) <= float(old["end_sec"]) + 1e-7]
            if any(start > float(phone["start_sec"]) + 1e-7 or
                   end < float(phone["end_sec"]) - 1e-7 for phone in children):
                QMessageBox.warning(self, "Boundary", "Word must contain its phones.")
                return
        edits = load_corrections(self.job_root, self.record["record_key"])
        bounds = dict(edits.get(f"{kind}_bounds", {}))
        bounds[str(index)] = [start, end]
        self._persist({f"{kind}_bounds": bounds}, f"Corrected {kind} boundary {index}")

    def _accept(self) -> None:
        if not self.job_root or not self.record:
            return
        self._persist({"decision": "ACCEPT", "issue": self.issue.currentText(),
                       "note": self.note.toPlainText()}, "Accepted reviewed alignment")

    def _flag(self) -> None:
        self._persist({"decision": "FLAG", "issue": self.issue.currentText(),
                       "note": self.note.toPlainText()}, "Flagged alignment issue")

    def _exclude(self) -> None:
        self._persist({"decision": "EXCLUDE", "issue": self.issue.currentText(),
                       "note": self.note.toPlainText()}, "Excluded recording")

    def _undo_change(self) -> None:
        if not self._undo or not self.job_root or not self.record:
            return
        current = load_corrections(self.job_root, self.record["record_key"])
        self._redo.append(current)
        previous = self._undo.pop()
        save_corrections(self.job_root, self.record["record_key"], {
            key: previous[key] for key in ("removed_word_indices", "word_bounds", "phone_bounds",
                                            "issue", "note", "decision")},
            self.reviewer.text(), "Undo correction")
        qc_job(self.job_root)
        self.changed.emit()

    def _redo_change(self) -> None:
        if not self._redo or not self.job_root or not self.record:
            return
        current = load_corrections(self.job_root, self.record["record_key"])
        self._undo.append(current)
        redo = self._redo.pop()
        save_corrections(self.job_root, self.record["record_key"], {
            key: redo[key] for key in ("removed_word_indices", "word_bounds", "phone_bounds",
                                        "issue", "note", "decision")},
            self.reviewer.text(), "Redo correction")
        qc_job(self.job_root)
        self.changed.emit()
