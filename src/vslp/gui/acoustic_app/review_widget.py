"""PySide workstation for persistent, task-neutral segmentation adjudication."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

from vslp.acoustic.segment.review import (
    finalize_segmentation_review, initialize_segmentation_review, load_review_state,
    preview_manual_segmentation, save_segmentation_review_entry,
)


class SegmentationReviewWidget(QWidget):
    finalized = Signal(object)
    continue_requested = Signal()

    def __init__(self, output_root: Callable[[], Path], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_root = output_root
        self._decisions = pd.DataFrame()
        self._overrides = pd.DataFrame()
        self._current_id = ""
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        outer = QVBoxLayout(self)
        top = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Pending review", "Review required", "All", "Accepted automatically",
                                    "Kept automatic after review", "Kept manual", "Excluded"])
        self.filter_combo.currentIndexChanged.connect(self._filter_rows)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Find recording or flag")
        self.search_edit.textChanged.connect(self._filter_rows)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        self.progress_label = QLabel("No review loaded")
        top.addWidget(self.filter_combo)
        top.addWidget(self.search_edit, 1)
        top.addWidget(refresh)
        top.addWidget(self.progress_label)
        outer.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        self.recording_list = QListWidget()
        self.recording_list.currentItemChanged.connect(self._show_selected)
        splitter.addWidget(self.recording_list)
        detail = QWidget()
        right = QVBoxLayout(detail)
        self.identity_label = QLabel("Select a recording")
        self.status_label = QLabel("")
        self.flags_label = QLabel("")
        right.addWidget(self.identity_label)
        right.addWidget(self.status_label)
        right.addWidget(self.flags_label)
        self.plot_label = QLabel()
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setMinimumSize(600, 320)
        plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setWidget(self.plot_label)
        right.addWidget(plot_scroll, 1)
        audio_row = QHBoxLayout()
        self.play_button = QPushButton("Play audio")
        self.play_button.clicked.connect(self._play)
        stop = QPushButton("Stop")
        stop.clicked.connect(self._player.stop)
        audio_row.addWidget(self.play_button)
        audio_row.addWidget(stop)
        audio_row.addStretch(1)
        right.addLayout(audio_row)
        self.reviewer_edit = QLineEdit()
        self.reviewer_edit.setPlaceholderText("Reviewer name")
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Review notes")
        self.notes_edit.setMaximumHeight(70)
        self.manual_edit = QPlainTextEdit()
        self.manual_edit.setPlaceholderText("One start_sec,end_sec interval per line")
        self.manual_edit.setMaximumHeight(95)
        self.manual_edit.setVisible(False)
        right.addWidget(self.reviewer_edit)
        right.addWidget(self.notes_edit)
        right.addWidget(self.manual_edit)
        actions = QHBoxLayout()
        self.keep_button = QPushButton("Keep automatic")
        self.keep_button.setToolTip("Retain the algorithm's exact intervals.")
        self.keep_button.clicked.connect(lambda: self._save("KEEP_AUTO"))
        self.edit_button = QPushButton("Edit boundaries")
        self.edit_button.clicked.connect(lambda: self.manual_edit.setVisible(True))
        self.preview_button = QPushButton("Preview manual")
        self.preview_button.clicked.connect(self._preview)
        self.save_manual_button = QPushButton("Save manual")
        self.save_manual_button.clicked.connect(lambda: self._save("KEEP_MANUAL"))
        self.exclude_button = QPushButton("Exclude")
        self.exclude_button.clicked.connect(lambda: self._save("EXCLUDE"))
        for button in (self.keep_button, self.edit_button, self.preview_button,
                       self.save_manual_button, self.exclude_button):
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
        navigation.addWidget(previous)
        navigation.addWidget(next_button)
        navigation.addStretch(1)
        navigation.addWidget(self.finalize_button)
        navigation.addWidget(self.continue_button)
        right.addLayout(navigation)
        splitter.addWidget(detail)
        splitter.setStretchFactor(1, 3)
        outer.addWidget(splitter, 1)

    def refresh(self) -> None:
        try:
            root = self._output_root()
            source = root / "acoustic" / "002_segmentation" / "tables" / "acoustic_segmentation_summary.csv"
            initialize_segmentation_review(source, root)
            self._decisions, self._overrides = load_review_state(root)
            frozen_path = (root / "acoustic" / "003_segmentation_review" / "tables" /
                           "final_segmentation_decisions.csv")
            frozen = frozen_path.exists()
            if frozen:
                self._decisions = pd.read_csv(frozen_path, dtype=str, keep_default_na=False)
            self.finalize_button.setEnabled(not frozen)
            self.continue_button.setEnabled(frozen)
            total = int(self._decisions.review_required.map(
                lambda v: str(v).lower() in {"true", "1", "yes"}).sum())
            pending = int(((self._decisions.final_decision == "") & self._decisions.review_required.map(
                lambda v: str(v).lower() in {"true", "1", "yes"})).sum())
            self.progress_label.setText(f"Required: {total - pending}/{total}")
            self._filter_rows()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Manual review", str(exc))

    def _filter_rows(self) -> None:
        if self._decisions.empty:
            return
        chosen = self._current_id
        frame = self._decisions.copy()
        state = self.filter_combo.currentText()
        required = frame.review_required.astype(str).str.lower().isin(["true", "1", "yes"])
        if state == "Pending review":
            frame = frame.loc[required & frame.final_decision.eq("")]
        elif state == "Review required":
            frame = frame.loc[required]
        elif state == "Accepted automatically":
            frame = frame.loc[frame.automatic_status.eq("ACCEPTED") & frame.reviewer.eq("")]
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
            from PySide6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(f"[{row.automatic_status}] {row.file_name}")
            item.setData(Qt.UserRole, str(row.recording_id))
            self.recording_list.addItem(item)
        self.recording_list.blockSignals(False)
        for index in range(self.recording_list.count()):
            if self.recording_list.item(index).data(Qt.UserRole) == chosen:
                self.recording_list.setCurrentRow(index)
                break
        else:
            if self.recording_list.count():
                self.recording_list.setCurrentRow(0)
            else:
                self._current_id = ""
                self.identity_label.setText("No recordings in this filter")
                self.plot_label.clear()

    def _show_selected(self, item, _previous=None) -> None:
        if item is None:
            return
        self._player.stop()
        self._current_id = str(item.data(Qt.UserRole))
        row = self._decisions.loc[self._decisions.recording_id.eq(self._current_id)].iloc[0]
        self.identity_label.setText(f"{row.file_name}  ·  {row.recording_id}")
        self.status_label.setText(f"{row.segmentation_method}  |  Automatic: {row.automatic_status}  |  Final: {row.final_decision or 'Pending'}")
        self.flags_label.setText(f"Flags: {row.automatic_flags or 'None'}")
        self.reviewer_edit.setText(row.reviewer)
        self.notes_edit.setPlainText(row.review_notes)
        saved = self._overrides.loc[self._overrides.recording_id.eq(self._current_id)]
        if not saved.empty:
            self.manual_edit.setPlainText("\n".join(f"{float(r.start_sec):.6f},{float(r.end_sec):.6f}"
                for r in saved.itertuples()))
            self.manual_edit.setVisible(True)
        else:
            self.manual_edit.clear()
            self.manual_edit.setVisible(False)
        plot = Path(row.plot_path or row.automatic_plot_path)
        self._display_plot(plot)
        frozen = not self.finalize_button.isEnabled()
        enabled = row.automatic_status != "FAILED" and not frozen
        for button in (self.keep_button, self.edit_button, self.preview_button,
                       self.save_manual_button, self.exclude_button):
            button.setEnabled(enabled)
        self.play_button.setEnabled(Path(row.analysis_wav_path).is_file())

    def _display_plot(self, path: Path) -> None:
        if path.is_file():
            pixmap = QPixmap(str(path))
            self.plot_label.setPixmap(pixmap.scaled(950, 490, Qt.KeepAspectRatio,
                                                   Qt.SmoothTransformation))
        else:
            self.plot_label.setText("Plot unavailable")

    def _play(self) -> None:
        if not self._current_id:
            return
        row = self._decisions.loc[self._decisions.recording_id.eq(self._current_id)].iloc[0]
        path = Path(row.analysis_wav_path)
        if path.is_file():
            self._player.setSource(QUrl.fromLocalFile(str(path.resolve())))
            self._player.play()

    def _navigate(self, offset: int) -> None:
        index = self.recording_list.currentRow() + offset
        if 0 <= index < self.recording_list.count():
            self.recording_list.setCurrentRow(index)

    def _save(self, decision: str) -> None:
        if not self._current_id:
            return
        try:
            save_segmentation_review_entry(self._output_root(), self._current_id, decision,
                self.reviewer_edit.text(), self.notes_edit.toPlainText(),
                self.manual_edit.toPlainText())
            self.refresh()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Review not saved", str(exc))

    def _preview(self) -> None:
        if not self._current_id:
            return
        try:
            path = preview_manual_segmentation(self._output_root(), self._current_id,
                                                self.manual_edit.toPlainText())
            self._display_plot(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Manual preview", str(exc))

    def _finalize(self) -> None:
        try:
            result = finalize_segmentation_review(self._output_root())
            self.refresh()
            self.finalized.emit(result)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Review incomplete", str(exc))
