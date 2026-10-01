"""Compact viewer and controls for the separate Alignment stage."""

from __future__ import annotations

import json
import time
import re
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyqtgraph as pg
import soundfile as sf
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtGui import QBrush, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QMessageBox, QDialog, QScrollArea,
    QPushButton, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QDoubleSpinBox,
    QProgressBar, QApplication,
)

from vslp.acoustic.alignment import AlignmentConfig, list_alignment_runs
from vslp.acoustic.alignment.mfa_provider import MfaProfile, MfaProvider
from vslp.acoustic.alignment.profiles import default_profile_path
from vslp.acoustic.alignment.task_workflow import (
    build_task_prompt_manifests, load_project_choices, project_alignment_context,
    resolve_prompt, save_project_choices,
)
from vslp.acoustic.alignment.trials import reviewed_trial_candidates
from vslp.acoustic.alignment.trials import validate_trial_draft
from vslp.acoustic.alignment.trial_review import set_trial_review, trial_review_table
from vslp.acoustic.alignment.trial_review import mark_recording_stale
from vslp.acoustic.alignment.corrections import (load_corrections, reset_corrections,
                                                 set_correction, reviewed_tables)
from .display_waveform import DisplayWaveformCache


_DISPLAY_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vslp-display")


class AlignmentWidget(QWidget):
    waveform_ready = Signal(object)
    return_to_features_requested = Signal()
    plot_requested = Signal(str, str, str)
    run_requested = Signal(object)
    freeze_requested = Signal(str)
    self_test_requested = Signal(str, str)
    preflight_requested = Signal(str, str)
    environment_requested = Signal(str)
    run_task_requested = Signal(str, str)
    rerun_recording_requested = Signal(str, str, str, str)

    def __init__(self, root_provider: Callable[[], Path]) -> None:
        super().__init__()
        self.root_provider = root_provider
        self._display_cache = DisplayWaveformCache()
        self._waveform_request = 0
        self.waveform_ready.connect(self._apply_waveform)
        self._review_cache_run = ""
        self._review_cache: pd.DataFrame | None = None
        self._feature_return_pending = False
        layout = QVBoxLayout(self)
        self.steps = QLabel("● 1 Confirm trials   →   ○ 2 Run alignment   →   ○ 3 Review   →   ○ 4 Freeze")
        layout.addWidget(self.steps)
        self.action_hint = QLabel("Choose a recording, check the shaded repetitions, then confirm them.")
        self.action_hint.setWordWrap(True)
        layout.addWidget(self.action_hint)
        self.primary = QPushButton("CONFIRM REPETITIONS")
        self.primary.setMinimumHeight(42)
        self.primary.clicked.connect(self._primary_action)
        self.next_action = QLabel("Next: Run Alignment after confirming trials")
        self.next_action.setWordWrap(True)
        self.run_progress = QProgressBar()
        self.run_progress.setRange(0, 0)
        self.run_progress.hide()
        layout.addWidget(self.run_progress)
        self.run_elapsed = QLabel("")
        self.run_elapsed.hide()
        layout.addWidget(self.run_elapsed)
        self._run_started_at = 0.0
        self._run_timer = QTimer(self)
        self._run_timer.timeout.connect(self._tick_run)
        self._running = False
        self._view_step = "trials"
        self._alignment_result_summary = ""
        self._alignment_had_issues = False
        self._editing_trials = False
        self._repair_ready = False
        self._repair_record_id = ""
        self._repair_source_run_id = ""
        self._editing_alignment = False
        self.task_brief = QLabel("Select a project in Setup")
        self.task_brief.setWordWrap(True)
        layout.addWidget(self.task_brief)
        self.readiness = QLabel("Review trial boundaries to prepare Alignment")
        layout.addWidget(self.readiness)
        form = QFormLayout()
        self.source = QComboBox()
        self.source.addItem("Montreal Forced Aligner", "mfa")
        self.source.addItem("Validated import", "external")
        self.source.currentIndexChanged.connect(self._source_changed)
        form.addRow("Provider", self.source)
        self.task_label = QLabel("Select a project in Setup")
        form.addRow("Task", self.task_label)
        self.stimulus = QComboBox()
        self.stimulus.currentIndexChanged.connect(self._stimulus_changed)
        form.addRow("Stimulus", self.stimulus)
        self.prompt_display = QLabel("No canonical prompt resolved")
        self.prompt_display.setWordWrap(True)
        form.addRow("Expected prompt", self.prompt_display)
        self.speaker_status = QLabel("Speaker identity not checked")
        self.speaker_status.setWordWrap(True)
        form.addRow("Speaker identity", self.speaker_status)
        self.reviewer = QLineEdit()
        self.reviewer.setPlaceholderText("Required only if an alignment transcript is corrected")
        self.reviewer.editingFinished.connect(self._save_reviewer)
        form.addRow("Transcript reviewer", self.reviewer)
        self.review_status = QLabel("Final reviewed segmentation not checked")
        form.addRow("Reviewed segmentation", self.review_status)
        self.prompt = self._path_row(form, "Prompt manifest", "JSON files (*.json)")
        self.words = self._path_row(form, "Validated words", "CSV files (*.csv)")
        self.phones = self._path_row(form, "Validated phones", "CSV files (*.csv)")
        self.profile = self._path_row(form, "Alignment profile", "JSON files (*.json)")
        self.profile.setText(default_profile_path())
        self.speakers = self._path_row(form, "Speaker mapping", "JSON files (*.json)")
        self.overrides = self._path_row(form, "Reviewed transcript overrides", "JSON files (*.json)")
        self.self_test_wav = self._path_row(form, "Select nonclinical test WAV (optional)", "WAV files (*.wav)")
        self.self_test_wav.setPlaceholderText("Select nonclinical test WAV if offline TTS is unavailable")
        self.environment = QLabel("MFA environment not checked")
        self.environment.setWordWrap(True)
        form.addRow("Environment", self.environment)
        self.preflight = QLabel("Preflight not run")
        self.preflight.setWordWrap(True)
        form.addRow("Preflight", self.preflight)
        self.recording_count = QLabel("Recordings: —")
        form.addRow("", self.recording_count)
        self.assignments = QTableWidget(0, 5)
        self.assignments.setHorizontalHeaderLabels(
            ["Recording", "Stimulus", "Speaker ID", "Alignment transcript", "Correction reason"])
        self.assignments.horizontalHeader().setStretchLastSection(True)
        self.assignments.itemChanged.connect(self._save_assignments)
        # Scientist-facing review controls remain visible; configuration lives below.
        self.normal_summary = QLabel("Select a recording and confirm each trial")
        self.normal_summary.setWordWrap(True)
        layout.addWidget(self.normal_summary)
        self.speaker_help = QLabel(
            "Speaker ID is required for Alignment. Enter each recording's deidentified "
            "speaker ID in the table, or paste one ID per missing row in table order. "
            "Use the same ID for recordings from the same speaker.")
        self.speaker_help.setWordWrap(True)
        self.speaker_help.hide()
        self.paste_speakers_button = QPushButton("Paste Speaker IDs for missing rows")
        self.paste_speakers_button.clicked.connect(self._paste_speaker_ids)
        self.paste_speakers_button.hide()
        check_environment = QPushButton("CHECK MFA ENVIRONMENT")
        check_environment.clicked.connect(self._check_environment)
        self.check_environment_button = check_environment
        self.self_test_button = QPushButton("RUN MFA SELF-TEST")
        self.self_test_button.clicked.connect(lambda: self.self_test_requested.emit(
            self.profile.text().strip(), self.self_test_wav.text().strip()))
        self.model = QLineEdit()
        self.model_version = QLineEdit()
        self.dictionary = QLineEdit()
        self.dictionary_version = QLineEdit()
        self.advanced_toggle = QPushButton("Diagnostics…")
        layout.addWidget(self.advanced_toggle)
        self.diagnostics_dialog = QDialog(self)
        self.diagnostics_dialog.setWindowTitle("Alignment diagnostics")
        self.diagnostics_dialog.resize(740, 550)
        diagnostic_layout = QVBoxLayout(self.diagnostics_dialog)
        diagnostic_scroll = QScrollArea()
        diagnostic_scroll.setWidgetResizable(True)
        diagnostic_layout.addWidget(diagnostic_scroll)
        self.advanced = QWidget()
        diagnostic_scroll.setWidget(self.advanced)
        advanced_form = QFormLayout(self.advanced)
        advanced_form.addRow(form)
        advanced_form.addRow("Recording assignments and transcript corrections", self.assignments)
        advanced_form.addRow("", self.speaker_help)
        advanced_form.addRow("", self.paste_speakers_button)
        advanced_form.addRow("MFA acoustic model", self.model)
        advanced_form.addRow("Model version", self.model_version)
        advanced_form.addRow("MFA dictionary", self.dictionary)
        advanced_form.addRow("Dictionary version", self.dictionary_version)
        advanced_form.addRow("", check_environment)
        advanced_form.addRow("", self.self_test_button)
        self.model.setReadOnly(True)
        self.model_version.setReadOnly(True)
        self.dictionary.setReadOnly(True)
        self.dictionary_version.setReadOnly(True)
        self.technical = QLabel("Select an alignment run to inspect provider details")
        self.technical.setWordWrap(True)
        advanced_form.addRow("Run diagnostics", self.technical)
        self.advanced_run_label = QLabel("Alignment run")
        self.advanced_toggle.clicked.connect(self.diagnostics_dialog.show)
        self.advanced_toggle.clicked.connect(self._source_changed)
        self.diagnostics_dialog.finished.connect(lambda _result: self._source_changed())
        self.source_fields = [form.labelForField(self.words._alignment_row), self.words._alignment_row,
                              form.labelForField(self.phones._alignment_row), self.phones._alignment_row]
        self.mfa_fields = [form.labelForField(self.prompt._alignment_row), self.prompt._alignment_row,
                           form.labelForField(self.profile._alignment_row), self.profile._alignment_row,
                           form.labelForField(self.speakers._alignment_row), self.speakers._alignment_row,
                           form.labelForField(self.overrides._alignment_row), self.overrides._alignment_row,
                           form.labelForField(self.self_test_wav._alignment_row), self.self_test_wav._alignment_row,
                           ]
        buttons = QHBoxLayout()
        run = QPushButton("RUN ALIGNMENT")
        run.clicked.connect(self._run)
        self.run_button = run
        run.setEnabled(False)
        buttons.addWidget(run)
        buttons.addWidget(QLabel("Alignment run"))
        self.runs = QComboBox()
        self.runs.currentIndexChanged.connect(self._run_changed)
        buttons.addWidget(self.runs, 1)
        new_plan = QPushButton("Define / rerun trials")
        new_plan.clicked.connect(self._new_plan)
        buttons.addWidget(new_plan)
        inspect = QPushButton("INSPECT ALIGNMENT")
        inspect.clicked.connect(self._mark_inspected)
        buttons.addWidget(inspect)
        freeze = QPushButton("FREEZE ALIGNMENT")
        freeze.clicked.connect(lambda: self.freeze_requested.emit(self.runs.currentData() or ""))
        self.freeze_button = freeze
        freeze.setEnabled(False)
        buttons.addWidget(freeze)
        advanced_form.addRow("Historical run", self.runs)
        self.summary = QLabel("No alignment run selected")
        layout.addWidget(self.summary)
        self.frozen_status = QLabel("No frozen Alignment")
        advanced_form.addRow("Frozen run", self.frozen_status)
        self.self_test_summary = QLabel("MFA self-test not run")
        self.self_test_summary.setWordWrap(True)
        advanced_form.addRow("System self-test", self.self_test_summary)
        self.view_self_test_diagnostics = QPushButton("View self-test diagnostics")
        self.view_self_test_diagnostics.hide()
        self.view_self_test_diagnostics.clicked.connect(self._open_self_test_diagnostics)
        advanced_form.addRow("", self.view_self_test_diagnostics)
        self._self_test_diagnostics_path = ""
        self.recordings = QComboBox()
        self.recordings.currentIndexChanged.connect(self._recording_changed)
        navigation = QHBoxLayout()
        self.previous_recording = QPushButton("◀ Recording")
        self.next_recording = QPushButton("Recording ▶")
        self.previous_recording.clicked.connect(lambda: self._step_recording(-1))
        self.next_recording.clicked.connect(lambda: self._step_recording(1))
        navigation.addWidget(self.previous_recording)
        navigation.addWidget(self.recordings, 1)
        navigation.addWidget(self.next_recording)
        layout.insertLayout(0, navigation)
        self.record_stimulus = QComboBox()
        self.record_stimulus.currentIndexChanged.connect(self._record_stimulus_changed)
        layout.insertWidget(1, self.record_stimulus)
        trial_controls = QHBoxLayout()
        self.trials = QComboBox()
        self.trials.currentIndexChanged.connect(self._trial_changed)
        self.previous_trial = QPushButton("◀ Trial")
        self.next_trial = QPushButton("Trial ▶")
        self.previous_trial.clicked.connect(lambda: self._step_trial(-1))
        self.next_trial.clicked.connect(lambda: self._step_trial(1))
        trial_controls.addWidget(self.previous_trial)
        trial_controls.addWidget(self.trials, 1)
        trial_controls.addWidget(self.next_trial)
        layout.insertLayout(1, trial_controls)
        self.trial_editor = QWidget()
        trial_editor = QHBoxLayout(self.trial_editor)
        self.speech_candidates = QComboBox()
        self.speech_candidates.currentIndexChanged.connect(self._candidate_changed)
        self.trial_start = QDoubleSpinBox()
        self.trial_end = QDoubleSpinBox()
        for control in (self.trial_start, self.trial_end):
            control.setRange(0, 86400)
            control.setDecimals(3)
            control.setSuffix(" s")
        add_trial = QPushButton("Confirm trial bounds")
        add_trial.clicked.connect(self._add_trial)
        remove_trial = QPushButton("Remove trial")
        remove_trial.clicked.connect(self._remove_trial)
        for control in (QLabel("Reviewed speech"), self.speech_candidates, QLabel("Start"),
                        self.trial_start, QLabel("End"), self.trial_end, add_trial, remove_trial):
            trial_editor.addWidget(control)
        advanced_form.addRow("Precise trial boundaries", self.trial_editor)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Original recording time", units="s")
        self.plot.setLabel("left", "Waveform")
        self.plot.showGrid(x=True, alpha=0.15)
        self.plot.scene().sigMouseClicked.connect(self._token_clicked)
        self.plot.viewport().installEventFilter(self)
        self.plot.setMinimumHeight(350)
        self.proposal_strip = QWidget()
        self.proposal_strip_layout = QHBoxLayout(self.proposal_strip)
        self.proposal_strip_layout.setContentsMargins(0, 0, 0, 0)
        self.trial_count = QLabel("Expected repetitions: —  ·  Current trials: —")
        self.trial_count.setWordWrap(True)
        layout.addWidget(self.trial_count)
        layout.addWidget(self.proposal_strip)
        layout.addWidget(self.plot, 1)
        self.trial_actions = QWidget()
        trial_actions = QHBoxLayout(self.trial_actions)
        trial_actions.setContentsMargins(0, 0, 0, 0)
        self.merge_button = QPushButton("MERGE")
        self.merge_button.clicked.connect(self._merge_proposals)
        self.split_button = QPushButton("SPLIT")
        self.split_button.clicked.connect(self._split_proposal)
        self.delete_button = QPushButton("DELETE")
        self.delete_button.clicked.connect(self._omit_proposal)
        self.add_button = QPushButton("ADD TRIAL")
        self.add_button.clicked.connect(self._begin_add_trial)
        self.undo_button = QPushButton("UNDO")
        self.undo_button.clicked.connect(self._undo_proposal)
        for control in (self.merge_button, self.split_button, self.delete_button,
                        self.add_button, self.undo_button):
            trial_actions.addWidget(control)
        layout.addWidget(self.trial_actions)
        playback = QHBoxLayout()
        self.player: QMediaPlayer | None = None
        self.audio_output: QAudioOutput | None = None
        self.play_button = QPushButton("Play trial")
        self.play_button.clicked.connect(self._play_pause)
        self.play_visible_button = QPushButton("Play visible region")
        self.play_visible_button.clicked.connect(self._play_visible)
        self.play_token_button = QPushButton("Play selected token")
        self.play_token_button.clicked.connect(self._play_token)
        self.accept_button = QPushButton("Accept Alignment")
        self.accept_button.clicked.connect(lambda: self._set_review("ACCEPTED"))
        self.needs_review_button = QPushButton("Needs Review")
        self.needs_review_button.clicked.connect(lambda: self._set_review("NEEDS_REVIEW"))
        self.keep_needs_review_button = QPushButton("Keep Needs Review — next trial")
        self.keep_needs_review_button.clicked.connect(
            lambda: self._navigate_unreviewed(after_current=True))
        for control in (self.play_button, self.play_visible_button,
                        self.play_token_button, self.accept_button,
                        self.needs_review_button):
            playback.addWidget(control)
        layout.addLayout(playback)
        layout.addWidget(self.keep_needs_review_button)
        correction_actions = QHBoxLayout()
        self.edit_trial_button = QPushButton("Edit Trials")
        self.edit_trial_button.clicked.connect(self._edit_trial_boundaries)
        self.edit_alignment_button = QPushButton("Edit Alignment")
        self.edit_alignment_button.clicked.connect(self._toggle_alignment_edit)
        self.reset_token_button = QPushButton("Reset token to MFA")
        self.reset_token_button.clicked.connect(self._reset_selected_token)
        self.reset_trial_button = QPushButton("Reset trial to MFA")
        self.reset_trial_button.clicked.connect(self._reset_current_trial)
        for control in (self.edit_trial_button, self.edit_alignment_button,
                        self.reset_token_button, self.reset_trial_button):
            correction_actions.addWidget(control)
        layout.addLayout(correction_actions)
        self.review_progress = QLabel("")
        layout.addWidget(self.review_progress)
        issue_nav = QHBoxLayout()
        self.previous_issue = QPushButton("◀ Previous issue")
        self.next_issue = QPushButton("Next issue ▶")
        self.previous_issue.clicked.connect(lambda: self._step_issue(-1))
        self.next_issue.clicked.connect(lambda: self._step_issue(1))
        self.view_auto_passed = QPushButton("View auto-passed cases")
        self.view_auto_passed.clicked.connect(self._view_auto_passed)
        for button in (self.previous_issue, self.next_issue, self.view_auto_passed):
            issue_nav.addWidget(button)
        layout.addLayout(issue_nav)
        self.plot_selected_button = QPushButton("Generate plot for selected trial")
        self.plot_selected_button.clicked.connect(self._request_selected_plot)
        advanced_form.addRow("Optional QC plot", self.plot_selected_button)

        self.proposal_regions: list[pg.LinearRegionItem] = []
        self.proposal_labels: list[pg.TextItem] = []
        self._selected_proposal = 0
        self._selected_proposals: set[int] = set()
        self._proposal_history: list[list[tuple[float, float]]] = []
        self._proposal_last_bounds: list[tuple[float, float]] = []
        self._rebuilding_proposals = False
        self._add_mode = False
        self._add_drag_start: float | None = None
        self._add_drag_preview: pg.LinearRegionItem | None = None
        self._proposal_cursor: float | None = None
        self.play_cursor = pg.InfiniteLine(angle=90, movable=False,
                                          pen=pg.mkPen("#ECA35D", width=2))
        self._play_end_ms = 0
        self._selected_token: tuple[float, float] | None = None
        self._selected_token_key: tuple[str, int] | None = None
        self._edit_handle: pg.LinearRegionItem | None = None
        self.token_detail = QLabel("Select a token on the timeline")
        layout.addWidget(self.token_detail)
        layout.addWidget(self.next_action)
        layout.addWidget(self.primary)
        self._tokens: list[tuple[float, float, str, str, str, int, str]] = []
        self._loading_assignments = False
        self._segmentation_display_tables: dict[str, tuple[tuple[int, int], pd.DataFrame]] = {}
        self._last_preflight_key = ""
        self._preflight_ready = False
        self._preflight_timer = QTimer(self)
        self._preflight_timer.setSingleShot(True)
        self._preflight_timer.timeout.connect(lambda: self.preflight_requested.emit(
            str(self.root_provider()), self.profile.text().strip()))
        self._source_changed()
        self._show_step()

    def _request_selected_plot(self) -> None:
        trial = self.trials.currentData()
        if self.runs.currentData() and self.recordings.currentData() and trial:
            self.plot_requested.emit(str(self.runs.currentData()),
                                     str(self.recordings.currentData()),
                                     str(trial["trial_id"]))

    def _path_row(self, form: QFormLayout, label: str, file_filter: str) -> QLineEdit:
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        choose = QPushButton("Browse")
        choose.clicked.connect(lambda: self._browse(edit, file_filter))
        line.addWidget(edit)
        line.addWidget(choose)
        form.addRow(label, row)
        edit._alignment_row = row
        return edit

    def _browse(self, field: QLineEdit, file_filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", file_filter)
        if path:
            field.setText(path)

    def _source_changed(self) -> None:
        external = self.source.currentData() == "external"
        advanced = self.diagnostics_dialog.isVisible()
        for widget in self.source_fields:
            widget.setVisible(external and advanced)
        for widget in self.mfa_fields:
            widget.setVisible(advanced)
        self.run_button.setEnabled(external or self._preflight_ready)
        self._show_step()

    def _project_trials_confirmed(self) -> bool:
        try:
            context = project_alignment_context(self.root_provider())
        except RuntimeError:
            return False
        records = context.get("records", [])
        choices = context.get("choices", {}).get("trials", {})
        return bool(records) and all(choices.get(row["recording_id"]) for row in records)

    def _trial_count_mismatch(self) -> tuple[int, int] | None:
        expected = self._expected_repetitions()
        current = (len(self.proposal_regions) if not self.runs.currentData()
                   else len(self._trial_entries()))
        if expected is None or not current or expected == current:
            return None
        return int(expected), current

    def _proposals_differ(self) -> bool:
        existing = sorted((float(item["start_sec"]), float(item["end_sec"]))
                          for item in self._trial_entries())
        current = sorted(tuple(map(float, region.getRegion()))
                         for region in self.proposal_regions)
        return len(existing) != len(current) or any(
            abs(left[0] - right[0]) > .001 or abs(left[1] - right[1]) > .001
            for left, right in zip(existing, current))

    def _reviews(self, run_id: str) -> pd.DataFrame:
        if self._review_cache is None or self._review_cache_run != str(run_id):
            self._review_cache = trial_review_table(self.root_provider(), run_id)
            self._review_cache_run = str(run_id)
        return self._review_cache

    def _issue_rows(self, *, unhandled_only: bool = False) -> pd.DataFrame:
        run_id = self.runs.currentData()
        if not run_id:
            return pd.DataFrame()
        table = self._reviews(run_id)
        if table.empty:
            return table
        if "queued_for_review" in table:
            flagged = table.queued_for_review.astype(str).str.lower().isin(["true", "1"])
        else:
            flagged = ~table.review_status.eq("AUTO_ACCEPTED_STRUCTURAL")
        if unhandled_only:
            mode = table.review_mode.astype(str) if "review_mode" in table else pd.Series(
                "", index=table.index)
            flagged &= ~mode.eq("HUMAN")
        return table.loc[flagged]

    def _step_issue(self, direction: int) -> None:
        issues = self._issue_rows()
        if issues.empty:
            return
        keys = list(zip(issues.recording_id.astype(str), issues.trial_id.astype(str)))
        current = (str(self.recordings.currentData()),
                   str((self.trials.currentData() or {}).get("trial_id", "")))
        index = keys.index(current) if current in keys else (0 if direction < 0 else -1)
        target = keys[(index + direction) % len(keys)]
        self.focus_trial(*target)

    def _view_auto_passed(self) -> None:
        run_id = self.runs.currentData()
        if not run_id:
            return
        table = self._reviews(run_id)
        accepted = table.loc[table.review_status.eq("AUTO_ACCEPTED_STRUCTURAL")]
        if not accepted.empty:
            first = accepted.iloc[0]
            self.focus_trial(str(first.recording_id), str(first.trial_id))

    def _review_counts(self) -> tuple[int, int, int]:
        run_id = self.runs.currentData() if hasattr(self, "runs") else None
        if not run_id:
            return 0, 0, 0
        table = self._reviews(run_id)
        return (len(table), int(table.review_status.isin(
            ["ACCEPTED", "AUTO_ACCEPTED_STRUCTURAL"]).sum()),
                int(table.review_status.eq("NEEDS_REVIEW").sum()))

    def _current_review_status(self) -> str:
        run_id = self.runs.currentData()
        entry = self.trials.currentData()
        if not run_id or not entry:
            return ""
        table = self._reviews(run_id)
        current = table.loc[table.recording_id.astype(str).eq(str(self.recordings.currentData()))
                            & table.trial_id.astype(str).eq(str(entry["trial_id"]))]
        return str(current.iloc[0].review_status) if len(current) == 1 else ""

    def _show_step(self) -> None:
        if not hasattr(self, "primary") or not hasattr(self, "runs"):
            return
        try:
            self.root_provider()
        except RuntimeError:
            self.steps.setText("● 1 Select a project   →   ○ 2 Run alignment   →   ○ 3 Review   →   ○ 4 Freeze")
            self.action_hint.setText("Select or open a project in Setup to begin Alignment.")
            self.primary.setEnabled(False)
            return
        total, accepted, issues = self._review_counts()
        review_table = (self._reviews(self.runs.currentData())
                        if self.runs.currentData() else pd.DataFrame())
        auto_accepted = (int(review_table.review_status.eq(
            "AUTO_ACCEPTED_STRUCTURAL").sum()) if not review_table.empty else 0)
        human_accepted = accepted - auto_accepted
        issue_table = self._issue_rows() if not review_table.empty else pd.DataFrame()
        issue_total = len(issue_table)
        handled = (int(issue_table.review_mode.astype(str).eq("HUMAN").sum())
                   if issue_total and "review_mode" in issue_table else 0)
        remaining_issues = issue_total - handled
        has_run = bool(self.runs.currentData())
        confirmed = self._project_trials_confirmed()
        mismatch = self._trial_count_mismatch() if not has_run else None
        acknowledged = load_project_choices(self.root_provider()).get(
            "trial_count_ack", {}).get(self.recordings.currentData(), 0)
        if self._running:
            step = "running"
        elif self._editing_trials or mismatch and acknowledged != mismatch[1]:
            step = "trials"
        elif has_run and total and accepted == total:
            step = "freeze"
        elif has_run:
            step = "review"
        elif confirmed or self._repair_ready:
            step = "run"
        else:
            step = "trials"
        self._view_step = step
        expected = self._expected_repetitions()
        current = (len(self.proposal_regions) if step == "trials" else
                   len(self._trial_entries()))
        if expected is None:
            self.trial_count.setText(f"Current trials: {current}  ·  No protocol count specified")
        elif expected == current:
            self.trial_count.setText(
                f"✓ Expected repetitions: {expected}  ·  Current trials: {current}")
        else:
            self.trial_count.setText(
                f"⚠ Expected repetitions: {expected}  ·  Current trials: {current}. "
                "Review, merge, delete, or adjust the shaded trials before Alignment.")
        self.trial_count.setVisible(step == "trials")
        self.steps.setText({
            "trials": "● 1 Confirm trials   →   ○ 2 Run alignment   →   ○ 3 Review   →   ○ 4 Freeze",
            "run": "✓ 1 Confirm trials   →   ● 2 Run alignment   →   ○ 3 Review   →   ○ 4 Freeze",
            "running": "✓ 1 Confirm trials   →   ● 2 Alignment running   →   ○ 3 Review   →   ○ 4 Freeze",
            "review": "✓ 1 Confirm trials   →   ✓ 2 Run alignment   →   ● 3 Review   →   ○ 4 Freeze",
            "freeze": "✓ 1 Confirm trials   →   ✓ 2 Run alignment   →   ✓ 3 Review   →   ● 4 Freeze",
        }[step])
        self.run_progress.setVisible(step == "running")
        self.run_elapsed.setVisible(step == "running")
        self.primary.setEnabled(step != "running" and (
            step != "run" or self._preflight_ready) and (step != "trials" or
            bool(self.recordings.currentData())))
        if step == "trials":
            candidates = reviewed_trial_candidates(self.root_provider(),
                                                    self.recordings.currentData() or "")
            existing = self._trial_entries()
            if mismatch and existing and not self._editing_trials and acknowledged != mismatch[1]:
                self.action_hint.setText(
                    f"⚠ Trial count differs from protocol: expected {mismatch[0]}, "
                    f"confirmed {mismatch[1]}. Select adjacent fragments to merge, "
                    "or edit the shaded trials on the waveform.")
                self.primary.setText("CONFIRM TRIALS")
            elif existing and not self._editing_trials:
                self.action_hint.setText(
                    f"Recording {self.recordings.currentIndex() + 1} of {self.recordings.count()} · "
                    f"{len(existing)} repetitions confirmed. Continue to the next recording.")
                self.primary.setText("NEXT RECORDING TO CONFIRM")
            else:
                self.action_hint.setText(
                    f"Recording {self.recordings.currentIndex() + 1} of {self.recordings.count()} · "
                    f"{len(self.proposal_regions) or len(candidates)} shaded trials. "
                    "Select a trial, drag its edge, or use the correction buttons below the waveform.")
                self.primary.setText("CONFIRM TRIALS")
        elif step == "run":
            self.action_hint.setText("Trials confirmed. " + (
                "Ready to align." if self._preflight_ready else
                "Checking the prompt, reviewed segmentation and Alignment environment…"))
            self.primary.setText("RERUN ALIGNMENT FOR THIS RECORDING" if self._repair_ready
                                 else "RUN ALIGNMENT")
        elif step == "running":
            self.primary.setText("ALIGNMENT RUNNING…")
        elif step == "review":
            if issue_total and not remaining_issues:
                self.action_hint.setText(
                    f"REVIEW COMPLETE — {handled} / {issue_total} flagged cases handled. "
                    + (f"{issues} still need correction before freeze."
                       if issues else "Ready to finalize."))
            else:
                self.action_hint.setText(
                    ("⚠ ALIGNMENT COMPLETED WITH ISSUES. " if self._alignment_had_issues else
                     "✓ ALIGNMENT COMPLETE. ") + self._alignment_result_summary + "\n" +
                    f"{auto_accepted} structurally auto-passed · "
                    f"{remaining_issues} flagged cases await review. "
                    "Inspect the waveform and tiers, listen, then choose a review decision.")
            self.primary.setText(
                f"REVIEW {remaining_issues} {'ISSUE' if remaining_issues == 1 else 'ISSUES'}"
                if remaining_issues else
                f"REVISIT {issues} {'ISSUE' if issues == 1 else 'ISSUES'}"
                if issues else "REVIEW ALIGNMENT")
        else:
            self.action_hint.setText(
                f"✓ {auto_accepted} structurally auto-passed · {human_accepted} human accepted "
                f"of {total} trials. Alignment can now be frozen.")
            if self._is_current_frozen():
                if self._feature_return_pending:
                    self.action_hint.setText("✓ Alignment requirements are satisfied.")
                    self.primary.setText("RETURN TO ACOUSTIC FEATURES")
                else:
                    self.action_hint.setText("✓ Alignment is frozen. Use Edit Trials to start a reviewed revision.")
                    self.primary.setText("ALIGNMENT FROZEN")
                    self.primary.setEnabled(False)
            else:
                self.primary.setText("FREEZE ALIGNMENT")
        self.review_progress.setText(
            f"{human_accepted + issues} / {total} reviewed · "
            f"{auto_accepted} auto-passed · {human_accepted} human accepted · "
            f"{issues} needs review · {total} total"
            if has_run and total else "")
        if issue_total:
            self.review_progress.setText(self.review_progress.text() +
                f"  |  Flagged: {handled} / {issue_total} reviewed · "
                f"{human_accepted} accepted · {issues} still need review · "
                f"{remaining_issues} remaining")
        self.previous_issue.setVisible(reviewing := step in {"review", "freeze"})
        self.next_issue.setVisible(reviewing)
        self.view_auto_passed.setVisible(reviewing and auto_accepted > 0)
        self.next_action.setText({
            "trials": "Next: Run Alignment after confirming these trials.",
            "run": "Next: Run Alignment, then review flagged cases.",
            "running": "Alignment is running. Flagged cases can be reviewed afterward.",
            "review": "Next: Review flagged cases, then finalize Alignment.",
            "freeze": "Next: Freeze the accepted Alignment.",
        }[step])
        self.proposal_strip.setVisible(step == "trials" and bool(self.proposal_regions))
        self.trial_actions.setVisible(step == "trials")
        mergeable_pair = (len(self._selected_proposals) == 2 and
                          max(self._selected_proposals) - min(self._selected_proposals) == 1)
        mergeable_single = (len(self._selected_proposals) == 1 and
                            len(self.proposal_regions) > 1)
        self.merge_button.setEnabled(step == "trials" and
                                      (mergeable_pair or mergeable_single))
        self.split_button.setEnabled(step == "trials" and
                                     len(self._selected_proposals) == 1 and
                                     self._proposal_cursor is not None)
        self.delete_button.setEnabled(step == "trials" and
                                      len(self._selected_proposals) == 1)
        self.undo_button.setEnabled(step == "trials" and bool(self._proposal_history))
        self.accept_button.setVisible(reviewing)
        self.needs_review_button.setVisible(reviewing)
        self.keep_needs_review_button.setVisible(reviewing and
            self._current_review_status() == "NEEDS_REVIEW")
        self.play_token_button.setVisible(reviewing)
        self.token_detail.setVisible(reviewing)
        self.edit_trial_button.setVisible(reviewing)
        self.edit_alignment_button.setVisible(reviewing)
        self.reset_token_button.setVisible(reviewing and self._editing_alignment)
        self.reset_trial_button.setVisible(reviewing and self._editing_alignment)

    def _expected_repetitions(self) -> int | None:
        try:
            context = project_alignment_context(self.root_provider())
        except RuntimeError:
            return None
        prompt_id = self.record_stimulus.currentData() if hasattr(self, "record_stimulus") else ""
        prompt = next((item for item in (context.get("task") or {}).get("prompts", [])
                       if item["prompt_id"] == prompt_id), None)
        return prompt.get("expected_repetitions") if prompt else None

    def _primary_action(self) -> None:
        if self._view_step == "trials":
            if (self._trial_count_mismatch() and self._trial_entries()
                    and not self._editing_trials and not self._proposals_differ()):
                self._acknowledge_trial_count()
                return
            self._confirm_proposals()
        elif self._view_step == "run":
            if self._repair_ready:
                self.rerun_recording_requested.emit(
                    str(self.root_provider()), self.profile.text().strip(),
                    self._repair_record_id, self._repair_source_run_id)
            else:
                self._run()
        elif self._view_step == "review":
            if self._issue_rows(unhandled_only=True).empty:
                self._step_issue(1)
            else:
                self._navigate_unreviewed()
        elif self._view_step == "freeze":
            if self._is_current_frozen() and self._feature_return_pending:
                self.return_to_features_requested.emit()
            else:
                self.freeze_requested.emit(self.runs.currentData() or "")

    def _tick_run(self) -> None:
        elapsed = int(time.monotonic() - self._run_started_at)
        self.run_elapsed.setText(f"Elapsed {elapsed // 60:02d}:{elapsed % 60:02d}")

    def alignment_started(self) -> None:
        self._running = True
        self._run_started_at = time.monotonic()
        self._run_timer.start(1000)
        self.action_hint.setText("ALIGNMENT RUNNING — Preparing corpus…")
        self._show_step()

    def alignment_progress(self, done: int, total: int, message: str) -> None:
        if self._running:
            self.action_hint.setText("ALIGNMENT RUNNING — " + message)
            if total > 0:
                self.run_progress.setRange(0, total)
                self.run_progress.setValue(done)
            else:
                self.run_progress.setRange(0, 0)

    def alignment_finished(self, error: str = "") -> None:
        self._running = False
        self._run_timer.stop()
        self._show_step()
        if error:
            self.action_hint.setText("Alignment could not complete. " + error.splitlines()[0])

    def _navigate_unreviewed(self, *, after_current: bool = False) -> None:
        run_id = self.runs.currentData()
        if not run_id:
            return
        table = self._reviews(run_id)
        pending = self._issue_rows(unhandled_only=True)
        if pending.empty:
            self.action_hint.setText("REVIEW COMPLETE — no unhandled issues remain. "
                                     "Resolve any cases kept Needs Review before freezing.")
            self._show_step()
            return
        if after_current:
            current_record = str(self.recordings.currentData())
            current_trial = str((self.trials.currentData() or {}).get("trial_id", ""))
            rows = table.reset_index(drop=True)
            current_rows = rows.index[rows.recording_id.astype(str).eq(current_record)
                                      & rows.trial_id.astype(str).eq(current_trial)]
            if len(current_rows):
                later = pending.loc[pending.index > current_rows[0]]
                if not later.empty:
                    pending = later
        first = pending.iloc[0]
        record_index = self.recordings.findData(str(first.recording_id))
        if record_index >= 0:
            self.recordings.setCurrentIndex(record_index)
            for index in range(self.trials.count()):
                if str(self.trials.itemData(index)["trial_id"]) == str(first.trial_id):
                    self.trials.setCurrentIndex(index)
                    break

    def focus_trial(self, recording_id: str, trial_id: str) -> None:
        """Show a specific trial identified by structural freeze validation."""
        record_index = self.recordings.findData(str(recording_id))
        if record_index < 0:
            return
        self.recordings.setCurrentIndex(record_index)
        for index in range(self.trials.count()):
            entry = self.trials.itemData(index) or {}
            if str(entry.get("trial_id")) == str(trial_id):
                self.trials.setCurrentIndex(index)
                break

    def _confirm_proposals(self) -> None:
        record_id = self.recordings.currentData()
        if not record_id or self.runs.currentData():
            return
        if self._trial_entries() and not self._editing_trials and not self._proposals_differ():
            choices = load_project_choices(self.root_provider())
            for index in range(self.recordings.count()):
                if not choices.get("trials", {}).get(self.recordings.itemData(index)):
                    self.recordings.setCurrentIndex(index)
                    return
        bounds = [tuple(region.getRegion()) for region in self.proposal_regions]
        if not bounds:
            self.action_hint.setText("No repetition was proposed. Review the speech bounds in Advanced / Diagnostics.")
            return
        choices = load_project_choices(self.root_provider())
        prompt_id = self.record_stimulus.currentData() or ""
        expected = self._expected_repetitions()
        if expected is not None and expected != len(bounds):
            answer = QMessageBox.question(
                self, "Confirm trial count",
                f"Protocol expects {expected} repetitions; you are confirming {len(bounds)}. "
                "Is this the actual number of valid repetitions in this recording?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        entries = [{"trial_id": f"trial_{number:03d}", "start_sec": left,
                    "end_sec": right, "prompt_id": prompt_id}
                   for number, (left, right) in enumerate(sorted(bounds), 1)]
        choices.setdefault("trials", {})[record_id] = entries
        context = project_alignment_context(self.root_provider())
        context["choices"] = choices
        context["records"] = [item for item in context["records"]
                              if item["recording_id"] == record_id]
        from vslp.acoustic.alignment.trials import validate_trial_choices
        try:
            validate_trial_choices(self.root_provider(), context)
        except ValueError as exc:
            self.action_hint.setText("Trial bounds need correction: " + str(exc))
            return
        save_project_choices(self.root_provider(), choices)
        if expected is not None and expected != len(entries):
            choices.setdefault("trial_count_ack", {})[record_id] = len(entries)
            save_project_choices(self.root_provider(), choices)
        if self._editing_trials and self._repair_source_run_id:
            frozen = (self.root_provider() / "acoustic" / "004_alignment" / "final" /
                      "final_alignment_manifest.json")
            if (frozen.is_file() and json.loads(frozen.read_text(encoding="utf-8")).get(
                    "alignment_run_id") == self._repair_source_run_id):
                choices.setdefault("pending_trial_revision", {})[record_id] = {
                    "source_run_id": self._repair_source_run_id,
                    "state": "AWAITING_TARGETED_RERUN"}
                save_project_choices(self.root_provider(), choices)
            else:
                mark_recording_stale(self.root_provider(), self._repair_source_run_id, record_id)
            self._repair_ready = True
            self._editing_trials = False
        self._recording_changed()
        self._request_preflight()
        if self._project_trials_confirmed():
            self._show_step()
        else:
            for index in range(self.recordings.count()):
                if not choices["trials"].get(self.recordings.itemData(index)):
                    self.recordings.setCurrentIndex(index)
                    break

    def _acknowledge_trial_count(self) -> None:
        mismatch = self._trial_count_mismatch()
        if not mismatch:
            return
        answer = QMessageBox.question(
            self, "Review repetitions",
            f"Protocol expects {mismatch[0]} repetitions; {mismatch[1]} are confirmed. "
            "Have you reviewed this difference on the waveform?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            choices = load_project_choices(self.root_provider())
            choices.setdefault("trial_count_ack", {})[self.recordings.currentData()] = mismatch[1]
            save_project_choices(self.root_provider(), choices)
            self._show_step()

    def _check_environment(self) -> None:
        self.environment.setText("Checking MFA environment...")
        self.environment_requested.emit(self.profile.text().strip())

    def show_environment_result(self, environment: dict) -> None:
        if environment.get("status") == "AVAILABLE":
            self.environment.setText("Ready")
            resources = environment.get("resources", {})
            self.model.setText(resources.get("acoustic", {}).get("identity", ""))
            self.dictionary.setText(resources.get("dictionary", {}).get("identity", ""))
        else:
            self.environment.setText(environment.get("status", "Environment unavailable"))

    def show_preflight_result(self, result: dict) -> None:
        environment = result.get("environment", {})
        self._preflight_ready = result.get("status") == "READY"
        if environment:
            self.show_environment_result(environment)
        if self._preflight_ready:
            self.preflight.setText("Ready — reviewed segmentation, prompt, speaker IDs and MFA environment available")
            gaps = result.get("repetition_gaps", {})
            self.readiness.setText(
                f"Ready to run Alignment · {len(gaps)} recording(s) have fewer confirmed "
                "trials than the protocol expects; missing trials are not fabricated."
                if gaps else "Ready to run Alignment")
        else:
            issue = result.get("issue", "ACTION_REQUIRED")
            if issue.startswith("TRIAL_BOUNDARIES_REQUIRED"):
                self.readiness.setText(
                    "Confirm each sentence repetition using reviewed speech bounds below")
            elif issue.startswith("INVALID_TRIAL_BOUNDARIES") or issue.startswith("TRIAL_CROSSES_EXCLUSION"):
                self.readiness.setText("Trial bounds need correction before Alignment")
            else:
                self.readiness.setText("Alignment needs attention — see details under Advanced")
            if issue == "SPEAKER_ID_REQUIRED":
                missing = result.get("missing_speaker_recordings", [])
                count = len(missing)
                self.speaker_status.setText(f"Missing for {count} recording{'s' if count != 1 else ''}")
                self.preflight.setText(
                    "Alignment is not ready yet.\n"
                    "✓ Reviewed segmentation\n✓ Prompt resolved\n"
                    f"✗ Speaker ID missing for {count} recording{'s' if count != 1 else ''}.\n"
                    "Enter the missing deidentified Speaker IDs in the table or use Paste Speaker IDs.")
            elif issue == "OOV":
                words = ", ".join(result.get("oov_words", []))
                self.preflight.setText(f"Alignment cannot run. Dictionary words missing: {words}")
            else:
                messages = {
                    "MISSING_FROZEN_SEGMENTATION": "Freeze reviewed segmentation before Alignment.",
                    "MISSING_REVIEWED_RECORDING": "A reviewed recording is missing or unreadable.",
                    "MISSING_CANONICAL_PROMPT": "This task has no approved prompt text.",
                    "PROMPT_SELECTION_REQUIRED": "Select the stimulus for each recording.",
                    "TASK_NOT_REGISTERED": "The Setup task is not a registered Alignment task. "
                                           "Check the task selected in Setup.",
                }
                self.preflight.setText("Alignment is not ready yet. " + messages.get(issue, issue))
        self.run_button.setEnabled(self.source.currentData() == "external" or self._preflight_ready)
        self._show_step()

    def show_self_test_result(self, result: object) -> None:
        passed = getattr(result, "result", "FAIL") == "PASS"
        if passed:
            self.self_test_summary.setText(
                "MFA SELF-TEST\n✓ Corpus created\n✓ Dictionary validation\n"
                "✓ MFA validation\n✓ Corpus alignment\n✓ Word parsing\n"
                "✓ Phone parsing\n✓ Structural validation\nSELF-TEST PASSED\n"
                f"{result.word_count} words · {result.phone_count} phones · "
                f"{result.elapsed_time_sec:.1f} s")
        else:
            self.self_test_summary.setText(
                "MFA SELF-TEST FAILED\n"
                f"Step: {getattr(result, 'failure_step', 'Unknown')}\n"
                f"Reason: {getattr(result, 'failure_message', 'Unknown failure')}")
        self._self_test_diagnostics_path = getattr(result, "diagnostic_log_path", "")
        self.view_self_test_diagnostics.setVisible(bool(self._self_test_diagnostics_path))

    def _open_self_test_diagnostics(self) -> None:
        if self._self_test_diagnostics_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._self_test_diagnostics_path))

    def _run(self) -> None:
        if self.source.currentData() == "mfa":
            if self._preflight_ready:
                self.run_task_requested.emit(str(self.root_provider()), self.profile.text().strip())
            return
        supplied_prompt = self.prompt.text().strip()
        try:
            prompt_paths = {} if supplied_prompt else build_task_prompt_manifests(self.root_provider())
        except ValueError as exc:
            QMessageBox.warning(self, "Alignment prompt required",
                "Choose an approved stimulus for every recording before importing Alignment. "
                f"Technical reason: {exc}")
            return
        self.run_requested.emit(AlignmentConfig(
            source=self.source.currentData(),
            prompt_manifest_path=supplied_prompt or next(iter(prompt_paths.values())),
            words_csv=self.words.text().strip(), phones_csv=self.phones.text().strip(),
            acoustic_model=self.model.text().strip(),
            acoustic_model_version=self.model_version.text().strip(),
            dictionary=self.dictionary.text().strip(),
            dictionary_version=self.dictionary_version.text().strip(),
            mfa_profile_path=self.profile.text().strip(),
            speaker_manifest_path=self.speakers.text().strip(),
            transcript_overrides_path=self.overrides.text().strip(),
            recording_prompt_manifest_paths=prompt_paths or None))

    def _stimulus_changed(self) -> None:
        if self._loading_assignments:
            return
        context = project_alignment_context(self.root_provider())
        prompt, issue = resolve_prompt(context, self.stimulus.currentData() or "")
        self.prompt_display.setText(prompt["exact_expected_text"] if prompt else issue)
        if prompt:
            choices = load_project_choices(self.root_provider())
            for record in context["records"]:
                entry = choices.setdefault("recordings", {}).setdefault(record["recording_id"], {})
                entry["prompt_id"] = prompt["prompt_id"]
                entry.pop("alignment_transcript", None)
                entry.pop("transcript_reason", None)
            save_project_choices(self.root_provider(), choices)
        self._preflight_ready = False
        self.run_button.setEnabled(False)
        self._request_preflight()

    def _save_assignments(self, item: QTableWidgetItem) -> None:
        if self._loading_assignments:
            return
        choices = load_project_choices(self.root_provider())
        row, column = item.row(), item.column()
        if column not in {2, 3, 4}:
            return
        identity = self.assignments.item(row, 0).data(Qt.UserRole)
        entry = choices.setdefault("recordings", {}).setdefault(identity, {})
        field = {2: "speaker_id", 3: "alignment_transcript", 4: "transcript_reason"}[column]
        if column == 2:
            authoritative = next((record["speaker_id"] for record in
                project_alignment_context(self.root_provider())["records"]
                if record["recording_id"] == identity), "")
            if item.text().strip() != authoritative:
                entry[field] = item.text().strip()
        else:
            entry[field] = item.text().strip()
        save_project_choices(self.root_provider(), choices)
        self._update_speaker_status()
        self._preflight_ready = False
        self.run_button.setEnabled(False)
        self._request_preflight()

    def _missing_speaker_rows(self) -> list[int]:
        return [row for row in range(self.assignments.rowCount())
                if not self.assignments.item(row, 2).text().strip()]

    def _update_speaker_status(self) -> None:
        missing = self._missing_speaker_rows()
        self.speaker_status.setText(
            f"Missing for {len(missing)} recording{'s' if len(missing) != 1 else ''} "
            "— highlighted empty Speaker ID cells" if missing else
            f"Available for all {self.assignments.rowCount()} recordings")
        previous = self._loading_assignments
        self._loading_assignments = True
        for row in range(self.assignments.rowCount()):
            item = self.assignments.item(row, 2)
            item.setBackground(QBrush(QColor("#FFE5E5" if row in missing else "#FFFFFF")))
        self._loading_assignments = previous

    def _paste_speaker_ids(self) -> None:
        missing = self._missing_speaker_rows()
        if not missing:
            return
        pasted, accepted = QInputDialog.getMultiLineText(
            self, "Paste deidentified Speaker IDs",
            f"Paste {len(missing)} Speaker IDs, one per missing table row in displayed order. "
            "Repeat an ID only for recordings from the same speaker:")
        if not accepted:
            return
        identities = [line.strip() for line in pasted.splitlines()]
        if len(identities) != len(missing) or any(
                not re.fullmatch(r"[A-Za-z0-9_]+", identity) for identity in identities):
            QMessageBox.warning(self, "Speaker IDs not applied",
                f"Enter exactly {len(missing)} nonempty IDs, one per line. "
                "Use letters, numbers and underscores only.")
            return
        self._loading_assignments = True
        for row, identity in zip(missing, identities):
            self.assignments.item(row, 2).setText(identity)
        self._loading_assignments = False
        choices = load_project_choices(self.root_provider())
        for row, identity in zip(missing, identities):
            record_id = self.assignments.item(row, 0).data(Qt.UserRole)
            choices.setdefault("recordings", {}).setdefault(record_id, {})["speaker_id"] = identity
        save_project_choices(self.root_provider(), choices)
        self._update_speaker_status()
        self._request_preflight()

    def _save_record_prompt(self, identity: str, prompt_id: str) -> None:
        if self._loading_assignments:
            return
        choices = load_project_choices(self.root_provider())
        entry = choices.setdefault("recordings", {}).setdefault(identity, {})
        entry["prompt_id"] = prompt_id
        entry.pop("alignment_transcript", None)
        entry.pop("transcript_reason", None)
        save_project_choices(self.root_provider(), choices)
        context = project_alignment_context(self.root_provider())
        selected, _issue = resolve_prompt(context, prompt_id)
        self._loading_assignments = True
        for row in range(self.assignments.rowCount()):
            if self.assignments.item(row, 0).data(Qt.UserRole) == identity:
                self.assignments.item(row, 3).setText(
                    selected["exact_expected_text"] if selected else "")
                self.assignments.item(row, 4).setText("")
                break
        self._loading_assignments = False
        self._preflight_ready = False
        self.run_button.setEnabled(False)
        self._request_preflight()

    def _save_reviewer(self) -> None:
        choices = load_project_choices(self.root_provider())
        choices["reviewer_id"] = self.reviewer.text().strip()
        save_project_choices(self.root_provider(), choices)
        self._preflight_ready = False
        self.run_button.setEnabled(False)
        self._request_preflight()

    def _request_preflight(self) -> None:
        if self.source.currentData() != "mfa":
            return
        context = project_alignment_context(self.root_provider())
        if context.get("task") and not context["task"]["alignment_applicable"]:
            self._preflight_ready = False
            self.run_button.setEnabled(False)
            self.readiness.setText(
                "Linguistic Alignment is not required. Confirm repeated task trials here if needed; "
                "DDK continues through reviewed DDK events.")
            return
        self.preflight.setText("Checking prerequisites...")
        self._preflight_timer.start(0)

    def refresh(self) -> None:
        final_manifest = (self.root_provider() / "acoustic" / "004_alignment" / "final" /
                          "final_alignment_manifest.json")
        if final_manifest.is_file():
            final_state = json.loads(final_manifest.read_text(encoding="utf-8"))
            self.frozen_status.setText(
                f"Frozen Alignment: {final_state.get('alignment_run_id', 'unknown')}")
        else:
            self.frozen_status.setText("No frozen Alignment")
        context = project_alignment_context(self.root_provider())
        if context.get("reviewed") and (context.get("task") or {}).get("alignment_applicable"):
            from vslp.acoustic.alignment.trials import structurally_proposed_trials
            before = json.dumps(context["choices"], sort_keys=True)
            context["choices"] = structurally_proposed_trials(self.root_provider(), context)
            if json.dumps(context["choices"], sort_keys=True) != before:
                save_project_choices(self.root_provider(), context["choices"])
        if context.get("issue"):
            self.task_label.setText("Select a project in Setup")
            self.run_button.setEnabled(False)
        else:
            self.task_label.setText(str(context["task_name"]))
            self.task_brief.setText(f"Task: {context['task_name']}")
            self.reviewer.setText(context["choices"].get("reviewer_id", ""))
            self.review_status.setText("Available" if context["reviewed"] else "Not frozen")
            self.recording_count.setText(f"Recordings: {len(context['records'])}")
            prompts = context["task"]["prompts"] if context["task"] else []
            self._loading_assignments = True
            self.stimulus.clear()
            for prompt in prompts:
                self.stimulus.addItem(f"{prompt['exact_expected_text']} · {prompt['prompt_version']}",
                                      prompt["prompt_id"])
            if len(prompts) == 1:
                self.stimulus.setCurrentIndex(0)
            elif prompts:
                selected = next((item["prompt_id"] for item in context["records"]
                                 if item["prompt_id"]), "")
                index = self.stimulus.findData(selected)
                self.stimulus.setCurrentIndex(index)
            prompt, issue = resolve_prompt(context, self.stimulus.currentData() or "")
            self.prompt_display.setText(
                prompt["exact_expected_text"] if prompt else
                ("Choose each recording's stimulus below" if len(prompts) > 1 else
                 "No linguistic prompt required" if issue == "ALIGNMENT_NOT_APPLICABLE" else
                 "Setup task is not registered for Alignment" if issue ==
                 "TASK_NOT_REGISTERED" else issue))
            expected = prompt.get("expected_repetitions") if prompt else None
            self.task_brief.setText(
                f"Task: {context['task_name']}  ·  Stimulus: {self.prompt_display.text()}"
                + (f"  ·  Protocol: {expected} repetitions expected" if expected else ""))
            self.stimulus.setVisible(False)
            self.assignments.setRowCount(len(context["records"]))
            for row, record in enumerate(context["records"]):
                filename = QTableWidgetItem(record["file_name"])
                filename.setData(Qt.UserRole, record["recording_id"])
                filename.setFlags(filename.flags() & ~Qt.ItemIsEditable)
                self.assignments.setItem(row, 0, filename)
                choice = QComboBox()
                if len(prompts) > 1:
                    choice.addItem("Select stimulus…", "")
                for option in prompts:
                    choice.addItem(option["exact_expected_text"], option["prompt_id"])
                if prompts:
                    if len(prompts) == 1:
                        choice.setCurrentIndex(0)
                    else:
                        choice.setCurrentIndex(max(0, choice.findData(record["prompt_id"])))
                choice.currentIndexChanged.connect(
                    lambda _index, selector=choice, identity=record["recording_id"]:
                    self._save_record_prompt(identity, selector.currentData()))
                self.assignments.setCellWidget(row, 1, choice)
                for column, value in ((2, record["speaker_id"]),
                                      (3, record["alignment_transcript"] or
                                       (prompt["exact_expected_text"] if prompt else "")),
                                      (4, record["transcript_reason"])):
                    self.assignments.setItem(row, column, QTableWidgetItem(value))
            self._loading_assignments = False
            self._update_speaker_status()
            key = f"{self.root_provider()}:{context['task_id']}:{len(context['records'])}"
            if key != self._last_preflight_key:
                self._last_preflight_key = key
                self._request_preflight()
        decisions = (self.root_provider() / "acoustic" / "003_segmentation_review" / "final" /
                     "final_segmentation_decisions.csv")
        if decisions.is_file():
            table = pd.read_csv(decisions, usecols=["final_decision"])
            count = table.final_decision.isin(["KEEP_AUTO", "KEEP_MANUAL"]).sum()
            self.recording_count.setText(f"Recordings: {count}")
        try:
            runs = list_alignment_runs(self.root_provider())
        except RuntimeError:
            return
        selected = self.runs.currentData()
        stored = load_project_choices(self.root_provider()).get("selected_run_id")
        self.runs.blockSignals(True)
        self.runs.clear()
        for run in runs:
            self.runs.addItem(f"{run['created_utc'][:19]} · {run['source']} · {run['alignment_run_id']}",
                              run["alignment_run_id"])
        index = self.runs.findData(stored or selected)
        self.runs.setCurrentIndex(index if index >= 0 else self.runs.count() - 1)
        self.runs.blockSignals(False)
        self._run_changed()
        if self.recordings.count() == 0 and context.get("records"):
            self.recordings.blockSignals(True)
            for record in context["records"]:
                self.recordings.addItem(record["file_name"], record["recording_id"])
            self.recordings.blockSignals(False)
            self._recording_changed()
        self._show_step()

    def _run_changed(self) -> None:
        run_id = self.runs.currentData()
        self._review_cache = None
        self._review_cache_run = ""
        self.recordings.clear()
        if run_id and run_id != self._repair_source_run_id:
            self._repair_ready = False
            self._editing_trials = False
            self._editing_alignment = False
        self._alignment_result_summary = ""
        self._alignment_had_issues = False
        if not run_id:
            self.freeze_button.setEnabled(False)
            self._show_step()
            return
        choices = load_project_choices(self.root_provider())
        if choices.get("selected_run_id") != run_id:
            choices["selected_run_id"] = run_id
            save_project_choices(self.root_provider(), choices)
        self.freeze_button.setEnabled(choices.get("inspected_run_id") == run_id)
        root = self.root_provider()
        manifest_path = root / "acoustic" / "004_alignment" / "runs" / run_id / "logs" / "stage_manifest.json"
        if not manifest_path.is_file():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("rerun_recording_id") and manifest.get("reused_alignment_run_id"):
            pending = load_project_choices(self.root_provider())
            identity = manifest["rerun_recording_id"]
            if pending.get("pending_trial_revision", {}).get(identity, {}).get(
                    "source_run_id") == manifest["reused_alignment_run_id"]:
                pending["pending_trial_revision"].pop(identity, None)
                save_project_choices(self.root_provider(), pending)
        if manifest.get("trial_contract_version") == "reviewed_trials_v1":
            self._update_freeze_state()
        # Historical run diagnostics belong to the selected run, not current preflight.
        self.technical.setText(
            f"Provider: {manifest.get('provider', '—')} {manifest.get('provider_version', '')}\n"
            f"Mode: {manifest.get('provider_mode', '—')}\n"
            f"Model: {manifest.get('acoustic_model', '—')}\n"
            f"Dictionary: {manifest.get('dictionary', '—')}\n"
            f"Phone set: {manifest.get('phone_set', '—')}\n"
            f"Working sample rate: {manifest.get('working_sample_rate_hz', '—')} Hz\n"
            f"Commands: {manifest.get('provider_commands', [])}")
        diagnostics = pd.read_csv(manifest["diagnostics_path"], keep_default_na=False)
        self._alignment_had_issues = bool(manifest.get("number_failed", 0) or
                                          manifest.get("number_partial", 0))
        self._alignment_result_summary = (
            f"{manifest['number_aligned']} recordings aligned · "
            f"{len(self._reviews(run_id))} trials · "
            f"{int(diagnostics.n_words.sum())} words · "
            f"{int(diagnostics.n_phones.sum())} phones. ")
        self.summary.setText(
            f"Aligned {manifest['number_aligned']} / {manifest['number_attempted']}  ·  "
            f"Partial {manifest.get('number_partial', 0)}  ·  Failed {manifest['number_failed']}  ·  "
            f"Needs review {manifest['number_below_coverage_threshold']}  ·  "
            f"Words {int(diagnostics.n_words.sum())}  ·  Phones {int(diagnostics.n_phones.sum())}")
        for row in diagnostics.itertuples():
            self.recordings.addItem(f"{row.file_name}  ·  {row.status}  ·  coverage {row.word_coverage:.0%}",
                                    str(row.recording_id))
        self._recording_changed()
        self._show_step()

    def _new_plan(self) -> None:
        self.runs.setCurrentIndex(-1)
        self.recordings.blockSignals(True)
        self.recordings.clear()
        for record in project_alignment_context(self.root_provider()).get("records", []):
            self.recordings.addItem(record["file_name"], record["recording_id"])
        self.recordings.blockSignals(False)
        self._recording_changed()

    def _edit_trial_boundaries(self) -> None:
        run_id = self.runs.currentData()
        record_id = self.recordings.currentData()
        if not run_id or not record_id:
            return
        self._repair_source_run_id = str(run_id)
        self._repair_record_id = str(record_id)
        self._repair_ready = False
        self._editing_trials = True
        self._editing_alignment = False
        self._new_plan()
        index = self.recordings.findData(record_id)
        if index >= 0:
            self.recordings.setCurrentIndex(index)
        self._show_step()

    def _toggle_alignment_edit(self) -> None:
        if self._is_current_frozen():
            QMessageBox.information(self, "Frozen Alignment",
                "This frozen Alignment is immutable. Create a new Alignment run before revising it.")
            return
        self._editing_alignment = not self._editing_alignment
        self.edit_alignment_button.setText("Finish boundary editing" if self._editing_alignment
                                           else "Edit Alignment")
        self._recording_changed()
        if self._editing_alignment:
            self.action_hint.setText(
                "Click a word on the yellow tier or a phone on the green tier. "
                "Drag the highlighted onset/offset handles, then Accept Alignment.")

    def _is_current_frozen(self) -> bool:
        path = (self.root_provider() / "acoustic" / "004_alignment" / "final" /
                "final_alignment_manifest.json")
        return (path.is_file() and
                json.loads(path.read_text(encoding="utf-8")).get("alignment_run_id") ==
                self.runs.currentData())

    def _save_token_region(self, tier: str, index: int, trial_id: str,
                           region: pg.LinearRegionItem) -> None:
        if not self._editing_alignment:
            return
        try:
            set_correction(self.root_provider(), self.runs.currentData(),
                           self.recordings.currentData(), trial_id, tier, index,
                           *map(float, region.getRegion()))
            self._review_cache = None
            self._recording_changed()
        except ValueError as exc:
            QMessageBox.warning(self, "Boundary correction", str(exc))
            self._recording_changed()

    def _reset_selected_token(self) -> None:
        if not self._selected_token_key or not self.trials.currentData():
            return
        tier, index = self._selected_token_key
        reset_corrections(self.root_provider(), self.runs.currentData(),
                          self.recordings.currentData(),
                          str(self.trials.currentData()["trial_id"]), tier, index)
        self._review_cache = None
        self._recording_changed()

    def _reset_current_trial(self) -> None:
        if not self.trials.currentData():
            return
        reset_corrections(self.root_provider(), self.runs.currentData(),
                          self.recordings.currentData(),
                          str(self.trials.currentData()["trial_id"]))
        self._review_cache = None
        self._recording_changed()

    def _mark_inspected(self) -> None:
        run_id = self.runs.currentData()
        if not run_id:
            return
        choices = load_project_choices(self.root_provider())
        choices["inspected_run_id"] = run_id
        save_project_choices(self.root_provider(), choices)
        self.freeze_button.setEnabled(True)
        self._recording_changed()

    def _segmentation_display_table(self, name: str) -> pd.DataFrame:
        path = (self.root_provider() / "acoustic" / "003_segmentation_review" /
                "final" / name)
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns)
        cached = self._segmentation_display_tables.get(str(path))
        if cached is not None and cached[0] == identity:
            return cached[1]
        table = pd.read_csv(path, keep_default_na=False)
        self._segmentation_display_tables[str(path)] = identity, table
        return table

    def _recording_changed(self) -> None:
        self._waveform_request += 1
        if self.player is not None:
            self.player.stop()
        self._selected_token = None
        self._selected_token_key = None
        self._edit_handle = None
        self.plot.clear()
        self.plot.addItem(self.play_cursor)
        self._tokens = []
        record_id = self.recordings.currentData()
        run_id = self.runs.currentData()
        if not record_id:
            context = project_alignment_context(self.root_provider())
            if context.get("records") and self.recordings.count() == 0:
                self.recordings.blockSignals(True)
                for record in context["records"]:
                    self.recordings.addItem(record["file_name"], record["recording_id"])
                self.recordings.blockSignals(False)
                record_id = self.recordings.currentData()
        if not record_id:
            return
        context = project_alignment_context(self.root_provider())
        prompts = context.get("task", {}).get("prompts", []) if context.get("task") else []
        selection = next((item["prompt_id"] for item in context.get("records", [])
                          if item["recording_id"] == record_id), "")
        self.record_stimulus.blockSignals(True)
        self.record_stimulus.clear()
        if len(prompts) > 1:
            self.record_stimulus.addItem("Select this recording's stimulus", "")
        for prompt in prompts:
            self.record_stimulus.addItem(prompt["exact_expected_text"], prompt["prompt_id"])
        index = self.record_stimulus.findData(selection)
        self.record_stimulus.setCurrentIndex(index if index >= 0 else 0)
        self.record_stimulus.setVisible(len(prompts) > 1)
        self.record_stimulus.blockSignals(False)
        selected_prompt = next((item for item in prompts
                                if item["prompt_id"] == self.record_stimulus.currentData()), None)
        if selected_prompt:
            self.prompt_display.setText(selected_prompt["exact_expected_text"])
            expected = selected_prompt.get("expected_repetitions")
            self.task_brief.setText(
                f"Task: {context['task_name']}  ·  Stimulus: {self.prompt_display.text()}"
                + (f"  ·  Protocol: {expected} repetitions expected" if expected else ""))
        root = self.root_provider()
        decisions = self._segmentation_display_table("final_segmentation_decisions.csv")
        matching = decisions.loc[decisions.recording_id.astype(str).eq(record_id)]
        if matching.empty:
            return
        if "analysis_wav_path" not in matching:
            self._refresh_trials()
            return
        path = Path(str(matching.iloc[0].analysis_wav_path))
        if not path.is_file():
            self._refresh_trials()
            return
        self._ensure_player().setSource(QUrl.fromLocalFile(str(path.resolve())))
        duration = sf.info(path).duration
        self.plot.setLimits(xMin=0, xMax=duration)
        self.plot.setXRange(0, duration, padding=.02)
        self.plot.setYRange(-1.8, 1.2, padding=0)
        request = self._waveform_request
        if path.stat().st_size > 2_000_000:
            owner = weakref.ref(self)
            future = _DISPLAY_EXECUTOR.submit(self._display_cache.get, path)

            def delivered(done):
                widget = owner()
                if widget is None:
                    return
                try:
                    payload = (request, done.result(), "")
                except (OSError, RuntimeError, ValueError) as exc:
                    payload = (request, None, str(exc))
                try:
                    widget.waveform_ready.emit(payload)
                except RuntimeError:
                    pass

            future.add_done_callback(delivered)
        else:
            self._apply_waveform((request, self._display_cache.get(path), ""))
        start = float(matching.iloc[0].analysis_start_sec)
        end = float(matching.iloc[0].analysis_end_sec)
        for left, right in ((0, start), (end, duration)):
            if right > left:
                self.plot.addItem(pg.LinearRegionItem([left, right],
                    brush=pg.mkBrush("#59616C55"), movable=False))
        interval_path = root / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_intervals.csv"
        if interval_path.is_file():
            intervals = self._segmentation_display_table("final_segmentation_intervals.csv")
            excluded = intervals.loc[intervals.recording_id.astype(str).eq(record_id)
                                     & intervals.segment_role.eq("manual_exclusion")]
            for row in excluded.itertuples():
                left, right = float(row.start_sec), float(row.end_sec)
                exclusion = pg.LinearRegionItem([left, right],
                    brush=pg.mkBrush("#C9696999"), movable=False)
                exclusion.setAcceptedMouseButtons(Qt.NoButton)
                exclusion.setZValue(6)
                self.plot.addItem(exclusion)
                tag = pg.TextItem("EXCLUDED", color="#F2A0A0", anchor=(0.5, 0.5))
                tag.setPos((left + right) / 2, -.88)
                tag.setZValue(7)
                self.plot.addItem(tag)
        candidates = (sorted((float(row.start_sec), float(row.end_sec))
                             for row in intervals.loc[intervals.recording_id.astype(str).eq(
                                 record_id) & intervals.segment_role.eq("speech")].itertuples())
                      if interval_path.is_file() else [])
        self.proposal_regions = []
        self.proposal_labels = []
        self._selected_proposal = 0
        self._selected_proposals = set()
        self._proposal_history.clear()
        self._proposal_last_bounds = []
        self._add_mode = False
        self.add_button.setText("ADD TRIAL")
        self._proposal_cursor = None
        while self.proposal_strip_layout.count():
            item = self.proposal_strip_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not run_id:
            proposed = ([(float(item["start_sec"]), float(item["end_sec"]))
                         for item in self._trial_entries()] or candidates)
            self._rebuild_proposals(proposed)
        self.speech_candidates.blockSignals(True)
        self.speech_candidates.clear()
        for number, (left, right) in enumerate(candidates, 1):
            self.speech_candidates.addItem(
                f"Speech {number}: {left:.2f}–{right:.2f} s", (left, right))
        self.speech_candidates.blockSignals(False)
        self._candidate_changed()
        self._refresh_trials()
        for number in range(self.trials.count() if run_id else 0):
            trial = self.trials.itemData(number)
            region = pg.LinearRegionItem(
                [float(trial["start_sec"]), float(trial["end_sec"])],
                brush=pg.mkBrush("#6C8ECA44"), movable=not bool(run_id))
            self.plot.addItem(region)
            label = pg.TextItem(f"Trial {number + 1}", color="#C4D7FC", anchor=(0.5, 0.5))
            label.setPos((float(trial["start_sec"]) + float(trial["end_sec"])) / 2, -1.72)
            self.plot.addItem(label)
        base = root / "acoustic" / "004_alignment" / "runs" / str(run_id) / "tables"
        if not run_id:
            return
        word_view, phone_view = reviewed_tables(root, str(run_id))
        links_path = base / "alignment_trial_tokens.csv"
        links = pd.read_csv(links_path, keep_default_na=False) if links_path.is_file() else pd.DataFrame()
        trial_id = str((self.trials.currentData() or {}).get("trial_id", ""))
        corrected = load_corrections(root, str(run_id))
        for kind, table, color in (("Word", word_view, "#F1C66D"),
                                   ("Phone", phone_view, "#94CCAD")):
            tier = kind.lower()
            for row in table.loc[table.recording_id.astype(str).eq(record_id)].itertuples():
                label = str(getattr(row, "word", getattr(row, "phone", "")))
                index = int(getattr(row, f"{tier}_index"))
                link = (links.loc[links.recording_id.astype(str).eq(record_id)
                                  & links.token_type.eq(tier)
                                  & links.token_index.astype(int).eq(index)]
                        if not links.empty else pd.DataFrame())
                owner = str(link.iloc[0].trial_id) if len(link) == 1 else ""
                manual = not corrected.loc[
                    corrected.recording_id.astype(str).eq(record_id)
                    & corrected.tier.eq(tier)
                    & corrected.token_index.astype(int).eq(index)].empty
                self._tokens.append((float(row.start_sec), float(row.end_sec), kind, label,
                                     "Manually corrected" if manual else "MFA", index, owner))
                region = pg.LinearRegionItem([float(row.start_sec), float(row.end_sec)],
                                             brush=pg.mkBrush(color + "35"),
                                             movable=False)
                self.plot.addItem(region)
                tag = pg.TextItem(label, color=color, anchor=(0.5, 0.5))
                tag.setPos((float(row.start_sec) + float(row.end_sec)) / 2,
                           -1.14 if kind == "Word" else -1.48)
                self.plot.addItem(tag)
        self.token_detail.setText("Yellow: words  ·  Green: phones  ·  Times on original recording")
        self._trial_changed()
        self._show_step()

    def _apply_waveform(self, payload: tuple) -> None:
        request, waveform, error = payload
        if request != self._waveform_request:
            return
        if error or waveform is None:
            self.token_detail.setText("Waveform display unavailable: " + error)
            return
        times, samples, _duration = waveform
        curve = self.plot.plot(times, samples, pen=pg.mkPen("#8AB9D7", width=1))
        curve.setZValue(-10)

    def _adjust_trial(self, trial_id: str, region: pg.LinearRegionItem) -> None:
        record_id = self.recordings.currentData()
        choices = load_project_choices(self.root_provider())
        entries = choices.get("trials", {}).get(record_id, [])
        entry = next((item for item in entries if str(item["trial_id"]) == trial_id), None)
        if entry is None:
            return
        old = (entry["start_sec"], entry["end_sec"])
        entry["start_sec"], entry["end_sec"] = map(float, region.getRegion())
        context = project_alignment_context(self.root_provider())
        context["choices"] = choices
        context["records"] = [item for item in context["records"]
                              if item["recording_id"] == record_id]
        from vslp.acoustic.alignment.trials import validate_trial_choices
        try:
            validate_trial_choices(self.root_provider(), context)
        except ValueError as exc:
            entry["start_sec"], entry["end_sec"] = old
            region.setRegion(old)
            self.action_hint.setText("Trial bounds need correction: " + str(exc))
            return
        save_project_choices(self.root_provider(), choices)
        self._request_preflight()

    def _token_clicked(self, event) -> None:
        if not self.plot.sceneBoundingRect().contains(event.scenePos()):
            return
        time = self.plot.plotItem.vb.mapSceneToView(event.scenePos()).x()
        if self._view_step == "trials":
            self._proposal_cursor = time
            for index, region in enumerate(self.proposal_regions):
                left, right = region.getRegion()
                if left <= time <= right:
                    self._select_proposal(index, bool(event.modifiers() & Qt.ControlModifier))
                    break
            self._show_step()
        y = self.plot.plotItem.vb.mapSceneToView(event.scenePos()).y()
        wanted = "Phone" if y < -1.3 else "Word"
        candidates = [item for item in self._tokens
                      if item[0] <= time <= item[1] and item[2] == wanted]
        if not candidates:
            candidates = [item for item in self._tokens if item[0] <= time <= item[1]]
        if candidates:
            start, end, kind, label, source, index, owner = min(
                candidates, key=lambda item: item[1] - item[0])
            self.token_detail.setText(
                f"{kind}: {label}  ·  {start:.3f}–{end:.3f} s  ·  "
                f"duration {end - start:.3f} s  ·  {source}")
            self._selected_token = (start, end)
            self._selected_token_key = (kind.lower(), index)
            if self._editing_alignment and owner == str(
                    (self.trials.currentData() or {}).get("trial_id", "")):
                if self._edit_handle is not None:
                    self.plot.removeItem(self._edit_handle)
                handle = pg.LinearRegionItem([start, end],
                    brush=pg.mkBrush("#F8F3A055"), movable=True)
                handle.setZValue(10)
                self.plot.addItem(handle)
                handle.sigRegionChangeFinished.connect(
                    lambda item=handle, tier=kind.lower(), token_index=index,
                           token_trial=owner: self._save_token_region(
                               tier, token_index, token_trial, item))
                self._edit_handle = handle

    def _step_recording(self, direction: int) -> None:
        index = self.recordings.currentIndex() + direction
        if 0 <= index < self.recordings.count():
            self.recordings.setCurrentIndex(index)

    def _record_stimulus_changed(self) -> None:
        identity = self.recordings.currentData()
        prompt_id = self.record_stimulus.currentData()
        if identity and prompt_id:
            self._save_record_prompt(identity, prompt_id)

    def _step_trial(self, direction: int) -> None:
        index = self.trials.currentIndex() + direction
        if 0 <= index < self.trials.count():
            self.trials.setCurrentIndex(index)

    def _candidate_changed(self) -> None:
        bounds = self.speech_candidates.currentData()
        if bounds:
            self.trial_start.setValue(bounds[0])
            self.trial_end.setValue(bounds[1])

    def _trial_entries(self) -> list[dict]:
        return load_project_choices(self.root_provider()).get("trials", {}).get(
            self.recordings.currentData(), [])

    def _refresh_trials(self) -> None:
        record_id = self.recordings.currentData()
        run_id = self.runs.currentData()
        entries = []
        if run_id:
            base = self.root_provider() / "acoustic" / "004_alignment" / "runs" / run_id
            planned = base / "configs" / "alignment_trials.json"
            if planned.is_file():
                entries = [item for item in json.loads(planned.read_text(encoding="utf-8"))["trials"]
                           if item["recording_id"] == record_id]
            else:
                path = base / "tables" / "alignment_trial_diagnostics.csv"
                if path.is_file():
                    table = pd.read_csv(path, keep_default_na=False)
                    entries = table.loc[table.recording_id.astype(str).eq(record_id)].to_dict("records")
                else:
                    decisions = (self.root_provider() / "acoustic" / "003_segmentation_review" /
                                 "final" / "final_segmentation_decisions.csv")
                    if decisions.is_file():
                        table = pd.read_csv(decisions, keep_default_na=False)
                        selected = table.loc[table.recording_id.astype(str).eq(record_id)]
                        if not selected.empty:
                            row = selected.iloc[0]
                            entries = [{"trial_id": "legacy_whole_recording",
                                        "start_sec": float(row.analysis_start_sec),
                                        "end_sec": float(row.analysis_end_sec)}]
        if not run_id:
            entries = self._trial_entries()
        self.trials.blockSignals(True)
        self.trials.clear()
        for number, entry in enumerate(entries, 1):
            self.trials.addItem(
                f"Trial {number}: {float(entry['start_sec']):.2f}–{float(entry['end_sec']):.2f} s",
                entry)
        self.trials.blockSignals(False)
        self._trial_changed()

    def _trial_changed(self) -> None:
        entry = self.trials.currentData()
        if not entry:
            self.normal_summary.setText("Confirm trial bounds from reviewed speech, then run Alignment")
            return
        start, end = float(entry["start_sec"]), float(entry["end_sec"])
        run_id = self.runs.currentData()
        if run_id:
            self.plot.setXRange(start, end, padding=.05)
        status = "UNREVIEWED"
        exists = False
        if run_id:
            table = self._reviews(run_id)
            matched = table.loc[table.recording_id.astype(str).eq(self.recordings.currentData())
                                & table.trial_id.astype(str).eq(str(entry["trial_id"]))]
            if not matched.empty:
                status = str(matched.iloc[0].review_status)
                exists = True
            else:
                status = ("LEGACY — trial review unavailable" if entry["trial_id"] ==
                          "legacy_whole_recording" else "ALIGNMENT_FAILED")
        self.normal_summary.setText(
            f"{self.task_label.text()}  ·  {self.prompt_display.text()}  ·  "
            f"Trial {self.trials.currentIndex() + 1}/{self.trials.count()}  ·  {status}")
        if run_id and exists and str(matched.iloc[0].get("flags_triggered", "")):
            try:
                flags = json.loads(str(matched.iloc[0].flags_triggered))
            except ValueError:
                flags = []
            if flags:
                self.normal_summary.setText(self.normal_summary.text() +
                    "  ·  Review: " + ", ".join(str(flag).replace("_", " ").lower()
                                              for flag in flags))
        if run_id and exists and status == "ACCEPTED":
            corrections = load_corrections(self.root_provider(), run_id)
            manual = not corrections.loc[
                corrections.recording_id.astype(str).eq(self.recordings.currentData())
                & corrections.trial_id.astype(str).eq(str(entry["trial_id"]))].empty
            self.normal_summary.setText(self.normal_summary.text() +
                (" — MANUALLY CORRECTED" if manual else " — MFA AS GENERATED"))
        self.accept_button.setEnabled(bool(run_id and exists))
        self.needs_review_button.setEnabled(bool(run_id and exists))
        self._show_step()

    def _add_trial(self) -> None:
        record_id = self.recordings.currentData()
        if not record_id:
            return
        choices = load_project_choices(self.root_provider())
        entries = choices.setdefault("trials", {}).setdefault(record_id, [])
        suffixes = [str(item.get("trial_id", "")).split("_")[-1] for item in entries]
        number = 1 + max((int(value) for value in suffixes if value.isdigit()), default=0)
        entries.append({"trial_id": f"trial_{number:03d}",
                        "start_sec": self.trial_start.value(),
                        "end_sec": self.trial_end.value(),
                        "prompt_id": next((item["prompt_id"] for item in
                            project_alignment_context(self.root_provider())["records"]
                            if item["recording_id"] == record_id), "")})
        context = project_alignment_context(self.root_provider())
        context["choices"] = choices
        context["records"] = [item for item in context["records"]
                              if item["recording_id"] == record_id]
        from vslp.acoustic.alignment.trials import validate_trial_choices
        try:
            validate_trial_choices(self.root_provider(), context)
        except ValueError as exc:
            if not str(exc).startswith("TRIAL_BOUNDARIES_REQUIRED:"):
                QMessageBox.warning(self, "Trial bounds", str(exc))
                return
        save_project_choices(self.root_provider(), choices)
        self._recording_changed()
        self._request_preflight()

    def _remove_trial(self) -> None:
        record_id = self.recordings.currentData()
        entry = self.trials.currentData()
        if not record_id or not entry or self.runs.currentData():
            return
        choices = load_project_choices(self.root_provider())
        choices.setdefault("trials", {})[record_id] = [item for item in self._trial_entries()
            if item["trial_id"] != entry["trial_id"]]
        save_project_choices(self.root_provider(), choices)
        self._recording_changed()
        self._request_preflight()

    def _set_review(self, status: str) -> None:
        entry = self.trials.currentData()
        run_id = self.runs.currentData()
        if not entry or not run_id:
            return
        try:
            set_trial_review(self.root_provider(), run_id, self.recordings.currentData(),
                             str(entry["trial_id"]), status)
        except ValueError as exc:
            QMessageBox.warning(self, "Alignment review", str(exc))
            return
        table = self._reviews(run_id)
        mask = table.recording_id.astype(str).eq(str(self.recordings.currentData())) & \
            table.trial_id.astype(str).eq(str(entry["trial_id"]))
        table.loc[mask, "review_status"] = status
        if "review_mode" in table:
            table.loc[mask, "review_mode"] = "HUMAN"
        self._trial_changed()
        self._update_freeze_state()
        self._navigate_unreviewed(after_current=True)
        self._show_step()

    def _update_freeze_state(self) -> None:
        run_id = self.runs.currentData()
        if not run_id:
            self.freeze_button.setEnabled(False)
            return
        table = self._reviews(run_id)
        if not table.empty:
            self.freeze_button.setEnabled(bool(table.review_status.isin(
                ["ACCEPTED", "AUTO_ACCEPTED_STRUCTURAL"]).all()))
        self._show_step()

    def _play_pause(self) -> None:
        player = self._ensure_player()
        if player.playbackState() == QMediaPlayer.PlayingState:
            player.pause()
            self.play_button.setText("Play trial")
            return
        entry = self.trials.currentData()
        if not entry and self._view_step == "trials" and self.proposal_regions:
            left, right = self.proposal_regions[self._selected_proposal].getRegion()
            entry = {"start_sec": left, "end_sec": right}
        if not entry:
            return
        player.setPosition(round(float(entry["start_sec"]) * 1000))
        self._play_end_ms = round(float(entry["end_sec"]) * 1000)
        player.play()
        self.play_button.setText("Pause")

    def _select_proposal(self, index: int, additive: bool = False) -> None:
        if not 0 <= index < len(self.proposal_regions):
            return
        if additive:
            if index in self._selected_proposals:
                self._selected_proposals.remove(index)
            else:
                self._selected_proposals.add(index)
        else:
            self._selected_proposals = {index}
        self._selected_proposal = index
        for number, region in enumerate(self.proposal_regions):
            selected = number in self._selected_proposals
            region.setBrush(pg.mkBrush("#E5B96C99" if selected else "#6C8ECA55"))
            for edge in region.lines:
                edge.setPen(pg.mkPen("#FFD388" if selected else "#9DC4FF", width=3))
            button = self.proposal_strip_layout.itemAt(number).widget()
            if button is not None:
                button.setStyleSheet("font-weight: bold; border: 2px solid #E5B96C" if
                                     selected else "")
        self._show_step()

    def _current_proposal_bounds(self) -> list[tuple[float, float]]:
        return [tuple(map(float, region.getRegion())) for region in self.proposal_regions]

    def _validate_proposal_bounds(self, bounds: list[tuple[float, float]]) -> bool:
        try:
            validate_trial_draft(self.root_provider(), str(self.recordings.currentData()), bounds)
        except ValueError as exc:
            self.action_hint.setText(
                "The proposed trial enters an excluded or unreviewed region, or overlaps "
                f"another trial. Adjust it before continuing. ({exc})")
            return False
        return True

    def _rebuild_proposals(self, bounds: list[tuple[float, float]],
                           selected: set[int] | None = None) -> None:
        self._rebuilding_proposals = True
        for item in [*self.proposal_regions, *self.proposal_labels]:
            self.plot.removeItem(item)
        self.proposal_regions = []
        self.proposal_labels = []
        for number, (left, right) in enumerate(sorted(bounds), 1):
            region = pg.LinearRegionItem([left, right],
                                         brush=pg.mkBrush("#6C8ECA55"), movable=True)
            region.setZValue(3)
            region.sigRegionChangeFinished.connect(
                lambda item=region: self._proposal_region_changed(item))
            self.plot.addItem(region)
            self.proposal_regions.append(region)
            label = pg.TextItem(f"Trial {number}", color="#C4D7FC", anchor=(0.5, 0.5))
            label.setPos((left + right) / 2, -1.72)
            label.setZValue(5)
            self.plot.addItem(label)
            self.proposal_labels.append(label)
        self._proposal_last_bounds = self._current_proposal_bounds()
        self._selected_proposals = selected or ({0} if bounds else set())
        self._selected_proposal = min(self._selected_proposals) if self._selected_proposals else 0
        self._rebuilding_proposals = False
        self._render_proposal_strip()
        if self._selected_proposals:
            self._select_proposal(self._selected_proposal)
        else:
            self._show_step()

    def _apply_proposal_bounds(self, bounds: list[tuple[float, float]],
                               selected: set[int] | None = None) -> bool:
        bounds = sorted(bounds)
        if not self._validate_proposal_bounds(bounds):
            return False
        old = self._current_proposal_bounds()
        if bounds == old:
            return True
        self._proposal_history.append(old)
        self._rebuild_proposals(bounds, selected)
        return True

    def _proposal_region_changed(self, region: pg.LinearRegionItem) -> None:
        if self._rebuilding_proposals or region not in self.proposal_regions:
            return
        index = self.proposal_regions.index(region)
        before = self._proposal_last_bounds.copy()
        current = self._current_proposal_bounds()
        if current == before:
            return
        if not self._validate_proposal_bounds(current):
            self._rebuilding_proposals = True
            region.setRegion(before[index])
            self._rebuilding_proposals = False
            return
        self._proposal_history.append(before)
        self._proposal_last_bounds = current
        self._render_proposal_strip()
        self._select_proposal(index)

    def _merge_proposals(self) -> None:
        selected = sorted(self._selected_proposals)
        if len(selected) == 1 and len(self.proposal_regions) > 1:
            neighbor = min(selected[0] + 1, len(self.proposal_regions) - 1)
            selected = sorted((selected[0], neighbor if neighbor != selected[0] else neighbor - 1))
        if len(selected) != 2 or selected[1] != selected[0] + 1:
            self.action_hint.setText("Select one trial to merge with its next neighbor, or select two adjacent trials.")
            return
        bounds = self._current_proposal_bounds()
        first, second = selected
        merged = (bounds[first][0], bounds[second][1])
        self._apply_proposal_bounds(bounds[:first] + [merged] + bounds[second + 1:], {first})

    def _render_proposal_strip(self) -> None:
        while self.proposal_strip_layout.count():
            item = self.proposal_strip_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for index, label in enumerate(self.proposal_labels):
            label.setText(f"Trial {index + 1}")
            left, right = self.proposal_regions[index].getRegion()
            label.setPos((left + right) / 2, -1.72)
            select = QPushButton(f"Trial {index + 1}")
            select.clicked.connect(lambda _checked=False, number=index:
                                   self._select_proposal(number, bool(
                                       QApplication.keyboardModifiers() & Qt.ControlModifier)))
            self.proposal_strip_layout.addWidget(select)

    def _begin_add_trial(self) -> None:
        self._add_mode = True
        self.add_button.setText("DRAG ON WAVEFORM…")
        self.action_hint.setText("Drag across the missing utterance on the waveform to add a trial.")

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        if watched is self.plot.viewport() and getattr(self, "_add_mode", False):
            if event.type() in {QEvent.MouseButtonPress, QEvent.MouseMove,
                                QEvent.MouseButtonRelease}:
                if event.type() != QEvent.MouseMove and event.button() != Qt.LeftButton:
                    return False
                scene = self.plot.mapToScene(event.position().toPoint())
                if not self.plot.plotItem.vb.sceneBoundingRect().contains(scene):
                    return False
                current = self.plot.plotItem.vb.mapSceneToView(scene).x()
                if event.type() == QEvent.MouseButtonPress:
                    self._add_drag_start = current
                    self._add_drag_preview = pg.LinearRegionItem(
                        [current, current], brush=pg.mkBrush("#E5B96C77"), movable=False)
                    self._add_drag_preview.setZValue(7)
                    self.plot.addItem(self._add_drag_preview)
                elif event.type() == QEvent.MouseMove and self._add_drag_start is not None:
                    self._add_drag_preview.setRegion(sorted((self._add_drag_start, current)))
                elif event.type() == QEvent.MouseButtonRelease and self._add_drag_start is not None:
                    start = self._add_drag_start
                    self.plot.removeItem(self._add_drag_preview)
                    self._add_drag_start = None
                    self._add_drag_preview = None
                    self._add_mode = False
                    self.add_button.setText("ADD TRIAL")
                    if abs(current - start) >= .01:
                        bounds = sorted((start, current))
                        proposed = sorted([*self._current_proposal_bounds(), tuple(bounds)])
                        self._apply_proposal_bounds(proposed, {proposed.index(tuple(bounds))})
                    else:
                        self.action_hint.setText("Drag across the full missing trial, then release.")
                return True
        return super().eventFilter(watched, event)

    def _omit_proposal(self) -> None:
        if len(self._selected_proposals) != 1:
            return
        index = next(iter(self._selected_proposals))
        bounds = self._current_proposal_bounds()
        self._apply_proposal_bounds(bounds[:index] + bounds[index + 1:],
                                    {min(index, len(bounds) - 2)} if len(bounds) > 1 else set())

    def _split_proposal(self) -> None:
        if len(self._selected_proposals) != 1 or self._proposal_cursor is None:
            self.action_hint.setText("Select one trial, click its split point on the waveform, then SPLIT.")
            return
        index = next(iter(self._selected_proposals))
        bounds = self._current_proposal_bounds()
        left, right = bounds[index]
        split = self._proposal_cursor
        if not left + .01 < split < right - .01:
            self.action_hint.setText("Click a split point inside the selected trial, away from its edges.")
            return
        self._apply_proposal_bounds(bounds[:index] + [(left, split), (split, right)] +
                                    bounds[index + 1:], {index})
        self._proposal_cursor = None

    def _undo_proposal(self) -> None:
        if self._proposal_history:
            previous = self._proposal_history.pop()
            self._rebuild_proposals(previous)

    def _play_visible(self) -> None:
        left, right = self.plot.viewRange()[0]
        player = self._ensure_player()
        player.setPosition(round(left * 1000))
        self._play_end_ms = round(right * 1000)
        player.play()
        self.play_button.setText("Pause")

    def _play_token(self) -> None:
        if not self._selected_token:
            return
        start, end = self._selected_token
        player = self._ensure_player()
        player.setPosition(round(start * 1000))
        self._play_end_ms = round(end * 1000)
        player.play()
        self.play_button.setText("Pause")

    def _playback_position(self, position_ms: int) -> None:
        self.play_cursor.setPos(position_ms / 1000)
        if self._play_end_ms and position_ms >= self._play_end_ms:
            if self.player is not None:
                self.player.pause()
            self.play_button.setText("Play trial")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._preflight_timer.stop()
        if self.player is not None:
            self.player.stop()
            self.player.setSource(QUrl())
        super().closeEvent(event)

    def _ensure_player(self) -> QMediaPlayer:
        if self.player is None:
            self.player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            self.player.setAudioOutput(self.audio_output)
            self.player.positionChanged.connect(self._playback_position)
        return self.player
