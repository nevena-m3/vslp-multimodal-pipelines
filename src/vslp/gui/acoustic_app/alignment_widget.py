"""Compact viewer and controls for the separate Alignment stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyqtgraph as pg
import soundfile as sf
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem,
)

from vslp.acoustic.alignment import AlignmentConfig, list_alignment_runs
from vslp.acoustic.alignment.mfa_provider import MfaProfile, MfaProvider
from vslp.acoustic.alignment.profiles import default_profile_path
from vslp.acoustic.alignment.task_workflow import (
    load_project_choices, project_alignment_context, resolve_prompt, save_project_choices,
)


class AlignmentWidget(QWidget):
    run_requested = Signal(object)
    freeze_requested = Signal(str)
    self_test_requested = Signal(str, str)
    preflight_requested = Signal(str, str)
    environment_requested = Signal(str)
    run_task_requested = Signal(str, str)

    def __init__(self, root_provider: Callable[[], Path]) -> None:
        super().__init__()
        self.root_provider = root_provider
        layout = QVBoxLayout(self)
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
        form.addRow("Preflight", self.preflight)
        self.recording_count = QLabel("Recordings: —")
        form.addRow("", self.recording_count)
        self.assignments = QTableWidget(0, 5)
        self.assignments.setHorizontalHeaderLabels(
            ["Recording", "Stimulus", "Speaker ID", "Alignment transcript", "Correction reason"])
        self.assignments.horizontalHeader().setStretchLastSection(True)
        self.assignments.itemChanged.connect(self._save_assignments)
        layout.addLayout(form)
        layout.addWidget(self.assignments)
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
        self.advanced_toggle = QPushButton("Advanced / diagnostics")
        self.advanced_toggle.setCheckable(True)
        layout.addWidget(self.advanced_toggle)
        self.advanced = QWidget()
        advanced_form = QFormLayout(self.advanced)
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
        self.advanced_toggle.toggled.connect(self.advanced.setVisible)
        self.advanced_toggle.toggled.connect(lambda _checked: self._source_changed())
        self.advanced.hide()
        layout.addWidget(self.advanced)
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
        inspect = QPushButton("INSPECT ALIGNMENT")
        inspect.clicked.connect(self._mark_inspected)
        buttons.addWidget(inspect)
        freeze = QPushButton("FREEZE ALIGNMENT")
        freeze.clicked.connect(lambda: self.freeze_requested.emit(self.runs.currentData() or ""))
        self.freeze_button = freeze
        freeze.setEnabled(False)
        buttons.addWidget(freeze)
        layout.addLayout(buttons)
        self.summary = QLabel("No alignment run selected")
        layout.addWidget(self.summary)
        self.frozen_status = QLabel("No frozen Alignment")
        layout.addWidget(self.frozen_status)
        self.self_test_summary = QLabel("MFA self-test not run")
        self.self_test_summary.setWordWrap(True)
        layout.addWidget(self.self_test_summary)
        self.view_self_test_diagnostics = QPushButton("View self-test diagnostics")
        self.view_self_test_diagnostics.hide()
        self.view_self_test_diagnostics.clicked.connect(self._open_self_test_diagnostics)
        layout.addWidget(self.view_self_test_diagnostics)
        self._self_test_diagnostics_path = ""
        self.recordings = QComboBox()
        self.recordings.currentIndexChanged.connect(self._recording_changed)
        layout.addWidget(self.recordings)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Original recording time", units="s")
        self.plot.setLabel("left", "Waveform")
        self.plot.showGrid(x=True, alpha=0.15)
        self.plot.scene().sigMouseClicked.connect(self._token_clicked)
        layout.addWidget(self.plot, 1)
        self.token_detail = QLabel("Select a token on the timeline")
        layout.addWidget(self.token_detail)
        self._tokens: list[tuple[float, float, str, str, str]] = []
        self._loading_assignments = False
        self._last_preflight_key = ""
        self._preflight_ready = False
        self._source_changed()

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
        advanced = self.advanced_toggle.isChecked()
        for widget in self.source_fields:
            widget.setVisible(external and advanced)
        for widget in self.mfa_fields:
            widget.setVisible(advanced)
        self.advanced.setVisible(advanced)
        self.run_button.setEnabled(external or self._preflight_ready)

    def _check_environment(self) -> None:
        self.environment.setText("Checking MFA environment...")
        self.environment_requested.emit(self.profile.text().strip())

    def show_environment_result(self, environment: dict) -> None:
        if environment.get("status") == "AVAILABLE":
            self.environment.setText(f"Ready — MFA {environment.get('version', '?')} · "
                                     f"{environment.get('conda_environment') or 'external'}")
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
            self.preflight.setText("READY")
        else:
            issue = result.get("issue", "ACTION_REQUIRED")
            if issue == "OOV":
                words = ", ".join(result.get("oov_words", []))
                self.preflight.setText(f"ACTION REQUIRED — dictionary words missing: {words}")
            else:
                self.preflight.setText(f"ACTION REQUIRED — {issue}")
        self.run_button.setEnabled(self.source.currentData() == "external" or self._preflight_ready)

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
        self.run_requested.emit(AlignmentConfig(
            source=self.source.currentData(), prompt_manifest_path=self.prompt.text().strip(),
            words_csv=self.words.text().strip(), phones_csv=self.phones.text().strip(),
            acoustic_model=self.model.text().strip(),
            acoustic_model_version=self.model_version.text().strip(),
            dictionary=self.dictionary.text().strip(),
            dictionary_version=self.dictionary_version.text().strip(),
            mfa_profile_path=self.profile.text().strip(),
            speaker_manifest_path=self.speakers.text().strip(),
            transcript_overrides_path=self.overrides.text().strip()))

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

    def _save_assignments(self, _item: QTableWidgetItem) -> None:
        if self._loading_assignments:
            return
        choices = load_project_choices(self.root_provider())
        for row in range(self.assignments.rowCount()):
            identity = self.assignments.item(row, 0).data(Qt.UserRole)
            entry = choices.setdefault("recordings", {}).setdefault(identity, {})
            for column, field in ((2, "speaker_id"), (3, "alignment_transcript"),
                                  (4, "transcript_reason")):
                entry[field] = self.assignments.item(row, column).text().strip()
        save_project_choices(self.root_provider(), choices)
        self._preflight_ready = False
        self.run_button.setEnabled(False)
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
        self.preflight.setText("Checking prerequisites...")
        QTimer.singleShot(0, lambda: self.preflight_requested.emit(
            str(self.root_provider()), self.profile.text().strip()))

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
        if context.get("issue"):
            self.task_label.setText("Select a project in Setup")
            self.run_button.setEnabled(False)
        else:
            self.task_label.setText(str(context["task_name"]))
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
                ("Choose each recording's stimulus below" if len(prompts) > 1 else issue))
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
            self.speaker_status.setText(
                f"Available ({len(context['records'])})" if all(
                    item["speaker_id"] for item in context["records"]) else "Speaker ID required")
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
        index = self.runs.findData(selected or stored)
        self.runs.setCurrentIndex(index if index >= 0 else self.runs.count() - 1)
        self.runs.blockSignals(False)
        self._run_changed()

    def _run_changed(self) -> None:
        run_id = self.runs.currentData()
        self.recordings.clear()
        if not run_id:
            self.freeze_button.setEnabled(False)
            return
        choices = load_project_choices(self.root_provider())
        if choices.get("selected_run_id") != run_id:
            choices["selected_run_id"] = run_id
            save_project_choices(self.root_provider(), choices)
        self.freeze_button.setEnabled(
            choices.get("inspected_run_id") == run_id)
        root = self.root_provider()
        manifest_path = root / "acoustic" / "004_alignment" / "runs" / run_id / "logs" / "stage_manifest.json"
        if not manifest_path.is_file():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        preflight = manifest_path.parent / "mfa_preflight.json"
        if preflight.is_file():
            state = json.loads(preflight.read_text(encoding="utf-8"))
            self.preflight.setText("Passed" if state.get("status") == "AVAILABLE"
                                   else str(state.get("status", "Issue")))
        else:
            self.preflight.setText("Validated import" if manifest.get("provider") == "external"
                                   else "Preflight not run")
        self.technical.setText(
            f"Provider: {manifest.get('provider', '—')} {manifest.get('provider_version', '')}\n"
            f"Mode: {manifest.get('provider_mode', '—')}\n"
            f"Model: {manifest.get('acoustic_model', '—')}\n"
            f"Dictionary: {manifest.get('dictionary', '—')}\n"
            f"Phone set: {manifest.get('phone_set', '—')}\n"
            f"Working sample rate: {manifest.get('working_sample_rate_hz', '—')} Hz\n"
            f"Commands: {manifest.get('provider_commands', [])}")
        diagnostics = pd.read_csv(manifest["diagnostics_path"], keep_default_na=False)
        self.summary.setText(
            f"Aligned {manifest['number_aligned']} / {manifest['number_attempted']}  ·  "
            f"Partial {manifest.get('number_partial', 0)}  ·  Failed {manifest['number_failed']}  ·  "
            f"Needs review {manifest['number_below_coverage_threshold']}  ·  "
            f"Words {int(diagnostics.n_words.sum())}  ·  Phones {int(diagnostics.n_phones.sum())}")
        for row in diagnostics.itertuples():
            self.recordings.addItem(f"{row.file_name}  ·  {row.status}  ·  coverage {row.word_coverage:.0%}",
                                    str(row.recording_id))
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

    def _recording_changed(self) -> None:
        self.plot.clear()
        self._tokens = []
        record_id = self.recordings.currentData()
        run_id = self.runs.currentData()
        if not record_id or not run_id:
            return
        root = self.root_provider()
        decisions = pd.read_csv(root / "acoustic" / "003_segmentation_review" / "final" /
                                "final_segmentation_decisions.csv", keep_default_na=False)
        matching = decisions.loc[decisions.recording_id.astype(str).eq(record_id)]
        if matching.empty:
            return
        path = Path(str(matching.iloc[0].analysis_wav_path))
        if not path.is_file():
            return
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        signal = audio.mean(axis=1)
        step = max(1, len(signal) // 15000)
        times = np.arange(0, len(signal), step) / rate
        self.plot.plot(times, signal[::step], pen=pg.mkPen("#8AB9D7", width=1))
        duration = len(signal) / rate
        self.plot.setLimits(xMin=0, xMax=duration)
        self.plot.setXRange(0, min(duration, 15), padding=0)
        start = float(matching.iloc[0].analysis_start_sec)
        end = float(matching.iloc[0].analysis_end_sec)
        for left, right in ((0, start), (end, duration)):
            if right > left:
                self.plot.addItem(pg.LinearRegionItem([left, right],
                    brush=pg.mkBrush("#59616C55"), movable=False))
        interval_path = root / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_intervals.csv"
        if interval_path.is_file():
            intervals = pd.read_csv(interval_path, keep_default_na=False)
            excluded = intervals.loc[intervals.recording_id.astype(str).eq(record_id)
                                     & intervals.segment_role.eq("manual_exclusion")]
            for row in excluded.itertuples():
                self.plot.addItem(pg.LinearRegionItem([float(row.start_sec), float(row.end_sec)],
                    brush=pg.mkBrush("#C9696955"), movable=False))
        base = root / "acoustic" / "004_alignment" / "runs" / run_id / "tables"
        for filename, color in (("alignment_words.csv", "#F1C66D"),
                                ("alignment_phones.csv", "#94CCAD")):
            path = base / filename
            if not path.is_file():
                continue
            table = pd.read_csv(path, keep_default_na=False)
            for row in table.loc[table.recording_id.astype(str).eq(record_id)].itertuples():
                label = str(getattr(row, "word", getattr(row, "phone", "")))
                kind = "Word" if "word" in filename else "Phone"
                self._tokens.append((float(row.start_sec), float(row.end_sec), kind, label,
                                     str(getattr(row, "alignment_source", ""))))
                region = pg.LinearRegionItem([float(row.start_sec), float(row.end_sec)],
                                             brush=pg.mkBrush(color + "35"), movable=False)
                self.plot.addItem(region)
                tag = pg.TextItem(label, color=color, anchor=(0.5, 0.5))
                tag.setPos((float(row.start_sec) + float(row.end_sec)) / 2,
                           0.7 if kind == "Word" else -0.7)
                self.plot.addItem(tag)
        self.token_detail.setText("Yellow: words  ·  Green: phones  ·  Times on original recording")

    def _token_clicked(self, event) -> None:
        if not self.plot.sceneBoundingRect().contains(event.scenePos()):
            return
        time = self.plot.plotItem.vb.mapSceneToView(event.scenePos()).x()
        candidates = [item for item in self._tokens if item[0] <= time <= item[1]]
        if candidates:
            start, end, kind, label, source = min(candidates, key=lambda item: item[1] - item[0])
            self.token_detail.setText(
                f"{kind}: {label}  ·  {start:.3f}–{end:.3f} s  ·  "
                f"duration {end - start:.3f} s  ·  {source}  ·  run {self.runs.currentData()}")
