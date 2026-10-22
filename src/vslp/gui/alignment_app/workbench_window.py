"""Task-first, batch-oriented VSLP Alignment workbench."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
from time import monotonic

from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDockWidget, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from vslp.acoustic.alignment.cancellable_runner import CancellableMfaRunner
from vslp.acoustic.alignment.mfa_provider import MfaProfile, MfaProvider
from vslp.acoustic.alignment.profiles import default_profile_path
from vslp.acoustic.alignment.task_workflow import task_registry
from vslp.acoustic.alignment.workbench import (
    align_job, canonical_task, create_job, discover_audio, inspect_audio_files,
    load_job, local_writable_workspace, preflight_job, prepare_corpus,
    realign_recording,
)
from vslp.acoustic.alignment.workbench_package import export_standalone_package
from vslp.gui.common.utils import open_path
from vslp.gui.common.worker import Worker
from vslp.gui.alignment_app.workbench_review import WorkbenchReview, load_recording_review


def _inspect_sources(paths: list[Path]) -> list[dict]:
    return inspect_audio_files(paths)


def _inspect_folder(path: Path) -> list[dict]:
    return inspect_audio_files(discover_audio(path))


def _create_job(workspace: Path, name: str, audio_rows: list[dict],
                task_id: str, repetitions: int) -> Path:
    return create_job(workspace, name=name, audio_rows=audio_rows,
                      task_id=task_id, repetitions=repetitions)


def _prepare_and_preflight(root: Path, runner: CancellableMfaRunner,
                           progress_callback=None) -> dict:
    preparation = prepare_corpus(root, progress_callback=progress_callback,
                                 cancelled=runner.cancelled)
    if preparation["status"] != "CORPUS_READY":
        return {"status": preparation["status"], "issues": preparation["issues"]}
    if progress_callback:
        progress_callback(0, 0, "Running MFA corpus validation and OOV checks...")
    return preflight_job(root, provider=MfaProvider(runner=runner))


def _align(root: Path, runner: CancellableMfaRunner, progress_callback=None) -> dict:
    if progress_callback:
        progress_callback(0, 0, "MFA corpus alignment running; progress is not estimated")
    result = align_job(root, provider=MfaProvider(runner=runner))
    if progress_callback:
        progress_callback(result["aligned"], result["total"],
                          "Parsing TextGrids and running structural QC complete")
    return result


def _environment() -> dict:
    return MfaProvider().inspect_environment(MfaProfile.load(default_profile_path()))


class WorkbenchWindow(QMainWindow):
    """One task definition per persistent job; heavy operations use one worker."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VSLP Alignment")
        self.resize(1450, 900)
        self.setMinimumSize(1050, 700)
        self.audio_rows: list[dict] = []
        self.job_root: Path | None = None
        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self._runner: CancellableMfaRunner | None = None
        self._exported: Path | None = None
        self._close_when_idle = False
        self._pending_review_record: dict | None = None
        self._operation_started_at = 0.0
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._refresh_elapsed)
        self._workspace_default = Path.home() / "Documents" / "Codex" / "VSLP Alignment"
        self._build()
        self._task_changed()
        QTimer.singleShot(0, lambda: self._start("environment", _environment, {}))

    def _build(self) -> None:
        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)
        title = QLabel("VSLP Alignment")
        title.setObjectName("AppTitleLabel")
        layout.addWidget(title)
        layout.addWidget(QLabel("1 Add audio   →   2 Choose task   →   3 Validate / Pre-flight   →   "
                                "4 Run alignment   →   5 Review flagged   →   6 Export"))
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.tabs.addTab(self._setup_page(), "1  Job setup")
        self.tabs.addTab(self._run_page(), "2  Run alignment")
        self.tabs.addTab(self._review_page(), "3  Review")
        self.tabs.addTab(self._export_page(), "4  Export")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        layout.addWidget(self.progress)
        self.progress_label = QLabel("Ready")
        layout.addWidget(self.progress_label)
        self.elapsed_label = QLabel("")
        layout.addWidget(self.elapsed_label)
        self._build_log()
        self.statusBar().showMessage("Ready")

    def _setup_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        heading = QLabel("ADD AUDIO")
        heading.setObjectName("SectionTitleLabel")
        layout.addWidget(heading)
        buttons = QHBoxLayout()
        for label, callback in (("Add files", self._add_files),
                                ("Add folder (recursive)", self._add_folder),
                                ("Open existing job", self._open_job_dialog)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.ingest_summary = QLabel("No recordings selected · WAV files only")
        layout.addWidget(self.ingest_summary)
        self.ingest_table = QTableWidget(0, 6)
        self.ingest_table.setHorizontalHeaderLabels(
            ["Recording", "Duration", "Sample rate", "Channels", "Size", "Status"])
        self.ingest_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ingest_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.ingest_table, 1)
        details = QHBoxLayout()
        task_box = QGroupBox("TASK")
        form = QFormLayout(task_box)
        self.task_type = QComboBox()
        self.task_type.addItem("Bamboo Passage", "bamboo_passage")
        self.task_type.addItem("Sentence", "sentence")
        self.task_type.currentIndexChanged.connect(self._task_changed)
        form.addRow("Task type", self.task_type)
        self.variant = QComboBox()
        for task in task_registry()["tasks"]:
            if task.get("alignment_applicable") and task["task_id"] != "bamboo_passage":
                self.variant.addItem(task["display_name"], task["task_id"])
        self.variant.currentIndexChanged.connect(self._task_changed)
        form.addRow("Sentence", self.variant)
        self.repetitions = QSpinBox()
        self.repetitions.setRange(1, 100)
        form.addRow("Expected repetitions", self.repetitions)
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setMaximumHeight(100)
        form.addRow("Canonical transcript", self.transcript)
        details.addWidget(task_box, 2)
        workspace_box = QGroupBox("JOB WORKSPACE")
        workspace_form = QFormLayout(workspace_box)
        self.job_name = QLineEdit("New Alignment Job")
        workspace_form.addRow("Job name", self.job_name)
        workspace_row = QHBoxLayout()
        self.workspace = QLineEdit(str(self._workspace_default))
        workspace_row.addWidget(self.workspace)
        choose = QPushButton("Change workspace")
        choose.clicked.connect(self._choose_workspace)
        workspace_row.addWidget(choose)
        workspace_form.addRow("Workspace", workspace_row)
        workspace_form.addRow(QLabel("Raw input files remain untouched. Each job has its own corpus, "
                                    "MFA output, review and export folders."))
        details.addWidget(workspace_box, 1)
        layout.addLayout(details)
        self.create_button = QPushButton("Create Alignment Job")
        self.create_button.setObjectName("RunButton")
        self.create_button.clicked.connect(self._create_job)
        layout.addWidget(self.create_button)
        return page

    def _run_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.job_heading = QLabel("Create or open an Alignment Job")
        self.job_heading.setObjectName("SectionTitleLabel")
        layout.addWidget(self.job_heading)
        self.engine_status = QLabel("Checking MFA environment...")
        layout.addWidget(self.engine_status)
        self.run_counts = QLabel("Ready 0    Aligned 0    Needs review 0    Failed 0")
        layout.addWidget(self.run_counts)
        actions = QHBoxLayout()
        self.preflight_button = QPushButton("Validate / Pre-flight")
        self.preflight_button.setToolTip("Check audio/transcripts, repetitions, workspace, MFA, "
                                         "model/dictionary, OOVs and corpus readiness.")
        self.preflight_button.clicked.connect(self._preflight)
        self.run_button = QPushButton("Run Alignment")
        self.run_button.setObjectName("RunButton")
        self.run_button.clicked.connect(self._run_alignment)
        self.cancel_button = QPushButton("Cancel Alignment")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_alignment)
        for button in (self.preflight_button, self.run_button, self.cancel_button):
            actions.addWidget(button)
        actions.addStretch()
        layout.addLayout(actions)
        self.preflight_details = QLabel("Corpus validation checks audio/transcript pairs, "
                                        "model, dictionary and OOVs before MFA alignment.")
        self.preflight_details.setWordWrap(True)
        layout.addWidget(self.preflight_details)
        self.batch_table = QTableWidget(0, 5)
        self.batch_table.setHorizontalHeaderLabels(
            ["Recording", "Duration", "Task", "Status", "Issue"])
        self.batch_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.batch_table, 1)
        return page

    def _review_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        navigation = QHBoxLayout()
        for title, step in (("← Previous flagged", -1), ("Next flagged →", 1)):
            button = QPushButton(title)
            button.clicked.connect(lambda _, direction=step: self._next_flagged(direction))
            navigation.addWidget(button)
        navigation.addWidget(QLabel("Queue"))
        self.review_filter = QComboBox()
        self.review_filter.addItems(["Needs Review", "Failed", "Reviewed", "Auto Pass", "All"])
        self.review_filter.currentIndexChanged.connect(self._filter_review)
        navigation.addWidget(self.review_filter)
        navigation.addStretch()
        layout.addLayout(navigation)
        splitter = QSplitter(Qt.Horizontal)
        self.review_table = QTableWidget(0, 4)
        self.review_table.setHorizontalHeaderLabels(
            ["Recording", "Status", "Aligned words", "QC issue"])
        self.review_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.review_table.horizontalHeader().setStretchLastSection(True)
        self.review_table.setMinimumWidth(300)
        self.review_table.setMaximumWidth(380)
        self.review_table.itemSelectionChanged.connect(self._review_selected)
        splitter.addWidget(self.review_table)
        self.reviewer = WorkbenchReview()
        self.reviewer.realign_requested.connect(self._realign_recording)
        self.reviewer.changed.connect(self._review_updated)
        review_scroll = QScrollArea()
        review_scroll.setWidgetResizable(True)
        review_scroll.setWidget(self.reviewer)
        splitter.addWidget(review_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter, 1)
        return page

    def _export_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        heading = QLabel("EXPORT VSLP ALIGNMENT PACKAGE V1")
        heading.setObjectName("SectionTitleLabel")
        layout.addWidget(heading)
        self.export_summary = QLabel("Only structurally accepted records can be exported.")
        layout.addWidget(self.export_summary)
        destination = QHBoxLayout()
        self.export_parent = QLineEdit()
        self.export_parent.setPlaceholderText("Choose a writable local output folder")
        destination.addWidget(self.export_parent)
        chooser = QPushButton("Choose folder")
        chooser.clicked.connect(self._choose_export)
        destination.addWidget(chooser)
        layout.addLayout(destination)
        self.export_button = QPushButton("Export VSLP Alignment Package")
        self.export_button.setObjectName("RunButton")
        self.export_button.clicked.connect(self._export)
        layout.addWidget(self.export_button)
        self.export_result = QLabel("")
        self.export_result.setWordWrap(True)
        layout.addWidget(self.export_result)
        self.open_export = QPushButton("Open package folder")
        self.open_export.setEnabled(False)
        self.open_export.clicked.connect(lambda: open_path(self._exported)
                                         if self._exported else None)
        layout.addWidget(self.open_export)
        layout.addStretch()
        return page

    def _build_log(self) -> None:
        self.log_dock = QDockWidget("Technical Log", self)
        self.log_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        controls = QHBoxLayout()
        self.log_toggle = QPushButton("+ Expand")
        self.log_toggle.clicked.connect(self._toggle_log)
        controls.addWidget(self.log_toggle)
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self.log.clear())
        controls.addWidget(clear)
        controls.addStretch()
        layout.addLayout(controls)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setVisible(False)
        layout.addWidget(self.log)
        self.log_dock.setWidget(panel)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.setMinimumHeight(72)
        self.log_dock.setMaximumHeight(72)
        self._log("INFO", "Application ready")

    def _toggle_log(self) -> None:
        visible = not self.log.isVisible()
        self.log.setVisible(visible)
        self.log_toggle.setText("− Collapse" if visible else "+ Expand")
        self.log_dock.setMinimumHeight(160 if visible else 72)
        self.log_dock.setMaximumHeight(16777215 if visible else 72)

    def _log(self, level: str, message: str) -> None:
        self.log.appendPlainText(f"{datetime.now():%H:%M:%S}  {level:<5}  {message}")
        if level in {"WARN", "ERROR"} and not self.log.isVisible():
            self._toggle_log()

    def _dialog_root(self) -> str:
        root = self.workspace.text().strip()
        path = Path(root) if root else self._workspace_default
        if str(path).startswith(("\\\\", "//")):
            return str(Path.cwd())
        return str(path if path.is_dir() else Path.cwd())

    def _choose_folder(self, title: str) -> str:
        return QFileDialog.getExistingDirectory(self, title, self._dialog_root())

    def _add_files(self) -> None:
        selected, _ = QFileDialog.getOpenFileNames(
            self, "Add WAV recordings", self._dialog_root(), "WAV audio (*.wav)")
        if selected:
            self._start("ingest", _inspect_sources, {"paths": [Path(p) for p in selected]})

    def _add_folder(self) -> None:
        selected = self._choose_folder("Add audio folder (recursive WAV discovery)")
        if selected:
            self._start("ingest", _inspect_folder, {"path": Path(selected)})

    def _choose_workspace(self) -> None:
        selected = self._choose_folder("Choose local Alignment workspace")
        if selected:
            self.workspace.setText(selected)

    def _choose_export(self) -> None:
        selected = self._choose_folder("Choose Alignment Package output folder")
        if selected:
            self.export_parent.setText(selected)

    def _task_changed(self) -> None:
        if not hasattr(self, "variant"):
            return
        bamboo = self.task_type.currentData() == "bamboo_passage"
        self.variant.setVisible(not bamboo)
        label = self.variant.parentWidget().layout().labelForField(self.variant)
        if label:
            label.setVisible(not bamboo)
        task_id = "bamboo_passage" if bamboo else self.variant.currentData()
        if task_id:
            task = next(t for t in task_registry()["tasks"] if t["task_id"] == task_id)
            prompt = task["prompts"][0]
            self.transcript.setPlainText(prompt["exact_expected_text"])
            self.repetitions.setValue(int(prompt.get("expected_repetitions") or 1))
            self.repetitions.setEnabled(not bamboo)

    def _create_job(self) -> None:
        if not any(row["status"] in {"READY", "CHANNEL WARNING"} for row in self.audio_rows):
            QMessageBox.warning(self, "Audio required", "Add readable WAV recordings first.")
            return
        task_id = ("bamboo_passage" if self.task_type.currentData() == "bamboo_passage"
                   else self.variant.currentData())
        try:
            canonical_task(task_id, self.repetitions.value())
            workspace = local_writable_workspace(self.workspace.text().strip())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Job setup", str(exc))
            return
        self._start("create_job", _create_job, {"workspace": workspace,
            "name": self.job_name.text(), "audio_rows": self.audio_rows,
            "task_id": task_id, "repetitions": self.repetitions.value()})

    def _open_job_dialog(self) -> None:
        folder = self._choose_folder("Open Alignment Job folder")
        if folder:
            self._start("open_job", load_job, {"root": Path(folder)})
            self._pending_job_root = Path(folder)

    def _preflight(self) -> None:
        if self.job_root is not None:
            self._runner = CancellableMfaRunner()
            self.cancel_button.setEnabled(True)
            self._start("preflight", _prepare_and_preflight,
                        {"root": self.job_root, "runner": self._runner})

    def _run_alignment(self) -> None:
        if self.job_root is None:
            return
        if load_job(self.job_root)["state"] != "READY_TO_ALIGN":
            QMessageBox.warning(self, "Validate first", "Run corpus preflight before alignment.")
            return
        self._runner = CancellableMfaRunner()
        self.cancel_button.setEnabled(True)
        self._start("alignment", _align, {"root": self.job_root, "runner": self._runner})

    def _realign_recording(self, record_key: str, spoken: str, reviewer: str) -> None:
        if self.job_root is None:
            return
        self._log("INFO", f"Re-aligning {record_key} using reviewed spoken transcript")
        self._start("realign", realign_recording,
                    {"root": self.job_root, "record_key": record_key,
                     "spoken_transcript": spoken, "reviewer": reviewer})

    def _cancel_alignment(self) -> None:
        if self._runner is not None:
            self._runner.cancel()
            self._log("WARN", "Cancellation requested; stopping MFA process tree")

    def _export(self) -> None:
        if self.job_root is None:
            return
        try:
            parent = local_writable_workspace(self.export_parent.text().strip())
        except (OSError, ValueError) as exc:
            self._log("ERROR", f"Export destination unavailable: {exc}")
            QMessageBox.warning(self, "Export destination", str(exc))
            return
        self._start("export", export_standalone_package,
                    {"job_root": self.job_root,
                     "destination": parent / f"VSLP_Alignment_Package_V1_{datetime.now():%Y%m%d_%H%M%S}"})

    def _start(self, name: str, function, kwargs: dict) -> None:
        if self._thread is not None:
            QMessageBox.information(self, "Operation running", "Wait or cancel the current operation.")
            return
        self.progress.setRange(0, 0)
        self.progress_label.setText(name.replace("_", " ").title() + "...")
        self._operation_started_at = monotonic()
        self._elapsed_timer.start(1000)
        self._refresh_elapsed()
        self.statusBar().showMessage(self.progress_label.text())
        self._log("INFO", self.progress_label.text())
        self._thread = QThread(self)
        self._worker = Worker(name, function, kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress)
        self._worker.finished.connect(self._finished)
        self._worker.failed.connect(self._failed)
        self._worker.finished.connect(self._cleanup)
        self._worker.failed.connect(self._cleanup)
        self._thread.start()

    def _progress(self, done: int, total: int, message: str) -> None:
        if total:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)

    def _refresh_elapsed(self) -> None:
        if self._operation_started_at:
            elapsed = int(monotonic() - self._operation_started_at)
            self.elapsed_label.setText(f"Elapsed {elapsed // 60:02d}:{elapsed % 60:02d}")

    def _finished(self, name: str, result: object) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress_label.setText(f"{name.replace('_', ' ').title()} complete")
        self.statusBar().showMessage(self.progress_label.text())
        self._log("INFO", self.progress_label.text())
        if name == "environment":
            report = result
            status = report.get("status", "UNKNOWN")
            self.engine_status.setText(
                f"MFA READY · {report.get('version', '')} · VSLP English Standard"
                if status == "AVAILABLE" else f"MFA SETUP REQUIRED · {status}")
            if status != "AVAILABLE":
                self._log("WARN", f"MFA environment: {status}")
        elif name == "ingest":
            self.audio_rows = list(result)
            self._show_ingest()
        elif name == "create_job":
            self.job_root = Path(result)
            self._show_job()
            self.tabs.setCurrentIndex(1)
        elif name == "open_job":
            self.job_root = self._pending_job_root
            self._show_job()
            self.tabs.setCurrentIndex(1)
        elif name == "review_load":
            if self.job_root and self._pending_review_record:
                self.reviewer.set_record(self.job_root, self._pending_review_record, result)
        elif name in {"preflight", "alignment"}:
            self._show_job()
            if name == "preflight":
                self.preflight_details.setText(
                    f"Preflight: {result.get('status')} · OOV: "
                    f"{', '.join(result.get('oov_words', [])) or 'none'}")
                if result.get("status") != "AVAILABLE":
                    self._log("WARN", self.preflight_details.text())
            else:
                self.tabs.setCurrentIndex(2)
        elif name == "realign":
            self._show_job()
            if self.job_root and self._pending_review_record:
                record = next((row for row in load_job(self.job_root)["recordings"]
                               if row["record_key"] == self._pending_review_record["record_key"]), None)
                if record:
                    self._pending_review_record = record
                    QTimer.singleShot(0, lambda: self._start(
                        "review_load", load_recording_review,
                        {"job_root": self.job_root, "record": record}))
        elif name == "export":
            self._exported = Path(result)
            self.export_result.setText(f"Package exported successfully\n{self._exported}")
            self.open_export.setEnabled(True)
            self._log("INFO", f"Package exported: {self._exported}")

    def _failed(self, name: str, error: str) -> None:
        summary = error.splitlines()[0]
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label.setText(f"{name} failed: {summary}")
        self.statusBar().showMessage(self.progress_label.text())
        self._log("ERROR", f"{name}: {summary}")
        self.log.appendPlainText(error)
        if self.job_root and name in {"alignment", "preflight"}:
            self._show_job()
        QMessageBox.warning(self, name.replace("_", " ").title(), summary)

    def _cleanup(self, *_args) -> None:
        self._elapsed_timer.stop()
        self._refresh_elapsed()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self._runner = None
        self.cancel_button.setEnabled(False)
        if self._close_when_idle:
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event) -> None:
        if self._thread is not None:
            self._close_when_idle = True
            if self._runner is not None:
                self._runner.cancel()
            self.statusBar().showMessage("Stopping operation before closing...")
            event.ignore()
            return
        super().closeEvent(event)

    def _show_ingest(self) -> None:
        rows = self.audio_rows
        ready = sum(row["status"] in {"READY", "CHANNEL WARNING"} for row in rows)
        self.ingest_summary.setText(f"{len(rows)} found · {ready} ready · "
                                    f"{len(rows) - ready} unavailable")
        self.ingest_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = (row["recording"], f"{row['duration_sec']:.2f} s"
                      if row["duration_sec"] is not None else "—",
                      str(row["sample_rate_hz"] or "—"), str(row["channels"] or "—"),
                      str(row["size_bytes"] or "—"), row["status"])
            for column, value in enumerate(values):
                self.ingest_table.setItem(index, column, QTableWidgetItem(value))

    def _show_job(self) -> None:
        if self.job_root is None:
            return
        job = load_job(self.job_root)
        rows = job["recordings"]
        self.job_heading.setText(f"{job['name']} · {job['task_specification']['task_name']} · "
                                 f"{len(rows)} recordings · "
                                 f"{job['task_specification']['expected_repetitions']} repetition(s)")
        counts = {state: sum(row["state"] == state for row in rows)
                  for state in ("QUEUED", "CORPUS_READY", "AUTO_PASS", "REVIEWED",
                                "NEEDS_REVIEW", "FAILED")}
        self.run_counts.setText("    ".join(f"{state.title()} {count}" for state, count in counts.items()))
        self.batch_table.setRowCount(len(rows))
        self.review_table.blockSignals(True)
        self.review_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            qc_detail = ", ".join(row.get("qc_flags", []))
            extra = row.get("extra_aligned_tokens") or []
            if extra:
                qc_detail += f" · Extra aligned token(s): {', '.join(extra)}"
            if row.get("first_aligned_word_start_sec") is not None:
                qc_detail += (f" · Initial span before first word: "
                              f"{row['first_aligned_word_start_sec']:.2f} s")
            qc_detail += (f" · Protocol words: {row.get('protocol_scientific_word_count', '—')}"
                          f" · MFA tokens: {row.get('aligned_word_count', '—')}")
            for column, value in enumerate((row["recording"],
                    f"{row['duration_sec']:.2f} s", job["task_specification"]["task_name"],
                    row["state"], qc_detail or row.get("issue", ""))):
                self.batch_table.setItem(index, column, QTableWidgetItem(str(value)))
            for column, value in enumerate((row["recording"], row["state"],
                    str(row.get("aligned_word_count", "—")),
                    qc_detail or "—")):
                self.review_table.setItem(index, column, QTableWidgetItem(str(value)))
        self.export_parent.setText(str(self.job_root / "exports"))
        self.export_summary.setText(f"Total {len(rows)} · Structurally accepted "
                                    f"{counts['AUTO_PASS']} · Reviewed {counts['REVIEWED']} · Needs review "
                                    f"{counts['NEEDS_REVIEW']} · Failed {counts['FAILED']}")
        self.review_table.blockSignals(False)
        self._filter_review()

    def _filter_review(self) -> None:
        if self.job_root is None:
            return
        status_filter = self.review_filter.currentText()
        allowed = {"Needs Review": {"NEEDS_REVIEW"}, "Failed": {"FAILED"},
                   "Reviewed": {"REVIEWED"}, "Auto Pass": {"AUTO_PASS"}}
        rows = load_job(self.job_root)["recordings"]
        for index, row in enumerate(rows):
            self.review_table.setRowHidden(index, status_filter != "All"
                                           and row["state"] not in allowed[status_filter])

    def _next_flagged(self, direction: int) -> None:
        if self.job_root is None:
            return
        rows = load_job(self.job_root)["recordings"]
        current = self.review_table.currentRow()
        for index in range(current + direction, len(rows) if direction > 0 else -1, direction):
            if rows[index]["state"] in {"NEEDS_REVIEW", "FAILED"}:
                self.review_filter.setCurrentText("Needs Review" if rows[index]["state"] ==
                                                  "NEEDS_REVIEW" else "Failed")
                self.review_table.selectRow(index)
                return

    def _review_selected(self) -> None:
        if (self.job_root is None or self._thread is not None or
                self.tabs.currentIndex() != 2):
            return
        index = self.review_table.currentRow()
        rows = load_job(self.job_root)["recordings"]
        if 0 <= index < len(rows) and rows[index].get("textgrid_path"):
            self._pending_review_record = rows[index]
            self._start("review_load", load_recording_review,
                        {"job_root": self.job_root, "record": rows[index]})

    def _review_updated(self) -> None:
        if self.job_root is None:
            return
        selected = self.review_table.currentRow()
        self._show_job()
        if selected >= 0:
            self.review_filter.setCurrentText("All")
            self.review_table.selectRow(selected)
