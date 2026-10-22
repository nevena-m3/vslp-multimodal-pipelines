"""Batch setup and existing review workflow for standalone VSLP Alignment."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

import pandas as pd
from PySide6.QtCore import QSettings, QThread, QTimer, Qt
from PySide6.QtWidgets import (
    QComboBox, QDockWidget, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from vslp.acoustic.alignment import (freeze_alignment, list_alignment_runs,
                                     load_final_alignment, run_acoustic_alignment)
from vslp.acoustic.alignment.corrections import load_corrections, reviewed_tables
from vslp.acoustic.alignment.package_v1 import export_alignment_package
from vslp.acoustic.alignment.self_test import run_mfa_self_test
from vslp.acoustic.alignment.task_workflow import (
    check_mfa_environment, check_task_preflight, load_project_choices,
    project_alignment_context, run_task_alignment, save_project_choices,
    task_registry,
)
from vslp.acoustic.run_setup import initialize_acoustic_run
from vslp.gui.alignment_app.inputs import prepare_alignment_inputs
from vslp.gui.alignment_app.review_editor import AlignmentReviewEditor
from vslp.gui.common.utils import open_path
from vslp.gui.common.worker import Worker


def alignment_tasks() -> list[dict]:
    return [task for task in task_registry()["tasks"]
            if task.get("alignment_applicable") and len(task.get("prompts", [])) == 1]


def _local_directory(path: str | Path | None) -> Path | None:
    """Never probe a UNC path or nonlocal drive on the GUI thread."""
    if not path:
        return None
    candidate = Path(path).expanduser()
    raw = str(candidate)
    if raw.startswith(("\\\\", "//")) or not candidate.is_absolute():
        return None
    if os.name == "nt":
        import ctypes

        drive = candidate.drive
        if not drive or ctypes.windll.kernel32.GetDriveTypeW(f"{drive}\\") != 3:
            return None
    return candidate if candidate.is_dir() else None


def validate_export_folder(path: str | Path) -> Path:
    """Create a local destination if needed and prove it accepts a small write."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or str(candidate).startswith(("\\\\", "//")):
        raise ValueError("Choose a local export folder.")
    if os.name == "nt":
        import ctypes

        if (not candidate.drive or
                ctypes.windll.kernel32.GetDriveTypeW(f"{candidate.drive}\\") != 3):
            raise ValueError("Choose a local export folder.")
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".vslp_export_check_", dir=candidate) as probe:
            probe.write(b"VSLP")
            probe.flush()
    except OSError as exc:
        raise PermissionError(f"Export folder is not writable: {candidate}") from exc
    return candidate


def load_project_snapshot(root: Path) -> dict:
    """Read project metadata away from Qt; never load waveforms here."""
    manifest = root / "project_manifest.json"
    if not manifest.is_file():
        raise ValueError("VSLP project_manifest.json is missing")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    context = project_alignment_context(root)
    final_path = root / "acoustic" / "004_alignment" / "final" / "final_alignment_contract_trials.csv"
    finalized = (set(pd.read_csv(final_path, usecols=["recording_id"])["recording_id"].astype(str))
                 if final_path.is_file() else set())
    job_file = root / "configs" / "alignment_job.json"
    job = json.loads(job_file.read_text(encoding="utf-8")) if job_file.is_file() else {}
    return {"root": root, "manifest": data, "context": context,
            "finalized": finalized, "job": job}


def prepare_alignment_project(*, input_files: list[Path], input_path: Path,
                              parent: Path, project_name: str, task: dict,
                              expected_repetitions: int, progress_callback=None) -> Path:
    """Copy, create and preprocess a project without occupying Qt's event loop."""
    if input_files:
        selected = parent / "alignment_input_selections" / uuid4().hex
        selected.mkdir(parents=True)
        for file in input_files:
            shutil.copy2(file, selected / file.name)
        input_path = selected
    if not input_path.is_dir():
        raise ValueError("Add audio files or a folder first.")
    prompt = task["prompts"][0]
    project = initialize_acoustic_run(
        project_name=project_name, task_name=task["display_name"],
        task_type=task["task_type"], input_folder=input_path,
        output_parent=parent, setup_values={
            "project_name": project_name, "task_name": task["display_name"],
            "task_type": task["task_type"], "input_folder": str(input_path),
            "output_parent": str(parent)},
    )
    (project.root / "configs" / "alignment_job.json").write_text(json.dumps({
        "contract_version": "VSLP_ALIGNMENT_JOB_V1", "task_id": task["task_id"],
        "task_type": task["task_type"], "task_variant": task["display_name"],
        "prompt_id": prompt["prompt_id"], "prompt_version": prompt["prompt_version"],
        "canonical_transcript": prompt["exact_expected_text"],
        "expected_repetitions": expected_repetitions,
    }, indent=2), encoding="utf-8")
    prepare_alignment_inputs(input_path, project.root, progress_callback=progress_callback)
    return project.root


def preload_review_data(root: Path) -> dict:
    """Parse review artifacts on a worker before rendering the active recording."""
    runs = list_alignment_runs(root)
    stored = load_project_choices(root).get("selected_run_id")
    run_id = stored if any(run["alignment_run_id"] == stored for run in runs) else (
        runs[-1]["alignment_run_id"] if runs else "")
    payload: dict = {"run_id": run_id}
    if run_id:
        payload["tables"] = reviewed_tables(root, run_id)
        payload["corrections"] = load_corrections(root, run_id)
        links = root / "acoustic" / "004_alignment" / "runs" / run_id / "tables" / "alignment_trial_tokens.csv"
        payload["links"] = (pd.read_csv(links, keep_default_na=False)
                            if links.is_file() else pd.DataFrame())
    payload["final_alignment"] = load_final_alignment(root)
    return payload


class AlignmentWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VSLP Alignment")
        self.resize(1450, 900)
        self.setMinimumSize(1080, 700)
        self._root: Path | None = None
        self._input_files: list[Path] = []
        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self._last_local_folder: Path | None = None
        self._last_exported_package: Path | None = None
        self._review_loaded = False
        self._build_ui()
        self._task_changed()

    def _project_root(self) -> Path:
        return self._root or Path.cwd() / "__no_alignment_project__"

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(16, 16, 16, 16)
        heading = QLabel("VSLP Alignment")
        heading.setObjectName("AppTitleLabel")
        outer.addWidget(heading)
        subtitle = QLabel("Setup  →  Align  →  Review  →  Export")
        subtitle.setObjectName("SubtitleLabel")
        outer.addWidget(subtitle)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.tabs.addTab(self._setup_tab(), "Setup")
        self.tabs.addTab(self._batch_tab(), "Align")
        self.review = AlignmentReviewEditor(self._project_root)
        self.review.run_requested.connect(lambda config: self._run("alignment", run_acoustic_alignment,
            {"output_root": self._project_root(), "config": config}))
        self.review.freeze_requested.connect(self._freeze)
        self.review.run_task_requested.connect(lambda root, profile: self._run(
            "alignment", run_task_alignment, {"output_root": root, "profile_path": profile}))
        self.review.rerun_recording_requested.connect(lambda root, profile, record, run: self._run(
            "alignment", run_task_alignment, {"output_root": root, "profile_path": profile,
                                                "rerun_recording_id": record, "reuse_run_id": run}))
        self.review.preflight_requested.connect(lambda root, profile: self._run(
            "alignment_preflight", check_task_preflight, {"root": root, "profile_path": profile}))
        self.review.environment_requested.connect(lambda profile: self._run(
            "alignment_environment", check_mfa_environment, {"profile_path": profile}))
        self.review.self_test_requested.connect(lambda profile, wav: self._run(
            "alignment_self_test", run_mfa_self_test,
            {"profile_path": profile, "fallback_wav_path": wav or None}))
        self.review.plot_requested.connect(self._plot_trial)
        # The review controls and three linked tracks need their natural height.
        # Scroll vertically in the standalone window instead of compressing them.
        self.review.setMinimumHeight(0)
        review_scroll = QScrollArea()
        review_scroll.setWidgetResizable(True)
        review_scroll.setWidget(self.review)
        self.review_scroll = review_scroll
        self.tabs.addTab(review_scroll, "Review")
        self.tabs.addTab(self._export_tab(), "Export")
        self.progress = QProgressBar()
        outer.addWidget(self.progress)
        self.progress_label = QLabel("Ready")
        outer.addWidget(self.progress_label)
        self.technical = QDockWidget("Technical Log", self)
        self.technical.setFeatures(QDockWidget.NoDockWidgetFeatures)
        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        log_controls = QHBoxLayout()
        self.log_toggle = QPushButton("− Collapse")
        self.log_toggle.clicked.connect(self._toggle_log)
        log_controls.addWidget(self.log_toggle)
        clear_log = QPushButton("Clear")
        clear_log.clicked.connect(lambda: self.log.clear())
        log_controls.addWidget(clear_log)
        log_controls.addStretch()
        log_layout.addLayout(log_controls)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(90)
        self._log_expanded = True
        self._log_user_toggled = False
        log_layout.addWidget(self.log)
        self.technical.setWidget(log_panel)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.technical)
        self.technical.setMinimumHeight(155)
        self.review.technical_details_requested.connect(self._expand_log)
        self.statusBar().showMessage("Ready")
        self._log_message("Application ready")

    def _toggle_log(self) -> None:
        if self.sender() is self.log_toggle:
            self._log_user_toggled = True
        expanded = not self._log_expanded
        self._log_expanded = expanded
        self.log.setVisible(expanded)
        self.log_toggle.setText("− Collapse" if expanded else "+ Expand")
        self.technical.setMinimumHeight(155 if expanded else 72)
        self.technical.setMaximumHeight(16777215 if expanded else 72)
        if expanded:
            self.resizeDocks([self.technical], [180], Qt.Vertical)

    def _expand_log(self) -> None:
        if not self._log_expanded:
            self._toggle_log()

    def _log_message(self, message: str, level: str = "INFO") -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"{stamp}  {level}  {message}")

    def _dialog_start(self) -> str:
        settings = QSettings("VSLP", "Alignment")
        candidates = (self._last_local_folder, settings.value("last_local_folder"),
                      self._root, Path.home() / "Documents", Path.cwd())
        for candidate in candidates:
            local = _local_directory(candidate)
            if local is not None:
                return str(local)
        return str(Path.cwd())

    def _choose_folder(self, title: str) -> str:
        folder = QFileDialog.getExistingDirectory(
            self, title, self._dialog_start(), QFileDialog.DontUseNativeDialog)
        if folder:
            local = _local_directory(folder)
            if local is None:
                self._log_message("Selected folder is not an accessible local folder", "WARNING")
                QMessageBox.warning(self, "Folder unavailable",
                                    "Choose an accessible local folder.")
                return ""
            self._last_local_folder = local
            QSettings("VSLP", "Alignment").setValue("last_local_folder", str(local))
        return folder

    def _setup_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        audio = QGroupBox("INPUT AUDIO")
        audio_layout = QVBoxLayout(audio)
        buttons = QHBoxLayout()
        for text, callback in (("Add files", self._add_files),
                               ("Add folder", self._add_folder),
                               ("Open existing VSLP project", self._open_project)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        audio_layout.addLayout(buttons)
        self.input_label = QLabel("No recordings selected")
        audio_layout.addWidget(self.input_label)
        layout.addWidget(audio)
        task_box = QGroupBox("TASK")
        form = QFormLayout(task_box)
        self.task_type = QComboBox()
        self.task_type.addItem("Bamboo Passage", "bamboo_passage")
        self.task_type.addItem("Sentence", "sentence")
        self.task_type.currentIndexChanged.connect(self._task_changed)
        form.addRow("Task type", self.task_type)
        self.variant = QComboBox()
        for task in alignment_tasks():
            if task["task_id"] != "bamboo_passage":
                label = "S1" if task["task_id"] == "wstg" else task["display_name"]
                self.variant.addItem(label, task["task_id"])
        self.variant.currentIndexChanged.connect(self._task_changed)
        form.addRow("Sentence", self.variant)
        self.repetitions = QSpinBox()
        self.repetitions.setRange(1, 100)
        form.addRow("Expected repetitions", self.repetitions)
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setMaximumHeight(125)
        form.addRow("Canonical transcript", self.transcript)
        layout.addWidget(task_box)
        destination = QGroupBox("PROJECT")
        destination_form = QFormLayout(destination)
        self.project_name = QLineEdit("VSLP Alignment")
        destination_form.addRow("Project name", self.project_name)
        destination_row = QHBoxLayout()
        self.output_parent = QLineEdit()
        destination_row.addWidget(self.output_parent)
        choose_output = QPushButton("Choose output folder")
        choose_output.clicked.connect(self._choose_output)
        destination_row.addWidget(choose_output)
        destination_form.addRow("Save project in", destination_row)
        layout.addWidget(destination)
        prepare = QPushButton("Continue to Alignment")
        prepare.setObjectName("RunButton")
        prepare.clicked.connect(self._prepare)
        layout.addWidget(prepare)
        layout.addStretch()
        return panel

    def _batch_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.batch_heading = QLabel("Select or prepare a project")
        layout.addWidget(self.batch_heading)
        self.batch_counts = QLabel("Current 0    Needs Review 0    Failed 0")
        layout.addWidget(self.batch_counts)
        actions = QHBoxLayout()
        self.run_button = QPushButton("Run Alignment")
        self.run_button.clicked.connect(self._run_missing_stale)
        retry_button = QPushButton("Retry Failed")
        retry_button.clicked.connect(self._run_missing_stale)
        self.review_button = QPushButton("Go to Review")
        self.review_button.clicked.connect(self._review_alignment)
        refresh = QPushButton("Refresh status")
        refresh.clicked.connect(self._refresh_batch)
        for button in (self.run_button, retry_button, self.review_button, refresh):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.batch_table = QTableWidget(0, 5)
        self.batch_table.setHorizontalHeaderLabels(
            ["Recording", "Task", "Repetitions", "Status", "Issue"])
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.batch_table, 1)
        return panel

    def _export_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("READY TO EXPORT")
        title.setObjectName("AppTitleLabel")
        layout.addWidget(title)
        self.export_counts = QLabel("Finalized 0    Needs Review 0    Failed 0")
        layout.addWidget(self.export_counts)
        layout.addWidget(QLabel("Output folder"))
        destination = QHBoxLayout()
        self.export_parent = QLineEdit()
        self.export_parent.setPlaceholderText("Choose a writable local export folder")
        destination.addWidget(self.export_parent, 1)
        self.choose_export_button = QPushButton("Choose output folder")
        self.choose_export_button.clicked.connect(self._choose_export_folder)
        destination.addWidget(self.choose_export_button)
        layout.addLayout(destination)
        self.export_button = QPushButton("Export Alignment Package")
        self.export_button.setObjectName("RunButton")
        self.export_button.clicked.connect(self._export)
        layout.addWidget(self.export_button)
        self.export_result = QLabel("")
        self.export_result.setWordWrap(True)
        layout.addWidget(self.export_result)
        self.open_export_button = QPushButton("Open folder")
        self.open_export_button.setEnabled(False)
        self.open_export_button.clicked.connect(self._open_export_folder)
        layout.addWidget(self.open_export_button)
        layout.addWidget(QLabel("Package format: VSLP_ALIGNMENT_PACKAGE_V1"))
        layout.addStretch()
        return panel

    def _export_candidates(self) -> tuple[Path, ...]:
        remembered = QSettings("VSLP", "Alignment").value("last_export_folder")
        candidates = []
        if remembered:
            candidates.append(Path(str(remembered)))
        if self._root is not None:
            candidates.append(self._root / "exports")
        candidates.extend((Path.home() / "Documents" / "AlignmentExports",
                           Path.home() / "Documents" / "Codex" / "AlignmentExports",
                           Path.cwd() / "AlignmentExports"))
        return tuple(candidates)

    def _safe_export_default(self) -> Path | None:
        for candidate in self._export_candidates():
            # A project can live under a redirected Desktop junction even when
            # its spelling begins with C:. Never silently default there.
            if any(part.lower() == "desktop" for part in candidate.parts):
                continue
            try:
                return validate_export_folder(candidate)
            except (OSError, ValueError) as exc:
                self._log_message(f"Export fallback rejected: {candidate}: {exc}", "WARNING")
        return None

    def _choose_export_folder(self) -> None:
        folder = self._choose_folder("Choose Alignment Package output folder")
        if folder:
            try:
                selected = validate_export_folder(folder)
            except (OSError, ValueError) as exc:
                self._log_message(f"Export path validation failed: {folder}: {exc}", "WARNING")
                QMessageBox.warning(self, "Export folder unavailable", str(exc))
                return
            self.export_parent.setText(str(selected))
            self.export_result.clear()

    def _open_export_folder(self) -> None:
        if self._last_exported_package is not None:
            open_path(self._last_exported_package)

    def _selected_task(self) -> dict:
        task_id = ("bamboo_passage" if self.task_type.currentData() == "bamboo_passage"
                   else self.variant.currentData())
        return next(task for task in alignment_tasks() if task["task_id"] == task_id)

    def _task_changed(self) -> None:
        if not hasattr(self, "variant"):
            return
        bamboo = self.task_type.currentData() == "bamboo_passage"
        self.variant.setVisible(not bamboo)
        self.variant.parentWidget().layout().labelForField(self.variant).setVisible(not bamboo)
        task = self._selected_task()
        prompt = task["prompts"][0]
        self.transcript.setPlainText(prompt["exact_expected_text"])
        self.repetitions.setValue(int(prompt.get("expected_repetitions") or 1))
        self.repetitions.setEnabled(not bamboo)

    def _add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Select alignment audio",
            self._dialog_start(),
            "Audio (*.wav *.mp3 *.mp4 *.m4a *.ogg *.flac *.aac *.aiff *.webm *.mov)",
            None, QFileDialog.DontUseNativeDialog)
        if files:
            self._input_files = [Path(file) for file in files]
            self.input_label.setText(f"{len(files)} recording(s) selected")
            self._last_local_folder = _local_directory(Path(files[0]).parent)

    def _add_folder(self) -> None:
        folder = self._choose_folder("Select audio folder")
        if folder:
            self._input_files = []
            self.input_label.setText(folder)

    def _choose_output(self) -> None:
        folder = self._choose_folder("Choose project output parent")
        if folder:
            self.output_parent.setText(folder)

    def _open_project(self) -> None:
        folder = self._choose_folder("Open existing VSLP project")
        if folder:
            self.open_project(Path(folder))

    def open_project(self, root: Path) -> None:
        self.progress_label.setText("Loading project...")
        self.statusBar().showMessage("Loading project...")
        self._log_message(f"Loading project: {root}")
        self._expand_log()
        self._run("project_load", load_project_snapshot, {"root": root})

    def _apply_project_snapshot(self, snapshot: dict) -> None:
        root = snapshot["root"]
        data = snapshot["manifest"]
        task_id = str(data.get("task_id", ""))
        if task_id in {"bamboo", "bamboo_passage"}:
            self.task_type.setCurrentIndex(0)
        else:
            variant = self.variant.findData(task_id)
            if variant < 0:
                raise ValueError(f"Task is not registered for Alignment: {task_id}")
            self.task_type.setCurrentIndex(1)
            self.variant.setCurrentIndex(variant)
        self._root = root
        self.input_label.setText(str(data.get("input_folder", "")))
        self.project_name.setText(str(data.get("project_name", "")))
        self.output_parent.setText(str(data.get("output_parent", "")))
        job = snapshot["job"]
        if job:
            self.repetitions.setValue(int(job["expected_repetitions"]))
        self._review_loaded = False
        self._apply_batch_snapshot(snapshot)
        export_default = self._safe_export_default()
        self.export_parent.setText(str(export_default) if export_default else "")
        self.tabs.setCurrentIndex(1)
        self._log_message(f"Found {len(snapshot['context'].get('records', []))} recordings")

    def _prepare(self) -> None:
        if self._root is not None:
            self._refresh_batch()
            self.tabs.setCurrentIndex(1)
            return
        parent_text = self.output_parent.text().strip()
        if not parent_text:
            QMessageBox.warning(self, "Output required", "Choose a project output folder.")
            return
        parent = Path(parent_text).expanduser()
        input_path = Path(self.input_label.text()) if not self._input_files else parent
        task = self._selected_task()
        count = int(self.repetitions.value())
        if task["task_id"] == "bamboo_passage" and count != 1:
            QMessageBox.warning(self, "Protocol conflict", "Bamboo requires one passage production.")
            return
        self._run("prepare", prepare_alignment_project,
                  {"input_files": list(self._input_files), "input_path": input_path,
                   "parent": parent, "project_name": self.project_name.text(),
                   "task": task, "expected_repetitions": count})

    def _run(self, name: str, function, kwargs: dict) -> None:
        if self._thread is not None:
            QMessageBox.information(self, "Busy", "An alignment operation is still running.")
            return
        self.progress.setRange(0, 0)
        self._thread = QThread()
        self._worker = Worker(name, function, kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.message.connect(self._log_message)
        self._worker.progress.connect(self._progress)
        self._worker.finished.connect(self._finished)
        self._worker.failed.connect(self._failed)
        self._worker.finished.connect(self._cleanup)
        self._worker.failed.connect(self._cleanup)
        if name == "alignment":
            self.review.alignment_started()
        self.statusBar().showMessage(f"{name.replace('_', ' ').title()}...")
        self._expand_log()
        self._thread.start()

    def _progress(self, done: int, total: int, message: str) -> None:
        if total:
            self.progress.setRange(0, total)
            self.progress.setValue(min(done, total))
            self.progress_label.setText(f"{round(100 * done / total)}% · {done}/{total} · {message}")
        else:
            self.progress.setRange(0, 0)
            self.progress_label.setText(message)
        self.statusBar().showMessage(self.progress_label.text())
        if self._worker is not None and self._worker.name == "alignment":
            self.review.alignment_progress(done, total, message)

    def _finished(self, name: str, result: object) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress_label.setText(f"{name} completed")
        self.statusBar().showMessage(self.progress_label.text())
        self._log_message(f"{name} completed")
        if name == "project_load":
            self._apply_project_snapshot(result)
            if self._log_expanded and not self._log_user_toggled:
                self._toggle_log()
        elif name == "batch_refresh":
            self._apply_batch_snapshot(result)
        elif name == "review_preload":
            self.review._standalone_review_payload = result
            QTimer.singleShot(0, self._load_review)
        elif name == "prepare":
            self._root = Path(result)
            self.tabs.setCurrentIndex(1)
        elif name == "alignment_preflight":
            self.review.show_preflight_result(result)
        elif name == "alignment_environment":
            self.review.show_environment_result(result)
        elif name == "alignment_self_test":
            self.review.show_self_test_result(result)
        elif name == "alignment":
            self.review.alignment_finished()
            manifest_path = getattr(result, "manifest_path", None)
            if manifest_path and Path(manifest_path).is_file():
                run_id = json.loads(Path(manifest_path).read_text(encoding="utf-8")).get(
                    "alignment_run_id", "")
                if run_id:
                    choices = load_project_choices(self._project_root())
                    choices["selected_run_id"] = run_id
                    save_project_choices(self._project_root(), choices)
            self._review_loaded = False
        elif name == "export":
            target = Path(result)
            self._last_exported_package = target
            QSettings("VSLP", "Alignment").setValue("last_export_folder", str(target.parent))
            manifest = json.loads((target / "package_manifest.json").read_text(encoding="utf-8"))
            message = (f"Package exported successfully\n{target}\n\n"
                       f"Recordings: {manifest['recording_count']}\n"
                       f"Finalized: {manifest['finalized_count']}\n"
                       f"Failed/not finalized: {manifest['failed_count']}")
            self.export_result.setText(message)
            self.open_export_button.setEnabled(True)
            self._log_message(f"Package exported successfully: {target}")
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Alignment Package exported")
            dialog.setText(message)
            open_button = dialog.addButton("Open package folder", QMessageBox.ActionRole)
            dialog.addButton(QMessageBox.Close)
            dialog.exec()
            if dialog.clickedButton() is open_button:
                open_path(target)
        if name in {"prepare", "alignment", "export"}:
            QTimer.singleShot(0, self._refresh_batch)

    def _failed(self, name: str, error: str) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label.setText(f"{name} failed")
        self.statusBar().showMessage(self.progress_label.text())
        self._log_message(f"{name}: {error}", "ERROR")
        if name == "alignment":
            self.review.alignment_finished(error)
        elif name == "alignment_preflight":
            self.review.show_preflight_result({"status": "ACTION_REQUIRED",
                                               "issue": error.splitlines()[0]})
        QMessageBox.critical(self, f"{name} failed", error.splitlines()[0])

    def _cleanup(self, *_args) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None

    def _freeze(self, run_id: str) -> None:
        if not run_id:
            return
        final = self._project_root() / "acoustic" / "004_alignment" / "final" / "final_alignment_manifest.json"
        supersede = False
        if final.is_file():
            current = json.loads(final.read_text(encoding="utf-8")).get("alignment_run_id")
            if current == run_id:
                QMessageBox.information(self, "Already finalized", "This run is already finalized.")
                return
            supersede = QMessageBox.question(self, "Replace Final Alignment",
                "Finalize this reviewed run and archive the previous Final Alignment?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes
            if not supersede:
                return
        self._run("alignment", freeze_alignment, {"output_root": self._project_root(),
            "alignment_run_id": run_id, "supersede": supersede})

    def _plot_trial(self, run_id: str, recording_id: str, trial_id: str) -> None:
        from vslp.acoustic.alignment.qc import generate_selected_trial_plot

        self._run("alignment_qc_plot", generate_selected_trial_plot,
            {"root": self._project_root(), "run_id": run_id,
             "recording_id": recording_id, "trial_id": trial_id})

    def _review_alignment(self) -> None:
        if self._root is None:
            QMessageBox.information(self, "Project required", "Prepare or open a project first.")
            return
        self.tabs.setCurrentWidget(self.review_scroll)
        if not self._review_loaded:
            self.progress_label.setText("Loading review data...")
            self.statusBar().showMessage("Loading review data...")
            self._log_message("Loading review data for active recording")
            self._run("review_preload", preload_review_data, {"root": self._root})

    def _load_review(self) -> None:
        self.review.refresh()
        self._review_loaded = True
        self.progress_label.setText("Review ready")
        self.statusBar().showMessage("Review ready")
        self._log_message("Review ready")
        if self._log_expanded and not self._log_user_toggled:
            self._toggle_log()

    def _run_missing_stale(self) -> None:
        if self._root is None:
            QMessageBox.information(self, "Project required", "Prepare or open a project first.")
            return
        context = project_alignment_context(self._root)
        if context.get("issue"):
            QMessageBox.information(self, "Review required", context["issue"])
            return
        trials = context.get("choices", {}).get("trials", {})
        if any(not trials.get(record["recording_id"]) or any(
                item.get("trial_status") != "READY_FOR_ALIGNMENT" for item in
                trials[record["recording_id"]]) for record in context["records"]):
            self._review_alignment()
            QMessageBox.information(self, "Confirm trials",
                "Review and confirm each recording's trial boundaries before running MFA.")
            return
        self._run("alignment", run_task_alignment,
                  {"output_root": self._root, "profile_path": ""})

    def _export(self) -> None:
        if self._root is None:
            QMessageBox.information(self, "Project required", "Open a project first.")
            return
        parent_text = self.export_parent.text().strip()
        if not parent_text:
            fallback = self._safe_export_default()
            if fallback is None:
                QMessageBox.warning(self, "Export folder required",
                                    "Choose a writable local output folder.")
                self._choose_export_folder()
                return
            self.export_parent.setText(str(fallback))
            parent_text = str(fallback)
        try:
            parent = validate_export_folder(parent_text)
        except (OSError, ValueError) as exc:
            self._log_message(f"Export path validation failed: {parent_text}: {exc}",
                              "WARNING")
            QMessageBox.warning(self, "Export folder unavailable",
                                "That output folder is not writable. Choose another location.")
            self._choose_export_folder()
            return
        self._log_message(f"Export folder validated: {parent}")
        target = parent / f"VSLP_Alignment_Package_V1_{uuid4().hex[:8]}"
        self._run("export", export_alignment_package,
                  {"project_root": self._root, "destination": target})

    def _refresh_batch(self) -> None:
        if self._root is None:
            return
        self._run("batch_refresh", load_project_snapshot, {"root": self._root})

    def _apply_batch_snapshot(self, snapshot: dict) -> None:
        try:
            context = snapshot["context"]
            records = context.get("records", [])
            choices = context.get("choices", {})
            finalized = snapshot["finalized"]
            task_name = context.get("task_name", "")
            expected = context.get("expected_repetitions") or self.repetitions.value()
            self.batch_heading.setText(
                f"{task_name} · Expected repetitions: {expected} · {len(records)} recordings")
            self.batch_table.setRowCount(len(records))
            counts = {"CURRENT": 0, "NEEDS REVIEW": 0, "QUEUED": 0, "FAILED": 0}
            for index, record in enumerate(records):
                identity = record["recording_id"]
                trials = choices.get("trials", {}).get(identity, [])
                state = ("CURRENT" if identity in finalized else
                         "NEEDS REVIEW" if trials and any(t.get("trial_status") not in
                             {"READY_FOR_ALIGNMENT", "FINALIZED"} for t in trials) else
                         "QUEUED")
                counts[state] += 1
                values = [record["file_name"], task_name, str(expected), state,
                          "; ".join(str(flag) for trial in trials for flag in trial.get("flags", []))]
                for column, value in enumerate(values):
                    self.batch_table.setItem(index, column, QTableWidgetItem(value))
            self.batch_counts.setText(
                f"Current {counts['CURRENT']}    Needs Review {counts['NEEDS REVIEW']}    "
                f"Failed {counts['FAILED']}    Queued {counts['QUEUED']}")
            self.export_counts.setText(
                f"Finalized {counts['CURRENT']}    Needs Review {counts['NEEDS REVIEW']}    "
                f"Failed {counts['FAILED']}")
            self.export_button.setEnabled(bool(finalized))
        except (OSError, ValueError, KeyError) as exc:
            self.batch_heading.setText(f"Alignment project issue: {exc}")
            self.export_button.setEnabled(False)
