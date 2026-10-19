"""Batch setup and existing review workflow for standalone VSLP Alignment."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from vslp.acoustic.alignment import freeze_alignment, run_acoustic_alignment
from vslp.acoustic.alignment.package_v1 import export_alignment_package
from vslp.acoustic.alignment.self_test import run_mfa_self_test
from vslp.acoustic.alignment.task_workflow import (
    check_mfa_environment, check_task_preflight, load_project_choices,
    project_alignment_context, run_task_alignment, save_project_choices,
    task_registry,
)
from vslp.acoustic.ingest.stage import run_acoustic_ingest
from vslp.acoustic.preprocess.stage import PreprocessConfig, run_acoustic_preprocess
from vslp.acoustic.run_setup import initialize_acoustic_run
from vslp.acoustic.segment.pipeline import SegmentationConfig, run_acoustic_segmentation, segmentation_runs
from vslp.acoustic.segment.review import initialize_segmentation_review, select_segmentation_run
from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget
from vslp.gui.acoustic_app.review_widget import SegmentationReviewWidget
from vslp.gui.common.utils import open_path
from vslp.gui.common.worker import Worker


def alignment_tasks() -> list[dict]:
    return [task for task in task_registry()["tasks"]
            if task.get("alignment_applicable") and len(task.get("prompts", [])) == 1]


def prepare_inputs(input_path: Path, output_root: Path, progress_callback=None) -> dict:
    """Use the existing segmentation workflow; human review remains required."""
    ingest = run_acoustic_ingest(input_path, output_root, progress_callback=progress_callback)
    preprocess = run_acoustic_preprocess(ingest.summary_table, output_root,
                                        PreprocessConfig(), progress_callback=progress_callback)
    segment = run_acoustic_segmentation(preprocess.summary_table, output_root,
                                        SegmentationConfig(), progress_callback=progress_callback)
    runs = segmentation_runs(output_root)
    if len(runs) == 1:
        select_segmentation_run(output_root, runs[0]["segmentation_run_id"])
        initialize_segmentation_review(segment.summary_table, output_root)
    return {"ingest": ingest, "preprocess": preprocess, "segment": segment}


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
        subtitle = QLabel("Prepare audio  →  Review trials  →  Run MFA  →  Finalize  →  Export package")
        subtitle.setObjectName("SubtitleLabel")
        outer.addWidget(subtitle)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.tabs.addTab(self._setup_tab(), "Setup")
        self.tabs.addTab(self._batch_tab(), "Batch")
        self.segmentation_review = SegmentationReviewWidget(self._project_root)
        self.segmentation_review.finalized.connect(lambda _result: self._refresh_batch())
        self.tabs.addTab(self.segmentation_review, "Segmentation Review")
        self.review = AlignmentWidget(self._project_root)
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
        self.review.setMinimumHeight(1050)
        review_scroll = QScrollArea()
        review_scroll.setWidgetResizable(True)
        review_scroll.setWidget(self.review)
        self.review_scroll = review_scroll
        self.tabs.addTab(review_scroll, "Alignment Review")
        self.progress = QProgressBar()
        outer.addWidget(self.progress)
        self.progress_label = QLabel("Ready")
        outer.addWidget(self.progress_label)
        self.technical = QGroupBox("Technical log")
        self.technical.setCheckable(True)
        self.technical.setChecked(False)
        self.technical.setMaximumHeight(32)
        self.technical.toggled.connect(lambda open_: self.technical.setMaximumHeight(
            16777215 if open_ else 32))
        log_layout = QVBoxLayout(self.technical)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        log_layout.addWidget(self.log)
        outer.addWidget(self.technical)
        self.review.technical_details_requested.connect(lambda: self.technical.setChecked(True))

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
        prepare = QPushButton("Prepare alignment")
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
        self.batch_counts = QLabel("CURRENT 0    QUEUED 0    RUNNING 0    NEEDS REVIEW 0    FAILED 0")
        layout.addWidget(self.batch_counts)
        actions = QHBoxLayout()
        self.run_button = QPushButton("Run missing/stale")
        self.run_button.clicked.connect(self._run_missing_stale)
        self.review_button = QPushButton("Review")
        self.review_button.clicked.connect(self._review_alignment)
        self.export_button = QPushButton("Export Alignment Package")
        self.export_button.clicked.connect(self._export)
        refresh = QPushButton("Refresh status")
        refresh.clicked.connect(self._refresh_batch)
        for button in (self.run_button, self.review_button, self.export_button, refresh):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.batch_table = QTableWidget(0, 7)
        self.batch_table.setHorizontalHeaderLabels(
            ["Recording", "Task", "Repetitions", "Status", "Progress", "Review", "Failure reason"])
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.batch_table, 1)
        return panel

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
        files, _ = QFileDialog.getOpenFileNames(self, "Select alignment audio", "",
            "Audio (*.wav *.mp3 *.mp4 *.m4a *.ogg *.flac *.aac *.aiff *.webm *.mov)")
        if files:
            self._input_files = [Path(file) for file in files]
            self.input_label.setText(f"{len(files)} recording(s) selected")

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select audio folder")
        if folder:
            self._input_files = []
            self.input_label.setText(folder)

    def _choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose project output parent")
        if folder:
            self.output_parent.setText(folder)

    def _open_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open existing VSLP project")
        if folder:
            try:
                self.open_project(Path(folder))
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Project unavailable", str(exc))

    def open_project(self, root: Path) -> None:
        manifest = root / "project_manifest.json"
        if not manifest.is_file():
            raise ValueError("VSLP project_manifest.json is missing")
        data = json.loads(manifest.read_text(encoding="utf-8"))
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
        job_file = root / "configs" / "alignment_job.json"
        if job_file.is_file():
            job = json.loads(job_file.read_text(encoding="utf-8"))
            self.repetitions.setValue(int(job["expected_repetitions"]))
        self.segmentation_review.refresh()
        self.review.refresh()
        self._refresh_batch()
        self.tabs.setCurrentIndex(1)

    def _prepare(self) -> None:
        if self._root is not None:
            self._refresh_batch()
            self.tabs.setCurrentIndex(1)
            return
        parent_text = self.output_parent.text().strip()
        if not parent_text:
            QMessageBox.warning(self, "Output required", "Choose a project output folder.")
            return
        parent = Path(parent_text).expanduser().resolve()
        if self._input_files:
            selected = parent / "alignment_input_selections" / uuid4().hex
            selected.mkdir(parents=True)
            for file in self._input_files:
                shutil.copy2(file, selected / file.name)
            input_path = selected
        else:
            input_path = Path(self.input_label.text())
        if not input_path.is_dir():
            QMessageBox.warning(self, "Audio required", "Add audio files or a folder first.")
            return
        task = self._selected_task()
        prompt = task["prompts"][0]
        count = int(self.repetitions.value())
        if task["task_id"] == "bamboo_passage" and count != 1:
            QMessageBox.warning(self, "Protocol conflict", "Bamboo requires one passage production.")
            return
        try:
            project = initialize_acoustic_run(
                project_name=self.project_name.text(), task_name=task["display_name"],
                task_type=task["task_type"], input_folder=input_path,
                output_parent=parent, setup_values={
                    "project_name": self.project_name.text(), "task_name": task["display_name"],
                    "task_type": task["task_type"], "input_folder": str(input_path),
                    "output_parent": str(parent)},
            )
            self._root = project.root
            (project.root / "configs" / "alignment_job.json").write_text(json.dumps({
                "contract_version": "VSLP_ALIGNMENT_JOB_V1", "task_id": task["task_id"],
                "task_type": task["task_type"], "task_variant": task["display_name"],
                "prompt_id": prompt["prompt_id"], "prompt_version": prompt["prompt_version"],
                "canonical_transcript": prompt["exact_expected_text"],
                "expected_repetitions": count,
            }, indent=2), encoding="utf-8")
            self._run("prepare", prepare_inputs,
                      {"input_path": input_path, "output_root": project.root})
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Preparation failed", str(exc))

    def _run(self, name: str, function, kwargs: dict) -> None:
        if self._thread is not None:
            QMessageBox.information(self, "Busy", "An alignment operation is still running.")
            return
        self._thread = QThread()
        self._worker = Worker(name, function, kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.message.connect(self.log.appendPlainText)
        self._worker.progress.connect(self._progress)
        self._worker.finished.connect(self._finished)
        self._worker.failed.connect(self._failed)
        self._worker.finished.connect(self._cleanup)
        self._worker.failed.connect(self._cleanup)
        if name == "alignment":
            self.review.alignment_started()
        self._thread.start()

    def _progress(self, done: int, total: int, message: str) -> None:
        if total:
            self.progress.setRange(0, total)
            self.progress.setValue(min(done, total))
            self.progress_label.setText(f"{round(100 * done / total)}% · {done}/{total} · {message}")
        else:
            self.progress.setRange(0, 0)
            self.progress_label.setText(message)
        if self._worker is not None and self._worker.name == "alignment":
            self.review.alignment_progress(done, total, message)

    def _finished(self, name: str, result: object) -> None:
        self.progress_label.setText(f"{name} completed")
        if name == "prepare":
            self.segmentation_review.refresh()
            self.tabs.setCurrentWidget(self.segmentation_review)
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
            self.review.refresh()
        elif name == "export":
            target = Path(result)
            manifest = json.loads((target / "package_manifest.json").read_text(encoding="utf-8"))
            message = (f"Package exported successfully\n{target}\n\n"
                       f"Recordings: {manifest['recording_count']}\n"
                       f"Finalized: {manifest['finalized_count']}\n"
                       f"Failed/not finalized: {manifest['failed_count']}")
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Alignment Package exported")
            dialog.setText(message)
            open_button = dialog.addButton("Open package folder", QMessageBox.ActionRole)
            dialog.addButton(QMessageBox.Close)
            dialog.exec()
            if dialog.clickedButton() is open_button:
                open_path(target)
        self._refresh_batch()

    def _failed(self, name: str, error: str) -> None:
        self.progress_label.setText(f"{name} failed")
        self.log.appendPlainText(error)
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
        self.review.refresh()
        self.tabs.setCurrentWidget(self.review_scroll)

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
        parent = QFileDialog.getExistingDirectory(self, "Choose package destination parent")
        if not parent:
            return
        target = Path(parent) / f"VSLP_Alignment_Package_V1_{uuid4().hex[:8]}"
        self._run("export", export_alignment_package,
                  {"project_root": self._root, "destination": target})

    def _refresh_batch(self) -> None:
        if self._root is None:
            return
        try:
            context = project_alignment_context(self._root)
            records = context.get("records", [])
            choices = context.get("choices", {})
            final_path = self._root / "acoustic" / "004_alignment" / "final" / "final_alignment_contract_trials.csv"
            finalized = set()
            if final_path.is_file():
                finalized = set(pd.read_csv(final_path, usecols=["recording_id"])[
                    "recording_id"].astype(str))
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
                          "100%" if state == "CURRENT" else "—",
                          "Finalized" if state == "CURRENT" else "Review required",
                          "; ".join(str(flag) for trial in trials for flag in trial.get("flags", []))]
                for column, value in enumerate(values):
                    self.batch_table.setItem(index, column, QTableWidgetItem(value))
            self.batch_counts.setText(
                f"CURRENT {counts['CURRENT']}    QUEUED {counts['QUEUED']}    "
                f"RUNNING {int(self._thread is not None)}    "
                f"NEEDS REVIEW {counts['NEEDS REVIEW']}    FAILED {counts['FAILED']}")
            self.export_button.setEnabled(bool(finalized))
        except (OSError, ValueError, KeyError) as exc:
            self.batch_heading.setText(f"Alignment project issue: {exc}")
            self.export_button.setEnabled(False)
