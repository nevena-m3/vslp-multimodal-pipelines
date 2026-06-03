"""VSLP Acoustic Pipeline GUI v0.2.

This first GUI wraps the validated backend stages:
- project setup
- ingest
- preprocess
- Silero segmentation

The goal is not full final polish yet, but a professional local application with
clear controls, strong reproducibility, and direct access to reports/artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import traceback

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from vslp.acoustic.features.registry import build_acoustic_feature_registry
from vslp.acoustic.features.stage import FeatureExtractionConfig, run_acoustic_feature_extraction
from vslp.acoustic.ingest.stage import run_acoustic_ingest
from vslp.acoustic.preprocess.stage import FilterConfig, PreprocessConfig, run_acoustic_preprocess
from vslp.acoustic.segment.stage import run_acoustic_segmentation_silero
from vslp.core.project import initialize_project
from vslp.gui.common.utils import open_in_browser, open_path


@dataclass
class StageRecord:
    status: str = "Not run"
    summary_path: str | None = None
    report_path: str | None = None
    manifest_path: str | None = None
    errors_path: str | None = None


class Worker(QObject):
    started = Signal(str)
    finished = Signal(str, object)
    failed = Signal(str, str)
    message = Signal(str)

    def __init__(self, name: str, func: Callable, kwargs: dict):
        super().__init__()
        self.name = name
        self.func = func
        self.kwargs = kwargs

    def run(self) -> None:
        self.started.emit(self.name)
        self.message.emit(f"Starting {self.name}...")
        try:
            result = self.func(**self.kwargs)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            self.failed.emit(self.name, f"{exc}\n\n{tb}")
            return
        self.message.emit(f"Finished {self.name}: {getattr(result, 'status', 'ok')}")
        self.finished.emit(self.name, result)


class AcousticPipelineWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VSLP - Acoustic Pipeline")
        self.resize(1400, 920)

        self.stage_records: dict[str, StageRecord] = {
            "project": StageRecord(),
            "ingest": StageRecord(),
            "preprocess": StageRecord(),
            "segment": StageRecord(),
            "features": StageRecord(),
        }
        self._thread: QThread | None = None
        self._worker: Worker | None = None

        self._build_ui()
        self._refresh_stage_cards()

    # ---------------------------- UI BUILD ----------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(280)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(16, 16, 16, 16)
        side_layout.setSpacing(12)

        title = QLabel("VSLP")
        title.setObjectName("TitleLabel")
        subtitle = QLabel("Acoustic Pipeline GUI v0.2\nValidated stages first, polished workflow next.")
        subtitle.setObjectName("SubtitleLabel")
        side_layout.addWidget(title)
        side_layout.addWidget(subtitle)

        self.stage_labels: dict[str, QLabel] = {}
        for key, label in [
            ("project", "Project"),
            ("ingest", "Ingest"),
            ("preprocess", "Preprocess"),
            ("segment", "Silero Segmentation"),
            ("features", "Feature Extraction"),
        ]:
            card = QFrame()
            card.setObjectName("Card")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 12, 12, 12)
            card_layout.addWidget(QLabel(label))
            status_lbl = QLabel("Not run")
            status_lbl.setObjectName("SubtitleLabel")
            card_layout.addWidget(status_lbl)
            self.stage_labels[key] = status_lbl
            side_layout.addWidget(card)

        side_layout.addStretch(1)

        self.run_all_btn = QPushButton("Run Full Acoustic Backend")
        self.run_all_btn.setObjectName("RunButton")
        self.run_all_btn.clicked.connect(self.run_all)
        side_layout.addWidget(self.run_all_btn)

        self.open_output_btn = QPushButton("Open Output Project Folder")
        self.open_output_btn.setObjectName("OpenButton")
        self.open_output_btn.clicked.connect(self.open_output_root)
        side_layout.addWidget(self.open_output_btn)

        root.addWidget(sidebar)

        main_col = QVBoxLayout()
        main_col.setSpacing(16)

        banner = QFrame()
        banner.setObjectName("TopBanner")
        banner_layout = QVBoxLayout(banner)
        banner_layout.setContentsMargins(16, 16, 16, 16)
        banner_title = QLabel("Professional local research workflow")
        banner_title.setObjectName("TitleLabel")
        banner_sub = QLabel(
            "This GUI wraps the validated CLI/backend stages for ingest, preprocessing, Silero segmentation, and first feature extraction. "
            "All outputs remain fully reproducible on disk."
        )
        banner_sub.setObjectName("SubtitleLabel")
        banner_layout.addWidget(banner_title)
        banner_layout.addWidget(banner_sub)
        main_col.addWidget(banner)

        tabs = QTabWidget()
        tabs.addTab(self._build_setup_tab(), "Setup")
        tabs.addTab(self._build_preprocess_tab(), "Preprocess")
        tabs.addTab(self._build_segment_tab(), "Segmentation")
        tabs.addTab(self._build_features_tab(), "Features")
        tabs.addTab(self._build_reports_tab(), "Reports & Outputs")
        main_col.addWidget(tabs, stretch=1)

        log_group = QGroupBox("Run Log")
        log_layout = QVBoxLayout(log_group)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(180)
        log_layout.addWidget(self.progress)
        log_layout.addWidget(self.log_box)
        main_col.addWidget(log_group)

        root.addLayout(main_col, stretch=1)

    def _build_setup_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        paths_group = QGroupBox("Project and Input Paths")
        form = QGridLayout(paths_group)

        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.project_name_edit = QLineEdit("VSLP Acoustic Project")

        browse_in = QPushButton("Browse Input Folder")
        browse_in.clicked.connect(self.browse_input_dir)
        browse_out = QPushButton("Browse Output Project Folder")
        browse_out.clicked.connect(self.browse_output_dir)

        form.addWidget(QLabel("Input audio folder"), 0, 0)
        form.addWidget(self.input_edit, 0, 1)
        form.addWidget(browse_in, 0, 2)

        form.addWidget(QLabel("Output project folder"), 1, 0)
        form.addWidget(self.output_edit, 1, 1)
        form.addWidget(browse_out, 1, 2)

        form.addWidget(QLabel("Project name"), 2, 0)
        form.addWidget(self.project_name_edit, 2, 1, 1, 2)

        btn_row = QHBoxLayout()
        init_btn = QPushButton("Initialize Project")
        init_btn.clicked.connect(self.run_project_init)
        ingest_btn = QPushButton("Run Ingest")
        ingest_btn.clicked.connect(self.run_ingest)
        btn_row.addWidget(init_btn)
        btn_row.addWidget(ingest_btn)

        layout.addWidget(paths_group)
        layout.addLayout(btn_row)
        layout.addStretch(1)
        return container

    def _build_preprocess_tab(self) -> QWidget:
        container = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)
        outer = QVBoxLayout(container)
        outer.addWidget(scroll)
        layout = QVBoxLayout(inner)

        simple_group = QGroupBox("Core preprocessing parameters")
        form = QFormLayout(simple_group)

        self.seg_sr_spin = QSpinBox()
        self.seg_sr_spin.setRange(8000, 48000)
        self.seg_sr_spin.setValue(16000)

        self.feature_sr_edit = QLineEdit()
        self.feature_sr_edit.setPlaceholderText("leave blank to keep original sample rate")

        form.addRow("Segmentation sample rate (Hz)", self.seg_sr_spin)
        form.addRow("Feature sample rate (optional)", self.feature_sr_edit)

        self.expert_mode_check = QCheckBox("Enable expert preprocessing controls")
        self.expert_mode_check.stateChanged.connect(self._toggle_expert_controls)

        self.expert_group = QGroupBox("Expert preprocessing controls")
        expert_form = QFormLayout(self.expert_group)
        self.filter_kind_combo = QComboBox()
        self.filter_kind_combo.addItems(["none", "lpf", "hpf", "bpf", "notch"])
        self.low_hz_spin = QDoubleSpinBox(); self.low_hz_spin.setRange(0, 50000); self.low_hz_spin.setValue(80.0)
        self.high_hz_spin = QDoubleSpinBox(); self.high_hz_spin.setRange(0, 50000); self.high_hz_spin.setValue(8000.0)
        self.notch_hz_combo = QComboBox(); self.notch_hz_combo.addItems(["", "50", "60"])
        expert_form.addRow("Filter kind", self.filter_kind_combo)
        expert_form.addRow("Low cutoff (Hz)", self.low_hz_spin)
        expert_form.addRow("High cutoff (Hz)", self.high_hz_spin)
        expert_form.addRow("Notch frequency (Hz)", self.notch_hz_combo)
        self.expert_group.setVisible(False)

        run_btn = QPushButton("Run Preprocess")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_preprocess)

        layout.addWidget(simple_group)
        layout.addWidget(self.expert_mode_check)
        layout.addWidget(self.expert_group)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        return container

    def _build_segment_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        group = QGroupBox("Silero segmentation parameters")
        form = QFormLayout(group)

        self.threshold_spin = QDoubleSpinBox(); self.threshold_spin.setDecimals(2); self.threshold_spin.setRange(0.0, 1.0); self.threshold_spin.setSingleStep(0.05); self.threshold_spin.setValue(0.50)
        self.min_speech_spin = QSpinBox(); self.min_speech_spin.setRange(0, 5000); self.min_speech_spin.setValue(250)
        self.min_silence_spin = QSpinBox(); self.min_silence_spin.setRange(0, 5000); self.min_silence_spin.setValue(100)
        self.speech_pad_spin = QSpinBox(); self.speech_pad_spin.setRange(0, 2000); self.speech_pad_spin.setValue(50)
        self.frame_ms_spin = QSpinBox(); self.frame_ms_spin.setRange(10, 1000); self.frame_ms_spin.setValue(30)
        self.force_reload_check = QCheckBox("Force reload Silero model")

        form.addRow("Threshold", self.threshold_spin)
        form.addRow("Min speech duration (ms)", self.min_speech_spin)
        form.addRow("Min silence duration (ms)", self.min_silence_spin)
        form.addRow("Speech pad (ms)", self.speech_pad_spin)
        form.addRow("Diagnostic frame size (ms)", self.frame_ms_spin)
        form.addRow("Advanced", self.force_reload_check)

        run_btn = QPushButton("Run Silero Segmentation")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_segmentation)

        layout.addWidget(group)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        return container

    def _build_features_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        intro = QLabel(
            "This first feature layer computes validated timing/respiratory features from Silero segments. "
            "The remaining registered acoustic features are listed and emitted as explicit NaN placeholders until their implementations are validated against the uploaded notebook."
        )
        intro.setWordWrap(True)
        intro.setObjectName("SubtitleLabel")
        layout.addWidget(intro)

        registry = build_acoustic_feature_registry()
        subsystems = sorted(registry["subsystem"].dropna().unique().tolist())

        self.feature_select_all = QCheckBox("Select all subsystems")
        self.feature_select_all.setChecked(True)
        self.feature_select_all.stateChanged.connect(self._toggle_feature_subsystems)
        layout.addWidget(self.feature_select_all)

        subsystem_group = QGroupBox("Feature subsystems")
        grid = QGridLayout(subsystem_group)
        self.feature_subsystem_checks: dict[str, QCheckBox] = {}
        for i, subsystem in enumerate(subsystems):
            count = int((registry["subsystem"] == subsystem).sum())
            cb = QCheckBox(f"{subsystem} ({count})")
            cb.setChecked(True)
            self.feature_subsystem_checks[subsystem] = cb
            grid.addWidget(cb, i // 2, i % 2)
        layout.addWidget(subsystem_group)

        param_group = QGroupBox("Feature parameters")
        form = QFormLayout(param_group)
        self.min_pause_feature_spin = QDoubleSpinBox()
        self.min_pause_feature_spin.setDecimals(2)
        self.min_pause_feature_spin.setRange(0.0, 5.0)
        self.min_pause_feature_spin.setSingleStep(0.05)
        self.min_pause_feature_spin.setValue(0.15)
        form.addRow("Minimum internal pause duration (s)", self.min_pause_feature_spin)
        layout.addWidget(param_group)

        preview = QPlainTextEdit()
        preview.setReadOnly(True)
        preview.setMaximumHeight(140)
        computed_count = len([f for f in registry["feature"].tolist() if f in {"speech_rate", "total_dur", "speech_dur", "percent_pause", "num_pause", "mean_pause_dur", "mean_phrase_dur", "cv_pause_dur", "cv_phrase_dur", "total_pause_dur"}])
        preview.setPlainText(
            f"Registered acoustic features: {len(registry)}\n"
            f"Implemented in this GUI/backend pass: {computed_count} timing/respiratory features\n"
            "Pending features are intentionally written as NaN with explicit status until formula-level implementation is validated."
        )
        layout.addWidget(preview)

        run_btn = QPushButton("Run Feature Extraction")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_features)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        return container

    def _build_reports_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        group = QGroupBox("Open outputs")
        btns = QGridLayout(group)

        open_ingest = QPushButton("Open Ingest Summary CSV")
        open_ingest.clicked.connect(lambda: self._open_stage_file("ingest", "summary"))
        open_pre_sum = QPushButton("Open Preprocess Summary CSV")
        open_pre_sum.clicked.connect(lambda: self._open_stage_file("preprocess", "summary"))
        open_pre_rep = QPushButton("Open Preprocess HTML Report")
        open_pre_rep.clicked.connect(lambda: self._open_stage_file("preprocess", "report"))
        open_seg_sum = QPushButton("Open Segmentation Summary CSV")
        open_seg_sum.clicked.connect(lambda: self._open_stage_file("segment", "summary"))
        open_seg_rep = QPushButton("Open Segmentation HTML Report")
        open_seg_rep.clicked.connect(lambda: self._open_stage_file("segment", "report"))
        open_feat_sum = QPushButton("Open Feature Table CSV")
        open_feat_sum.clicked.connect(lambda: self._open_stage_file("features", "summary"))
        open_feat_rep = QPushButton("Open Feature HTML Report")
        open_feat_rep.clicked.connect(lambda: self._open_stage_file("features", "report"))
        open_stage_dir = QPushButton("Open Acoustic Output Folder")
        open_stage_dir.clicked.connect(self.open_output_root)

        for i, btn in enumerate([open_ingest, open_pre_sum, open_pre_rep, open_seg_sum, open_seg_rep, open_feat_sum, open_feat_rep, open_stage_dir]):
            btn.setObjectName("OpenButton")
            btns.addWidget(btn, i // 2, i % 2)

        notes = QLabel(
            "Tip: every stage writes tables, reports, logs, plots, and manifests to disk. "
            "Use these buttons to inspect artifacts outside the GUI."
        )
        notes.setWordWrap(True)
        notes.setObjectName("SubtitleLabel")

        layout.addWidget(group)
        layout.addWidget(notes)
        layout.addStretch(1)
        return container

    # ---------------------------- HELPERS ----------------------------
    def _toggle_expert_controls(self) -> None:
        self.expert_group.setVisible(self.expert_mode_check.isChecked())

    def append_log(self, text: str) -> None:
        self.log_box.appendPlainText(text)

    def _require_paths(self) -> tuple[Path, Path] | None:
        input_text = self.input_edit.text().strip()
        output_text = self.output_edit.text().strip()
        if not input_text or not output_text:
            QMessageBox.warning(self, "Missing paths", "Please select both the input audio folder and the output project folder.")
            return None
        return Path(input_text).expanduser(), Path(output_text).expanduser()

    def _preprocess_summary_path(self) -> Path:
        output_root = Path(self.output_edit.text().strip()).expanduser()
        return output_root / "acoustic" / "002_preprocess" / "tables" / "acoustic_preprocess_summary.csv"

    def _segmentation_summary_path(self) -> Path:
        output_root = Path(self.output_edit.text().strip()).expanduser()
        return output_root / "acoustic" / "003_segmentation" / "tables" / "acoustic_segmentation_summary.csv"

    def _toggle_feature_subsystems(self) -> None:
        checked = self.feature_select_all.isChecked()
        for cb in getattr(self, "feature_subsystem_checks", {}).values():
            cb.setChecked(checked)

    def browse_input_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select input audio folder")
        if folder:
            self.input_edit.setText(folder)

    def browse_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select output project folder")
        if folder:
            self.output_edit.setText(folder)

    def _set_busy(self, busy: bool) -> None:
        for btn in self.findChildren(QPushButton):
            btn.setEnabled(not busy)
        self.progress.setRange(0, 0 if busy else 100)
        if not busy:
            self.progress.setValue(100)

    def _run_worker(self, name: str, func: Callable, kwargs: dict) -> None:
        if self._thread is not None:
            QMessageBox.information(self, "Busy", "Another stage is still running. Please wait until it finishes.")
            return
        self._thread = QThread()
        self._worker = Worker(name=name, func=func, kwargs=kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.started.connect(self._on_worker_started)
        self._worker.message.connect(self.append_log)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._cleanup_thread)
        self._worker.failed.connect(self._cleanup_thread)
        self._thread.start()

    def _cleanup_thread(self, *_args) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self._set_busy(False)

    def _on_worker_started(self, name: str) -> None:
        self._set_busy(True)
        self.append_log(f"=== {name} ===")

    def _on_worker_finished(self, name: str, result: object) -> None:
        status = getattr(result, "status", "completed")
        manifest = getattr(result, "manifest_path", None)
        summary = getattr(result, "summary_table", None)
        report = getattr(result, "report_path", None)
        errors = getattr(result, "error_table", None)
        rec = self.stage_records.get(name, StageRecord())
        rec.status = str(status)
        rec.manifest_path = str(manifest) if manifest else None
        rec.summary_path = str(summary) if summary else None
        rec.report_path = str(report) if report else None
        rec.errors_path = str(errors) if errors else None
        self.stage_records[name] = rec
        self._refresh_stage_cards()
        self.append_log(f"{name} completed with status: {status}")
        if report:
            self.append_log(f"Report: {report}")
        if summary:
            self.append_log(f"Summary: {summary}")

    def _on_worker_failed(self, name: str, err: str) -> None:
        rec = self.stage_records.get(name, StageRecord())
        rec.status = "failed"
        self.stage_records[name] = rec
        self._refresh_stage_cards()
        self.append_log(f"ERROR in {name}:\n{err}")
        QMessageBox.critical(self, f"{name} failed", err)

    def _refresh_stage_cards(self) -> None:
        for key, lbl in self.stage_labels.items():
            lbl.setText(self.stage_records[key].status)

    def _open_stage_file(self, stage_key: str, kind: str) -> None:
        rec = self.stage_records.get(stage_key)
        if rec is None:
            return
        target = None
        if kind == "summary":
            target = rec.summary_path
        elif kind == "report":
            target = rec.report_path
        if not target or not Path(target).exists():
            QMessageBox.information(self, "Not available", f"No {kind} file is available yet for stage: {stage_key}.")
            return
        if kind == "report":
            open_in_browser(target)
        else:
            open_path(target)

    def open_output_root(self) -> None:
        out = self.output_edit.text().strip()
        if not out:
            QMessageBox.information(self, "No output folder", "Please choose an output project folder first.")
            return
        p = Path(out).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        open_path(p)

    # ---------------------------- ACTIONS ----------------------------
    def run_project_init(self) -> None:
        paths = self._require_paths()
        if paths is None:
            return
        _input_path, output_root = paths
        project_name = self.project_name_edit.text().strip() or "VSLP Acoustic Project"
        self._run_worker("project", initialize_project, {"output_root": output_root, "project_name": project_name})

    def run_ingest(self) -> None:
        paths = self._require_paths()
        if paths is None:
            return
        input_path, output_root = paths
        initialize_project(output_root=output_root, project_name=self.project_name_edit.text().strip() or "VSLP Acoustic Project")
        self._run_worker("ingest", run_acoustic_ingest, {"input_path": input_path, "output_root": output_root})

    def run_preprocess(self) -> None:
        paths = self._require_paths()
        if paths is None:
            return
        input_path, output_root = paths
        initialize_project(output_root=output_root, project_name=self.project_name_edit.text().strip() or "VSLP Acoustic Project")
        feature_sr_text = self.feature_sr_edit.text().strip()
        feature_sr = int(feature_sr_text) if feature_sr_text else None
        filter_kind = self.filter_kind_combo.currentText()
        notch_text = self.notch_hz_combo.currentText().strip()
        filter_cfg = FilterConfig(
            enabled=(filter_kind != "none"),
            kind=filter_kind,
            low_hz=float(self.low_hz_spin.value()) if filter_kind in {"hpf", "bpf"} else None,
            high_hz=float(self.high_hz_spin.value()) if filter_kind in {"lpf", "bpf"} else None,
            notch_hz=float(notch_text) if filter_kind == "notch" and notch_text else None,
        )
        cfg = PreprocessConfig(
            segmentation_sample_rate_hz=int(self.seg_sr_spin.value()),
            feature_sample_rate_hz=feature_sr,
            filter=filter_cfg,
        )
        self._run_worker("preprocess", run_acoustic_preprocess, {"input_path": input_path, "output_root": output_root, "config": cfg})

    def run_segmentation(self) -> None:
        paths = self._require_paths()
        if paths is None:
            return
        _input_path, output_root = paths
        preprocess_summary = self._preprocess_summary_path()
        if not preprocess_summary.exists():
            QMessageBox.warning(self, "Preprocess required", "Run preprocessing first. The preprocess summary CSV was not found.")
            return
        self._run_worker(
            "segment",
            run_acoustic_segmentation_silero,
            {
                "preprocess_summary_csv": preprocess_summary,
                "output_root": output_root,
                "threshold": float(self.threshold_spin.value()),
                "min_speech_duration_ms": int(self.min_speech_spin.value()),
                "min_silence_duration_ms": int(self.min_silence_spin.value()),
                "speech_pad_ms": int(self.speech_pad_spin.value()),
                "frame_ms": int(self.frame_ms_spin.value()),
                "force_reload": bool(self.force_reload_check.isChecked()),
            },
        )

    def run_features(self) -> None:
        paths = self._require_paths()
        if paths is None:
            return
        _input_path, output_root = paths
        segmentation_summary = self._segmentation_summary_path()
        if not segmentation_summary.exists():
            QMessageBox.warning(self, "Segmentation required", "Run Silero segmentation first. The segmentation summary CSV was not found.")
            return
        selected_subsystems = [
            subsystem for subsystem, cb in getattr(self, "feature_subsystem_checks", {}).items() if cb.isChecked()
        ]
        cfg = FeatureExtractionConfig(
            selected_subsystems=selected_subsystems,
            minimum_pause_duration_sec=float(self.min_pause_feature_spin.value()),
        )
        self._run_worker(
            "features",
            run_acoustic_feature_extraction,
            {
                "segmentation_summary_csv": segmentation_summary,
                "output_root": output_root,
                "config": cfg,
            },
        )

    def run_all(self) -> None:
        paths = self._require_paths()
        if paths is None:
            return
        input_path, output_root = paths
        initialize_project(output_root=output_root, project_name=self.project_name_edit.text().strip() or "VSLP Acoustic Project")

        # Run a simple chained workflow to keep the first GUI dependable.
        def full_run(input_path: Path, output_root: Path):
            ingest_result = run_acoustic_ingest(input_path=input_path, output_root=output_root)
            preprocess_cfg = PreprocessConfig(
                segmentation_sample_rate_hz=int(self.seg_sr_spin.value()),
                feature_sample_rate_hz=int(self.feature_sr_edit.text().strip()) if self.feature_sr_edit.text().strip() else None,
                filter=FilterConfig(enabled=False, kind="none"),
            )
            preprocess_result = run_acoustic_preprocess(input_path=input_path, output_root=output_root, config=preprocess_cfg)
            segment_result = run_acoustic_segmentation_silero(
                preprocess_summary_csv=output_root / "acoustic" / "002_preprocess" / "tables" / "acoustic_preprocess_summary.csv",
                output_root=output_root,
                threshold=float(self.threshold_spin.value()),
                min_speech_duration_ms=int(self.min_speech_spin.value()),
                min_silence_duration_ms=int(self.min_silence_spin.value()),
                speech_pad_ms=int(self.speech_pad_spin.value()),
                frame_ms=int(self.frame_ms_spin.value()),
                force_reload=bool(self.force_reload_check.isChecked()),
            )
            features_result = run_acoustic_feature_extraction(
                segmentation_summary_csv=output_root / "acoustic" / "003_segmentation" / "tables" / "acoustic_segmentation_summary.csv",
                output_root=output_root,
                config=FeatureExtractionConfig(minimum_pause_duration_sec=float(self.min_pause_feature_spin.value())),
            )
            # Update stage records manually inside the final result pack.
            return {
                "ingest": ingest_result,
                "preprocess": preprocess_result,
                "segment": segment_result,
                "features": features_result,
            }

        self._run_worker("full_run", full_run, {"input_path": input_path, "output_root": output_root})

    def _on_worker_finished(self, name: str, result: object) -> None:  # type: ignore[override]
        if name == "full_run" and isinstance(result, dict):
            for stage_name in ["ingest", "preprocess", "segment", "features"]:
                self._on_worker_finished(stage_name, result[stage_name])
            self.append_log("Full acoustic backend run finished.")
            return
        status = getattr(result, "status", "completed")
        manifest = getattr(result, "manifest_path", None)
        summary = getattr(result, "summary_table", None)
        report = getattr(result, "report_path", None)
        errors = getattr(result, "error_table", None)
        rec = self.stage_records.get(name, StageRecord())
        rec.status = str(status)
        rec.manifest_path = str(manifest) if manifest else None
        rec.summary_path = str(summary) if summary else None
        rec.report_path = str(report) if report else None
        rec.errors_path = str(errors) if errors else None
        self.stage_records[name] = rec
        self._refresh_stage_cards()
        self.append_log(f"{name} completed with status: {status}")
        if report:
            self.append_log(f"Report: {report}")
        if summary:
            self.append_log(f"Summary: {summary}")
