"""VSLP Acoustic Pipeline GUI v0.20.

V0.17 Setup/Ingest refinement:
- stage-aware workflow guidance with scientific rationale;
- embedded CSV and plot previews;
- latest-output detection;
- feature-level selection inside each subsystem;
- quick selectors for all / implemented / proxy / pending features;
- region-aware feature extraction;
- aggregation and QC dashboard stages;
- clearer separation of clinical/simple controls and expert controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import traceback

import pandas as pd
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vslp.acoustic.aggregate.stage import AggregationConfig, run_acoustic_aggregation
from vslp.acoustic.features.registry import build_acoustic_feature_registry
from vslp.acoustic.features.stage import (
    IMPLEMENTED_FEATURES,
    PROXY_FEATURES,
    FeatureExtractionConfig,
    run_acoustic_feature_extraction,
)
from vslp.acoustic.ingest.stage import run_acoustic_ingest
from vslp.acoustic.metadata.stage import MetadataConfig, run_acoustic_metadata
from vslp.acoustic.preprocess.stage import FilterConfig, PreprocessConfig, run_acoustic_preprocess
from vslp.acoustic.segment.stage import run_acoustic_segmentation_silero
from vslp.acoustic.qc.stage import AcousticQCConfig, run_acoustic_qc_dashboard
from vslp.acoustic.quality.stage import QC_FAMILIES, QualityControlConfig, run_acoustic_quality_control
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
        self.resize(1480, 900)
        self.setMinimumSize(1180, 760)
        self._current_preview_pixmap: QPixmap | None = None
        self._current_preview_path: Path | None = None

        self.registry = build_acoustic_feature_registry()
        self._updating_feature_tree = False
        self.stage_records: dict[str, StageRecord] = {
            "project": StageRecord(),
            "metadata": StageRecord(),
            "ingest": StageRecord(),
            "preprocess": StageRecord(),
            "segment": StageRecord(),
            "quality": StageRecord(),
            "features": StageRecord(),
            "aggregate": StageRecord(),
            "qc": StageRecord(),
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
        sidebar.setFixedWidth(260)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(7)

        title = QLabel("VSLP")
        title.setObjectName("AppTitleLabel")
        subtitle = QLabel("Acoustic Pipeline GUI v0.21")
        subtitle.setObjectName("SubtitleLabel")
        ip_notice = QLabel(
            "© 2026 Nevena Musikic & Yana Yunusova\n"
            "Speech Production Lab, University of Toronto"
        )
        ip_notice.setObjectName("IPNoticeLabel")
        ip_notice.setWordWrap(True)
        side_layout.addWidget(title)
        side_layout.addWidget(subtitle)
        side_layout.addWidget(ip_notice)

        self.stage_labels: dict[str, QLabel] = {}
        self.stage_cards: dict[str, QFrame] = {}
        for key, label in [
            ("project", "Project"),
            ("metadata", "Metadata"),
            ("ingest", "Ingest"),
            ("preprocess", "Preprocess"),
            ("segment", "Data Segmentation"),
            ("quality", "Quality Control"),
            ("features", "Feature Extraction"),
            ("aggregate", "Aggregation"),
            ("qc", "QC Dashboard"),
        ]:
            card = QFrame()
            card.setObjectName("Card")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(9, 7, 9, 7)
            label_widget = QLabel(label)
            label_widget.setStyleSheet("font-weight: 700;")
            status_lbl = QLabel("Not run")
            status_lbl.setObjectName("SubtitleLabel")
            card_layout.addWidget(label_widget)
            card_layout.addWidget(status_lbl)
            card.setMaximumHeight(62)
            self.stage_labels[key] = status_lbl
            self.stage_cards[key] = card
            side_layout.addWidget(card)

        side_layout.addStretch(1)

        self.refresh_outputs_btn = QPushButton("Refresh Latest Outputs")
        self.refresh_outputs_btn.clicked.connect(self.refresh_latest_outputs)
        side_layout.addWidget(self.refresh_outputs_btn)

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

        branding_bar = self._build_branding_bar()
        main_col.addWidget(branding_bar)

        tabs = QTabWidget()
        tabs.setUsesScrollButtons(True)
        tabs.setElideMode(Qt.ElideRight)
        tabs.addTab(self._build_setup_tab(), "Setup")
        tabs.addTab(self._build_metadata_tab(), "Metadata")
        tabs.addTab(self._build_preprocess_tab(), "Preprocess")
        tabs.addTab(self._build_segment_tab(), "Segmentation")
        tabs.addTab(self._build_quality_tab(), "Quality Control")
        tabs.addTab(self._build_features_tab(), "Features")
        tabs.addTab(self._build_aggregation_tab(), "Aggregation")
        tabs.addTab(self._build_qc_tab(), "QC Dashboard")
        tabs.addTab(self._build_inspector_tab(), "Inspector")
        tabs.addTab(self._build_reports_tab(), "Reports & Outputs")
        main_col.addWidget(tabs, stretch=1)

        log_group = QGroupBox("Run Log")
        log_layout = QVBoxLayout(log_group)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(90)
        self.log_box.setMaximumHeight(125)
        log_layout.addWidget(self.progress)
        log_layout.addWidget(self.log_box)
        log_group.setMaximumHeight(180)
        main_col.addWidget(log_group)

        root.addLayout(main_col, stretch=1)


    def _build_branding_bar(self) -> QFrame:
        """Build a compact institutional branding strip without marketing copy.

        Logos are optional. If these files exist, they are displayed:
        - src/vslp/gui/assets/branding/speech_production_lab_logo.png
        - src/vslp/gui/assets/branding/uoft_logo.png
        """
        bar = QFrame()
        bar.setObjectName("BrandingBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        label = QLabel("VSLP Acoustic Pipeline")
        label.setObjectName("BrandingTitle")
        layout.addWidget(label)
        layout.addStretch(1)

        for logo_name, fallback in [
            ("speech_production_lab_logo.png", "Speech Production Lab"),
            ("uoft_logo.png", "University of Toronto"),
        ]:
            logo_widget = self._logo_or_text(logo_name, fallback)
            layout.addWidget(logo_widget)

        bar.setMaximumHeight(58)
        return bar

    def _logo_or_text(self, logo_file: str, fallback: str) -> QLabel:
        assets_dir = Path(__file__).resolve().parents[1] / "assets" / "branding"
        logo_path = assets_dir / logo_file
        widget = QLabel()
        widget.setObjectName("LogoPlaceholder")
        widget.setAlignment(Qt.AlignCenter)
        widget.setMinimumWidth(150)
        widget.setMaximumHeight(42)
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            if not pixmap.isNull():
                widget.setPixmap(pixmap.scaledToHeight(34, Qt.SmoothTransformation))
                widget.setToolTip(str(logo_path))
                return widget
        widget.setText(fallback)
        widget.setToolTip(f"Optional logo file not found: {logo_path}")
        return widget

    # ---------------------------- VISUAL HELPERS ----------------------------
    def _info_panel(self, title: str, body: str) -> QFrame:
        panel = QFrame()
        panel.setObjectName("InfoPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        title_label = QLabel(title)
        title_label.setObjectName("InfoTitle")
        body_label = QLabel(body)
        body_label.setObjectName("InfoBody")
        body_label.setWordWrap(True)
        panel.setMaximumHeight(118)
        layout.addWidget(title_label)
        layout.addWidget(body_label)
        return panel

    def _metric_card(self, label: str, value: str, tooltip: str = "") -> QFrame:
        card = QFrame()
        card.setObjectName("MetricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        value_label = QLabel(value)
        value_label.setObjectName("MetricValue")
        label_widget = QLabel(label)
        label_widget.setObjectName("MetricLabel")
        label_widget.setWordWrap(True)
        layout.addWidget(value_label)
        layout.addWidget(label_widget)
        if tooltip:
            card.setToolTip(tooltip)
        return card

    def _set_tooltip(self, widget: QWidget, text: str) -> None:
        widget.setToolTip(text)

    def _scrollable(self, content: QWidget) -> QWidget:
        """Return a scrollable tab wrapper so every control remains reachable on laptops."""
        wrapper = QWidget()
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return wrapper

    def _stage_status_text(self, status: str) -> str:
        if status in {"completed", "detected"}:
            return f"● {status}"
        if status == "completed_with_warnings":
            return "● completed with warnings"
        if status == "failed":
            return "● failed"
        if status == "running":
            return "● running"
        return f"○ {status}"

    def _build_setup_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        paths_group = QGroupBox("Project setup")
        form = QGridLayout(paths_group)

        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.project_name_edit = QLineEdit("VSLP Acoustic Project")
        self._set_tooltip(self.input_edit, "Folder containing raw audio/video files. Subfolders are searched recursively. Recommendation: process one speech task at a time when possible.")
        self._set_tooltip(self.output_edit, "Root folder where VSLP writes all outputs: tables, plots, reports, logs, errors, artifacts, and manifests.")
        self._set_tooltip(self.project_name_edit, "Human-readable project label stored in project_manifest.json.")

        browse_in = QPushButton("Browse Input Folder")
        browse_in.clicked.connect(self.browse_input_dir)
        browse_out = QPushButton("Browse Output Folder")
        browse_out.clicked.connect(self.browse_output_dir)

        form.addWidget(QLabel("Input audio folder"), 0, 0)
        form.addWidget(self.input_edit, 0, 1)
        form.addWidget(browse_in, 0, 2)
        form.addWidget(QLabel("Output project folder"), 1, 0)
        form.addWidget(self.output_edit, 1, 1)
        form.addWidget(browse_out, 1, 2)
        form.addWidget(QLabel("Project name"), 2, 0)
        form.addWidget(self.project_name_edit, 2, 1, 1, 2)

        workflow_group = QGroupBox("2. Project gate")
        workflow_layout = QVBoxLayout(workflow_group)
        self.project_gate_label = QLabel("")
        self.project_gate_label.setVisible(False)

        btn_row = QHBoxLayout()
        self.init_project_btn = QPushButton("Initialize Project")
        self.init_project_btn.setObjectName("RunButton")
        self.init_project_btn.clicked.connect(self.run_project_init)
        self.ingest_btn = QPushButton("Run Ingest")
        self.ingest_btn.clicked.connect(self.run_ingest)
        self.ingest_btn.setToolTip("Runs ffprobe-based media digest. Requires project initialization first.")
        btn_row.addWidget(self.init_project_btn)
        btn_row.addWidget(self.ingest_btn)
        workflow_layout.addLayout(btn_row)

        ingest_group = QGroupBox("3. Ingest summary")
        ingest_layout = QVBoxLayout(ingest_group)
        self.ingest_summary_label = QLabel(
            "No ingest results yet. After ingest, this panel will show total files loaded, failed files, and the detected format/codec distribution."
        )
        self.ingest_summary_label.setWordWrap(True)
        self.ingest_summary_label.setObjectName("SubtitleLabel")
        ingest_layout.addWidget(self.ingest_summary_label)

        self.ingest_format_table = QTableWidget(0, 3)
        self.ingest_format_table.setHorizontalHeaderLabels(["Detected format", "Audio codec", "Files"])
        self.ingest_format_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ingest_format_table.setMaximumHeight(170)
        self.ingest_format_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ingest_format_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        ingest_layout.addWidget(self.ingest_format_table)

        layout.addWidget(paths_group)
        layout.addWidget(workflow_group)
        layout.addWidget(ingest_group)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_metadata_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        layout.addWidget(self._info_panel(
            "Info",
            "The minimum clinically useful identity is subject_id, session_id, iteration, task, recording_date, diagnosis, severity_score, severity_bin, and file_name. If no CSV is provided, VSLP parses filenames conservatively and leaves clinical labels blank."
        ))

        group = QGroupBox("Demographics / metadata CSV")
        grid = QGridLayout(group)
        self.demographics_csv_edit = QLineEdit()
        self._set_tooltip(self.demographics_csv_edit, "Optional CSV with one row per recording. VSLP detects flexible column names and links only ingested files to matching metadata rows.")
        self.demographics_csv_edit.setPlaceholderText("optional CSV; flexible column names are detected automatically")
        browse_demo = QPushButton("Browse CSV")
        browse_demo.clicked.connect(self.browse_demographics_csv)
        grid.addWidget(QLabel("Metadata CSV"), 0, 0)
        grid.addWidget(self.demographics_csv_edit, 0, 1)
        grid.addWidget(browse_demo, 0, 2)
        layout.addWidget(group)

        required = QPlainTextEdit()
        required.setReadOnly(True)
        required.setMaximumHeight(135)
        required.setPlainText(
            "Metadata CSV can contain more rows than uploaded files. VSLP links only the ingested files.\n"
            "Column names are detected flexibly, for example: Raw Media File name, SubjectID, Protocol ID, Iteration, Task Name, Recording date, Visit date, ALSFRS total score, ALSFRS bulbar subscore.\n\n"
            "Outputs include a project file index, column-mapping table, linkage summary, unmatched ingested files, and unused metadata row sample."
        )
        layout.addWidget(required)

        run_btn = QPushButton("Run Metadata Indexing")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_metadata)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_preprocess_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        group = QGroupBox("Preprocessing")
        form = QFormLayout(group)

        self.remove_dc_check = QCheckBox("Remove DC offset")
        self.remove_dc_check.setChecked(True)
        self._set_tooltip(self.remove_dc_check, "Recommended default. Removes constant waveform offset before segmentation and feature extraction.")

        self.normalize_check = QCheckBox("Peak-normalize waveform")
        self.normalize_check.setChecked(False)
        self._set_tooltip(self.normalize_check, "Off by default. Normalization can alter amplitude/intensity features; enable only when needed for a documented ML experiment.")
        self.normalize_target_spin = QDoubleSpinBox()
        self.normalize_target_spin.setDecimals(2)
        self.normalize_target_spin.setRange(0.10, 1.00)
        self.normalize_target_spin.setSingleStep(0.05)
        self.normalize_target_spin.setValue(0.95)
        self.normalize_target_spin.setEnabled(False)
        self.normalize_check.stateChanged.connect(lambda *_: self.normalize_target_spin.setEnabled(self.normalize_check.isChecked()))

        self.make_segmentation_wav_check = QCheckBox("Create segmentation WAV")
        self.make_segmentation_wav_check.setChecked(True)
        self._set_tooltip(self.make_segmentation_wav_check, "Keep enabled when using Silero segmentation. Default sample rate is 16 kHz.")
        self.seg_sr_spin = QSpinBox()
        self.seg_sr_spin.setRange(8000, 48000)
        self.seg_sr_spin.setValue(16000)

        self.make_feature_wav_check = QCheckBox("Create feature WAV")
        self.make_feature_wav_check.setChecked(True)
        self._set_tooltip(self.make_feature_wav_check, "Feature WAVs are used by downstream acoustic feature extraction.")
        self.feature_resample_check = QCheckBox("Resample feature WAV")
        self.feature_resample_check.setChecked(False)
        self._set_tooltip(self.feature_resample_check, "Off by default. Leave feature WAV at original sample rate unless a study-specific reason requires resampling.")
        self.feature_sr_spin = QSpinBox()
        self.feature_sr_spin.setRange(8000, 96000)
        self.feature_sr_spin.setValue(16000)
        self.feature_sr_spin.setEnabled(False)
        self.feature_resample_check.stateChanged.connect(lambda *_: self.feature_sr_spin.setEnabled(self.feature_resample_check.isChecked()))

        form.addRow("DC offset", self.remove_dc_check)
        form.addRow("Normalize", self.normalize_check)
        form.addRow("Target peak", self.normalize_target_spin)
        form.addRow("Segmentation WAV", self.make_segmentation_wav_check)
        form.addRow("Segmentation sample rate", self.seg_sr_spin)
        form.addRow("Feature WAV", self.make_feature_wav_check)
        form.addRow("Feature resampling", self.feature_resample_check)
        form.addRow("Feature sample rate", self.feature_sr_spin)
        layout.addWidget(group)

        self.expert_mode_check = QCheckBox("Show filters")
        self.expert_mode_check.stateChanged.connect(self._toggle_expert_controls)
        layout.addWidget(self.expert_mode_check)

        self.expert_group = QGroupBox("Optional filters")
        expert_form = QFormLayout(self.expert_group)
        self.filter_preset_combo = QComboBox()
        self.filter_preset_combo.addItems([
            "none",
            "high-pass Butterworth",
            "low-pass Butterworth",
            "band-pass Butterworth",
            "notch 50 Hz",
            "notch 60 Hz",
        ])
        self._set_tooltip(self.filter_preset_combo, "Filtering is off by default. Use only when justified by recording conditions or analysis plan.")
        self.low_hz_spin = QDoubleSpinBox(); self.low_hz_spin.setRange(0, 50000); self.low_hz_spin.setValue(50.0)
        self.high_hz_spin = QDoubleSpinBox(); self.high_hz_spin.setRange(0, 50000); self.high_hz_spin.setValue(8000.0)
        self.filter_order_spin = QSpinBox(); self.filter_order_spin.setRange(1, 10); self.filter_order_spin.setValue(4)
        expert_form.addRow("Preset", self.filter_preset_combo)
        expert_form.addRow("Low cutoff (Hz)", self.low_hz_spin)
        expert_form.addRow("High cutoff (Hz)", self.high_hz_spin)
        expert_form.addRow("Filter order", self.filter_order_spin)
        self.expert_group.setVisible(False)
        layout.addWidget(self.expert_group)

        summary_group = QGroupBox("Preprocessing summary")
        summary_layout = QVBoxLayout(summary_group)
        self.preprocess_feedback = QPlainTextEdit()
        self.preprocess_feedback.setReadOnly(True)
        self.preprocess_feedback.setMaximumHeight(145)
        self.preprocess_feedback.setPlainText("Run preprocessing to summarize SNR, clipping, DC offset, powerline flags, and generated WAVs.")
        summary_layout.addWidget(self.preprocess_feedback)
        layout.addWidget(summary_group)

        run_btn = QPushButton("Run Preprocess")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_preprocess)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_segment_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "Info",
            "Select the segmentation method for this dataset. Silero is recommended for passages, sentences, and free speech. SPA and energy/RMS segmentation are reserved for task-specific workflows such as DDKs or custom syllable/event detection. Parameters shown below update based on the selected method."
        ))

        method_group = QGroupBox("Segmentation method")
        method_form = QFormLayout(method_group)
        self.segmentation_method_combo = QComboBox()
        self.segmentation_method_combo.addItems([
            "Silero VAD - speech/pause segmentation",
            "SPA - task-specific segmentation (placeholder)",
            "Energy/RMS - amplitude-based segmentation (placeholder)",
            "Custom method - plugin slot (placeholder)",
        ])
        self.segmentation_method_combo.currentTextChanged.connect(self._update_segmentation_method_ui)
        self._set_tooltip(self.segmentation_method_combo, "Choose the segmentation algorithm. Only Silero is fully implemented in this version; other methods are visible placeholders for later plugin implementation.")
        method_form.addRow("Method", self.segmentation_method_combo)
        layout.addWidget(method_group)

        self.silero_group = QGroupBox("Silero parameters")
        form = QFormLayout(self.silero_group)
        self.threshold_spin = QDoubleSpinBox(); self.threshold_spin.setDecimals(2); self.threshold_spin.setRange(0.0, 1.0); self.threshold_spin.setSingleStep(0.05); self.threshold_spin.setValue(0.50)
        self._set_tooltip(self.threshold_spin, "Speech probability threshold. 0.50 is the neutral starting value; higher values are more conservative.")
        self.min_speech_spin = QSpinBox(); self.min_speech_spin.setRange(0, 5000); self.min_speech_spin.setValue(250)
        self._set_tooltip(self.min_speech_spin, "Minimum accepted speech segment duration. Default 250 ms reduces transient false detections.")
        self.min_silence_spin = QSpinBox(); self.min_silence_spin.setRange(0, 5000); self.min_silence_spin.setValue(100)
        self._set_tooltip(self.min_silence_spin, "Minimum nonspeech gap separating speech chunks. Default 100 ms preserves short pauses.")
        self.speech_pad_spin = QSpinBox(); self.speech_pad_spin.setRange(0, 2000); self.speech_pad_spin.setValue(50)
        self._set_tooltip(self.speech_pad_spin, "Padding added around speech regions to reduce boundary truncation.")
        self.frame_ms_spin = QSpinBox(); self.frame_ms_spin.setRange(10, 1000); self.frame_ms_spin.setValue(30)
        self._set_tooltip(self.frame_ms_spin, "Diagnostic frame size for frame-level tables and plots.")
        self.force_reload_check = QCheckBox("Force reload Silero model")
        form.addRow("Threshold", self.threshold_spin)
        form.addRow("Min speech duration (ms)", self.min_speech_spin)
        form.addRow("Min silence duration (ms)", self.min_silence_spin)
        form.addRow("Speech pad (ms)", self.speech_pad_spin)
        form.addRow("Diagnostic frame size (ms)", self.frame_ms_spin)
        form.addRow("Advanced", self.force_reload_check)
        layout.addWidget(self.silero_group)

        self.segmentation_placeholder_group = QGroupBox("Method status")
        placeholder_layout = QVBoxLayout(self.segmentation_placeholder_group)
        self.segmentation_method_status = QLabel("Silero is fully implemented. Other methods are intentionally reserved as plugin slots.")
        self.segmentation_method_status.setWordWrap(True)
        placeholder_layout.addWidget(self.segmentation_method_status)
        self.segmentation_placeholder_group.setVisible(False)
        layout.addWidget(self.segmentation_placeholder_group)

        output_group = QGroupBox("Segmentation output")
        output_layout = QVBoxLayout(output_group)
        self.segmentation_feedback = QPlainTextEdit()
        self.segmentation_feedback.setReadOnly(True)
        self.segmentation_feedback.setMaximumHeight(150)
        self.segmentation_feedback.setPlainText("Run data segmentation to summarize speech/pause regions, segment counts, speech fraction, and output tables.")
        output_layout.addWidget(self.segmentation_feedback)
        layout.addWidget(output_group)

        run_btn = QPushButton("Run Data Segmentation")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_segmentation)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        self._update_segmentation_method_ui()
        return self._scrollable(container)


    def _update_segmentation_method_ui(self) -> None:
        method = self.segmentation_method_combo.currentText() if hasattr(self, "segmentation_method_combo") else "Silero"
        is_silero = method.startswith("Silero")
        if hasattr(self, "silero_group"):
            self.silero_group.setVisible(is_silero)
        if hasattr(self, "segmentation_placeholder_group"):
            self.segmentation_placeholder_group.setVisible(not is_silero)
        if hasattr(self, "segmentation_method_status"):
            if method.startswith("SPA"):
                self.segmentation_method_status.setText("SPA segmentation is reserved for DDK/task-specific segmentation. The plugin slot is present, but the method is not yet implemented in this build.")
            elif method.startswith("Energy"):
                self.segmentation_method_status.setText("Energy/RMS segmentation is reserved as a simple amplitude-based fallback. The GUI slot is present for future implementation and validation.")
            elif method.startswith("Custom"):
                self.segmentation_method_status.setText("Custom segmentation will support lab-specific algorithms through a plugin interface. This method is not yet implemented in this build.")
            else:
                self.segmentation_method_status.setText("Silero is fully implemented and recommended for passages, sentences, and free speech.")

    def _build_quality_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "Info",
            "Quality Control extracts segmentation-informed quality features before acoustic biomarkers are computed. Select one or more artifact families. These measures flag recordings for review; they do not reject data automatically."
        ))

        family_group = QGroupBox("QC feature families")
        family_layout = QVBoxLayout(family_group)
        self.qc_family_checks = {}
        family_descriptions = {
            "additive_interference": "Background noise or hum added to the signal, especially during pauses.",
            "gain_dynamics": "Amplitude instability, level drift, or automatic-gain-control effects during speech.",
            "reverberation_echo": "Speech tails continuing into pauses, suggesting echo or room reverberation.",
            "channel_device": "Device/channel spectral coloration, bandwidth limitation, or microphone response differences.",
            "nonlinear_distortion": "Clipping, saturation, peak flattening, or overload at high levels.",
            "temporal_discontinuity": "Dropouts, silent holes, frozen windows, or abrupt waveform breaks.",
        }
        for family, meta in QC_FAMILIES.items():
            cb = QCheckBox(meta.get("label", family.replace("_", " ")))
            cb.setChecked(True)
            cb.setToolTip(family_descriptions.get(family, ""))
            self.qc_family_checks[family] = cb
            desc = QLabel(family_descriptions.get(family, ""))
            desc.setObjectName("SubtitleLabel")
            desc.setWordWrap(True)
            row = QFrame(); row.setObjectName("Card")
            row_layout = QVBoxLayout(row); row_layout.setContentsMargins(10, 8, 10, 8)
            row_layout.addWidget(cb); row_layout.addWidget(desc)
            family_layout.addWidget(row)

        param_group = QGroupBox("QC parameters")
        form = QFormLayout(param_group)
        self.qc_pause_sec_spin = QDoubleSpinBox(); self.qc_pause_sec_spin.setDecimals(2); self.qc_pause_sec_spin.setRange(0.0, 5.0); self.qc_pause_sec_spin.setSingleStep(0.05); self.qc_pause_sec_spin.setValue(0.15)
        self.qc_high_level_spin = QDoubleSpinBox(); self.qc_high_level_spin.setDecimals(1); self.qc_high_level_spin.setRange(50.0, 99.9); self.qc_high_level_spin.setSingleStep(1.0); self.qc_high_level_spin.setValue(90.0)
        self.qc_hard_clip_spin = QDoubleSpinBox(); self.qc_hard_clip_spin.setDecimals(3); self.qc_hard_clip_spin.setRange(0.5, 1.0); self.qc_hard_clip_spin.setSingleStep(0.005); self.qc_hard_clip_spin.setValue(0.995)
        self.qc_near_clip_spin = QDoubleSpinBox(); self.qc_near_clip_spin.setDecimals(3); self.qc_near_clip_spin.setRange(0.5, 1.0); self.qc_near_clip_spin.setSingleStep(0.005); self.qc_near_clip_spin.setValue(0.950)
        form.addRow("Minimum internal pause (s)", self.qc_pause_sec_spin)
        form.addRow("High-level speech percentile", self.qc_high_level_spin)
        form.addRow("Hard clipping threshold", self.qc_hard_clip_spin)
        form.addRow("Near-clipping threshold", self.qc_near_clip_spin)

        self.quality_summary_label = QLabel("Quality Control has not been run.")
        self.quality_summary_label.setObjectName("SubtitleLabel")
        self.quality_summary_label.setWordWrap(True)
        run_btn = QPushButton("Run Quality Control")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_quality_control)
        layout.addWidget(family_group)
        layout.addWidget(param_group)
        layout.addWidget(run_btn)
        layout.addWidget(self.quality_summary_label)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_features_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "Region-aware feature extraction",
            "Many acoustic features should not be computed over the full file. Timing and pause features use Silero speech/nonspeech segments. Signal features default to speech_only regions so leading silence, trailing silence, instructions, and dead time do not contaminate the measurement. Pending features remain explicit NaN placeholders until validated."
        ))

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget(); left_layout = QVBoxLayout(left)
        selector_group = QGroupBox("Feature selector")
        selector_layout = QVBoxLayout(selector_group)
        quick = QHBoxLayout()
        for label, callback in [
            ("All", self.select_all_features),
            ("Implemented", self.select_implemented_features),
            ("Implemented + Proxy", self.select_implemented_and_proxy_features),
            ("Clear", self.clear_feature_selection),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(callback)
            quick.addWidget(btn)
        selector_layout.addLayout(quick)

        self.feature_tree = QTreeWidget()
        self.feature_tree.setHeaderLabels(["Feature / subsystem", "Status", "Unit"])
        self.feature_tree.setMinimumHeight(360)
        self.feature_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.feature_tree.itemChanged.connect(self._on_feature_tree_item_changed)
        self.feature_tree.itemSelectionChanged.connect(self._update_feature_detail)
        self.feature_items: dict[str, QTreeWidgetItem] = {}
        self.subsystem_items: dict[str, QTreeWidgetItem] = {}
        self._populate_feature_tree()
        selector_layout.addWidget(self.feature_tree)
        left_layout.addWidget(selector_group)

        right = QWidget(); right_layout = QVBoxLayout(right)
        self.feature_count_label = QLabel("")
        self.feature_count_label.setObjectName("SectionHeader")
        right_layout.addWidget(self.feature_count_label)
        self.feature_detail_box = QPlainTextEdit()
        self.feature_detail_box.setReadOnly(True)
        self.feature_detail_box.setMinimumHeight(150)
        self.feature_detail_box.setMaximumHeight(210)
        right_layout.addWidget(self.feature_detail_box)
        param_group = QGroupBox("Feature parameters")
        form = QFormLayout(param_group)
        self.min_pause_feature_spin = QDoubleSpinBox(); self.min_pause_feature_spin.setDecimals(2); self.min_pause_feature_spin.setRange(0.0, 5.0); self.min_pause_feature_spin.setSingleStep(0.05); self.min_pause_feature_spin.setValue(0.15)
        self._set_tooltip(self.min_pause_feature_spin, "Minimum internal nonspeech duration counted as a pause. 150 ms is a conservative starting point for speech pause analysis.")
        self.region_policy_combo = QComboBox(); self.region_policy_combo.addItems(["speech_only", "effective_task", "full_file"])
        self._set_tooltip(self.region_policy_combo, "Default speech_only computes signal features from detected speech regions. effective_task includes internal pauses. full_file is exploratory only.")
        form.addRow("Minimum internal pause duration (s)", self.min_pause_feature_spin)
        form.addRow("Signal-feature analysis region", self.region_policy_combo)
        region_note = QLabel("Recommended default: speech_only. Timing features always use speech/pause segments. Use full_file only for explicit full-recording exploratory features.")
        region_note.setWordWrap(True); region_note.setObjectName("SubtitleLabel")
        form.addRow("Region guidance", region_note)
        right_layout.addWidget(param_group)
        run_btn = QPushButton("Run Feature Extraction")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_features)
        right_layout.addWidget(run_btn)
        right_layout.addStretch(1)
        splitter.addWidget(left); splitter.addWidget(right); splitter.setSizes([720, 430]); splitter.setMinimumHeight(540)
        layout.addWidget(splitter)
        self._refresh_feature_count_label()
        return self._scrollable(container)


    def _build_aggregation_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "Aggregation rationale",
            "Aggregation converts per-file feature outputs into analysis-ready tables while preserving subject/session/iteration/task structure. The default grouping is designed for longitudinal clinical monitoring and ML leakage control."
        ))
        group = QGroupBox("Aggregation configuration")
        form = QFormLayout(group)
        self.agg_group_columns_edit = QLineEdit("subject_id,session_id,iteration,task")
        self.agg_numeric_policy_combo = QComboBox(); self.agg_numeric_policy_combo.addItems(["mean", "median"])
        self.agg_missing_policy_combo = QComboBox(); self.agg_missing_policy_combo.addItems(["preserve", "drop_features_over_threshold", "impute_group_median"])
        self.agg_missing_threshold_spin = QDoubleSpinBox(); self.agg_missing_threshold_spin.setDecimals(2); self.agg_missing_threshold_spin.setRange(0.0, 1.0); self.agg_missing_threshold_spin.setSingleStep(0.05); self.agg_missing_threshold_spin.setValue(0.40)
        form.addRow("Group columns", self.agg_group_columns_edit)
        form.addRow("Numeric aggregation", self.agg_numeric_policy_combo)
        form.addRow("Missing-value policy", self.agg_missing_policy_combo)
        form.addRow("Max missing fraction", self.agg_missing_threshold_spin)
        guidance = QLabel("Aggregation creates analysis-ready tables for the later Feature Analysis and ML GUIs. Default grouping is subject × session × iteration × task.")
        guidance.setWordWrap(True); guidance.setObjectName("SubtitleLabel")
        run_btn = QPushButton("Run Aggregation")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_aggregation)
        layout.addWidget(group)
        layout.addWidget(guidance)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_qc_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "QC rationale",
            "QC flags recordings for review; it does not reject data automatically. Thresholds are intentionally visible and editable because acceptable quality depends on task, device, patient speech impairment, and study design."
        ))
        group = QGroupBox("QC dashboard thresholds")
        form = QFormLayout(group)
        self.qc_min_snr_spin = QDoubleSpinBox(); self.qc_min_snr_spin.setDecimals(1); self.qc_min_snr_spin.setRange(-50.0, 80.0); self.qc_min_snr_spin.setValue(10.0)
        self._set_tooltip(self.qc_min_snr_spin, "Research-screening SNR proxy threshold. 10 dB is a conservative initial flag, not a hard exclusion rule.")
        self.qc_clip_spin = QDoubleSpinBox(); self.qc_clip_spin.setDecimals(4); self.qc_clip_spin.setRange(0.0, 1.0); self.qc_clip_spin.setSingleStep(0.0005); self.qc_clip_spin.setValue(0.001)
        self._set_tooltip(self.qc_clip_spin, "Fraction of samples allowed near full scale before flagging clipping. Default 0.001 = 0.1%.")
        self.qc_min_speech_spin = QDoubleSpinBox(); self.qc_min_speech_spin.setDecimals(2); self.qc_min_speech_spin.setRange(0.0, 1.0); self.qc_min_speech_spin.setValue(0.05)
        self._set_tooltip(self.qc_min_speech_spin, "Very low speech fraction may indicate wrong file, failed recording, or VAD failure.")
        self.qc_max_speech_spin = QDoubleSpinBox(); self.qc_max_speech_spin.setDecimals(2); self.qc_max_speech_spin.setRange(0.0, 1.0); self.qc_max_speech_spin.setValue(0.98)
        self._set_tooltip(self.qc_max_speech_spin, "Very high speech fraction may indicate missing pause detection or excessive background/sustained activity.")
        form.addRow("Minimum estimated SNR (dB)", self.qc_min_snr_spin)
        form.addRow("Maximum clipping fraction", self.qc_clip_spin)
        form.addRow("Minimum speech fraction", self.qc_min_speech_spin)
        form.addRow("Maximum speech fraction", self.qc_max_speech_spin)
        guidance = QLabel("QC flags files for review using preprocessing, segmentation, and feature-completeness outputs. These are research-screening thresholds, not clinical acceptance criteria.")
        guidance.setWordWrap(True); guidance.setObjectName("SubtitleLabel")
        run_btn = QPushButton("Run QC Dashboard")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_qc_dashboard)
        layout.addWidget(group)
        layout.addWidget(guidance)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_inspector_tab(self) -> QWidget:
        """Build an inspector with left-side controls and right-side previews.

        Product design rule for v0.15:
        - controls live in a compact left rail;
        - previews live in a large right canvas;
        - plot preview receives a near-square high-contrast viewport;
        - tables retain the same professional outline and are not mixed with plot controls.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.addWidget(self._info_panel(
            "Visual inspection layer",
            "Preview generated tables and plots without leaving the GUI. Use the left panel to choose an artifact; the right panel provides a large, audit-friendly preview. Full files remain on disk for reproducibility and publication-quality inspection."
        ))

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        controls_panel = QFrame()
        controls_panel.setObjectName("Card")
        controls_panel.setMinimumWidth(260)
        controls_panel.setMaximumWidth(360)
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(10)

        table_group = QGroupBox("Table previews")
        table_layout = QVBoxLayout(table_group)
        table_layout.setSpacing(7)
        table_buttons = [
            ("Metadata file index", lambda: self.preview_csv(self._metadata_index_path())),
            ("Metadata column mapping", lambda: self.preview_csv(self._stage_path("metadata", "column_mapping"))),
            ("Metadata linkage", lambda: self.preview_csv(self._stage_path("metadata", "linkage"))),
            ("Metadata unmatched files", lambda: self.preview_csv(self._stage_path("metadata", "unmatched"))),
            ("Ingest summary", lambda: self.preview_csv(self._stage_path("ingest", "summary"))),
            ("Preprocess main summary", lambda: self.preview_csv(self._stage_path("preprocess", "main_summary"))),
            ("Preprocess detailed summary", lambda: self.preview_csv(self._stage_path("preprocess", "summary"))),
            ("Preprocess QC flags", lambda: self.preview_csv(self._stage_path("preprocess", "qc_flags"))),
            ("Segmentation summary", lambda: self.preview_csv(self._stage_path("segment", "summary"))),
            ("Quality features", lambda: self.preview_csv(self._stage_path("quality", "summary"))),
            ("Quality family status", lambda: self.preview_csv(self._stage_path("quality", "family_status"))),
            ("Feature table", lambda: self.preview_csv(self._stage_path("features", "summary"))),
            ("Aggregated table", lambda: self.preview_csv(self._stage_path("aggregate", "summary"))),
            ("QC dashboard", lambda: self.preview_csv(self._stage_path("qc", "summary"))),
        ]
        for label, callback in table_buttons:
            btn = QPushButton(label)
            btn.setObjectName("OpenButton")
            btn.clicked.connect(callback)
            table_layout.addWidget(btn)

        plot_group = QGroupBox("Plot previews")
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setSpacing(7)
        plot_buttons = [
            ("Latest segmentation plot", self.preview_latest_segmentation_plot),
            ("Preprocess SNR", lambda: self.preview_image(self._output_root() / "acoustic" / "002_preprocess" / "plots" / "preprocess_snr_distribution.png")),
            ("Preprocess clipping", lambda: self.preview_image(self._output_root() / "acoustic" / "002_preprocess" / "plots" / "preprocess_clipping_distribution.png")),
            ("Preprocess DC offset", lambda: self.preview_image(self._output_root() / "acoustic" / "002_preprocess" / "plots" / "preprocess_dc_offset_before_after.png")),
            ("Quality family status", lambda: self.preview_image(self._output_root() / "acoustic" / "003_quality_control" / "plots" / "quality_family_status.png")),
            ("Quality overview", lambda: self.preview_image(self._output_root() / "acoustic" / "003_quality_control" / "plots" / "quality_main_features_overview.png")),
            ("Feature missingness", lambda: self.preview_image(self._features_plot_path("feature_missingness.png"))),
            ("Feature implementation status", lambda: self.preview_image(self._features_plot_path("feature_subsystem_implementation_status.png"))),
            ("Feature distributions", lambda: self.preview_image(self._features_plot_path("implemented_feature_distributions.png"))),
            ("Aggregation missingness", lambda: self.preview_image(self._output_root() / "acoustic" / "005_aggregation" / "plots" / "aggregation_missingness_top30.png")),
            ("Aggregation group counts", lambda: self.preview_image(self._output_root() / "acoustic" / "005_aggregation" / "plots" / "aggregation_group_counts.png")),
            ("QC flag counts", lambda: self.preview_image(self._output_root() / "acoustic" / "006_qc_dashboard" / "plots" / "qc_flag_counts.png")),
            ("QC SNR distribution", lambda: self.preview_image(self._output_root() / "acoustic" / "006_qc_dashboard" / "plots" / "qc_snr_distribution.png")),
            ("QC speech fraction", lambda: self.preview_image(self._output_root() / "acoustic" / "006_qc_dashboard" / "plots" / "qc_speech_fraction_distribution.png")),
        ]
        for label, callback in plot_buttons:
            btn = QPushButton(label)
            btn.setObjectName("OpenButton")
            btn.clicked.connect(callback)
            plot_layout.addWidget(btn)

        utility_group = QGroupBox("Utilities")
        utility_layout = QVBoxLayout(utility_group)
        utility_layout.setSpacing(7)
        refresh_btn = QPushButton("Refresh latest outputs")
        refresh_btn.clicked.connect(self.refresh_latest_outputs)
        open_current_btn = QPushButton("Open current preview file")
        open_current_btn.setObjectName("OpenButton")
        open_current_btn.clicked.connect(self.open_current_preview_file)
        open_output_btn = QPushButton("Open output project folder")
        open_output_btn.setObjectName("OpenButton")
        open_output_btn.clicked.connect(self.open_output_root)
        utility_layout.addWidget(refresh_btn)
        utility_layout.addWidget(open_current_btn)
        utility_layout.addWidget(open_output_btn)

        controls_layout.addWidget(table_group)
        controls_layout.addWidget(plot_group)
        controls_layout.addWidget(utility_group)
        controls_layout.addStretch(1)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_panel)
        controls_scroll.setMinimumWidth(290)
        controls_scroll.setMaximumWidth(390)

        preview_panel = QFrame()
        preview_panel.setObjectName("Card")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(10)

        self.inspector_status = QLabel("Select a table or plot from the left panel.")
        self.inspector_status.setObjectName("SubtitleLabel")
        self.inspector_status.setWordWrap(True)
        preview_layout.addWidget(self.inspector_status)

        self.preview_tabs = QTabWidget()
        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.preview_table.verticalHeader().setVisible(False)

        plot_canvas = QFrame()
        plot_canvas.setObjectName("PlotCanvas")
        plot_canvas_layout = QVBoxLayout(plot_canvas)
        plot_canvas_layout.setContentsMargins(12, 12, 12, 12)
        self.preview_image_label = QLabel("No plot selected")
        self.preview_image_label.setObjectName("PlotPreviewLabel")
        self.preview_image_label.setAlignment(Qt.AlignCenter)
        self.preview_image_label.setMinimumSize(560, 560)
        self.preview_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_image_label.setScaledContents(False)
        plot_canvas_layout.addWidget(self.preview_image_label, stretch=1)

        self.preview_tabs.addTab(self.preview_table, "Table Preview")
        self.preview_tabs.addTab(plot_canvas, "Plot Preview")
        preview_layout.addWidget(self.preview_tabs, stretch=1)

        splitter.addWidget(controls_scroll)
        splitter.addWidget(preview_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 980])

        layout.addWidget(splitter, stretch=1)
        return container

    def _build_reports_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        group = QGroupBox("Open outputs")
        btns = QGridLayout(group)
        buttons = [
            ("Open Metadata File Index CSV", lambda: self._open_stage_file("metadata", "summary")),
            ("Open Metadata Column Mapping CSV", lambda: self._open_stage_file("metadata", "column_mapping")),
            ("Open Metadata Linkage CSV", lambda: self._open_stage_file("metadata", "linkage")),
            ("Open Metadata HTML Report", lambda: self._open_stage_file("metadata", "report")),
            ("Open Ingest Summary CSV", lambda: self._open_stage_file("ingest", "summary")),
            ("Open Preprocess Detailed CSV", lambda: self._open_stage_file("preprocess", "summary")),
            ("Open Preprocess HTML Report", lambda: self._open_stage_file("preprocess", "report")),
            ("Open Data Segmentation Summary CSV", lambda: self._open_stage_file("segment", "summary")),
            ("Open Data Segmentation HTML Report", lambda: self._open_stage_file("segment", "report")),
            ("Open Quality Features CSV", lambda: self._open_stage_file("quality", "summary")),
            ("Open Quality Control HTML Report", lambda: self._open_stage_file("quality", "report")),
            ("Open Feature Table CSV", lambda: self._open_stage_file("features", "summary")),
            ("Open Feature HTML Report", lambda: self._open_stage_file("features", "report")),
            ("Open Aggregated Feature CSV", lambda: self._open_stage_file("aggregate", "summary")),
            ("Open Aggregation HTML Report", lambda: self._open_stage_file("aggregate", "report")),
            ("Open QC Dashboard CSV", lambda: self._open_stage_file("qc", "summary")),
            ("Open QC Dashboard HTML", lambda: self._open_stage_file("qc", "report")),
            ("Open Acoustic Output Folder", self.open_output_root),
        ]
        for i, (label, callback) in enumerate(buttons):
            btn = QPushButton(label)
            btn.setObjectName("OpenButton")
            btn.clicked.connect(callback)
            btns.addWidget(btn, i // 2, i % 2)
        notes = QLabel(
            "Every stage writes tables, reports, logs, plots, errors, and manifests to disk. The Inspector tab shows quick previews; this tab opens the native files/folders."
        )
        notes.setWordWrap(True); notes.setObjectName("SubtitleLabel")
        layout.addWidget(group)
        layout.addWidget(notes)
        layout.addStretch(1)
        return self._scrollable(container)

    # ---------------------------- FEATURE TREE ----------------------------
    def _feature_status(self, name: str) -> str:
        if name in PROXY_FEATURES:
            return "proxy"
        if name in IMPLEMENTED_FEATURES:
            return "implemented"
        return "pending"

    def _populate_feature_tree(self) -> None:
        self._updating_feature_tree = True
        self.feature_tree.clear()
        for subsystem, sdf in self.registry.groupby("subsystem", sort=True):
            parent = QTreeWidgetItem([f"{subsystem} ({len(sdf)})", "subsystem", ""])
            parent.setFlags(parent.flags() | Qt.ItemIsUserCheckable)
            parent.setCheckState(0, Qt.Checked)
            self.feature_tree.addTopLevelItem(parent)
            self.subsystem_items[str(subsystem)] = parent
            for _, row in sdf.sort_values("feature").iterrows():
                feature = str(row["feature"])
                status = self._feature_status(feature)
                item = QTreeWidgetItem([feature, status, str(row.get("unit", ""))])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked)
                item.setData(0, Qt.UserRole, feature)
                item.setToolTip(0, str(row.get("meaning", "")))
                item.setToolTip(1, "implemented = computed; proxy = engineering estimate needing validation; pending = explicit NaN placeholder")
                if status == "implemented":
                    color = QColor("#8EF2C6")
                elif status == "proxy":
                    color = QColor("#FFD98E")
                else:
                    color = QColor("#8FA9BD")
                item.setForeground(1, QBrush(color))
                item.setForeground(0, QBrush(QColor("#EAF2F8")))
                parent.addChild(item)
                self.feature_items[feature] = item
            parent.setExpanded(True)
        self.feature_tree.resizeColumnToContents(0)
        self._updating_feature_tree = False

    def _on_feature_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating_feature_tree or column != 0:
            return
        self._updating_feature_tree = True
        try:
            if item.childCount() > 0:
                state = item.checkState(0)
                for i in range(item.childCount()):
                    item.child(i).setCheckState(0, state)
            else:
                parent = item.parent()
                if parent is not None:
                    checked = sum(parent.child(i).checkState(0) == Qt.Checked for i in range(parent.childCount()))
                    if checked == parent.childCount():
                        parent.setCheckState(0, Qt.Checked)
                    elif checked == 0:
                        parent.setCheckState(0, Qt.Unchecked)
                    else:
                        parent.setCheckState(0, Qt.PartiallyChecked)
        finally:
            self._updating_feature_tree = False
        self._refresh_feature_count_label()

    def _set_feature_selection(self, predicate: Callable[[str], bool]) -> None:
        self._updating_feature_tree = True
        try:
            for feature, item in self.feature_items.items():
                item.setCheckState(0, Qt.Checked if predicate(feature) else Qt.Unchecked)
            for parent in self.subsystem_items.values():
                checked = sum(parent.child(i).checkState(0) == Qt.Checked for i in range(parent.childCount()))
                if checked == parent.childCount():
                    parent.setCheckState(0, Qt.Checked)
                elif checked == 0:
                    parent.setCheckState(0, Qt.Unchecked)
                else:
                    parent.setCheckState(0, Qt.PartiallyChecked)
        finally:
            self._updating_feature_tree = False
        self._refresh_feature_count_label()

    def select_all_features(self) -> None:
        self._set_feature_selection(lambda _f: True)

    def select_implemented_features(self) -> None:
        self._set_feature_selection(lambda f: f in IMPLEMENTED_FEATURES and f not in PROXY_FEATURES)

    def select_implemented_and_proxy_features(self) -> None:
        self._set_feature_selection(lambda f: f in IMPLEMENTED_FEATURES or f in PROXY_FEATURES)

    def clear_feature_selection(self) -> None:
        self._set_feature_selection(lambda _f: False)

    def _selected_feature_names(self) -> list[str]:
        return [f for f, item in self.feature_items.items() if item.checkState(0) == Qt.Checked]

    def _refresh_feature_count_label(self) -> None:
        if not hasattr(self, "feature_count_label"):
            return
        selected = self._selected_feature_names()
        impl = sum(f in IMPLEMENTED_FEATURES and f not in PROXY_FEATURES for f in selected)
        proxy = sum(f in PROXY_FEATURES for f in selected)
        pending = len(selected) - impl - proxy
        self.feature_count_label.setText(f"Selected: {len(selected)} | implemented: {impl} | proxy: {proxy} | pending placeholders: {pending}")

    def _update_feature_detail(self) -> None:
        items = self.feature_tree.selectedItems()
        if not items or not hasattr(self, "feature_detail_box"):
            return
        item = items[0]
        feature = item.data(0, Qt.UserRole)
        if not feature:
            subsystem_text = item.text(0)
            self.feature_detail_box.setPlainText(f"Subsystem: {subsystem_text}\n\nSelect a child feature to view details.")
            return
        row = self.registry.loc[self.registry["feature"].astype(str) == str(feature)].iloc[0]
        status = self._feature_status(str(feature))
        status_note = {
            "implemented": "Computed by the current backend and ready for engineering review.",
            "proxy": "Computed as a proxy. Use for pipeline testing/exploration; validate before clinical interpretation.",
            "pending": "Registered but not implemented yet. Output remains NaN intentionally."
        }.get(status, "")
        detail = (
            f"Feature: {feature}\n"
            f"Subsystem: {row['subsystem']}\n"
            f"Status: {status} — {status_note}\n"
            f"Unit: {row.get('unit', '')}\n\n"
            f"Scientific meaning:\n{row.get('meaning', '')}\n\n"
            f"Computation note:\n{row.get('computation_note', '')}\n\n"
            "Region policy note:\nTiming and pause features use Silero segment tables. Signal features use the selected analysis region, with speech_only recommended by default."
        )
        self.feature_detail_box.setPlainText(detail)

    # ---------------------------- PATHS/PREVIEWS ----------------------------
    def _output_root(self) -> Path:
        return Path(self.output_edit.text().strip()).expanduser()


    def _project_manifest_path(self) -> Path:
        return self._output_root() / "project_manifest.json"

    def _is_project_initialized(self) -> bool:
        try:
            return self._project_manifest_path().exists()
        except Exception:
            return False

    def _require_project_initialized(self) -> bool:
        if self._is_project_initialized():
            return True
        QMessageBox.warning(
            self,
            "Initialize project first",
            "Please initialize the project before running this stage.\n\n"
            "Go to Setup → select input/output folders → click Initialize Project.",
        )
        return False

    def _refresh_project_gate(self) -> None:
        if not hasattr(self, "project_gate_label"):
            return
        if self._is_project_initialized():
            self.stage_records["project"].status = "detected"
        self._refresh_stage_cards()

    def _update_ingest_feedback(self) -> None:
        if not hasattr(self, "ingest_summary_label"):
            return
        summary_path = self._stage_path("ingest", "summary")
        errors_path = self._output_root() / "acoustic" / "001_ingest" / "errors" / "audio_ingest_errors.csv"
        duplicates_path = self._output_root() / "acoustic" / "001_ingest" / "tables" / "audio_ingest_skipped_duplicates.csv"
        if not summary_path.exists():
            self.ingest_summary_label.setText(
                "No ingest results yet. After ingest, this panel will show total files loaded, failed files, and the detected format/codec distribution."
            )
            self.ingest_format_table.setRowCount(0)
            return
        try:
            df = pd.read_csv(summary_path)
        except Exception as exc:  # noqa: BLE001
            self.ingest_summary_label.setText(f"Ingest summary exists but could not be read: {exc}")
            return
        n_ok = len(df)
        n_failed = 0
        if errors_path.exists():
            try:
                n_failed = len(pd.read_csv(errors_path))
            except Exception:
                n_failed = 0
        duplicate_skipped_count = 0
        if duplicates_path.exists():
            try:
                duplicate_skipped_count = len(pd.read_csv(duplicates_path))
            except Exception:
                duplicate_skipped_count = 0
        subfolder_count = 0
        try:
            input_root = Path(self.input_edit.text().strip()).expanduser().resolve()
            if input_root.exists() and input_root.is_dir() and "file_path" in df.columns:
                parents = {Path(str(p)).expanduser().resolve().parent for p in df["file_path"].dropna()}
                subfolder_count = sum(1 for parent in parents if parent != input_root)
        except Exception:
            subfolder_count = 0
        suffix_counts = df["suffix"].astype(str).value_counts().to_dict() if "suffix" in df.columns else {}
        suffix_txt = ", ".join(f"{k}: {v}" for k, v in suffix_counts.items()) if suffix_counts else "not available"
        self.ingest_summary_label.setText(
            f"Loaded/probed files: {n_ok} | Failed files: {n_failed} | Duplicate files skipped: {duplicate_skipped_count} | Subfolders represented: {subfolder_count}\n"
            f"File extensions found: {suffix_txt}"
        )

        group_cols = [c for c in ["format_name", "audio_codec"] if c in df.columns]
        if group_cols:
            fmt = df.groupby(group_cols, dropna=False).size().reset_index(name="files")
        else:
            fmt = pd.DataFrame({"format_name": ["unknown"], "audio_codec": ["unknown"], "files": [n_ok]})
        self.ingest_format_table.setRowCount(len(fmt))
        self.ingest_format_table.setColumnCount(3)
        self.ingest_format_table.setHorizontalHeaderLabels(["Detected format", "Audio codec", "Files"])
        for r, row in fmt.iterrows():
            self.ingest_format_table.setItem(r, 0, QTableWidgetItem(str(row.get("format_name", ""))))
            self.ingest_format_table.setItem(r, 1, QTableWidgetItem(str(row.get("audio_codec", ""))))
            self.ingest_format_table.setItem(r, 2, QTableWidgetItem(str(row.get("files", ""))))

    def _preprocess_summary_path(self) -> Path:
        return self._output_root() / "acoustic" / "002_preprocess" / "tables" / "acoustic_preprocess_summary.csv"

    def _segmentation_summary_path(self) -> Path:
        return self._output_root() / "acoustic" / "003_segmentation" / "tables" / "acoustic_segmentation_summary.csv"

    def _metadata_index_path(self) -> Path:
        return self._output_root() / "acoustic" / "000_metadata" / "tables" / "project_file_index.csv"

    def _stage_path(self, stage: str, kind: str) -> Path:
        mapping = {
            ("metadata", "summary"): self._metadata_index_path(),
            ("metadata", "report"): self._output_root() / "acoustic" / "000_metadata" / "reports" / "metadata_report.html",
            ("metadata", "column_mapping"): self._output_root() / "acoustic" / "000_metadata" / "tables" / "metadata_column_mapping.csv",
            ("metadata", "linkage"): self._output_root() / "acoustic" / "000_metadata" / "tables" / "metadata_linkage_summary.csv",
            ("metadata", "unmatched"): self._output_root() / "acoustic" / "000_metadata" / "tables" / "metadata_unmatched_ingested_files.csv",
            ("metadata", "unused"): self._output_root() / "acoustic" / "000_metadata" / "tables" / "metadata_unused_rows_sample.csv",
            ("ingest", "summary"): self._output_root() / "acoustic" / "001_ingest" / "tables" / "audio_ingest_summary.csv",
            ("preprocess", "summary"): self._preprocess_summary_path(),
            ("preprocess", "main_summary"): self._output_root() / "acoustic" / "002_preprocess" / "tables" / "acoustic_preprocess_main_summary.csv",
            ("preprocess", "qc_flags"): self._output_root() / "acoustic" / "002_preprocess" / "tables" / "acoustic_preprocess_qc_flags.csv",
            ("preprocess", "report"): self._output_root() / "acoustic" / "002_preprocess" / "reports" / "acoustic_preprocess_report.html",
            ("segment", "summary"): self._segmentation_summary_path(),
            ("segment", "main_summary"): self._output_root() / "acoustic" / "003_segmentation" / "tables" / "acoustic_segmentation_main_summary.csv",
            ("segment", "report"): self._output_root() / "acoustic" / "003_segmentation" / "reports" / "acoustic_segmentation_report.html",
            ("quality", "summary"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_features.csv",
            ("quality", "main_summary"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_main_summary.csv",
            ("quality", "family_status"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_family_status.csv",
            ("quality", "family_summary"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_family_summary.csv",
            ("quality", "report"): self._output_root() / "acoustic" / "003_quality_control" / "reports" / "acoustic_quality_control_report.html",
            ("features", "summary"): self._output_root() / "acoustic" / "004_features" / "tables" / "acoustic_features_per_file.csv",
            ("features", "report"): self._output_root() / "acoustic" / "004_features" / "reports" / "acoustic_feature_report.html",
            ("aggregate", "summary"): self._output_root() / "acoustic" / "005_aggregation" / "tables" / "acoustic_features_aggregated.csv",
            ("aggregate", "report"): self._output_root() / "acoustic" / "005_aggregation" / "reports" / "acoustic_aggregation_report.html",
            ("qc", "summary"): self._output_root() / "acoustic" / "006_qc_dashboard" / "tables" / "acoustic_qc_dashboard.csv",
            ("qc", "report"): self._output_root() / "acoustic" / "006_qc_dashboard" / "reports" / "acoustic_qc_dashboard.html",
        }
        return mapping[(stage, kind)]

    def _features_plot_path(self, name: str) -> Path:
        return self._output_root() / "acoustic" / "004_features" / "plots" / name

    def refresh_latest_outputs(self) -> None:
        if not self.output_edit.text().strip():
            QMessageBox.information(self, "No output folder", "Select an output project folder first.")
            return
        if self._project_manifest_path().exists():
            self.stage_records["project"].status = "detected"
        known = {
            "metadata": (self._stage_path("metadata", "summary"), self._stage_path("metadata", "report")),
            "ingest": (self._stage_path("ingest", "summary"), None),
            "preprocess": (self._stage_path("preprocess", "summary"), self._stage_path("preprocess", "report")),
            "segment": (self._stage_path("segment", "summary"), self._stage_path("segment", "report")),
            "quality": (self._stage_path("quality", "summary"), self._stage_path("quality", "report")),
            "features": (self._stage_path("features", "summary"), self._stage_path("features", "report")),
            "aggregate": (self._stage_path("aggregate", "summary"), self._stage_path("aggregate", "report")),
            "qc": (self._stage_path("qc", "summary"), self._stage_path("qc", "report")),
        }
        for stage, (summary, report) in known.items():
            rec = self.stage_records[stage]
            if summary and summary.exists():
                rec.status = "detected"
                rec.summary_path = str(summary)
            if report and report.exists():
                rec.report_path = str(report)
            self.stage_records[stage] = rec
        self._refresh_stage_cards()
        self._refresh_project_gate()
        self._update_ingest_feedback()
        self.append_log("Refreshed latest output paths from disk.")

    def preview_csv(self, path: Path) -> None:
        if not path.exists():
            QMessageBox.information(self, "Missing output", f"CSV not found:\n{path}")
            return
        try:
            df = pd.read_csv(path, nrows=200)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Preview failed", str(exc))
            return
        self.preview_table.setRowCount(len(df))
        self.preview_table.setColumnCount(len(df.columns))
        self.preview_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                self.preview_table.setItem(r, c, QTableWidgetItem(str(df.iloc[r, c])))
        self._current_preview_path = path
        self.inspector_status.setText(f"Table preview: first {len(df)} rows from {path}")
        self.preview_tabs.setCurrentWidget(self.preview_table)

    def preview_image(self, path: Path) -> None:
        if not path.exists():
            QMessageBox.information(self, "Missing plot", f"Plot not found:\n{path}")
            return
        pix = QPixmap(str(path))
        if pix.isNull():
            QMessageBox.warning(self, "Preview failed", f"Could not load image:\n{path}")
            return
        self._current_preview_pixmap = pix
        self._current_preview_path = path
        self.preview_tabs.setCurrentIndex(1)
        self._update_preview_image_scaled()
        self.inspector_status.setText(f"Plot preview: {path}")

    def _update_preview_image_scaled(self) -> None:
        if not getattr(self, "_current_preview_pixmap", None) or not hasattr(self, "preview_image_label"):
            return
        # Use a near-square canvas for scientific plot inspection.
        # The image is never distorted; it is fitted into the available canvas.
        avail_w = max(420, self.preview_image_label.width() - 20)
        avail_h = max(420, self.preview_image_label.height() - 20)
        square = max(420, min(avail_w, avail_h))
        scaled = self._current_preview_pixmap.scaled(square, square, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._update_preview_image_scaled()


    def open_current_preview_file(self) -> None:
        if not getattr(self, "_current_preview_path", None):
            QMessageBox.information(self, "No preview selected", "Select a table or plot preview first.")
            return
        path = Path(self._current_preview_path)
        if not path.exists():
            QMessageBox.information(self, "Missing file", f"The current preview file no longer exists:\n{path}")
            return
        open_path(path)

    def preview_latest_segmentation_plot(self) -> None:
        plot_dir = self._output_root() / "acoustic" / "003_segmentation" / "plots"
        plots = sorted(plot_dir.glob("*.png")) if plot_dir.exists() else []
        if not plots:
            QMessageBox.information(self, "No plots", f"No segmentation plots found in:\n{plot_dir}")
            return
        self.preview_image(plots[0])

    def _update_preprocess_feedback(self) -> None:
        if not hasattr(self, "preprocess_feedback"):
            return
        main_path = self._stage_path("preprocess", "main_summary")
        flags_path = self._stage_path("preprocess", "qc_flags")
        if not main_path.exists():
            self.preprocess_feedback.setPlainText("Run preprocessing to summarize SNR, clipping, DC offset, powerline flags, and generated WAVs.")
            return
        try:
            df = pd.read_csv(main_path)
            flags = pd.read_csv(flags_path) if flags_path.exists() else pd.DataFrame()
            n = len(df)
            review = int((flags.get("qc_status") == "review").sum()) if not flags.empty and "qc_status" in flags.columns else 0
            snr = pd.to_numeric(df.get("snr_db_estimate"), errors="coerce")
            clip = pd.to_numeric(df.get("clipping_fraction_near_full_scale"), errors="coerce")
            dc_before = pd.to_numeric(df.get("raw_dc_offset"), errors="coerce").abs()
            dc_after = pd.to_numeric(df.get("processed_dc_offset"), errors="coerce").abs()
            f50 = int(df.get("powerline_50hz_flag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if "powerline_50hz_flag" in df else 0
            f60 = int(df.get("powerline_60hz_flag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if "powerline_60hz_flag" in df else 0
            normalized = int(df.get("normalize_peak", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if "normalize_peak" in df else 0
            filtered = int(df.get("filter_enabled", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if "filter_enabled" in df else 0
            self.preprocess_feedback.setPlainText(
                f"Files processed: {n}\n"
                f"Files flagged for review: {review}\n"
                f"Median estimated SNR: {snr.median():.2f} dB\n" if snr.notna().any() else f"Files processed: {n}\nFiles flagged for review: {review}\nMedian estimated SNR: unavailable\n"
            )
            extra = (
                f"Max clipping fraction: {clip.max():.6f}\n" if clip.notna().any() else "Max clipping fraction: unavailable\n"
            )
            extra += (
                f"Median |DC offset| raw → processed: {dc_before.median():.6f} → {dc_after.median():.6f}\n" if dc_before.notna().any() and dc_after.notna().any() else "Median |DC offset| raw → processed: unavailable\n"
            )
            extra += f"Powerline flags: 50 Hz={f50}, 60 Hz={f60}\n"
            extra += f"Normalized files: {normalized}; Filtered files: {filtered}"
            self.preprocess_feedback.appendPlainText(extra)
        except Exception as exc:  # noqa: BLE001
            self.preprocess_feedback.setPlainText(f"Could not summarize preprocessing outputs: {exc}")


    def _read_csv_safe(self, path: Path) -> pd.DataFrame:
        """Read a CSV if it exists and is non-empty; otherwise return an empty DataFrame."""
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _update_segmentation_feedback(self) -> None:
        if not hasattr(self, "segmentation_feedback"):
            return
        main_path = self._stage_path("segment", "main_summary")
        errors_path = self._output_root() / "acoustic" / "003_segmentation" / "errors" / "acoustic_segmentation_errors.csv"
        if not main_path.exists():
            self.segmentation_feedback.setPlainText("Run data segmentation to summarize speech/pause regions, segment counts, speech fraction, and output tables.")
            return
        try:
            df = self._read_csv_safe(main_path)
            err_df = self._read_csv_safe(errors_path)
            err_n = len(err_df)
            n = len(df)
            lines = [
                f"Files segmented: {n}",
                f"Files failed: {err_n}",
            ]
            if n == 0:
                lines.append("No successful segmentation rows were available. Check the segmentation errors table for details.")
                self.segmentation_feedback.setPlainText("\n".join(lines))
                return
            speech_fraction = pd.to_numeric(df.get("speech_fraction"), errors="coerce")
            speech_segments = pd.to_numeric(df.get("n_speech_segments"), errors="coerce")
            internal_pause = pd.to_numeric(df.get("n_internal_nonspeech_segments"), errors="coerce")
            longest_pause = pd.to_numeric(df.get("longest_internal_nonspeech_sec"), errors="coerce")
            if speech_fraction.notna().any():
                lines.append(f"Median speech fraction: {speech_fraction.median():.3f}")
            if speech_segments.notna().any():
                lines.append(f"Median speech segments/file: {speech_segments.median():.1f}")
            if internal_pause.notna().any():
                lines.append(f"Median internal pause segments/file: {internal_pause.median():.1f}")
            if longest_pause.notna().any():
                lines.append(f"Median longest internal pause: {longest_pause.median():.2f} s")
            self.segmentation_feedback.setPlainText("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            self.segmentation_feedback.setPlainText(f"Could not summarize segmentation outputs: {exc}")

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

    def browse_input_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select input audio folder")
        if folder:
            self.input_edit.setText(folder)

    def browse_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select output project folder")
        if folder:
            self.output_edit.setText(folder)
            self.refresh_latest_outputs()

    def browse_demographics_csv(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select demographics CSV", "", "CSV files (*.csv);;All files (*)")
        if file_path:
            self.demographics_csv_edit.setText(file_path)

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
            self._thread.quit(); self._thread.wait()
        self._thread = None; self._worker = None; self._set_busy(False)

    def _on_worker_started(self, name: str) -> None:
        self._set_busy(True)
        if name in self.stage_records:
            self.stage_records[name].status = "running"
            self._refresh_stage_cards()
        self.append_log(f"=== {name} ===")

    def _on_worker_failed(self, name: str, err: str) -> None:
        rec = self.stage_records.get(name, StageRecord())
        rec.status = "failed"; self.stage_records[name] = rec
        self._refresh_stage_cards()
        self.append_log(f"ERROR in {name}:\n{err}")
        QMessageBox.critical(self, f"{name} failed", err)

    def _refresh_stage_cards(self) -> None:
        for key, lbl in self.stage_labels.items():
            status = self.stage_records[key].status
            lbl.setText(self._stage_status_text(status))
            if status in {"completed", "detected"}:
                lbl.setStyleSheet("color:#8EF2C6; font-weight:800;")
            elif status == "completed_with_warnings":
                lbl.setStyleSheet("color:#FFD98E; font-weight:800;")
            elif status == "failed":
                lbl.setStyleSheet("color:#FF8EA3; font-weight:800;")
            elif status == "running":
                lbl.setStyleSheet("color:#8FCBFF; font-weight:800;")
            else:
                lbl.setStyleSheet("color:#BFD5E6;")

    def _open_stage_file(self, stage_key: str, kind: str) -> None:
        target = self._stage_path(stage_key, kind)
        if not target.exists():
            QMessageBox.information(self, "Not available", f"No {kind} file is available yet for stage: {stage_key}.\n\nExpected:\n{target}")
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
        p = Path(out).expanduser(); p.mkdir(parents=True, exist_ok=True); open_path(p)

    # ---------------------------- ACTIONS ----------------------------
    def run_project_init(self) -> None:
        paths = self._require_paths()
        if paths is None: return
        _input_path, output_root = paths
        self._run_worker("project", initialize_project, {"output_root": output_root, "project_name": self.project_name_edit.text().strip() or "VSLP Acoustic Project"})

    def run_metadata(self) -> None:
        paths = self._require_paths()
        if paths is None: return
        if not self._require_project_initialized(): return
        input_path, output_root = paths
        demo_text = self.demographics_csv_edit.text().strip() if hasattr(self, "demographics_csv_edit") else ""
        cfg = MetadataConfig(demographics_csv=demo_text or None)
        self._run_worker("metadata", run_acoustic_metadata, {"input_path": input_path, "output_root": output_root, "config": cfg})

    def run_ingest(self) -> None:
        paths = self._require_paths()
        if paths is None: return
        if not self._require_project_initialized(): return
        input_path, output_root = paths
        self._run_worker("ingest", run_acoustic_ingest, {"input_path": input_path, "output_root": output_root})

    def _preprocess_filter_config_from_gui(self) -> FilterConfig:
        preset = self.filter_preset_combo.currentText() if hasattr(self, "filter_preset_combo") else "none"
        order = int(self.filter_order_spin.value()) if hasattr(self, "filter_order_spin") else 4
        low = float(self.low_hz_spin.value()) if hasattr(self, "low_hz_spin") else None
        high = float(self.high_hz_spin.value()) if hasattr(self, "high_hz_spin") else None
        if preset == "high-pass Butterworth":
            return FilterConfig(enabled=True, kind="hpf", low_hz=low, order=order)
        if preset == "low-pass Butterworth":
            return FilterConfig(enabled=True, kind="lpf", high_hz=high, order=order)
        if preset == "band-pass Butterworth":
            return FilterConfig(enabled=True, kind="bpf", low_hz=low, high_hz=high, order=order)
        if preset == "notch 50 Hz":
            return FilterConfig(enabled=True, kind="notch", notch_hz=50.0, order=order)
        if preset == "notch 60 Hz":
            return FilterConfig(enabled=True, kind="notch", notch_hz=60.0, order=order)
        return FilterConfig(enabled=False, kind="none", order=order)

    def _preprocess_config_from_gui(self) -> PreprocessConfig:
        feature_sr = int(self.feature_sr_spin.value()) if getattr(self, "feature_resample_check", None) and self.feature_resample_check.isChecked() else None
        return PreprocessConfig(
            segmentation_sample_rate_hz=int(self.seg_sr_spin.value()),
            feature_sample_rate_hz=feature_sr,
            make_segmentation_wav=bool(self.make_segmentation_wav_check.isChecked()),
            make_feature_wav=bool(self.make_feature_wav_check.isChecked()),
            remove_dc_offset=bool(self.remove_dc_check.isChecked()),
            normalize_peak=bool(self.normalize_check.isChecked()),
            normalize_peak_target_abs=float(self.normalize_target_spin.value()),
            filter=self._preprocess_filter_config_from_gui(),
        )

    def run_preprocess(self) -> None:
        paths = self._require_paths()
        if paths is None: return
        if not self._require_project_initialized(): return
        input_path, output_root = paths
        cfg = self._preprocess_config_from_gui()
        self._run_worker("preprocess", run_acoustic_preprocess, {"input_path": input_path, "output_root": output_root, "config": cfg})

    def run_segmentation(self) -> None:
        paths = self._require_paths()
        if paths is None: return
        if not self._require_project_initialized(): return
        method = self.segmentation_method_combo.currentText() if hasattr(self, "segmentation_method_combo") else "Silero"
        if not method.startswith("Silero"):
            QMessageBox.information(
                self,
                "Method not implemented",
                "This segmentation method is reserved as a plugin slot and is not implemented yet. Please use Silero for the current build."
            )
            return
        _input_path, output_root = paths
        preprocess_summary = self._preprocess_summary_path()
        if not preprocess_summary.exists():
            QMessageBox.warning(self, "Preprocess required", "Run preprocessing first. The preprocess summary CSV was not found.")
            return
        self._run_worker("segment", run_acoustic_segmentation_silero, {
            "preprocess_summary_csv": preprocess_summary, "output_root": output_root,
            "threshold": float(self.threshold_spin.value()),
            "min_speech_duration_ms": int(self.min_speech_spin.value()),
            "min_silence_duration_ms": int(self.min_silence_spin.value()),
            "speech_pad_ms": int(self.speech_pad_spin.value()),
            "frame_ms": int(self.frame_ms_spin.value()),
            "force_reload": bool(self.force_reload_check.isChecked()),
        })

    def _update_quality_feedback(self) -> None:
        if not hasattr(self, "quality_summary_label"):
            return
        path = self._stage_path("quality", "family_status")
        if not path.exists():
            self.quality_summary_label.setText("Quality Control has not been run.")
            return
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            self.quality_summary_label.setText(f"Could not summarize QC outputs: {exc}")
            return
        if df.empty:
            self.quality_summary_label.setText("Quality Control produced no family-status rows.")
            return
        n_files = df["file_name"].nunique() if "file_name" in df.columns else 0
        computed = int(df["status"].astype(str).str.contains("computed", na=False).sum()) if "status" in df.columns else 0
        failed = int((df["status"].astype(str) == "failed").sum()) if "status" in df.columns else 0
        selected = int((df["status"].astype(str) != "not_selected").sum()) if "status" in df.columns else len(df)
        self.quality_summary_label.setText(
            f"Files evaluated: {n_files}\n"
            f"Selected family evaluations: {selected}\n"
            f"Computed/proxy evaluations: {computed}\n"
            f"Failed evaluations: {failed}"
        )

    def run_quality_control(self) -> None:
        paths = self._require_paths()
        if paths is None: return
        if not self._require_project_initialized(): return
        _input_path, output_root = paths
        segmentation_summary = self._segmentation_summary_path()
        if not segmentation_summary.exists():
            QMessageBox.warning(self, "Segmentation required", "Run Data Segmentation first. Quality Control uses segmentation frames and segment tables.")
            return
        selected = [family for family, cb in getattr(self, "qc_family_checks", {}).items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "No QC families selected", "Select at least one QC family.")
            return
        cfg = QualityControlConfig(
            selected_families=selected,
            minimum_internal_pause_sec=float(self.qc_pause_sec_spin.value()),
            high_level_percentile=float(self.qc_high_level_spin.value()),
            hard_clip_threshold=float(self.qc_hard_clip_spin.value()),
            near_clip_threshold=float(self.qc_near_clip_spin.value()),
        )
        self._run_worker("quality", run_acoustic_quality_control, {"segmentation_summary_csv": segmentation_summary, "output_root": output_root, "config": cfg})

    def run_features(self) -> None:
        paths = self._require_paths()
        if paths is None: return
        if not self._require_project_initialized(): return
        _input_path, output_root = paths
        segmentation_summary = self._segmentation_summary_path()
        if not segmentation_summary.exists():
            QMessageBox.warning(self, "Segmentation required", "Run Silero segmentation first. The segmentation summary CSV was not found.")
            return
        selected_features = self._selected_feature_names()
        if not selected_features:
            QMessageBox.warning(self, "No features selected", "Select at least one feature in the feature tree.")
            return
        metadata_index = self._metadata_index_path()
        cfg = FeatureExtractionConfig(
            selected_features=selected_features,
            minimum_pause_duration_sec=float(self.min_pause_feature_spin.value()),
            metadata_csv=str(metadata_index) if metadata_index.exists() else None,
            acoustic_region_policy=self.region_policy_combo.currentText(),
        )
        self._run_worker("features", run_acoustic_feature_extraction, {"segmentation_summary_csv": segmentation_summary, "output_root": output_root, "config": cfg})


    def run_aggregation(self) -> None:
        paths = self._require_paths()
        if paths is None: return
        if not self._require_project_initialized(): return
        _input_path, output_root = paths
        features_csv = self._stage_path("features", "summary")
        if not features_csv.exists():
            QMessageBox.warning(self, "Features required", "Run feature extraction first. The feature table was not found.")
            return
        cfg = AggregationConfig(
            group_columns=[c.strip() for c in self.agg_group_columns_edit.text().split(",") if c.strip()],
            numeric_policy=self.agg_numeric_policy_combo.currentText(),
            missing_policy=self.agg_missing_policy_combo.currentText(),
            max_missing_fraction=float(self.agg_missing_threshold_spin.value()),
        )
        self._run_worker("aggregate", run_acoustic_aggregation, {"features_csv": features_csv, "output_root": output_root, "config": cfg})

    def run_qc_dashboard(self) -> None:
        paths = self._require_paths()
        if paths is None: return
        if not self._require_project_initialized(): return
        _input_path, output_root = paths
        preprocess_csv = self._stage_path("preprocess", "summary")
        if not preprocess_csv.exists():
            QMessageBox.warning(self, "Preprocess required", "Run preprocessing first. QC uses preprocessing outputs at minimum.")
            return
        cfg = AcousticQCConfig(
            min_snr_db=float(self.qc_min_snr_spin.value()),
            max_clipping_fraction=float(self.qc_clip_spin.value()),
            min_speech_fraction=float(self.qc_min_speech_spin.value()),
            max_speech_fraction=float(self.qc_max_speech_spin.value()),
        )
        self._run_worker("qc", run_acoustic_qc_dashboard, {"output_root": output_root, "config": cfg})

    def run_all(self) -> None:
        paths = self._require_paths()
        if paths is None: return
        if not self._require_project_initialized(): return
        input_path, output_root = paths

        selected_features = self._selected_feature_names()
        def full_run(input_path: Path, output_root: Path):
            demo_text = self.demographics_csv_edit.text().strip() if hasattr(self, "demographics_csv_edit") else ""
            metadata_result = run_acoustic_metadata(input_path=input_path, output_root=output_root, config=MetadataConfig(demographics_csv=demo_text or None))
            ingest_result = run_acoustic_ingest(input_path=input_path, output_root=output_root)
            preprocess_cfg = self._preprocess_config_from_gui()
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
                config=FeatureExtractionConfig(
                    selected_features=selected_features,
                    minimum_pause_duration_sec=float(self.min_pause_feature_spin.value()),
                    metadata_csv=str(output_root / "acoustic" / "000_metadata" / "tables" / "project_file_index.csv"),
                    acoustic_region_policy=self.region_policy_combo.currentText(),
                ),
            )
            aggregate_result = run_acoustic_aggregation(
                features_csv=output_root / "acoustic" / "004_features" / "tables" / "acoustic_features_per_file.csv",
                output_root=output_root,
                config=AggregationConfig(
                    group_columns=[c.strip() for c in self.agg_group_columns_edit.text().split(",") if c.strip()],
                    numeric_policy=self.agg_numeric_policy_combo.currentText(),
                    missing_policy=self.agg_missing_policy_combo.currentText(),
                    max_missing_fraction=float(self.agg_missing_threshold_spin.value()),
                ),
            )
            qc_result = run_acoustic_qc_dashboard(
                output_root=output_root,
                config=AcousticQCConfig(
                    min_snr_db=float(self.qc_min_snr_spin.value()),
                    max_clipping_fraction=float(self.qc_clip_spin.value()),
                    min_speech_fraction=float(self.qc_min_speech_spin.value()),
                    max_speech_fraction=float(self.qc_max_speech_spin.value()),
                ),
            )
            return {"metadata": metadata_result, "ingest": ingest_result, "preprocess": preprocess_result, "segment": segment_result, "features": features_result, "aggregate": aggregate_result, "qc": qc_result}
        self._run_worker("full_run", full_run, {"input_path": input_path, "output_root": output_root})

    def _on_worker_finished(self, name: str, result: object) -> None:  # type: ignore[override]
        if name == "full_run" and isinstance(result, dict):
            for stage_name in ["metadata", "ingest", "preprocess", "segment", "quality", "features", "aggregate", "qc"]:
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
        if report: self.append_log(f"Report: {report}")
        if summary: self.append_log(f"Summary: {summary}")
        if name == "project":
            self._refresh_project_gate()
        if name == "ingest":
            self._update_ingest_feedback()
            if errors:
                self.append_log(f"Ingest errors: {errors}")
        if name == "preprocess":
            self._update_preprocess_feedback()
        if name == "segment":
            self._update_segmentation_feedback()
        if name == "quality":
            self._update_quality_feedback()
        self.refresh_latest_outputs()
