"""VSLP Acoustic Pipeline GUI v0.38.

V0.38 final polish:
- stage-aware workflow guidance with scientific rationale;
- embedded CSV and plot previews;
- latest-output detection;
- feature-level selection inside each subsystem;
- quick selectors for all / implemented / proxy / pending features;
- region-aware feature extraction;
- dedicated Quality Control stage;
- clearer separation of clinical/simple controls and expert controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
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

from vslp.acoustic.features.registry import build_acoustic_feature_registry
from vslp.acoustic.features.scales import build_feature_family_policy_summary
from vslp.acoustic.features.stage import (
    IMPLEMENTED_FEATURES,
    PROXY_FEATURES,
    FeatureExtractionConfig,
    run_acoustic_feature_extraction,
)
from vslp.acoustic.ingest.stage import run_acoustic_ingest
from vslp.acoustic.run_setup import initialize_acoustic_run
from vslp.acoustic.preprocess.stage import PreprocessConfig, run_acoustic_preprocess
from vslp.acoustic.segment.pipeline import SegmentationConfig, run_acoustic_segmentation
from vslp.acoustic.segment.selection import SILERO, DDK, PHONATION, CUSTOM, recommend_method
from vslp.acoustic.segment.task_methods import DDKConfig, PhonationConfig
from vslp.acoustic.quality.stage import (
    QC_FAMILIES,
    FAMILY_FEATURES,
    QualityControlConfig,
    quality_feature_registry,
    run_acoustic_quality_control,
)
from vslp.core.project import prune_empty_acoustic_directories, task_run_folder_name
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult
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
        self.output_root = kwargs.get("output_root")

    def run(self) -> None:
        self.started.emit(self.name)
        self.message.emit(f"Starting {self.name}...")
        try:
            result = self.func(**self.kwargs)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            if self.output_root is not None:
                prune_empty_acoustic_directories(self.output_root)
            self.failed.emit(self.name, f"{exc}\n\n{tb}")
            return
        if self.output_root is not None:
            prune_empty_acoustic_directories(self.output_root)
        self.message.emit(f"Finished {self.name}: {getattr(result, 'status', 'ok')}")
        self.finished.emit(self.name, result)


class AcousticPipelineWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VSLP - Acoustic Pipeline")
        self.resize(1480, 900)
        self.setMinimumSize(1180, 760)
        self._run_root: Path | None = None
        self._current_preview_pixmap: QPixmap | None = None
        self._current_preview_path: Path | None = None

        self.registry = build_acoustic_feature_registry()
        self._updating_feature_tree = False
        self.stage_records: dict[str, StageRecord] = {
            "project": StageRecord(),
            "ingest": StageRecord(),
            "preprocess": StageRecord(),
            "segment": StageRecord(),
            "quality": StageRecord(),
            "features": StageRecord(),
            "reports": StageRecord(),
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
        subtitle = QLabel("Acoustic Pipeline GUI v0.38")
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
            ("ingest", "Ingest"),
            ("preprocess", "Preprocess"),
            ("segment", "Data Segmentation"),
            ("quality", "Quality Control"),
            ("features", "Feature Extraction"),
            ("reports", "Reports"),
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

        self.run_all_btn = QPushButton("Run Full Acoustic Workflow")
        self.run_all_btn.setObjectName("RunButton")
        self.run_all_btn.clicked.connect(self.run_all)
        side_layout.addWidget(self.run_all_btn)

        self.open_output_btn = QPushButton("Open Run Folder")
        self.open_output_btn.setObjectName("OpenButton")
        self.open_output_btn.clicked.connect(self.open_output_root)
        side_layout.addWidget(self.open_output_btn)

        root.addWidget(sidebar)

        main_col = QVBoxLayout()
        main_col.setSpacing(16)

        branding_bar = self._build_branding_bar()
        main_col.addWidget(branding_bar)

        tabs = QTabWidget()
        self.tabs = tabs
        tabs.setUsesScrollButtons(True)
        tabs.setElideMode(Qt.ElideRight)
        tabs.addTab(self._build_setup_tab(), "Setup")
        tabs.addTab(self._build_preprocess_tab(), "Preprocess")
        tabs.addTab(self._build_segment_tab(), "Segmentation")
        tabs.addTab(self._build_quality_tab(), "Quality Control")
        tabs.addTab(self._build_features_tab(), "Features")
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

        lab_logo = self._logo_or_text("speech_production_lab_logo.png", "Speech Production Lab", max_width=260, max_height=64)
        uoft_logo = self._logo_or_text("uoft_logo.png", "University of Toronto", max_width=300, max_height=60)
        layout.addWidget(lab_logo)
        layout.addWidget(uoft_logo)

        bar.setMaximumHeight(86)
        return bar

    def _logo_or_text(self, logo_file: str, fallback: str, max_width: int = 170, max_height: int = 42) -> QLabel:
        assets_dir = Path(__file__).resolve().parents[1] / "assets" / "branding"
        logo_path = assets_dir / logo_file
        widget = QLabel()
        widget.setObjectName("LogoPlaceholder")
        widget.setAlignment(Qt.AlignCenter)
        widget.setMinimumWidth(min(max_width, 220))
        widget.setMaximumWidth(max_width)
        widget.setMaximumHeight(max_height)
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            if not pixmap.isNull():
                widget.setObjectName("LogoImage")
                widget.setPixmap(pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))
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
        self.project_name_edit = QLineEdit()
        self.project_name_edit.setPlaceholderText("Required project name")
        self.task_name_edit = QLineEdit()
        self._set_tooltip(self.input_edit, "Folder containing raw audio/video files. Subfolders are searched recursively. Recommendation: process one speech task at a time when possible.")
        self._set_tooltip(self.output_edit, "Parent folder for a new acoustic run. It is never used as a stage output root.")
        self._set_tooltip(self.project_name_edit, "Required. Stored in the immutable run manifest and setup config.")
        self._set_tooltip(self.task_name_edit, "Required. The human-readable name is preserved; a safe slug names the run folder.")

        self.browse_input_btn = QPushButton("Browse Input Folder")
        self.browse_input_btn.clicked.connect(self.browse_input_dir)
        self.browse_output_btn = QPushButton("Browse Output Folder")
        self.browse_output_btn.clicked.connect(self.browse_output_dir)

        form.addWidget(QLabel("Input audio folder"), 0, 0)
        form.addWidget(self.input_edit, 0, 1)
        form.addWidget(self.browse_input_btn, 0, 2)
        form.addWidget(QLabel("Output parent folder"), 1, 0)
        form.addWidget(self.output_edit, 1, 1)
        form.addWidget(self.browse_output_btn, 1, 2)
        form.addWidget(QLabel("Project name"), 2, 0)
        form.addWidget(self.project_name_edit, 2, 1, 1, 2)
        form.addWidget(QLabel("Task name"), 3, 0)
        form.addWidget(self.task_name_edit, 3, 1, 1, 2)
        self.run_root_edit = QLineEdit()
        self.run_root_edit.setReadOnly(True)
        self.run_root_edit.setPlaceholderText("Created after Initialize Project")
        form.addWidget(QLabel("Generated run folder"), 4, 0)
        form.addWidget(self.run_root_edit, 4, 1, 1, 2)
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
            "No ingest results yet."
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

    def _build_preprocess_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        options = QGroupBox("Preprocessing")
        options_layout = QVBoxLayout(options)
        self.remove_dc_check = QCheckBox("Remove DC offset")
        self.remove_dc_check.setChecked(True)
        options_layout.addWidget(self.remove_dc_check)
        layout.addWidget(options)

        run_btn = QPushButton("Run Preprocessing")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_preprocess)
        layout.addWidget(run_btn)

        self.preprocess_feedback = QLabel("Not run")
        self.preprocess_feedback.setObjectName("SubtitleLabel")
        layout.addWidget(self.preprocess_feedback)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_segment_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        method_group = QGroupBox("Segmentation method")
        form = QFormLayout(method_group)
        self.segmentation_method_combo = QComboBox()
        for label, code in (
            ("Silero VAD", SILERO), ("DDK energy-envelope", DDK),
            ("Sustained phonation", PHONATION), ("Custom/plugin", CUSTOM),
        ):
            self.segmentation_method_combo.addItem(label, code)
        self.segmentation_method_combo.currentIndexChanged.connect(self._update_segmentation_method_ui)
        form.addRow("Method", self.segmentation_method_combo)
        layout.addWidget(method_group)

        self.silero_group = QGroupBox("Silero VAD")
        form = QFormLayout(self.silero_group)
        self.threshold_spin = QDoubleSpinBox(); self.threshold_spin.setDecimals(2); self.threshold_spin.setRange(0.01, 0.99); self.threshold_spin.setSingleStep(0.05); self.threshold_spin.setValue(0.50)
        self._set_tooltip(self.threshold_spin, "Speech evidence: higher may miss weak or breathy speech; lower may retain noise.")
        self.min_speech_spin = QSpinBox(); self.min_speech_spin.setRange(0, 5000); self.min_speech_spin.setValue(250)
        self._set_tooltip(self.min_speech_spin, "Shortest retained speech event; longer may remove fragmented speech.")
        self.min_silence_spin = QSpinBox(); self.min_silence_spin.setRange(0, 5000); self.min_silence_spin.setValue(100)
        self._set_tooltip(self.min_silence_spin, "Gap needed to separate speech; longer values bridge brief pauses.")
        self.speech_pad_spin = QSpinBox(); self.speech_pad_spin.setRange(0, 2000); self.speech_pad_spin.setValue(0)
        self._set_tooltip(self.speech_pad_spin, "Artificial boundary expansion; 0 ms preserves measured onset and offset.")
        self.silero_profile_combo = QComboBox(); self.silero_profile_combo.addItems(["default", "conservative", "permissive"])
        self._set_tooltip(self.silero_profile_combo, "Sensitivity check: conservative needs stronger speech evidence; permissive retains weaker speech.")
        form.addRow("Threshold", self.threshold_spin)
        form.addRow("Minimum speech (ms)", self.min_speech_spin)
        form.addRow("Minimum silence (ms)", self.min_silence_spin)
        form.addRow("Speech padding (ms)", self.speech_pad_spin)
        form.addRow("Sensitivity profile", self.silero_profile_combo)
        layout.addWidget(self.silero_group)

        self.ddk_group = QGroupBox("DDK energy-envelope")
        form = QFormLayout(self.ddk_group)
        self.ddk_threshold_spin = QDoubleSpinBox(); self.ddk_threshold_spin.setRange(0.05, 0.90); self.ddk_threshold_spin.setSingleStep(0.05); self.ddk_threshold_spin.setValue(0.35)
        self._set_tooltip(self.ddk_threshold_spin, "Local energy fraction separating syllable events from valleys; lower retains weak syllables.")
        self.ddk_min_event_spin = QSpinBox(); self.ddk_min_event_spin.setRange(10, 500); self.ddk_min_event_spin.setValue(35)
        self._set_tooltip(self.ddk_min_event_spin, "Shortest measurable syllable energy event; longer values suppress brief noise bursts.")
        self.ddk_local_window_spin = QSpinBox(); self.ddk_local_window_spin.setRange(100, 2000); self.ddk_local_window_spin.setValue(600)
        self._set_tooltip(self.ddk_local_window_spin, "Time span used to adapt to changing loudness across repetitions.")
        form.addRow("Local threshold fraction", self.ddk_threshold_spin)
        form.addRow("Minimum event (ms)", self.ddk_min_event_spin)
        form.addRow("Local window (ms)", self.ddk_local_window_spin)
        layout.addWidget(self.ddk_group)

        self.phonation_group = QGroupBox("Sustained phonation")
        form = QFormLayout(self.phonation_group)
        self.phon_min_spin = QSpinBox(); self.phon_min_spin.setRange(100, 10000); self.phon_min_spin.setValue(500)
        self._set_tooltip(self.phon_min_spin, "Shortest detectable phonation episode; shorter episodes remain available for review.")
        self.phon_onset_spin = QSpinBox(); self.phon_onset_spin.setRange(0, 5000); self.phon_onset_spin.setValue(1000)
        self._set_tooltip(self.phon_onset_spin, "Excludes initial vocal-fold and airflow build-up from the stable region.")
        self.phon_offset_spin = QSpinBox(); self.phon_offset_spin.setRange(0, 5000); self.phon_offset_spin.setValue(300)
        self._set_tooltip(self.phon_offset_spin, "Excludes terminal breath and vibration decay from the stable region.")
        self.phon_stable_spin = QSpinBox(); self.phon_stable_spin.setRange(100, 10000); self.phon_stable_spin.setValue(2000)
        self._set_tooltip(self.phon_stable_spin, "Interior duration for stationary voice measures; full episode retains voice breaks.")
        self.phon_activity_spin = QDoubleSpinBox(); self.phon_activity_spin.setRange(0.02, 0.90); self.phon_activity_spin.setSingleStep(0.02); self.phon_activity_spin.setValue(0.18)
        self._set_tooltip(self.phon_activity_spin, "Energy above room baseline needed to mark active phonation.")
        self.phon_gap_spin = QSpinBox(); self.phon_gap_spin.setRange(0, 2000); self.phon_gap_spin.setValue(500)
        self._set_tooltip(self.phon_gap_spin, "Keep brief internal voice breaks inside the full episode; breaks remain visible.")
        form.addRow("Minimum phonation (ms)", self.phon_min_spin)
        form.addRow("Onset guard (ms)", self.phon_onset_spin)
        form.addRow("Offset guard (ms)", self.phon_offset_spin)
        form.addRow("Stable duration (ms)", self.phon_stable_spin)
        form.addRow("Activity fraction", self.phon_activity_spin)
        form.addRow("Internal break span (ms)", self.phon_gap_spin)
        layout.addWidget(self.phonation_group)

        self.segmentation_recommendation = QLabel()
        layout.addWidget(self.segmentation_recommendation)
        self.segmentation_feedback = QPlainTextEdit(); self.segmentation_feedback.setReadOnly(True); self.segmentation_feedback.setMaximumHeight(110)
        self.segmentation_feedback.setPlainText("Not run")
        layout.addWidget(self.segmentation_feedback)
        run_btn = QPushButton("Run Segmentation"); run_btn.setObjectName("RunButton"); run_btn.clicked.connect(self.run_segmentation)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        self.task_name_edit.textChanged.connect(self._recommend_segmentation_method)
        self._update_segmentation_method_ui()
        return self._scrollable(container)

    def _recommend_segmentation_method(self, task: str) -> None:
        method = recommend_method(task)
        index = self.segmentation_method_combo.findData(method)
        if index >= 0:
            self.segmentation_method_combo.setCurrentIndex(index)

    def _update_segmentation_method_ui(self) -> None:
        method = self.segmentation_method_combo.currentData()
        self.silero_group.setVisible(method == SILERO)
        self.ddk_group.setVisible(method == DDK)
        self.phonation_group.setVisible(method == PHONATION)
        recommendations = {
            SILERO: "Recommended for: Bamboo Passage · Buy Bobby a Puppy · WSTG · connected speech",
            DDK: "Recommended for: pa · ta · ka · ba · pataka · repetitive oral DDK",
            PHONATION: "Recommended for: sustained vowels /a/, /i/, etc.",
            CUSTOM: "Custom plugin method is not installed.",
        }
        self.segmentation_recommendation.setText(recommendations[method])

    def _segmentation_config_from_gui(self) -> SegmentationConfig:
        return SegmentationConfig(
            method=self.segmentation_method_combo.currentData(),
            threshold=float(self.threshold_spin.value()),
            min_speech_duration_ms=int(self.min_speech_spin.value()),
            min_silence_duration_ms=int(self.min_silence_spin.value()),
            speech_pad_ms=int(self.speech_pad_spin.value()),
            sensitivity_profile=self.silero_profile_combo.currentText(),
            ddk=DDKConfig(threshold_fraction=float(self.ddk_threshold_spin.value()),
                          min_event_ms=float(self.ddk_min_event_spin.value()),
                          local_window_ms=float(self.ddk_local_window_spin.value())),
            phonation=PhonationConfig(min_phonation_ms=float(self.phon_min_spin.value()),
                                     onset_guard_ms=float(self.phon_onset_spin.value()),
                                     offset_guard_ms=float(self.phon_offset_spin.value()),
                                     stable_duration_ms=float(self.phon_stable_spin.value()),
                                     activity_fraction=float(self.phon_activity_spin.value()),
                                     max_internal_gap_ms=float(self.phon_gap_spin.value())),
        )

    def _build_quality_tab(self) -> QWidget:
        """Build the segmentation-informed Quality Control workstation."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.addWidget(self._info_panel(
            "Info",
            "Quality Control extracts artifact-oriented features from segmented recordings. Use this block to select QC families/features, tune transparent screening parameters, and review recording-quality outputs before acoustic feature extraction."
        ))

        qc_tabs = QTabWidget()
        qc_tabs.addTab(self._build_quality_configure_panel(), "Configure")
        qc_tabs.addTab(self._build_quality_outputs_panel(), "Outputs")
        layout.addWidget(qc_tabs, stretch=1)
        self._refresh_qc_count_label()
        self._update_qc_parameter_relevance()
        return self._scrollable(container)

    def _build_quality_configure_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        selector_group = QGroupBox("QC feature selection")
        selector_layout = QVBoxLayout(selector_group)
        quick = QHBoxLayout()
        for label, callback in [
            ("All", self.select_all_qc_features),
            ("Recommended", self.select_recommended_qc_features),
            ("Clear", self.clear_qc_features),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(callback)
            quick.addWidget(btn)
        selector_layout.addLayout(quick)

        self.qc_feature_tree = QTreeWidget()
        self.qc_feature_tree.setHeaderLabels(["QC family / feature", "Role", "Parameters"])
        self.qc_feature_tree.setMinimumHeight(430)
        self.qc_feature_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.qc_feature_tree.itemChanged.connect(self._on_qc_tree_item_changed)
        self.qc_feature_tree.itemSelectionChanged.connect(self._update_qc_feature_detail)
        self.qc_family_items: dict[str, QTreeWidgetItem] = {}
        self.qc_feature_items: dict[str, QTreeWidgetItem] = {}
        self._populate_qc_feature_tree()
        selector_layout.addWidget(self.qc_feature_tree, stretch=1)
        left_layout.addWidget(selector_group)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)

        plan_group = QGroupBox("QC plan")
        plan_layout = QVBoxLayout(plan_group)
        self.qc_count_label = QLabel("")
        self.qc_count_label.setObjectName("SectionHeader")
        self.qc_parameter_relevance_label = QLabel("")
        self.qc_parameter_relevance_label.setObjectName("SubtitleLabel")
        self.qc_parameter_relevance_label.setWordWrap(True)
        plan_layout.addWidget(self.qc_count_label)
        plan_layout.addWidget(self.qc_parameter_relevance_label)
        right_layout.addWidget(plan_group)

        detail_group = QGroupBox("Selected feature / family")
        detail_layout = QVBoxLayout(detail_group)
        self.qc_feature_detail_box = QPlainTextEdit()
        self.qc_feature_detail_box.setReadOnly(True)
        self.qc_feature_detail_box.setMinimumHeight(150)
        self.qc_feature_detail_box.setMaximumHeight(220)
        detail_layout.addWidget(self.qc_feature_detail_box)
        right_layout.addWidget(detail_group)

        param_group = QGroupBox("Computation parameters")
        form = QFormLayout(param_group)
        self.qc_pause_sec_spin = QDoubleSpinBox(); self.qc_pause_sec_spin.setDecimals(2); self.qc_pause_sec_spin.setRange(0.0, 5.0); self.qc_pause_sec_spin.setSingleStep(0.05); self.qc_pause_sec_spin.setValue(0.15)
        self.qc_pause_sec_spin.setToolTip("Pause-support threshold. Pauses shorter than this are ignored for pause-region noise and reverberation estimates.")
        self.qc_high_level_spin = QDoubleSpinBox(); self.qc_high_level_spin.setDecimals(1); self.qc_high_level_spin.setRange(50.0, 99.9); self.qc_high_level_spin.setSingleStep(1.0); self.qc_high_level_spin.setValue(90.0)
        self.qc_high_level_spin.setToolTip("Speech energy percentile used to identify high-level speech frames for gain and nonlinear-distortion screening.")
        self.qc_hard_clip_spin = QDoubleSpinBox(); self.qc_hard_clip_spin.setDecimals(3); self.qc_hard_clip_spin.setRange(0.5, 1.0); self.qc_hard_clip_spin.setSingleStep(0.005); self.qc_hard_clip_spin.setValue(0.995)
        self.qc_hard_clip_spin.setToolTip("Absolute waveform threshold for severe clipping/saturation in normalized audio.")
        self.qc_near_clip_spin = QDoubleSpinBox(); self.qc_near_clip_spin.setDecimals(3); self.qc_near_clip_spin.setRange(0.5, 1.0); self.qc_near_clip_spin.setSingleStep(0.005); self.qc_near_clip_spin.setValue(0.950)
        self.qc_near_clip_spin.setToolTip("Softer overload-risk threshold for near-clipping detection.")
        form.addRow("Minimum internal pause (s)", self.qc_pause_sec_spin)
        form.addRow("High-level speech percentile", self.qc_high_level_spin)
        form.addRow("Hard clipping threshold", self.qc_hard_clip_spin)
        form.addRow("Near-clipping threshold", self.qc_near_clip_spin)
        reset_btn = QPushButton("Reset defaults")
        reset_btn.clicked.connect(self.reset_qc_parameters)
        form.addRow("", reset_btn)
        right_layout.addWidget(param_group)

        run_btn = QPushButton("Run Quality Control")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_quality_control)
        right_layout.addWidget(run_btn)
        right_layout.addStretch(1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([760, 520])
        splitter.setMinimumHeight(620)
        layout.addWidget(splitter)
        return panel

    def _build_quality_outputs_panel(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        left = QGroupBox("Output previews")
        left_layout = QVBoxLayout(left)
        self.quality_summary_label = QLabel("Quality Control has not been run.")
        self.quality_summary_label.setObjectName("SubtitleLabel")
        self.quality_summary_label.setWordWrap(True)
        left_layout.addWidget(self.quality_summary_label)

        tables_group = QGroupBox("Tables")
        tables_layout = QGridLayout(tables_group)
        table_buttons = [
            ("Features", lambda: self.preview_csv(self._stage_path("quality", "summary"))),
            ("Main summary", lambda: self.preview_csv(self._stage_path("quality", "main_summary"))),
            ("Warnings", lambda: self.preview_csv(self._stage_path("quality", "warnings"))),
            ("Recommendations", lambda: self.preview_csv(self._stage_path("quality", "recommendations"))),
            ("Family scores", lambda: self.preview_csv(self._stage_path("quality", "family_scores"))),
            ("Review rank", lambda: self.preview_csv(self._stage_path("quality", "review_rank"))),
        ]
        for i, (label, callback) in enumerate(table_buttons):
            btn = QPushButton(label)
            btn.clicked.connect(callback)
            tables_layout.addWidget(btn, i // 2, i % 2)
        left_layout.addWidget(tables_group)

        plots_group = QGroupBox("Plots")
        plots_layout = QGridLayout(plots_group)
        plot_buttons = [
            ("Family burden", "quality_family_score_distributions.png"),
            ("Review ranking", "quality_recording_review_rank.png"),
            ("Warning heatmap", "quality_warning_heatmap.png"),
            ("Recommendations", "quality_recommendation_summary.png"),
            ("Feature correlations", "quality_feature_correlation_heatmap.png"),
            ("Family correlations", "quality_family_correlation_heatmap.png"),
            ("Feature coverage", "quality_missingness_feature_coverage.png"),
            ("PCA embedding", "quality_pca_embedding.png"),
        ]
        for i, (label, filename) in enumerate(plot_buttons):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _checked=False, fn=filename: self.preview_image(self._output_root() / "acoustic" / "003_quality_control" / "plots" / fn))
            plots_layout.addWidget(btn, i // 2, i % 2)
        left_layout.addWidget(plots_group)

        open_report = QPushButton("Open QC HTML report")
        open_report.setObjectName("OpenButton")
        open_report.clicked.connect(lambda: self._open_stage_file("quality", "report"))
        left_layout.addWidget(open_report)
        left_layout.addStretch(1)

        right = QGroupBox("Interpretation notes")
        right_layout = QVBoxLayout(right)
        self.quality_interpretation_box = QPlainTextEdit()
        self.quality_interpretation_box.setReadOnly(True)
        self.quality_interpretation_box.setPlainText(
            "Run Quality Control to generate artifact-family scores, warning summaries, and recording review rankings.\n\n"
            "Interpretation is descriptive: warnings identify recordings/families for review and should not be treated as automatic exclusion rules."
        )
        right_layout.addWidget(self.quality_interpretation_box)
        layout.addWidget(left, stretch=1)
        layout.addWidget(right, stretch=1)
        return panel

    def _populate_qc_feature_tree(self) -> None:
        self.qc_feature_tree.blockSignals(True)
        self.qc_feature_tree.clear()
        self.qc_family_items = {}
        self.qc_feature_items = {}
        registry = quality_feature_registry()
        for family, meta in QC_FAMILIES.items():
            fam_item = QTreeWidgetItem([meta.get("label", family), "family", ""])
            fam_item.setFlags(fam_item.flags() | Qt.ItemIsUserCheckable)
            fam_item.setCheckState(0, Qt.Checked)
            fam_item.setData(0, Qt.UserRole, {"kind": "family", "family": family})
            self.qc_feature_tree.addTopLevelItem(fam_item)
            self.qc_family_items[family] = fam_item
            sub = registry[registry["family"].eq(family)]
            for _, r in sub.iterrows():
                feat = str(r["feature"])
                role = str(r["role"])
                params = str(r["parameter_dependencies"])
                child = QTreeWidgetItem([feat, role, params])
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setCheckState(0, Qt.Checked if bool(r.get("default_selected", True)) or role == "support" else Qt.Unchecked)
                if role == "support":
                    child.setForeground(1, QBrush(QColor("#9CB3C9")))
                child.setData(0, Qt.UserRole, {"kind": "feature", "family": family, "feature": feat, "meaning": str(r["meaning"]), "params": params, "role": role})
                fam_item.addChild(child)
                self.qc_feature_items[feat] = child
            fam_item.setExpanded(False)
        self.qc_feature_tree.resizeColumnToContents(0)
        self.qc_feature_tree.blockSignals(False)

    def _on_qc_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        data = item.data(0, Qt.UserRole) or {}
        if data.get("kind") == "family":
            state = item.checkState(0)
            self.qc_feature_tree.blockSignals(True)
            for i in range(item.childCount()):
                item.child(i).setCheckState(0, state)
            self.qc_feature_tree.blockSignals(False)
        elif data.get("kind") == "feature":
            family = data.get("family")
            fam_item = self.qc_family_items.get(family)
            if fam_item:
                checked = sum(1 for i in range(fam_item.childCount()) if fam_item.child(i).checkState(0) == Qt.Checked)
                self.qc_feature_tree.blockSignals(True)
                fam_item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                self.qc_feature_tree.blockSignals(False)
        self._refresh_qc_count_label()

    def _selected_qc_features(self) -> list[str]:
        selected = []
        for feat, item in getattr(self, "qc_feature_items", {}).items():
            data = item.data(0, Qt.UserRole) or {}
            if item.checkState(0) == Qt.Checked and data.get("role") != "support":
                selected.append(feat)
        return selected

    def _selected_qc_families(self) -> list[str]:
        selected_features = set(self._selected_qc_features())
        families = []
        for family, feats in FAMILY_FEATURES.items():
            numeric = [f for f in feats if not f.endswith("_status") and not f.endswith("_flags")]
            if any(f in selected_features for f in numeric):
                families.append(family)
        return families

    def _refresh_qc_count_label(self) -> None:
        if not hasattr(self, "qc_count_label"):
            return
        n_features = len(self._selected_qc_features()) if hasattr(self, "qc_feature_items") else 0
        n_families = len(self._selected_qc_families()) if hasattr(self, "qc_feature_items") else 0
        self.qc_count_label.setText(f"Selected QC features: {n_features} across {n_families} families")
        self._update_qc_parameter_relevance()

    def _update_qc_feature_detail(self) -> None:
        if not hasattr(self, "qc_feature_detail_box"):
            return
        items = self.qc_feature_tree.selectedItems() if hasattr(self, "qc_feature_tree") else []
        if not items:
            self.qc_feature_detail_box.setPlainText("Select a QC family or feature to inspect its meaning and parameter dependencies.")
            return
        data = items[0].data(0, Qt.UserRole) or {}
        if data.get("kind") == "family":
            fam = data.get("family")
            meta = QC_FAMILIES.get(fam, {})
            text = f"Family: {meta.get('label', fam)}\n\nMeaning: {meta.get('meaning', '')}\n\nTypical use: review whether this artifact family may affect downstream acoustic features."
        else:
            text = (
                f"Feature: {data.get('feature')}\n"
                f"Family: {QC_FAMILIES.get(data.get('family'), {}).get('label', data.get('family'))}\n"
                f"Role: {data.get('role')}\n"
                f"Parameter dependencies: {data.get('params')}\n\n"
                f"Meaning: {data.get('meaning')}"
            )
        self.qc_feature_detail_box.setPlainText(text)

    def select_all_qc_features(self) -> None:
        self.qc_feature_tree.blockSignals(True)
        for item in self.qc_feature_items.values():
            item.setCheckState(0, Qt.Checked)
        for item in self.qc_family_items.values():
            item.setCheckState(0, Qt.Checked)
        self.qc_feature_tree.blockSignals(False)
        self._refresh_qc_count_label()

    def select_recommended_qc_features(self) -> None:
        registry = quality_feature_registry()
        recommended = set(registry.loc[registry["default_selected"].eq(True), "feature"].astype(str))
        self.qc_feature_tree.blockSignals(True)
        for feat, item in self.qc_feature_items.items():
            item.setCheckState(0, Qt.Checked if feat in recommended or feat.endswith("_status") or feat.endswith("_flags") else Qt.Unchecked)
        for family, fam_item in self.qc_family_items.items():
            checked = sum(1 for i in range(fam_item.childCount()) if fam_item.child(i).checkState(0) == Qt.Checked)
            fam_item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
        self.qc_feature_tree.blockSignals(False)
        self._refresh_qc_count_label()

    def clear_qc_features(self) -> None:
        self.qc_feature_tree.blockSignals(True)
        for item in self.qc_feature_items.values():
            item.setCheckState(0, Qt.Unchecked)
        for item in self.qc_family_items.values():
            item.setCheckState(0, Qt.Unchecked)
        self.qc_feature_tree.blockSignals(False)
        self._refresh_qc_count_label()



    def reset_qc_parameters(self) -> None:
        """Reset QC computation parameters to conservative v1 defaults."""
        self.qc_pause_sec_spin.setValue(0.15)
        self.qc_high_level_spin.setValue(90.0)
        self.qc_hard_clip_spin.setValue(0.995)
        self.qc_near_clip_spin.setValue(0.950)
        self._update_qc_parameter_relevance()

    def _update_qc_parameter_relevance(self) -> None:
        """Show which QC parameters are currently relevant to selected features."""
        if not hasattr(self, "qc_parameter_relevance_label") or not hasattr(self, "qc_feature_items"):
            return
        selected = self._selected_qc_features()
        registry = quality_feature_registry()
        if not selected:
            self.qc_parameter_relevance_label.setText("No QC features selected.")
            return
        deps = []
        if not registry.empty and "feature" in registry.columns:
            sub = registry[registry["feature"].astype(str).isin(selected)]
            for value in sub.get("parameter_dependencies", pd.Series(dtype=str)).astype(str):
                for part in value.split(";"):
                    part = part.strip()
                    if part and part.lower() != "none":
                        deps.append(part)
        deps = sorted(set(deps))
        if not deps:
            self.qc_parameter_relevance_label.setText("Selected QC features do not require adjustable parameters.")
        else:
            self.qc_parameter_relevance_label.setText("Active parameter dependencies: " + ", ".join(deps))

    def _build_feature_policy_table(self) -> QTableWidget:
        """Build a compact, GUI-facing feature computation policy table."""
        df = build_feature_family_policy_summary()
        table = QTableWidget()
        columns = ["family", "best_tasks", "native_scale", "default_region", "scalar_reduction", "selectable_modes", "core_rule"]
        headers = ["Feature family", "Best tasks", "Native scale", "Default region", "Scalar reduction", "Selectable modes", "Rule"]
        table.setColumnCount(len(columns))
        table.setRowCount(len(df))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setWordWrap(True)
        table.verticalHeader().setVisible(False)
        table.setMinimumHeight(260)
        table.setMaximumHeight(360)
        for r, (_, row) in enumerate(df.iterrows()):
            for c, col in enumerate(columns):
                item = QTableWidgetItem(str(row.get(col, "")))
                item.setToolTip(str(row.get(col, "")))
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.horizontalHeader().setStretchLastSection(True)
        table.setColumnWidth(0, 170)
        table.setColumnWidth(1, 210)
        table.setColumnWidth(2, 190)
        table.setColumnWidth(3, 210)
        table.setColumnWidth(4, 260)
        table.setColumnWidth(5, 230)
        return table

    def _build_features_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "Info",
            "Select feature families or individual features. Each feature has a defined native measurement scale, analysis region, and scalar-reduction rule."
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
        self.feature_tree.setHeaderLabels(["Feature / subsystem", "Status", "Tier", "Task / unit"])
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
        self.min_pause_feature_spin = QDoubleSpinBox(); self.min_pause_feature_spin.setDecimals(2); self.min_pause_feature_spin.setRange(0.0, 5.0); self.min_pause_feature_spin.setSingleStep(0.05); self.min_pause_feature_spin.setValue(0.30)
        self._set_tooltip(self.min_pause_feature_spin, "Minimum internal nonspeech duration counted as a pause and minimum phrase duration. The supplied VSLP feature protocol specifies 300 ms.")
        self.computation_mode_combo = QComboBox(); self.computation_mode_combo.addItems([
            "validated_default",
            "speech_only_concatenated",
            "effective_task_with_pauses",
            "per_segment_robust",
            "full_file_exploratory",
        ])
        self._set_tooltip(self.computation_mode_combo, "Validated default is recommended. Other modes are applied only where scientifically valid and are recorded in the scalar-reduction audit table.")
        self.region_policy_combo = QComboBox(); self.region_policy_combo.addItems(["speech_only", "effective_task", "full_file"])
        self._set_tooltip(self.region_policy_combo, "Family defaults override this where needed. This is kept as a low-level fallback for signal features.")
        form.addRow("Minimum internal pause duration (s)", self.min_pause_feature_spin)
        form.addRow("Computation mode", self.computation_mode_combo)
        form.addRow("Fallback signal region", self.region_policy_combo)
        right_layout.addWidget(param_group)

        strategy_group = QGroupBox("Computation policy")
        strategy_layout = QVBoxLayout(strategy_group)
        strategy_layout.addWidget(self._build_feature_policy_table())
        right_layout.addWidget(strategy_group)

        run_btn = QPushButton("Run Feature Extraction")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_features)
        right_layout.addWidget(run_btn)
        right_layout.addStretch(1)
        splitter.addWidget(left); splitter.addWidget(right); splitter.setSizes([720, 430]); splitter.setMinimumHeight(540)
        layout.addWidget(splitter)
        self._refresh_feature_count_label()
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
            ("Ingest summary", lambda: self.preview_csv(self._stage_path("ingest", "summary"))),
            ("Preprocess main summary", lambda: self.preview_csv(self._stage_path("preprocess", "main_summary"))),
            ("Preprocess detailed summary", lambda: self.preview_csv(self._stage_path("preprocess", "summary"))),
            ("Preprocess QC flags", lambda: self.preview_csv(self._stage_path("preprocess", "qc_flags"))),
            ("Segmentation summary", lambda: self.preview_csv(self._stage_path("segment", "summary"))),
            ("Quality features", lambda: self.preview_csv(self._stage_path("quality", "summary"))),
            ("Quality main summary", lambda: self.preview_csv(self._stage_path("quality", "main_summary"))),
            ("Quality feature registry", lambda: self.preview_csv(self._stage_path("quality", "feature_registry"))),
            ("Quality warnings", lambda: self.preview_csv(self._stage_path("quality", "warnings"))),
            ("Quality recommendations", lambda: self.preview_csv(self._stage_path("quality", "recommendations"))),
            ("Quality distribution summary", lambda: self.preview_csv(self._stage_path("quality", "distribution"))),
            ("Quality family scores", lambda: self.preview_csv(self._stage_path("quality", "family_scores"))),
            ("Quality review rank", lambda: self.preview_csv(self._stage_path("quality", "review_rank"))),
            ("Quality feature correlation", lambda: self.preview_csv(self._stage_path("quality", "feature_corr"))),
            ("Quality PCA variance", lambda: self.preview_csv(self._stage_path("quality", "pca_variance"))),
            ("Quality family status", lambda: self.preview_csv(self._stage_path("quality", "family_status"))),
            ("Feature table", lambda: self.preview_csv(self._stage_path("features", "summary"))),
            ("Feature registry", lambda: self.preview_csv(self._stage_path("features", "registry"))),
            ("Feature status", lambda: self.preview_csv(self._stage_path("features", "status"))),
            ("Feature distribution audit", lambda: self.preview_csv(self._stage_path("features", "audit"))),
            ("Feature expected-range flags", lambda: self.preview_csv(self._stage_path("features", "range_flags"))),
            ("Feature measurement scales", lambda: self.preview_csv(self._output_root() / "acoustic" / "004_features" / "tables" / "acoustic_feature_measurement_scale_registry.csv")),
            ("Feature computation policy", lambda: self.preview_csv(self._stage_path("features", "computation_policy"))),
            ("Feature scalar-reduction audit", lambda: self.preview_csv(self._stage_path("features", "reduction_audit"))),
            ("Native segment events", lambda: self.preview_csv(self._output_root() / "acoustic" / "004_features" / "tables" / "native_measurements" / "acoustic_native_segment_events.csv")),
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
            ("Preprocess SNR", lambda: self.preview_image(self._output_root() / "acoustic" / "001_preprocess" / "plots" / "preprocess_snr_distribution.png")),
            ("Preprocess clipping", lambda: self.preview_image(self._output_root() / "acoustic" / "001_preprocess" / "plots" / "preprocess_clipping_distribution.png")),
            ("Preprocess DC offset", lambda: self.preview_image(self._output_root() / "acoustic" / "001_preprocess" / "plots" / "preprocess_dc_offset_before_after.png")),
            ("Quality family status", lambda: self.preview_image(self._output_root() / "acoustic" / "003_quality_control" / "plots" / "quality_family_status.png")),
            ("Quality overview", lambda: self.preview_image(self._output_root() / "acoustic" / "003_quality_control" / "plots" / "quality_main_features_overview.png")),
            ("Quality feature distributions", lambda: self.preview_image(self._output_root() / "acoustic" / "003_quality_control" / "plots" / "quality_feature_distributions.png")),
            ("Quality warning heatmap", lambda: self.preview_image(self._output_root() / "acoustic" / "003_quality_control" / "plots" / "quality_warning_heatmap.png")),
            ("Quality recommendations", lambda: self.preview_image(self._output_root() / "acoustic" / "003_quality_control" / "plots" / "quality_recommendation_summary.png")),
            ("Quality family score distributions", lambda: self.preview_image(self._output_root() / "acoustic" / "003_quality_control" / "plots" / "quality_family_score_distributions.png")),
            ("Quality review rank", lambda: self.preview_image(self._output_root() / "acoustic" / "003_quality_control" / "plots" / "quality_recording_review_rank.png")),
            ("Quality feature correlation", lambda: self.preview_image(self._output_root() / "acoustic" / "003_quality_control" / "plots" / "quality_feature_correlation_heatmap.png")),
            ("Quality family correlation", lambda: self.preview_image(self._output_root() / "acoustic" / "003_quality_control" / "plots" / "quality_family_correlation_heatmap.png")),
            ("Quality feature coverage", lambda: self.preview_image(self._output_root() / "acoustic" / "003_quality_control" / "plots" / "quality_missingness_feature_coverage.png")),
            ("Quality PCA scree", lambda: self.preview_image(self._output_root() / "acoustic" / "003_quality_control" / "plots" / "quality_pca_scree.png")),
            ("Quality PCA embedding", lambda: self.preview_image(self._output_root() / "acoustic" / "003_quality_control" / "plots" / "quality_pca_embedding.png")),
            ("Feature missingness", lambda: self.preview_image(self._features_plot_path("feature_missingness.png"))),
            ("Feature implementation status", lambda: self.preview_image(self._features_plot_path("feature_subsystem_implementation_status.png"))),
            ("Feature distributions", lambda: self.preview_image(self._features_plot_path("implemented_feature_distributions.png"))),
            ("Feature distribution audit", lambda: self.preview_image(self._features_plot_path("feature_distribution_audit.png"))),
            ("Feature expected-range flags", lambda: self.preview_image(self._features_plot_path("feature_expected_range_flags.png"))),
            ("Feature correlation heatmap", lambda: self.preview_image(self._features_plot_path("feature_correlation_heatmap.png"))),
            ("Feature subsystem coverage", lambda: self.preview_image(self._features_plot_path("feature_subsystem_distributions.png"))),
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
        layout.setSpacing(12)
        layout.addWidget(self._info_panel(
            "Info",
            "Use Main Feature GUI Handoff for downstream analysis. Supplementary outputs preserve stage reports, diagnostics, plots, logs, and audit evidence."
        ))

        handoff_group = QGroupBox("Feature GUI handoff")
        handoff_layout = QVBoxLayout(handoff_group)
        handoff_note = QLabel(
            "Main contains the feature values, registry, per-feature status, optional QC covariates, and mapped recording context. "
            "Supplementary indexes detailed stage outputs and is intended for review and troubleshooting."
        )
        handoff_note.setWordWrap(True)
        handoff_note.setObjectName("SubtitleLabel")
        handoff_buttons = QHBoxLayout()
        open_main = QPushButton("Open Main Feature GUI Handoff")
        open_main.setObjectName("RunButton")
        open_main.clicked.connect(lambda: self._open_feature_handoff("main"))
        open_supplementary = QPushButton("Open Supplementary Outputs")
        open_supplementary.setObjectName("OpenButton")
        open_supplementary.clicked.connect(lambda: self._open_feature_handoff("supplementary"))
        handoff_buttons.addWidget(open_main)
        handoff_buttons.addWidget(open_supplementary)
        handoff_layout.addWidget(handoff_note)
        handoff_layout.addLayout(handoff_buttons)

        generate_group = QGroupBox("Run summary")
        generate_layout = QVBoxLayout(generate_group)
        summary_note = QLabel("Generate an index of available tables/reports/plots and a compact HTML summary for the current output project.")
        summary_note.setObjectName("SubtitleLabel")
        summary_note.setWordWrap(True)
        generate_btn = QPushButton("Generate Run Summary")
        generate_btn.setObjectName("RunButton")
        generate_btn.clicked.connect(self.run_generate_report_summary)
        generate_layout.addWidget(summary_note)
        generate_layout.addWidget(generate_btn)

        stage_reports = QGroupBox("Stage reports")
        stage_grid = QGridLayout(stage_reports)
        stage_buttons = [
            ("Preprocess report", lambda: self._open_stage_file("preprocess", "report")),
            ("Segmentation report", lambda: self._open_stage_file("segment", "report")),
            ("Quality Control report", lambda: self._open_stage_file("quality", "report")),
            ("Feature Extraction report", lambda: self._open_stage_file("features", "report")),
            ("Run summary report", lambda: self._open_stage_file("reports", "report")),
        ]
        for i, (label, callback) in enumerate(stage_buttons):
            btn = QPushButton(label); btn.setObjectName("OpenButton"); btn.clicked.connect(callback)
            stage_grid.addWidget(btn, i // 2, i % 2)

        primary_tables = QGroupBox("Stage and audit tables (supplementary)")
        table_grid = QGridLayout(primary_tables)
        table_buttons = [
            ("Ingest summary", lambda: self._open_stage_file("ingest", "summary")),
            ("Preprocess summary", lambda: self._open_stage_file("preprocess", "main_summary")),
            ("Segmentation summary", lambda: self._open_stage_file("segment", "main_summary")),
            ("Quality main summary", lambda: self._open_stage_file("quality", "main_summary")),
            ("Quality recommendations", lambda: self._open_stage_file("quality", "recommendations")),
            ("Feature table", lambda: self._open_stage_file("features", "summary")),
            ("Feature status", lambda: self._open_stage_file("features", "status")),
            ("Feature scalar-reduction audit", lambda: self._open_stage_file("features", "reduction_audit")),
            ("Run output manifest", lambda: self._open_stage_file("reports", "summary")),
        ]
        for i, (label, callback) in enumerate(table_buttons):
            btn = QPushButton(label); btn.setObjectName("OpenButton"); btn.clicked.connect(callback)
            table_grid.addWidget(btn, i // 2, i % 2)

        folders = QGroupBox("Folders")
        folder_layout = QHBoxLayout(folders)
        open_acoustic = QPushButton("Open Acoustic Output Folder")
        open_acoustic.setObjectName("OpenButton")
        open_acoustic.clicked.connect(self.open_output_root)
        open_plots = QPushButton("Open Plots Folder")
        open_plots.setObjectName("OpenButton")
        open_plots.clicked.connect(lambda: open_path(self._output_root() / "acoustic"))
        folder_layout.addWidget(open_acoustic)
        folder_layout.addWidget(open_plots)

        layout.addWidget(handoff_group)
        layout.addWidget(generate_group)
        layout.addWidget(stage_reports)
        layout.addWidget(primary_tables)
        layout.addWidget(folders)
        layout.addStretch(1)
        return self._scrollable(container)

    def _open_feature_handoff(self, section: str) -> None:
        path = self._output_root() / "acoustic" / "feature_handoff" / section
        if not path.exists():
            QMessageBox.information(self, "Handoff not available", "Run Acoustic Feature Extraction first.")
            return
        open_path(path)

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
                tier = str(row.get("evidence_tier", ""))
                unit = str(row.get("unit", ""))
                task = str(row.get("task_scope", ""))
                task_unit = f"{task} | {unit}" if task and unit else (task or unit)
                item = QTreeWidgetItem([feature, status, tier, task_unit])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked)
                item.setData(0, Qt.UserRole, feature)
                item.setToolTip(0, str(row.get("meaning", "")))
                item.setToolTip(1, "implemented = computed; proxy = engineering estimate needing validation; pending = explicit NaN placeholder")
                item.setToolTip(2, "Evidence tier from the feature map: A established, B strong, C inconsistent/exploratory, D low priority.")
                item.setToolTip(3, "Task compatibility and units.")
                if status == "implemented":
                    color = QColor("#8EF2C6")
                elif status == "proxy":
                    color = QColor("#FFD98E")
                else:
                    color = QColor("#8FA9BD")
                tier_color = {"A": QColor("#8EF2C6"), "B": QColor("#5FD3C4"), "C": QColor("#FFD98E"), "D": QColor("#D77A6A")}.get(tier, QColor("#8FA9BD"))
                item.setForeground(1, QBrush(color))
                item.setForeground(2, QBrush(tier_color))
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
        if self._run_root is None:
            raise RuntimeError("Initialize an acoustic run before accessing stage outputs.")
        return self._run_root

    def _project_manifest_path(self) -> Path:
        return self._output_root() / "project_manifest.json"

    def _is_project_initialized(self) -> bool:
        if self._run_root is None:
            return False
        try:
            manifest = json.loads(self._project_manifest_path().read_text(encoding="utf-8"))
            return (
                manifest.get("modality") == "acoustic"
                and manifest.get("run_root") == str(self._run_root)
                and manifest.get("project_name") == self.project_name_edit.text().strip()
                and manifest.get("task_name") == self.task_name_edit.text().strip()
                and (self._run_root / "acoustic").is_dir()
            )
        except (OSError, ValueError, TypeError):
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
        errors_path = self._output_root() / "acoustic" / "000_ingest" / "errors" / "audio_ingest_errors.csv"
        duplicates_path = self._output_root() / "acoustic" / "000_ingest" / "tables" / "audio_ingest_skipped_duplicates.csv"
        if not summary_path.exists():
            self.ingest_summary_label.setText(
                "No ingest results yet."
            )
            self.ingest_format_table.setRowCount(0)
            return
        try:
            df = pd.read_csv(summary_path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
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
            f"Discovered: {n_ok + n_failed + duplicate_skipped_count} | Accepted: {n_ok} | Duplicate-skipped: {duplicate_skipped_count} | Failed: {n_failed} | Subfolders represented: {subfolder_count}\n"
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
        return self._output_root() / "acoustic" / "001_preprocess" / "tables" / "acoustic_preprocess_summary.csv"

    def _segmentation_summary_path(self) -> Path:
        return self._output_root() / "acoustic" / "002_segmentation" / "tables" / "acoustic_segmentation_summary.csv"

    def _stage_path(self, stage: str, kind: str) -> Path:
        mapping = {
            ("ingest", "summary"): self._output_root() / "acoustic" / "000_ingest" / "tables" / "audio_ingest_summary.csv",
            ("preprocess", "summary"): self._preprocess_summary_path(),
            ("preprocess", "main_summary"): self._output_root() / "acoustic" / "001_preprocess" / "tables" / "acoustic_preprocess_main_summary.csv",
            ("preprocess", "report"): self._output_root() / "acoustic" / "001_preprocess" / "reports" / "acoustic_preprocess_report.html",
            ("segment", "summary"): self._segmentation_summary_path(),
            ("segment", "main_summary"): self._output_root() / "acoustic" / "002_segmentation" / "tables" / "acoustic_segmentation_main_summary.csv",
            ("segment", "report"): self._output_root() / "acoustic" / "002_segmentation" / "reports" / "acoustic_segmentation_report.html",
            ("quality", "summary"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_features.csv",
            ("quality", "main_summary"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_main_summary.csv",
            ("quality", "family_status"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_family_status.csv",
            ("quality", "family_summary"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_family_summary.csv",
            ("quality", "feature_registry"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_feature_registry.csv",
            ("quality", "warnings"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_warnings.csv",
            ("quality", "recommendations"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_recommendations.csv",
            ("quality", "distribution"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_distribution_summary.csv",
            ("quality", "processed"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_processed_features.csv",
            ("quality", "family_scores"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_family_scores.csv",
            ("quality", "feature_corr"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_feature_spearman_correlation.csv",
            ("quality", "family_corr"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_family_spearman_correlation.csv",
            ("quality", "review_rank"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_recording_review_rank.csv",
            ("quality", "pca_variance"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_pca_variance.csv",
            ("quality", "pca_scores"): self._output_root() / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_pca_scores.csv",
            ("quality", "report"): self._output_root() / "acoustic" / "003_quality_control" / "reports" / "acoustic_quality_control_report.html",
            ("features", "summary"): self._output_root() / "acoustic" / "004_features" / "tables" / "acoustic_features_per_file.csv",
            ("features", "report"): self._output_root() / "acoustic" / "004_features" / "reports" / "acoustic_feature_report.html",
            ("reports", "summary"): self._output_root() / "acoustic" / "005_run_summary" / "tables" / "vslp_acoustic_output_manifest.csv",
            ("reports", "report"): self._output_root() / "acoustic" / "005_run_summary" / "reports" / "vslp_acoustic_run_summary.html",
            ("features", "registry"): self._output_root() / "acoustic" / "004_features" / "tables" / "selected_acoustic_feature_registry.csv",
            ("features", "status"): self._output_root() / "acoustic" / "004_features" / "tables" / "acoustic_feature_status_long.csv",
            ("features", "audit"): self._output_root() / "acoustic" / "004_features" / "tables" / "acoustic_feature_distribution_audit.csv",
            ("features", "range_flags"): self._output_root() / "acoustic" / "004_features" / "tables" / "acoustic_feature_expected_range_flags.csv",
            ("features", "computation_policy"): self._output_root() / "acoustic" / "004_features" / "tables" / "acoustic_feature_computation_policy.csv",
            ("features", "reduction_audit"): self._output_root() / "acoustic" / "004_features" / "tables" / "acoustic_feature_scalar_reduction_audit.csv",
        }
        return mapping[(stage, kind)]

    def _features_plot_path(self, name: str) -> Path:
        return self._output_root() / "acoustic" / "004_features" / "plots" / name

    def refresh_latest_outputs(self) -> None:
        if self._run_root is None:
            QMessageBox.information(self, "No run folder", "Initialize an acoustic run first.")
            return
        if self._project_manifest_path().exists():
            self.stage_records["project"].status = "detected"
        known = {
            "ingest": (self._stage_path("ingest", "summary"), None),
            "preprocess": (self._stage_path("preprocess", "summary"), self._stage_path("preprocess", "report")),
            "segment": (self._stage_path("segment", "summary"), self._stage_path("segment", "report")),
            "quality": (self._stage_path("quality", "summary"), self._stage_path("quality", "report")),
            "features": (self._stage_path("features", "summary"), self._stage_path("features", "report")),
            "reports": (self._stage_path("reports", "summary"), self._stage_path("reports", "report")),
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
        plot_dir = self._output_root() / "acoustic" / "002_segmentation" / "plots"
        plots = sorted(plot_dir.rglob("*.png")) if plot_dir.exists() else []
        if not plots:
            QMessageBox.information(self, "No plots", f"No segmentation plots found in:\n{plot_dir}")
            return
        self.preview_image(plots[0])

    def _update_preprocess_feedback(self) -> None:
        if not hasattr(self, "preprocess_feedback"):
            return
        main_path = self._stage_path("preprocess", "main_summary")
        if not main_path.exists():
            self.preprocess_feedback.setText("Not run")
            return
        try:
            df = self._read_csv_safe(main_path)
            if df.empty:
                self.preprocess_feedback.setText("No preprocessing results")
                return
            status = df.get("status", pd.Series("", index=df.index)).astype(str)
            n_ok = int(status.eq("ok").sum())
            n_review = int(status.eq("needs_channel_review").sum())
            n_failed = int(status.eq("failed").sum())
            self.preprocess_feedback.setText(
                f"Processed: {n_ok}   |   Review: {n_review}   |   Failed: {n_failed}"
            )
        except Exception as exc:  # noqa: BLE001
            self.preprocess_feedback.setText(f"Could not read preprocessing results: {exc}")

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
        if not main_path.exists():
            self.segmentation_feedback.setPlainText("Not run")
            return
        try:
            df = self._read_csv_safe(main_path)
            status = df.get("automatic_status", pd.Series(dtype=str)).astype(str).str.upper()
            counts = {name: int(status.eq(name).sum()) for name in ("ACCEPTED", "REVIEW", "EXCLUDED", "FAILED")}
            self.segmentation_feedback.setPlainText(
                "  |  ".join(f"{name.title()}: {count}" for name, count in counts.items())
            )
        except Exception as exc:  # noqa: BLE001
            self.segmentation_feedback.setPlainText(f"Could not read segmentation outputs: {exc}")

    def append_log(self, text: str) -> None:
        self.log_box.appendPlainText(text)

    def _require_paths(self) -> tuple[Path, Path] | None:
        if self._run_root is None:
            QMessageBox.warning(self, "Initialize run first", "Complete Setup and initialize an acoustic run first.")
            return None
        return Path(self.input_edit.text().strip()).expanduser(), self._run_root

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
            if self._run_root is not None:
                for btn in (self.browse_input_btn, self.browse_output_btn, self.init_project_btn):
                    btn.setEnabled(False)

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
        if self._run_root is None:
            QMessageBox.information(self, "No run folder", "Initialize an acoustic run first.")
            return
        open_path(self._run_root)

    # ---------------------------- ACTIONS ----------------------------
    def _freeze_setup(self) -> None:
        for field in (self.project_name_edit, self.task_name_edit, self.input_edit, self.output_edit):
            field.setReadOnly(True)
        for button in (self.browse_input_btn, self.browse_output_btn, self.init_project_btn):
            button.setEnabled(False)

    def run_project_init(self) -> None:
        if self._run_root is not None:
            QMessageBox.information(self, "Run already initialized", "Open a new Acoustic GUI window to create a different run.")
            return
        raw_values = {
            "project_name": self.project_name_edit.text(),
            "task_name": self.task_name_edit.text(),
            "input_folder": self.input_edit.text(),
            "output_parent": self.output_edit.text(),
        }
        project_name = raw_values["project_name"].strip()
        task_name = raw_values["task_name"].strip()
        input_text = raw_values["input_folder"].strip()
        parent_text = raw_values["output_parent"].strip()
        if not all((project_name, task_name, input_text, parent_text)):
            QMessageBox.warning(self, "Incomplete Setup", "Project name, Task name, Input folder, and Output parent folder are all required.")
            return
        input_path = Path(input_text).expanduser().resolve()
        if not input_path.is_dir():
            QMessageBox.warning(self, "Input folder missing", "Select an existing input audio folder.")
            return
        try:
            task_run_folder_name(task_name)
            output_parent = Path(parent_text).expanduser().resolve()
            if output_parent.is_relative_to(input_path):
                raise ValueError("Output parent must be outside the input folder.")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid Setup", str(exc))
            return
        self._run_worker("project", initialize_acoustic_run, {
            "project_name": project_name,
            "task_name": task_name,
            "input_folder": input_path,
            "output_parent": output_parent,
            "setup_values": raw_values,
            "gui_version": "0.38",
            "pipeline_version": "0.38.0",
        })

    def run_ingest(self) -> None:
        paths = self._require_paths()
        if paths is None: return
        if not self._require_project_initialized(): return
        input_path, output_root = paths
        self._run_worker("ingest", run_acoustic_ingest, {"input_path": input_path, "output_root": output_root})

    def _preprocess_config_from_gui(self) -> PreprocessConfig:
        return PreprocessConfig(
            remove_dc_offset=bool(self.remove_dc_check.isChecked()),
        )

    def run_preprocess(self) -> None:
        paths = self._require_paths()
        if paths is None:
            return
        if not self._require_project_initialized():
            return
        _input_path, output_root = paths
        ingest_summary = self._stage_path("ingest", "summary")
        if not ingest_summary.exists():
            QMessageBox.warning(self, "Ingest required", "Run Ingest first.")
            return
        self._run_worker(
            "preprocess",
            run_acoustic_preprocess,
            {
                "ingest_summary_csv": ingest_summary,
                "output_root": output_root,
                "config": self._preprocess_config_from_gui(),
            },
        )

    def run_segmentation(self) -> None:
        paths = self._require_paths()
        if paths is None or not self._require_project_initialized(): return
        if self.segmentation_method_combo.currentData() == CUSTOM:
            QMessageBox.information(self, "Plugin required", "Install a segmentation plugin before selecting Custom.")
            return
        _input_path, output_root = paths
        preprocess_summary = self._preprocess_summary_path()
        if not preprocess_summary.exists():
            QMessageBox.warning(self, "Preprocess required", "Run preprocessing first.")
            return
        self._run_worker("segment", run_acoustic_segmentation, {
            "preprocess_summary_csv": preprocess_summary, "output_root": output_root,
            "config": self._segmentation_config_from_gui(),
        })

    def _update_quality_feedback(self) -> None:
        if not hasattr(self, "quality_summary_label"):
            return
        family_status_path = self._stage_path("quality", "family_status")
        warnings_path = self._stage_path("quality", "warnings")
        recommendations_path = self._stage_path("quality", "recommendations")
        review_path = self._stage_path("quality", "review_rank")
        if not family_status_path.exists():
            self.quality_summary_label.setText("Quality Control has not been run.")
            if hasattr(self, "quality_interpretation_box"):
                self.quality_interpretation_box.setPlainText(
                    "Run Quality Control to generate artifact-family scores, warning summaries, and recording review rankings.\n\n"
                    "Interpretation is descriptive: warnings identify recordings/families for review and should not be treated as automatic exclusion rules."
                )
            return
        try:
            df = pd.read_csv(family_status_path)
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

        n_warning_rows = 0
        n_flagged_files = 0
        top_families = "none"
        if warnings_path.exists():
            try:
                wdf = pd.read_csv(warnings_path)
                n_warning_rows = len(wdf)
                if not wdf.empty:
                    n_flagged_files = wdf["file_name"].nunique() if "file_name" in wdf.columns else 0
                    if "family" in wdf.columns:
                        fam_counts = wdf["family"].astype(str).value_counts().head(3)
                        top_families = ", ".join(f"{k} ({v})" for k, v in fam_counts.items())
            except Exception:
                pass

        n_rec = 0
        rec_families = "none"
        if recommendations_path.exists():
            try:
                rdf = pd.read_csv(recommendations_path)
                n_rec = len(rdf)
                if not rdf.empty and "family" in rdf.columns:
                    rec_families = ", ".join(rdf["family"].astype(str).head(4).tolist())
            except Exception:
                pass

        top_review = "not available"
        if review_path.exists():
            try:
                rnk = pd.read_csv(review_path)
                if not rnk.empty:
                    top_names = rnk["file_name"].astype(str).head(3).tolist() if "file_name" in rnk.columns else []
                    top_review = ", ".join(top_names) if top_names else "not available"
            except Exception:
                pass

        self.quality_summary_label.setText(
            f"Files evaluated: {n_files}\n"
            f"Selected family evaluations: {selected}\n"
            f"Computed/proxy evaluations: {computed}\n"
            f"Failed evaluations: {failed}\n"
            f"Warning rows: {n_warning_rows} across {n_flagged_files} recordings"
        )
        if hasattr(self, "quality_interpretation_box"):
            self.quality_interpretation_box.setPlainText(
                "Dataset-level QC snapshot\n"
                f"• Files evaluated: {n_files}\n"
                f"• Recordings with QC warning rows: {n_flagged_files}\n"
                f"• Most frequent warning families: {top_families}\n"
                f"• Recommendation families: {rec_families}\n"
                f"• Highest-priority recordings for review: {top_review}\n\n"
                "Use the plots in this tab to inspect artifact burden distributions, warning heatmaps, and QC feature correlation structure. These outputs are descriptive screening tools and should guide review/sensitivity analysis, not automatic exclusion."
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
        selected_features = self._selected_qc_features()
        selected_families = self._selected_qc_families()
        if not selected_features or not selected_families:
            QMessageBox.warning(self, "No QC features selected", "Select at least one QC feature.")
            return
        cfg = QualityControlConfig(
            selected_families=selected_families,
            selected_features=selected_features,
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
        cfg = FeatureExtractionConfig(
            selected_features=selected_features,
            minimum_pause_duration_sec=float(self.min_pause_feature_spin.value()),
            acoustic_region_policy=self.region_policy_combo.currentText(),
            computation_mode=self.computation_mode_combo.currentText(),
        )
        self._run_worker("features", run_acoustic_feature_extraction, {"segmentation_summary_csv": segmentation_summary, "output_root": output_root, "config": cfg})


    def run_generate_report_summary(self) -> None:
        paths = self._require_paths()
        if paths is None: return
        if not self._require_project_initialized(): return
        _input_path, output_root = paths
        self._run_worker("reports", self._write_run_summary_files, {"output_root": output_root})

    def _write_run_summary_files(self, output_root: str | Path) -> StageResult:
        output_root = Path(output_root)
        stage_dir = output_root / "acoustic" / "005_run_summary"
        table_dir = stage_dir / "tables"
        report_dir = stage_dir / "reports"
        manifest_dir = stage_dir / "logs"
        table_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)

        artifacts = [
            ("ingest", "audio_ingest_summary", self._stage_path("ingest", "summary"), "table"),
            ("preprocess", "preprocess_main_summary", self._stage_path("preprocess", "main_summary"), "table"),
            ("preprocess", "preprocess_report", self._stage_path("preprocess", "report"), "report"),
            ("segmentation", "segmentation_main_summary", self._stage_path("segment", "main_summary"), "table"),
            ("segmentation", "segmentation_report", self._stage_path("segment", "report"), "report"),
            ("quality_control", "quality_main_summary", self._stage_path("quality", "main_summary"), "table"),
            ("quality_control", "quality_warnings", self._stage_path("quality", "warnings"), "table"),
            ("quality_control", "quality_recommendations", self._stage_path("quality", "recommendations"), "table"),
            ("quality_control", "quality_report", self._stage_path("quality", "report"), "report"),
            ("feature_extraction", "feature_table", self._stage_path("features", "summary"), "table"),
            ("feature_extraction", "feature_status", self._stage_path("features", "status"), "table"),
            ("feature_extraction", "feature_reduction_audit", self._stage_path("features", "reduction_audit"), "table"),
            ("feature_extraction", "feature_report", self._stage_path("features", "report"), "report"),
        ]
        rows = []
        for stage, name, path, kind in artifacts:
            rows.append({
                "stage": stage,
                "artifact": name,
                "kind": kind,
                "exists": bool(path.exists()),
                "path": str(path),
            })
        manifest_csv = table_dir / "vslp_acoustic_output_manifest.csv"
        pd.DataFrame(rows).to_csv(manifest_csv, index=False)

        def csv_count(path: Path) -> str:
            if not path.exists():
                return "not available"
            try:
                return str(len(pd.read_csv(path)))
            except Exception:
                return "available"

        task_value = "not specified"
        project_value = "not specified"
        try:
            manifest = json.loads((output_root / "project_manifest.json").read_text(encoding="utf-8"))
            project_value = manifest.get("project_name") or "not specified"
            task_value = manifest.get("task_name") or "not specified"
        except (OSError, ValueError, TypeError):
            pass
        metrics = [
            ("Project name", project_value),
            ("Primary task", task_value),
            ("Ingested files", csv_count(self._stage_path("ingest", "summary"))),
            ("Segmented files", csv_count(self._stage_path("segment", "main_summary"))),
            ("QC feature rows", csv_count(self._stage_path("quality", "summary"))),
            ("Feature rows", csv_count(self._stage_path("features", "summary"))),
        ]
        available = sum(1 for r in rows if r["exists"])
        total = len(rows)
        report_path = report_dir / "vslp_acoustic_run_summary.html"
        metric_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics)
        artifact_rows = "".join(
            f"<tr><td>{r['stage']}</td><td>{r['artifact']}</td><td>{r['kind']}</td><td>{'yes' if r['exists'] else 'no'}</td><td><code>{r['path']}</code></td></tr>"
            for r in rows
        )
        report_path.write_text(f"""
<!doctype html><html><head><meta charset='utf-8'><title>VSLP Acoustic Run Summary</title>
<style>
body{{font-family:Arial, sans-serif;background:#0b1320;color:#e8eef8;margin:32px;}}
h1,h2{{color:#ffffff}} .card{{background:#111d2e;border:1px solid #26364d;border-radius:12px;padding:18px;margin:16px 0;}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #26364d;padding:8px;text-align:left;vertical-align:top}}th{{color:#9bc7ff}}code{{color:#8de0d2;word-break:break-all}}
.ok{{color:#8ef2c6}} .warn{{color:#ffd98e}}
</style></head><body>
<h1>VSLP Acoustic Run Summary</h1>
<div class='card'><p>This report indexes the outputs produced by the current acoustic pipeline run.</p>
<p><b>Available artifacts:</b> <span class='ok'>{available}</span> / {total}</p></div>
<div class='card'><h2>Stage counts</h2><table><tr><th>Item</th><th>Count/status</th></tr>{metric_rows}</table></div>
<div class='card'><h2>Output manifest</h2><table><tr><th>Stage</th><th>Artifact</th><th>Kind</th><th>Exists</th><th>Path</th></tr>{artifact_rows}</table></div>
</body></html>
""", encoding="utf-8")
        manifest = StageManifest(
            stage_name="acoustic_run_summary",
            stage_version="0.38.0",
            status="completed",
            output_artifacts=[
                ArtifactRef(path=str(manifest_csv), role="output_manifest", media_type="text/csv"),
                ArtifactRef(path=str(report_path), role="run_summary_report", media_type="text/html"),
            ],
            notes=["Run summary indexes existing outputs. It does not compute new QC decisions."],
        )
        manifest_path = manifest.write_json(manifest_dir / "stage_manifest.json")
        return StageResult(status="completed", manifest_path=manifest_path, summary_table=manifest_csv, report_path=report_path)

    def run_all(self) -> None:
        paths = self._require_paths()
        if paths is None: return
        if not self._require_project_initialized(): return
        input_path, output_root = paths

        selected_features = self._selected_feature_names()
        segmentation_config = self._segmentation_config_from_gui()
        if segmentation_config.method == CUSTOM:
            QMessageBox.information(self, "Plugin required", "Install a segmentation plugin before selecting Custom.")
            return
        def full_run(input_path: Path, output_root: Path):
            ingest_result = run_acoustic_ingest(input_path=input_path, output_root=output_root)
            if ingest_result.summary_table is None:
                raise RuntimeError("Ingest completed without a summary table; preprocessing cannot proceed")
            preprocess_result = run_acoustic_preprocess(
                ingest_summary_csv=ingest_result.summary_table,
                output_root=output_root,
                config=self._preprocess_config_from_gui(),
            )
            segment_result = run_acoustic_segmentation(
                preprocess_summary_csv=output_root / "acoustic" / "001_preprocess" / "tables" / "acoustic_preprocess_summary.csv",
                output_root=output_root,
                config=segmentation_config,
            )
            quality_result = run_acoustic_quality_control(
                segmentation_summary_csv=output_root / "acoustic" / "002_segmentation" / "tables" / "acoustic_segmentation_summary.csv",
                output_root=output_root,
                config=QualityControlConfig(
                    selected_families=self._selected_qc_families(),
                    selected_features=self._selected_qc_features(),
                    minimum_internal_pause_sec=float(self.qc_pause_sec_spin.value()),
                    high_level_percentile=float(self.qc_high_level_spin.value()),
                    hard_clip_threshold=float(self.qc_hard_clip_spin.value()),
                    near_clip_threshold=float(self.qc_near_clip_spin.value()),
                ),
            )
            features_result = run_acoustic_feature_extraction(
                segmentation_summary_csv=output_root / "acoustic" / "002_segmentation" / "tables" / "acoustic_segmentation_summary.csv",
                output_root=output_root,
                config=FeatureExtractionConfig(
                    selected_features=selected_features,
                    minimum_pause_duration_sec=float(self.min_pause_feature_spin.value()),
                            acoustic_region_policy=self.region_policy_combo.currentText(),
                    computation_mode=self.computation_mode_combo.currentText(),
                ),
            )
            reports_result = self._write_run_summary_files(output_root)
            return {"ingest": ingest_result, "preprocess": preprocess_result, "segment": segment_result, "quality": quality_result, "features": features_result, "reports": reports_result}
        self._run_worker("full_run", full_run, {"input_path": input_path, "output_root": output_root})

    def _on_worker_finished(self, name: str, result: object) -> None:  # type: ignore[override]
        if name == "full_run" and isinstance(result, dict):
            for stage_name in ["ingest", "preprocess", "segment", "quality", "features", "reports"]:
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
            self._run_root = result.root
            self.run_root_edit.setText(str(result.root))
            self._freeze_setup()
            self.append_log(f"Active acoustic run folder: {result.root}")
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
