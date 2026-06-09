"""VSLP Kinematics Pipeline GUI v0.61.

This GUI intentionally mirrors the acoustic pipeline layout: left stage sidebar,
institutional branding strip, top tabs, run log, and compact scientific workflow
panels. Heavy MediaPipe extraction and final feature computation are connected in
later patches; this pass establishes the production-quality outline and user flow.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

try:
    from PySide6.QtCore import QPointF, QObject, QThread, Qt, Signal
    from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QFileDialog,
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
        QSpinBox,
        QDoubleSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PySide6 is required to launch the kinematics GUI. Install with: pip install -e '.[gui]'"
    ) from exc

from vslp.analysis.kinematics import (
    AGGREGATION_PROFILES,
    LANDMARK_PRESETS,
    NORMALIZATION_METHODS,
    LandmarkRunConfig,
    VideoIngestConfig,
    bootstrap_mediapipe_runtime,
    download_default_model,
    run_mediapipe_landmarks,
    verify_mediapipe_runtime,
    mediapipe_environment_status,
    link_metadata,
    mediapipe_capability_note,
    run_ingest,
    write_landmark_plan,
    write_normalization_config,
    write_scaffold_report,
)
from vslp.analysis.kinematics.schemas import DEFAULT_VIDEO_EXTENSIONS, parse_int_list

APP_VERSION = "v0.61"
BRAND_DIR = Path(__file__).resolve().parent / "assets" / "branding"
LAB_LOGO = BRAND_DIR / "lab_logo.png"
UOFT_LOGO = BRAND_DIR / "uoft_logo.png"


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
            import traceback

            self.failed.emit(self.name, f"{exc}\n\n{traceback.format_exc()}")
            return
        self.message.emit(f"Finished {self.name}.")
        self.finished.emit(self.name, result)


class LandmarkMeshCanvas(QWidget):
    """Interactive 2D landmark selector for MediaPipe face-mesh points.

    The canvas displays either median x/y positions from an extracted landmark CSV
    or a synthetic face-like fallback layout. Users can click points to toggle
    inclusion in the active landmark subset. This is intentionally visual and
    auditable: the text box remains the source of truth, and the canvas simply
    makes the point set understandable.
    """

    selection_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(520, 520)
        self.setMouseTracking(True)
        self.points: dict[int, tuple[float, float]] = self._synthetic_points(478)
        self.selected: set[int] = set()
        self.hover_idx: int | None = None
        self.show_all_labels = False
        self.source_label = "Generic MediaPipe face-mesh template"

    @staticmethod
    def _synthetic_points(n: int) -> dict[int, tuple[float, float]]:
        import math
        pts: dict[int, tuple[float, float]] = {}
        for i in range(n):
            # Deterministic face-like oval fallback. This is not anatomical ground truth;
            # it simply prevents a blank canvas before real landmark CSVs exist.
            ring = i % 37
            band = (i // 37) % 13
            theta = 2.0 * math.pi * ring / 37.0
            rx = 0.18 + 0.26 * (band / 12.0)
            ry = 0.12 + 0.34 * (band / 12.0)
            x = 0.50 + rx * math.cos(theta) * (0.85 + 0.15 * math.sin(band))
            y = 0.50 + ry * math.sin(theta)
            pts[i] = (float(min(max(x, 0.04), 0.96)), float(min(max(y, 0.04), 0.96)))
        return pts

    def set_points(self, points: dict[int, tuple[float, float]], source_label: str) -> None:
        if points:
            self.points = points
            self.source_label = source_label
            self.update()

    def set_selected(self, indices: list[int] | tuple[int, ...] | set[int]) -> None:
        self.selected = {int(i) for i in indices if int(i) in self.points}
        self.selection_changed.emit(", ".join(map(str, sorted(self.selected))))
        self.update()

    def selected_text(self) -> str:
        return ", ".join(map(str, sorted(self.selected)))

    def _plot_rect(self):
        margin = 36
        return margin, margin + 18, self.width() - 2 * margin, self.height() - 2 * margin - 40

    def _to_screen(self, x: float, y: float) -> QPointF:
        left, top, w, h = self._plot_rect()
        return QPointF(left + x * w, top + y * h)

    def _nearest(self, pos) -> tuple[int | None, float]:
        best_idx = None
        best_d = 1e9
        for idx, (x, y) in self.points.items():
            p = self._to_screen(x, y)
            d = ((p.x() - pos.x()) ** 2 + (p.y() - pos.y()) ** 2) ** 0.5
            if d < best_d:
                best_idx, best_d = idx, d
        return best_idx, best_d

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt override
        idx, d = self._nearest(event.position())
        self.hover_idx = idx if d <= 12 else None
        if self.hover_idx is not None:
            self.setToolTip(f"Landmark {self.hover_idx}: click to toggle selection")
        self.update()

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if event.button() != Qt.LeftButton:
            return
        idx, d = self._nearest(event.position())
        if idx is not None and d <= 14:
            if idx in self.selected:
                self.selected.remove(idx)
            else:
                self.selected.add(idx)
            self.selection_changed.emit(self.selected_text())
            self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#F7FAFD"))
        left, top, w, h = self._plot_rect()
        painter.setPen(QPen(QColor("#C8D6E4"), 1))
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.drawRoundedRect(left, top, w, h, 12, 12)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#0B2740"))
        painter.drawText(18, 22, "Interactive MediaPipe face-mesh selector")
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#3E5B73"))
        painter.drawText(18, self.height() - 12, f"Source: {self.source_label} | selected: {len(self.selected)} | click points to add/remove")

        # Draw coarse facial orientation guides.
        painter.setPen(QPen(QColor("#D7E2EC"), 1, Qt.DashLine))
        painter.drawLine(int(left + 0.5 * w), top + 8, int(left + 0.5 * w), top + h - 8)
        painter.drawLine(left + 8, int(top + 0.5 * h), left + w - 8, int(top + 0.5 * h))

        # All landmarks.
        painter.setPen(Qt.NoPen)
        for idx, (x, y) in self.points.items():
            p = self._to_screen(x, y)
            if idx in self.selected:
                continue
            painter.setBrush(QBrush(QColor("#7B8FA3") if idx != self.hover_idx else QColor("#F3B13B")))
            radius = 2.2 if idx != self.hover_idx else 4.4
            painter.drawEllipse(p, radius, radius)

        # Selected landmarks on top.
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        for idx in sorted(self.selected):
            x, y = self.points[idx]
            p = self._to_screen(x, y)
            painter.setPen(QPen(QColor("#FFFFFF"), 1))
            painter.setBrush(QBrush(QColor("#0E9F6E")))
            painter.drawEllipse(p, 6.2, 6.2)
            painter.setPen(QColor("#063221"))
            painter.drawText(int(p.x() + 7), int(p.y() - 7), str(idx))

        if self.hover_idx is not None and self.hover_idx not in self.selected:
            x, y = self.points[self.hover_idx]
            p = self._to_screen(x, y)
            painter.setPen(QPen(QColor("#B66B00"), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(p, 8.5, 8.5)
            painter.setPen(QColor("#723B00"))
            painter.drawText(int(p.x() + 8), int(p.y() - 8), str(self.hover_idx))


class KinematicsPipelineWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VSLP - Kinematics Pipeline")
        self.resize(1480, 900)
        self.setMinimumSize(1180, 760)

        self.input_root: Path | None = None
        self.output_root: Path | None = None
        self.ingest_manifest_csv: Path | None = None
        self.metadata_path: Path | None = None
        self.last_report_html: Path | None = None
        self.landmark_indices: tuple[int, ...] = LANDMARK_PRESETS["ALS oral-motor core 15"]
        self._thread: QThread | None = None
        self._worker: Worker | None = None

        self.stage_records: dict[str, StageRecord] = {
            "project": StageRecord(),
            "metadata": StageRecord(),
            "ingest": StageRecord(),
            "landmarks": StageRecord(),
            "selection": StageRecord(),
            "normalization": StageRecord(),
            "qc": StageRecord(),
            "features": StageRecord(),
            "aggregation": StageRecord(),
            "reports": StageRecord(),
        }
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
        subtitle = QLabel(f"Kinematics Pipeline GUI {APP_VERSION}")
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
            ("landmarks", "Face Landmarks"),
            ("selection", "Landmark Selection"),
            ("normalization", "Normalization"),
            ("qc", "Video QC"),
            ("features", "Feature Computation"),
            ("aggregation", "Temporal Aggregation"),
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

        self.run_all_btn = QPushButton("Run Full Kinematics Workflow")
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
        main_col.addWidget(self._build_branding_bar())

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.ElideRight)
        self.tabs.addTab(self._build_setup_tab(), "Setup")
        self.tabs.addTab(self._build_metadata_tab(), "Metadata")
        self.tabs.addTab(self._build_landmarks_tab(), "Landmarks")
        self.tabs.addTab(self._build_selection_tab(), "Landmark Selection")
        self.tabs.addTab(self._build_normalization_tab(), "Normalization")
        self.tabs.addTab(self._build_qc_tab(), "Video QC")
        self.tabs.addTab(self._build_features_tab(), "Features")
        self.tabs.addTab(self._build_aggregation_tab(), "Aggregation")
        self.tabs.addTab(self._build_inspector_tab(), "Inspector")
        self.tabs.addTab(self._build_reports_tab(), "Reports & Outputs")
        main_col.addWidget(self.tabs, stretch=1)

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
        bar = QFrame()
        bar.setObjectName("BrandingBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)
        label = QLabel("VSLP Kinematics Pipeline")
        label.setObjectName("BrandingTitle")
        layout.addWidget(label)
        layout.addStretch(1)
        layout.addWidget(self._logo_or_text(LAB_LOGO, "Speech Production Lab", max_width=210, max_height=56))
        layout.addWidget(self._logo_or_text(UOFT_LOGO, "University of Toronto", max_width=250, max_height=54))
        bar.setMaximumHeight(76)
        return bar

    def _logo_or_text(self, logo_path: Path, fallback: str, max_width: int = 170, max_height: int = 42) -> QLabel:
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

    # ---------------------------- HELPERS ----------------------------
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

    def _scrollable(self, content: QWidget) -> QWidget:
        wrapper = QWidget()
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return wrapper

    def _set_tooltip(self, widget: QWidget, text: str) -> None:
        widget.setToolTip(text)

    def _path_or_warn(self, edit: QLineEdit, label: str) -> Path | None:
        text = edit.text().strip()
        if not text:
            QMessageBox.warning(self, "Missing path", f"Please select {label}.")
            return None
        return Path(text).expanduser().resolve()

    def _log(self, message: str) -> None:
        self.log_box.appendPlainText(message)

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

    def _refresh_stage_cards(self) -> None:
        for key, record in self.stage_records.items():
            if key in self.stage_labels:
                self.stage_labels[key].setText(self._stage_status_text(record.status))

    def _fill_table(self, table: QTableWidget, df: pd.DataFrame, max_rows: int = 300) -> None:
        if df is None or df.empty:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        df = df.head(max_rows).copy()
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        table.setRowCount(len(df))
        for r, (_, row) in enumerate(df.iterrows()):
            for c, value in enumerate(row):
                item = QTableWidgetItem("" if pd.isna(value) else str(value))
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)

    # ---------------------------- TABS ----------------------------
    def _build_setup_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        paths_group = QGroupBox("Project setup")
        form = QGridLayout(paths_group)
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.project_name_edit = QLineEdit("VSLP Kinematics Project")
        self.task_name_edit = QLineEdit()
        self.task_name_edit.setPlaceholderText("e.g., Bamboo passage video, DDK-pa, DDK-pataka, smile, mouth open-close")
        self._set_tooltip(self.input_edit, "Folder containing raw participant videos. Subfolders are searched recursively.")
        self._set_tooltip(self.output_edit, "Root folder where VSLP writes kinematics outputs: tables, reports, logs, manifests, and future artifacts.")
        self._set_tooltip(self.task_name_edit, "Optional task label used as fallback when filename/folder parsing is inconclusive.")
        browse_in = QPushButton("Browse Input Folder")
        browse_in.clicked.connect(self.browse_input_dir)
        browse_out = QPushButton("Browse Output Folder")
        browse_out.clicked.connect(self.browse_output_dir)
        form.addWidget(QLabel("Input video folder"), 0, 0)
        form.addWidget(self.input_edit, 0, 1)
        form.addWidget(browse_in, 0, 2)
        form.addWidget(QLabel("Output project folder"), 1, 0)
        form.addWidget(self.output_edit, 1, 1)
        form.addWidget(browse_out, 1, 2)
        form.addWidget(QLabel("Project name"), 2, 0)
        form.addWidget(self.project_name_edit, 2, 1, 1, 2)
        form.addWidget(QLabel("Task being analyzed"), 3, 0)
        form.addWidget(self.task_name_edit, 3, 1, 1, 2)
        task_hint = QLabel("Examples: Bamboo passage video, DDK-pa, DDK-pataka, smile, mouth opening/closing, facial expression task")
        task_hint.setObjectName("SubtitleLabel")
        task_hint.setWordWrap(True)
        form.addWidget(task_hint, 4, 1, 1, 2)

        workflow_group = QGroupBox("2. Project gate")
        workflow_layout = QVBoxLayout(workflow_group)
        btn_row = QHBoxLayout()
        self.init_project_btn = QPushButton("Initialize Project")
        self.init_project_btn.setObjectName("RunButton")
        self.init_project_btn.clicked.connect(self.run_project_init)
        self.ingest_btn = QPushButton("Run Video Ingest")
        self.ingest_btn.clicked.connect(self.run_ingest_stage)
        btn_row.addWidget(self.init_project_btn)
        btn_row.addWidget(self.ingest_btn)
        workflow_layout.addLayout(btn_row)

        ingest_group = QGroupBox("3. Ingest summary")
        ingest_layout = QVBoxLayout(ingest_group)
        self.ingest_summary_label = QLabel("No ingest results yet.")
        self.ingest_summary_label.setWordWrap(True)
        self.ingest_summary_label.setObjectName("SubtitleLabel")
        ingest_layout.addWidget(self.ingest_summary_label)
        self.ingest_format_table = QTableWidget(0, 4)
        self.ingest_format_table.setHorizontalHeaderLabels(["Detected format", "Video codec", "Resolution", "Files"])
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
            "Metadata is optional for landmark extraction but essential for later linkage to subject, session, task, diagnosis, severity, and visit structure. The GUI links conservatively and never forces uncertain matches.",
        ))
        group = QGroupBox("Demographics / metadata CSV or XLSX")
        grid = QGridLayout(group)
        self.metadata_edit = QLineEdit()
        self.metadata_edit.setPlaceholderText("optional metadata table; CSV/XLSX supported")
        browse_btn = QPushButton("Browse Metadata")
        browse_btn.clicked.connect(self.browse_metadata_file)
        grid.addWidget(QLabel("Metadata file"), 0, 0)
        grid.addWidget(self.metadata_edit, 0, 1)
        grid.addWidget(browse_btn, 0, 2)
        layout.addWidget(group)
        required = QPlainTextEdit()
        required.setReadOnly(True)
        required.setMaximumHeight(130)
        required.setPlainText(
            "Recommended metadata: subject_id, session_id, visit/date, task, diagnosis/group, severity, video filename, device/camera if available.\n\n"
            "Outputs will include metadata link preview, unmatched videos, unused metadata row samples, and a conservative linkage summary."
        )
        layout.addWidget(required)
        run_btn = QPushButton("Run Metadata Linking")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_metadata_stage)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_landmarks_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "Info",
            "Run Google MediaPipe Face Landmarker over each accepted video and write one per-frame landmark CSV per video. Frames with no detected face are preserved as NaN rows instead of being silently removed, so landmark gaps are auditable.",
        ))
        group = QGroupBox("MediaPipe Face Landmarker configuration")
        form = QGridLayout(group)
        self.model_path_edit = QLineEdit("models/face_landmarker.task")
        self.detection_conf_spin = QDoubleSpinBox(); self.detection_conf_spin.setRange(0.0, 1.0); self.detection_conf_spin.setSingleStep(0.05); self.detection_conf_spin.setValue(0.50)
        self.presence_conf_spin = QDoubleSpinBox(); self.presence_conf_spin.setRange(0.0, 1.0); self.presence_conf_spin.setSingleStep(0.05); self.presence_conf_spin.setValue(0.50)
        self.tracking_conf_spin = QDoubleSpinBox(); self.tracking_conf_spin.setRange(0.0, 1.0); self.tracking_conf_spin.setSingleStep(0.05); self.tracking_conf_spin.setValue(0.50)
        self.expected_landmarks_spin = QSpinBox(); self.expected_landmarks_spin.setRange(468, 500); self.expected_landmarks_spin.setValue(478)
        browse_model = QPushButton("Browse Model")
        browse_model.clicked.connect(self.browse_model_file)
        download_model = QPushButton("Download Default Model")
        download_model.clicked.connect(self.download_landmarker_model)
        install_runtime = QPushButton("Install / Verify MediaPipe Runtime")
        install_runtime.setObjectName("RunButton")
        install_runtime.clicked.connect(self.bootstrap_mediapipe_runtime_stage)
        self.auto_prepare_mediapipe_check = QCheckBox("Auto-install missing runtime and download model before extraction")
        self.auto_prepare_mediapipe_check.setChecked(True)
        self.auto_prepare_mediapipe_check.setToolTip(
            "Recommended for normal users. The GUI will install opencv-python/mediapipe into the active VSLP virtual environment if missing, "
            "then download face_landmarker.task if needed before running extraction."
        )
        form.addWidget(QLabel("FaceLandmarker .task model"), 0, 0)
        form.addWidget(self.model_path_edit, 0, 1)
        form.addWidget(browse_model, 0, 2)
        form.addWidget(download_model, 0, 3)
        form.addWidget(QLabel("Detection confidence"), 1, 0); form.addWidget(self.detection_conf_spin, 1, 1)
        form.addWidget(QLabel("Presence confidence"), 2, 0); form.addWidget(self.presence_conf_spin, 2, 1)
        form.addWidget(QLabel("Tracking confidence"), 3, 0); form.addWidget(self.tracking_conf_spin, 3, 1)
        form.addWidget(QLabel("Expected landmark columns"), 4, 0); form.addWidget(self.expected_landmarks_spin, 4, 1)
        form.addWidget(QLabel("Runtime setup"), 5, 0); form.addWidget(install_runtime, 5, 1, 1, 1); form.addWidget(self.auto_prepare_mediapipe_check, 5, 2, 1, 2)
        layout.addWidget(group)
        note = QPlainTextEdit(); note.setReadOnly(True); note.setMaximumHeight(150)
        note.setPlainText(mediapipe_capability_note())
        layout.addWidget(note)
        btn_row = QHBoxLayout()
        plan_btn = QPushButton("Write Landmark Extraction Plan")
        plan_btn.clicked.connect(self.run_landmark_plan_stage)
        run_btn = QPushButton("Run MediaPipe Landmark Extraction")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_landmark_extraction_stage)
        btn_row.addWidget(plan_btn)
        btn_row.addWidget(run_btn)
        layout.addLayout(btn_row)
        self.landmark_runtime_label = QLabel("Runtime not checked yet.")
        self.landmark_runtime_label.setObjectName("SubtitleLabel")
        self.landmark_runtime_label.setWordWrap(True)
        layout.addWidget(self.landmark_runtime_label)
        self.landmark_summary_table = QTableWidget(0, 7)
        self.landmark_summary_table.setHorizontalHeaderLabels(["Video", "Status", "Frames", "Face frames", "Dropped", "Detected %", "Output"])
        self.landmark_summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.landmark_summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.landmark_summary_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.landmark_summary_table)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_selection_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "Info",
            "Select the landmark subset that will drive default kinematic feature computation. Use the visual face-mesh selector to inspect extracted MediaPipe landmarks, click points to add/remove them, and save the selection for downstream normalization/features.",
        ))

        group = QGroupBox("Landmark preset and visual selection")
        grid = QGridLayout(group)
        self.preset_combo = QComboBox(); self.preset_combo.addItems(list(LANDMARK_PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self.apply_landmark_preset)
        self.landmark_text = QPlainTextEdit(); self.landmark_text.setMaximumHeight(88)
        self.landmark_text.setPlainText(", ".join(map(str, self.landmark_indices)))
        apply_btn = QPushButton("Apply / Save Selected Landmarks")
        apply_btn.setObjectName("RunButton")
        apply_btn.clicked.connect(self.run_selection_stage)
        load_mesh_btn = QPushButton("Load Mesh From Extracted Landmarks")
        load_mesh_btn.clicked.connect(self.load_mesh_from_latest_landmarks)
        clear_btn = QPushButton("Clear Visual Selection")
        clear_btn.clicked.connect(self.clear_visual_landmarks)
        preview_btn = QPushButton("Save Mesh Preview PNG")
        preview_btn.clicked.connect(self.save_landmark_mesh_preview)
        grid.addWidget(QLabel("Preset"), 0, 0); grid.addWidget(self.preset_combo, 0, 1); grid.addWidget(apply_btn, 0, 2)
        grid.addWidget(QLabel("Selected landmark indices"), 1, 0); grid.addWidget(self.landmark_text, 1, 1, 1, 2)
        grid.addWidget(load_mesh_btn, 2, 1); grid.addWidget(clear_btn, 2, 2); grid.addWidget(preview_btn, 2, 3)
        layout.addWidget(group)

        mesh_row = QHBoxLayout()
        self.landmark_canvas = LandmarkMeshCanvas()
        self.landmark_canvas.set_selected(self.landmark_indices)
        self.landmark_canvas.selection_changed.connect(self._canvas_selection_changed)
        mesh_row.addWidget(self.landmark_canvas, stretch=2)
        guide = QTextBrowser()
        guide.setMinimumWidth(360)
        guide.setMaximumWidth(460)
        guide.setHtml(
            "<h3>How to use this selector</h3>"
            "<p><b>1.</b> Run landmark extraction first, then click <b>Load Mesh From Extracted Landmarks</b>. "
            "The canvas will use median x/y positions from a real extracted video, not an abstract diagram.</p>"
            "<p><b>2.</b> Choose a preset such as <b>ALS oral-motor core 15</b>, then click individual points to add/remove landmarks.</p>"
            "<p><b>3.</b> Save the selected set. This writes <code>selected_landmarks.json</code> and makes the selection explicit for normalization and feature computation.</p>"
            "<p><b>Interpretation:</b> mouth/jaw/lip points are usually used for oral kinematics; eye/canthus points are often anchors for scale normalization. Full landmark CSVs remain available for audit.</p>"
            "<p><b>Quality note:</b> if the visual mesh looks distorted or sparse, inspect Face Landmarks and Video QC before trusting downstream features.</p>"
        )
        mesh_row.addWidget(guide, stretch=1)
        layout.addLayout(mesh_row)

        self.preset_table = QTableWidget(0, 4)
        self.preset_table.setHorizontalHeaderLabels(["Preset", "N landmarks", "Recommended use", "Indices"])
        rows = []
        use_map = {
            "ALS oral-motor core 15": "Default oral/jaw/lip kinematics and normalization anchors.",
            "Lower-face jaw/lip kinematics": "Jaw opening, lower lip, commissure, and mouth-aperture signals.",
            "Lip symmetry and lateralization": "Left/right commissure asymmetry and lateralized movement.",
            "Parkinson hypomimia / facial expressivity": "Broader facial expressivity and reduced movement amplitude screening.",
            "Broad audit 30": "Exploratory QC/audit set before narrowing to a task-specific subset.",
        }
        for name, vals in LANDMARK_PRESETS.items():
            rows.append({"Preset": name, "N landmarks": len(vals), "Recommended use": use_map.get(name, "Task-specific or exploratory review."), "Indices": ", ".join(map(str, vals))})
        self._fill_table(self.preset_table, pd.DataFrame(rows), max_rows=20)
        layout.addWidget(self.preset_table)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_normalization_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "Info",
            "Normalization defines how coordinates and distances are scaled. Intercanthal distance is the default starting point for oral/jaw kinematics because it reduces camera-distance effects while staying anatomically interpretable.",
        ))
        group = QGroupBox("Normalization policy")
        grid = QGridLayout(group)
        self.norm_combo = QComboBox(); self.norm_combo.addItems(list(NORMALIZATION_METHODS.keys()))
        self.norm_combo.setCurrentText("intercanthal_distance")
        self.norm_desc = QLabel(NORMALIZATION_METHODS["intercanthal_distance"]); self.norm_desc.setWordWrap(True); self.norm_desc.setObjectName("SubtitleLabel")
        self.norm_combo.currentTextChanged.connect(lambda name: self.norm_desc.setText(NORMALIZATION_METHODS.get(name, "")))
        self.head_stabilize_check = QCheckBox("Enable future head-pose stabilization placeholder")
        self.head_stabilize_check.setChecked(False)
        btn = QPushButton("Write Normalization Config")
        btn.setObjectName("RunButton"); btn.clicked.connect(self.run_normalization_stage)
        grid.addWidget(QLabel("Method"), 0, 0); grid.addWidget(self.norm_combo, 0, 1); grid.addWidget(btn, 0, 2)
        grid.addWidget(QLabel("Interpretation"), 1, 0); grid.addWidget(self.norm_desc, 1, 1, 1, 2)
        grid.addWidget(QLabel("Head stabilization"), 2, 0); grid.addWidget(self.head_stabilize_check, 2, 1, 1, 2)
        layout.addWidget(group)
        self.norm_table = QTableWidget(0, 2)
        self.norm_table.setHorizontalHeaderLabels(["Method", "Use / caution"])
        self._fill_table(self.norm_table, pd.DataFrame([{"Method": k, "Use / caution": v} for k, v in NORMALIZATION_METHODS.items()]), max_rows=30)
        layout.addWidget(self.norm_table)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_qc_tab(self) -> QWidget:
        container = QWidget(); layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "Info",
            "Video QC will separate visual/acquisition problems from facial motor signal. For now this page defines the QC taxonomy that will be quantified after landmark extraction is connected.",
        ))
        qc_rows = [
            ("Decode / container QC", "Unreadable files, fps problems, duration/frame-count inconsistencies, codec/container warnings."),
            ("Face visibility QC", "Face-detected fraction, long no-face gaps, partial face visibility, occlusion risk."),
            ("Pose / head-motion QC", "Head rotation, large translations, off-axis views, pose instability."),
            ("Illumination QC", "Low light, overexposure, flicker, contrast instability."),
            ("Landmark stability QC", "Tracking jitter, coordinate jumps, interpolation burden, landmark dropout."),
            ("Task / adherence QC", "Wrong task, failed repetition structure, mouth hidden, off-screen movement, non-target behavior."),
        ]
        table = QTableWidget(0, 2); table.setHorizontalHeaderLabels(["QC family", "What it will measure"])
        self._fill_table(table, pd.DataFrame(qc_rows, columns=["QC family", "What it will measure"]), max_rows=20)
        layout.addWidget(table)
        btn = QPushButton("Write Video QC Placeholder")
        btn.setObjectName("RunButton"); btn.clicked.connect(self.run_qc_placeholder)
        layout.addWidget(btn)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_features_tab(self) -> QWidget:
        container = QWidget(); layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "Info",
            "Feature computation will convert cleaned, normalized landmark trajectories into mouth/jaw/lip movement features. This stage will plug in the uploaded computation scripts in later patches.",
        ))
        rows = [
            ("Trajectory cleaning", "Interpolate missing frames, remove outliers, smooth trajectories."),
            ("Geometry", "Mouth aperture, lip spread, jaw/lip distances, symmetry and lateralization."),
            ("Kinematics", "Velocity, acceleration, path length, range of motion, timing of movement segments."),
            ("Movement segmentation", "Open/close repetitions or task-specific movement windows."),
            ("Audit outputs", "Per-video feature manifest, flags, configuration and computation policy."),
        ]
        table = QTableWidget(0, 2); table.setHorizontalHeaderLabels(["Feature layer", "Planned computation"])
        self._fill_table(table, pd.DataFrame(rows, columns=["Feature layer", "Planned computation"]), max_rows=20)
        layout.addWidget(table)
        btn = QPushButton("Write Feature Computation Placeholder")
        btn.setObjectName("RunButton"); btn.clicked.connect(self.run_features_placeholder)
        layout.addWidget(btn)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_aggregation_tab(self) -> QWidget:
        container = QWidget(); layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "Info",
            "Frame-level features are time series. Temporal aggregation defines how to create one scalar feature row per video without hiding the policy used to collapse movement over time.",
        ))
        group = QGroupBox("Temporal aggregation profile")
        grid = QGridLayout(group)
        self.agg_combo = QComboBox(); self.agg_combo.addItems(list(AGGREGATION_PROFILES.keys()))
        self.agg_combo.setCurrentText("robust_default")
        self.agg_desc = QLabel(AGGREGATION_PROFILES["robust_default"]); self.agg_desc.setWordWrap(True); self.agg_desc.setObjectName("SubtitleLabel")
        self.agg_combo.currentTextChanged.connect(lambda name: self.agg_desc.setText(AGGREGATION_PROFILES.get(name, "")))
        btn = QPushButton("Write Aggregation Plan")
        btn.setObjectName("RunButton"); btn.clicked.connect(self.run_aggregation_placeholder)
        grid.addWidget(QLabel("Aggregation profile"), 0, 0); grid.addWidget(self.agg_combo, 0, 1); grid.addWidget(btn, 0, 2)
        grid.addWidget(QLabel("Interpretation"), 1, 0); grid.addWidget(self.agg_desc, 1, 1, 1, 2)
        layout.addWidget(group)
        table = QTableWidget(0, 2); table.setHorizontalHeaderLabels(["Profile", "Recommended use"])
        self._fill_table(table, pd.DataFrame([{"Profile": k, "Recommended use": v} for k, v in AGGREGATION_PROFILES.items()]), max_rows=20)
        layout.addWidget(table)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_inspector_tab(self) -> QWidget:
        container = QWidget(); layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel("Info", "Preview the current kinematics stage tables and manifests before trusting downstream outputs."))
        btn_row = QHBoxLayout()
        refresh = QPushButton("Refresh Inspector")
        refresh.clicked.connect(self.refresh_inspector)
        open_out = QPushButton("Open Output Project Folder")
        open_out.clicked.connect(self.open_output_root)
        btn_row.addWidget(refresh); btn_row.addWidget(open_out); btn_row.addStretch(1)
        layout.addLayout(btn_row)
        self.inspector_label = QLabel("No table loaded yet."); self.inspector_label.setObjectName("SubtitleLabel")
        layout.addWidget(self.inspector_label)
        self.inspector_table = QTableWidget(0, 0)
        self.inspector_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.inspector_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.inspector_table)
        return self._scrollable(container)

    def _build_reports_tab(self) -> QWidget:
        container = QWidget(); layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel("Info", "Create a reproducible kinematics workflow report and open output artifacts."))
        btn_row = QHBoxLayout()
        create = QPushButton("Create Scaffold Report")
        create.setObjectName("RunButton"); create.clicked.connect(self.run_report_stage)
        open_report = QPushButton("Open HTML Report")
        open_report.clicked.connect(self.open_report)
        open_folder = QPushButton("Open Output Project Folder")
        open_folder.clicked.connect(self.open_output_root)
        btn_row.addWidget(create); btn_row.addWidget(open_report); btn_row.addWidget(open_folder); btn_row.addStretch(1)
        layout.addLayout(btn_row)
        self.report_box = QTextBrowser(); self.report_box.setMinimumHeight(340)
        self.report_box.setHtml("<b>No report generated yet.</b><br>Run the report stage after configuring the workflow outline.")
        layout.addWidget(self.report_box)
        return self._scrollable(container)

    # ---------------------------- ACTIONS ----------------------------
    def browse_input_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select input video folder")
        if path:
            self.input_edit.setText(path); self.input_root = Path(path)

    def browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output project folder")
        if path:
            self.output_edit.setText(path); self.output_root = Path(path)

    def browse_metadata_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select metadata file", filter="Tables (*.csv *.xlsx *.xls);;All files (*.*)")
        if path:
            self.metadata_edit.setText(path); self.metadata_path = Path(path)

    def browse_model_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select MediaPipe FaceLandmarker model", filter="MediaPipe task (*.task);;All files (*.*)")
        if path:
            self.model_path_edit.setText(path)

    def run_project_init(self) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out is None:
            return
        self.output_root = out
        project_dir = out / "kinematics"
        for sub in ["000_ingest", "001_metadata", "002_landmarks", "003_selection", "004_normalization", "005_video_qc", "006_features", "007_aggregation", "008_inspector", "009_reports"]:
            (project_dir / sub).mkdir(parents=True, exist_ok=True)
        manifest = project_dir / "project_manifest.json"
        manifest.write_text(json.dumps({"project_name": self.project_name_edit.text(), "task": self.task_name_edit.text(), "schema": "vslp_kinematics_project_v0.59"}, indent=2), encoding="utf-8")
        self.stage_records["project"] = StageRecord(status="completed", manifest_path=str(manifest))
        self._refresh_stage_cards()
        self._log(f"Initialized kinematics project: {project_dir}")

    def run_ingest_stage(self) -> None:
        inp = self._path_or_warn(self.input_edit, "an input video folder")
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if inp is None or out is None:
            return
        self.input_root, self.output_root = inp, out
        cfg = VideoIngestConfig(input_root=inp, output_root=out, recursive=True, extensions=DEFAULT_VIDEO_EXTENSIONS)
        try:
            res = run_ingest(cfg)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ingest failed", str(exc)); return
        self.ingest_manifest_csv = Path(res["manifest_csv"])
        self.stage_records["ingest"] = StageRecord(status="completed", manifest_path=str(self.ingest_manifest_csv))
        self._refresh_stage_cards()
        self._log(f"Video ingest completed: {res['n_videos']} video(s).")
        self._load_ingest_summary()

    def _load_ingest_summary(self) -> None:
        if not self.ingest_manifest_csv or not self.ingest_manifest_csv.exists():
            return
        df = pd.read_csv(self.ingest_manifest_csv)
        self.ingest_summary_label.setText(f"{len(df)} candidate videos detected. Formats/codecs/resolutions summarized below.")
        if not df.empty:
            tmp = df.copy()
            tmp["Resolution"] = tmp["width"].fillna(0).astype(int).astype(str) + " × " + tmp["height"].fillna(0).astype(int).astype(str)
            grouped = tmp.groupby(["extension", "codec_name", "Resolution"], dropna=False).size().reset_index(name="Files")
            grouped.columns = ["Detected format", "Video codec", "Resolution", "Files"]
            self._fill_table(self.ingest_format_table, grouped, max_rows=60)

    def run_metadata_stage(self) -> None:
        if not self.ingest_manifest_csv or not self.ingest_manifest_csv.exists():
            QMessageBox.warning(self, "Missing ingest", "Run video ingest before metadata linking."); return
        meta = Path(self.metadata_edit.text()).expanduser().resolve() if self.metadata_edit.text().strip() else None
        try:
            out = link_metadata(self.ingest_manifest_csv, meta, Path(self.output_edit.text()).expanduser().resolve())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Metadata failed", str(exc)); return
        self.stage_records["metadata"] = StageRecord(status="completed", manifest_path=str(out))
        self._refresh_stage_cards(); self._log(f"Metadata link preview written: {out}")

    def apply_landmark_preset(self, name: str) -> None:
        vals = LANDMARK_PRESETS.get(name, ())
        self.landmark_text.setPlainText(", ".join(map(str, vals)))
        if hasattr(self, "landmark_canvas"):
            self.landmark_canvas.set_selected(vals)

    def _canvas_selection_changed(self, text: str) -> None:
        if hasattr(self, "landmark_text"):
            self.landmark_text.setPlainText(text)

    def clear_visual_landmarks(self) -> None:
        if hasattr(self, "landmark_canvas"):
            self.landmark_canvas.set_selected([])
        self.landmark_text.setPlainText("")

    def _latest_landmark_csv(self) -> Path | None:
        out_text = self.output_edit.text().strip()
        if not out_text:
            return None
        tables = Path(out_text).expanduser().resolve() / "kinematics" / "002_landmarks" / "tables"
        if not tables.exists():
            return None
        csvs = [p for p in tables.glob("*-lmks.csv") if p.is_file()]
        if not csvs:
            csvs = [p for p in tables.glob("**/*-lmks.csv") if p.is_file()]
        if not csvs:
            return None
        return max(csvs, key=lambda p: p.stat().st_mtime)

    def _landmark_points_from_csv(self, csv_path: Path) -> dict[int, tuple[float, float]]:
        df = pd.read_csv(csv_path)
        if "face_detected" in df.columns:
            detected = df[df["face_detected"].astype(str).str.lower().isin(["true", "1", "yes"])]
            if not detected.empty:
                df = detected
        points: dict[int, tuple[float, float]] = {}
        for i in range(0, 478):
            xcol, ycol = f"{i}_x", f"{i}_y"
            if xcol not in df.columns or ycol not in df.columns:
                continue
            x = pd.to_numeric(df[xcol], errors="coerce").median()
            y = pd.to_numeric(df[ycol], errors="coerce").median()
            if pd.notna(x) and pd.notna(y):
                # Clamp MediaPipe normalized coordinates to the visible canvas range.
                points[i] = (float(max(0.0, min(1.0, x))), float(max(0.0, min(1.0, y))))
        return points

    def load_mesh_from_latest_landmarks(self) -> None:
        path = self._latest_landmark_csv()
        if path is None:
            QMessageBox.information(
                self,
                "No landmark CSV found",
                "Run Landmarks → Run MediaPipe Landmark Extraction first. The visual selector can then display the median face mesh from an extracted video.",
            )
            return
        try:
            points = self._landmark_points_from_csv(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not load landmark mesh", str(exc))
            return
        if len(points) < 20:
            QMessageBox.warning(
                self,
                "Sparse landmark file",
                f"Only {len(points)} usable landmark points were found in:\n{path}\n\nCheck face_detected coverage before using this selection.",
            )
            return
        self.landmark_canvas.set_points(points, f"Median mesh from {path.name}")
        try:
            current = parse_int_list(self.landmark_text.toPlainText())
        except Exception:
            current = []
        self.landmark_canvas.set_selected(current)
        self._log(f"Loaded visual face mesh from: {path} ({len(points)} usable landmark points)")

    def save_landmark_mesh_preview(self) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out is None or not hasattr(self, "landmark_canvas"):
            return
        fig_dir = out / "kinematics" / "003_selection" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        path = fig_dir / "selected_landmark_mesh_preview.png"
        self.landmark_canvas.grab().save(str(path))
        QMessageBox.information(self, "Mesh preview saved", f"Saved current landmark-selection preview:\n{path}")
        self._log(f"Saved landmark mesh preview: {path}")

    def _landmark_config(self) -> LandmarkRunConfig:
        return LandmarkRunConfig(
            model_path=self.model_path_edit.text().strip() or "models/face_landmarker.task",
            selected_preset=self.preset_combo.currentText() if hasattr(self, "preset_combo") else "ALS oral-motor core 15",
            selected_landmarks=parse_int_list(self.landmark_text.toPlainText()),
            min_face_detection_confidence=float(self.detection_conf_spin.value()),
            min_face_presence_confidence=float(self.presence_conf_spin.value()),
            min_tracking_confidence=float(self.tracking_conf_spin.value()),
            normalization_method=self.norm_combo.currentText() if hasattr(self, "norm_combo") else "intercanthal_distance",
            n_landmarks=int(self.expected_landmarks_spin.value()) if hasattr(self, "expected_landmarks_spin") else 478,
        )

    def _require_ingest_manifest(self) -> Path | None:
        if self.ingest_manifest_csv and self.ingest_manifest_csv.exists():
            return self.ingest_manifest_csv
        out_text = self.output_edit.text().strip()
        if out_text:
            candidate = Path(out_text).expanduser().resolve() / "kinematics" / "000_ingest" / "tables" / "video_ingest_manifest.csv"
            if candidate.exists():
                self.ingest_manifest_csv = candidate
                return candidate
        QMessageBox.warning(self, "Missing ingest", "Run Setup → Run Video Ingest before landmark extraction.")
        return None

    def _update_landmark_runtime_label(self) -> None:
        status = mediapipe_environment_status()
        self.landmark_runtime_label.setText(
            f"OpenCV: {'available' if status.opencv_available else 'missing'}"
            f"{f' ({status.opencv_version})' if status.opencv_version else ''}; "
            f"MediaPipe: {'available' if status.mediapipe_available else 'missing'}"
            f"{f' ({status.mediapipe_version})' if status.mediapipe_version else ''}. "
            f"{status.message}"
        )

    def bootstrap_mediapipe_runtime_stage(self) -> None:
        self._log("Checking/installing MediaPipe runtime into the active Python environment...")
        try:
            result = bootstrap_mediapipe_runtime()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "MediaPipe runtime setup failed", str(exc))
            self._log(f"MediaPipe runtime setup failed: {exc}")
            return
        self._update_landmark_runtime_label()
        if result.get("ok"):
            QMessageBox.information(
                self,
                "MediaPipe runtime ready",
                "opencv-python and mediapipe are available in this environment. "
                "If this was a fresh install, restart the GUI if extraction still reports missing imports.",
            )
        else:
            QMessageBox.warning(
                self,
                "MediaPipe runtime incomplete",
                "The install command finished but the runtime is still not fully available. "
                "Open the run log for details, or restart the GUI and try again.",
            )
        self._log(f"MediaPipe runtime setup command: {result.get('command')}")
        if result.get("stderr"):
            self._log("pip stderr tail: " + str(result.get("stderr"))[-1200:])

    def download_landmarker_model(self) -> None:
        try:
            path = download_default_model(self._path_or_warn(self.output_edit, "an output project folder") or Path.cwd(), self._landmark_config())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Model download failed", str(exc)); return
        self.model_path_edit.setText(str(path))
        QMessageBox.information(self, "Model ready", f"FaceLandmarker model is available at:\n{path}")
        self._log(f"MediaPipe FaceLandmarker model ready: {path}")

    def run_landmark_plan_stage(self) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        manifest = self._require_ingest_manifest()
        if out is None or manifest is None:
            return
        cfg = self._landmark_config()
        path = write_landmark_plan(out, cfg, manifest_csv=manifest)
        self.stage_records["landmarks"] = StageRecord(status="planned", manifest_path=str(path))
        self._update_landmark_runtime_label()
        self._refresh_stage_cards(); self._log(f"Landmark extraction plan written: {path}")

    def run_landmark_extraction_stage(self) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        manifest = self._require_ingest_manifest()
        if out is None or manifest is None:
            return
        cfg = self._landmark_config()
        self._update_landmark_runtime_label()
        status = mediapipe_environment_status()
        auto_prepare = bool(getattr(self, "auto_prepare_mediapipe_check", None) and self.auto_prepare_mediapipe_check.isChecked())
        if (not status.opencv_available or not status.mediapipe_available) and auto_prepare:
            reply = QMessageBox.question(
                self,
                "Install MediaPipe runtime?",
                "OpenCV and/or MediaPipe are missing from this VSLP Python environment.\n\n"
                "The GUI can install the required packages into the active virtual environment now:\n"
                "  opencv-python\n"
                "  mediapipe\n\n"
                "This can take a few minutes and requires internet access. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
            self.bootstrap_mediapipe_runtime_stage()
            status = mediapipe_environment_status()
        if not status.opencv_available or not status.mediapipe_available:
            QMessageBox.critical(
                self,
                "MediaPipe runtime missing",
                "Real landmark extraction cannot run until opencv-python and mediapipe are installed.\n\n"
                "Click 'Install / Verify MediaPipe Runtime' on the Landmarks tab, or run:\n"
                "python -m pip install opencv-python mediapipe",
            )
            return
        try:
            res = run_mediapipe_landmarks(out, cfg, manifest)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "MediaPipe extraction failed", str(exc)); return
        manifest_csv = Path(res["manifest_csv"])
        self.stage_records["landmarks"] = StageRecord(status="completed" if int(res.get("n_error", 0)) == 0 else "completed_with_warnings", manifest_path=str(manifest_csv))
        self._refresh_stage_cards()
        self._log(f"MediaPipe landmarks completed: {res['n_ok']} ok, {res['n_error']} error, {res['n_skipped_existing']} skipped. Manifest: {manifest_csv}")
        self._load_landmark_summary(manifest_csv)
        QMessageBox.information(self, "Landmark extraction complete", f"Processed {res['n_ok']} video(s).\nManifest:\n{manifest_csv}")

    def _load_landmark_summary(self, manifest_csv: Path) -> None:
        if not manifest_csv.exists():
            return
        df = pd.read_csv(manifest_csv)
        if df.empty:
            self.landmark_summary_table.setRowCount(0); self.landmark_summary_table.setColumnCount(0); return
        preview = pd.DataFrame({
            "Video": df.get("video_id", ""),
            "Status": df.get("status", ""),
            "Frames": df.get("n_frames", ""),
            "Face frames": df.get("n_faces_detected", ""),
            "Dropped": df.get("n_dropped", ""),
            "Detected %": (pd.to_numeric(df.get("face_detected_fraction"), errors="coerce") * 100).round(1),
            "Output": df.get("output_csv", ""),
        })
        self._fill_table(self.landmark_summary_table, preview, max_rows=100)

    def run_selection_stage(self) -> None:
        try:
            self.landmark_indices = parse_int_list(self.landmark_text.toPlainText())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Invalid landmarks", str(exc)); return
        if hasattr(self, "landmark_canvas"):
            self.landmark_canvas.set_selected(self.landmark_indices)
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out:
            path = out / "kinematics" / "003_selection" / "tables" / "selected_landmarks.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            preview = out / "kinematics" / "003_selection" / "figures" / "selected_landmark_mesh_preview.png"
            preview.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(self, "landmark_canvas"):
                self.landmark_canvas.grab().save(str(preview))
            payload = {
                "selected_landmarks": list(self.landmark_indices),
                "n_selected": len(self.landmark_indices),
                "preset": self.preset_combo.currentText(),
                "mesh_source": getattr(getattr(self, "landmark_canvas", None), "source_label", "not_available"),
                "preview_png": str(preview) if preview.exists() else None,
                "interpretation": "Selected landmarks define the default subset for normalization and kinematic feature computation. Full extracted MediaPipe landmarks remain available for audit.",
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.stage_records["selection"] = StageRecord(status="completed", manifest_path=str(path))
            self._refresh_stage_cards(); self._log(f"Selected {len(self.landmark_indices)} landmarks: {path}")

    def run_normalization_stage(self) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out is None:
            return
        path = write_normalization_config(out, self.norm_combo.currentText())
        self.stage_records["normalization"] = StageRecord(status="completed", manifest_path=str(path))
        self._refresh_stage_cards(); self._log(f"Normalization config written: {path}")

    def _write_placeholder(self, stage_key: str, rel: str, payload: dict, message: str) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out is None:
            return
        path = out / "kinematics" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.stage_records[stage_key] = StageRecord(status="completed", manifest_path=str(path))
        self._refresh_stage_cards(); self._log(f"{message}: {path}")

    def run_qc_placeholder(self) -> None:
        self._write_placeholder("qc", "005_video_qc/tables/video_qc_plan.json", {"status": "placeholder", "families": ["decode", "face_visibility", "pose", "illumination", "landmark_stability", "task_adherence"]}, "Video QC plan written")

    def run_features_placeholder(self) -> None:
        self._write_placeholder("features", "006_features/tables/feature_computation_plan.json", {"status": "placeholder", "selected_landmarks": list(self.landmark_indices)}, "Feature computation plan written")

    def run_aggregation_placeholder(self) -> None:
        self._write_placeholder("aggregation", "007_aggregation/tables/aggregation_plan.json", {"profile": self.agg_combo.currentText(), "description": AGGREGATION_PROFILES.get(self.agg_combo.currentText(), "")}, "Aggregation plan written")

    def refresh_inspector(self) -> None:
        out_text = self.output_edit.text().strip()
        if not out_text:
            return
        root = Path(out_text) / "kinematics"
        candidates = sorted(root.glob("**/*.csv"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        if not candidates:
            self.inspector_label.setText("No CSV outputs found yet."); self.inspector_table.setRowCount(0); self.inspector_table.setColumnCount(0); return
        path = candidates[0]
        df = pd.read_csv(path)
        self.inspector_label.setText(f"Previewing latest CSV: {path}")
        self._fill_table(self.inspector_table, df, max_rows=300)

    def run_report_stage(self) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out is None:
            return
        report = write_scaffold_report(out)
        self.last_report_html = report
        self.stage_records["reports"] = StageRecord(status="completed", report_path=str(report))
        self._refresh_stage_cards()
        self.report_box.setHtml(report.read_text(encoding="utf-8"))
        self._log(f"Kinematics report written: {report}")

    def refresh_latest_outputs(self) -> None:
        self._load_ingest_summary()
        self.refresh_inspector()
        self._log("Refreshed latest kinematics outputs.")

    def open_output_root(self) -> None:
        path_text = self.output_edit.text().strip()
        if not path_text:
            QMessageBox.information(self, "No output folder", "Select an output folder first."); return
        path = Path(path_text).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        import webbrowser
        webbrowser.open(path.as_uri())

    def open_report(self) -> None:
        if self.last_report_html and self.last_report_html.exists():
            import webbrowser
            webbrowser.open(self.last_report_html.as_uri())
        else:
            QMessageBox.information(self, "No report", "Create the report first.")

    def run_all(self) -> None:
        self.run_project_init()
        self.run_ingest_stage()
        self.run_metadata_stage()
        self.run_landmark_plan_stage()
        self.run_selection_stage()
        self.run_normalization_stage()
        self.run_qc_placeholder()
        self.run_features_placeholder()
        self.run_aggregation_placeholder()
        self.run_report_stage()


def launch_kinematics_gui() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("VSLP")
    try:
        from vslp.gui.theme import build_dark_stylesheet
        app.setStyleSheet(build_dark_stylesheet())
    except Exception:
        # Fallback mirrors the acoustic dark theme closely enough if the shared theme is unavailable.
        app.setStyleSheet(
            """
            QWidget { background: #071727; color: #D8E8F8; font-family: 'Segoe UI', Arial, sans-serif; font-size: 10pt; }
            QFrame#Sidebar { background: #0E253A; border: 1px solid #1B3A57; border-radius: 10px; }
            QLabel#AppTitleLabel { font-size: 24pt; font-weight: 900; color: #FFFFFF; }
            QLabel#SubtitleLabel, QLabel#IPNoticeLabel { color: #AFC8DE; font-size: 8.5pt; }
            QFrame#Card, QGroupBox { background: #0C2033; border: 1px solid #1B3A57; border-radius: 8px; }
            QFrame#BrandingBar { background: #F6F8FB; border-radius: 10px; }
            QLabel#BrandingTitle { color: #06213A; font-weight: 800; }
            QPushButton { background: #1D6791; color: white; border: 1px solid #2E7DAA; border-radius: 6px; padding: 7px 10px; font-weight: 700; }
            QPushButton#RunButton { background: #16896F; }
            QPushButton#OpenButton { background: #415A75; }
            QLineEdit, QPlainTextEdit, QTextBrowser, QComboBox, QSpinBox, QDoubleSpinBox { background: #0B1D2E; color: #E7F3FF; border: 1px solid #244B6C; border-radius: 5px; padding: 5px; }
            QTableWidget { background: #0B1D2E; color: #E7F3FF; gridline-color: #244B6C; selection-background-color: #254D70; selection-color: white; }
            QHeaderView::section { background: #183B59; color: #FFFFFF; padding: 6px; border: 1px solid #244B6C; }
            QTabWidget::pane { border: 1px solid #244B6C; }
            QTabBar::tab { background: #102A42; color: #D8E8F8; padding: 8px 13px; border-top-left-radius: 5px; border-top-right-radius: 5px; }
            QTabBar::tab:selected { background: #1D6791; color: #FFFFFF; }
            """
        )
    win = KinematicsPipelineWindow()
    win.show()
    return app.exec()


def main() -> int:
    return launch_kinematics_gui()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
