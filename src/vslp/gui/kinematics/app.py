"""VSLP Kinematics GUI scaffold and commercial workflow outline.

This version is intentionally a polished, user-facing scaffold. It defines the
full kinematics workflow, visual structure, recommended decision points, and
initial lightweight backends for ingest/metadata/planning. Heavy MediaPipe and
feature computation execution will be connected in later patches.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from PySide6.QtCore import Qt, QSize, QUrl
    from PySide6.QtGui import QDesktopServices, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextBrowser,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PySide6 is required to launch the kinematics GUI. Install with: pip install -e '.[gui]'"
    ) from exc

import pandas as pd

from vslp.analysis.kinematics import (
    AGGREGATION_PROFILES,
    LANDMARK_PRESETS,
    NORMALIZATION_METHODS,
    LandmarkRunConfig,
    VideoIngestConfig,
    link_metadata,
    mediapipe_capability_note,
    run_ingest,
    write_landmark_plan,
    write_normalization_config,
    write_scaffold_report,
)
from vslp.analysis.kinematics.schemas import DEFAULT_VIDEO_EXTENSIONS, parse_int_list

APP_TITLE = "VSLP Kinematics Pipeline"
APP_VERSION = "v0.57 outline"
COPYRIGHT_TEXT = "© 2026 Speech Production Lab, University of Toronto. Internal research software; not a clinical diagnostic device."
BRAND_DIR = Path(__file__).parent / "assets" / "branding"
LAB_LOGO = BRAND_DIR / "lab_logo.png"
UOFT_LOGO = BRAND_DIR / "uoft_logo.png"

NAV_ITEMS = [
    ("setup", "1  Setup / Ingest"),
    ("metadata", "2  Metadata"),
    ("landmarks", "3  Face Landmarks"),
    ("selection", "4  Landmark Selection"),
    ("normalization", "5  Normalization"),
    ("qc", "6  Video QC"),
    ("features", "7  Feature Computation"),
    ("aggregation", "8  Temporal Aggregation"),
    ("inspector", "9  Data Inspector"),
    ("reports", "10 Reports & Outputs"),
]

WORKFLOW_STAGES = [
    ("Setup", "Discover and structurally probe videos", "000_ingest"),
    ("Metadata", "Link subjects, sessions, tasks, clinical labels", "001_metadata"),
    ("Landmarks", "Extract MediaPipe Face Landmarker trajectories", "002_landmarks"),
    ("Selection", "Choose clinically meaningful landmark subsets", "003_selection"),
    ("Normalization", "Scale/stabilize coordinates before features", "004_normalization"),
    ("Video QC", "Quantify landmark/visibility/acquisition quality", "005_video_qc"),
    ("Features", "Compute frame/movement-level kinematics", "006_features"),
    ("Aggregation", "Collapse time series transparently", "007_aggregation"),
    ("Inspector", "Review tables, manifests, and QC evidence", "008_inspector"),
    ("Reports", "Export reproducible package and report", "009_reports"),
]

STAGE_GUIDANCE = {
    "setup": {
        "purpose": "Create a reproducible inventory of all candidate videos before any transformation.",
        "inputs": "Folder containing MP4/WebM/MOV/MKV/AVI/WMV/3GP or other supported video containers.",
        "outputs": "video_ingest_manifest.csv/json with source paths, codec/container, fps, duration, frame count estimate, task guess, and warnings.",
        "decision": "Confirm that expected videos are present, readable, and assigned plausible task guesses before moving on.",
    },
    "metadata": {
        "purpose": "Attach participant/session/task/clinical context without requiring it for landmark extraction.",
        "inputs": "Optional CSV/XLSX metadata table.",
        "outputs": "Conservative link preview and metadata summary.",
        "decision": "Verify that IDs/tasks/sessions align; unresolved metadata should be documented rather than forced.",
    },
    "landmarks": {
        "purpose": "Configure the MediaPipe Face Landmarker stage and document extraction assumptions.",
        "inputs": "Ingest manifest and optional FaceLandmarker .task model.",
        "outputs": "landmark extraction plan, run configuration, and later per-video landmark CSVs.",
        "decision": "Keep confidence thresholds traceable. Higher thresholds increase missing frames; lower thresholds may accept uncertain frames.",
    },
    "selection": {
        "purpose": "Reduce the full face mesh to reproducible, anatomically meaningful landmark sets.",
        "inputs": "MediaPipe landmark indices and recommended presets.",
        "outputs": "Selected landmark list for downstream features; full landmarks remain available for audit.",
        "decision": "Use ALS oral-motor core by default; choose broader sets for exploratory facial expressivity or hypomimia analysis.",
    },
    "normalization": {
        "purpose": "Define how distances and movements are scaled so values are comparable across camera distance and face size.",
        "inputs": "Selected anchor landmarks and normalization method.",
        "outputs": "normalization_config.json describing the chosen coordinate policy.",
        "decision": "Intercanthal distance is default for oral/jaw kinematics; raw normalized coordinates are audit only.",
    },
    "qc": {
        "purpose": "Separate visual/acquisition problems from true facial movement signals.",
        "inputs": "Video structural metrics, face-detected frames, landmark trajectories, illumination/pose/stability indicators.",
        "outputs": "Video QC table, QC flags, visual-quality covariates, and interpretation notes.",
        "decision": "QC should flag and explain risk. It should not automatically exclude videos without analyst review.",
    },
    "features": {
        "purpose": "Compute interpretable kinematic signals from cleaned, normalized landmark trajectories.",
        "inputs": "Landmark CSVs, selected landmarks, normalization configuration, QC indicators.",
        "outputs": "Frame-level trajectories, movement-level summaries, and feature-computation audit tables.",
        "decision": "Document which features are raw trajectories, movement-derived summaries, or exploratory outputs.",
    },
    "aggregation": {
        "purpose": "Turn time-series features into one row per video while preserving clinically relevant variability.",
        "inputs": "Frame-level and movement-level kinematic features.",
        "outputs": "Per-video scalar feature matrix plus aggregation policy/audit.",
        "decision": "Use robust default summaries for ML-ready exports; retain movement/time-series evidence for audit.",
    },
    "inspector": {
        "purpose": "Let the analyst inspect stage outputs before trusting downstream tables.",
        "inputs": "Any stage table or manifest created by the GUI.",
        "outputs": "Previewed tables and opened artifacts.",
        "decision": "Use this to detect wrong input paths, metadata mismatch, missing landmark outputs, or unexpected warnings.",
    },
    "reports": {
        "purpose": "Package the run into a reproducible, SOP-aligned report.",
        "inputs": "Stage manifests, configurations, selected presets, normalization choices, QC/feature outputs.",
        "outputs": "HTML report, output folder, and future export bundle.",
        "decision": "Use the report as the handoff artifact before feature-analysis/ML stages.",
    },
}

PRESET_DESCRIPTIONS = {
    "ALS oral-motor core 15": "Default starting set for visible oral-motor kinematics. Focuses on mouth aperture, lip spread, commissures, jaw/lower face and eye/canthus anchors for scale normalization.",
    "Lower-face jaw/lip kinematics": "More lower-face and jaw-emphasis landmarks for mouth opening, closing, jaw excursion, and lip aperture analyses.",
    "Lip symmetry and lateralization": "Commissure- and eye-anchor-heavy set for left/right symmetry, lateralized deviation and coordination checks.",
    "Parkinson hypomimia / facial expressivity": "Adds brow and peri-orbital landmarks to support reduced expressivity/hypomimia-oriented exploration.",
    "Broad audit 30": "A broader quality and feature-audit set. Useful during development, but not recommended as the first ML-ready feature subset.",
}


def set_app_style(app: QApplication) -> None:
    app.setStyleSheet("""
    QWidget { background: #F4F7FB; color: #172033; font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; }
    QMainWindow { background: #F4F7FB; }
    QLabel#Title { font-size: 22pt; font-weight: 800; color: #10233F; }
    QLabel#Subtitle { color: #526173; font-size: 10.5pt; line-height: 135%; }
    QLabel#Eyebrow { color: #1769AA; font-size: 8.5pt; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; }
    QLabel#CardTitle { font-size: 12.5pt; font-weight: 800; color: #12345A; }
    QLabel#MetricValue { font-size: 19pt; font-weight: 800; color: #12345A; }
    QLabel#MetricLabel { font-size: 8.8pt; font-weight: 700; color: #526173; }
    QLabel#Footer { color: #657386; font-size: 8.4pt; }
    QFrame#Card, QGroupBox { background: #FFFFFF; border: 1px solid #DCE5F2; border-radius: 16px; padding: 12px; }
    QFrame#MiniCard { background: #FFFFFF; border: 1px solid #DCE5F2; border-radius: 14px; padding: 10px; }
    QFrame#StageBlock { background: #F9FBFE; border: 1px solid #DCE5F2; border-left: 5px solid #1769AA; border-radius: 14px; padding: 10px; }
    QFrame#StageBlockDone { background: #F0F8F3; border: 1px solid #CBE8D3; border-left: 5px solid #2D9C55; border-radius: 14px; padding: 10px; }
    QFrame#StageBlockFuture { background: #FFF9EC; border: 1px solid #F3D899; border-left: 5px solid #D69B00; border-radius: 14px; padding: 10px; }
    QGroupBox::title { color: #12345A; font-weight: 800; subcontrol-origin: margin; left: 12px; padding: 0 6px; }
    QPushButton { background: #FFFFFF; color: #10233F; border: 1px solid #B8C7DA; border-radius: 9px; padding: 8px 12px; font-weight: 700; }
    QPushButton:hover { background: #EEF6FF; border-color: #78A7D8; color: #10233F; }
    QPushButton:pressed { background: #DCEEFF; color: #10233F; }
    QPushButton#Primary { background: #1769AA; color: #FFFFFF; border-color: #1769AA; }
    QPushButton#Primary:hover { background: #0F5B96; color: #FFFFFF; }
    QPushButton#Success { background: #2D9C55; color: #FFFFFF; border-color: #2D9C55; }
    QPushButton#Success:hover { background: #248B49; color: #FFFFFF; }
    QLineEdit, QPlainTextEdit, QTextEdit, QTextBrowser, QComboBox, QSpinBox { background: #FFFFFF; color: #172033; border: 1px solid #B8C7DA; border-radius: 8px; padding: 6px; selection-background-color: #CFE5FF; selection-color: #172033; }
    QTextBrowser { line-height: 135%; }
    QComboBox QAbstractItemView { background: #FFFFFF; color: #172033; selection-background-color: #D9ECFF; selection-color: #172033; border: 1px solid #B8C7DA; }
    QTableWidget { background: #FFFFFF; alternate-background-color: #F6FAFF; gridline-color: #E1E9F3; color: #172033; selection-background-color: #D9ECFF; selection-color: #172033; border: 1px solid #DCE5F2; border-radius: 8px; }
    QHeaderView::section { background: #EAF1F8; color: #10233F; border: 0px; border-right: 1px solid #D2DEEA; padding: 6px; font-weight: 800; }
    QListWidget#Nav { background: #10233F; color: #DDEBFA; border: 0px; padding: 8px; font-weight: 650; }
    QListWidget#Nav::item { padding: 11px 12px; border-radius: 8px; margin: 3px; }
    QListWidget#Nav::item:selected { background: #1E78BE; color: #FFFFFF; }
    QListWidget#Nav::item:hover { background: #193756; color: #FFFFFF; }
    QScrollArea { border: 0px; background: #F4F7FB; }
    QMessageBox { background: #FFFFFF; color: #172033; }
    QMessageBox QLabel { background: #FFFFFF; color: #172033; }
    QMessageBox QPushButton { background: #FFFFFF; color: #172033; border: 1px solid #B8C7DA; border-radius: 8px; padding: 7px 14px; }
    """)


class Card(QFrame):
    def __init__(self, title: str | None = None, subtitle: str | None = None):
        super().__init__()
        self.setObjectName("Card")
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)
        if title:
            lab = QLabel(title)
            lab.setObjectName("CardTitle")
            lab.setWordWrap(True)
            self.layout.addWidget(lab)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("Subtitle")
            sub.setWordWrap(True)
            self.layout.addWidget(sub)


class KinematicsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1540, 940)
        self.input_root: Path | None = None
        self.output_root: Path | None = None
        self.ingest_manifest_csv: Path | None = None
        self.metadata_path: Path | None = None
        self.last_report_html: Path | None = None
        self.landmark_indices = LANDMARK_PRESETS["ALS oral-motor core 15"]

        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        self.nav = QListWidget()
        self.nav.setObjectName("Nav")
        self.nav.setFixedWidth(260)
        for _, label in NAV_ITEMS:
            self.nav.addItem(label)
        main.addWidget(self.nav)

        self.stack = QStackedWidget()
        main.addWidget(self.stack, 1)
        self.pages = {
            "setup": self._setup_page(),
            "metadata": self._metadata_page(),
            "landmarks": self._landmarks_page(),
            "selection": self._selection_page(),
            "normalization": self._normalization_page(),
            "qc": self._qc_page(),
            "features": self._features_page(),
            "aggregation": self._aggregation_page(),
            "inspector": self._inspector_page(),
            "reports": self._reports_page(),
        }
        for key, _ in NAV_ITEMS:
            self.stack.addWidget(self.pages[key])
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

    # ------------------------------------------------------------------
    # Shared UI helpers
    # ------------------------------------------------------------------
    def _logo(self, path: Path, fallback: str, width: int = 138, height: int = 52) -> QLabel:
        lab = QLabel()
        lab.setFixedSize(width, height)
        lab.setAlignment(Qt.AlignCenter)
        if path.exists():
            pix = QPixmap(str(path))
            if not pix.isNull():
                lab.setPixmap(pix.scaled(QSize(width, height), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                return lab
        lab.setText(fallback)
        lab.setStyleSheet("font-weight:800;color:#12345A;background:#FFFFFF;border:1px solid #DCE5F2;border-radius:8px;")
        return lab

    def _page(self, title: str, subtitle: str, key: str) -> QWidget:
        outer = QWidget()
        outer_lay = QVBoxLayout(outer)
        outer_lay.setContentsMargins(0, 0, 0, 0)
        outer_lay.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(28, 22, 28, 22)
        lay.setSpacing(14)
        scroll.setWidget(content)
        outer_lay.addWidget(scroll, 1)

        header = Card()
        header_l = QHBoxLayout()
        header_l.setContentsMargins(0, 0, 0, 0)
        left = QVBoxLayout()
        eyebrow = QLabel(f"{APP_TITLE} · {APP_VERSION}")
        eyebrow.setObjectName("Eyebrow")
        t = QLabel(title)
        t.setObjectName("Title")
        s = QLabel(subtitle)
        s.setObjectName("Subtitle")
        s.setWordWrap(True)
        left.addWidget(eyebrow)
        left.addWidget(t)
        left.addWidget(s)
        header_l.addLayout(left, 1)
        header_l.addWidget(self._logo(LAB_LOGO, "SPL", 120, 52))
        header_l.addWidget(self._logo(UOFT_LOGO, "UofT", 142, 52))
        header.layout.addLayout(header_l)
        lay.addWidget(header)

        guide = self._guidance_card(key)
        if guide:
            lay.addWidget(guide)
        outer.body = lay  # type: ignore[attr-defined]
        return outer

    def _guidance_card(self, key: str) -> Card | None:
        info = STAGE_GUIDANCE.get(key)
        if not info:
            return None
        card = Card("Stage objective and decision rule")
        grid = QGridLayout()
        labels = [
            ("Purpose", info["purpose"]),
            ("Inputs", info["inputs"]),
            ("Outputs", info["outputs"]),
            ("Decision", info["decision"]),
        ]
        for r, (lab, txt) in enumerate(labels):
            h = QLabel(lab)
            h.setObjectName("Eyebrow")
            h.setFixedWidth(110)
            v = QLabel(txt)
            v.setWordWrap(True)
            grid.addWidget(h, r, 0, Qt.AlignTop)
            grid.addWidget(v, r, 1)
        card.layout.addLayout(grid)
        return card

    def _workflow_graphic(self) -> Card:
        card = Card("Commercial workflow outline", "The kinematics GUI is organized as a controlled, reproducible pipeline. Early stages inventory and annotate videos; middle stages extract/normalize landmarks and compute features; final stages aggregate, inspect, and report results.")
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        for i, (name, desc, folder) in enumerate(WORKFLOW_STAGES):
            block = QFrame()
            block.setObjectName("StageBlockDone" if i <= 4 else "StageBlockFuture")
            bl = QVBoxLayout(block)
            bl.setContentsMargins(8, 8, 8, 8)
            n = QLabel(f"{i+1}. {name}")
            n.setStyleSheet("font-weight:800;color:#10233F;background:transparent;")
            d = QLabel(desc)
            d.setWordWrap(True)
            d.setStyleSheet("color:#526173;background:transparent;font-size:9pt;")
            f = QLabel(folder)
            f.setStyleSheet("color:#1769AA;background:transparent;font-size:8.5pt;font-weight:700;")
            bl.addWidget(n)
            bl.addWidget(d)
            bl.addWidget(f)
            grid.addWidget(block, i // 5, i % 5)
        card.layout.addLayout(grid)
        note = QLabel("Green blocks are scaffolded and ready for configuration. Gold blocks are outlined now and will receive full backends in staged patches. This preserves the same professional development pattern used for the acoustic GUI.")
        note.setWordWrap(True)
        note.setObjectName("Subtitle")
        card.layout.addWidget(note)
        return card

    def _metric_card(self, label: str, value: str, note: str = "") -> QFrame:
        f = QFrame()
        f.setObjectName("MiniCard")
        lay = QVBoxLayout(f)
        lay.setSpacing(3)
        v = QLabel(value)
        v.setObjectName("MetricValue")
        l = QLabel(label)
        l.setObjectName("MetricLabel")
        l.setWordWrap(True)
        lay.addWidget(v)
        lay.addWidget(l)
        if note:
            n = QLabel(note)
            n.setWordWrap(True)
            n.setObjectName("Subtitle")
            lay.addWidget(n)
        return f

    def _rich_text(self, html: str, minimum_height: int = 150) -> QTextBrowser:
        box = QTextBrowser()
        box.setOpenExternalLinks(True)
        box.setMinimumHeight(minimum_height)
        box.setHtml(html)
        return box

    def _path_row(self, label: str, line: QLineEdit, callback) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label))
        row.addWidget(line, 1)
        btn = QPushButton("Browse…")
        btn.clicked.connect(callback)
        row.addWidget(btn)
        return w

    def _table(self) -> QTableWidget:
        tbl = QTableWidget()
        tbl.setAlternatingRowColors(True)
        tbl.setSortingEnabled(True)
        tbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return tbl

    def _fill_table(self, table: QTableWidget, df: pd.DataFrame, max_rows: int = 1000) -> None:
        table.setSortingEnabled(False)
        show = df.head(max_rows).copy()
        table.setRowCount(len(show))
        table.setColumnCount(len(show.columns))
        table.setHorizontalHeaderLabels([str(c) for c in show.columns])
        for r in range(len(show)):
            for c, col in enumerate(show.columns):
                table.setItem(r, c, QTableWidgetItem("" if pd.isna(show.iloc[r, c]) else str(show.iloc[r, c])))
        table.resizeColumnsToContents()
        table.setSortingEnabled(True)

    # ------------------------------------------------------------------
    # Setup / ingest
    # ------------------------------------------------------------------
    def _choose_input(self):
        d = QFileDialog.getExistingDirectory(self, "Select folder containing videos")
        if d:
            self.input_root = Path(d)
            self.input_line.setText(d)

    def _choose_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.output_root = Path(d)
            self.output_line.setText(d)

    def _setup_page(self) -> QWidget:
        page = self._page("Setup / Ingest", "Select a video folder, recursively discover supported video containers, probe structural video properties, and create an ingest manifest. Videos are read in place; they are not copied or converted at this stage.", "setup")
        page.body.addWidget(self._workflow_graphic())  # type: ignore[attr-defined]
        card = Card("Input / output", "Use a study-level folder when possible. The scanner accepts broad video formats and recursively preserves relative paths.")
        self.input_line = QLineEdit()
        self.output_line = QLineEdit()
        card.layout.addWidget(self._path_row("Video folder", self.input_line, self._choose_input))
        card.layout.addWidget(self._path_row("Output folder", self.output_line, self._choose_output))
        opts = QHBoxLayout()
        self.recursive_check = QCheckBox("Search subfolders recursively")
        self.recursive_check.setChecked(True)
        self.ext_line = QLineEdit(", ".join(sorted(DEFAULT_VIDEO_EXTENSIONS)))
        opts.addWidget(self.recursive_check)
        opts.addWidget(QLabel("Extensions"))
        opts.addWidget(self.ext_line, 1)
        card.layout.addLayout(opts)
        run = QPushButton("Run Video Ingest")
        run.setObjectName("Primary")
        run.clicked.connect(self._run_ingest)
        card.layout.addWidget(run)
        page.body.addWidget(card)  # type: ignore[attr-defined]

        self.ingest_summary = QTextEdit()
        self.ingest_summary.setReadOnly(True)
        self.ingest_table = self._table()
        page.body.addWidget(self.ingest_summary)
        page.body.addWidget(self.ingest_table, 1)
        return page

    def _run_ingest(self):
        try:
            self.input_root = Path(self.input_line.text()).expanduser()
            self.output_root = Path(self.output_line.text()).expanduser()
            exts = frozenset(
                e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
                for e in self.ext_line.text().split(",")
                if e.strip()
            )
            res = run_ingest(
                VideoIngestConfig(
                    input_root=self.input_root,
                    output_root=self.output_root,
                    recursive=self.recursive_check.isChecked(),
                    extensions=exts,
                )
            )
            self.ingest_manifest_csv = Path(res["manifest_csv"])
            df = pd.read_csv(self.ingest_manifest_csv)
            self._fill_table(self.ingest_table, df)
            warnings = int((df.get("warning", pd.Series(dtype=str)).fillna("") != "").sum()) if len(df) else 0
            self.ingest_summary.setText(
                f"Ingest complete.\n\n"
                f"Videos discovered: {res['n_videos']}\n"
                f"Probe warnings: {warnings}\n"
                f"Manifest: {self.ingest_manifest_csv}\n\n"
                "Next: review warnings, then load metadata if available. Broad format support is intentional because browser recordings may arrive as WebM, MP4, MOV, MKV, AVI, WMV, or other containers."
            )
        except Exception as e:
            QMessageBox.critical(self, "Ingest failed", str(e))

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def _metadata_page(self) -> QWidget:
        page = self._page("Metadata", "Load optional participant/session/task/clinical metadata and create a conservative link preview. This mirrors the acoustic GUI metadata philosophy: metadata is helpful but not required for landmark extraction.", "metadata")
        card = Card("Metadata file", "Metadata should be linked conservatively. Ambiguous joins are reported rather than silently forced.")
        self.metadata_line = QLineEdit()
        card.layout.addWidget(self._path_row("CSV/XLSX metadata", self.metadata_line, self._choose_metadata))
        btn = QPushButton("Load and Link Metadata Preview")
        btn.setObjectName("Primary")
        btn.clicked.connect(self._link_metadata)
        card.layout.addWidget(btn)
        page.body.addWidget(card)  # type: ignore[attr-defined]

        help_card = Card("Recommended metadata fields")
        help_card.layout.addWidget(self._rich_text("""
        <ul>
          <li><b>Participant/session keys:</b> subject_id, session_id, visit, recording_date.</li>
          <li><b>Task labels:</b> Bamboo passage, DDK, smile, mouth open/close, sustained posture, other task-specific labels.</li>
          <li><b>Clinical context:</b> diagnosis, ALSFRS-R total, ALSFRS-R bulbar, severity bins, medication state when relevant.</li>
          <li><b>Acquisition context:</b> device, camera type, platform/browser, external webcam indicator, environment notes.</li>
        </ul>
        """, 145))
        page.body.addWidget(help_card)  # type: ignore[attr-defined]

        self.metadata_summary = QTextEdit()
        self.metadata_summary.setReadOnly(True)
        self.metadata_table = self._table()
        page.body.addWidget(self.metadata_summary)
        page.body.addWidget(self.metadata_table, 1)
        return page

    def _choose_metadata(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select metadata CSV/XLSX", filter="Data files (*.csv *.xlsx *.xls)")
        if f:
            self.metadata_path = Path(f)
            self.metadata_line.setText(f)

    def _link_metadata(self):
        try:
            if not self.ingest_manifest_csv:
                raise RuntimeError("Run Setup / Ingest first.")
            if not self.metadata_line.text():
                raise RuntimeError("Select a metadata file first.")
            res = link_metadata(self.ingest_manifest_csv, Path(self.metadata_line.text()), self.output_root or Path.cwd())
            df = pd.read_csv(res["link_preview"])
            self._fill_table(self.metadata_table, df)
            self.metadata_summary.setText(f"Metadata rows: {res['n_metadata_rows']}\nLink mode: {res['link_mode']}\nPreview: {res['link_preview']}")
        except Exception as e:
            QMessageBox.critical(self, "Metadata failed", str(e))

    # ------------------------------------------------------------------
    # Face landmarks
    # ------------------------------------------------------------------
    def _landmarks_page(self) -> QWidget:
        page = self._page("Face Landmarks", "Configure Google MediaPipe Face Landmarker extraction. This scaffold records the run plan now; full extraction will be connected to the uploaded MediaPipe runner in the next patch.", "landmarks")
        card = Card("MediaPipe configuration", "The GUI records all thresholds and model paths so landmark outputs are reproducible and auditable.")
        self.model_line = QLineEdit()
        card.layout.addWidget(self._path_row("FaceLandmarker .task model", self.model_line, self._choose_model))
        grid = QGridLayout()
        self.det_conf = QSpinBox(); self.det_conf.setRange(1, 99); self.det_conf.setValue(50); self.det_conf.setSuffix(" %")
        self.pres_conf = QSpinBox(); self.pres_conf.setRange(1, 99); self.pres_conf.setValue(50); self.pres_conf.setSuffix(" %")
        self.track_conf = QSpinBox(); self.track_conf.setRange(1, 99); self.track_conf.setValue(50); self.track_conf.setSuffix(" %")
        grid.addWidget(QLabel("Detection confidence threshold"), 0, 0); grid.addWidget(self.det_conf, 0, 1)
        grid.addWidget(QLabel("Presence confidence threshold"), 1, 0); grid.addWidget(self.pres_conf, 1, 1)
        grid.addWidget(QLabel("Tracking confidence threshold"), 2, 0); grid.addWidget(self.track_conf, 2, 1)
        card.layout.addLayout(grid)
        card.layout.addWidget(self._rich_text("<b>Quality note.</b> " + mediapipe_capability_note(), 120))
        btn = QPushButton("Write Landmark Extraction Plan")
        btn.setObjectName("Primary")
        btn.clicked.connect(self._write_landmark_plan)
        card.layout.addWidget(btn)
        page.body.addWidget(card)  # type: ignore[attr-defined]

        pipeline = Card("Landmark extraction output contract")
        pipeline.layout.addWidget(self._rich_text("""
        <p>Expected future output for each accepted video:</p>
        <ul>
          <li><code>&lt;video_id&gt;-lmks.csv</code> containing frame, timestamp, face_detected, and x/y/z for each face landmark.</li>
          <li>Frames with no detected face remain in the table as missing coordinates; they are not silently dropped.</li>
          <li>Missing-frame burden, long gaps, tracking jumps, and interpolation burden will feed Video QC.</li>
        </ul>
        """, 145))
        page.body.addWidget(pipeline)  # type: ignore[attr-defined]

        self.landmark_summary = QTextEdit()
        self.landmark_summary.setReadOnly(True)
        self.landmark_table = self._table()
        page.body.addWidget(self.landmark_summary)
        page.body.addWidget(self.landmark_table, 1)
        return page

    def _choose_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select MediaPipe FaceLandmarker model", filter="MediaPipe model (*.task);;All files (*.*)")
        if f:
            self.model_line.setText(f)

    def _write_landmark_plan(self):
        try:
            if not self.ingest_manifest_csv:
                raise RuntimeError("Run Setup / Ingest first.")
            cfg = LandmarkRunConfig(
                manifest_csv=self.ingest_manifest_csv,
                output_root=self.output_root or Path.cwd(),
                model_path=Path(self.model_line.text()) if self.model_line.text() else None,
                selected_preset=self.preset_combo.currentText() if hasattr(self, "preset_combo") else "ALS oral-motor core 15",
                selected_indices=self.landmark_indices,
                min_face_detection_confidence=self.det_conf.value() / 100,
                min_face_presence_confidence=self.pres_conf.value() / 100,
                min_tracking_confidence=self.track_conf.value() / 100,
            )
            res = write_landmark_plan(cfg)
            df = pd.read_csv(res["plan_csv"])
            self._fill_table(self.landmark_table, df)
            self.landmark_summary.setText(f"Landmark plan written for {res['n_videos']} videos.\nPlan: {res['plan_csv']}\nConfig: {res['config_json']}")
        except Exception as e:
            QMessageBox.critical(self, "Landmark planning failed", str(e))

    # ------------------------------------------------------------------
    # Landmark selection
    # ------------------------------------------------------------------
    def _selection_page(self) -> QWidget:
        page = self._page("Landmark Selection", "Select a clinically meaningful subset of face landmarks for downstream kinematic features. The full landmark CSV remains available for audit; selected sets control feature computation defaults.", "selection")
        card = Card("Landmark preset", "Choose a validated starting configuration, then manually edit landmark indices only when the protocol requires it.")
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(LANDMARK_PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._preset_changed)
        card.layout.addWidget(QLabel("Recommended configurations"))
        card.layout.addWidget(self.preset_combo)
        self.preset_info = self._rich_text(PRESET_DESCRIPTIONS[self.preset_combo.currentText()], 90)
        card.layout.addWidget(self.preset_info)
        self.landmark_text = QPlainTextEdit()
        self.landmark_text.setPlainText(", ".join(map(str, self.landmark_indices)))
        card.layout.addWidget(QLabel("Selected landmark indices"))
        card.layout.addWidget(self.landmark_text)
        row = QHBoxLayout()
        apply = QPushButton("Use These Landmarks")
        apply.setObjectName("Primary")
        apply.clicked.connect(self._apply_landmark_text)
        row.addWidget(apply)
        row.addStretch(1)
        card.layout.addLayout(row)
        page.body.addWidget(card)  # type: ignore[attr-defined]

        preset_card = Card("Preset comparison")
        df = pd.DataFrame([
            {"preset": k, "n_landmarks": len(v), "indices": ", ".join(map(str, v)), "recommended_use": PRESET_DESCRIPTIONS.get(k, "")}
            for k, v in LANDMARK_PRESETS.items()
        ])
        self.preset_table = self._table()
        self._fill_table(self.preset_table, df)
        preset_card.layout.addWidget(self.preset_table)
        page.body.addWidget(preset_card, 1)  # type: ignore[attr-defined]
        return page

    def _preset_changed(self, name: str):
        self.landmark_indices = LANDMARK_PRESETS[name]
        self.landmark_text.setPlainText(", ".join(map(str, self.landmark_indices)))
        if hasattr(self, "preset_info"):
            self.preset_info.setHtml(PRESET_DESCRIPTIONS.get(name, ""))

    def _apply_landmark_text(self):
        try:
            self.landmark_indices = parse_int_list(self.landmark_text.toPlainText())
            QMessageBox.information(self, "Landmarks updated", f"Selected {len(self.landmark_indices)} unique landmarks.")
        except Exception as e:
            QMessageBox.critical(self, "Invalid landmarks", str(e))

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def _normalization_page(self) -> QWidget:
        page = self._page("Normalization", "Choose how frame-level landmark coordinates should be scaled/stabilized before feature computation. Intercanthal normalization is the recommended default for mouth/jaw kinematics.", "normalization")
        card = Card("Normalization method", "Normalization is not cosmetic: it defines the measurement scale for all downstream kinematic features.")
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(list(NORMALIZATION_METHODS.keys()))
        self.norm_desc = self._rich_text(NORMALIZATION_METHODS[self.norm_combo.currentText()], 100)
        self.norm_combo.currentTextChanged.connect(lambda n: self.norm_desc.setHtml(NORMALIZATION_METHODS[n]))
        card.layout.addWidget(self.norm_combo)
        card.layout.addWidget(self.norm_desc)
        btn = QPushButton("Save Normalization Config")
        btn.setObjectName("Primary")
        btn.clicked.connect(self._save_norm)
        card.layout.addWidget(btn)
        page.body.addWidget(card)  # type: ignore[attr-defined]

        method_card = Card("Method comparison")
        norm_df = pd.DataFrame([{"method": k, "interpretation": v} for k, v in NORMALIZATION_METHODS.items()])
        self.norm_table = self._table()
        self._fill_table(self.norm_table, norm_df)
        method_card.layout.addWidget(self.norm_table)
        page.body.addWidget(method_card, 1)  # type: ignore[attr-defined]
        return page

    def _save_norm(self):
        try:
            path = write_normalization_config(self.output_root or Path.cwd(), self.norm_combo.currentText())
            QMessageBox.information(self, "Normalization saved", f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    # ------------------------------------------------------------------
    # Video QC outline
    # ------------------------------------------------------------------
    def _qc_page(self) -> QWidget:
        page = self._page("Video QC", "Planned multidimensional video/landmark QC. This will parallel acoustic QC but for visual acquisition and landmark-tracking validity.", "qc")
        card = Card("Video QC families", "These families convert raw tracking problems into interpretable covariates and flags.")
        qc_df = pd.DataFrame([
            {"QC family": "Decode/container", "What it checks": "unreadable videos, variable fps, corrupted frames, duration mismatch", "Why it matters": "bad timing or missing frames distort derivatives and aggregation"},
            {"QC family": "Face visibility", "What it checks": "no-face frames, long gaps, partial face, occlusion, off-screen face", "Why it matters": "missing landmarks drive interpolation and feature loss"},
            {"QC family": "Pose/head motion", "What it checks": "excessive yaw/pitch/roll, rapid head movement, unstable camera", "Why it matters": "2D/3D landmark distances can change due to pose rather than articulator movement"},
            {"QC family": "Illumination", "What it checks": "underexposure, overexposure, flicker, shadows, low contrast", "Why it matters": "landmark detection and stability can degrade under poor lighting"},
            {"QC family": "Landmark stability", "What it checks": "jitter, impossible jumps, asymmetric tracking failure, high interpolation burden", "Why it matters": "movement features may reflect tracking noise"},
            {"QC family": "Task/adherence", "What it checks": "wrong task, face not visible during target movement, non-target events", "Why it matters": "feature interpretation depends on correct task context"},
        ])
        self.qc_outline_table = self._table()
        self._fill_table(self.qc_outline_table, qc_df)
        card.layout.addWidget(self.qc_outline_table)
        page.body.addWidget(card, 1)  # type: ignore[attr-defined]
        return page

    # ------------------------------------------------------------------
    # Feature computation outline
    # ------------------------------------------------------------------
    def _features_page(self) -> QWidget:
        page = self._page("Feature Computation", "Planned frame-level and movement-level kinematic feature computation. Your uploaded scripts will be connected here: cleaning, smoothing, movement segmentation, geometry, kinematic derivatives, and per-frame timeseries export.", "features")
        card = Card("Feature computation families")
        feature_df = pd.DataFrame([
            {"family": "Mouth opening / jaw displacement", "examples": "vertical lower-lip/jaw displacement, aperture range", "primary signal": "normalized distance trajectory"},
            {"family": "Lip spread / aspect ratio", "examples": "horizontal spread, vertical-to-horizontal aperture ratio", "primary signal": "commissure and lip aperture geometry"},
            {"family": "Jaw lateralization", "examples": "left/right jaw deviation ratio", "primary signal": "distance to eye/canthus anchors"},
            {"family": "Lip symmetry / coordination", "examples": "left-right commissure symmetry, cross-correlation", "primary signal": "paired lateral trajectories"},
            {"family": "Kinematic derivatives", "examples": "velocity, acceleration, path length, range of motion", "primary signal": "time derivative of cleaned trajectories"},
            {"family": "Movement segmentation", "examples": "opening/closing cycles, peak/trough windows", "primary signal": "task-specific repetitive motion"},
        ])
        self.feature_outline_table = self._table()
        self._fill_table(self.feature_outline_table, feature_df)
        card.layout.addWidget(self.feature_outline_table)
        page.body.addWidget(card, 1)  # type: ignore[attr-defined]

        guard = Card("Scientific guardrails")
        guard.layout.addWidget(self._rich_text("""
        <ul>
          <li>Frame-level trajectories are not scalar biomarkers until cleaning, normalization, QC review, feature computation, and aggregation are documented.</li>
          <li>Velocity and acceleration are highly sensitive to fps, smoothing, tracking jitter, and interpolation burden.</li>
          <li>Movement-segmented features should only be used for tasks with interpretable repeated movements.</li>
          <li>Feature computation should always export provenance: selected landmarks, normalization, smoothing, segmentation settings, and aggregation policy.</li>
        </ul>
        """, 150))
        page.body.addWidget(guard)  # type: ignore[attr-defined]
        return page

    # ------------------------------------------------------------------
    # Aggregation outline
    # ------------------------------------------------------------------
    def _aggregation_page(self) -> QWidget:
        page = self._page("Temporal Aggregation", "Choose how frame-level kinematic timeseries are collapsed into one scalar row per video without hiding clinically meaningful variability.", "aggregation")
        card = Card("Aggregation profile", "A scalar feature table is useful, but the collapse policy must be explicit because facial kinematics are naturally time-varying.")
        self.agg_combo = QComboBox()
        self.agg_combo.addItems(list(AGGREGATION_PROFILES.keys()))
        self.agg_desc = self._rich_text(AGGREGATION_PROFILES[self.agg_combo.currentText()], 95)
        self.agg_combo.currentTextChanged.connect(lambda n: self.agg_desc.setHtml(AGGREGATION_PROFILES[n]))
        card.layout.addWidget(self.agg_combo)
        card.layout.addWidget(self.agg_desc)
        page.body.addWidget(card)  # type: ignore[attr-defined]

        agg_card = Card("Aggregation strategy comparison")
        agg_df = pd.DataFrame([{"profile": k, "description": v} for k, v in AGGREGATION_PROFILES.items()])
        self.agg_table = self._table()
        self._fill_table(self.agg_table, agg_df)
        agg_card.layout.addWidget(self.agg_table)
        page.body.addWidget(agg_card, 1)  # type: ignore[attr-defined]
        return page

    # ------------------------------------------------------------------
    # Inspector / reports
    # ------------------------------------------------------------------
    def _inspector_page(self) -> QWidget:
        page = self._page("Data Inspector", "Open and inspect stage outputs. This lightweight first version focuses on ingest, metadata, and landmark planning tables.", "inspector")
        card = Card("Table preview")
        row = QHBoxLayout()
        btn_ing = QPushButton("Preview ingest manifest")
        btn_ing.clicked.connect(lambda: self._preview_csv(self.ingest_manifest_csv))
        btn_land = QPushButton("Preview landmark plan")
        btn_land.clicked.connect(self._preview_landmark_plan)
        row.addWidget(btn_ing)
        row.addWidget(btn_land)
        row.addStretch(1)
        card.layout.addLayout(row)
        self.inspect_table = self._table()
        card.layout.addWidget(self.inspect_table)
        page.body.addWidget(card, 1)  # type: ignore[attr-defined]
        return page

    def _preview_csv(self, path: Path | None):
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "No table", "The requested table does not exist yet.")
            return
        self._fill_table(self.inspect_table, pd.read_csv(path))

    def _preview_landmark_plan(self):
        if not self.output_root:
            QMessageBox.warning(self, "No output", "Select an output folder first.")
            return
        candidates = sorted((self.output_root / "kinematics" / "002_landmarks" / "tables").glob("landmark_extraction_plan.csv"))
        self._preview_csv(candidates[0] if candidates else None)

    def _reports_page(self) -> QWidget:
        page = self._page("Reports & Outputs", "Create an initial HTML scaffold report and open the output folder. Full reports will expand as each kinematic computation stage is implemented.", "reports")
        card = Card("Report actions")
        row = QHBoxLayout()
        b1 = QPushButton("Create Scaffold Report")
        b1.setObjectName("Primary")
        b1.clicked.connect(self._write_report)
        b2 = QPushButton("Open HTML Report")
        b2.clicked.connect(self._open_report)
        b3 = QPushButton("Open Output Folder")
        b3.clicked.connect(self._open_output)
        row.addWidget(b1)
        row.addWidget(b2)
        row.addWidget(b3)
        row.addStretch(1)
        card.layout.addLayout(row)
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        card.layout.addWidget(self.report_text)
        page.body.addWidget(card)  # type: ignore[attr-defined]

        doc = Card("Commercial/document-control notes")
        doc.layout.addWidget(self._rich_text(f"""
        <p><b>Copyright / ownership.</b> {COPYRIGHT_TEXT}</p>
        <p><b>Intended use.</b> The kinematics GUI is a research workflow for video-based facial/oral kinematic feature preparation. It does not provide clinical diagnosis or automated clinical decision-making.</p>
        <p><b>Reproducibility.</b> Every implemented stage should write configuration, manifest, and provenance files before downstream analysis.</p>
        <p><b>Next backend milestones.</b> MediaPipe extraction, landmark/video QC, feature computation, aggregation/export, and then integration with the already completed Feature Analysis GUI.</p>
        """, 190))
        page.body.addWidget(doc)  # type: ignore[attr-defined]
        return page

    def _write_report(self):
        try:
            self.last_report_html = write_scaffold_report(self.output_root or Path.cwd())
            self.report_text.setText(f"Report written:\n{self.last_report_html}")
            QMessageBox.information(self, "Report complete", f"Report written:\n{self.last_report_html}")
        except Exception as e:
            QMessageBox.critical(self, "Report failed", str(e))

    def _open_report(self):
        if self.last_report_html and self.last_report_html.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_report_html)))
        else:
            QMessageBox.warning(self, "No report", "Create the scaffold report first.")

    def _open_output(self):
        path = self.output_root or Path.cwd()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def main() -> int:
    app = QApplication(sys.argv)
    set_app_style(app)
    win = KinematicsGUI()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
