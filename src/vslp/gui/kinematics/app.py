"""VSLP Kinematics GUI scaffold.

This first kinematics GUI pass mirrors the acoustic workflow structure while
keeping heavy MediaPipe/video feature computation behind staged buttons. It is
safe to launch before mediapipe is installed; landmark extraction is currently a
configuration/planning stage with later patches plugging in the full runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
        QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow,
        QMessageBox, QPushButton, QPlainTextEdit, QSizePolicy, QSpinBox,
        QStackedWidget, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout,
        QWidget,
    )
    from PySide6.QtCore import QUrl
except Exception as exc:  # pragma: no cover
    raise SystemExit("PySide6 is required to launch the kinematics GUI. Install with: pip install -e '.[gui]'") from exc

import pandas as pd

from vslp.analysis.kinematics import (
    AGGREGATION_PROFILES, LANDMARK_PRESETS, NORMALIZATION_METHODS,
    LandmarkRunConfig, VideoIngestConfig, link_metadata, mediapipe_capability_note,
    run_ingest, write_landmark_plan, write_normalization_config, write_scaffold_report,
)
from vslp.analysis.kinematics.schemas import DEFAULT_VIDEO_EXTENSIONS, parse_int_list

APP_TITLE = "VSLP Kinematics Pipeline"
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


def set_app_style(app: QApplication) -> None:
    app.setStyleSheet("""
    QWidget { background: #F5F7FB; color: #172033; font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; }
    QMainWindow { background: #F5F7FB; }
    QLabel#Title { font-size: 20pt; font-weight: 700; color: #10233F; }
    QLabel#Subtitle { color: #526173; font-size: 10pt; }
    QLabel#CardTitle { font-size: 12pt; font-weight: 700; color: #12345A; }
    QFrame#Card, QGroupBox { background: #FFFFFF; border: 1px solid #DCE5F2; border-radius: 14px; padding: 12px; }
    QGroupBox::title { color: #12345A; font-weight: 700; subcontrol-origin: margin; left: 12px; padding: 0 6px; }
    QPushButton { background: #FFFFFF; color: #10233F; border: 1px solid #B8C7DA; border-radius: 9px; padding: 8px 12px; font-weight: 600; }
    QPushButton:hover { background: #EEF6FF; border-color: #78A7D8; }
    QPushButton:pressed { background: #DCEEFF; color: #10233F; }
    QPushButton#Primary { background: #1769AA; color: #FFFFFF; border-color: #1769AA; }
    QPushButton#Primary:hover { background: #0F5B96; color: #FFFFFF; }
    QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox { background: #FFFFFF; color: #172033; border: 1px solid #B8C7DA; border-radius: 8px; padding: 6px; selection-background-color: #CFE5FF; selection-color: #172033; }
    QComboBox QAbstractItemView { background: #FFFFFF; color: #172033; selection-background-color: #D9ECFF; selection-color: #172033; border: 1px solid #B8C7DA; }
    QTableWidget { background: #FFFFFF; alternate-background-color: #F6FAFF; gridline-color: #E1E9F3; color: #172033; selection-background-color: #D9ECFF; selection-color: #172033; border: 1px solid #DCE5F2; border-radius: 8px; }
    QHeaderView::section { background: #EAF1F8; color: #10233F; border: 0px; border-right: 1px solid #D2DEEA; padding: 6px; font-weight: 700; }
    QListWidget#Nav { background: #10233F; color: #DDEBFA; border: 0px; padding: 8px; }
    QListWidget#Nav::item { padding: 11px 12px; border-radius: 8px; margin: 3px; }
    QListWidget#Nav::item:selected { background: #1E78BE; color: #FFFFFF; }
    QMessageBox { background: #FFFFFF; color: #172033; }
    QMessageBox QLabel { background: #FFFFFF; color: #172033; }
    QMessageBox QPushButton { background: #FFFFFF; color: #172033; border: 1px solid #B8C7DA; border-radius: 8px; padding: 7px 14px; }
    """)


class Card(QFrame):
    def __init__(self, title: str | None = None):
        super().__init__()
        self.setObjectName("Card")
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)
        if title:
            lab = QLabel(title)
            lab.setObjectName("CardTitle")
            self.layout.addWidget(lab)


class KinematicsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1480, 900)
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
        self.nav.setFixedWidth(245)
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

    def _page(self, title: str, subtitle: str) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(26, 22, 26, 22)
        lay.setSpacing(14)
        t = QLabel(title); t.setObjectName("Title")
        s = QLabel(subtitle); s.setObjectName("Subtitle"); s.setWordWrap(True)
        lay.addWidget(t); lay.addWidget(s)
        page.body = lay  # type: ignore[attr-defined]
        return page

    def _path_row(self, label: str, line: QLineEdit, callback) -> QWidget:
        w = QWidget(); row = QHBoxLayout(w); row.setContentsMargins(0,0,0,0)
        row.addWidget(QLabel(label)); row.addWidget(line, 1)
        btn = QPushButton("Browse…"); btn.clicked.connect(callback); row.addWidget(btn)
        return w

    def _table(self) -> QTableWidget:
        tbl = QTableWidget(); tbl.setAlternatingRowColors(True); tbl.setSortingEnabled(True)
        tbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return tbl

    def _fill_table(self, table: QTableWidget, df: pd.DataFrame, max_rows: int = 1000) -> None:
        table.setSortingEnabled(False)
        show = df.head(max_rows).copy()
        table.setRowCount(len(show)); table.setColumnCount(len(show.columns)); table.setHorizontalHeaderLabels([str(c) for c in show.columns])
        for r in range(len(show)):
            for c, col in enumerate(show.columns):
                table.setItem(r, c, QTableWidgetItem("" if pd.isna(show.iloc[r, c]) else str(show.iloc[r, c])))
        table.resizeColumnsToContents(); table.setSortingEnabled(True)

    def _choose_input(self):
        d = QFileDialog.getExistingDirectory(self, "Select folder containing videos")
        if d:
            self.input_root = Path(d); self.input_line.setText(d)

    def _choose_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.output_root = Path(d); self.output_line.setText(d)

    def _setup_page(self) -> QWidget:
        page = self._page("Setup / Ingest", "Select a video folder, recursively discover supported video containers, probe structural video properties, and create an ingest manifest. Videos are read in place; they are not copied or converted at this stage.")
        card = Card("Input / output")
        self.input_line = QLineEdit(); self.output_line = QLineEdit()
        card.layout.addWidget(self._path_row("Video folder", self.input_line, self._choose_input))
        card.layout.addWidget(self._path_row("Output folder", self.output_line, self._choose_output))
        opts = QHBoxLayout(); self.recursive_check = QCheckBox("Search subfolders recursively"); self.recursive_check.setChecked(True)
        self.ext_line = QLineEdit(", ".join(sorted(DEFAULT_VIDEO_EXTENSIONS)))
        opts.addWidget(self.recursive_check); opts.addWidget(QLabel("Extensions")); opts.addWidget(self.ext_line, 1)
        card.layout.addLayout(opts)
        run = QPushButton("Run Video Ingest"); run.setObjectName("Primary"); run.clicked.connect(self._run_ingest)
        card.layout.addWidget(run)
        page.body.addWidget(card)  # type: ignore[attr-defined]
        self.ingest_summary = QTextEdit(); self.ingest_summary.setReadOnly(True)
        self.ingest_table = self._table()
        page.body.addWidget(self.ingest_summary)
        page.body.addWidget(self.ingest_table, 1)
        return page

    def _run_ingest(self):
        try:
            self.input_root = Path(self.input_line.text()).expanduser()
            self.output_root = Path(self.output_line.text()).expanduser()
            exts = frozenset(e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}" for e in self.ext_line.text().split(",") if e.strip())
            res = run_ingest(VideoIngestConfig(input_root=self.input_root, output_root=self.output_root, recursive=self.recursive_check.isChecked(), extensions=exts))
            self.ingest_manifest_csv = Path(res["manifest_csv"])
            df = pd.read_csv(self.ingest_manifest_csv)
            self._fill_table(self.ingest_table, df)
            self.ingest_summary.setText(f"Ingest complete. Videos discovered: {res['n_videos']}\nManifest: {self.ingest_manifest_csv}\n\nSupported containers are intentionally broad; ffprobe warnings remain visible for manual review.")
        except Exception as e:
            QMessageBox.critical(self, "Ingest failed", str(e))

    def _metadata_page(self) -> QWidget:
        page = self._page("Metadata", "Load optional participant/session/task/clinical metadata and create a conservative link preview. This mirrors the acoustic GUI metadata philosophy: metadata is helpful but not required for landmark extraction.")
        card = Card("Metadata file")
        self.metadata_line = QLineEdit()
        card.layout.addWidget(self._path_row("CSV/XLSX metadata", self.metadata_line, self._choose_metadata))
        btn = QPushButton("Load and Link Metadata Preview"); btn.setObjectName("Primary"); btn.clicked.connect(self._link_metadata)
        card.layout.addWidget(btn)
        page.body.addWidget(card)  # type: ignore[attr-defined]
        self.metadata_summary = QTextEdit(); self.metadata_summary.setReadOnly(True)
        self.metadata_table = self._table()
        page.body.addWidget(self.metadata_summary); page.body.addWidget(self.metadata_table, 1)
        return page

    def _choose_metadata(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select metadata CSV/XLSX", filter="Data files (*.csv *.xlsx *.xls)")
        if f:
            self.metadata_path = Path(f); self.metadata_line.setText(f)

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

    def _landmarks_page(self) -> QWidget:
        page = self._page("Face Landmarks", "Configure Google MediaPipe Face Landmarker extraction. The scaffold records the run plan now; full extraction will be connected to the uploaded MediaPipe runner in the next patch.")
        card = Card("MediaPipe configuration")
        self.model_line = QLineEdit(); card.layout.addWidget(self._path_row("FaceLandmarker .task model", self.model_line, self._choose_model))
        grid = QGridLayout()
        self.det_conf = QSpinBox(); self.det_conf.setRange(1,99); self.det_conf.setValue(50); self.det_conf.setSuffix(" %")
        self.pres_conf = QSpinBox(); self.pres_conf.setRange(1,99); self.pres_conf.setValue(50); self.pres_conf.setSuffix(" %")
        self.track_conf = QSpinBox(); self.track_conf.setRange(1,99); self.track_conf.setValue(50); self.track_conf.setSuffix(" %")
        grid.addWidget(QLabel("Detection confidence threshold"),0,0); grid.addWidget(self.det_conf,0,1)
        grid.addWidget(QLabel("Presence confidence threshold"),1,0); grid.addWidget(self.pres_conf,1,1)
        grid.addWidget(QLabel("Tracking confidence threshold"),2,0); grid.addWidget(self.track_conf,2,1)
        card.layout.addLayout(grid)
        note = QTextEdit(); note.setReadOnly(True); note.setText(mediapipe_capability_note())
        card.layout.addWidget(note)
        btn = QPushButton("Write Landmark Extraction Plan"); btn.setObjectName("Primary"); btn.clicked.connect(self._write_landmark_plan)
        card.layout.addWidget(btn)
        page.body.addWidget(card)  # type: ignore[attr-defined]
        self.landmark_summary = QTextEdit(); self.landmark_summary.setReadOnly(True)
        self.landmark_table = self._table()
        page.body.addWidget(self.landmark_summary); page.body.addWidget(self.landmark_table, 1)
        return page

    def _choose_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select MediaPipe FaceLandmarker model", filter="MediaPipe model (*.task);;All files (*.*)")
        if f: self.model_line.setText(f)

    def _write_landmark_plan(self):
        try:
            if not self.ingest_manifest_csv: raise RuntimeError("Run Setup / Ingest first.")
            cfg = LandmarkRunConfig(
                manifest_csv=self.ingest_manifest_csv,
                output_root=self.output_root or Path.cwd(),
                model_path=Path(self.model_line.text()) if self.model_line.text() else None,
                selected_preset=self.preset_combo.currentText() if hasattr(self, "preset_combo") else "ALS oral-motor core 15",
                selected_indices=self.landmark_indices,
                min_face_detection_confidence=self.det_conf.value()/100,
                min_face_presence_confidence=self.pres_conf.value()/100,
                min_tracking_confidence=self.track_conf.value()/100,
            )
            res = write_landmark_plan(cfg)
            df = pd.read_csv(res["plan_csv"])
            self._fill_table(self.landmark_table, df)
            self.landmark_summary.setText(f"Landmark plan written for {res['n_videos']} videos.\nPlan: {res['plan_csv']}\nConfig: {res['config_json']}")
        except Exception as e:
            QMessageBox.critical(self, "Landmark planning failed", str(e))

    def _selection_page(self) -> QWidget:
        page = self._page("Landmark Selection", "Select a clinically meaningful subset of face landmarks for downstream kinematic features. The full landmark CSV remains available for audit; selected sets control feature computation defaults.")
        card = Card("Landmark preset")
        self.preset_combo = QComboBox(); self.preset_combo.addItems(list(LANDMARK_PRESETS.keys())); self.preset_combo.currentTextChanged.connect(self._preset_changed)
        card.layout.addWidget(QLabel("Recommended configurations")); card.layout.addWidget(self.preset_combo)
        self.landmark_text = QPlainTextEdit(); self.landmark_text.setPlainText(", ".join(map(str, self.landmark_indices)))
        card.layout.addWidget(QLabel("Selected landmark indices")); card.layout.addWidget(self.landmark_text)
        row = QHBoxLayout()
        apply = QPushButton("Use These Landmarks"); apply.setObjectName("Primary"); apply.clicked.connect(self._apply_landmark_text)
        row.addWidget(apply); row.addStretch(1); card.layout.addLayout(row)
        guidance = QTextEdit(); guidance.setReadOnly(True); guidance.setText("Default sets are starting configurations for visible facial/oral kinematics. They should be validated against task, diagnosis, QC, and feature stability. For ALS oral-motor analysis, the core set emphasizes mouth aperture, lip spread, commissures, lower face/jaw, and eye/canthus anchors for normalization.")
        card.layout.addWidget(guidance)
        page.body.addWidget(card, 1)  # type: ignore[attr-defined]
        return page

    def _preset_changed(self, name: str):
        self.landmark_indices = LANDMARK_PRESETS[name]
        self.landmark_text.setPlainText(", ".join(map(str, self.landmark_indices)))

    def _apply_landmark_text(self):
        try:
            self.landmark_indices = parse_int_list(self.landmark_text.toPlainText())
            QMessageBox.information(self, "Landmarks updated", f"Selected {len(self.landmark_indices)} unique landmarks.")
        except Exception as e:
            QMessageBox.critical(self, "Invalid landmarks", str(e))

    def _normalization_page(self) -> QWidget:
        page = self._page("Normalization", "Choose how frame-level landmark coordinates should be scaled/stabilized before feature computation. Intercanthal normalization is the recommended default for mouth/jaw kinematics.")
        card = Card("Normalization method")
        self.norm_combo = QComboBox(); self.norm_combo.addItems(list(NORMALIZATION_METHODS.keys()))
        self.norm_desc = QTextEdit(); self.norm_desc.setReadOnly(True)
        self.norm_combo.currentTextChanged.connect(lambda n: self.norm_desc.setText(NORMALIZATION_METHODS[n]))
        self.norm_desc.setText(NORMALIZATION_METHODS[self.norm_combo.currentText()])
        card.layout.addWidget(self.norm_combo); card.layout.addWidget(self.norm_desc)
        btn = QPushButton("Save Normalization Config"); btn.setObjectName("Primary"); btn.clicked.connect(self._save_norm)
        card.layout.addWidget(btn)
        page.body.addWidget(card, 1)  # type: ignore[attr-defined]
        return page

    def _save_norm(self):
        try:
            path = write_normalization_config(self.output_root or Path.cwd(), self.norm_combo.currentText())
            QMessageBox.information(self, "Normalization saved", f"Saved: {path}")
        except Exception as e: QMessageBox.critical(self, "Save failed", str(e))

    def _qc_page(self) -> QWidget:
        page = self._page("Video QC", "Placeholder for multidimensional video/landmark QC. This will parallel acoustic QC but for visual acquisition and landmark-tracking validity.")
        card = Card("Planned QC families")
        txt = QTextEdit(); txt.setReadOnly(True); txt.setText("""
Planned video QC families:

1. Decode/container QC: unreadable videos, variable fps, corrupted frames, duration mismatch.
2. Face visibility QC: no-face frames, long landmark gaps, partial face, occlusion, off-screen face.
3. Pose/head-motion QC: excessive yaw/pitch/roll, rapid head movement, unstable camera.
4. Illumination QC: underexposure, overexposure, flicker, low contrast, shadows.
5. Landmark stability QC: jitter, impossible jumps, asymmetric tracking failure, high interpolation burden.
6. Task/adherence QC: wrong task, face not visible during target movement, non-target events.

This stage will not automatically exclude videos. It will create visual-quality covariates and flags for later feature interpretation.
""")
        card.layout.addWidget(txt); page.body.addWidget(card, 1)  # type: ignore[attr-defined]
        return page

    def _features_page(self) -> QWidget:
        page = self._page("Feature Computation", "Placeholder for frame-level and movement-level kinematic feature computation. Your uploaded scripts will be connected here: cleaning, smoothing, movement segmentation, geometry, kinematic derivatives, and per-frame timeseries export.")
        card = Card("Planned feature families")
        txt = QTextEdit(); txt.setReadOnly(True); txt.setText("""
Planned computation families:

- Mouth opening / jaw displacement
- Lip spread and lip aspect ratio
- Jaw lateralization
- Lip symmetry and commissure coordination
- Velocity, acceleration, path length, range of motion
- Movement segmentation: open/close cycles when task supports it
- Frame-level timeseries outputs plus per-video scalar summaries

The uploaded feature code already includes cleaning, smoothing, movement segmentation, normalized distance geometry, velocity/acceleration, path length, ROM, and summary statistics. This GUI stage will become the user-facing control layer for those functions.
""")
        card.layout.addWidget(txt); page.body.addWidget(card, 1)  # type: ignore[attr-defined]
        return page

    def _aggregation_page(self) -> QWidget:
        page = self._page("Temporal Aggregation", "Choose how frame-level kinematic timeseries are collapsed into one scalar row per video without hiding clinically meaningful variability.")
        card = Card("Aggregation profile")
        self.agg_combo = QComboBox(); self.agg_combo.addItems(list(AGGREGATION_PROFILES.keys()))
        self.agg_desc = QTextEdit(); self.agg_desc.setReadOnly(True)
        self.agg_combo.currentTextChanged.connect(lambda n: self.agg_desc.setText(AGGREGATION_PROFILES[n]))
        self.agg_desc.setText(AGGREGATION_PROFILES[self.agg_combo.currentText()])
        card.layout.addWidget(self.agg_combo); card.layout.addWidget(self.agg_desc)
        page.body.addWidget(card, 1)  # type: ignore[attr-defined]
        return page

    def _inspector_page(self) -> QWidget:
        page = self._page("Data Inspector", "Open and inspect stage outputs. This lightweight first version focuses on ingest, metadata, and landmark planning tables.")
        card = Card("Table preview")
        row = QHBoxLayout()
        btn_ing = QPushButton("Open ingest manifest"); btn_ing.clicked.connect(lambda: self._preview_csv(self.ingest_manifest_csv))
        btn_land = QPushButton("Open landmark plan"); btn_land.clicked.connect(self._preview_landmark_plan)
        row.addWidget(btn_ing); row.addWidget(btn_land); row.addStretch(1)
        card.layout.addLayout(row)
        self.inspect_table = self._table(); card.layout.addWidget(self.inspect_table)
        page.body.addWidget(card, 1)  # type: ignore[attr-defined]
        return page

    def _preview_csv(self, path: Path | None):
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "No table", "The requested table does not exist yet."); return
        self._fill_table(self.inspect_table, pd.read_csv(path))

    def _preview_landmark_plan(self):
        if not self.output_root:
            QMessageBox.warning(self, "No output", "Select an output folder first."); return
        candidates = sorted((self.output_root / "kinematics" / "002_landmarks" / "tables").glob("landmark_extraction_plan.csv"))
        self._preview_csv(candidates[0] if candidates else None)

    def _reports_page(self) -> QWidget:
        page = self._page("Reports & Outputs", "Create an initial HTML scaffold report and open the output folder. Full reports will expand as each kinematic computation stage is implemented.")
        card = Card("Report actions")
        row = QHBoxLayout()
        b1 = QPushButton("Create Scaffold Report"); b1.setObjectName("Primary"); b1.clicked.connect(self._write_report)
        b2 = QPushButton("Open HTML Report"); b2.clicked.connect(self._open_report)
        b3 = QPushButton("Open Output Folder"); b3.clicked.connect(self._open_output)
        row.addWidget(b1); row.addWidget(b2); row.addWidget(b3); row.addStretch(1)
        card.layout.addLayout(row)
        self.report_text = QTextEdit(); self.report_text.setReadOnly(True); card.layout.addWidget(self.report_text)
        page.body.addWidget(card, 1)  # type: ignore[attr-defined]
        return page

    def _write_report(self):
        try:
            self.last_report_html = write_scaffold_report(self.output_root or Path.cwd())
            self.report_text.setText(f"Report written:\n{self.last_report_html}")
            QMessageBox.information(self, "Report complete", f"Report written:\n{self.last_report_html}")
        except Exception as e: QMessageBox.critical(self, "Report failed", str(e))

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
    win = KinematicsGUI(); win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
