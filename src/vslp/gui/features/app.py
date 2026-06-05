from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from PySide6.QtCore import Qt, QThread, Signal, QSize
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
        QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QSplitter, QTabWidget,
        QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget, QMainWindow,
        QHeaderView, QAbstractItemView, QSizePolicy, QScrollArea
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PySide6 is required for the Feature Analysis GUI. Install the gui extras first.") from exc

from vslp.analysis.features import AnalysisInputs, run_feature_analysis
from vslp.analysis.features.loaders import read_table
from vslp.analysis.features.column_mapping import infer_column_roles


NAVY = "#07172b"
NAVY2 = "#0b223d"
NAVY3 = "#123b63"
TEXT = "#e8edf7"
MUTED = "#a8bad6"
TEAL = "#14b8a6"
WHITE = "#ffffff"
INK = "#0b1f3a"

APP_STYLE = f"""
QMainWindow, QWidget {{ background: {NAVY}; color: {TEXT}; font-family: Arial; font-size: 13px; }}
QFrame#BrandBar {{ background: #ffffff; border: 1px solid #d6dbe4; border-radius: 10px; }}
QFrame#SideBar {{ background: #051124; border: 1px solid #17385c; border-radius: 12px; }}
QFrame#Card {{ background: #0a1b32; border: 1px solid #254161; border-radius: 12px; }}
QLabel#AppTitle {{ color: #ffffff; font-size: 20px; font-weight: 700; }}
QLabel#AppSubTitle {{ color: #bdd0ee; font-size: 12px; }}
QLabel#BrandTitle {{ color: {INK}; font-size: 19px; font-weight: 700; }}
QLabel#BrandSubTitle {{ color: #344054; font-size: 12px; }}
QLabel#SectionTitle {{ color: #ffffff; font-size: 16px; font-weight: 700; }}
QLabel#FinePrint {{ color: #9fb4d4; font-size: 11px; }}
QLabel#InfoText {{ color: #c8d7ee; font-size: 12px; }}
QLabel#DarkOnLight {{ color: {INK}; font-size: 12px; }}
QGroupBox {{ border: 1px solid #254161; border-radius: 10px; margin-top: 10px; padding: 12px; font-weight: 600; color: #e8edf7; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #b7cdf0; }}
QPushButton {{ background: {NAVY3}; color: #ffffff; border: 1px solid #2d5f91; border-radius: 8px; padding: 8px 12px; min-height: 26px; }}
QPushButton:hover {{ background: #174c7e; }}
QPushButton#Primary {{ background: #0f766e; border-color: {TEAL}; font-weight: 700; }}
QPushButton#NavButton {{ text-align: left; background: #071b33; color: #c7d7ef; border: 1px solid #183a5d; border-radius: 8px; padding: 8px 10px; }}
QPushButton#NavButton:checked {{ background: #0f766e; color: white; border-color: {TEAL}; font-weight: 700; }}
QLineEdit, QComboBox, QTextEdit, QTableWidget, QListWidget {{ background: {NAVY2}; color: {TEXT}; border: 1px solid #254161; border-radius: 6px; padding: 5px; selection-background-color: #0f766e; }}
QComboBox {{ min-height: 30px; min-width: 170px; padding-left: 8px; }}
QComboBox QAbstractItemView {{ background: #ffffff; color: {INK}; border: 1px solid #8ca3c3; selection-background-color: #dbeafe; selection-color: {INK}; padding: 4px; outline: 0px; }}
QTabWidget::pane {{ border: 1px solid #254161; border-radius: 8px; top: -1px; }}
QTabBar::tab {{ background: {NAVY2}; color: #c6d4e8; padding: 8px 12px; border-top-left-radius: 7px; border-top-right-radius: 7px; min-width: 110px; }}
QTabBar::tab:selected {{ background: {NAVY3}; color: #ffffff; }}
QHeaderView::section {{ background: {NAVY3}; color: #ffffff; padding: 5px; border: 0px; }}
QTableWidget {{ gridline-color: #254161; }}
QScrollArea {{ border: 0px; }}
"""


def _find_asset(name: str) -> Optional[Path]:
    candidates = [
        Path(__file__).resolve().parents[2] / "assets" / "branding" / name,
        Path.cwd() / "src" / "vslp" / "gui" / "assets" / "branding" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _fit_pixmap(path: Path, width: int, height: int) -> QPixmap:
    pix = QPixmap(str(path))
    if pix.isNull():
        return pix
    return pix.scaled(QSize(width, height), Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _set_table(widget: QTableWidget, df: pd.DataFrame, max_rows: int = 500) -> None:
    widget.clear()
    if df is None or df.empty:
        widget.setRowCount(0)
        widget.setColumnCount(0)
        return
    show = df.head(max_rows).copy()
    widget.setRowCount(show.shape[0])
    widget.setColumnCount(show.shape[1])
    widget.setHorizontalHeaderLabels([str(c) for c in show.columns])
    for r in range(show.shape[0]):
        for c in range(show.shape[1]):
            val = show.iat[r, c]
            widget.setItem(r, c, QTableWidgetItem("" if pd.isna(val) else str(val)))
    widget.resizeColumnsToContents()
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)


class AnalysisWorker(QThread):
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, inputs: AnalysisInputs):
        super().__init__()
        self.inputs = inputs

    def run(self):
        try:
            result = run_feature_analysis(self.inputs)
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class FeatureAnalysisWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VSLP Feature Analysis GUI")
        self.resize(1480, 920)
        self.result = None
        self.worker = None
        self.nav_buttons: list[QPushButton] = []
        self.status_labels: dict[str, QLabel] = {}
        self._build()

    def _build(self):
        root = QWidget()
        outer = QHBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        outer.addWidget(self._sidebar())

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)
        main_layout.addWidget(self._brand_bar())
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, 1)
        outer.addWidget(main, 1)

        self._build_project_tab()
        self._build_mapping_tab()
        self._build_overview_tab()
        self._build_missing_tab()
        self._build_dist_tab()
        self._build_qc_tab()
        self._build_reliability_tab()
        self._build_export_tab()
        self._wire_nav_buttons()
        self.setCentralWidget(root)

    def _sidebar(self):
        side = QFrame()
        side.setObjectName("SideBar")
        side.setFixedWidth(255)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(14, 16, 14, 16)
        lay.setSpacing(9)

        title = QLabel("VSLP")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Feature Analysis GUI v0.40")
        subtitle.setObjectName("AppSubTitle")
        lay.addWidget(title)
        lay.addWidget(subtitle)

        credit = QLabel("© 2026 Nevena Musikic & Yana Yunusova\nSpeech Production Lab, University of Toronto")
        credit.setObjectName("FinePrint")
        credit.setWordWrap(True)
        lay.addWidget(credit)

        lay.addSpacing(10)
        steps = [
            ("Project", "Upload & project setup"),
            ("Column Mapping", "Column roles"),
            ("Overview", "Dataset inventory"),
            ("Missingness", "Coverage audit"),
            ("Distributions", "Outliers & ranges"),
            ("QC Integration", "Feature-QC links"),
            ("Reliability", "Review screen"),
            ("Export", "Report & outputs"),
        ]
        for i, (name, hint) in enumerate(steps):
            btn = QPushButton(f"{i+1}. {name}")
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            btn.setToolTip(hint)
            self.nav_buttons.append(btn)
            lay.addWidget(btn)
            status = QLabel("○ not run")
            status.setObjectName("FinePrint")
            self.status_labels[name] = status
            lay.addWidget(status)

        lay.addStretch(1)
        note = QLabel("Feature audit only. Model training belongs in the separate ML GUI.")
        note.setObjectName("FinePrint")
        note.setWordWrap(True)
        lay.addWidget(note)
        return side

    def _wire_nav_buttons(self):
        for i, btn in enumerate(self.nav_buttons):
            btn.clicked.connect(lambda checked=False, ix=i: self.tabs.setCurrentIndex(ix))
        self.tabs.currentChanged.connect(self._sync_nav)
        self._sync_nav(0)

    def _sync_nav(self, index: int):
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

    def _brand_bar(self):
        bar = QFrame()
        bar.setObjectName("BrandBar")
        bar.setFixedHeight(104)
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 10, 16, 10)
        h.setSpacing(16)

        lab_logo = QLabel()
        lab_logo.setFixedSize(210, 72)
        lab_logo.setAlignment(Qt.AlignCenter)
        uoft_logo = QLabel()
        uoft_logo.setFixedSize(235, 72)
        uoft_logo.setAlignment(Qt.AlignCenter)

        lab_path = _find_asset("speech_production_lab_logo.png") or _find_asset("Lab_logo_final.jpg")
        uoft_path = _find_asset("uoft_logo.png") or _find_asset("University-of-Toronto.png.webp.png")
        if lab_path:
            lab_logo.setPixmap(_fit_pixmap(lab_path, 205, 68))
        else:
            lab_logo.setText("Speech Production Lab")
            lab_logo.setObjectName("DarkOnLight")
        if uoft_path:
            uoft_logo.setPixmap(_fit_pixmap(uoft_path, 230, 68))
        else:
            uoft_logo.setText("University of Toronto")
            uoft_logo.setObjectName("DarkOnLight")

        title_box = QVBoxLayout()
        title = QLabel("VSLP Feature Analysis")
        title.setObjectName("BrandTitle")
        sub = QLabel("Feature audit · QC integration · missingness · distributions · reliability screening")
        sub.setObjectName("BrandSubTitle")
        title_box.addStretch(1)
        title_box.addWidget(title)
        title_box.addWidget(sub)
        title_box.addStretch(1)

        h.addWidget(lab_logo)
        h.addLayout(title_box, 1)
        h.addWidget(uoft_logo)
        return bar

    def _path_row(self, label, line, button_text, callback, tooltip: str | None = None):
        row = QHBoxLayout()
        lab = QLabel(label)
        lab.setMinimumWidth(185)
        row.addWidget(lab)
        row.addWidget(line, 1)
        btn = QPushButton(button_text)
        btn.clicked.connect(callback)
        if tooltip:
            lab.setToolTip(tooltip)
            line.setToolTip(tooltip)
            btn.setToolTip(tooltip)
        row.addWidget(btn)
        return row

    def _card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("Card")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 12, 14, 12)
        lab = QLabel(title)
        lab.setObjectName("SectionTitle")
        lay.addWidget(lab)
        return frame, lay

    def _scrollable(self, widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    def _build_project_tab(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(12)

        card, b = self._card("Project setup")
        self.feature_path = QLineEdit()
        self.qc_path = QLineEdit()
        self.meta_path = QLineEdit()
        self.registry_path = QLineEdit()
        self.output_path = QLineEdit()
        b.addLayout(self._path_row("Primary feature table", self.feature_path, "Browse", lambda: self._browse_file(self.feature_path), "Required. Main features table from acoustic, kinematic, mixed, or generic source."))
        b.addLayout(self._path_row("Optional QC table", self.qc_path, "Browse", lambda: self._browse_file(self.qc_path), "Optional. Quality-control metrics linked to the feature table."))
        b.addLayout(self._path_row("Optional metadata table", self.meta_path, "Browse", lambda: self._browse_file(self.meta_path), "Optional. Subject/session/task/diagnosis/severity covariates."))
        b.addLayout(self._path_row("Optional feature registry / policy", self.registry_path, "Browse", lambda: self._browse_file(self.registry_path), "Optional. Feature definitions, subsystem labels, units, computation policy, expected ranges, and implementation status."))
        b.addLayout(self._path_row("Output folder", self.output_path, "Browse", lambda: self._browse_dir(self.output_path), "Required. Feature-analysis reports, plots, and audit tables are written here."))

        config = QHBoxLayout()
        self.modality = QComboBox()
        self.modality.addItems(["auto", "acoustic", "kinematic", "mixed", "generic"])
        self.modality.setMinimumWidth(200)
        self.modality.setMaxVisibleItems(8)
        self.modality.setToolTip("Use auto for VSLP outputs. Select generic for external feature tables.")
        self.join_key = QLineEdit("auto")
        self.join_key.setMinimumWidth(220)
        self.join_key.setToolTip("Join key for feature/QC/metadata tables. Use auto unless you need to force file_name, record_key, or subject_id.")
        config.addWidget(QLabel("Modality"))
        config.addWidget(self.modality)
        config.addSpacing(14)
        config.addWidget(QLabel("Join key"))
        config.addWidget(self.join_key)
        config.addStretch(1)
        b.addLayout(config)

        info = QLabel("Feature registry/policy is optional. When supplied, it improves labeling, subsystem grouping, expected-range review, and interpretation. For VSLP acoustic outputs, use the feature computation policy or registry table from acoustic/004_features/tables.")
        info.setObjectName("InfoText")
        info.setWordWrap(True)
        b.addWidget(info)

        run = QPushButton("Run Feature Analysis")
        run.setObjectName("Primary")
        run.clicked.connect(self.run_analysis)
        b.addWidget(run)
        lay.addWidget(card)

        log_card, log_lay = self._card("Run log")
        self.project_log = QTextEdit()
        self.project_log.setReadOnly(True)
        self.project_log.setMinimumHeight(160)
        self.project_log.setPlainText("Load a feature table, optionally add QC/metadata/registry tables, choose an output folder, then run analysis.")
        log_lay.addWidget(self.project_log)
        lay.addWidget(log_card, 1)
        self.tabs.addTab(self._scrollable(page), "Project")

    def _build_mapping_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(14, 14, 14, 14)
        top, top_lay = self._card("Column role mapping")
        txt = QLabel("Preview how VSLP classifies columns as identifiers, features, QC variables, targets, covariates, task variables, time variables, or ignored fields.")
        txt.setObjectName("InfoText")
        txt.setWordWrap(True)
        top_lay.addWidget(txt)
        btn = QPushButton("Preview column mapping")
        btn.clicked.connect(self.preview_mapping)
        top_lay.addWidget(btn)
        lay.addWidget(top)
        self.mapping_table = QTableWidget()
        self.mapping_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.mapping_table, 1)
        self.tabs.addTab(tab, "Column Mapping")

    def _simple_table_tab(self, name, description: str = ""):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(14, 14, 14, 14)
        if description:
            card, card_lay = self._card(name)
            info = QLabel(description)
            info.setObjectName("InfoText")
            info.setWordWrap(True)
            card_lay.addWidget(info)
            lay.addWidget(card)
        tbl = QTableWidget()
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        lay.addWidget(tbl, 1)
        self.tabs.addTab(tab, name)
        return tbl

    def _build_overview_tab(self):
        self.overview_table = self._simple_table_tab("Overview", "Dataset inventory: rows, columns, detected feature variables, identifiers, targets, task variables, and QC columns.")

    def _build_missing_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(14, 14, 14, 14)
        card, card_lay = self._card("Missingness and availability")
        info = QLabel("Review feature coverage before modeling. High missingness may indicate task incompatibility, failed signal tracking, severe impairment, or recording quality problems.")
        info.setObjectName("InfoText")
        info.setWordWrap(True)
        card_lay.addWidget(info)
        lay.addWidget(card)
        sp = QSplitter(Qt.Horizontal)
        self.missing_table = QTableWidget()
        self.missing_plot = QLabel("Run analysis to preview missingness plot.")
        self.missing_plot.setAlignment(Qt.AlignCenter)
        self.missing_plot.setMinimumSize(420, 360)
        sp.addWidget(self.missing_table)
        sp.addWidget(self.missing_plot)
        sp.setSizes([620, 620])
        lay.addWidget(sp, 1)
        self.tabs.addTab(tab, "Missingness")

    def _build_dist_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(14, 14, 14, 14)
        card, card_lay = self._card("Distributions and outliers")
        info = QLabel("Inspect robust distribution summaries and outlier flags. This screen is descriptive; it does not automatically exclude features.")
        info.setObjectName("InfoText")
        info.setWordWrap(True)
        card_lay.addWidget(info)
        lay.addWidget(card)
        sp = QSplitter(Qt.Horizontal)
        self.dist_table = QTableWidget()
        self.dist_plot = QLabel("Run analysis to preview distribution plot.")
        self.dist_plot.setAlignment(Qt.AlignCenter)
        self.dist_plot.setMinimumSize(420, 360)
        sp.addWidget(self.dist_table)
        sp.addWidget(self.dist_plot)
        sp.setSizes([620, 620])
        lay.addWidget(sp, 1)
        self.tabs.addTab(tab, "Distributions")

    def _build_qc_tab(self):
        self.qc_table = self._simple_table_tab("QC Integration", "Feature-QC correlations help identify variables that may be strongly influenced by recording quality, segmentation quality, noise, clipping, reverberation, or other artifact domains.")

    def _build_reliability_tab(self):
        self.rel_table = self._simple_table_tab("Reliability", "Initial feature reliability screen based on missingness, zero variance, robust outliers, and QC association. Recommendations are review aids, not automatic scientific decisions.")

    def _build_export_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(14, 14, 14, 14)
        card, card_lay = self._card("Reports and outputs")
        self.report_label = QLabel("Run analysis to create report and export tables.")
        self.report_label.setObjectName("InfoText")
        self.report_label.setWordWrap(True)
        card_lay.addWidget(self.report_label)
        btns = QHBoxLayout()
        open_report = QPushButton("Open HTML Report")
        open_report.clicked.connect(self.open_report)
        open_folder = QPushButton("Open Output Folder")
        open_folder.clicked.connect(self.open_output_folder)
        btns.addWidget(open_report)
        btns.addWidget(open_folder)
        btns.addStretch(1)
        card_lay.addLayout(btns)
        lay.addWidget(card)
        self.export_log = QTextEdit()
        self.export_log.setReadOnly(True)
        lay.addWidget(self.export_log, 1)
        self.tabs.addTab(tab, "Export")

    def _browse_file(self, line):
        p, _ = QFileDialog.getOpenFileName(self, "Select table", str(Path.home()), "Tables (*.csv *.tsv *.txt *.parquet);;All files (*)")
        if p:
            line.setText(p)

    def _browse_dir(self, line):
        p = QFileDialog.getExistingDirectory(self, "Select output folder", str(Path.home()))
        if p:
            line.setText(p)

    def preview_mapping(self):
        if not self.feature_path.text().strip():
            QMessageBox.warning(self, "Missing feature table", "Please select a primary feature table first.")
            return
        try:
            df = read_table(Path(self.feature_path.text().strip()))
            mapping = infer_column_roles(df, "features")
            _set_table(self.mapping_table, mapping)
            self._set_stage_status("Column Mapping", "● previewed")
        except Exception as exc:
            QMessageBox.critical(self, "Mapping failed", str(exc))

    def run_analysis(self):
        if not self.feature_path.text().strip():
            QMessageBox.warning(self, "Missing feature table", "Please select a primary feature table.")
            return
        if not self.output_path.text().strip():
            QMessageBox.warning(self, "Missing output folder", "Please select an output folder.")
            return
        inputs = AnalysisInputs(
            feature_table=Path(self.feature_path.text().strip()),
            qc_table=Path(self.qc_path.text().strip()) if self.qc_path.text().strip() else None,
            metadata_table=Path(self.meta_path.text().strip()) if self.meta_path.text().strip() else None,
            feature_registry=Path(self.registry_path.text().strip()) if self.registry_path.text().strip() else None,
            output_root=Path(self.output_path.text().strip()),
            modality=self.modality.currentText(),
            join_key=self.join_key.text().strip() or "auto",
        )
        self.project_log.append("Running feature analysis...")
        self._set_stage_status("Project", "● running")
        self.worker = AnalysisWorker(inputs)
        self.worker.finished_ok.connect(self._analysis_done)
        self.worker.failed.connect(self._analysis_failed)
        self.worker.start()

    def _analysis_failed(self, msg):
        QMessageBox.critical(self, "Feature analysis failed", msg)
        self.project_log.append("FAILED: " + msg)
        self._set_stage_status("Project", "● failed")

    def _set_stage_status(self, name: str, status: str):
        lab = self.status_labels.get(name)
        if lab:
            lab.setText(status)

    def _analysis_done(self, result):
        self.result = result
        self.project_log.append(f"Done. Report: {result.report_path}")
        self.report_label.setText(str(result.report_path))
        self.export_log.setPlainText("\n".join([f"{k}: {v}" for k, v in {**result.tables, **result.plots}.items()]))
        self._load_outputs()
        for name in ["Project", "Overview", "Missingness", "Distributions", "QC Integration", "Reliability", "Export"]:
            self._set_stage_status(name, "● completed")
        if result.tables.get("feature_column_mapping"):
            self._set_stage_status("Column Mapping", "● completed")
        QMessageBox.information(self, "Analysis complete", "Feature analysis completed successfully.")

    def _read_table(self, key):
        if not self.result or key not in self.result.tables:
            return pd.DataFrame()
        try:
            return pd.read_csv(self.result.tables[key])
        except Exception:
            return pd.DataFrame()

    def _show_plot(self, label, key):
        if not self.result or key not in self.result.plots:
            return
        pix = QPixmap(str(self.result.plots[key]))
        if not pix.isNull():
            label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.result:
            self._show_plot(self.missing_plot, "missingness_top_features")
            self._show_plot(self.dist_plot, "feature_distribution_grid")

    def _load_outputs(self):
        _set_table(self.overview_table, self._read_table("dataset_inventory"))
        _set_table(self.missing_table, self._read_table("feature_distribution_summary"))
        _set_table(self.dist_table, self._read_table("robust_outlier_flags"))
        _set_table(self.qc_table, self._read_table("feature_qc_spearman_correlation"))
        _set_table(self.rel_table, self._read_table("feature_reliability_screen"))
        self._show_plot(self.missing_plot, "missingness_top_features")
        self._show_plot(self.dist_plot, "feature_distribution_grid")

    def open_report(self):
        if self.result:
            import subprocess
            subprocess.run(["open", str(self.result.report_path)], check=False)

    def open_output_folder(self):
        if self.result:
            import subprocess
            subprocess.run(["open", str(self.result.output_root)], check=False)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    win = FeatureAnalysisWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
