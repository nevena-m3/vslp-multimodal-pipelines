from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
        QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QSplitter, QTabWidget,
        QTableWidget, QTableWidgetItem, QTextEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
        QWidget, QMainWindow, QHeaderView, QAbstractItemView
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PySide6 is required for the Feature Analysis GUI. Install the gui extras first.") from exc

from vslp.analysis.features import AnalysisInputs, run_feature_analysis
from vslp.analysis.features.loaders import read_table
from vslp.analysis.features.column_mapping import infer_column_roles


APP_STYLE = """
QMainWindow, QWidget { background: #07172b; color: #e8edf7; font-family: Arial; font-size: 13px; }
QFrame#BrandBar { background: #ffffff; border: 0px; }
QLabel#Title { color: #0b1f3a; font-size: 20px; font-weight: 700; }
QLabel#SubTitle { color: #344054; font-size: 12px; }
QGroupBox { border: 1px solid #254161; border-radius: 10px; margin-top: 10px; padding: 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #b7cdf0; }
QPushButton { background: #123b63; color: #ffffff; border: 1px solid #2d5f91; border-radius: 8px; padding: 7px 12px; }
QPushButton:hover { background: #174c7e; }
QPushButton#Primary { background: #0f766e; border-color: #14b8a6; font-weight: 700; }
QLineEdit, QComboBox, QTextEdit, QTableWidget, QTreeWidget, QListWidget { background: #0b223d; color: #e8edf7; border: 1px solid #254161; border-radius: 6px; padding: 4px; }
QTabWidget::pane { border: 1px solid #254161; border-radius: 8px; }
QTabBar::tab { background: #0b223d; color: #c6d4e8; padding: 8px 12px; border-top-left-radius: 7px; border-top-right-radius: 7px; }
QTabBar::tab:selected { background: #123b63; color: #ffffff; }
QHeaderView::section { background: #123b63; color: #ffffff; padding: 5px; border: 0px; }
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


def _set_table(widget: QTableWidget, df: pd.DataFrame, max_rows: int = 500) -> None:
    widget.clear()
    if df is None or df.empty:
        widget.setRowCount(0); widget.setColumnCount(0); return
    show = df.head(max_rows).copy()
    widget.setRowCount(show.shape[0]); widget.setColumnCount(show.shape[1])
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
        self.resize(1450, 900)
        self.result = None
        self.worker = None
        self._build()

    def _build(self):
        root = QWidget(); layout = QVBoxLayout(root); layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self._brand_bar())
        self.tabs = QTabWidget(); layout.addWidget(self.tabs, 1)
        self._build_project_tab(); self._build_mapping_tab(); self._build_overview_tab(); self._build_missing_tab(); self._build_dist_tab(); self._build_qc_tab(); self._build_reliability_tab(); self._build_export_tab()
        self.setCentralWidget(root)

    def _brand_bar(self):
        bar = QFrame(); bar.setObjectName("BrandBar"); bar.setFixedHeight(96)
        h = QHBoxLayout(bar); h.setContentsMargins(16, 8, 16, 8)
        lab_logo = QLabel(); uoft_logo = QLabel()
        for label, names, maxw in [(lab_logo,["speech_production_lab_logo.png","Lab_logo_final.jpg"],220),(uoft_logo,["uoft_logo.png","University-of-Toronto.png.webp.png"],310)]:
            for n in names:
                p = _find_asset(n)
                if p:
                    pix = QPixmap(str(p)); label.setPixmap(pix.scaledToWidth(maxw, Qt.SmoothTransformation)); break
            label.setStyleSheet("background:white;")
        title_box = QVBoxLayout()
        title = QLabel("VSLP Feature Analysis GUI"); title.setObjectName("Title")
        sub = QLabel("Modality-neutral feature audit, QC integration, distributions, missingness, outliers, and export"); sub.setObjectName("SubTitle")
        title_box.addWidget(title); title_box.addWidget(sub)
        h.addWidget(lab_logo); h.addSpacing(16); h.addLayout(title_box, 1); h.addWidget(uoft_logo)
        return bar

    def _path_row(self, label, line, button_text, callback):
        row = QHBoxLayout(); row.addWidget(QLabel(label)); row.addWidget(line, 1); btn = QPushButton(button_text); btn.clicked.connect(callback); row.addWidget(btn); return row

    def _build_project_tab(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        box = QGroupBox("Upload tables"); b = QVBoxLayout(box)
        self.feature_path = QLineEdit(); self.qc_path = QLineEdit(); self.meta_path = QLineEdit(); self.registry_path = QLineEdit(); self.output_path = QLineEdit()
        b.addLayout(self._path_row("Primary feature table", self.feature_path, "Browse", lambda: self._browse_file(self.feature_path)))
        b.addLayout(self._path_row("Optional QC table", self.qc_path, "Browse", lambda: self._browse_file(self.qc_path)))
        b.addLayout(self._path_row("Optional metadata table", self.meta_path, "Browse", lambda: self._browse_file(self.meta_path)))
        b.addLayout(self._path_row("Optional feature registry/policy", self.registry_path, "Browse", lambda: self._browse_file(self.registry_path)))
        b.addLayout(self._path_row("Output folder", self.output_path, "Browse", lambda: self._browse_dir(self.output_path)))
        config = QHBoxLayout(); self.modality = QComboBox(); self.modality.addItems(["auto", "acoustic", "kinematic", "mixed", "generic"]); self.join_key = QLineEdit("auto")
        config.addWidget(QLabel("Modality")); config.addWidget(self.modality); config.addWidget(QLabel("Join key")); config.addWidget(self.join_key); b.addLayout(config)
        run = QPushButton("Run Feature Analysis"); run.setObjectName("Primary"); run.clicked.connect(self.run_analysis); b.addWidget(run)
        lay.addWidget(box)
        self.project_log = QTextEdit(); self.project_log.setReadOnly(True); self.project_log.setPlainText("Load a feature table, optionally add QC/metadata, choose an output folder, then run analysis.")
        lay.addWidget(self.project_log, 1)
        self.tabs.addTab(tab, "Project")

    def _build_mapping_tab(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        btn = QPushButton("Preview column mapping"); btn.clicked.connect(self.preview_mapping); lay.addWidget(btn)
        self.mapping_table = QTableWidget(); self.mapping_table.setEditTriggers(QAbstractItemView.NoEditTriggers); lay.addWidget(self.mapping_table, 1)
        self.tabs.addTab(tab, "Column Mapping")

    def _simple_table_tab(self, name):
        tab = QWidget(); lay = QVBoxLayout(tab); tbl = QTableWidget(); tbl.setEditTriggers(QAbstractItemView.NoEditTriggers); lay.addWidget(tbl, 1); self.tabs.addTab(tab, name); return tbl

    def _build_overview_tab(self): self.overview_table = self._simple_table_tab("Overview")
    def _build_missing_tab(self):
        tab = QWidget(); sp = QSplitter(Qt.Horizontal); lay = QVBoxLayout(tab); lay.addWidget(sp)
        self.missing_table = QTableWidget(); self.missing_plot = QLabel("Run analysis to preview missingness plot."); self.missing_plot.setAlignment(Qt.AlignCenter); self.missing_plot.setScaledContents(False)
        sp.addWidget(self.missing_table); sp.addWidget(self.missing_plot); self.tabs.addTab(tab, "Missingness")
    def _build_dist_tab(self):
        tab = QWidget(); sp = QSplitter(Qt.Horizontal); lay = QVBoxLayout(tab); lay.addWidget(sp)
        self.dist_table = QTableWidget(); self.dist_plot = QLabel("Run analysis to preview distribution plot."); self.dist_plot.setAlignment(Qt.AlignCenter)
        sp.addWidget(self.dist_table); sp.addWidget(self.dist_plot); self.tabs.addTab(tab, "Distributions / Outliers")
    def _build_qc_tab(self): self.qc_table = self._simple_table_tab("QC Integration")
    def _build_reliability_tab(self): self.rel_table = self._simple_table_tab("Reliability")
    def _build_export_tab(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        self.report_label = QLabel("Run analysis to create report and export tables."); lay.addWidget(self.report_label)
        btns = QHBoxLayout(); open_report = QPushButton("Open HTML Report"); open_report.clicked.connect(self.open_report); open_folder = QPushButton("Open Output Folder"); open_folder.clicked.connect(self.open_output_folder); btns.addWidget(open_report); btns.addWidget(open_folder); lay.addLayout(btns)
        self.export_log = QTextEdit(); self.export_log.setReadOnly(True); lay.addWidget(self.export_log, 1)
        self.tabs.addTab(tab, "Export / Report")

    def _browse_file(self, line):
        p, _ = QFileDialog.getOpenFileName(self, "Select table", str(Path.home()), "Tables (*.csv *.tsv *.txt *.parquet);;All files (*)")
        if p: line.setText(p)
    def _browse_dir(self, line):
        p = QFileDialog.getExistingDirectory(self, "Select output folder", str(Path.home()))
        if p: line.setText(p)

    def preview_mapping(self):
        if not self.feature_path.text().strip():
            QMessageBox.warning(self, "Missing feature table", "Please select a primary feature table first."); return
        try:
            df = read_table(Path(self.feature_path.text().strip()))
            mapping = infer_column_roles(df, "features")
            _set_table(self.mapping_table, mapping)
        except Exception as exc:
            QMessageBox.critical(self, "Mapping failed", str(exc))

    def run_analysis(self):
        if not self.feature_path.text().strip():
            QMessageBox.warning(self, "Missing feature table", "Please select a primary feature table."); return
        if not self.output_path.text().strip():
            QMessageBox.warning(self, "Missing output folder", "Please select an output folder."); return
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
        self.worker = AnalysisWorker(inputs); self.worker.finished_ok.connect(self._analysis_done); self.worker.failed.connect(self._analysis_failed); self.worker.start()

    def _analysis_failed(self, msg):
        QMessageBox.critical(self, "Feature analysis failed", msg); self.project_log.append("FAILED: " + msg)

    def _analysis_done(self, result):
        self.result = result; self.project_log.append(f"Done. Report: {result.report_path}")
        self.report_label.setText(str(result.report_path))
        self.export_log.setPlainText("\n".join([f"{k}: {v}" for k, v in {**result.tables, **result.plots}.items()]))
        self._load_outputs()
        QMessageBox.information(self, "Analysis complete", "Feature analysis completed successfully.")

    def _read_table(self, key):
        if not self.result or key not in self.result.tables: return pd.DataFrame()
        try: return pd.read_csv(self.result.tables[key])
        except Exception: return pd.DataFrame()

    def _show_plot(self, label, key):
        if not self.result or key not in self.result.plots: return
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
            import subprocess; subprocess.run(["open", str(self.result.report_path)], check=False)
    def open_output_folder(self):
        if self.result:
            import subprocess; subprocess.run(["open", str(self.result.output_root)], check=False)


def main():
    app = QApplication(sys.argv); app.setStyleSheet(APP_STYLE); win = FeatureAnalysisWindow(); win.show(); sys.exit(app.exec())


if __name__ == "__main__":
    main()
