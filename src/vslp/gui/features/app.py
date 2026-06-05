"""VSLP Feature Analysis GUI.

Professional, modality-neutral feature audit interface for acoustic, kinematic,
multimodal, and generic feature tables.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from PySide6.QtCore import Qt, QSize
    from PySide6.QtGui import QPixmap, QFont
    from PySide6.QtWidgets import (
        QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QGroupBox,
        QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
        QPushButton, QSizePolicy, QStackedWidget, QTableWidget, QTableWidgetItem,
        QTextEdit, QVBoxLayout, QWidget, QSplitter, QScrollArea
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Feature Analysis GUI requires PySide6. Install with pip install -e '.[gui]'.") from exc

from vslp.analysis.features.column_mapping import (
    ROLE_OPTIONS, classify_columns, infer_table_kind, role_lists, summarize_roles
)
from vslp.analysis.features.audit import (
    read_table, dataset_inventory, feature_distribution_summary,
    feature_qc_correlations, reliability_screen
)

APP_VERSION = "v0.41"

NAVY = "#071A33"
NAVY2 = "#0B2442"
INK = "#0E1726"
PANEL = "#102A49"
CARD = "#FFFFFF"
SOFT = "#F4F7FB"
LINE = "#D9E2EF"
TEAL = "#2DB7B0"
GOLD = "#B68B2D"
RED = "#B42318"
MUTED = "#607089"


def repo_root_guess() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "src" / "vslp").exists() or (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def asset_path(*parts: str) -> Path:
    root = repo_root_guess()
    candidates = [
        root / "src" / "vslp" / "gui" / "assets" / "branding" / Path(*parts),
        root / "src" / "vslp" / "gui" / "features" / "assets" / Path(*parts),
        root / Path(*parts),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def set_app_style(app: QApplication) -> None:
    app.setStyleSheet(f"""
    QWidget {{
        font-family: Arial, Helvetica, sans-serif;
        font-size: 13px;
        color: {INK};
    }}
    QMainWindow {{ background: {SOFT}; }}
    QLineEdit, QTextEdit, QComboBox {{
        background: #FFFFFF;
        border: 1px solid {LINE};
        border-radius: 8px;
        padding: 7px 10px;
        min-height: 28px;
        color: {INK};
        selection-background-color: {TEAL};
    }}
    QComboBox {{ min-width: 220px; }}
    QComboBox::drop-down {{
        border: none;
        width: 28px;
    }}
    QComboBox QAbstractItemView {{
        background: #FFFFFF;
        color: {INK};
        border: 1px solid {LINE};
        selection-background-color: #DDF6F4;
        selection-color: {INK};
        padding: 6px;
        outline: none;
        min-width: 260px;
    }}
    QPushButton {{
        background: {NAVY};
        color: white;
        border: 1px solid #12385E;
        border-radius: 9px;
        padding: 8px 14px;
        font-weight: 600;
    }}
    QPushButton:hover {{ background: #0E3156; }}
    QPushButton:pressed {{ background: #061426; }}
    QPushButton:disabled {{ background: #B8C3D3; color: #EDF1F7; border-color: #B8C3D3; }}
    QPushButton[secondary="true"] {{
        background: #FFFFFF;
        color: {NAVY};
        border: 1px solid {LINE};
    }}
    QPushButton[secondary="true"]:hover {{ background: #F3F8FF; }}
    QTableWidget {{
        background: #FFFFFF;
        border: 1px solid {LINE};
        border-radius: 8px;
        gridline-color: #EDF2F7;
        alternate-background-color: #F8FBFE;
        selection-background-color: #DDF6F4;
        selection-color: {INK};
    }}
    QHeaderView::section {{
        background: #EEF4FA;
        color: {NAVY};
        border: none;
        border-right: 1px solid {LINE};
        border-bottom: 1px solid {LINE};
        padding: 7px;
        font-weight: 700;
    }}
    QGroupBox {{
        border: 1px solid {LINE};
        border-radius: 12px;
        margin-top: 14px;
        padding: 14px;
        background: #FFFFFF;
        font-weight: 700;
        color: {NAVY};
    }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; }}
    QScrollArea {{ border: none; background: transparent; }}
    """)


class LogoBar(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("LogoBar")
        self.setStyleSheet(f"""
        QFrame#LogoBar {{
            background: #FFFFFF;
            border-bottom: 1px solid {LINE};
        }}
        QLabel#Title {{ color: {NAVY}; font-size: 22px; font-weight: 800; letter-spacing: 0.5px; }}
        QLabel#Subtitle {{ color: {MUTED}; font-size: 12px; font-weight: 500; }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 10, 18, 10)
        layout.setSpacing(18)

        self.lab_logo = QLabel()
        self.lab_logo.setFixedSize(170, 58)
        self.lab_logo.setAlignment(Qt.AlignCenter)
        self._set_logo(self.lab_logo, ["Lab_logo_final.jpg", "lab_logo_for_doc.png", "speech_production_lab_logo.png"])

        title_box = QVBoxLayout()
        title = QLabel("VSLP Feature Analysis GUI")
        title.setObjectName("Title")
        sub = QLabel("Feature audit · QC integration · reliability screening · export")
        sub.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(sub)
        title_box.addStretch(1)

        self.uoft_logo = QLabel()
        self.uoft_logo.setFixedSize(230, 58)
        self.uoft_logo.setAlignment(Qt.AlignCenter)
        self._set_logo(self.uoft_logo, ["University-of-Toronto.png.webp.png", "uoft_logo_for_doc.png", "uoft_logo.png"])

        layout.addWidget(self.lab_logo)
        layout.addLayout(title_box, 1)
        layout.addWidget(self.uoft_logo)

    def _set_logo(self, label: QLabel, names: list[str]) -> None:
        for name in names:
            p = asset_path(name)
            if p.exists():
                pix = QPixmap(str(p))
                if not pix.isNull():
                    label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    return
        label.setText("Logo")
        label.setStyleSheet(f"color:{MUTED}; border:1px solid {LINE}; border-radius:8px;")


class Sidebar(QFrame):
    def __init__(self, on_select) -> None:
        super().__init__()
        self.on_select = on_select
        self.buttons: dict[str, QPushButton] = {}
        self.setFixedWidth(250)
        self.setStyleSheet(f"""
        QFrame {{ background: {NAVY}; color: #FFFFFF; }}
        QLabel#Brand {{ color: #FFFFFF; font-size: 26px; font-weight: 900; letter-spacing: 2px; }}
        QLabel#Sub {{ color: #C9D6E6; font-size: 12px; }}
        QLabel#Credit {{ color: #AFC0D5; font-size: 11px; }}
        QPushButton {{
            text-align: left;
            background: transparent;
            color: #D9E6F5;
            border: 1px solid transparent;
            border-radius: 10px;
            padding: 10px 12px;
            font-weight: 600;
        }}
        QPushButton:hover {{ background: #0F2D4F; border-color: #1C4E7E; }}
        QPushButton[active="true"] {{ background: #123A63; border-color: {TEAL}; color: #FFFFFF; }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(8)
        brand = QLabel("VSLP")
        brand.setObjectName("Brand")
        sub = QLabel(f"Feature Analysis GUI {APP_VERSION}")
        sub.setObjectName("Sub")
        credit = QLabel("© 2026 Nevena Musikic & Yana Yunusova\nSpeech Production Lab\nUniversity of Toronto")
        credit.setObjectName("Credit")
        credit.setWordWrap(True)
        layout.addWidget(brand)
        layout.addWidget(sub)
        layout.addSpacing(8)
        layout.addWidget(credit)
        layout.addSpacing(18)
        for key, text in [
            ("project", "○  Project"),
            ("mapping", "○  Column Mapping"),
            ("overview", "○  Overview"),
            ("missing", "○  Missingness"),
            ("dist", "○  Distributions"),
            ("qc", "○  QC Integration"),
            ("reliability", "○  Reliability"),
            ("export", "○  Export / Report"),
        ]:
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, k=key: self.on_select(k))
            layout.addWidget(b)
            self.buttons[key] = b
        layout.addStretch(1)

    def set_active(self, key: str) -> None:
        for k, b in self.buttons.items():
            b.setProperty("active", k == key)
            prefix = "●" if k == key else "○"
            b.setText(prefix + b.text()[1:])
            b.style().unpolish(b); b.style().polish(b)


class Card(QFrame):
    def __init__(self, title: str, subtitle: str | None = None) -> None:
        super().__init__()
        self.setStyleSheet(f"""
        QFrame {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 14px; }}
        QLabel#CardTitle {{ color: {NAVY}; font-size: 17px; font-weight: 800; border: none; }}
        QLabel#CardSubtitle {{ color: {MUTED}; font-size: 12px; border: none; }}
        """)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 16)
        self.layout.setSpacing(10)
        title_label = QLabel(title)
        title_label.setObjectName("CardTitle")
        self.layout.addWidget(title_label)
        if subtitle:
            st = QLabel(subtitle)
            st.setObjectName("CardSubtitle")
            st.setWordWrap(True)
            self.layout.addWidget(st)


class FilePicker(QWidget):
    def __init__(self, label: str, optional: bool = False) -> None:
        super().__init__()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Optional" if optional else "Required")
        self.button = QPushButton("Browse")
        self.button.setProperty("secondary", True)
        self.button.clicked.connect(self.pick_file)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(label)
        lab.setStyleSheet(f"color:{NAVY}; font-weight:700;")
        layout.addWidget(lab, 0, 0)
        layout.addWidget(self.path_edit, 1, 0)
        layout.addWidget(self.button, 1, 1)

    def pick_file(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Select table", "", "Tables (*.csv *.tsv *.txt *.parquet);;All files (*.*)")
        if p:
            self.path_edit.setText(p)

    @property
    def path(self) -> str:
        return self.path_edit.text().strip()


class FeatureAnalysisGUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"VSLP Feature Analysis GUI {APP_VERSION}")
        self.resize(1400, 880)
        self.feature_df: Optional[pd.DataFrame] = None
        self.qc_df: Optional[pd.DataFrame] = None
        self.meta_df: Optional[pd.DataFrame] = None
        self.registry_df: Optional[pd.DataFrame] = None
        self.mapping_df = pd.DataFrame()
        self.outputs: dict[str, pd.DataFrame] = {}
        self.output_dir: Optional[Path] = None
        self.page_keys = ["project", "mapping", "overview", "missing", "dist", "qc", "reliability", "export"]

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(LogoBar())

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)
        outer.addLayout(content, 1)

        self.sidebar = Sidebar(self.show_page)
        content.addWidget(self.sidebar)
        self.stack = QStackedWidget()
        content.addWidget(self.stack, 1)

        self.pages = {
            "project": self._project_page(),
            "mapping": self._mapping_page(),
            "overview": self._table_page("Overview", "Dataset inventory and role counts."),
            "missing": self._table_page("Missingness", "Feature-level and row-level missingness summaries."),
            "dist": self._table_page("Distributions / Outliers", "Distribution summaries and robust outlier screening."),
            "qc": self._table_page("QC Integration", "Feature-QC association screening, when a QC table is supplied."),
            "reliability": self._table_page("Reliability", "Initial reliability screen for downstream analysis."),
            "export": self._export_page(),
        }
        for key in self.page_keys:
            self.stack.addWidget(self.pages[key])
        self.show_page("project")

    def show_page(self, key: str) -> None:
        self.stack.setCurrentIndex(self.page_keys.index(key))
        self.sidebar.set_active(key)

    def _wrap_scroll(self, widget: QWidget) -> QScrollArea:
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(widget)
        return sc

    def _project_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        intro = Card("Project", "Upload a feature table and optional QC, metadata, and registry/policy tables. The analysis is local and does not modify inputs.")
        grid = QGridLayout()
        self.feature_picker = FilePicker("Primary feature table", optional=False)
        self.qc_picker = FilePicker("Optional QC table", optional=True)
        self.meta_picker = FilePicker("Optional metadata table", optional=True)
        self.registry_picker = FilePicker("Optional feature registry / policy", optional=True)
        grid.addWidget(self.feature_picker, 0, 0)
        grid.addWidget(self.qc_picker, 0, 1)
        grid.addWidget(self.meta_picker, 1, 0)
        grid.addWidget(self.registry_picker, 1, 1)
        intro.layout.addLayout(grid)

        policy_note = QLabel("Registry / policy files help the GUI understand feature names, subsystems, units, expected ranges, computation policies, and implementation status. If omitted, VSLP uses conservative automatic column-role detection.")
        policy_note.setWordWrap(True)
        policy_note.setStyleSheet(f"color:{MUTED}; background:#F7FAFD; border:1px solid {LINE}; border-radius:8px; padding:10px;")
        intro.layout.addWidget(policy_note)

        row = QHBoxLayout()
        row.addWidget(QLabel("Modality:"))
        self.modality_combo = QComboBox()
        self.modality_combo.addItems(["Auto-detect", "Acoustic", "Kinematic", "Mixed acoustic + kinematic", "Generic"])
        self.modality_combo.setMinimumWidth(290)
        row.addWidget(self.modality_combo)
        row.addSpacing(16)
        row.addWidget(QLabel("Output folder:"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Required output folder for analysis results")
        row.addWidget(self.output_edit, 1)
        out_button = QPushButton("Browse")
        out_button.setProperty("secondary", True)
        out_button.clicked.connect(self.pick_output_folder)
        row.addWidget(out_button)
        intro.layout.addLayout(row)

        actions = QHBoxLayout()
        load_btn = QPushButton("Load and Map Tables")
        load_btn.clicked.connect(self.load_and_map)
        run_btn = QPushButton("Run Feature Analysis")
        run_btn.clicked.connect(self.run_analysis)
        actions.addWidget(load_btn)
        actions.addWidget(run_btn)
        actions.addStretch(1)
        intro.layout.addLayout(actions)

        self.project_status = QTextEdit()
        self.project_status.setReadOnly(True)
        self.project_status.setMinimumHeight(170)
        self.project_status.setPlaceholderText("Status messages will appear here.")
        intro.layout.addWidget(self.project_status)
        layout.addWidget(intro)
        layout.addStretch(1)
        return self._wrap_scroll(body)

    def _mapping_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        card = Card("Column Mapping", "Review detected roles. Numeric primary-table columns are treated as features unless a stronger rule identifies them as identifiers, QC variables, audit/status fields, or exact clinical labels.")
        self.mapping_table = QTableWidget(0, 7)
        self.mapping_table.setHorizontalHeaderLabels(["Column", "Role", "Confidence", "Reason", "dtype", "Missing", "Unique"])
        self.mapping_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.mapping_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.mapping_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.mapping_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.mapping_table.setAlternatingRowColors(True)
        card.layout.addWidget(self.mapping_table)
        btns = QHBoxLayout()
        refresh = QPushButton("Refresh Mapping")
        refresh.clicked.connect(self.refresh_mapping_table)
        btns.addWidget(refresh)
        btns.addStretch(1)
        card.layout.addLayout(btns)
        layout.addWidget(card)
        return self._wrap_scroll(body)

    def _table_page(self, title: str, subtitle: str) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        card = Card(title, subtitle)
        table = QTableWidget(0, 0)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setObjectName(title.replace(" ", "_").lower())
        card.layout.addWidget(table)
        setattr(self, f"table_{table.objectName()}", table)
        layout.addWidget(card)
        return self._wrap_scroll(body)

    def _export_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        card = Card("Export / Report", "Write analysis tables and a compact HTML report to disk.")
        self.export_status = QTextEdit()
        self.export_status.setReadOnly(True)
        self.export_status.setMinimumHeight(360)
        card.layout.addWidget(self.export_status)
        btn = QPushButton("Open Output Folder")
        btn.clicked.connect(self.open_output_folder)
        btn.setProperty("secondary", True)
        card.layout.addWidget(btn)
        layout.addWidget(card)
        return self._wrap_scroll(body)

    def pick_output_folder(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Select output folder")
        if p:
            self.output_edit.setText(p)

    def log(self, text: str) -> None:
        self.project_status.append(text)
        self.export_status.append(text)

    def load_and_map(self) -> None:
        try:
            if not self.feature_picker.path:
                QMessageBox.warning(self, "Missing feature table", "Please select a primary feature table.")
                return
            self.feature_df = read_table(self.feature_picker.path)
            self.qc_df = read_table(self.qc_picker.path) if self.qc_picker.path else None
            self.meta_df = read_table(self.meta_picker.path) if self.meta_picker.path else None
            self.registry_df = read_table(self.registry_picker.path) if self.registry_picker.path else None
            kind = infer_table_kind(self.feature_picker.path, explicit="feature")
            self.mapping_df = classify_columns(self.feature_df, table_kind=kind, registry=self.registry_df)
            self.refresh_mapping_table()
            roles = summarize_roles(self.mapping_df)
            self.log(f"Loaded feature table: {self.feature_df.shape[0]} rows × {self.feature_df.shape[1]} columns")
            if self.qc_df is not None:
                self.log(f"Loaded QC table: {self.qc_df.shape[0]} rows × {self.qc_df.shape[1]} columns")
            if self.meta_df is not None:
                self.log(f"Loaded metadata table: {self.meta_df.shape[0]} rows × {self.meta_df.shape[1]} columns")
            if self.registry_df is not None:
                self.log(f"Loaded registry/policy table: {self.registry_df.shape[0]} rows × {self.registry_df.shape[1]} columns")
            self.log("Role counts:\n" + roles.to_string(index=False))
            self.show_page("mapping")
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    def refresh_mapping_table(self) -> None:
        if self.mapping_df.empty:
            return
        df = self.mapping_df.copy()
        self.mapping_table.setRowCount(len(df))
        self.mapping_table.setColumnCount(7)
        for i, r in df.iterrows():
            self.mapping_table.setItem(i, 0, QTableWidgetItem(str(r["column"])))
            combo = QComboBox()
            combo.addItems(ROLE_OPTIONS)
            combo.setCurrentText(str(r["role"]))
            self.mapping_table.setCellWidget(i, 1, combo)
            self.mapping_table.setItem(i, 2, QTableWidgetItem(f"{float(r['confidence']):.2f}"))
            self.mapping_table.setItem(i, 3, QTableWidgetItem(str(r["reason"])))
            self.mapping_table.setItem(i, 4, QTableWidgetItem(str(r["dtype"])))
            self.mapping_table.setItem(i, 5, QTableWidgetItem(f"{float(r['missing_fraction']):.3f}"))
            self.mapping_table.setItem(i, 6, QTableWidgetItem(str(r["unique_values"])))
        self.mapping_table.resizeRowsToContents()

    def collect_mapping_from_table(self) -> pd.DataFrame:
        if self.mapping_df.empty:
            return self.mapping_df
        df = self.mapping_df.copy()
        roles = []
        for i in range(self.mapping_table.rowCount()):
            widget = self.mapping_table.cellWidget(i, 1)
            roles.append(widget.currentText() if isinstance(widget, QComboBox) else df.iloc[i]["role"])
        df["role"] = roles
        self.mapping_df = df
        return df

    def run_analysis(self) -> None:
        try:
            if self.feature_df is None:
                self.load_and_map()
                if self.feature_df is None:
                    return
            if not self.output_edit.text().strip():
                QMessageBox.warning(self, "Missing output folder", "Please select an output folder for the Feature Analysis results.")
                return
            self.output_dir = Path(self.output_edit.text().strip()) / "feature_analysis"
            tables_dir = self.output_dir / "tables"
            reports_dir = self.output_dir / "reports"
            tables_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)

            mapping = self.collect_mapping_from_table()
            roles = role_lists(mapping)
            feature_cols = roles.get("Feature", [])
            inventory = dataset_inventory(self.feature_df, self.qc_df, self.meta_df, mapping)
            dist = feature_distribution_summary(self.feature_df, feature_cols)
            row_missing = pd.DataFrame({
                "row_index": range(len(self.feature_df)),
                "missing_fraction_all_columns": self.feature_df.isna().mean(axis=1).values,
                "missing_fraction_feature_columns": self.feature_df[feature_cols].isna().mean(axis=1).values if feature_cols else [],
            }) if feature_cols else pd.DataFrame({"row_index": range(len(self.feature_df)), "missing_fraction_all_columns": self.feature_df.isna().mean(axis=1).values})
            qc_corr = feature_qc_correlations(self.feature_df, self.qc_df, feature_cols)
            reliability = reliability_screen(dist, qc_corr)

            outputs = {
                "dataset_inventory": inventory,
                "feature_column_mapping": mapping,
                "feature_distribution_summary": dist,
                "missingness_by_row": row_missing,
                "feature_qc_spearman_correlation": qc_corr,
                "feature_reliability_screen": reliability,
            }
            self.outputs = outputs
            for name, df in outputs.items():
                df.to_csv(tables_dir / f"{name}.csv", index=False)
            self.write_report(reports_dir / "vslp_feature_analysis_report.html", outputs)
            self.populate_output_tables(outputs)
            self.log(f"Analysis complete. Outputs written to: {self.output_dir}")
            self.show_page("overview")
        except Exception as exc:
            QMessageBox.critical(self, "Analysis failed", str(exc))

    def populate_output_tables(self, outputs: dict[str, pd.DataFrame]) -> None:
        targets = {
            "overview": "dataset_inventory",
            "missing": "missingness_by_row",
            "distributions___outliers": "feature_distribution_summary",
            "qc_integration": "feature_qc_spearman_correlation",
            "reliability": "feature_reliability_screen",
        }
        for obj_suffix, key in targets.items():
            table = getattr(self, f"table_{obj_suffix}", None)
            if table is not None and key in outputs:
                self._fill_table(table, outputs[key])

    def _fill_table(self, table: QTableWidget, df: pd.DataFrame, max_rows: int = 500) -> None:
        show = df.head(max_rows).copy()
        table.setRowCount(len(show))
        table.setColumnCount(len(show.columns))
        table.setHorizontalHeaderLabels([str(c) for c in show.columns])
        for i in range(len(show)):
            for j, c in enumerate(show.columns):
                table.setItem(i, j, QTableWidgetItem(str(show.iloc[i, j])))
        table.resizeRowsToContents()

    def write_report(self, path: Path, outputs: dict[str, pd.DataFrame]) -> None:
        inv = outputs.get("dataset_inventory", pd.DataFrame()).to_html(index=False, escape=False)
        roles = summarize_roles(outputs.get("feature_column_mapping", pd.DataFrame())).to_html(index=False, escape=False)
        rel = outputs.get("feature_reliability_screen", pd.DataFrame()).head(80).to_html(index=False, escape=False)
        html = f"""<!doctype html><html><head><meta charset='utf-8'><title>VSLP Feature Analysis Report</title>
        <style>body{{font-family:Arial,sans-serif;margin:32px;color:#0E1726}} h1,h2{{color:#071A33}} table{{border-collapse:collapse;width:100%;font-size:12px;margin-bottom:24px}} td,th{{border:1px solid #D9E2EF;padding:6px}} th{{background:#EEF4FA}}</style></head><body>
        <h1>VSLP Feature Analysis Report</h1><p>Version {APP_VERSION}. Descriptive feature audit only; not a diagnostic or ML-training report.</p>
        <h2>Dataset inventory</h2>{inv}<h2>Column-role summary</h2>{roles}<h2>Initial feature reliability screen</h2>{rel}</body></html>"""
        path.write_text(html, encoding="utf-8")

    def open_output_folder(self) -> None:
        if not self.output_dir or not self.output_dir.exists():
            QMessageBox.information(self, "No output folder", "Run Feature Analysis first.")
            return
        import subprocess, platform
        if platform.system() == "Darwin":
            subprocess.run(["open", str(self.output_dir)], check=False)
        elif platform.system() == "Windows":
            subprocess.run(["explorer", str(self.output_dir)], check=False)
        else:
            subprocess.run(["xdg-open", str(self.output_dir)], check=False)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    set_app_style(app)
    win = FeatureAnalysisGUI()
    win.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
