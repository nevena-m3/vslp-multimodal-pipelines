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
    from PySide6.QtCore import Qt, QSize, QUrl
    from PySide6.QtGui import QPixmap, QFont, QDesktopServices
    from PySide6.QtWidgets import (
        QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QGroupBox,
        QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
        QPushButton, QSizePolicy, QStackedWidget, QTableWidget, QTableWidgetItem,
        QTextEdit, QVBoxLayout, QWidget, QSplitter, QScrollArea, QAbstractItemView,
        QTabWidget
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Feature Analysis GUI requires PySide6. Install with pip install -e '.[gui]'.") from exc

from vslp.analysis.features.column_mapping import (
    ROLE_OPTIONS, ROLE_FEATURE, ROLE_TARGET, ROLE_IDENTIFIER, ROLE_QC, ROLE_COVARIATE, ROLE_IGNORE, classify_columns, infer_table_kind, role_lists, summarize_roles
)
from vslp.analysis.features.audit import (
    read_table, dataset_inventory, role_summary, design_overview,
    feature_family_overview, group_counts, feature_distribution_summary,
    feature_qc_correlations, reliability_screen,
    missingness_feature_summary, missingness_row_summary, missingness_group_summary,
    missingness_family_summary, missingness_comissing_pairs,
    robust_outlier_flags, expected_range_flags, distribution_review_summary,
    distribution_shape_audit, row_outlier_burden_summary,
    overview_readiness_summary, overview_feature_quality_landscape,
    qc_metric_catalog, qc_row_burden_summary, qc_family_burden_summary,
    feature_qc_family_association, qc_missingness_associations,
    qc_outlier_associations, qc_integration_summary
)
from vslp.analysis.features.plots import (
    plot_role_counts, plot_group_counts, plot_feature_family_counts,
    plot_missingness, plot_feature_availability_heatmap,
    plot_row_missingness_distribution, plot_missingness_by_group,
    plot_missingness_family_summary, plot_comissing_heatmap,
    plot_distribution_review_summary, plot_expected_range_flags,
    plot_selected_feature_distribution, plot_selected_feature_diagnostic, plot_group_feature_boxplot, plot_distribution_grid, plot_outlier_counts,
    plot_distribution_shape_summary, plot_distribution_shape_landscape, plot_row_outlier_burden, plot_variance_screen,
    plot_overview_readiness_scorecard, plot_dataset_design_tiles,
    plot_feature_quality_landscape, plot_feature_family_quality, plot_subject_task_matrix,
    plot_qc_family_burden, plot_qc_metric_distributions, plot_qc_feature_association_heatmap,
    plot_qc_top_feature_associations, plot_qc_missingness_associations,
    plot_qc_row_burden, plot_selected_feature_qc_scatter, plot_qc_artifact_model
)

APP_VERSION = "v0.49"

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
        selection-background-color: #DDF6F4;
        selection-color: {INK};
    }}
    QComboBox {{
        min-width: 220px;
        background: #FFFFFF;
        color: {INK};
    }}
    QComboBox:hover {{ border: 1px solid #B7C7DA; background: #FFFFFF; color: {INK}; }}
    QComboBox:focus {{ border: 1px solid {TEAL}; background: #FFFFFF; color: {INK}; }}
    QComboBox:on {{ background: #FFFFFF; color: {INK}; }}
    QComboBox::drop-down {{
        border: none;
        width: 30px;
        background: #FFFFFF;
        border-top-right-radius: 8px;
        border-bottom-right-radius: 8px;
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
    QComboBox QAbstractItemView::item {{
        background: #FFFFFF;
        color: {INK};
        min-height: 26px;
        padding: 6px 10px;
    }}
    QComboBox QAbstractItemView::item:hover {{ background: #F3FAF9; color: {INK}; }}
    QComboBox QAbstractItemView::item:selected {{ background: #DDF6F4; color: {INK}; }}
    QPushButton {{
        background: #FFFFFF;
        color: {NAVY};
        border: 1px solid {LINE};
        border-radius: 9px;
        padding: 8px 14px;
        font-weight: 700;
    }}
    QPushButton:hover {{ background: #F3FAF9; border-color: {TEAL}; color: {NAVY}; }}
    QPushButton:pressed {{ background: #DDF6F4; border-color: {TEAL}; color: {NAVY}; }}
    QPushButton:focus {{ border: 1px solid {TEAL}; background: #FFFFFF; color: {NAVY}; }}
    QPushButton:checked {{ background: #EAF8F7; border-color: {TEAL}; color: {NAVY}; }}
    QPushButton:disabled {{ background: #F4F6F9; color: #8A99AA; border-color: #D9E2EF; }}
    QPushButton[secondary="true"] {{
        background: #FFFFFF;
        color: {NAVY};
        border: 1px solid {LINE};
    }}
    QPushButton[secondary="true"]:hover {{ background: #F3FAF9; border-color: {TEAL}; color: {NAVY}; }}
    QPushButton[primary="true"] {{
        background: #EAF8F7;
        color: {NAVY};
        border: 1px solid {TEAL};
    }}
    QPushButton[primary="true"]:hover {{ background: #DDF6F4; color: {NAVY}; }}
    QPushButton[primary="true"]:pressed {{ background: #C8F0ED; color: {NAVY}; }}
    QAbstractButton {{
        color: {NAVY};
        background: #FFFFFF;
        selection-background-color: #DDF6F4;
        selection-color: {INK};
    }}
    QCheckBox, QRadioButton {{
        color: {INK};
        background: transparent;
        spacing: 8px;
        min-height: 24px;
    }}
    QCheckBox:hover, QRadioButton:hover {{ color: {NAVY}; background: #F7FAFD; }}
    QListView, QTreeView, QTableView {{
        background: #FFFFFF;
        color: {INK};
        alternate-background-color: #F8FBFE;
        selection-background-color: #DDF6F4;
        selection-color: {INK};
    }}
    QTabWidget::pane {{
        background: #FFFFFF;
        border: 1px solid {LINE};
        border-radius: 8px;
    }}
    QTabBar::tab {{
        background: #F7FAFD;
        color: {NAVY};
        border: 1px solid {LINE};
        border-bottom: none;
        padding: 8px 14px;
        min-height: 22px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
    }}
    QTabBar::tab:selected {{
        background: #FFFFFF;
        color: {NAVY};
        border-top: 2px solid {TEAL};
        font-weight: 700;
    }}
    QTabBar::tab:hover {{ background: #F3FAF9; color: {NAVY}; }}
    QMenu {{
        background: #FFFFFF;
        color: {INK};
        border: 1px solid {LINE};
    }}
    QMenu::item {{ background: #FFFFFF; color: {INK}; padding: 7px 14px; }}
    QMenu::item:selected {{ background: #DDF6F4; color: {INK}; }}
    QToolTip {{
        background: #FFFFFF;
        color: {INK};
        border: 1px solid {LINE};
        padding: 6px;
    }}
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
        QLabel#Title {{ color: {NAVY}; background: transparent; border: none; font-size: 22px; font-weight: 800; letter-spacing: 0.5px; }}
        QLabel#Subtitle {{ color: {MUTED}; background: transparent; border: none; font-size: 12px; font-weight: 500; }}
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
        QPushButton:hover {{ background: #0F2D4F; border-color: #1C4E7E; color: #FFFFFF; }}
        QPushButton:focus {{ background: #0B2442; border-color: {TEAL}; color: #FFFFFF; }}
        QPushButton:pressed {{ background: #123A63; border-color: {TEAL}; color: #FFFFFF; }}
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
            "overview": self._overview_page(),
            "missing": self._missingness_page(),
            "dist": self._distributions_page(),
            "qc": self._qc_page(),
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

        intro = Card("Project", "Upload a primary feature table and optional QC, metadata, and feature-definition tables. The analysis is local and non-destructive.")
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
        self.mapping_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.mapping_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.mapping_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.mapping_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.mapping_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.mapping_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.mapping_table.setAlternatingRowColors(True)
        card.layout.addWidget(self.mapping_table)
        self.mapping_summary_label = QLabel("Load a table to inspect proposed roles. You can accept the proposed mapping or manually change any row.")
        self.mapping_summary_label.setWordWrap(True)
        self.mapping_summary_label.setStyleSheet(f"color:{MUTED}; background:#F7FAFD; border:1px solid {LINE}; border-radius:8px; padding:9px;")
        card.layout.addWidget(self.mapping_summary_label)

        quick = QHBoxLayout()
        for label, role in [
            ("Set selected: Feature", ROLE_FEATURE),
            ("Target", ROLE_TARGET),
            ("Identifier", ROLE_IDENTIFIER),
            ("QC", ROLE_QC),
            ("Covariate", ROLE_COVARIATE),
            ("Ignore / Exclude", ROLE_IGNORE),
        ]:
            b = QPushButton(label)
            b.setProperty("secondary", True)
            b.clicked.connect(lambda _=False, r=role: self.set_selected_role(r))
            quick.addWidget(b)
        quick.addStretch(1)
        card.layout.addLayout(quick)

        btns = QHBoxLayout()
        refresh = QPushButton("Refresh Proposed Mapping")
        refresh.setProperty("secondary", True)
        refresh.clicked.connect(self.reload_proposed_mapping)
        accept = QPushButton("Accept Mapping and Continue")
        accept.clicked.connect(self.accept_mapping_and_continue)
        btns.addWidget(refresh)
        btns.addWidget(accept)
        btns.addStretch(1)
        card.layout.addLayout(btns)
        layout.addWidget(card)
        return self._wrap_scroll(body)


    def _overview_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        card = Card(
            "Overview",
            "Dataset orientation, design structure, feature coverage, and readiness for downstream feature review. This screen is descriptive; it does not perform ML."
        )

        self.overview_metric_grid = QGridLayout()
        self.overview_metric_grid.setHorizontalSpacing(12)
        self.overview_metric_grid.setVerticalSpacing(12)
        card.layout.addLayout(self.overview_metric_grid)

        self.overview_note = QLabel("Load tables, review/accept column mapping, then run Feature Analysis to populate this dashboard.")
        self.overview_note.setWordWrap(True)
        self.overview_note.setStyleSheet(f"color:{MUTED}; background:#F7FAFD; border:1px solid {LINE}; border-radius:8px; padding:10px;")
        card.layout.addWidget(self.overview_note)

        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {LINE}; border-radius: 8px; background: #FFFFFF; }}
            QTabBar::tab {{ background:#F7FAFD; color:{NAVY}; border:1px solid {LINE}; border-bottom:none; padding:8px 14px; min-height:22px; }}
            QTabBar::tab:selected {{ background:#FFFFFF; color:{NAVY}; border-top:2px solid {TEAL}; font-weight:700; }}
            QTabBar::tab:hover {{ background:#F3FAF9; color:{NAVY}; }}
        """)

        self.overview_readiness_table = QTableWidget(0, 0)
        self.overview_inventory_table = QTableWidget(0, 0)
        self.overview_design_table = QTableWidget(0, 0)
        self.overview_roles_table = QTableWidget(0, 0)
        self.overview_family_table = QTableWidget(0, 0)
        self.overview_quality_table = QTableWidget(0, 0)
        for t in [
            self.overview_readiness_table, self.overview_inventory_table,
            self.overview_design_table, self.overview_roles_table,
            self.overview_family_table, self.overview_quality_table,
        ]:
            t.setAlternatingRowColors(True)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        tabs.addTab(self.overview_readiness_table, "Readiness")
        tabs.addTab(self.overview_inventory_table, "Inventory")
        tabs.addTab(self.overview_roles_table, "Roles")
        tabs.addTab(self.overview_design_table, "Design variables")
        tabs.addTab(self.overview_family_table, "Feature families")
        tabs.addTab(self.overview_quality_table, "Feature quality")
        card.layout.addWidget(tabs)

        plot_panel = QFrame()
        plot_panel.setStyleSheet(f"QFrame {{ background:#F8FBFE; border:1px solid {LINE}; border-radius:12px; }}")
        plot_panel_layout = QHBoxLayout(plot_panel)
        plot_panel_layout.setContentsMargins(14, 14, 14, 14)
        plot_panel_layout.setSpacing(14)

        plot_controls = QFrame()
        plot_controls.setStyleSheet("QFrame { border:none; background:transparent; }")
        plot_controls_layout = QVBoxLayout(plot_controls)
        plot_controls_layout.setContentsMargins(0, 0, 0, 0)
        plot_controls_layout.setSpacing(8)
        plot_title = QLabel("Overview plot gallery")
        plot_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        plot_controls_layout.addWidget(plot_title)
        plot_note = QLabel("Use this as a visual orientation board before Missingness, Distributions, QC, and ML export. Each plot is regenerated from the accepted column mapping.")
        plot_note.setWordWrap(True)
        plot_note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent;")
        plot_controls_layout.addWidget(plot_note)

        self.overview_plot_combo = QComboBox()
        self.overview_plot_combo.addItems([
            "Readiness scorecard",
            "Dataset design tiles",
            "Role counts",
            "Task / group counts",
            "Subject × task coverage",
            "Feature-family coverage",
            "Feature quality landscape",
            "Feature availability heatmap",
            "Top missing features",
        ])
        plot_controls_layout.addWidget(self.overview_plot_combo)
        show_btn = QPushButton("Show selected plot")
        show_btn.clicked.connect(self.preview_selected_overview_plot)
        plot_controls_layout.addWidget(show_btn)

        for label, attr in [
            ("Readiness", "overview_readiness_scorecard"),
            ("Design tiles", "overview_design_tiles"),
            ("Feature quality", "overview_feature_quality_landscape"),
            ("Subject × task", "overview_subject_task_matrix"),
        ]:
            b = QPushButton(label)
            b.setProperty("secondary", True)
            b.clicked.connect(lambda _=False, a=attr: self.preview_plot(a))
            plot_controls_layout.addWidget(b)

        regen = QPushButton("Regenerate overview visuals")
        regen.clicked.connect(self.regenerate_overview_plots)
        plot_controls_layout.addWidget(regen)
        open_current = QPushButton("Open current plot file")
        open_current.setProperty("secondary", True)
        open_current.clicked.connect(self.open_current_overview_plot)
        plot_controls_layout.addWidget(open_current)
        plot_controls_layout.addStretch(1)
        plot_controls.setFixedWidth(270)
        plot_panel_layout.addWidget(plot_controls)

        preview_box = QFrame()
        preview_box.setStyleSheet("QFrame { border:none; background:transparent; }")
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)
        self.overview_plot_caption = QLabel("Run Feature Analysis, then select a plot. The plot preview is embedded here and saved to feature_analysis/plots.")
        self.overview_plot_caption.setWordWrap(True)
        self.overview_plot_caption.setStyleSheet(f"color:{MUTED}; background:#FFFFFF; border:1px solid {LINE}; border-radius:8px; padding:9px;")
        preview_layout.addWidget(self.overview_plot_caption)
        self.overview_plot_preview = QLabel("Run Feature Analysis, then select a plot on the left.")
        self.overview_plot_preview.setAlignment(Qt.AlignCenter)
        self.overview_plot_preview.setMinimumHeight(500)
        self.overview_plot_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.overview_plot_preview.setStyleSheet(f"QLabel {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; color:{MUTED}; padding:16px; }}")
        preview_layout.addWidget(self.overview_plot_preview, 1)
        plot_panel_layout.addWidget(preview_box, 1)
        card.layout.addWidget(plot_panel)

        layout.addWidget(card)
        return self._wrap_scroll(body)

    def _metric_tile(self, title: str, value: object, subtitle: str = "") -> QFrame:
        tile = QFrame()
        tile.setStyleSheet(f"QFrame {{ background:#F8FBFE; border:1px solid {LINE}; border-radius:12px; }} QLabel#MetricTitle {{ color:{MUTED}; font-size:11px; font-weight:700; text-transform:uppercase; border:none; }} QLabel#MetricValue {{ color:{NAVY}; font-size:24px; font-weight:900; border:none; }} QLabel#MetricSub {{ color:{MUTED}; font-size:11px; border:none; }}")
        lay = QVBoxLayout(tile)
        lay.setContentsMargins(12, 10, 12, 10)
        lab = QLabel(str(title)); lab.setObjectName("MetricTitle")
        val = QLabel(str(value)); val.setObjectName("MetricValue")
        sub = QLabel(str(subtitle)); sub.setObjectName("MetricSub"); sub.setWordWrap(True)
        lay.addWidget(lab); lay.addWidget(val); lay.addWidget(sub)
        return tile

    def update_overview_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        # Clear metric grid.
        while self.overview_metric_grid.count():
            item = self.overview_metric_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        inv = outputs.get("dataset_inventory", pd.DataFrame())
        readiness = outputs.get("overview_readiness_summary", pd.DataFrame())
        quality = outputs.get("overview_feature_quality_landscape", pd.DataFrame())

        def metric_value(name: str, default: object = "—") -> object:
            if inv.empty or "metric" not in inv.columns:
                return default
            row = inv.loc[inv["metric"].eq(name)]
            return row["value"].iloc[0] if not row.empty else default

        def score_value(name: str, default: object = "—") -> object:
            if readiness.empty or "dimension" not in readiness.columns:
                return default
            row = readiness.loc[readiness["dimension"].astype(str).eq(name)]
            if row.empty:
                return default
            return f"{float(row['score_0_100'].iloc[0]):.0f}"

        n_review = 0
        n_monitor = 0
        if quality is not None and not quality.empty and "quality_status" in quality.columns:
            n_review = int((quality["quality_status"].astype(str) == "review").sum())
            n_monitor = int((quality["quality_status"].astype(str) == "monitor").sum())

        tiles = [
            ("Rows", metric_value("feature_table_rows"), "records / files"),
            ("Features", metric_value("detected_feature_columns"), "mapped feature columns"),
            ("Numeric", metric_value("numeric_feature_columns"), "usable for quantitative audit"),
            ("Completeness", score_value("Feature completeness"), "0–100 orientation score"),
            ("Design", score_value("Design richness"), "group/task/session context"),
            ("Review flags", n_review, f"monitor: {n_monitor}"),
            ("Subjects", metric_value("unique_subjects"), "if subject_id exists"),
            ("Tasks", metric_value("unique_tasks"), "if task exists"),
            ("QC", "yes" if str(metric_value("qc_table_loaded", False)).lower() == "true" else "no", "artifact context"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.overview_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx // 3, idx % 3)

        self._fill_table(self.overview_readiness_table, outputs.get("overview_readiness_summary", pd.DataFrame()))
        self._fill_table(self.overview_inventory_table, outputs.get("dataset_inventory", pd.DataFrame()))
        self._fill_table(self.overview_roles_table, outputs.get("feature_role_summary", pd.DataFrame()))
        self._fill_table(self.overview_design_table, outputs.get("dataset_design_overview", pd.DataFrame()))
        self._fill_table(self.overview_family_table, outputs.get("feature_family_overview", pd.DataFrame()))
        self._fill_table(self.overview_quality_table, outputs.get("overview_feature_quality_landscape", pd.DataFrame()))
        self.overview_note.setText(
            "Overview generated. Use this page to verify design context, mapped feature coverage, available metadata/QC context, and major feature-quality risks before interpreting downstream modules or exporting to ML."
        )

    def _analysis_dirs(self) -> tuple[Path, Path, Path, Path]:
        if not getattr(self, "output_dir", None):
            self.output_dir = Path(self.output_edit.text().strip()) / "feature_analysis"
        tables_dir = self.output_dir / "tables"
        reports_dir = self.output_dir / "reports"
        plots_dir = self.output_dir / "plots"
        for d in (tables_dir, reports_dir, plots_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self.output_dir, tables_dir, reports_dir, plots_dir

    def _build_analysis_outputs(self) -> tuple[dict[str, pd.DataFrame], list[str]]:
        if self.feature_df is None:
            self.load_and_map()
        if self.feature_df is None:
            raise RuntimeError("Load a primary feature table first.")
        mapping = self.collect_mapping_from_table()
        roles = role_lists(mapping)
        feature_cols = roles.get("Feature", [])
        inventory = dataset_inventory(self.feature_df, self.qc_df, self.meta_df, mapping)
        role_sum = role_summary(mapping)
        design = design_overview(self.feature_df, mapping)
        family = feature_family_overview(self.feature_df, feature_cols, self.registry_df)
        groups = group_counts(self.feature_df)
        dist = feature_distribution_summary(self.feature_df, feature_cols)
        outlier_flags = robust_outlier_flags(self.feature_df, feature_cols, self.registry_df)
        range_flags = expected_range_flags(self.feature_df, feature_cols, self.registry_df)
        dist_review = distribution_review_summary(dist, range_flags)
        shape_audit = distribution_shape_audit(dist, range_flags)
        row_outlier_burden = row_outlier_burden_summary(outlier_flags, len(feature_cols))
        feature_missing = missingness_feature_summary(self.feature_df, feature_cols, self.registry_df)
        row_missing = missingness_row_summary(self.feature_df, feature_cols)
        group_missing = missingness_group_summary(self.feature_df, feature_cols)
        family_missing = missingness_family_summary(feature_missing)
        comissing = missingness_comissing_pairs(self.feature_df, feature_cols)
        quality_landscape = overview_feature_quality_landscape(dist, self.registry_df)
        readiness = overview_readiness_summary(self.feature_df, self.qc_df, self.meta_df, mapping, dist, feature_missing, groups)
        qc_corr = feature_qc_correlations(self.feature_df, self.qc_df, feature_cols)
        qc_catalog = qc_metric_catalog(self.qc_df)
        qc_family = qc_family_burden_summary(self.qc_df)
        qc_row_burden = qc_row_burden_summary(self.qc_df)
        qc_family_assoc = feature_qc_family_association(qc_corr)
        qc_missing_assoc = qc_missingness_associations(self.feature_df, self.qc_df, feature_cols)
        qc_outlier_assoc = qc_outlier_associations(outlier_flags, self.qc_df)
        qc_summary = qc_integration_summary(self.qc_df, qc_corr, qc_family, qc_missing_assoc, qc_outlier_assoc)
        reliability = reliability_screen(dist, qc_corr)
        outputs = {
            "dataset_inventory": inventory,
            "feature_role_summary": role_sum,
            "dataset_design_overview": design,
            "feature_family_overview": family,
            "overview_readiness_summary": readiness,
            "overview_feature_quality_landscape": quality_landscape,
            "group_counts": groups,
            "feature_column_mapping": mapping,
            "feature_distribution_summary": dist,
            "robust_outlier_flags": outlier_flags,
            "feature_expected_range_flags": range_flags,
            "distribution_review_summary": dist_review,
            "distribution_shape_audit": shape_audit,
            "row_outlier_burden_summary": row_outlier_burden,
            "missingness_by_feature": feature_missing,
            "missingness_by_row": row_missing,
            "missingness_by_group": group_missing,
            "missingness_by_family": family_missing,
            "missingness_comissing_pairs": comissing,
            "feature_qc_spearman_correlation": qc_corr,
            "qc_metric_catalog": qc_catalog,
            "qc_family_burden_summary": qc_family,
            "qc_row_burden_summary": qc_row_burden,
            "feature_qc_family_association": qc_family_assoc,
            "qc_missingness_associations": qc_missing_assoc,
            "qc_outlier_associations": qc_outlier_assoc,
            "qc_integration_summary": qc_summary,
            "feature_reliability_screen": reliability,
        }
        return outputs, feature_cols

    def _write_outputs(self, outputs: dict[str, pd.DataFrame], tables_dir: Path) -> None:
        for name, df in outputs.items():
            df.to_csv(tables_dir / f"{name}.csv", index=False)

    def _generate_overview_plots(self, outputs: dict[str, pd.DataFrame], feature_cols: list[str], plots_dir: Path) -> dict[str, str]:
        paths = {}
        paths["overview_readiness_scorecard"] = str(plot_overview_readiness_scorecard(outputs.get("overview_readiness_summary", pd.DataFrame()), plots_dir / "overview_readiness_scorecard.png"))
        paths["overview_design_tiles"] = str(plot_dataset_design_tiles(outputs.get("dataset_design_overview", pd.DataFrame()), plots_dir / "overview_design_tiles.png"))
        paths["role_counts"] = str(plot_role_counts(outputs.get("feature_role_summary", pd.DataFrame()), plots_dir / "overview_role_counts.png"))
        paths["group_counts"] = str(plot_group_counts(outputs.get("group_counts", pd.DataFrame()), plots_dir / "overview_group_counts.png"))
        paths["overview_subject_task_matrix"] = str(plot_subject_task_matrix(self.feature_df, plots_dir / "overview_subject_task_matrix.png"))
        paths["feature_family_counts"] = str(plot_feature_family_counts(outputs.get("feature_family_overview", pd.DataFrame()), plots_dir / "overview_feature_family_counts.png"))
        paths["overview_feature_family_quality"] = str(plot_feature_family_quality(outputs.get("feature_family_overview", pd.DataFrame()), plots_dir / "overview_feature_family_quality.png"))
        paths["overview_feature_quality_landscape"] = str(plot_feature_quality_landscape(outputs.get("overview_feature_quality_landscape", pd.DataFrame()), plots_dir / "overview_feature_quality_landscape.png"))
        paths["missingness_top_features"] = str(plot_missingness(outputs.get("feature_distribution_summary", pd.DataFrame()), plots_dir / "missingness_top_features.png"))
        paths["feature_availability_heatmap"] = str(plot_feature_availability_heatmap(self.feature_df, feature_cols, plots_dir / "feature_availability_heatmap.png"))
        # Missingness-specific plots are generated at the same time so that the Missingness page is immediately usable.
        paths["missingness_row_distribution"] = str(plot_row_missingness_distribution(outputs.get("missingness_by_row", pd.DataFrame()), plots_dir / "missingness_row_distribution.png"))
        paths["missingness_by_group"] = str(plot_missingness_by_group(outputs.get("missingness_by_group", pd.DataFrame()), plots_dir / "missingness_by_group.png"))
        paths["missingness_by_family"] = str(plot_missingness_family_summary(outputs.get("missingness_by_family", pd.DataFrame()), plots_dir / "missingness_by_family.png"))
        paths["missingness_comissing_heatmap"] = str(plot_comissing_heatmap(self.feature_df, feature_cols, plots_dir / "missingness_comissing_heatmap.png"))
        paths["distribution_review_status"] = str(plot_distribution_review_summary(outputs.get("distribution_review_summary", pd.DataFrame()), plots_dir / "distribution_review_status.png"))
        paths["distribution_shape_summary"] = str(plot_distribution_shape_summary(outputs.get("distribution_shape_audit", pd.DataFrame()), plots_dir / "distribution_shape_summary.png"))
        paths["distribution_shape_landscape"] = str(plot_distribution_shape_landscape(outputs.get("distribution_shape_audit", pd.DataFrame()), plots_dir / "distribution_shape_landscape.png"))
        paths["row_outlier_burden"] = str(plot_row_outlier_burden(outputs.get("row_outlier_burden_summary", pd.DataFrame()), plots_dir / "row_outlier_burden.png"))
        paths["variance_screen"] = str(plot_variance_screen(outputs.get("feature_distribution_summary", pd.DataFrame()), plots_dir / "variance_screen.png"))
        paths["expected_range_flags"] = str(plot_expected_range_flags(outputs.get("feature_expected_range_flags", pd.DataFrame()), plots_dir / "feature_expected_range_flags.png"))
        paths["outlier_counts"] = str(plot_outlier_counts(outputs.get("robust_outlier_flags", pd.DataFrame()), plots_dir / "outlier_counts.png"))
        focus_cols = list(feature_cols)
        shape_for_grid = outputs.get("distribution_shape_audit", pd.DataFrame())
        if shape_for_grid is not None and not shape_for_grid.empty and "feature" in shape_for_grid.columns:
            focus_cols = [c for c in shape_for_grid["feature"].astype(str).tolist() if c in self.feature_df.columns] or focus_cols
        paths["feature_distribution_grid"] = str(plot_distribution_grid(self.feature_df, focus_cols, plots_dir / "feature_distribution_grid.png"))
        first_feature = feature_cols[0] if feature_cols else None
        if first_feature:
            paths["selected_feature_distribution"] = str(plot_selected_feature_diagnostic(self.feature_df, first_feature, plots_dir / "selected_feature_distribution.png"))
            paths["selected_feature_by_group"] = str(plot_group_feature_boxplot(self.feature_df, first_feature, plots_dir / "selected_feature_by_group.png"))
        paths["qc_artifact_model"] = str(plot_qc_artifact_model(plots_dir / "qc_artifact_model.png"))
        paths["qc_family_burden"] = str(plot_qc_family_burden(outputs.get("qc_family_burden_summary", pd.DataFrame()), plots_dir / "qc_family_burden.png"))
        paths["qc_metric_distributions"] = str(plot_qc_metric_distributions(self.qc_df, outputs.get("qc_metric_catalog", pd.DataFrame()), plots_dir / "qc_metric_distributions.png"))
        paths["qc_feature_association_heatmap"] = str(plot_qc_feature_association_heatmap(outputs.get("feature_qc_family_association", pd.DataFrame()), plots_dir / "qc_feature_association_heatmap.png"))
        paths["qc_top_feature_associations"] = str(plot_qc_top_feature_associations(outputs.get("feature_qc_spearman_correlation", pd.DataFrame()), plots_dir / "qc_top_feature_associations.png"))
        paths["qc_missingness_associations"] = str(plot_qc_missingness_associations(outputs.get("qc_missingness_associations", pd.DataFrame()), plots_dir / "qc_missingness_associations.png"))
        paths["qc_row_burden"] = str(plot_qc_row_burden(outputs.get("qc_row_burden_summary", pd.DataFrame()), plots_dir / "qc_row_burden.png"))
        qcols = outputs.get("qc_metric_catalog", pd.DataFrame()).get("qc_variable", pd.Series(dtype=str)).astype(str).tolist()
        if first_feature and qcols:
            paths["selected_feature_qc_scatter"] = str(plot_selected_feature_qc_scatter(self.feature_df, self.qc_df, first_feature, qcols[0], plots_dir / "selected_feature_qc_scatter.png"))
        return paths

    def preview_selected_overview_plot(self) -> None:
        label = self.overview_plot_combo.currentText() if hasattr(self, "overview_plot_combo") else ""
        key_map = {
            "Readiness scorecard": "overview_readiness_scorecard",
            "Dataset design tiles": "overview_design_tiles",
            "Role counts": "role_counts",
            "Task / group counts": "group_counts",
            "Subject × task coverage": "overview_subject_task_matrix",
            "Feature-family coverage": "overview_feature_family_quality",
            "Feature quality landscape": "overview_feature_quality_landscape",
            "Feature availability heatmap": "feature_availability_heatmap",
            "Top missing features": "missingness_top_features",
        }
        self.preview_plot(key_map.get(label, "overview_readiness_scorecard"))

    def _overview_plot_caption_text(self, key: str) -> str:
        captions = {
            "overview_readiness_scorecard": "Readiness scorecard: quick orientation across completeness, numeric analyzability, metadata context, QC context, design richness, row depth, and feature breadth. It is not an ML score.",
            "overview_design_tiles": "Dataset design tiles: shows whether subject, session, task, diagnosis/severity, sex/gender, and device columns are detected. Missing design variables limit stratified analyses.",
            "role_counts": "Role counts: verifies that columns are mapped as features, identifiers, targets, covariates, QC variables, or ignored columns before analysis proceeds.",
            "group_counts": "Group counts: shows balance across the most relevant detected grouping variable, usually task or diagnosis. Severe imbalance affects interpretation and downstream ML splitting.",
            "overview_subject_task_matrix": "Subject × task coverage: shows repeated-measures/task coverage. Empty cells reveal missing task coverage and potential bias in task-specific summaries.",
            "feature_family_counts": "Feature-family counts: shows coverage across acoustic/kinematic/other feature subsystems when registry labels are available.",
            "overview_feature_family_quality": "Feature-family quality: combines family coverage with average missingness. Use it to see whether an entire subsystem is weak, not just individual features.",
            "overview_feature_quality_landscape": "Feature quality landscape: each feature is positioned by missingness and robust outlier burden. Review-zone features need inspection before ML export.",
            "feature_availability_heatmap": "Feature availability heatmap: row-by-feature matrix showing available versus missing feature values. Blocks of missingness often indicate task or computation support issues.",
            "missingness_top_features": "Top missing features: first-pass ranking of features with the highest missingness. Detailed missingness review is handled in the Missingness menu.",
        }
        return captions.get(key, "Overview plot.")

    def regenerate_overview_plots(self) -> None:
        try:
            if self.feature_df is None:
                self.load_and_map()
            if self.feature_df is None:
                return
            if not self.output_edit.text().strip():
                QMessageBox.warning(self, "Missing output folder", "Please select an output folder before generating plots.")
                return
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            outputs, feature_cols = self._build_analysis_outputs()
            self.outputs = outputs
            self._write_outputs(outputs, tables_dir)
            self.plot_paths = self._generate_overview_plots(outputs, feature_cols, plots_dir)
            self.populate_output_tables(outputs)
            self.update_overview_dashboard(outputs)
            self.update_missingness_dashboard(outputs)
            self.update_distribution_dashboard(outputs)
            self.update_qc_dashboard(outputs)
            self.log(f"Overview/missingness plots generated in: {plots_dir}")
            self.preview_plot("role_counts", generate_if_missing=False)
        except Exception as exc:
            QMessageBox.critical(self, "Could not generate overview plots", str(exc))

    def preview_plot(self, key: str, generate_if_missing: bool = True) -> None:
        if generate_if_missing and (not hasattr(self, "plot_paths") or key not in self.plot_paths or not Path(self.plot_paths.get(key, "")).exists()):
            self.regenerate_overview_plots()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths:
            QMessageBox.information(self, "Plot unavailable", "Run Feature Analysis first, or this plot was not generated for the current dataset.")
            return
        path = Path(self.plot_paths[key])
        if not path.exists():
            QMessageBox.information(self, "Plot unavailable", f"Plot file not found:\n{path}")
            return
        self.current_overview_plot = path
        if hasattr(self, "overview_plot_caption"):
            self.overview_plot_caption.setText(self._overview_plot_caption_text(key))
        self._show_overview_plot(path)

    def _show_overview_plot(self, path: Path) -> None:
        if not hasattr(self, "overview_plot_preview"):
            return
        pix = QPixmap(str(path))
        if pix.isNull():
            self.overview_plot_preview.setText(f"Could not load plot:\n{path}")
            return
        target_size = self.overview_plot_preview.size()
        scaled = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.overview_plot_preview.setPixmap(scaled)
        self.overview_plot_preview.setToolTip(str(path))

    def open_current_overview_plot(self) -> None:
        path = getattr(self, "current_overview_plot", None)
        if not path:
            QMessageBox.information(self, "No current plot", "Preview a plot first, then use this button to open the full-resolution file.")
            return
        self.open_file(Path(path))

    def open_file(self, path: Path) -> None:
        if not path.exists():
            QMessageBox.information(self, "File unavailable", f"File not found:\n{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def open_plot(self, key: str) -> None:
        if not hasattr(self, "plot_paths") or key not in self.plot_paths:
            QMessageBox.information(self, "Plot unavailable", "Run Feature Analysis first, or this plot was not generated for the current dataset.")
            return
        path = Path(self.plot_paths[key])
        if not path.exists():
            QMessageBox.information(self, "Plot unavailable", f"Plot file not found:\n{path}")
            return
        self.open_file(path)


    def _missingness_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        card = Card(
            "Missingness",
            "Audit feature failure and data availability. Missingness may be informative in clinical datasets; inspect patterns before imputation, exclusion, or ML."
        )

        self.missing_metric_grid = QGridLayout()
        self.missing_metric_grid.setHorizontalSpacing(12)
        self.missing_metric_grid.setVerticalSpacing(12)
        card.layout.addLayout(self.missing_metric_grid)

        self.missing_note = QLabel("Run Feature Analysis to populate missingness tables and plots. Review feature-level, row-level, family-level, and group-level patterns before modeling.")
        self.missing_note.setWordWrap(True)
        self.missing_note.setStyleSheet(f"color:{MUTED}; background:#F7FAFD; border:1px solid {LINE}; border-radius:8px; padding:10px;")
        card.layout.addWidget(self.missing_note)

        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {LINE}; border-radius: 8px; background: #FFFFFF; }}
            QTabBar::tab {{ background:#F7FAFD; color:{NAVY}; border:1px solid {LINE}; border-bottom:none; padding:8px 14px; min-height:22px; }}
            QTabBar::tab:selected {{ background:#FFFFFF; color:{NAVY}; border-top:2px solid {TEAL}; font-weight:700; }}
            QTabBar::tab:hover {{ background:#F3FAF9; color:{NAVY}; }}
        """)
        self.missing_feature_table = QTableWidget(0, 0)
        self.missing_row_table = QTableWidget(0, 0)
        self.missing_group_table = QTableWidget(0, 0)
        self.missing_family_table = QTableWidget(0, 0)
        self.missing_comissing_table = QTableWidget(0, 0)
        for t in [self.missing_feature_table, self.missing_row_table, self.missing_group_table, self.missing_family_table, self.missing_comissing_table]:
            t.setAlternatingRowColors(True)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tabs.addTab(self.missing_feature_table, "By feature")
        tabs.addTab(self.missing_row_table, "By row / recording")
        tabs.addTab(self.missing_group_table, "By group")
        tabs.addTab(self.missing_family_table, "By family")
        tabs.addTab(self.missing_comissing_table, "Co-missing pairs")
        card.layout.addWidget(tabs)

        plot_panel = QFrame()
        plot_panel.setStyleSheet(f"QFrame {{ background:#F8FBFE; border:1px solid {LINE}; border-radius:12px; }}")
        plot_panel_layout = QHBoxLayout(plot_panel)
        plot_panel_layout.setContentsMargins(14, 14, 14, 14)
        plot_panel_layout.setSpacing(14)

        controls = QFrame()
        controls.setStyleSheet("QFrame { border:none; background:transparent; }")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        title = QLabel("Missingness plots")
        title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        controls_layout.addWidget(title)
        note = QLabel("These plots identify whether feature failure is global, feature-specific, subject/task-specific, or family-specific. They are descriptive screening tools, not exclusion rules.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent;")
        controls_layout.addWidget(note)
        guide = QLabel("Interpretation guide:<br>• &lt;20%: usually low concern<br>• 20–50%: monitor mechanism<br>• ≥50%: review before ML<br>• blocks/clusters: possible task, QC, or computation support problem")
        guide.setWordWrap(True)
        guide.setStyleSheet(f"color:{INK}; background:#FFFFFF; border:1px solid {LINE}; border-radius:8px; padding:9px; font-size:12px;")
        controls_layout.addWidget(guide)
        for label, key in [
            ("Top missing features", "missingness_top_features"),
            ("Row-level missingness", "missingness_row_distribution"),
            ("Missingness by group", "missingness_by_group"),
            ("Missingness by family", "missingness_by_family"),
            ("Feature availability", "feature_availability_heatmap"),
            ("Co-missing heatmap", "missingness_comissing_heatmap"),
        ]:
            b = QPushButton(label)
            b.setProperty("secondary", True)
            b.clicked.connect(lambda _=False, k=key: self.preview_missingness_plot(k))
            controls_layout.addWidget(b)
        regen = QPushButton("Regenerate missingness plots")
        regen.clicked.connect(self.regenerate_overview_plots)
        controls_layout.addWidget(regen)
        open_current = QPushButton("Open current plot file")
        open_current.setProperty("secondary", True)
        open_current.clicked.connect(self.open_current_missingness_plot)
        controls_layout.addWidget(open_current)
        controls_layout.addStretch(1)
        controls.setFixedWidth(260)
        plot_panel_layout.addWidget(controls)

        self.missing_plot_preview = QLabel("Run Feature Analysis, then select a missingness plot on the left.")
        self.missing_plot_preview.setAlignment(Qt.AlignCenter)
        self.missing_plot_preview.setMinimumHeight(430)
        self.missing_plot_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.missing_plot_preview.setStyleSheet(f"QLabel {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; color:{MUTED}; padding:16px; }}")
        right_preview = QFrame()
        right_preview.setStyleSheet("QFrame { border:none; background:transparent; }")
        right_layout = QVBoxLayout(right_preview)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        right_layout.addWidget(self.missing_plot_preview, 1)
        self.missing_plot_caption = QLabel("Select a missingness plot. A short interpretation guide will appear here so the plot can be read without guessing.")
        self.missing_plot_caption.setWordWrap(True)
        self.missing_plot_caption.setStyleSheet(f"color:{INK}; background:#FFFFFF; border:1px solid {LINE}; border-radius:8px; padding:10px;")
        right_layout.addWidget(self.missing_plot_caption)
        plot_panel_layout.addWidget(right_preview, 1)
        card.layout.addWidget(plot_panel)

        layout.addWidget(card)
        return self._wrap_scroll(body)

    def update_missingness_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "missing_metric_grid"):
            return
        while self.missing_metric_grid.count():
            item = self.missing_metric_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        feat = outputs.get("missingness_by_feature", pd.DataFrame())
        row = outputs.get("missingness_by_row", pd.DataFrame())
        group = outputs.get("missingness_by_group", pd.DataFrame())
        n_features = len(feat) if feat is not None else 0
        mean_miss = "—"
        high_features = "—"
        high_rows = "—"
        groups = "—"
        if feat is not None and not feat.empty and "missing_fraction" in feat.columns:
            mean_miss = f"{pd.to_numeric(feat['missing_fraction'], errors='coerce').mean():.3f}"
            high_features = int(feat.get("missingness_status", pd.Series(dtype=str)).isin(["review", "high_review"]).sum())
        if row is not None and not row.empty and "missing_fraction_feature_columns" in row.columns:
            high_rows = int((pd.to_numeric(row["missing_fraction_feature_columns"], errors="coerce") >= 0.50).sum())
        if group is not None and not group.empty and "group_variable" in group.columns:
            groups = int(group["group_variable"].nunique())
        tiles = [
            ("Features audited", n_features, "selected feature columns"),
            ("Mean missingness", mean_miss, "across selected features"),
            ("Review features", high_features, "≥50% missing or worse"),
            ("Review rows", high_rows, "≥50% selected features missing"),
            ("Group screens", groups, "available metadata strata"),
            ("Default policy", "do not auto-impute", "review mechanism first"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.missing_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx // 3, idx % 3)
        self._fill_table(self.missing_feature_table, outputs.get("missingness_by_feature", pd.DataFrame()))
        self._fill_table(self.missing_row_table, outputs.get("missingness_by_row", pd.DataFrame()))
        self._fill_table(self.missing_group_table, outputs.get("missingness_by_group", pd.DataFrame()))
        self._fill_table(self.missing_family_table, outputs.get("missingness_by_family", pd.DataFrame()))
        self._fill_table(self.missing_comissing_table, outputs.get("missingness_comissing_pairs", pd.DataFrame()))
        self.missing_note.setText("Missingness audit generated. Read this menu as a missing-data mechanism screen: first identify high-missing features, then check whether missingness is concentrated in rows, tasks/groups, feature families, or co-missing clusters. Imputation and complete-case exclusion should be deferred to ML only after this review.")

    def _missingness_plot_caption_text(self, key: str) -> str:
        captions = {
            "missingness_top_features": "Top missing features: ranks feature columns by missing fraction. Features near the top may be unsupported for some tasks, sensitive to signal quality, or computationally unstable. Do not exclude automatically; first check whether missingness is task-, group-, or QC-linked.",
            "missingness_row_distribution": "Row-level missingness: shows how much feature information is lost per recording/row. A right-shifted distribution means many recordings have broad feature failure, which can reduce usable sample size and bias ML training.",
            "missingness_by_group": "Missingness by group: compares average feature missingness across detected task, diagnosis, severity, device, session, or similar groups. Group differences suggest missingness may be non-random and should not be handled by naive complete-case analysis.",
            "missingness_by_family": "Missingness by family: summarizes failure by feature subsystem. A high family-level value suggests a systematic issue, such as formant tracking, voicing detection, segmentation support, or missing registry labels, rather than isolated bad features.",
            "feature_availability_heatmap": "Feature availability heatmap: rows are recordings and columns are features. Contiguous missing blocks suggest structured missingness by task, modality, subject group, or computation mode; scattered gaps suggest more local feature failure.",
            "missingness_comissing_heatmap": "Co-missing heatmap: shows features that fail together. Strong co-missing clusters often indicate shared algorithmic dependencies or task support limitations, and they should be reviewed as groups rather than one feature at a time.",
        }
        return captions.get(key, "Missingness plot. Use it to determine whether feature absence is isolated, structured, or associated with design/QC variables.")

    def preview_missingness_plot(self, key: str) -> None:
        if not hasattr(self, "plot_paths") or key not in self.plot_paths or not Path(self.plot_paths.get(key, "")).exists():
            self.regenerate_overview_plots()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths:
            QMessageBox.information(self, "Plot unavailable", "Run Feature Analysis first, or this plot was not generated for the current dataset.")
            return
        path = Path(self.plot_paths[key])
        if not path.exists():
            QMessageBox.information(self, "Plot unavailable", f"Plot file not found:\n{path}")
            return
        self.current_missingness_plot = path
        if hasattr(self, "missing_plot_caption"):
            self.missing_plot_caption.setText(self._missingness_plot_caption_text(key))
        pix = QPixmap(str(path))
        if pix.isNull():
            self.missing_plot_preview.setText(f"Could not load plot:\n{path}")
            return
        scaled = pix.scaled(self.missing_plot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.missing_plot_preview.setPixmap(scaled)
        self.missing_plot_preview.setToolTip(str(path))

    def open_current_missingness_plot(self) -> None:
        path = getattr(self, "current_missingness_plot", None)
        if not path:
            QMessageBox.information(self, "No current plot", "Preview a plot first, then use this button to open the full-resolution file.")
            return
        self.open_file(Path(path))


    def _distributions_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        card = Card(
            "Distributions / Outliers",
            "Inspect feature distributions, robust outliers, expected-range flags, and group/context overlays. These are review tools, not automatic exclusion decisions."
        )

        self.dist_metric_grid = QGridLayout()
        self.dist_metric_grid.setHorizontalSpacing(12)
        self.dist_metric_grid.setVerticalSpacing(12)
        card.layout.addLayout(self.dist_metric_grid)

        self.dist_note = QLabel("Run Feature Analysis to populate distribution diagnostics. Review outliers in context: task, QC, device, diagnosis/severity, and implementation status can all affect feature values.")
        self.dist_note.setWordWrap(True)
        self.dist_note.setStyleSheet(f"color:{MUTED}; background:#F7FAFD; border:1px solid {LINE}; border-radius:8px; padding:10px;")
        card.layout.addWidget(self.dist_note)

        top_controls = QHBoxLayout()
        top_controls.addWidget(QLabel("Feature to inspect:"))
        self.dist_feature_combo = QComboBox()
        self.dist_feature_combo.setMinimumWidth(360)
        self.dist_feature_combo.currentTextChanged.connect(lambda _: self.generate_selected_distribution_plot())
        top_controls.addWidget(self.dist_feature_combo)
        top_controls.addWidget(QLabel("Group overlay:"))
        self.dist_group_combo = QComboBox()
        self.dist_group_combo.setMinimumWidth(240)
        self.dist_group_combo.currentTextChanged.connect(lambda _: self.generate_selected_group_plot())
        top_controls.addWidget(self.dist_group_combo)
        top_controls.addStretch(1)
        card.layout.addLayout(top_controls)

        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {LINE}; border-radius: 8px; background: #FFFFFF; }}
            QTabBar::tab {{ background:#F7FAFD; color:{NAVY}; border:1px solid {LINE}; border-bottom:none; padding:8px 14px; min-height:22px; }}
            QTabBar::tab:selected {{ background:#FFFFFF; color:{NAVY}; border-top:2px solid {TEAL}; font-weight:700; }}
            QTabBar::tab:hover {{ background:#F3FAF9; color:{NAVY}; }}
        """)
        self.dist_summary_table = QTableWidget(0, 0)
        self.dist_review_table = QTableWidget(0, 0)
        self.dist_outlier_table = QTableWidget(0, 0)
        self.dist_range_table = QTableWidget(0, 0)
        self.dist_shape_table = QTableWidget(0, 0)
        self.dist_row_burden_table = QTableWidget(0, 0)
        for t in [self.dist_summary_table, self.dist_review_table, self.dist_outlier_table, self.dist_range_table, self.dist_shape_table, self.dist_row_burden_table]:
            t.setAlternatingRowColors(True)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tabs.addTab(self.dist_summary_table, "Distribution summary")
        tabs.addTab(self.dist_review_table, "Review summary")
        tabs.addTab(self.dist_outlier_table, "Row-level outliers")
        tabs.addTab(self.dist_range_table, "Expected ranges")
        tabs.addTab(self.dist_shape_table, "Shape audit")
        tabs.addTab(self.dist_row_burden_table, "Row burden")
        card.layout.addWidget(tabs)

        plot_panel = QFrame()
        plot_panel.setStyleSheet(f"QFrame {{ background:#F8FBFE; border:1px solid {LINE}; border-radius:12px; }}")
        plot_panel_layout = QHBoxLayout(plot_panel)
        plot_panel_layout.setContentsMargins(14, 14, 14, 14)
        plot_panel_layout.setSpacing(14)

        controls = QFrame()
        controls.setStyleSheet("QFrame { border:none; background:transparent; }")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        title = QLabel("Distribution plots")
        title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        controls_layout.addWidget(title)
        note = QLabel("Use the feature-specific plots for clinical review. A statistical outlier is not automatically wrong; it may be true physiology, task effect, device/QC artifact, or computation failure.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent;")
        controls_layout.addWidget(note)
        for label, key in [
            ("Review status", "distribution_review_status"),
            ("Shape priority", "distribution_shape_summary"),
            ("Shape landscape", "distribution_shape_landscape"),
            ("Expected-range flags", "expected_range_flags"),
            ("Outlier counts", "outlier_counts"),
            ("Row outlier burden", "row_outlier_burden"),
            ("Variance screen", "variance_screen"),
            ("Feature diagnostic", "selected_feature_distribution"),
            ("Selected feature by group", "selected_feature_by_group"),
            ("Top feature grid", "feature_distribution_grid"),
        ]:
            b = QPushButton(label)
            b.setProperty("secondary", True)
            b.clicked.connect(lambda _=False, k=key: self.preview_distribution_plot(k))
            controls_layout.addWidget(b)
        regen = QPushButton("Regenerate distribution plots")
        regen.clicked.connect(self.regenerate_overview_plots)
        controls_layout.addWidget(regen)
        open_current = QPushButton("Open current plot file")
        open_current.setProperty("secondary", True)
        open_current.clicked.connect(self.open_current_distribution_plot)
        controls_layout.addWidget(open_current)
        controls_layout.addStretch(1)
        controls.setFixedWidth(270)
        plot_panel_layout.addWidget(controls)

        self.dist_plot_preview = QLabel("Run Feature Analysis, then select a distribution plot on the left.")
        self.dist_plot_preview.setAlignment(Qt.AlignCenter)
        self.dist_plot_preview.setMinimumHeight(520)
        self.dist_plot_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dist_plot_preview.setStyleSheet(f"QLabel {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; color:{MUTED}; padding:16px; }}")
        plot_panel_layout.addWidget(self.dist_plot_preview, 1)

        self.dist_interpretation_label = QLabel("Select a plot to see: what it shows, what is concerning, what not to conclude, and what to check next.")
        self.dist_interpretation_label.setWordWrap(True)
        self.dist_interpretation_label.setMinimumWidth(300)
        self.dist_interpretation_label.setMaximumWidth(390)
        self.dist_interpretation_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.dist_interpretation_label.setStyleSheet(f"QLabel {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; color:{INK}; padding:14px; font-size:12px; line-height:140%; }}")
        plot_panel_layout.addWidget(self.dist_interpretation_label)
        card.layout.addWidget(plot_panel)

        layout.addWidget(card)
        return self._wrap_scroll(body)

    def update_distribution_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "dist_metric_grid"):
            return
        while self.dist_metric_grid.count():
            item = self.dist_metric_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        dist = outputs.get("feature_distribution_summary", pd.DataFrame())
        review = outputs.get("distribution_review_summary", pd.DataFrame())
        outliers = outputs.get("robust_outlier_flags", pd.DataFrame())
        ranges = outputs.get("feature_expected_range_flags", pd.DataFrame())
        shape = outputs.get("distribution_shape_audit", pd.DataFrame())
        row_burden = outputs.get("row_outlier_burden_summary", pd.DataFrame())
        n_features = len(dist) if dist is not None else 0
        monitor = review.get("distribution_status", pd.Series(dtype=str)).isin(["monitor"]).sum() if review is not None and not review.empty else 0
        review_n = review.get("distribution_status", pd.Series(dtype=str)).isin(["review"]).sum() if review is not None and not review.empty else 0
        out_n = len(outliers) if outliers is not None else 0
        range_n = int(pd.to_numeric(ranges.get("fraction_outside_expected", pd.Series(dtype=float)), errors="coerce").gt(0).sum()) if ranges is not None and not ranges.empty else 0
        zero_n = int(dist.get("zero_variance", pd.Series(dtype=bool)).fillna(False).sum()) if dist is not None and not dist.empty else 0
        shape_review = int(shape.get("priority", pd.Series(dtype=str)).astype(str).eq("review").sum()) if shape is not None and not shape.empty else 0
        row_review = int(row_burden.get("review_level", pd.Series(dtype=str)).astype(str).eq("review").sum()) if row_burden is not None and not row_burden.empty else 0
        tiles = [
            ("Features audited", n_features, "numeric selected features"),
            ("Monitor", int(monitor), "moderate distribution issues"),
            ("Review", int(review_n), "high-risk distribution issues"),
            ("Outlier rows", out_n, "row-level robust/range flags"),
            ("Range flags", range_n, "features outside supplied ranges"),
            ("Zero variance", zero_n, "not useful for ML"),
            ("Shape review", shape_review, "features needing shape review"),
            ("Row burden", row_review, "recordings with many flags"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.dist_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx // 3, idx % 3)
        self._fill_table(self.dist_summary_table, dist)
        self._fill_table(self.dist_review_table, review)
        self._fill_table(self.dist_outlier_table, outliers)
        self._fill_table(self.dist_range_table, ranges)
        self._fill_table(self.dist_shape_table, shape)
        self._fill_table(self.dist_row_burden_table, row_burden)
        if hasattr(self, "dist_feature_combo"):
            current = self.dist_feature_combo.currentText()
            self.dist_feature_combo.blockSignals(True)
            self.dist_feature_combo.clear()
            if dist is not None and not dist.empty and "feature" in dist.columns:
                self.dist_feature_combo.addItems([str(x) for x in dist["feature"].dropna().tolist()])
            if current:
                ix = self.dist_feature_combo.findText(current)
                if ix >= 0:
                    self.dist_feature_combo.setCurrentIndex(ix)
            self.dist_feature_combo.blockSignals(False)
        if hasattr(self, "dist_group_combo"):
            current = self.dist_group_combo.currentText()
            self.dist_group_combo.blockSignals(True)
            self.dist_group_combo.clear()
            self.dist_group_combo.addItem("Auto")
            if self.feature_df is not None:
                for c in ["task", "diagnosis", "severity_bin", "modality", "sex", "gender", "session_id", "subject_id"]:
                    if c in self.feature_df.columns:
                        self.dist_group_combo.addItem(c)
            if current:
                ix = self.dist_group_combo.findText(current)
                if ix >= 0:
                    self.dist_group_combo.setCurrentIndex(ix)
            self.dist_group_combo.blockSignals(False)
        self.dist_note.setText("Distribution audit generated. Review statistical outliers together with expected ranges, QC artifacts, task compatibility, and clinical/context metadata before excluding or transforming features.")

    def _selected_distribution_feature(self) -> str | None:
        if not hasattr(self, "dist_feature_combo"):
            return None
        val = self.dist_feature_combo.currentText().strip()
        return val or None

    def _selected_distribution_group(self) -> str | None:
        if not hasattr(self, "dist_group_combo"):
            return None
        val = self.dist_group_combo.currentText().strip()
        return None if val in ["", "Auto"] else val

    def _selected_expected_bounds(self, feature: str | None) -> tuple[float | None, float | None]:
        if not feature or not getattr(self, "outputs", None):
            return None, None
        ranges = self.outputs.get("feature_expected_range_flags", pd.DataFrame())
        if ranges is None or ranges.empty or "feature" not in ranges.columns:
            return None, None
        row = ranges[ranges["feature"].astype(str).eq(str(feature))]
        if row.empty:
            return None, None
        low = pd.to_numeric(row.iloc[0].get("expected_low"), errors="coerce")
        high = pd.to_numeric(row.iloc[0].get("expected_high"), errors="coerce")
        return (float(low) if pd.notna(low) else None, float(high) if pd.notna(high) else None)

    def generate_selected_distribution_plot(self) -> None:
        if self.feature_df is None or not self.output_edit.text().strip():
            return
        feature = self._selected_distribution_feature()
        if not feature:
            return
        try:
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            low, high = self._selected_expected_bounds(feature)
            path = plot_selected_feature_diagnostic(self.feature_df, feature, plots_dir / "selected_feature_distribution.png", low, high, self._selected_distribution_group())
            if not hasattr(self, "plot_paths"):
                self.plot_paths = {}
            self.plot_paths["selected_feature_distribution"] = str(path)
            self.preview_distribution_plot("selected_feature_distribution")
        except Exception:
            pass

    def generate_selected_group_plot(self) -> None:
        if self.feature_df is None or not self.output_edit.text().strip():
            return
        feature = self._selected_distribution_feature()
        if not feature:
            return
        try:
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            path = plot_group_feature_boxplot(self.feature_df, feature, plots_dir / "selected_feature_by_group.png", self._selected_distribution_group())
            if not hasattr(self, "plot_paths"):
                self.plot_paths = {}
            self.plot_paths["selected_feature_by_group"] = str(path)
        except Exception:
            pass

    def preview_distribution_plot(self, key: str) -> None:
        if key == "selected_feature_distribution":
            self.generate_selected_distribution_plot()
        if key == "selected_feature_by_group":
            self.generate_selected_group_plot()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths or not Path(self.plot_paths.get(key, "")).exists():
            self.regenerate_overview_plots()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths:
            QMessageBox.information(self, "Plot unavailable", "Run Feature Analysis first, or this plot was not generated for the current dataset.")
            return
        path = Path(self.plot_paths[key])
        if not path.exists():
            QMessageBox.information(self, "Plot unavailable", f"Plot file not found:\n{path}")
            return
        self.current_distribution_plot = path
        self.update_distribution_interpretation(key)
        pix = QPixmap(str(path))
        if pix.isNull():
            self.dist_plot_preview.setText(f"Could not load plot:\n{path}")
            return
        scaled = pix.scaled(self.dist_plot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.dist_plot_preview.setPixmap(scaled)
        self.dist_plot_preview.setToolTip(str(path))

    def update_distribution_interpretation(self, key: str) -> None:
        if not hasattr(self, "dist_interpretation_label"):
            return
        feature = self._selected_distribution_feature() or "selected feature"
        group = self._selected_distribution_group() or "detected group"
        captions = {
            "distribution_review_status": (
                "<b>What it shows</b><br>Counts of features labelled ok, monitor, or review after combining valid n, missingness, zero variance, robust outlier burden, and expected-range flags.<br><br>"
                "<b>Concerning pattern</b><br>Many features in review means the dataset may contain unstable or poorly supported variables before modelling.<br><br>"
                "<b>Do not overinterpret</b><br>This is not an exclusion rule and not a measure of clinical importance.<br><br>"
                "<b>Next check</b><br>Open Shape audit, Expected ranges, Row-level outliers, and the selected-feature diagnostic for the highest-priority features."
            ),
            "distribution_shape_summary": (
                "<b>What it shows</b><br>Feature-level distribution-shape priority: compact/regular, monitor, or review.<br><br>"
                "<b>Concerning pattern</b><br>Many review features suggest severe skew, heavy tails, floor/ceiling effects, high missingness, or too few valid values.<br><br>"
                "<b>Do not overinterpret</b><br>Speech features are not required to be normally distributed; non-normality alone is not a failure.<br><br>"
                "<b>Next check</b><br>Use the feature diagnostic and raw/QC context before deciding whether to exclude, transform, or retain a feature."
            ),
            "distribution_shape_landscape": (
                "<b>What it shows</b><br>Each feature is positioned by skew proxy and tail heaviness. Review features are labelled when possible.<br><br>"
                "<b>Concerning pattern</b><br>Features far from zero skew or with very high tail ratio may be driven by extreme rows, bounded scales, or task/QC artifacts.<br><br>"
                "<b>Do not overinterpret</b><br>Skewed physiological signals can be real. This plot indicates shape, not validity.<br><br>"
                "<b>Next check</b><br>Inspect selected-feature histogram/ECDF/QQ-style view, then check task, group, QC, and expected range."
            ),
            "expected_range_flags": (
                "<b>What it shows</b><br>Features with values outside supplied expected ranges from the registry/policy table.<br><br>"
                "<b>Concerning pattern</b><br>A high fraction outside range suggests unit mismatch, computation error, task incompatibility, or unusual recordings.<br><br>"
                "<b>Do not overinterpret</b><br>If no registry bounds were supplied, absence of range flags does not prove values are plausible.<br><br>"
                "<b>Next check</b><br>Verify units, computation mode, task type, and row-level flags."
            ),
            "outlier_counts": (
                "<b>What it shows</b><br>Which features have the largest number of row-level robust outlier or expected-range flags.<br><br>"
                "<b>Concerning pattern</b><br>One feature with many flags may be unstable; many features with flags on the same rows may indicate bad recordings/QC issues.<br><br>"
                "<b>Do not overinterpret</b><br>A robust outlier is not automatically an artifact; it can represent real disease severity or task behavior.<br><br>"
                "<b>Next check</b><br>Open Row outlier burden and selected-feature diagnostic."
            ),
            "row_outlier_burden": (
                "<b>What it shows</b><br>Rows/recordings that accumulate many flagged features.<br><br>"
                "<b>Concerning pattern</b><br>Rows with many simultaneous flags often indicate acquisition/QC problems, wrong task, segmentation failure, or metadata mismatch.<br><br>"
                "<b>Do not overinterpret</b><br>Do not delete rows automatically; review audio/QC and clinical context first.<br><br>"
                "<b>Next check</b><br>Open the row-level outlier table and inspect source file, task, subject, and QC metrics."
            ),
            "variance_screen": (
                "<b>What it shows</b><br>Features with zero or near-zero spread based on IQR and unique values.<br><br>"
                "<b>Concerning pattern</b><br>Zero-variance features cannot contribute to modelling and may indicate constants, placeholders, or failed computation.<br><br>"
                "<b>Do not overinterpret</b><br>Low variance can be expected in homogeneous samples; it is still weak for prediction in this dataset.<br><br>"
                "<b>Next check</b><br>Confirm feature definition and whether this feature should be exported or marked review/exclude."
            ),
            "selected_feature_distribution": (
                f"<b>What it shows</b><br>A four-panel diagnostic for <b>{feature}</b>: histogram, box/strip view, empirical CDF, and QQ-style normality check.<br><br>"
                "<b>Concerning pattern</b><br>Isolated extreme points, strong skew, flat/step-like ECDF, or divergence from the QQ reference line may require review.<br><br>"
                "<b>Do not overinterpret</b><br>Normality is not required. This view is for understanding shape and plausible preprocessing needs.<br><br>"
                "<b>Next check</b><br>Compare by group/task, inspect outlier rows, and review expected ranges/QC."
            ),
            "selected_feature_by_group": (
                f"<b>What it shows</b><br>Distribution of <b>{feature}</b> across <b>{group}</b> using box/strip display.<br><br>"
                "<b>Concerning pattern</b><br>Group separation accompanied by unequal spread, very small n, or group-specific outliers may reflect bias or real signal.<br><br>"
                "<b>Do not overinterpret</b><br>This is descriptive, not a hypothesis test and not an ML result.<br><br>"
                "<b>Next check</b><br>Check group sample sizes, task balance, QC burden, and later univariate screening."
            ),
            "feature_distribution_grid": (
                "<b>What it shows</b><br>A compact grid of the highest-priority features from the shape audit, not every feature.<br><br>"
                "<b>Concerning pattern</b><br>Degenerate, highly skewed, or multi-peaked panels indicate features needing closer inspection.<br><br>"
                "<b>Do not overinterpret</b><br>The grid is a triage board. Use selected-feature diagnostics for decisions.<br><br>"
                "<b>Next check</b><br>Select one suspicious feature from the dropdown and inspect it in detail."
            ),
        }
        self.dist_interpretation_label.setText(captions.get(key, "Select a plot to see structured interpretation guidance."))

    def open_current_distribution_plot(self) -> None:
        path = getattr(self, "current_distribution_plot", None)
        if not path:
            QMessageBox.information(self, "No current plot", "Preview a plot first, then use this button to open the full-resolution file.")
            return
        self.open_file(Path(path))


    def _simple_table(self) -> QTableWidget:
        table = QTableWidget(0, 0)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        return table

    def _qc_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        card = Card(
            "QC Integration",
            "Artifact-aware interpretation layer. This screen asks whether feature values, missingness, or outliers may be explained by acquisition quality rather than physiology."
        )
        self.qc_metric_grid = QGridLayout()
        self.qc_metric_grid.setHorizontalSpacing(12)
        self.qc_metric_grid.setVerticalSpacing(12)
        card.layout.addLayout(self.qc_metric_grid)

        self.qc_note = QLabel(
            "Load an optional QC table in Project, accept Column Mapping, and run Feature Analysis. QC is interpreted as a multidimensional profile: additive interference, gain/level dynamics, reverberation/echo, channel/device/platform, nonlinear distortion, and temporal discontinuities."
        )
        self.qc_note.setWordWrap(True)
        self.qc_note.setStyleSheet(f"color:{INK}; background:#F7FAFD; border:1px solid {LINE}; border-radius:10px; padding:10px;")
        card.layout.addWidget(self.qc_note)

        tabs = QTabWidget()
        self.qc_summary_table = self._simple_table()
        self.qc_catalog_table = self._simple_table()
        self.qc_family_table = self._simple_table()
        self.qc_assoc_table = self._simple_table()
        self.qc_family_assoc_table = self._simple_table()
        self.qc_missing_table = self._simple_table()
        self.qc_outlier_table = self._simple_table()
        self.qc_row_table = self._simple_table()
        tabs.addTab(self.qc_summary_table, "Summary")
        tabs.addTab(self.qc_catalog_table, "QC metrics")
        tabs.addTab(self.qc_family_table, "Family burden")
        tabs.addTab(self.qc_assoc_table, "Feature × QC")
        tabs.addTab(self.qc_family_assoc_table, "Feature × family")
        tabs.addTab(self.qc_missing_table, "Missingness × QC")
        tabs.addTab(self.qc_outlier_table, "Outliers × QC")
        tabs.addTab(self.qc_row_table, "Row QC burden")
        card.layout.addWidget(tabs)
        layout.addWidget(card)

        plot_card = Card(
            "QC plot board",
            "Use these plots to decide whether acoustic feature behavior may be acquisition-sensitive. Every plot is descriptive and should be interpreted with task, clinical, segmentation, and raw-audio context."
        )
        controls = QHBoxLayout()
        self.qc_feature_combo = QComboBox()
        self.qc_feature_combo.setMinimumWidth(270)
        self.qc_metric_combo = QComboBox()
        self.qc_metric_combo.setMinimumWidth(270)
        controls.addWidget(QLabel("Feature:")); controls.addWidget(self.qc_feature_combo)
        controls.addSpacing(12)
        controls.addWidget(QLabel("QC metric:")); controls.addWidget(self.qc_metric_combo)
        controls.addStretch(1)
        plot_card.layout.addLayout(controls)

        buttons = QHBoxLayout()
        for label, key in [
            ("QC framework", "qc_artifact_model"),
            ("Family burden", "qc_family_burden"),
            ("QC metric distributions", "qc_metric_distributions"),
            ("Feature × QC-family heatmap", "qc_feature_association_heatmap"),
            ("Top feature-QC associations", "qc_top_feature_associations"),
            ("Missingness linked to QC", "qc_missingness_associations"),
            ("Row QC burden", "qc_row_burden"),
            ("Selected feature × selected QC", "selected_feature_qc_scatter"),
        ]:
            b = QPushButton(label)
            b.setProperty("secondary", True)
            b.clicked.connect(lambda _=False, k=key: self.preview_qc_plot(k))
            buttons.addWidget(b)
        buttons.addStretch(1)
        plot_card.layout.addLayout(buttons)

        splitter = QSplitter(Qt.Horizontal)
        self.qc_plot_preview = QLabel("Run Feature Analysis, then select a QC plot.")
        self.qc_plot_preview.setAlignment(Qt.AlignCenter)
        self.qc_plot_preview.setMinimumHeight(520)
        self.qc_plot_preview.setStyleSheet(f"background:#FFFFFF; color:{MUTED}; border:1px solid {LINE}; border-radius:10px;")
        self.qc_plot_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.qc_interpretation_label = QLabel("Select a QC plot to see structured interpretation guidance.")
        self.qc_interpretation_label.setWordWrap(True)
        self.qc_interpretation_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.qc_interpretation_label.setMinimumWidth(340)
        self.qc_interpretation_label.setStyleSheet(f"background:#F7FAFD; color:{INK}; border:1px solid {LINE}; border-radius:10px; padding:14px; line-height:145%;")
        splitter.addWidget(self.qc_plot_preview)
        splitter.addWidget(self.qc_interpretation_label)
        splitter.setSizes([860, 360])
        plot_card.layout.addWidget(splitter)

        open_btn = QPushButton("Open current plot full size")
        open_btn.setProperty("secondary", True)
        open_btn.clicked.connect(self.open_current_qc_plot)
        plot_card.layout.addWidget(open_btn)
        layout.addWidget(plot_card)
        return self._wrap_scroll(body)

    def update_qc_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "qc_metric_grid"):
            return
        while self.qc_metric_grid.count():
            item = self.qc_metric_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        summary = outputs.get("qc_integration_summary", pd.DataFrame())
        def metric_value(name, default="—"):
            if summary is None or summary.empty or "metric" not in summary.columns:
                return default
            hit = summary[summary["metric"].astype(str).eq(name)]
            return hit["value"].iloc[0] if not hit.empty else default
        tiles = [
            ("QC table", "yes" if str(metric_value("qc_table_loaded", False)).lower() == "true" else "no", "artifact context supplied"),
            ("QC rows", metric_value("qc_rows"), "recordings with QC data"),
            ("QC metrics", metric_value("numeric_qc_metrics"), "numeric artifact indicators"),
            ("Families", metric_value("artifact_families_detected"), "recognized QC domains"),
            ("Feature-QC pairs ≥ .30", metric_value("feature_qc_pairs_abs_rho_ge_0_30", 0), "monitor associations"),
            ("Feature-QC pairs ≥ .50", metric_value("feature_qc_pairs_abs_rho_ge_0_50", 0), "review associations"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.qc_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx // 3, idx % 3)

        self._fill_table(self.qc_summary_table, summary)
        self._fill_table(self.qc_catalog_table, outputs.get("qc_metric_catalog", pd.DataFrame()))
        self._fill_table(self.qc_family_table, outputs.get("qc_family_burden_summary", pd.DataFrame()))
        self._fill_table(self.qc_assoc_table, outputs.get("feature_qc_spearman_correlation", pd.DataFrame()))
        self._fill_table(self.qc_family_assoc_table, outputs.get("feature_qc_family_association", pd.DataFrame()))
        self._fill_table(self.qc_missing_table, outputs.get("qc_missingness_associations", pd.DataFrame()))
        self._fill_table(self.qc_outlier_table, outputs.get("qc_outlier_associations", pd.DataFrame()))
        self._fill_table(self.qc_row_table, outputs.get("qc_row_burden_summary", pd.DataFrame()))

        self.qc_feature_combo.clear()
        dist = outputs.get("feature_distribution_summary", pd.DataFrame())
        if dist is not None and not dist.empty and "feature" in dist.columns:
            self.qc_feature_combo.addItems(dist["feature"].astype(str).tolist())
        self.qc_metric_combo.clear()
        catalog = outputs.get("qc_metric_catalog", pd.DataFrame())
        if catalog is not None and not catalog.empty and "qc_variable" in catalog.columns:
            self.qc_metric_combo.addItems(catalog["qc_variable"].astype(str).tolist())
        if self.qc_df is None or self.qc_df.empty:
            self.qc_note.setText("No QC table is loaded. This menu will become active when Project includes an optional QC table with qadd/qgain/qrev/qchan/qdist/qtemp/qdrop metrics or equivalent QC indicators.")
        else:
            self.qc_note.setText("QC integration generated. Use this menu to distinguish acquisition-sensitive feature behavior from plausible speech-physiology variation. QC associations are descriptive screening signals, not automatic exclusion rules.")

    def _selected_qc_feature(self) -> str | None:
        if hasattr(self, "qc_feature_combo") and self.qc_feature_combo.count() > 0:
            return self.qc_feature_combo.currentText()
        return None

    def _selected_qc_metric(self) -> str | None:
        if hasattr(self, "qc_metric_combo") and self.qc_metric_combo.count() > 0:
            return self.qc_metric_combo.currentText()
        return None

    def generate_selected_qc_scatter(self) -> None:
        feature = self._selected_qc_feature()
        metric = self._selected_qc_metric()
        if not feature or not metric:
            return
        try:
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            path = plot_selected_feature_qc_scatter(self.feature_df, self.qc_df, feature, metric, plots_dir / "selected_feature_qc_scatter.png")
            if not hasattr(self, "plot_paths"):
                self.plot_paths = {}
            self.plot_paths["selected_feature_qc_scatter"] = str(path)
        except Exception:
            pass

    def preview_qc_plot(self, key: str) -> None:
        if key == "selected_feature_qc_scatter":
            self.generate_selected_qc_scatter()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths or not Path(self.plot_paths.get(key, "")).exists():
            self.regenerate_overview_plots()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths:
            QMessageBox.information(self, "Plot unavailable", "Run Feature Analysis first, or this QC plot could not be generated for the current dataset.")
            return
        path = Path(self.plot_paths[key])
        if not path.exists():
            QMessageBox.information(self, "Plot unavailable", f"Plot file not found:\n{path}")
            return
        self.current_qc_plot = path
        self.update_qc_interpretation(key)
        pix = QPixmap(str(path))
        if pix.isNull():
            self.qc_plot_preview.setText(f"Could not load plot:\n{path}")
            return
        self.qc_plot_preview.setPixmap(pix.scaled(self.qc_plot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.qc_plot_preview.setToolTip(str(path))

    def update_qc_interpretation(self, key: str) -> None:
        if not hasattr(self, "qc_interpretation_label"):
            return
        feature = self._selected_qc_feature() or "selected feature"
        qcvar = self._selected_qc_metric() or "selected QC metric"
        captions = {
            "qc_artifact_model": (
                "<b>What it shows</b><br>The conceptual QC model: recording quality is a vector of artifact families, not a single good/bad score.<br><br>"
                "<b>Concerning pattern</b><br>Any family can perturb features differently. A recording can be acceptable overall but still artifact-sensitive for specific feature families.<br><br>"
                "<b>Do not overinterpret</b><br>QC families are proxy estimates of latent acquisition processes, not direct physical measurements of the room/device.<br><br>"
                "<b>Next check</b><br>Inspect family burden, feature-QC associations, and selected feature × QC scatter plots."
            ),
            "qc_family_burden": (
                "<b>What it shows</b><br>Which artifact families have elevated QC metrics across recordings.<br><br>"
                "<b>Concerning pattern</b><br>High burden in additive, gain, channel, reverberation, distortion, or temporal families indicates acquisition effects that may bias feature interpretation.<br><br>"
                "<b>Do not overinterpret</b><br>High family burden does not automatically mean recordings are unusable; it means downstream features need artifact-aware interpretation.<br><br>"
                "<b>Next check</b><br>Use Feature × QC-family heatmap to see which acoustic features are sensitive to that family."
            ),
            "qc_metric_distributions": (
                "<b>What it shows</b><br>Distributions of the highest-spread QC metrics, grouped by artifact family.<br><br>"
                "<b>Concerning pattern</b><br>Heavy tails, spikes at zero, or extreme values can indicate rare but important artifacts such as clipping or dropouts.<br><br>"
                "<b>Do not overinterpret</b><br>Different QC metrics have different units and supports; compare shape and burden, not raw magnitude across metrics.<br><br>"
                "<b>Next check</b><br>Review the QC metric catalog and row-level QC burden table."
            ),
            "qc_feature_association_heatmap": (
                "<b>What it shows</b><br>Maximum absolute monotonic association between each feature and each QC artifact family.<br><br>"
                "<b>Concerning pattern</b><br>A block of high associations for one feature family suggests acquisition-sensitive measurements.<br><br>"
                "<b>Do not overinterpret</b><br>Correlation does not prove artifact causation; QC may also be linked to task, severity, device, or cohort.<br><br>"
                "<b>Next check</b><br>Inspect top feature-QC associations and selected feature × selected QC scatter."
            ),
            "qc_top_feature_associations": (
                "<b>What it shows</b><br>The strongest individual feature-QC metric associations ranked by |Spearman rho|.<br><br>"
                "<b>Concerning pattern</b><br>|rho| ≥ .30 warrants monitoring; |rho| ≥ .50 should be reviewed before using the feature in ML or clinical interpretation.<br><br>"
                "<b>Do not overinterpret</b><br>A strong association can represent true task/acquisition structure, not necessarily bad data.<br><br>"
                "<b>Next check</b><br>Check whether the association remains after stratifying by task/group and after reviewing row-level QC."
            ),
            "qc_missingness_associations": (
                "<b>What it shows</b><br>Whether feature absence is associated with QC burden.<br><br>"
                "<b>Concerning pattern</b><br>If missingness increases with QC artifacts, the missing-data mechanism may be acquisition-dependent rather than random.<br><br>"
                "<b>Do not overinterpret</b><br>Absence of association does not prove missingness is random; sample size and alignment matter.<br><br>"
                "<b>Next check</b><br>Compare with Missingness by group/task and Feature availability heatmap."
            ),
            "qc_row_burden": (
                "<b>What it shows</b><br>Recordings with the largest number of elevated QC metrics.<br><br>"
                "<b>Concerning pattern</b><br>Rows with multiple elevated QC families may be acquisition-dominated and should be reviewed before row exclusion or model training.<br><br>"
                "<b>Do not overinterpret</b><br>Do not delete rows automatically; severe ALS speech may also produce unusual signal patterns.<br><br>"
                "<b>Next check</b><br>Open the row QC burden table and inspect raw audio, segmentation, task, and clinical context."
            ),
            "selected_feature_qc_scatter": (
                f"<b>What it shows</b><br>Recording-level relationship between <b>{feature}</b> and <b>{qcvar}</b>.<br><br>"
                "<b>Concerning pattern</b><br>Monotonic trends, funnel shapes, or clusters suggest that feature values may depend on acquisition quality.<br><br>"
                "<b>Do not overinterpret</b><br>This is descriptive, not a causal model. Check task, diagnosis, severity, device, and subject repetition.<br><br>"
                "<b>Next check</b><br>If clinically important, plan QC sensitivity analyses or use QC covariates in the later ML/statistical pipeline."
            ),
        }
        self.qc_interpretation_label.setText(captions.get(key, "Select a QC plot to see structured interpretation guidance."))

    def open_current_qc_plot(self) -> None:
        path = getattr(self, "current_qc_plot", None)
        if not path:
            QMessageBox.information(self, "No current plot", "Preview a QC plot first, then open the full-resolution file.")
            return
        self.open_file(Path(path))

    def _table_page(self, title: str, subtitle: str) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        card = Card(title, subtitle)
        table = QTableWidget(0, 0)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setObjectName("".join(ch.lower() if ch.isalnum() else "_" for ch in title).strip("_"))
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
            self.log("Proposed role counts:\n" + roles.to_string(index=False))
            self.log("Review Column Mapping, then click Accept Mapping and Continue or manually revise roles.")
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
        self.update_mapping_summary()

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
        self.update_mapping_summary()
        return df

    def set_selected_role(self, role: str) -> None:
        rows = sorted({idx.row() for idx in self.mapping_table.selectedIndexes()})
        if not rows:
            QMessageBox.information(self, "No rows selected", "Select one or more rows in the mapping table first.")
            return
        for row in rows:
            widget = self.mapping_table.cellWidget(row, 1)
            if isinstance(widget, QComboBox):
                widget.setCurrentText(role)
        self.collect_mapping_from_table()
        self.update_mapping_summary()

    def reload_proposed_mapping(self) -> None:
        if self.feature_df is None:
            QMessageBox.information(self, "No table loaded", "Load a primary feature table first.")
            return
        self.mapping_df = classify_columns(self.feature_df, table_kind="feature", registry=self.registry_df)
        self.refresh_mapping_table()

    def accept_mapping_and_continue(self) -> None:
        self.collect_mapping_from_table()
        self.update_mapping_summary()
        roles = role_lists(self.mapping_df)
        n_features = len(roles.get(ROLE_FEATURE, []))
        if n_features == 0:
            QMessageBox.warning(self, "No feature columns selected", "At least one column must be assigned the role 'Feature' before analysis.")
            return
        self.log(f"Column mapping accepted: {n_features} feature columns selected.")
        self.show_page("overview")

    def update_mapping_summary(self) -> None:
        if self.mapping_df.empty or not hasattr(self, "mapping_summary_label"):
            return
        summary = summarize_roles(self.mapping_df)
        parts = [f"{r['role']}: {int(r['n_columns'])}" for _, r in summary.iterrows()]
        self.mapping_summary_label.setText("Current mapping · " + " | ".join(parts))

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
            plots_dir = self.output_dir / "plots"
            tables_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)
            plots_dir.mkdir(parents=True, exist_ok=True)

            outputs, feature_cols = self._build_analysis_outputs()
            self.outputs = outputs
            self._write_outputs(outputs, tables_dir)
            self.plot_paths = self._generate_overview_plots(outputs, feature_cols, plots_dir)

            self.write_report(reports_dir / "vslp_feature_analysis_report.html", outputs)
            self.populate_output_tables(outputs)
            self.update_overview_dashboard(outputs)
            self.update_missingness_dashboard(outputs)
            self.update_distribution_dashboard(outputs)
            self.update_qc_dashboard(outputs)
            self.log(f"Analysis complete. Outputs written to: {self.output_dir}")
            self.show_page("overview")
        except Exception as exc:
            QMessageBox.critical(self, "Analysis failed", str(exc))

    def populate_output_tables(self, outputs: dict[str, pd.DataFrame]) -> None:
        targets = {
            "overview": "dataset_inventory",
            "missing": "missingness_by_feature",
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

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        super().resizeEvent(event)
        path = getattr(self, "current_overview_plot", None)
        if path and hasattr(self, "overview_plot_preview"):
            self._show_overview_plot(Path(path))
        mpath = getattr(self, "current_missingness_plot", None)
        if mpath and hasattr(self, "missing_plot_preview"):
            pix = QPixmap(str(mpath))
            if not pix.isNull():
                self.missing_plot_preview.setPixmap(pix.scaled(self.missing_plot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        dpath = getattr(self, "current_distribution_plot", None)
        if dpath and hasattr(self, "dist_plot_preview"):
            pix = QPixmap(str(dpath))
            if not pix.isNull():
                self.dist_plot_preview.setPixmap(pix.scaled(self.dist_plot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        qpath = getattr(self, "current_qc_plot", None)
        if qpath and hasattr(self, "qc_plot_preview"):
            pix = QPixmap(str(qpath))
            if not pix.isNull():
                self.qc_plot_preview.setPixmap(pix.scaled(self.qc_plot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def write_report(self, path: Path, outputs: dict[str, pd.DataFrame]) -> None:
        inv = outputs.get("dataset_inventory", pd.DataFrame()).to_html(index=False, escape=False)
        roles = outputs.get("feature_role_summary", summarize_roles(outputs.get("feature_column_mapping", pd.DataFrame()))).to_html(index=False, escape=False)
        design = outputs.get("dataset_design_overview", pd.DataFrame()).to_html(index=False, escape=False)
        fam = outputs.get("feature_family_overview", pd.DataFrame()).to_html(index=False, escape=False)
        miss = outputs.get("missingness_by_feature", pd.DataFrame()).head(80).to_html(index=False, escape=False)
        missgrp = outputs.get("missingness_by_group", pd.DataFrame()).head(80).to_html(index=False, escape=False)
        distreview = outputs.get("distribution_review_summary", pd.DataFrame()).head(100).to_html(index=False, escape=False)
        outrows = outputs.get("robust_outlier_flags", pd.DataFrame()).head(100).to_html(index=False, escape=False)
        rel = outputs.get("feature_reliability_screen", pd.DataFrame()).head(80).to_html(index=False, escape=False)
        html = f"""<!doctype html><html><head><meta charset='utf-8'><title>VSLP Feature Analysis Report</title>
        <style>body{{font-family:Arial,sans-serif;margin:32px;color:#0E1726}} h1,h2{{color:#071A33}} table{{border-collapse:collapse;width:100%;font-size:12px;margin-bottom:24px}} td,th{{border:1px solid #D9E2EF;padding:6px}} th{{background:#EEF4FA}}</style></head><body>
        <h1>VSLP Feature Analysis Report</h1><p>Version {APP_VERSION}. Descriptive feature audit only; not a diagnostic or ML-training report.</p>
        <h2>Dataset inventory</h2>{inv}
        <h2>Column-role summary</h2>{roles}
        <h2>Dataset design overview</h2>{design}
        <h2>Feature-family overview</h2>{fam}
        <h2>Missingness by feature</h2>{miss}
        <h2>Missingness by group</h2>{missgrp}
        <h2>Distribution / outlier review</h2>{distreview}
        <h2>Row-level outlier flags</h2>{outrows}
        <h2>Initial feature reliability screen</h2>{rel}</body></html>"""
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
