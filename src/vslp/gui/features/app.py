"""VSLP Feature Analysis GUI.

Professional, modality-neutral feature audit interface for acoustic, kinematic,
multimodal, and generic feature tables.
"""
from __future__ import annotations

import sys
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from PySide6.QtCore import Qt, QSize, QUrl
    from PySide6.QtGui import QPixmap, QFont, QDesktopServices, QColor, QBrush
    from PySide6.QtWidgets import (
        QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QGroupBox,
        QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
        QPushButton, QSizePolicy, QStackedWidget, QTableWidget, QTableWidgetItem,
        QTextEdit, QVBoxLayout, QWidget, QSplitter, QScrollArea, QAbstractItemView,
        QTabWidget, QProgressBar
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
    qc_outlier_associations, qc_integration_summary,
    feature_relationship_summary, feature_correlation_long_table, redundant_feature_pairs,
    feature_relationship_modules, feature_family_correlation_matrix,
    feature_pca_summary, feature_pca_loadings, feature_pca_scores,
    build_group_outcome_screening,
    reliability_design_summary, reliability_subject_record_counts,
    feature_repeatability_summary, reliability_family_summary,
    feature_recommendation_table, feature_recommendation_summary,
    feature_recommendation_reason_counts, feature_recommendation_family_summary
)
from vslp.analysis.features.ml_export_builder import build_ml_export_package
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
    plot_qc_row_burden, plot_selected_feature_qc_scatter, plot_qc_artifact_model,
    plot_relationship_correlation_heatmap, plot_relationship_redundant_pairs,
    plot_relationship_family_matrix, plot_relationship_pca_scree,
    plot_relationship_pca_scores, plot_relationship_pca_loadings,
    plot_selected_feature_correlations,
    plot_screening_group_balance, plot_screening_effect_ranking,
    plot_screening_continuous_heatmap, plot_screening_group_heatmap,
    plot_screening_effect_landscape, plot_selected_feature_outcome,
    plot_reliability_status_counts, plot_reliability_icc_ranking,
    plot_reliability_variance_landscape, plot_reliability_family_summary,
    plot_reliability_subject_counts, plot_selected_feature_reliability,
    plot_recommendation_counts, plot_recommendation_score_landscape,
    plot_recommendation_reason_counts, plot_recommendation_family_summary,
    plot_ml_export_manifest_summary
)

APP_VERSION = "v0.58.0"

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

    QMessageBox, QMessageBox QLabel, QMessageBox QPushButton {{
        background: #FFFFFF;
        color: {INK};
    }}
    QMessageBox {{
        background: #FFFFFF;
        color: {INK};
    }}
    QMessageBox QLabel {{
        background: #FFFFFF;
        color: {INK};
        font-size: 13px;
        padding: 4px;
    }}
    QMessageBox QPushButton {{
        background: #FFFFFF;
        color: {NAVY};
        border: 1px solid {LINE};
        border-radius: 8px;
        padding: 7px 14px;
        min-width: 78px;
        font-weight: 700;
    }}
    QMessageBox QPushButton:hover {{
        background: #F3FAF9;
        border-color: {TEAL};
        color: {NAVY};
    }}
    QMessageBox QPushButton:pressed {{
        background: #DDF6F4;
        color: {NAVY};
    }}
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{
        background: #EEF4FA;
        width: 14px;
        margin: 2px;
        border-radius: 7px;
    }}
    QScrollBar::handle:vertical {{
        background: #B8C9DC;
        min-height: 36px;
        border-radius: 7px;
    }}
    QScrollBar::handle:vertical:hover {{ background: #8EA7C2; }}
    QScrollBar:horizontal {{
        background: #EEF4FA;
        height: 14px;
        margin: 2px;
        border-radius: 7px;
    }}
    QScrollBar::handle:horizontal {{
        background: #B8C9DC;
        min-width: 36px;
        border-radius: 7px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: #8EA7C2; }}
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
            ("relationships", "○  Feature Relationships"),
            ("screening", "○  Group / Outcome Screening"),
            ("reliability", "○  Reliability"),
            ("recommendations", "○  Recommendations"),
            ("ml_export", "○  ML Export Builder"),
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
        self.page_keys = ["project", "mapping", "overview", "missing", "dist", "qc", "relationships", "screening", "reliability", "recommendations", "ml_export", "export"]

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(LogoBar())

        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setChildrenCollapsible(False)
        outer.addWidget(main_splitter, 1)

        content_widget = QWidget()
        content = QHBoxLayout(content_widget)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)

        self.sidebar = Sidebar(self.show_page)
        content.addWidget(self.sidebar)
        self.stack = QStackedWidget()
        content.addWidget(self.stack, 1)
        main_splitter.addWidget(content_widget)

        self.run_log_panel = self._build_run_log_panel()
        main_splitter.addWidget(self.run_log_panel)
        main_splitter.setStretchFactor(0, 8)
        main_splitter.setStretchFactor(1, 2)
        main_splitter.setSizes([700, 170])

        self.pages = {
            "project": self._project_page(),
            "mapping": self._mapping_page(),
            "overview": self._overview_page(),
            "missing": self._missingness_page(),
            "dist": self._distributions_page(),
            "qc": self._qc_page(),
            "relationships": self._relationships_page(),
            "screening": self._screening_page(),
            "reliability": self._reliability_page(),
            "recommendations": self._recommendations_page(),
            "ml_export": self._ml_export_builder_page(),
            "export": self._export_page(),
        }
        for key in self.page_keys:
            self.stack.addWidget(self.pages[key])
        self.show_page("project")

    def show_page(self, key: str) -> None:
        self.stack.setCurrentIndex(self.page_keys.index(key))
        self.sidebar.set_active(key)

    def _build_run_log_panel(self) -> QFrame:
        panel = QFrame()
        panel.setStyleSheet(f"""
        QFrame {{ background: #FFFFFF; border-top: 1px solid {LINE}; }}
        QLabel#RunLogTitle {{ color: {NAVY}; font-size: 13px; font-weight: 800; border: none; }}
        QLabel#RunLogHint {{ color: {MUTED}; font-size: 11px; border: none; }}
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 8, 14, 10)
        layout.setSpacing(6)
        header = QHBoxLayout()
        title = QLabel(f"Run Log · Feature Analysis GUI {APP_VERSION}")
        title.setObjectName("RunLogTitle")
        hint = QLabel("Persistent messages for loading, mapping, analysis, export, and errors")
        hint.setObjectName("RunLogHint")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(hint)
        layout.addLayout(header)
        self.run_log = QTextEdit()
        self.run_log.setReadOnly(True)
        self.run_log.setMinimumHeight(96)
        self.run_log.setPlaceholderText("Run messages will appear here.")
        layout.addWidget(self.run_log)
        self.run_progress = QProgressBar()
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(0)
        self.run_progress.setTextVisible(True)
        layout.addWidget(self.run_progress)
        return panel

    def _wrap_scroll(self, widget: QWidget) -> QScrollArea:
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(widget)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        widget.setMinimumWidth(980)
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
        rel_summary = feature_relationship_summary(self.feature_df, feature_cols, self.registry_df)
        rel_corr_long = feature_correlation_long_table(self.feature_df, feature_cols)
        rel_redundant = redundant_feature_pairs(rel_corr_long, self.registry_df)
        rel_modules = feature_relationship_modules(rel_corr_long, self.registry_df)
        rel_family_matrix = feature_family_correlation_matrix(rel_corr_long, self.registry_df)
        rel_pca_summary = feature_pca_summary(self.feature_df, feature_cols)
        rel_pca_loadings = feature_pca_loadings(self.feature_df, feature_cols, self.registry_df)
        rel_pca_scores = feature_pca_scores(self.feature_df, feature_cols)
        screening = build_group_outcome_screening(self.feature_df, feature_cols, mapping)
        reliability = reliability_screen(dist, qc_corr)
        reliability_design = reliability_design_summary(self.feature_df, feature_cols, mapping)
        reliability_subjects = reliability_subject_record_counts(self.feature_df, mapping)
        reliability_repeatability = feature_repeatability_summary(self.feature_df, feature_cols, mapping, self.registry_df)
        reliability_family = reliability_family_summary(reliability_repeatability)
        recommendation_inputs = {
            "screening_continuous_outcome_associations": screening.get("screening_continuous_outcome_associations", pd.DataFrame()),
            "screening_categorical_group_associations": screening.get("screening_categorical_group_associations", pd.DataFrame()),
        }
        feature_recs = feature_recommendation_table(
            dist, feature_missing, shape_audit, qc_corr, rel_redundant, reliability_repeatability, recommendation_inputs, self.registry_df
        )
        rec_summary = feature_recommendation_summary(feature_recs)
        rec_reasons = feature_recommendation_reason_counts(feature_recs)
        rec_family = feature_recommendation_family_summary(feature_recs)
        ml_export_manifest = feature_recs[[c for c in [
            "feature", "family_or_subsystem", "readiness_recommendation", "readiness_score",
            "ml_export_default", "primary_reasons", "recommended_action"
        ] if c in feature_recs.columns]].copy()
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
            "feature_relationship_summary": rel_summary,
            "feature_correlation_long": rel_corr_long,
            "feature_redundant_pairs": rel_redundant,
            "feature_relationship_modules": rel_modules,
            "feature_family_correlation_matrix": rel_family_matrix,
            "feature_pca_summary": rel_pca_summary,
            "feature_pca_loadings": rel_pca_loadings,
            "feature_pca_scores": rel_pca_scores,
            "screening_summary": screening.get("screening_summary", pd.DataFrame()),
            "screening_variable_catalog": screening.get("screening_variable_catalog", pd.DataFrame()),
            "screening_continuous_outcome_associations": screening.get("screening_continuous_outcome_associations", pd.DataFrame()),
            "screening_categorical_group_associations": screening.get("screening_categorical_group_associations", pd.DataFrame()),
            "screening_group_balance": screening.get("screening_group_balance", pd.DataFrame()),
            "feature_reliability_screen": reliability,
            "reliability_design_summary": reliability_design,
            "reliability_subject_record_counts": reliability_subjects,
            "feature_repeatability_summary": reliability_repeatability,
            "reliability_family_summary": reliability_family,
            "feature_recommendations": feature_recs,
            "feature_recommendation_summary": rec_summary,
            "feature_recommendation_reason_counts": rec_reasons,
            "feature_recommendation_family_summary": rec_family,
            "ml_export_manifest": ml_export_manifest,
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
        paths["relationship_correlation_heatmap"] = str(plot_relationship_correlation_heatmap(outputs.get("feature_correlation_long", pd.DataFrame()), plots_dir / "relationship_correlation_heatmap.png"))
        paths["relationship_redundant_pairs"] = str(plot_relationship_redundant_pairs(outputs.get("feature_redundant_pairs", pd.DataFrame()), plots_dir / "relationship_redundant_pairs.png"))
        paths["relationship_family_matrix"] = str(plot_relationship_family_matrix(outputs.get("feature_family_correlation_matrix", pd.DataFrame()), plots_dir / "relationship_family_matrix.png"))
        paths["relationship_pca_scree"] = str(plot_relationship_pca_scree(outputs.get("feature_pca_summary", pd.DataFrame()), plots_dir / "relationship_pca_scree.png"))
        paths["relationship_pca_scores"] = str(plot_relationship_pca_scores(outputs.get("feature_pca_scores", pd.DataFrame()), self.feature_df, plots_dir / "relationship_pca_scores.png"))
        paths["relationship_pca_loadings"] = str(plot_relationship_pca_loadings(outputs.get("feature_pca_loadings", pd.DataFrame()), plots_dir / "relationship_pca_loadings.png"))
        if first_feature:
            paths["selected_feature_correlations"] = str(plot_selected_feature_correlations(outputs.get("feature_correlation_long", pd.DataFrame()), first_feature, plots_dir / "selected_feature_correlations.png"))
        paths["screening_group_balance"] = str(plot_screening_group_balance(outputs.get("screening_group_balance", pd.DataFrame()), plots_dir / "screening_group_balance.png"))
        paths["screening_effect_ranking"] = str(plot_screening_effect_ranking(outputs.get("screening_continuous_outcome_associations", pd.DataFrame()), outputs.get("screening_categorical_group_associations", pd.DataFrame()), plots_dir / "screening_effect_ranking.png"))
        paths["screening_continuous_heatmap"] = str(plot_screening_continuous_heatmap(outputs.get("screening_continuous_outcome_associations", pd.DataFrame()), plots_dir / "screening_continuous_heatmap.png"))
        paths["screening_group_heatmap"] = str(plot_screening_group_heatmap(outputs.get("screening_categorical_group_associations", pd.DataFrame()), plots_dir / "screening_group_heatmap.png"))
        paths["screening_effect_landscape"] = str(plot_screening_effect_landscape(outputs.get("screening_continuous_outcome_associations", pd.DataFrame()), outputs.get("screening_categorical_group_associations", pd.DataFrame()), plots_dir / "screening_effect_landscape.png"))
        screen_vars = outputs.get("screening_variable_catalog", pd.DataFrame())
        screen_var_list = screen_vars.get("variable", pd.Series(dtype=str)).astype(str).tolist() if screen_vars is not None and not screen_vars.empty else []
        if first_feature and screen_var_list:
            paths["selected_feature_outcome"] = str(plot_selected_feature_outcome(self.feature_df, first_feature, screen_var_list[0], plots_dir / "selected_feature_outcome.png"))
        paths["reliability_status_counts"] = str(plot_reliability_status_counts(outputs.get("feature_repeatability_summary", pd.DataFrame()), plots_dir / "reliability_status_counts.png"))
        paths["reliability_icc_ranking"] = str(plot_reliability_icc_ranking(outputs.get("feature_repeatability_summary", pd.DataFrame()), plots_dir / "reliability_icc_ranking.png"))
        paths["reliability_variance_landscape"] = str(plot_reliability_variance_landscape(outputs.get("feature_repeatability_summary", pd.DataFrame()), plots_dir / "reliability_variance_landscape.png"))
        paths["reliability_family_summary"] = str(plot_reliability_family_summary(outputs.get("reliability_family_summary", pd.DataFrame()), plots_dir / "reliability_family_summary.png"))
        paths["reliability_subject_counts"] = str(plot_reliability_subject_counts(outputs.get("reliability_subject_record_counts", pd.DataFrame()), plots_dir / "reliability_subject_counts.png"))
        if first_feature:
            paths["selected_feature_reliability"] = str(plot_selected_feature_reliability(self.feature_df, first_feature, plots_dir / "selected_feature_reliability.png"))
        paths["recommendation_counts"] = str(plot_recommendation_counts(outputs.get("feature_recommendations", pd.DataFrame()), plots_dir / "recommendation_counts.png"))
        paths["recommendation_score_landscape"] = str(plot_recommendation_score_landscape(outputs.get("feature_recommendations", pd.DataFrame()), plots_dir / "recommendation_score_landscape.png"))
        paths["recommendation_reason_counts"] = str(plot_recommendation_reason_counts(outputs.get("feature_recommendation_reason_counts", pd.DataFrame()), plots_dir / "recommendation_reason_counts.png"))
        paths["recommendation_family_summary"] = str(plot_recommendation_family_summary(outputs.get("feature_recommendation_family_summary", pd.DataFrame()), plots_dir / "recommendation_family_summary.png"))
        paths["ml_export_manifest_summary"] = str(plot_ml_export_manifest_summary(outputs.get("ml_export_manifest", pd.DataFrame()), plots_dir / "ml_export_manifest_summary.png"))
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
            self.update_relationships_dashboard(outputs)
            self.update_screening_dashboard(outputs)
            self.update_reliability_dashboard(outputs)
            self.update_recommendations_dashboard(outputs)
            self.update_export_dashboard(outputs)
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
        except Exception as exc:
            QMessageBox.warning(self, "Distribution plot failed", str(exc))

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
        elif key == "selected_feature_by_group":
            self.generate_selected_group_plot()
        elif not hasattr(self, "plot_paths") or key not in self.plot_paths or not Path(self.plot_paths.get(key, "")).exists():
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

    def _make_table(self) -> QTableWidget:
        """Compatibility alias used by newer pages."""
        return self._simple_table()

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


    def _relationships_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        card = Card(
            "Feature Relationships",
            "Correlation, redundancy, family structure, and exploratory PCA. This screen asks whether features form interpretable subsystems or redundant blocks before any ML work."
        )
        self.relationship_metric_grid = QGridLayout()
        card.layout.addLayout(self.relationship_metric_grid)

        controls = QHBoxLayout()
        self.relationship_feature_combo = QComboBox()
        self.relationship_feature_combo.setMinimumWidth(360)
        self.relationship_feature_combo.currentIndexChanged.connect(lambda _=0: self.preview_relationship_plot("selected_feature_correlations"))
        controls.addWidget(QLabel("Selected feature:"))
        controls.addWidget(self.relationship_feature_combo)
        controls.addStretch(1)
        open_btn = QPushButton("Open current plot")
        open_btn.setProperty("secondary", True)
        open_btn.clicked.connect(self.open_current_relationship_plot)
        controls.addWidget(open_btn)
        card.layout.addLayout(controls)

        tabs = QTabWidget()
        self.relationship_summary_table = self._simple_table()
        self.relationship_redundant_table = self._simple_table()
        self.relationship_modules_table = self._simple_table()
        self.relationship_family_matrix_table = self._simple_table()
        self.relationship_pca_table = self._simple_table()
        self.relationship_loadings_table = self._simple_table()
        for title, tbl in [
            ("Summary", self.relationship_summary_table),
            ("Redundant pairs", self.relationship_redundant_table),
            ("Modules", self.relationship_modules_table),
            ("Family matrix", self.relationship_family_matrix_table),
            ("PCA summary", self.relationship_pca_table),
            ("PCA loadings", self.relationship_loadings_table),
        ]:
            tabs.addTab(tbl, title)
        card.layout.addWidget(tabs, 1)

        plot_card = Card("Relationship plots", "Each plot includes interpretation guidance. Use these to distinguish useful feature blocks from avoidable redundancy.")
        split = QSplitter(Qt.Horizontal)
        left = QWidget(); left_l = QVBoxLayout(left); left_l.setContentsMargins(0,0,0,0); left_l.setSpacing(8)
        for label, key in [
            ("Correlation heatmap", "relationship_correlation_heatmap"),
            ("Top redundant pairs", "relationship_redundant_pairs"),
            ("Family/block matrix", "relationship_family_matrix"),
            ("PCA scree", "relationship_pca_scree"),
            ("PCA recording map", "relationship_pca_scores"),
            ("PCA top loadings", "relationship_pca_loadings"),
            ("Selected feature links", "selected_feature_correlations"),
        ]:
            b = QPushButton(label)
            b.setProperty("secondary", True)
            b.clicked.connect(lambda _=False, k=key: self.preview_relationship_plot(k))
            left_l.addWidget(b)
        left_l.addStretch(1)
        right = QWidget(); right_l = QVBoxLayout(right); right_l.setContentsMargins(0,0,0,0); right_l.setSpacing(10)
        self.relationship_plot_preview = QLabel("Run Feature Analysis, then select a relationship plot.")
        self.relationship_plot_preview.setAlignment(Qt.AlignCenter)
        self.relationship_plot_preview.setMinimumHeight(520)
        self.relationship_plot_preview.setStyleSheet(f"background:#FFFFFF; border:1px solid {LINE}; border-radius:12px; color:{MUTED};")
        self.relationship_interpretation_label = QLabel("Select a plot to see structured interpretation guidance.")
        self.relationship_interpretation_label.setWordWrap(True)
        self.relationship_interpretation_label.setStyleSheet(f"background:#FFFFFF; color:{INK}; border:1px solid {LINE}; border-radius:10px; padding:12px;")
        right_l.addWidget(self.relationship_plot_preview, 1)
        right_l.addWidget(self.relationship_interpretation_label)
        split.addWidget(left); split.addWidget(right); split.setStretchFactor(1, 1)
        plot_card.layout.addWidget(split)
        layout.addWidget(card)
        layout.addWidget(plot_card, 1)
        return self._wrap_scroll(body)

    def update_relationships_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "relationship_summary_table"):
            return
        while self.relationship_metric_grid.count():
            item = self.relationship_metric_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        summary = outputs.get("feature_relationship_summary", pd.DataFrame())
        def metric(name: str, default: object = "—") -> object:
            if summary is None or summary.empty or "metric" not in summary.columns:
                return default
            row = summary.loc[summary["metric"].astype(str).eq(name)]
            return row["value"].iloc[0] if not row.empty else default
        tiles = [
            ("Numeric features", metric("numeric_features"), "usable for relationships"),
            ("Pairwise links", metric("feature_pairs_evaluated"), "Spearman pairs"),
            ("|ρ| ≥ .80", metric("redundant_pairs_abs_rho_ge_0_80"), "strong redundancy"),
            ("Modules", metric("correlation_modules_abs_rho_ge_0_70"), "connected blocks"),
            ("PC1 variance", metric("pc1_variance_percent"), "% if PCA available"),
            ("PC1–PC3", metric("pc1_pc3_cumulative_percent"), "cumulative variance"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.relationship_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx // 3, idx % 3)
        self._fill_table(self.relationship_summary_table, summary)
        self._fill_table(self.relationship_redundant_table, outputs.get("feature_redundant_pairs", pd.DataFrame()))
        self._fill_table(self.relationship_modules_table, outputs.get("feature_relationship_modules", pd.DataFrame()))
        self._fill_table(self.relationship_family_matrix_table, outputs.get("feature_family_correlation_matrix", pd.DataFrame()))
        self._fill_table(self.relationship_pca_table, outputs.get("feature_pca_summary", pd.DataFrame()))
        self._fill_table(self.relationship_loadings_table, outputs.get("feature_pca_loadings", pd.DataFrame()))
        if hasattr(self, "relationship_feature_combo"):
            current = self.relationship_feature_combo.currentText()
            self.relationship_feature_combo.blockSignals(True)
            self.relationship_feature_combo.clear()
            corr = outputs.get("feature_correlation_long", pd.DataFrame())
            feats = []
            if corr is not None and not corr.empty:
                feats = sorted(set(corr.get("feature_1", pd.Series(dtype=str)).astype(str)).union(set(corr.get("feature_2", pd.Series(dtype=str)).astype(str))))
            self.relationship_feature_combo.addItems(feats[:500])
            if current and current in feats:
                self.relationship_feature_combo.setCurrentText(current)
            self.relationship_feature_combo.blockSignals(False)

    def _selected_relationship_feature(self) -> str | None:
        if hasattr(self, "relationship_feature_combo") and self.relationship_feature_combo.count() > 0:
            return self.relationship_feature_combo.currentText()
        return None

    def generate_selected_feature_correlations(self) -> None:
        feature = self._selected_relationship_feature()
        if not feature:
            return
        try:
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            path = plot_selected_feature_correlations(self.outputs.get("feature_correlation_long", pd.DataFrame()), feature, plots_dir / "selected_feature_correlations.png")
            if not hasattr(self, "plot_paths"):
                self.plot_paths = {}
            self.plot_paths["selected_feature_correlations"] = str(path)
        except Exception:
            pass

    def preview_relationship_plot(self, key: str) -> None:
        if key == "selected_feature_correlations":
            self.generate_selected_feature_correlations()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths or not Path(self.plot_paths.get(key, "")).exists():
            self.regenerate_overview_plots()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths:
            QMessageBox.information(self, "Plot unavailable", "Run Feature Analysis first, or this relationship plot could not be generated for the current dataset.")
            return
        path = Path(self.plot_paths[key])
        if not path.exists():
            QMessageBox.information(self, "Plot unavailable", f"Plot file not found:\n{path}")
            return
        self.current_relationship_plot = path
        self.update_relationship_interpretation(key)
        pix = QPixmap(str(path))
        if pix.isNull():
            self.relationship_plot_preview.setText(f"Could not load plot:\n{path}")
            return
        self.relationship_plot_preview.setPixmap(pix.scaled(self.relationship_plot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.relationship_plot_preview.setToolTip(str(path))

    def update_relationship_interpretation(self, key: str) -> None:
        if not hasattr(self, "relationship_interpretation_label"):
            return
        feature = self._selected_relationship_feature() or "selected feature"
        captions = {
            "relationship_correlation_heatmap": (
                "<b>What it shows</b><br>Spearman feature-feature correlation among the most connected numeric features.<br><br>"
                "<b>Concerning pattern</b><br>Large solid blocks mean many features carry overlapping information. That can inflate interpretability claims and destabilize ML models.<br><br>"
                "<b>Do not overinterpret</b><br>Correlation is not physiology by itself. Correlated features may share computation, task, QC sensitivity, or true subsystem behavior.<br><br>"
                "<b>Next check</b><br>Review redundant pairs, family matrix, PCA loadings, and QC associations for the same block."
            ),
            "relationship_redundant_pairs": (
                "<b>What it shows</b><br>The strongest feature-feature associations ranked by |Spearman rho|.<br><br>"
                "<b>Concerning pattern</b><br>|ρ| ≥ .80 suggests near-duplicate information. |ρ| ≥ .90 often indicates variables should not all enter the same small-sample ML model.<br><br>"
                "<b>Do not overinterpret</b><br>Redundancy is not automatically bad for descriptive science; it can validate a feature family. It is mainly a problem for ML and feature-count inflation.<br><br>"
                "<b>Next check</b><br>Use the module table to decide whether to keep one representative or carry the block forward as a family."
            ),
            "relationship_family_matrix": (
                "<b>What it shows</b><br>Average absolute correlation within and between feature families/subsystems.<br><br>"
                "<b>Concerning pattern</b><br>Very high cross-family coupling suggests features may be dominated by shared acquisition, task, or scaling effects rather than distinct physiology.<br><br>"
                "<b>Do not overinterpret</b><br>Some cross-family coupling is expected in connected speech because speech subsystems are coordinated.<br><br>"
                "<b>Next check</b><br>Compare family matrix with QC Integration and PCA loadings."
            ),
            "relationship_pca_scree": (
                "<b>What it shows</b><br>How much feature variance is captured by each principal component after robust scaling.<br><br>"
                "<b>Concerning pattern</b><br>A very dominant PC1 may indicate a global size/quality/task axis rather than independent feature domains.<br><br>"
                "<b>Do not overinterpret</b><br>PCA is exploratory and unsupervised. It is not a clinical classifier or biomarker result.<br><br>"
                "<b>Next check</b><br>Inspect PCA loadings and the PCA recording map to understand what drives each component."
            ),
            "relationship_pca_scores": (
                "<b>What it shows</b><br>Recordings plotted on PC1 and PC2 using numeric features after robust scaling.<br><br>"
                "<b>Concerning pattern</b><br>Clear clusters may reflect task, cohort, device, QC, or real physiological differences; they require annotation before interpretation.<br><br>"
                "<b>Do not overinterpret</b><br>Separation on PCA is not evidence of diagnostic performance. It is structure discovery only.<br><br>"
                "<b>Next check</b><br>Color/stratify by task, diagnosis, QC, or severity in later modules."
            ),
            "relationship_pca_loadings": (
                "<b>What it shows</b><br>The features with the largest absolute loadings on early PCs.<br><br>"
                "<b>Concerning pattern</b><br>If one QC-sensitive or high-missingness feature family dominates PC1, the main structure may be technical rather than physiological.<br><br>"
                "<b>Do not overinterpret</b><br>Loadings describe variance structure, not outcome association or clinical validity.<br><br>"
                "<b>Next check</b><br>Compare dominant loading features with Distributions, QC Integration, and later outcome screening."
            ),
            "selected_feature_correlations": (
                f"<b>What it shows</b><br>The strongest relationships involving <b>{feature}</b>.<br><br>"
                "<b>Concerning pattern</b><br>If a feature has many very high correlations, it may be redundant or part of a broad latent block.<br><br>"
                "<b>Do not overinterpret</b><br>A feature with many correlations is not necessarily better; it may simply share computation with many variables.<br><br>"
                "<b>Next check</b><br>Inspect whether linked features are from the same family, same task, or same QC-sensitive subsystem."
            ),
        }
        self.relationship_interpretation_label.setText(captions.get(key, "Select a relationship plot to see structured interpretation guidance."))

    def open_current_relationship_plot(self) -> None:
        path = getattr(self, "current_relationship_plot", None)
        if not path:
            QMessageBox.information(self, "No current plot", "Preview a relationship plot first, then open the full-resolution file.")
            return
        self.open_file(Path(path))

    def _screening_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        card = Card(
            "Group / Outcome Screening",
            "Descriptive univariate screening of feature associations with clinical labels, task/group variables, and continuous outcomes. This is not modelling, prediction, biomarker discovery, or adjusted inference."
        )
        self.screening_metric_grid = QGridLayout()
        card.layout.addLayout(self.screening_metric_grid)
        tabs = QTabWidget()
        self.screening_summary_table = self._make_table()
        self.screening_catalog_table = self._make_table()
        self.screening_continuous_table = self._make_table()
        self.screening_group_table = self._make_table()
        self.screening_balance_table = self._make_table()
        tabs.addTab(self.screening_summary_table, "Summary")
        tabs.addTab(self.screening_catalog_table, "Variable catalog")
        tabs.addTab(self.screening_continuous_table, "Continuous outcomes")
        tabs.addTab(self.screening_group_table, "Categorical groups")
        tabs.addTab(self.screening_balance_table, "Group balance")
        card.layout.addWidget(tabs)

        plot_card = Card(
            "Screening plots",
            "Use these plots to identify candidate feature-label patterns that need task, QC, covariate, and repeated-measures review. Effects here are screening signals only."
        )
        controls = QHBoxLayout()
        self.screening_feature_combo = QComboBox()
        self.screening_variable_combo = QComboBox()
        controls.addWidget(QLabel("Feature:")); controls.addWidget(self.screening_feature_combo)
        controls.addWidget(QLabel("Outcome / group:")); controls.addWidget(self.screening_variable_combo)
        plot_card.layout.addLayout(controls)
        split = QSplitter(Qt.Horizontal)
        left = QWidget(); left_l = QVBoxLayout(left); left_l.setContentsMargins(0,0,0,0)
        for label, key in [
            ("Group balance", "screening_group_balance"),
            ("Top screening effects", "screening_effect_ranking"),
            ("Feature × continuous outcome heatmap", "screening_continuous_heatmap"),
            ("Feature × categorical group heatmap", "screening_group_heatmap"),
            ("Effect-size landscape", "screening_effect_landscape"),
            ("Selected feature vs selected outcome/group", "selected_feature_outcome"),
        ]:
            b = QPushButton(label)
            b.setProperty("secondary", True)
            b.clicked.connect(lambda _=False, k=key: self.preview_screening_plot(k))
            left_l.addWidget(b)
        open_btn = QPushButton("Open current plot file")
        open_btn.setProperty("primary", True)
        open_btn.clicked.connect(self.open_current_screening_plot)
        left_l.addWidget(open_btn)
        left_l.addStretch(1)
        right = QWidget(); right_l = QVBoxLayout(right); right_l.setContentsMargins(0,0,0,0)
        self.screening_plot_preview = QLabel("Run Feature Analysis, then choose a screening plot.")
        self.screening_plot_preview.setAlignment(Qt.AlignCenter)
        self.screening_plot_preview.setMinimumHeight(430)
        self.screening_plot_preview.setStyleSheet(f"background:#FFFFFF; border:1px solid {LINE}; border-radius:12px; color:{MUTED};")
        self.screening_interpretation_label = QLabel("Select a plot to see structured interpretation guidance.")
        self.screening_interpretation_label.setWordWrap(True)
        self.screening_interpretation_label.setStyleSheet(f"background:#FFFFFF; color:{INK}; border:1px solid {LINE}; border-radius:10px; padding:12px;")
        right_l.addWidget(self.screening_plot_preview, 1)
        right_l.addWidget(self.screening_interpretation_label)
        split.addWidget(left); split.addWidget(right); split.setStretchFactor(1, 1)
        plot_card.layout.addWidget(split)
        layout.addWidget(card)
        layout.addWidget(plot_card, 1)
        return self._wrap_scroll(body)

    def update_screening_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "screening_summary_table"):
            return
        while self.screening_metric_grid.count():
            item = self.screening_metric_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        summary = outputs.get("screening_summary", pd.DataFrame())
        def metric(name: str, default: object = "—") -> object:
            if summary is None or summary.empty or "metric" not in summary.columns:
                return default
            row = summary.loc[summary["metric"].astype(str).eq(name)]
            return row["value"].iloc[0] if not row.empty else default
        cont = outputs.get("screening_continuous_outcome_associations", pd.DataFrame())
        cat = outputs.get("screening_categorical_group_associations", pd.DataFrame())
        tiles = [
            ("Screening variables", metric("screening_variables_detected"), "outcomes/groups/covariates"),
            ("Continuous screens", metric("continuous_feature_outcome_tests"), "Spearman screens"),
            ("Group screens", metric("categorical_group_feature_tests"), "robust contrasts"),
            ("Continuous |effect| ≥ .30", metric("continuous_abs_effect_ge_0_30"), "monitor/review"),
            ("Group |effect| ≥ .30", metric("group_abs_effect_ge_0_30"), "monitor/review"),
            ("Top effect", "—" if (cont is None or cont.empty) and (cat is None or cat.empty) else round(float(pd.concat([d for d in [cont.get('abs_effect') if cont is not None and not cont.empty else pd.Series(dtype=float), cat.get('abs_effect') if cat is not None and not cat.empty else pd.Series(dtype=float)]], ignore_index=True).max()), 3), "descriptive only"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.screening_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx // 3, idx % 3)
        self._fill_table(self.screening_summary_table, summary)
        self._fill_table(self.screening_catalog_table, outputs.get("screening_variable_catalog", pd.DataFrame()))
        self._fill_table(self.screening_continuous_table, cont)
        self._fill_table(self.screening_group_table, cat)
        self._fill_table(self.screening_balance_table, outputs.get("screening_group_balance", pd.DataFrame()))
        current_f = self.screening_feature_combo.currentText() if hasattr(self, "screening_feature_combo") else ""
        current_v = self.screening_variable_combo.currentText() if hasattr(self, "screening_variable_combo") else ""
        self.screening_feature_combo.blockSignals(True); self.screening_variable_combo.blockSignals(True)
        self.screening_feature_combo.clear(); self.screening_variable_combo.clear()
        feats = []
        for df in [cont, cat]:
            if df is not None and not df.empty and "feature" in df.columns:
                feats.extend(df["feature"].astype(str).tolist())
        vars_ = []
        if cont is not None and not cont.empty and "outcome_variable" in cont.columns:
            vars_.extend(cont["outcome_variable"].astype(str).tolist())
        if cat is not None and not cat.empty and "group_variable" in cat.columns:
            vars_.extend(cat["group_variable"].astype(str).tolist())
        feats = sorted(set(feats)); vars_ = sorted(set(vars_))
        self.screening_feature_combo.addItems(feats[:500])
        self.screening_variable_combo.addItems(vars_[:100])
        if current_f and current_f in feats: self.screening_feature_combo.setCurrentText(current_f)
        if current_v and current_v in vars_: self.screening_variable_combo.setCurrentText(current_v)
        self.screening_feature_combo.blockSignals(False); self.screening_variable_combo.blockSignals(False)

    def _selected_screening_feature(self) -> str | None:
        if hasattr(self, "screening_feature_combo") and self.screening_feature_combo.count() > 0:
            return self.screening_feature_combo.currentText()
        return None

    def _selected_screening_variable(self) -> str | None:
        if hasattr(self, "screening_variable_combo") and self.screening_variable_combo.count() > 0:
            return self.screening_variable_combo.currentText()
        return None

    def generate_selected_feature_outcome(self) -> None:
        feature = self._selected_screening_feature()
        variable = self._selected_screening_variable()
        if not feature or not variable:
            return
        try:
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            path = plot_selected_feature_outcome(self.feature_df, feature, variable, plots_dir / "selected_feature_outcome.png")
            if not hasattr(self, "plot_paths"):
                self.plot_paths = {}
            self.plot_paths["selected_feature_outcome"] = str(path)
        except Exception:
            pass

    def preview_screening_plot(self, key: str) -> None:
        if key == "selected_feature_outcome":
            self.generate_selected_feature_outcome()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths or not Path(self.plot_paths.get(key, "")).exists():
            self.regenerate_overview_plots()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths:
            QMessageBox.information(self, "Plot unavailable", "Run Feature Analysis first, or this screening plot could not be generated for the current dataset.")
            return
        path = Path(self.plot_paths[key])
        if not path.exists():
            QMessageBox.information(self, "Plot unavailable", f"Plot file not found:\n{path}")
            return
        self.current_screening_plot = path
        self.update_screening_interpretation(key)
        pix = QPixmap(str(path))
        if pix.isNull():
            self.screening_plot_preview.setText(f"Could not load plot:\n{path}")
            return
        self.screening_plot_preview.setPixmap(pix.scaled(self.screening_plot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.screening_plot_preview.setToolTip(str(path))

    def update_screening_interpretation(self, key: str) -> None:
        if not hasattr(self, "screening_interpretation_label"):
            return
        feature = self._selected_screening_feature() or "selected feature"
        variable = self._selected_screening_variable() or "selected outcome/group"
        captions = {
            "screening_group_balance": (
                "<b>What it shows</b><br>Sample size per level for the most relevant detected group/outcome variable.<br><br>"
                "<b>Concerning pattern</b><br>Small or highly imbalanced levels make group contrasts unstable and can bias downstream ML splits.<br><br>"
                "<b>Do not overinterpret</b><br>Balance does not prove comparability; groups can still differ in task, QC, device, age, severity, or visit count.<br><br>"
                "<b>Next check</b><br>Review the group-balance table and compare with QC Integration and Missingness by group."
            ),
            "screening_effect_ranking": (
                "<b>What it shows</b><br>The largest descriptive univariate feature-outcome or feature-group effects.<br><br>"
                "<b>Concerning pattern</b><br>Large effects are interesting only if they are not driven by small group sizes, QC, missingness, outliers, or task imbalance.<br><br>"
                "<b>Do not overinterpret</b><br>This is a screening ranking, not biomarker discovery, not adjusted inference, and not predictive validation.<br><br>"
                "<b>Next check</b><br>Inspect the selected feature plot and verify the same feature in Distributions, QC Integration, and Feature Relationships."
            ),
            "screening_continuous_heatmap": (
                "<b>What it shows</b><br>Spearman correlations between features and continuous outcomes such as ALSFRS scores, severity scores, progression rate, or intelligibility.<br><br>"
                "<b>Concerning pattern</b><br>Broad high associations across many features may indicate a global task/QC/severity axis rather than specific physiology.<br><br>"
                "<b>Do not overinterpret</b><br>Spearman screens are unadjusted and ignore repeated measures. They do not replace mixed models or ML validation.<br><br>"
                "<b>Next check</b><br>Check whether candidate features are QC-sensitive, redundant, or missing non-randomly."
            ),
            "screening_group_heatmap": (
                "<b>What it shows</b><br>Robust descriptive feature contrasts across categorical groups such as diagnosis, task, severity_bin, session, device, or sex/gender.<br><br>"
                "<b>Concerning pattern</b><br>Large effects for task or device can masquerade as clinical effects if not handled later.<br><br>"
                "<b>Do not overinterpret</b><br>Group contrast screens are not corrected clinical tests and are not adjusted for covariates or repeated recordings.<br><br>"
                "<b>Next check</b><br>Review group balance, task distribution, QC family burden, and selected feature plots."
            ),
            "screening_effect_landscape": (
                "<b>What it shows</b><br>Effect magnitude versus pairwise sample size for all evaluated screening relationships.<br><br>"
                "<b>Concerning pattern</b><br>Very large effects with very small N are unstable and should not drive conclusions.<br><br>"
                "<b>Do not overinterpret</b><br>A small effect with high N may be reliable but clinically small; a large effect with low N may be noise.<br><br>"
                "<b>Next check</b><br>Prioritize effects that are moderate/large, adequate-N, distributionally plausible, and not QC-driven."
            ),
            "selected_feature_outcome": (
                f"<b>What it shows</b><br>Direct descriptive view of <b>{feature}</b> against <b>{variable}</b>.<br><br>"
                "<b>Concerning pattern</b><br>One or two points driving the pattern, unequal variance, sparse groups, or visible outliers mean the association needs review.<br><br>"
                "<b>Do not overinterpret</b><br>This plot is descriptive. It does not establish diagnosis, prognosis, causality, or model performance.<br><br>"
                "<b>Next check</b><br>Check the feature's missingness, distribution, QC sensitivity, redundancy, and reliability before interpreting clinically."
            ),
        }
        self.screening_interpretation_label.setText(captions.get(key, "Select a screening plot to see structured interpretation guidance."))

    def open_current_screening_plot(self) -> None:
        path = getattr(self, "current_screening_plot", None)
        if not path:
            QMessageBox.information(self, "No current plot", "Preview a screening plot first, then open the full-resolution file.")
            return
        self.open_file(Path(path))

    def _reliability_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        card = Card(
            "Reliability / Repeatability",
            "Repeated-measures stability review. This screen asks whether features are stable across repeated recordings/sessions, or mainly session-, task-, QC-, or acquisition-dependent."
        )
        self.reliability_metric_grid = QGridLayout()
        card.layout.addLayout(self.reliability_metric_grid)

        controls = QHBoxLayout()
        self.reliability_feature_combo = QComboBox()
        self.reliability_feature_combo.setMinimumWidth(360)
        self.reliability_feature_combo.currentIndexChanged.connect(lambda _=0: self.preview_reliability_plot("selected_feature_reliability"))
        controls.addWidget(QLabel("Selected feature:"))
        controls.addWidget(self.reliability_feature_combo)
        controls.addStretch(1)
        open_btn = QPushButton("Open current plot file")
        open_btn.setProperty("primary", True)
        open_btn.clicked.connect(self.open_current_reliability_plot)
        controls.addWidget(open_btn)
        card.layout.addLayout(controls)

        tabs = QTabWidget()
        self.reliability_design_table = self._simple_table()
        self.reliability_repeatability_table = self._simple_table()
        self.reliability_family_table = self._simple_table()
        self.reliability_subjects_table = self._simple_table()
        self.reliability_screen_table = self._simple_table()
        for title, tbl in [
            ("Design support", self.reliability_design_table),
            ("Feature repeatability", self.reliability_repeatability_table),
            ("Family summary", self.reliability_family_table),
            ("Subject counts", self.reliability_subjects_table),
            ("Readiness screen", self.reliability_screen_table),
        ]:
            tabs.addTab(tbl, title)
        card.layout.addWidget(tabs, 1)

        plot_card = Card(
            "Reliability plots",
            "Use these plots to distinguish stable participant/setup traits from session-level variability. Interpretation depends on repeated-record design support."
        )
        split = QSplitter(Qt.Horizontal)
        left = QWidget(); left_l = QVBoxLayout(left); left_l.setContentsMargins(0,0,0,0); left_l.setSpacing(8)
        for label, key in [
            ("Reliability status counts", "reliability_status_counts"),
            ("ICC ranking", "reliability_icc_ranking"),
            ("Within vs between variance", "reliability_variance_landscape"),
            ("Reliability by family", "reliability_family_summary"),
            ("Subject record counts", "reliability_subject_counts"),
            ("Selected feature spaghetti", "selected_feature_reliability"),
        ]:
            b = QPushButton(label)
            b.setProperty("secondary", True)
            b.clicked.connect(lambda _=False, k=key: self.preview_reliability_plot(k))
            left_l.addWidget(b)
        left_l.addStretch(1)
        right = QWidget(); right_l = QVBoxLayout(right); right_l.setContentsMargins(0,0,0,0); right_l.setSpacing(10)
        self.reliability_plot_preview = QLabel("Run Feature Analysis, then choose a reliability plot.")
        self.reliability_plot_preview.setAlignment(Qt.AlignCenter)
        self.reliability_plot_preview.setMinimumHeight(500)
        self.reliability_plot_preview.setStyleSheet(f"background:#FFFFFF; border:1px solid {LINE}; border-radius:12px; color:{MUTED};")
        self.reliability_interpretation_label = QLabel("Select a plot to see structured interpretation guidance.")
        self.reliability_interpretation_label.setWordWrap(True)
        self.reliability_interpretation_label.setStyleSheet(f"background:#FFFFFF; color:{INK}; border:1px solid {LINE}; border-radius:10px; padding:12px;")
        right_l.addWidget(self.reliability_plot_preview, 1)
        right_l.addWidget(self.reliability_interpretation_label)
        split.addWidget(left); split.addWidget(right); split.setStretchFactor(1, 1)
        plot_card.layout.addWidget(split)
        layout.addWidget(card)
        layout.addWidget(plot_card, 1)
        return self._wrap_scroll(body)

    def update_reliability_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "reliability_repeatability_table"):
            return
        while self.reliability_metric_grid.count():
            item = self.reliability_metric_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        design = outputs.get("reliability_design_summary", pd.DataFrame())
        rep = outputs.get("feature_repeatability_summary", pd.DataFrame())
        fam = outputs.get("reliability_family_summary", pd.DataFrame())
        subj = outputs.get("reliability_subject_record_counts", pd.DataFrame())
        def metric(name: str, default: object = "—") -> object:
            if design is None or design.empty or "metric" not in design.columns:
                return default
            row = design.loc[design["metric"].astype(str).eq(name)]
            return row["value"].iloc[0] if not row.empty else default
        stable = int(rep["reliability_status"].astype(str).eq("stable").sum()) if rep is not None and not rep.empty and "reliability_status" in rep.columns else 0
        moderate = int(rep["reliability_status"].astype(str).eq("moderate").sum()) if rep is not None and not rep.empty and "reliability_status" in rep.columns else 0
        evaluable = int(pd.to_numeric(rep.get("icc1_proxy", pd.Series(dtype=float)), errors="coerce").notna().sum()) if rep is not None and not rep.empty else 0
        tiles = [
            ("Subject column", metric("subject_column"), "repeatability anchor"),
            ("Subjects with repeats", metric("subjects_with_repeats", 0), "required for ICC-style review"),
            ("Median records / subject", metric("median_records_per_subject", 0), "design depth"),
            ("Evaluable features", evaluable, "ICC-style estimates"),
            ("Stable features", stable, "ICC proxy ≥ .75"),
            ("Moderate features", moderate, "ICC proxy .50-.75"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.reliability_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx // 3, idx % 3)
        self._fill_table(self.reliability_design_table, design)
        self._fill_table(self.reliability_repeatability_table, rep)
        self._fill_table(self.reliability_family_table, fam)
        self._fill_table(self.reliability_subjects_table, subj)
        self._fill_table(self.reliability_screen_table, outputs.get("feature_reliability_screen", pd.DataFrame()))
        current = self.reliability_feature_combo.currentText() if hasattr(self, "reliability_feature_combo") else ""
        self.reliability_feature_combo.blockSignals(True)
        self.reliability_feature_combo.clear()
        feats = rep["feature"].astype(str).tolist() if rep is not None and not rep.empty and "feature" in rep.columns else []
        self.reliability_feature_combo.addItems(feats[:500])
        if current and current in feats:
            self.reliability_feature_combo.setCurrentText(current)
        self.reliability_feature_combo.blockSignals(False)

    def _selected_reliability_feature(self) -> str | None:
        if hasattr(self, "reliability_feature_combo") and self.reliability_feature_combo.count() > 0:
            return self.reliability_feature_combo.currentText()
        return None

    def generate_selected_feature_reliability(self) -> None:
        feature = self._selected_reliability_feature()
        if not feature:
            return
        try:
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            path = plot_selected_feature_reliability(self.feature_df, feature, plots_dir / "selected_feature_reliability.png")
            if not hasattr(self, "plot_paths"):
                self.plot_paths = {}
            self.plot_paths["selected_feature_reliability"] = str(path)
        except Exception as exc:
            QMessageBox.warning(self, "Reliability plot failed", str(exc))

    def preview_reliability_plot(self, key: str) -> None:
        if key == "selected_feature_reliability":
            self.generate_selected_feature_reliability()
        elif not hasattr(self, "plot_paths") or key not in self.plot_paths or not Path(self.plot_paths.get(key, "")).exists():
            self.regenerate_overview_plots()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths:
            QMessageBox.information(self, "Plot unavailable", "Run Feature Analysis first, or this reliability plot could not be generated for the current dataset.")
            return
        path = Path(self.plot_paths[key])
        if not path.exists():
            QMessageBox.information(self, "Plot unavailable", f"Plot file not found:\n{path}")
            return
        self.current_reliability_plot = path
        self.update_reliability_interpretation(key)
        pix = QPixmap(str(path))
        if pix.isNull():
            self.reliability_plot_preview.setText(f"Could not load plot:\n{path}")
            return
        self.reliability_plot_preview.setPixmap(pix.scaled(self.reliability_plot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.reliability_plot_preview.setToolTip(str(path))

    def update_reliability_interpretation(self, key: str) -> None:
        if not hasattr(self, "reliability_interpretation_label"):
            return
        feature = self._selected_reliability_feature() or "selected feature"
        captions = {
            "reliability_status_counts": (
                "<b>What it shows</b><br>Counts of features classified as stable, moderate, variable, unstable, or not evaluable based on repeated-record variance structure.<br><br>"
                "<b>Concerning pattern</b><br>Many not-evaluable or unstable features means the dataset may not support longitudinal interpretation for those measures.<br><br>"
                "<b>Do not overinterpret</b><br>These are screening labels, not formal mixed-effects reliability estimates.<br><br>"
                "<b>Next check</b><br>Inspect design support and selected-feature spaghetti plots."
            ),
            "reliability_icc_ranking": (
                "<b>What it shows</b><br>Features ranked by ICC(1)-style variance-ratio proxy: between-subject variance divided by total between+within variance.<br><br>"
                "<b>Concerning pattern</b><br>Low ICC means repeated recordings from the same subject vary substantially; this can weaken longitudinal monitoring.<br><br>"
                "<b>Do not overinterpret</b><br>High ICC can reflect stable device/setup effects as well as stable physiology. Compare with QC Integration.<br><br>"
                "<b>Next check</b><br>Review family summary, QC sensitivity, and selected feature trajectories."
            ),
            "reliability_variance_landscape": (
                "<b>What it shows</b><br>Within-subject variance versus between-subject variance for evaluable features. Larger points have higher ICC proxy.<br><br>"
                "<b>Concerning pattern</b><br>High within-subject variance suggests session/task/acquisition instability; high between-subject variance can be useful but may include stable device effects.<br><br>"
                "<b>Do not overinterpret</b><br>This does not separate physiology from acquisition unless paired with QC and task context.<br><br>"
                "<b>Next check</b><br>Inspect features with high between-subject and low within-subject variance."
            ),
            "reliability_family_summary": (
                "<b>What it shows</b><br>Median repeatability by feature family/subsystem.<br><br>"
                "<b>Concerning pattern</b><br>A whole family with low repeatability may reflect task support problems, algorithm instability, or acquisition sensitivity.<br><br>"
                "<b>Do not overinterpret</b><br>Family medians can hide individual strong features.<br><br>"
                "<b>Next check</b><br>Open the feature repeatability table and compare with distributions/QC."
            ),
            "reliability_subject_counts": (
                "<b>What it shows</b><br>How many repeated records each subject contributes.<br><br>"
                "<b>Concerning pattern</b><br>Few repeated subjects or very unequal recording counts make reliability estimates fragile.<br><br>"
                "<b>Do not overinterpret</b><br>More records per subject improve reliability estimation but do not guarantee task or QC comparability.<br><br>"
                "<b>Next check</b><br>Review task/session structure and subject-level QC burden."
            ),
            "selected_feature_reliability": (
                f"<b>What it shows</b><br>Repeated-record trajectory for <b>{feature}</b> within each subject when subject labels are available.<br><br>"
                "<b>Concerning pattern</b><br>Large within-subject swings suggest session effects, task differences, QC artifacts, or genuine longitudinal change.<br><br>"
                "<b>Do not overinterpret</b><br>Spaghetti plots are descriptive; they do not prove progression or stability.<br><br>"
                "<b>Next check</b><br>Compare against visit timing, task, severity, and QC."
            ),
        }
        self.reliability_interpretation_label.setText(captions.get(key, "Select a reliability plot to see structured interpretation guidance."))

    def open_current_reliability_plot(self) -> None:
        path = getattr(self, "current_reliability_plot", None)
        if not path:
            QMessageBox.information(self, "No current plot", "Preview a reliability plot first, then open the full-resolution file.")
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

    def _recommendations_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        card = Card(
            "Feature Recommendation",
            "Integrated feature-readiness review. This screen combines missingness, distributions, QC sensitivity, redundancy, screening signal, and repeatability into transparent recommendations. It is not automatic ML feature selection."
        )
        self.recommendation_metric_grid = QGridLayout()
        card.layout.addLayout(self.recommendation_metric_grid)
        note = QLabel(
            "Use this page after all diagnostic modules. Recommended features are default candidates for export, while caution/review/exclude labels document why a feature needs sensitivity checks, recomputation, or manual review."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"background:#F7FAFD; color:{INK}; border:1px solid {LINE}; border-radius:10px; padding:10px;")
        card.layout.addWidget(note)
        tabs = QTabWidget()
        self.recommendation_summary_table = self._simple_table()
        self.recommendation_table = self._simple_table()
        self.recommendation_reasons_table = self._simple_table()
        self.recommendation_family_table = self._simple_table()
        self.ml_export_manifest_table = self._simple_table()
        for title, tbl in [
            ("Summary", self.recommendation_summary_table),
            ("Feature recommendations", self.recommendation_table),
            ("Reason counts", self.recommendation_reasons_table),
            ("Family summary", self.recommendation_family_table),
            ("ML export manifest", self.ml_export_manifest_table),
        ]:
            tabs.addTab(tbl, title)
        card.layout.addWidget(tabs, 1)
        layout.addWidget(card)

        plot_card = Card(
            "Recommendation plot board",
            "These plots explain the integrated readiness decision. Use them to document why features are exported, held for review, or excluded by default."
        )
        split = QSplitter(Qt.Horizontal)
        left = QWidget(); left_l = QVBoxLayout(left); left_l.setContentsMargins(0,0,0,0); left_l.setSpacing(8)
        for label, key in [
            ("Readiness category counts", "recommendation_counts"),
            ("Readiness landscape", "recommendation_score_landscape"),
            ("Review reason counts", "recommendation_reason_counts"),
            ("Readiness by family", "recommendation_family_summary"),
            ("ML export manifest", "ml_export_manifest_summary"),
        ]:
            b = QPushButton(label)
            b.setProperty("secondary", True)
            b.clicked.connect(lambda _=False, k=key: self.preview_recommendation_plot(k))
            left_l.addWidget(b)
        left_l.addStretch(1)
        open_btn = QPushButton("Open current plot file")
        open_btn.setProperty("primary", True)
        open_btn.clicked.connect(self.open_current_recommendation_plot)
        left_l.addWidget(open_btn)

        right = QWidget(); right_l = QVBoxLayout(right); right_l.setContentsMargins(0,0,0,0); right_l.setSpacing(10)
        self.recommendation_plot_preview = QLabel("Run Feature Analysis, then choose a recommendation plot.")
        self.recommendation_plot_preview.setAlignment(Qt.AlignCenter)
        self.recommendation_plot_preview.setMinimumHeight(500)
        self.recommendation_plot_preview.setStyleSheet(f"background:#FFFFFF; border:1px solid {LINE}; border-radius:12px; color:{MUTED};")
        self.recommendation_interpretation_label = QLabel("Select a plot to see structured interpretation guidance.")
        self.recommendation_interpretation_label.setWordWrap(True)
        self.recommendation_interpretation_label.setStyleSheet(f"background:#FFFFFF; color:{INK}; border:1px solid {LINE}; border-radius:10px; padding:12px;")
        right_l.addWidget(self.recommendation_plot_preview, 1)
        right_l.addWidget(self.recommendation_interpretation_label)
        split.addWidget(left); split.addWidget(right); split.setStretchFactor(1, 1)
        plot_card.layout.addWidget(split)
        layout.addWidget(plot_card, 1)
        return self._wrap_scroll(body)

    def update_recommendations_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "recommendation_table"):
            return
        while self.recommendation_metric_grid.count():
            item = self.recommendation_metric_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        recs = outputs.get("feature_recommendations", pd.DataFrame())
        summary = outputs.get("feature_recommendation_summary", pd.DataFrame())
        reasons = outputs.get("feature_recommendation_reason_counts", pd.DataFrame())
        fam = outputs.get("feature_recommendation_family_summary", pd.DataFrame())
        manifest = outputs.get("ml_export_manifest", pd.DataFrame())
        def m(metric: str, default: object = 0) -> object:
            if summary is None or summary.empty or "metric" not in summary.columns:
                return default
            row = summary.loc[summary["metric"].astype(str).eq(metric)]
            return row["value"].iloc[0] if not row.empty else default
        tiles = [
            ("Features reviewed", m("features_reviewed", 0), "integrated audit"),
            ("Recommended", m("recommended", 0), "clean default candidates"),
            ("With caution", m("recommended_with_caution", 0), "export but document risk"),
            ("Review before use", m("review_before_use", 0), "manual review / sensitivity"),
            ("Exclude/recompute", int(m("exclude_or_recompute", 0)) + int(m("exclude_by_default", 0)), "hold out by default"),
            ("Default ML export", m("default_ml_export_features", 0), "transparent manifest"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.recommendation_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx // 3, idx % 3)
        self._fill_table(self.recommendation_summary_table, summary)
        self._fill_table(self.recommendation_table, recs)
        self._fill_table(self.recommendation_reasons_table, reasons)
        self._fill_table(self.recommendation_family_table, fam)
        self._fill_table(self.ml_export_manifest_table, manifest)

    def preview_recommendation_plot(self, key: str) -> None:
        if not hasattr(self, "plot_paths") or key not in self.plot_paths or not Path(self.plot_paths.get(key, "")).exists():
            self.regenerate_overview_plots()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths:
            QMessageBox.information(self, "Plot unavailable", "Run Feature Analysis first, or this recommendation plot could not be generated for the current dataset.")
            return
        path = Path(self.plot_paths[key])
        if not path.exists():
            QMessageBox.information(self, "Plot unavailable", f"Plot file not found:\n{path}")
            return
        self.current_recommendation_plot = path
        self.update_recommendation_interpretation(key)
        pix = QPixmap(str(path))
        if pix.isNull():
            self.recommendation_plot_preview.setText(f"Could not load plot:\n{path}")
            return
        self.recommendation_plot_preview.setPixmap(pix.scaled(self.recommendation_plot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.recommendation_plot_preview.setToolTip(str(path))

    def update_recommendation_interpretation(self, key: str) -> None:
        if not hasattr(self, "recommendation_interpretation_label"):
            return
        captions = {
            "recommendation_counts": (
                "<b>What it shows</b><br>Number of features in each integrated readiness category.<br><br>"
                "<b>Concerning pattern</b><br>Many review/exclude features means the dataset still has substantial missingness, QC, distributional, redundancy, or reliability risk.<br><br>"
                "<b>Do not overinterpret</b><br>These are transparent review defaults, not automated feature selection.<br><br>"
                "<b>Next check</b><br>Open the feature recommendations table and inspect reasons."
            ),
            "recommendation_score_landscape": (
                "<b>What it shows</b><br>Feature readiness score versus missingness, with larger points indicating stronger QC association.<br><br>"
                "<b>Concerning pattern</b><br>Low score, high missingness, and large point size together suggest a risky feature for ML export.<br><br>"
                "<b>Do not overinterpret</b><br>A lower score does not prove the feature is invalid; it means more review is required.<br><br>"
                "<b>Next check</b><br>Review Missingness, QC Integration, Distributions, and Reliability for that feature."
            ),
            "recommendation_reason_counts": (
                "<b>What it shows</b><br>The most common reasons features were cautioned, reviewed, or excluded by default.<br><br>"
                "<b>Concerning pattern</b><br>Dominant QC or missingness reasons suggest a dataset-level problem, not just individual bad features.<br><br>"
                "<b>Do not overinterpret</b><br>Reason counts are not weighted by severity or clinical importance.<br><br>"
                "<b>Next check</b><br>Use family summary to see whether reasons cluster by subsystem."
            ),
            "recommendation_family_summary": (
                "<b>What it shows</b><br>Median readiness score by feature family/subsystem.<br><br>"
                "<b>Concerning pattern</b><br>A low-scoring family may indicate subsystem-level computation, task-support, or acquisition-sensitivity problems.<br><br>"
                "<b>Do not overinterpret</b><br>Family medians can hide strong individual features.<br><br>"
                "<b>Next check</b><br>Inspect the family table and individual feature recommendations."
            ),
            "ml_export_manifest_summary": (
                "<b>What it shows</b><br>How many features are included in the default ML export manifest versus held for review.<br><br>"
                "<b>Concerning pattern</b><br>A very small export set means most features need review, recomputation, or better data support.<br><br>"
                "<b>Do not overinterpret</b><br>This is not final ML selection; the ML GUI must still do fold-safe preprocessing and selection.<br><br>"
                "<b>Next check</b><br>Export the manifest and use it as the starting feature set for ML."
            ),
        }
        self.recommendation_interpretation_label.setText(captions.get(key, "Select a recommendation plot to see structured interpretation guidance."))

    def open_current_recommendation_plot(self) -> None:
        path = getattr(self, "current_recommendation_plot", None)
        if not path:
            QMessageBox.information(self, "No current plot", "Preview a recommendation plot first, then open the full-resolution file.")
            return
        self.open_file(Path(path))



    def _ml_export_builder_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        card = Card(
            "ML Export Builder",
            "Curate acoustic and kinematic feature outputs into clean acoustic-only, kinematic-only, and early-fusion ML-ready tables. This stage does not train models."
        )

        note = QLabel(
            "Use this page as the bridge between the Acoustic/Kinematics GUIs and the future ML GUI. "
            "Load completed feature outputs and registries, optionally add metadata/labels, then write standardized ML-ready tables and a unified feature manifest. "
            "The ML GUI should consume these exports rather than raw acoustic/kinematic engineering tables."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{INK}; background:#F7FAFD; border:1px solid {LINE}; border-radius:10px; padding:10px;")
        card.layout.addWidget(note)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        self.ml_acoustic_features_picker = FilePicker("Acoustic features per file", optional=True)
        self.ml_acoustic_registry_picker = FilePicker("Selected acoustic feature registry", optional=True)
        self.ml_acoustic_scale_picker = FilePicker("Acoustic measurement-scale registry", optional=True)
        self.ml_kinematic_features_picker = FilePicker("Kinematic aggregated features", optional=True)
        self.ml_kinematic_manifest_picker = FilePicker("Kinematic feature manifest", optional=True)
        self.ml_metadata_picker = FilePicker("Metadata / labels", optional=True)
        grid.addWidget(self.ml_acoustic_features_picker, 0, 0)
        grid.addWidget(self.ml_acoustic_registry_picker, 0, 1)
        grid.addWidget(self.ml_acoustic_scale_picker, 1, 0)
        grid.addWidget(self.ml_kinematic_features_picker, 1, 1)
        grid.addWidget(self.ml_kinematic_manifest_picker, 2, 0)
        grid.addWidget(self.ml_metadata_picker, 2, 1)
        card.layout.addLayout(grid)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output folder:"))
        self.ml_export_output_edit = QLineEdit()
        self.ml_export_output_edit.setPlaceholderText("Required output folder for Feature-GUI ML export package")
        out_row.addWidget(self.ml_export_output_edit, 1)
        browse = QPushButton("Browse")
        browse.setProperty("secondary", True)
        browse.clicked.connect(self.pick_ml_export_output_folder)
        out_row.addWidget(browse)
        card.layout.addLayout(out_row)

        action_row = QHBoxLayout()
        build = QPushButton("Build ML Export Package")
        build.setProperty("primary", True)
        build.clicked.connect(self.build_feature_ml_export_package)
        action_row.addWidget(build)
        open_btn = QPushButton("Open ML Export Folder")
        open_btn.setProperty("secondary", True)
        open_btn.clicked.connect(self.open_ml_export_folder)
        action_row.addWidget(open_btn)
        action_row.addStretch(1)
        card.layout.addLayout(action_row)

        self.ml_export_status = QTextEdit()
        self.ml_export_status.setReadOnly(True)
        self.ml_export_status.setMinimumHeight(150)
        self.ml_export_status.setPlaceholderText("ML export builder status messages will appear here.")
        card.layout.addWidget(self.ml_export_status)

        tabs = QTabWidget()
        self.ml_export_summary_table = self._simple_table()
        self.ml_export_overlap_table = self._simple_table()
        self.ml_export_manifest_table = self._simple_table()
        self.ml_export_acoustic_table = self._simple_table()
        self.ml_export_kinematic_table = self._simple_table()
        self.ml_export_fusion_table = self._simple_table()
        self.ml_export_exclusions_table = self._simple_table()
        tabs.addTab(self.ml_export_summary_table, "Export summary")
        tabs.addTab(self.ml_export_overlap_table, "Row alignment")
        tabs.addTab(self.ml_export_manifest_table, "Unified manifest")
        tabs.addTab(self.ml_export_acoustic_table, "Acoustic ML-ready")
        tabs.addTab(self.ml_export_kinematic_table, "Kinematic ML-ready")
        tabs.addTab(self.ml_export_fusion_table, "Early fusion")
        tabs.addTab(self.ml_export_exclusions_table, "Exclusions / notes")
        card.layout.addWidget(tabs)

        guide = Card(
            "Design boundary",
            "This stage prepares model-facing tables but deliberately does not impute, scale, select features inside folds, split subjects, or train models. Those steps belong in the ML GUI to avoid leakage."
        )
        guide_text = QLabel(
            "Outputs written under feature_analysis/ml_export_builder/tables:\n"
            "• acoustic_ml_ready.csv and acoustic_features_only.csv\n"
            "• kinematic_ml_ready.csv and kinematic_features_only.csv\n"
            "• multimodal_early_fusion_ml_ready.csv when a safe shared key exists\n"
            "• feature_manifest_unified.csv\n"
            "• row_alignment_report.csv, row_exclusions.csv, ml_export_manifest.json"
        )
        guide_text.setWordWrap(True)
        guide_text.setStyleSheet(f"color:{MUTED}; background:#FFFFFF; border:none; padding:4px;")
        guide.layout.addWidget(guide_text)

        layout.addWidget(card)
        layout.addWidget(guide)
        return self._wrap_scroll(body)

    def pick_ml_export_output_folder(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Select Feature-GUI ML export output folder")
        if p:
            self.ml_export_output_edit.setText(p)

    def build_feature_ml_export_package(self) -> None:
        try:
            out = self.ml_export_output_edit.text().strip() or self.output_edit.text().strip()
            if not out:
                QMessageBox.warning(self, "Missing output folder", "Select an output folder for the ML export package.")
                return
            result = build_ml_export_package(
                output_root=out,
                acoustic_features=self.ml_acoustic_features_picker.path or None,
                acoustic_registry=self.ml_acoustic_registry_picker.path or None,
                acoustic_scale_registry=self.ml_acoustic_scale_picker.path or None,
                kinematic_features=self.ml_kinematic_features_picker.path or None,
                kinematic_manifest=self.ml_kinematic_manifest_picker.path or None,
                metadata=self.ml_metadata_picker.path or None,
            )
            self.latest_ml_export_dir = result.output_dir
            summary_rows = [
                {"metric": k, "value": json.dumps(v) if isinstance(v, (list, dict)) else v}
                for k, v in result.summary.items()
                if k not in {"outputs", "notes"}
            ]
            for idx, note in enumerate(result.summary.get("notes", []), start=1):
                summary_rows.append({"metric": f"note_{idx}", "value": note})
            self._fill_table(self.ml_export_summary_table, pd.DataFrame(summary_rows))
            self._fill_table(self.ml_export_overlap_table, result.tables.get("row_alignment_report", pd.DataFrame()))
            self._fill_table(self.ml_export_manifest_table, result.tables.get("feature_manifest_unified", pd.DataFrame()), max_rows=1000)
            self._fill_table(self.ml_export_acoustic_table, result.tables.get("acoustic_ml_ready", pd.DataFrame()), max_rows=200)
            self._fill_table(self.ml_export_kinematic_table, result.tables.get("kinematic_ml_ready", pd.DataFrame()), max_rows=200)
            self._fill_table(self.ml_export_fusion_table, result.tables.get("multimodal_early_fusion_ml_ready", pd.DataFrame()), max_rows=200)
            self._fill_table(self.ml_export_exclusions_table, result.tables.get("row_exclusions", pd.DataFrame()))
            self.ml_export_status.append(f"ML export package created: {result.output_dir}")
            for name, path in result.paths.items():
                self.ml_export_status.append(f"{name}: {path}")
            QMessageBox.information(self, "ML export complete", f"ML export package created:\n{result.output_dir}")
        except Exception as exc:
            QMessageBox.critical(self, "ML export failed", str(exc))

    def open_ml_export_folder(self) -> None:
        path = getattr(self, "latest_ml_export_dir", None)
        if not path:
            p = self.ml_export_output_edit.text().strip() if hasattr(self, "ml_export_output_edit") else ""
            if p:
                candidate = Path(p) / "feature_analysis" / "ml_export_builder"
                path = candidate if candidate.exists() else Path(p)
        if not path:
            QMessageBox.information(self, "No ML export yet", "Build the ML export package first.")
            return
        self.open_file(Path(path))


    def _export_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        card = Card(
            "Export / Report",
            "Final packaging screen. Review feature-readiness decisions, choose an export profile, write ML-ready tables, and generate a professional HTML report. No model is trained here."
        )

        self.export_metric_grid = QGridLayout()
        self.export_metric_grid.setHorizontalSpacing(12)
        self.export_metric_grid.setVerticalSpacing(12)
        card.layout.addLayout(self.export_metric_grid)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("Export profile:"))
        self.export_profile_combo = QComboBox()
        self.export_profile_combo.addItems([
            "Recommended + caution (default ML starting set)",
            "Recommended only (strict)",
            "Review set (recommended + caution + review)",
            "Full audit set (all features; not ML default)",
        ])
        self.export_profile_combo.setMinimumWidth(430)
        self.export_profile_combo.currentTextChanged.connect(lambda _=None: self.update_export_dashboard(getattr(self, "outputs", {})))
        control_row.addWidget(self.export_profile_combo)
        control_row.addStretch(1)
        refresh_btn = QPushButton("Refresh Preview")
        refresh_btn.setProperty("secondary", True)
        refresh_btn.clicked.connect(lambda: self.update_export_dashboard(getattr(self, "outputs", {})))
        control_row.addWidget(refresh_btn)
        package_btn = QPushButton("Create Export Package")
        package_btn.clicked.connect(self.create_export_package)
        control_row.addWidget(package_btn)
        card.layout.addLayout(control_row)

        guidance = QLabel(
            "Use the profile selector to choose how conservative the exported feature matrix should be. Green features are clean default candidates; light-green features are usable with documented caution; gold/orange/red features should be reviewed, recomputed, or held out by default. The exported manifest always explains why each feature was included or held."
        )
        guidance.setWordWrap(True)
        guidance.setStyleSheet(f"color:{INK}; background:#F7FAFD; border:1px solid {LINE}; border-radius:10px; padding:10px;")
        card.layout.addWidget(guidance)

        splitter = QSplitter(Qt.Horizontal)
        left = QFrame(); left.setStyleSheet("QFrame { border:none; background:transparent; }")
        left_layout = QVBoxLayout(left); left_layout.setContentsMargins(0,0,0,0); left_layout.setSpacing(8)
        self.export_interpretation = QTextEdit()
        self.export_interpretation.setReadOnly(True)
        self.export_interpretation.setMinimumHeight(210)
        left_layout.addWidget(self.export_interpretation)
        self.export_status = QTextEdit()
        self.export_status.setReadOnly(True)
        self.export_status.setMinimumHeight(170)
        self.export_status.setPlaceholderText("Export status messages will appear here.")
        left_layout.addWidget(self.export_status)
        splitter.addWidget(left)

        right = QFrame(); right.setStyleSheet("QFrame { border:none; background:transparent; }")
        right_layout = QVBoxLayout(right); right_layout.setContentsMargins(0,0,0,0); right_layout.setSpacing(8)
        self.export_plot_preview = QLabel("Run Feature Analysis, then use Export / Report to preview the manifest summary.")
        self.export_plot_preview.setAlignment(Qt.AlignCenter)
        self.export_plot_preview.setMinimumHeight(360)
        self.export_plot_preview.setStyleSheet(f"QLabel {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; color:{MUTED}; padding:16px; }}")
        right_layout.addWidget(self.export_plot_preview, 1)
        plot_buttons = QHBoxLayout()
        open_plot_btn = QPushButton("Open Current Plot")
        open_plot_btn.setProperty("secondary", True)
        open_plot_btn.clicked.connect(self.open_current_export_plot)
        plot_buttons.addWidget(open_plot_btn)
        open_report_btn = QPushButton("Open HTML Report")
        open_report_btn.setProperty("secondary", True)
        open_report_btn.clicked.connect(self.open_html_report)
        plot_buttons.addWidget(open_report_btn)
        open_folder_btn = QPushButton("Open Output Folder")
        open_folder_btn.setProperty("secondary", True)
        open_folder_btn.clicked.connect(self.open_output_folder)
        plot_buttons.addWidget(open_folder_btn)
        right_layout.addLayout(plot_buttons)
        splitter.addWidget(right)
        splitter.setSizes([430, 720])
        card.layout.addWidget(splitter)

        tabs = QTabWidget()
        self.export_manifest_table = self._simple_table()
        self.export_summary_table = self._simple_table()
        self.export_profile_table = self._simple_table()
        self.export_readme_table = self._simple_table()
        tabs.addTab(self.export_manifest_table, "Color-coded feature manifest")
        tabs.addTab(self.export_summary_table, "Readiness summary")
        tabs.addTab(self.export_profile_table, "Exported tables")
        tabs.addTab(self.export_readme_table, "Decision legend")
        card.layout.addWidget(tabs)

        layout.addWidget(card)
        return self._wrap_scroll(body)

    def _export_profile_key(self) -> str:
        label = self.export_profile_combo.currentText() if hasattr(self, "export_profile_combo") else ""
        if label.startswith("Recommended only"):
            return "recommended_only"
        if label.startswith("Review set"):
            return "review_set"
        if label.startswith("Full audit"):
            return "full_audit"
        return "recommended_plus_caution"

    def _export_profile_label(self, profile: str) -> str:
        return {
            "recommended_only": "Recommended only (strict)",
            "recommended_plus_caution": "Recommended + caution (default ML starting set)",
            "review_set": "Review set (recommended + caution + review)",
            "full_audit": "Full audit set (all features; not ML default)",
        }.get(profile, profile)

    def _selected_export_features(self, recs: pd.DataFrame, profile: str) -> pd.Series:
        if recs is None or recs.empty or "feature" not in recs.columns:
            return pd.Series(dtype=bool)
        r = recs.get("readiness_recommendation", pd.Series([""] * len(recs))).astype(str)
        if profile == "recommended_only":
            return r.eq("recommended")
        if profile == "review_set":
            return r.isin(["recommended", "recommended_with_caution", "review_before_use"])
        if profile == "full_audit":
            return pd.Series([True] * len(recs), index=recs.index)
        # Default: export clean and caution features. Prefer manifest flag when available.
        if "ml_export_default" in recs.columns:
            return recs["ml_export_default"].astype(str).str.lower().isin(["true", "1", "yes"]) | r.isin(["recommended", "recommended_with_caution"])
        return r.isin(["recommended", "recommended_with_caution"])

    def _export_decision_legend(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"color": "green", "label": "recommended", "meaning": "Clean default candidate", "action": "Include in default export."},
            {"color": "light green", "label": "recommended_with_caution", "meaning": "Usable with documented risk", "action": "Include in default export, but carry reasons forward."},
            {"color": "gold", "label": "review_before_use", "meaning": "Manual review needed", "action": "Hold from default ML export unless justified."},
            {"color": "orange", "label": "exclude_or_recompute", "meaning": "Technical/scientific issue likely", "action": "Recompute, sensitivity-check, or hold out."},
            {"color": "red", "label": "exclude_by_default", "meaning": "Major support problem", "action": "Exclude by default; retain only for audit."},
        ])

    def _build_export_preview_tables(self, outputs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        recs = outputs.get("feature_recommendations", pd.DataFrame()).copy() if outputs else pd.DataFrame()
        profile = self._export_profile_key()
        if recs.empty:
            manifest = pd.DataFrame(columns=["feature", "final_include", "readiness_recommendation", "readiness_score", "primary_reasons", "recommended_action"])
        else:
            mask = self._selected_export_features(recs, profile)
            manifest = recs.copy()
            manifest["export_profile"] = self._export_profile_label(profile)
            manifest["final_include"] = mask.values if len(mask) == len(manifest) else False
            manifest["export_decision"] = manifest["final_include"].map({True: "include", False: "hold / exclude"})
            keep_cols = [c for c in [
                "feature", "final_include", "export_decision", "readiness_recommendation", "readiness_score",
                "family_or_subsystem", "primary_reasons", "recommended_action", "missing_fraction",
                "robust_outlier_fraction", "max_abs_qc_spearman", "max_abs_redundancy", "icc1_proxy",
                "max_screening_effect", "export_profile"
            ] if c in manifest.columns]
            manifest = manifest[keep_cols].sort_values(["final_include", "readiness_score"], ascending=[False, False])

        n_total = int(len(recs))
        n_inc = int(manifest["final_include"].astype(bool).sum()) if "final_include" in manifest.columns else 0
        n_hold = int(n_total - n_inc)
        profile_table = pd.DataFrame([
            {"export_profile": self._export_profile_label(profile), "n_total_features": n_total, "n_included_features": n_inc, "n_held_or_excluded_features": n_hold, "purpose": "Choose the feature matrix breadth before downstream ML."},
            {"export_profile": "recommended_only", "n_total_features": n_total, "n_included_features": int((recs.get("readiness_recommendation", pd.Series(dtype=str)).astype(str) == "recommended").sum()) if not recs.empty else 0, "n_held_or_excluded_features": "—", "purpose": "Strictest clean set."},
            {"export_profile": "recommended_plus_caution", "n_total_features": n_total, "n_included_features": int(recs.get("readiness_recommendation", pd.Series(dtype=str)).astype(str).isin(["recommended", "recommended_with_caution"]).sum()) if not recs.empty else 0, "n_held_or_excluded_features": "—", "purpose": "Default starting set for ML."},
            {"export_profile": "review_set", "n_total_features": n_total, "n_included_features": int(recs.get("readiness_recommendation", pd.Series(dtype=str)).astype(str).isin(["recommended", "recommended_with_caution", "review_before_use"]).sum()) if not recs.empty else 0, "n_held_or_excluded_features": "—", "purpose": "Broad sensitivity/review set."},
        ])
        summary = outputs.get("feature_recommendation_summary", pd.DataFrame()).copy() if outputs else pd.DataFrame()
        return manifest, summary, profile_table

    def update_export_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "export_metric_grid"):
            return
        outputs = outputs or getattr(self, "outputs", {}) or {}
        manifest, summary, profile_table = self._build_export_preview_tables(outputs)
        while self.export_metric_grid.count():
            item = self.export_metric_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        recs = outputs.get("feature_recommendations", pd.DataFrame()) if outputs else pd.DataFrame()
        included = int(manifest["final_include"].astype(bool).sum()) if "final_include" in manifest.columns else 0
        total = int(len(recs))
        held = max(total - included, 0)
        strict = int(recs.get("readiness_recommendation", pd.Series(dtype=str)).astype(str).eq("recommended").sum()) if not recs.empty else 0
        caution = int(recs.get("readiness_recommendation", pd.Series(dtype=str)).astype(str).eq("recommended_with_caution").sum()) if not recs.empty else 0
        review = int(recs.get("readiness_recommendation", pd.Series(dtype=str)).astype(str).eq("review_before_use").sum()) if not recs.empty else 0
        excluded = int(recs.get("readiness_recommendation", pd.Series(dtype=str)).astype(str).isin(["exclude_or_recompute", "exclude_by_default"]).sum()) if not recs.empty else 0
        tiles = [
            ("Profile", self._export_profile_label(self._export_profile_key()).split("(")[0].strip(), "selected export rule"),
            ("Included", included, "features in exported matrix"),
            ("Held", held, "review/exclude/audit only"),
            ("Recommended", strict, "green"),
            ("Caution", caution, "light green"),
            ("Review", review, "gold"),
            ("Excluded", excluded, "orange/red"),
            ("Total", total, "feature recommendations"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.export_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx // 4, idx % 4)
        self._fill_export_manifest_table(self.export_manifest_table, manifest)
        self._fill_table(self.export_summary_table, summary)
        self._fill_table(self.export_profile_table, profile_table)
        self._fill_table(self.export_readme_table, self._export_decision_legend())
        self.export_interpretation.setHtml(self._export_interpretation_html(included, held, total))
        path = None
        if hasattr(self, "plot_paths"):
            path = self.plot_paths.get("ml_export_manifest_summary") or self.plot_paths.get("recommendation_counts")
        if path and Path(path).exists():
            self.current_export_plot = Path(path)
            pix = QPixmap(str(path))
            if not pix.isNull():
                self.export_plot_preview.setPixmap(pix.scaled(self.export_plot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.export_plot_preview.setToolTip(str(path))

    def _export_interpretation_html(self, included: int, held: int, total: int) -> str:
        profile = self._export_profile_label(self._export_profile_key())
        return f"""
        <b>Export profile</b><br>{profile}<br><br>
        <b>What this stage does</b><br>Creates transparent analysis and ML-preparation files from the completed Feature Analysis audit. The feature matrix includes <b>{included}</b> of <b>{total}</b> features under the selected rule; <b>{held}</b> features are held for review, recomputation, exclusion, or audit-only use.<br><br>
        <b>Why include a feature?</b><br>Green/light-green features have adequate support across missingness, distribution shape, QC sensitivity, redundancy, reliability, and integrated recommendation score. Caution features are included only because their risks are documented and can be handled later in sensitivity analyses or fold-safe ML pipelines.<br><br>
        <b>Why exclude or hold a feature?</b><br>Gold/orange/red features may have high missingness, zero/near-zero variance, implausible distribution, strong QC sensitivity, excessive outliers, redundancy, or insufficient repeatability. Holding them prevents silent leakage of low-quality measurements into modelling.<br><br>
        <b>Important boundary</b><br>This export is not final ML feature selection. The future ML GUI must still perform imputation, scaling, transformations, feature selection, and validation inside cross-validation or training folds.
        """

    def _fill_export_manifest_table(self, table: QTableWidget, df: pd.DataFrame, max_rows: int = 1000) -> None:
        self._fill_table(table, df, max_rows=max_rows)
        if df is None or df.empty:
            return
        rec_col = list(df.columns).index("readiness_recommendation") if "readiness_recommendation" in df.columns else None
        inc_col = list(df.columns).index("final_include") if "final_include" in df.columns else None
        colors = {
            "recommended": "#DDF6E8",
            "recommended_with_caution": "#EEF8DD",
            "review_before_use": "#FFF4D6",
            "exclude_or_recompute": "#FFE2C2",
            "exclude_by_default": "#FDE2DF",
        }
        for i in range(table.rowCount()):
            rec = table.item(i, rec_col).text() if rec_col is not None and table.item(i, rec_col) else ""
            include = table.item(i, inc_col).text().lower() in ["true", "1", "yes"] if inc_col is not None and table.item(i, inc_col) else False
            bg = QColor(colors.get(rec, "#FFFFFF"))
            for j in range(table.columnCount()):
                item = table.item(i, j)
                if item:
                    item.setBackground(QBrush(bg))
                    item.setForeground(QBrush(QColor(INK)))
                    if j == inc_col:
                        item.setText("INCLUDE" if include else "HOLD")
                        item.setForeground(QBrush(QColor("#047857" if include else RED)))

    def _make_export_package_tables(self) -> tuple[Path, dict[str, Path]]:
        if self.feature_df is None:
            self.load_and_map()
        if self.feature_df is None:
            raise RuntimeError("Load a primary feature table before exporting.")
        if not getattr(self, "outputs", None):
            outputs, feature_cols = self._build_analysis_outputs()
            self.outputs = outputs
        outputs = self.outputs
        profile = self._export_profile_key()
        profile_slug = profile.replace("_", "-")
        base_dir = (self.output_dir if getattr(self, "output_dir", None) else Path(self.output_edit.text().strip()) / "feature_analysis")
        export_dir = base_dir / "exports" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{profile_slug}"
        export_dir.mkdir(parents=True, exist_ok=True)
        manifest, summary, profile_table = self._build_export_preview_tables(outputs)
        mapping = self.collect_mapping_from_table()
        roles = role_lists(mapping)
        id_cols = [c for c in roles.get(ROLE_IDENTIFIER, []) if c in self.feature_df.columns]
        target_cols = [c for c in roles.get(ROLE_TARGET, []) if c in self.feature_df.columns]
        cov_cols = [c for c in roles.get(ROLE_COVARIATE, []) if c in self.feature_df.columns]
        qc_cols = [c for c in roles.get(ROLE_QC, []) if c in self.feature_df.columns]
        include_features = manifest.loc[manifest.get("final_include", False).astype(bool), "feature"].astype(str).tolist() if not manifest.empty and "feature" in manifest.columns else []
        include_features = [c for c in include_features if c in self.feature_df.columns]
        paths: dict[str, Path] = {}
        def write_df(name: str, df: pd.DataFrame) -> None:
            out = export_dir / name
            df.to_csv(out, index=False)
            paths[name] = out
        write_df("feature_export_manifest.csv", manifest)
        write_df("feature_recommendation_summary.csv", summary)
        write_df("export_profile_summary.csv", profile_table)
        write_df("feature_recommendation_legend.csv", self._export_decision_legend())
        matrix_cols = id_cols + include_features
        write_df("ml_ready_feature_matrix.csv", self.feature_df[matrix_cols].copy() if matrix_cols else pd.DataFrame())
        write_df("ml_target_table.csv", self.feature_df[id_cols + target_cols].copy() if target_cols else pd.DataFrame(columns=id_cols))
        write_df("ml_covariate_table.csv", self.feature_df[id_cols + cov_cols].copy() if cov_cols else pd.DataFrame(columns=id_cols))
        write_df("ml_qc_covariate_table_from_feature_table.csv", self.feature_df[id_cols + qc_cols].copy() if qc_cols else pd.DataFrame(columns=id_cols))
        if self.qc_df is not None:
            write_df("linked_qc_table.csv", self.qc_df.copy())
        if self.meta_df is not None:
            write_df("linked_metadata_table.csv", self.meta_df.copy())
        config = {
            "app_version": APP_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "export_profile": self._export_profile_label(profile),
            "n_total_features": int(len(outputs.get("feature_recommendations", pd.DataFrame()))),
            "n_included_features": int(len(include_features)),
            "feature_table": self.feature_picker.path,
            "qc_table": self.qc_picker.path,
            "metadata_table": self.meta_picker.path,
            "registry_table": self.registry_picker.path,
            "boundary": "Feature Analysis export only; no model trained; ML preprocessing and feature selection must be fold-safe.",
        }
        (export_dir / "export_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        readme = f"""# VSLP Feature Analysis Export Package

Export profile: {self._export_profile_label(profile)}

Included feature count: {len(include_features)}

This package is a downstream-analysis starting point, not a trained model and not final ML feature selection. The manifest documents why each feature was included or held. Imputation, scaling, transformations, feature selection, model fitting, and validation must occur inside the future ML pipeline to avoid leakage.

Core files:
- `ml_ready_feature_matrix.csv`: identifiers plus selected feature columns.
- `ml_target_table.csv`: identifiers plus mapped target/outcome columns, if available.
- `ml_covariate_table.csv`: identifiers plus mapped covariates, if available.
- `ml_qc_covariate_table_from_feature_table.csv`: identifiers plus QC columns from the feature table, if available.
- `linked_qc_table.csv`: optional externally loaded QC table.
- `feature_export_manifest.csv`: color-coded decision logic in table form.
- `export_config.json`: reproducibility metadata.

Decision colors:
- green = recommended, include by default.
- light green = recommended with caution, include but document risk.
- gold = review before use, hold from default ML unless justified.
- orange/red = exclude, recompute, or audit-only by default.
"""
        (export_dir / "README.md").write_text(readme, encoding="utf-8")
        report_path = export_dir / "vslp_feature_analysis_export_report.html"
        self.write_report(report_path, outputs, export_manifest=manifest, export_profile=self._export_profile_label(profile))
        zip_base = shutil.make_archive(str(export_dir), "zip", root_dir=export_dir)
        paths["export_zip"] = Path(zip_base)
        return export_dir, paths

    def create_export_package(self) -> None:
        try:
            export_dir, paths = self._make_export_package_tables()
            self.export_status.append(f"Export package created: {export_dir}")
            if "export_zip" in paths:
                self.export_status.append(f"Zip package: {paths['export_zip']}")
            self.update_export_dashboard(getattr(self, "outputs", {}))
            QMessageBox.information(self, "Export complete", f"Export package created:\n{export_dir}")
            self.open_file(export_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def open_current_export_plot(self) -> None:
        path = getattr(self, "current_export_plot", None)
        if not path:
            QMessageBox.information(self, "No current plot", "Run Feature Analysis first, then preview the Export / Report page.")
            return
        self.open_file(Path(path))

    def open_html_report(self) -> None:
        if not getattr(self, "output_dir", None):
            QMessageBox.information(self, "No report yet", "Run Feature Analysis first.")
            return
        report = self.output_dir / "reports" / "vslp_feature_analysis_report.html"
        if not report.exists():
            QMessageBox.information(self, "No report yet", "Run Feature Analysis first to write the HTML report.")
            return
        self.open_file(report)

    def pick_output_folder(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Select output folder")
        if p:
            self.output_edit.setText(p)

    def log(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{stamp}] {text}"
        for name in ("run_log", "project_status", "export_status"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.append(msg)
        if hasattr(self, "run_progress"):
            low = text.lower()
            if "loaded feature table" in low:
                self.run_progress.setValue(20)
            elif "column mapping accepted" in low:
                self.run_progress.setValue(40)
            elif "analysis complete" in low or "export package created" in low or "ml export package created" in low:
                self.run_progress.setValue(100)

    def log_error(self, context: str, exc: Exception) -> None:
        self.log(f"ERROR · {context}: {exc}")

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
            self.log_error("Load failed", exc)
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
            self.update_relationships_dashboard(outputs)
            self.update_screening_dashboard(outputs)
            self.update_reliability_dashboard(outputs)
            self.update_recommendations_dashboard(outputs)
            self.update_export_dashboard(outputs)
            self.log(f"Analysis complete. Outputs written to: {self.output_dir}")
            self.show_page("overview")
        except Exception as exc:
            self.log_error("Analysis failed", exc)
            QMessageBox.critical(self, "Analysis failed", str(exc))

    def populate_output_tables(self, outputs: dict[str, pd.DataFrame]) -> None:
        targets = {
            "overview": "dataset_inventory",
            "missing": "missingness_by_feature",
            "distributions___outliers": "feature_distribution_summary",
            "qc_integration": "feature_qc_spearman_correlation",
            "feature_relationships": "feature_relationship_summary",
            "group___outcome_screening": "screening_summary",
            "reliability": "feature_repeatability_summary",
            "recommendations": "feature_recommendations",
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
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setSortingEnabled(True)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.horizontalHeader().setStretchLastSection(False)
        table.resizeRowsToContents()
        table.resizeColumnsToContents()

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
        rpath = getattr(self, "current_relationship_plot", None)
        if rpath and hasattr(self, "relationship_plot_preview"):
            pix = QPixmap(str(rpath))
            if not pix.isNull():
                self.relationship_plot_preview.setPixmap(pix.scaled(self.relationship_plot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        spath = getattr(self, "current_screening_plot", None)
        if spath and hasattr(self, "screening_plot_preview"):
            pix = QPixmap(str(spath))
            if not pix.isNull():
                self.screening_plot_preview.setPixmap(pix.scaled(self.screening_plot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        relpath = getattr(self, "current_reliability_plot", None)
        if relpath and hasattr(self, "reliability_plot_preview"):
            pix = QPixmap(str(relpath))
            if not pix.isNull():
                self.reliability_plot_preview.setPixmap(pix.scaled(self.reliability_plot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        recpath = getattr(self, "current_recommendation_plot", None)
        if recpath and hasattr(self, "recommendation_plot_preview"):
            pix = QPixmap(str(recpath))
            if not pix.isNull():
                self.recommendation_plot_preview.setPixmap(pix.scaled(self.recommendation_plot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        epath = getattr(self, "current_export_plot", None)
        if epath and hasattr(self, "export_plot_preview"):
            pix = QPixmap(str(epath))
            if not pix.isNull():
                self.export_plot_preview.setPixmap(pix.scaled(self.export_plot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def write_report(self, path: Path, outputs: dict[str, pd.DataFrame], export_manifest: pd.DataFrame | None = None, export_profile: str = "Default export") -> None:
        inv = outputs.get("dataset_inventory", pd.DataFrame()).to_html(index=False, escape=False)
        roles = outputs.get("feature_role_summary", summarize_roles(outputs.get("feature_column_mapping", pd.DataFrame()))).to_html(index=False, escape=False)
        design = outputs.get("dataset_design_overview", pd.DataFrame()).to_html(index=False, escape=False)
        fam = outputs.get("feature_family_overview", pd.DataFrame()).to_html(index=False, escape=False)
        miss = outputs.get("missingness_by_feature", pd.DataFrame()).head(80).to_html(index=False, escape=False)
        missgrp = outputs.get("missingness_by_group", pd.DataFrame()).head(80).to_html(index=False, escape=False)
        distreview = outputs.get("distribution_review_summary", pd.DataFrame()).head(100).to_html(index=False, escape=False)
        outrows = outputs.get("robust_outlier_flags", pd.DataFrame()).head(100).to_html(index=False, escape=False)
        rel = outputs.get("feature_reliability_screen", pd.DataFrame()).head(80).to_html(index=False, escape=False)
        recs = outputs.get("feature_recommendations", pd.DataFrame()).head(120).to_html(index=False, escape=False)
        manifest_df = export_manifest if export_manifest is not None else outputs.get("ml_export_manifest", pd.DataFrame())
        manifest = manifest_df.head(300).to_html(index=False, escape=False)
        rec_summary = outputs.get("feature_recommendation_summary", pd.DataFrame()).to_html(index=False, escape=False)
        qc = outputs.get("qc_integration_summary", pd.DataFrame()).to_html(index=False, escape=False)
        relationships = outputs.get("feature_relationship_summary", pd.DataFrame()).to_html(index=False, escape=False)
        html = f"""<!doctype html><html><head><meta charset='utf-8'><title>VSLP Feature Analysis Report</title>
        <style>
        body{{font-family:Arial,sans-serif;margin:32px;color:#0E1726;background:#F7FAFD}}
        .page{{max-width:1280px;margin:auto;background:white;padding:30px;border:1px solid #D9E2EF;border-radius:16px}}
        h1,h2,h3{{color:#071A33}}
        .note{{background:#EEF8FF;border-left:5px solid #2DB7B0;padding:12px;margin:16px 0;border-radius:8px}}
        .warn{{background:#FFF4D6;border-left:5px solid #B68B2D;padding:12px;margin:16px 0;border-radius:8px}}
        table{{border-collapse:collapse;width:100%;font-size:12px;margin-bottom:24px}}
        td,th{{border:1px solid #D9E2EF;padding:6px;vertical-align:top}} th{{background:#EEF4FA;color:#071A33}}
        tr:has(td:nth-child(4):contains('recommended')){{background:#DDF6E8}}
        </style></head><body><div class='page'>
        <h1>VSLP Feature Analysis Report</h1>
        <p><b>Version:</b> {APP_VERSION}. <b>Export profile:</b> {export_profile}.</p>
        <div class='note'>This report is a descriptive feature audit and export-preparation report. It is not a diagnostic report and no machine-learning model has been trained.</div>
        <div class='warn'>Imputation, scaling, transformation, feature selection, and model training must occur inside the downstream ML pipeline to avoid data leakage.</div>
        <h2>Export decision summary</h2>{rec_summary}
        <h2>Dataset inventory</h2>{inv}
        <h2>Column-role summary</h2>{roles}
        <h2>Dataset design overview</h2>{design}
        <h2>Feature-family overview</h2>{fam}
        <h2>Missingness by feature</h2>{miss}
        <h2>Missingness by group</h2>{missgrp}
        <h2>Distribution / outlier review</h2>{distreview}
        <h2>Row-level outlier flags</h2>{outrows}
        <h2>QC integration summary</h2>{qc}
        <h2>Feature relationship summary</h2>{relationships}
        <h2>Initial feature reliability screen</h2>{rel}
        <h2>Integrated feature recommendations</h2>{recs}
        <h2>Export manifest preview</h2>{manifest}
        </div></body></html>"""
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
