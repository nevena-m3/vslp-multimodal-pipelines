"""VSLP Feature Analysis GUI.

Professional, modality-neutral feature audit interface for acoustic, kinematic,
multimodal, and generic feature tables.
"""
from __future__ import annotations

import sys
import json
import shutil
import re
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
    plot_ml_export_manifest_summary,
    plot_task_counts, plot_task_subject_matrix, plot_task_label_context,
    plot_task_feature_support, plot_task_clinical_context, plot_task_clinical_filtered_counts, plot_longitudinal_subject_records,
    plot_longitudinal_session_matrix, plot_longitudinal_iteration_counts,
    plot_longitudinal_date_timeline
)

APP_VERSION = "v0.73.0"

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
        sub = QLabel("Feature audit | QC integration | reliability screening | export")
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
        credit = QLabel("(c) 2026 Nevena Musikic & Yana Yunusova\nSpeech Production Lab\nUniversity of Toronto")
        credit.setObjectName("Credit")
        credit.setWordWrap(True)
        layout.addWidget(brand)
        layout.addWidget(sub)
        layout.addSpacing(8)
        layout.addWidget(credit)
        layout.addSpacing(18)
        for key, text in [
            ("project", "o  Project"),
            ("mapping", "o  Column Mapping"),
            ("overview", "o  Overview"),
            ("missing", "o  Missingness"),
            ("dist", "o  Distributions"),
            ("qc", "o  QC Integration"),
            ("relationships", "o  Feature Relationships"),
            ("task_review", "o  Task Review"),
            ("longitudinal", "o  Longitudinal / Iterations"),
            ("screening", "o  Group / Outcome Screening"),
            ("reliability", "o  Reliability"),
            ("recommendations", "o  Recommendations"),
            ("ml_export", "o  ML Export Builder"),
            ("export", "o  Export / Report"),
        ]:
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, k=key: self.on_select(k))
            layout.addWidget(b)
            self.buttons[key] = b
        layout.addStretch(1)

    def set_active(self, key: str) -> None:
        for k, b in self.buttons.items():
            b.setProperty("active", k == key)
            prefix = "*" if k == key else "o"
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




class NoWheelComboBox(QComboBox):
    """Combo box that cannot change roles from accidental mouse-wheel scrolling.

    Role changes should be deliberate clicks/keyboard edits, not incidental table
    scrolling while the cursor is over the Role column.
    """

    def wheelEvent(self, event):  # noqa: N802 - Qt override
        event.ignore()


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
        self.proposed_mapping_df = pd.DataFrame()
        self.analysis_df: Optional[pd.DataFrame] = None
        self.metadata_join_strategy = "feature_table_only"
        self.mapping_modified = False
        self.mapping_accepted = False
        self.outputs: dict[str, pd.DataFrame] = {}
        self.output_dir: Optional[Path] = None
        self.page_keys = ["project", "mapping", "overview", "missing", "dist", "qc", "relationships", "task_review", "longitudinal", "screening", "reliability", "recommendations", "ml_export", "export"]

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
            "task_review": self._task_review_page(),
            "longitudinal": self._longitudinal_page(),
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
        title = QLabel(f"Run Log | Feature Analysis GUI {APP_VERSION}")
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
        self.mapping_table.setHorizontalHeaderLabels(["Column", "Role", "Confidence", "Reason / rationale", "dtype", "Missing", "Unique"])
        self.mapping_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.mapping_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.mapping_table.setAlternatingRowColors(True)
        self.mapping_table.setWordWrap(False)
        self.mapping_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.mapping_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.mapping_table.verticalHeader().setDefaultSectionSize(30)
        header = self.mapping_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        self.mapping_table.setColumnWidth(0, 285)
        self.mapping_table.setColumnWidth(1, 150)
        self.mapping_table.setColumnWidth(2, 85)
        self.mapping_table.setColumnWidth(3, 520)
        self.mapping_table.setColumnWidth(4, 115)
        self.mapping_table.setColumnWidth(5, 85)
        self.mapping_table.setColumnWidth(6, 85)
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
            "High-level dataset orientation. This page summarizes design context, column roles, feature-family coverage, and first-pass feature quality without duplicating the deeper Missingness, Distribution, QC, or ML Export menus."
        )

        self.overview_note = QLabel("Load tables, review/accept column mapping, then run Feature Analysis to populate this dashboard.")
        self.overview_note.setWordWrap(True)
        self.overview_note.setStyleSheet(f"color:{MUTED}; background:#F7FAFD; border:1px solid {LINE}; border-radius:8px; padding:10px;")
        card.layout.addWidget(self.overview_note)

        overview_split = QHBoxLayout()
        overview_split.setSpacing(14)

        # Main visual area: one plot selector toolbar, one large preview.  Overview only
        # contains non-redundant orientation plots; detailed missingness/distribution/QC
        # plots live in their own menus.
        plot_panel = QFrame()
        plot_panel.setStyleSheet(f"QFrame {{ background:#F8FBFE; border:1px solid {LINE}; border-radius:12px; }}")
        plot_panel_layout = QVBoxLayout(plot_panel)
        plot_panel_layout.setContentsMargins(14, 14, 14, 14)
        plot_panel_layout.setSpacing(10)

        plot_header = QHBoxLayout()
        plot_header.setSpacing(10)
        plot_title = QLabel("Overview plot")
        plot_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        plot_header.addWidget(plot_title)

        self.overview_plot_combo = QComboBox()
        self.overview_plot_combo.setMinimumWidth(360)
        self.overview_plot_combo.addItems([
            "Readiness scorecard",
            "Design context",
            "Role mapping summary",
            "Feature-family coverage",
            "Feature-quality landscape",
            "Subject x task coverage",
        ])
        plot_header.addWidget(self.overview_plot_combo, 1)

        show_btn = QPushButton("Show")
        show_btn.setProperty("secondary", True)
        show_btn.clicked.connect(self.preview_selected_overview_plot)
        plot_header.addWidget(show_btn)

        regen = QPushButton("Regenerate")
        regen.clicked.connect(self.regenerate_missingness_scope_plots)
        plot_header.addWidget(regen)

        plot_header.addStretch(1)

        open_current = QPushButton("Open current plot")
        open_current.setProperty("secondary", True)
        open_current.clicked.connect(self.open_current_overview_plot)
        plot_header.addWidget(open_current)
        plot_panel_layout.addLayout(plot_header)

        self.overview_plot_caption = QLabel("Overview uses a deliberately small plot set: readiness, design context, role mapping, feature-family coverage, feature quality, and subjectxtask coverage. Missingness, distribution, QC, and ML-export plots are handled in their own menus.")
        self.overview_plot_caption.setWordWrap(True)
        self.overview_plot_caption.setStyleSheet(f"color:{MUTED}; background:#FFFFFF; border:1px solid {LINE}; border-radius:8px; padding:9px;")
        plot_panel_layout.addWidget(self.overview_plot_caption)

        self.overview_plot_preview = QLabel("Run Feature Analysis, then choose one overview plot.")
        self.overview_plot_preview.setAlignment(Qt.AlignCenter)
        self.overview_plot_preview.setMinimumHeight(520)
        self.overview_plot_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.overview_plot_preview.setStyleSheet(f"QLabel {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; color:{MUTED}; padding:16px; }}")
        plot_panel_layout.addWidget(self.overview_plot_preview, 1)
        overview_split.addWidget(plot_panel, 1)

        # Side summary keeps metric tiles visible without occupying the top of the page.
        side_panel = QFrame()
        side_panel.setStyleSheet(f"QFrame {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:12px; }}")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        side_title = QLabel("Dataset snapshot")
        side_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        side_layout.addWidget(side_title)
        side_note = QLabel("Compact orientation metrics. Values use the accepted mapping and metadata-augmented analysis table when metadata can be joined safely.")
        side_note.setWordWrap(True)
        side_note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent; font-size:12px;")
        side_layout.addWidget(side_note)
        self.overview_metric_grid = QGridLayout()
        self.overview_metric_grid.setHorizontalSpacing(8)
        self.overview_metric_grid.setVerticalSpacing(8)
        side_layout.addLayout(self.overview_metric_grid)
        side_layout.addStretch(1)
        side_panel.setFixedWidth(300)
        overview_split.addWidget(side_panel)
        card.layout.addLayout(overview_split)

        tables_header = QLabel("Detailed overview tables")
        tables_header.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; padding-top:8px;")
        card.layout.addWidget(tables_header)

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
            t.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
            t.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

        tabs.addTab(self.overview_readiness_table, "Readiness")
        tabs.addTab(self.overview_inventory_table, "Inventory")
        tabs.addTab(self.overview_roles_table, "Roles")
        tabs.addTab(self.overview_design_table, "Design variables")
        tabs.addTab(self.overview_family_table, "Feature families")
        tabs.addTab(self.overview_quality_table, "Feature quality")
        card.layout.addWidget(tabs)

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

        def metric_value(name: str, default: object = "-") -> object:
            if inv.empty or "metric" not in inv.columns:
                return default
            row = inv.loc[inv["metric"].eq(name)]
            return row["value"].iloc[0] if not row.empty else default

        def score_value(name: str, default: object = "-") -> object:
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
            ("Features", metric_value("detected_feature_columns"), "mapped predictors"),
            ("Subjects", metric_value("unique_subjects"), "metadata-aware if joined"),
            ("Tasks", metric_value("unique_tasks"), "metadata-aware if joined"),
            ("Completeness", score_value("Feature completeness"), "0-100 audit score"),
            ("Design", score_value("Design richness"), "context richness"),
            ("Review flags", n_review, f"monitor: {n_monitor}"),
            ("QC", "yes" if str(metric_value("qc_table_loaded", False)).lower() == "true" else "no", "artifact context"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.overview_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx, 0)

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

    def _normal_col_lookup(self, df: pd.DataFrame) -> dict[str, str]:
        from vslp.analysis.features.column_mapping import normalize_name
        return {normalize_name(c): str(c) for c in df.columns}

    def _canonical_metadata_name(self, column: str) -> str:
        """Map common REDCap/export names to the Feature GUI's canonical field names."""
        from vslp.analysis.features.column_mapping import normalize_name

        n = normalize_name(column)
        aliases = {
            "raw_media_file_name": "file_name",
            "media_file_name": "file_name",
            "filename": "file_name",
            "file": "file_name",
            "subjectid": "subject_id",
            "subject_id": "subject_id",
            "participant_id": "subject_id",
            "patient_id": "subject_id",
            "clinical_visit_id": "visit_id",
            "visit_id": "visit_id",
            "session_id": "session_id",
            "recording_session": "session_id",
            "iteration": "iteration",
            "recording_date": "recording_date",
            "assessment_date": "assessment_date",
            "task_name": "task",
            "task": "task",
            "diagnosis": "diagnosis",
            "sex": "sex_or_gender",
            "gender": "sex_or_gender",
            "sex_or_gender": "sex_or_gender",
            "alsfrs_total_score": "severity_score",
            "severity_score": "severity_score",
            "severity_bin": "severity_bin",
            "frame_rate": "frame_rate",
            "sampling_rate": "sampling_rate",
            "duration_s": "duration_sec",
            "duration_sec": "duration_sec",
            "protocol_id": "protocol_id",
            "extension": "extension",
            "device": "device",
            "organization_name": "site",
        }
        return aliases.get(n, n)

    def _basename_series(self, values: pd.Series) -> pd.Series:
        """Return lower-case file basenames from Windows, POSIX, or plain filename strings."""
        def one(v: object) -> str:
            if pd.isna(v):
                return ""
            s = str(v).strip().strip('"').strip("'")
            if not s:
                return ""
            s = re.split(r"[\\/]", s)[-1]
            return s.lower()
        return values.map(one)

    def _add_file_match_helpers(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        source = None
        for c in ["file_name", "source_file_path", "raw_media_file_name", "segmentation_wav_path", "video_id"]:
            if c in out.columns:
                source = c
                break
        if source is not None:
            out["_match_file_basename"] = self._basename_series(out[source])
            out["_match_file_stem"] = out["_match_file_basename"].str.replace(r"\.[a-z0-9]+$", "", regex=True)
        return out

    def _standardize_metadata_table(self, meta_df: pd.DataFrame) -> pd.DataFrame:
        """Rename common metadata columns and add match helpers without discarding originals."""
        if meta_df is None or meta_df.empty:
            return meta_df
        out = meta_df.copy()
        rename: dict[str, str] = {}
        used = set(str(c) for c in out.columns)
        for col in out.columns:
            canon = self._canonical_metadata_name(str(col))
            if canon != str(col):
                if canon not in used:
                    rename[str(col)] = canon
                    used.add(canon)
                else:
                    rename[str(col)] = f"metadata__{canon}"
        if rename:
            out = out.rename(columns=rename)
        return self._add_file_match_helpers(out)

    def _standardize_feature_match_helpers(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        return self._add_file_match_helpers(feature_df)

    def _metadata_join_candidates(self) -> list[list[str]]:
        return [
            ["record_key"],
            ["recording_id"],
            ["file_name"],
            ["_match_file_basename"],
            ["_match_file_stem"],
            ["subject_id", "session_id", "task"],
            ["subject_id", "visit_id", "task"],
            ["subject_id", "session_id"],
            ["subject_id", "visit_id"],
            ["subject_id", "task"],
            ["subject_id", "recording_date", "task"],
            ["subject_id"],
        ]

    def _is_effectively_empty(self, s: pd.Series) -> pd.Series:
        return s.isna() | s.astype(str).str.strip().isin(["", "nan", "NaN", "None", "none"])

    def _metadata_extra_columns(self, feature_df: pd.DataFrame, meta_df: pd.DataFrame, keys: list[str] | None) -> list[str]:
        keyset = set(keys or [])
        extras = []
        helper_cols = {"_match_file_basename", "_match_file_stem"}
        for col in meta_df.columns:
            if str(col) in keyset or str(col) in helper_cols:
                continue
            extras.append(str(col))
        return extras

    def _merge_metadata_context(self, feature_df: pd.DataFrame, feature_mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        """Return feature table enriched with safe metadata labels/covariates when available.

        The Feature GUI must search both the feature table and metadata table for
        subject/session/task/diagnosis/context variables. Metadata is joined only
        when a defensible key is available. When metadata fills an empty canonical
        feature-table column, the canonical column is populated so all downstream
        menus see the same information.
        """
        if self.meta_df is None or self.meta_df.empty:
            return feature_df.copy(), feature_mapping.copy(), "feature_table_only"

        feature_base = self._standardize_feature_match_helpers(feature_df)
        meta_df = self._standardize_metadata_table(self.meta_df.copy())
        feature_lookup = self._normal_col_lookup(feature_base)
        meta_lookup = self._normal_col_lookup(meta_df)

        chosen_keys: list[str] = []
        strategy = "metadata_loaded_not_merged"
        merged = feature_base.copy()
        duplicate_canonical_cols: list[tuple[str, str]] = []

        for candidate in self._metadata_join_candidates():
            if all(k in feature_lookup and k in meta_lookup for k in candidate):
                left_keys = [feature_lookup[k] for k in candidate]
                right_keys = [meta_lookup[k] for k in candidate]
                right = meta_df.copy()
                if left_keys != right_keys:
                    right = right.rename(columns={rk: lk for lk, rk in zip(left_keys, right_keys)})

                if right.duplicated(subset=left_keys).any():
                    continue

                extra_cols = self._metadata_extra_columns(merged, right, left_keys)
                rename_map: dict[str, str] = {}
                duplicate_canonical_cols = []
                for c in extra_cols:
                    if c in merged.columns:
                        new_name = f"metadata__{c}"
                        rename_map[c] = new_name
                        duplicate_canonical_cols.append((c, new_name))
                right = right[left_keys + extra_cols].rename(columns=rename_map)
                merged = merged.merge(right, on=left_keys, how="left", validate="m:1")
                chosen_keys = left_keys
                strategy = "metadata_key_join:" + "+".join(left_keys)
                break

        if not chosen_keys and len(meta_df) == len(feature_df):
            add = meta_df.reset_index(drop=True).copy()
            rename_map: dict[str, str] = {}
            duplicate_canonical_cols = []
            for c in add.columns:
                if c in {"_match_file_basename", "_match_file_stem"}:
                    rename_map[c] = f"metadata__{c}"
                    continue
                if c in merged.columns:
                    new_name = f"metadata__{c}"
                    rename_map[c] = new_name
                    duplicate_canonical_cols.append((c, new_name))
            add = add.rename(columns=rename_map)
            add = add[[c for c in add.columns if c not in merged.columns]]
            merged = pd.concat([merged.reset_index(drop=True), add], axis=1)
            strategy = "metadata_row_order_join:same_row_count"

        for canonical, metadata_col in duplicate_canonical_cols:
            if canonical in merged.columns and metadata_col in merged.columns:
                empty_mask = self._is_effectively_empty(merged[canonical])
                # Metadata may contain strings for canonical fields that pandas read as
                # all-missing float columns in the feature table. Cast before assignment
                # so filling subject/session/task/diagnosis never raises a dtype error.
                if bool(empty_mask.any()):
                    merged[canonical] = merged[canonical].astype("object")
                    fill_values = merged.loc[empty_mask, metadata_col].astype("object")
                    merged.loc[empty_mask, canonical] = fill_values

        helper_cols = [c for c in ["_match_file_basename", "_match_file_stem"] if c in merged.columns]
        if helper_cols:
            merged = merged.drop(columns=helper_cols)

        if merged.shape[1] == feature_df.shape[1] and strategy == "metadata_loaded_not_merged":
            return merged, feature_mapping.copy(), strategy

        added_cols = [c for c in merged.columns if c not in feature_df.columns]
        meta_mapping = classify_columns(merged[added_cols], table_kind="metadata", registry=None) if added_cols else pd.DataFrame()
        if not meta_mapping.empty:
            meta_mapping["reason"] = meta_mapping["reason"].astype(str) + f" Source: metadata table ({strategy})."
        combined = pd.concat([feature_mapping, meta_mapping], ignore_index=True) if not meta_mapping.empty else feature_mapping.copy()
        return merged, combined, strategy

    def mark_mapping_modified(self, *_args) -> None:
        self.mapping_modified = True
        self.mapping_accepted = False
        if hasattr(self, "mapping_summary_label"):
            self.mapping_summary_label.setText((self.mapping_summary_label.text() or "Current mapping") + "  | unsaved edits")

    def _accepted_mapping_path(self) -> Path | None:
        out = self.output_edit.text().strip() if hasattr(self, "output_edit") else ""
        if not out:
            return None
        tables_dir = Path(out) / "feature_analysis" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        return tables_dir / "accepted_column_mapping.csv"

    def save_accepted_mapping(self) -> Path | None:
        path = self._accepted_mapping_path()
        if path is None:
            return None
        self.mapping_df.to_csv(path, index=False)
        return path

    def _n_mapping_changes_from_proposal(self) -> int:
        if self.proposed_mapping_df.empty or self.mapping_df.empty:
            return 0
        left = self.proposed_mapping_df[["column", "role"]].rename(columns={"role": "proposed_role"})
        right = self.mapping_df[["column", "role"]].rename(columns={"role": "current_role"})
        comp = left.merge(right, on="column", how="outer")
        return int((comp["proposed_role"].astype(str) != comp["current_role"].astype(str)).sum())

    def _build_analysis_outputs(self) -> tuple[dict[str, pd.DataFrame], list[str]]:
        if self.feature_df is None:
            self.load_and_map()
        if self.feature_df is None:
            raise RuntimeError("Load a primary feature table first.")
        mapping = self.collect_mapping_from_table()
        active_df = self.analysis_df if self.analysis_df is not None else self.feature_df
        roles = role_lists(mapping)
        feature_cols = [c for c in roles.get("Feature", []) if c in active_df.columns]
        inventory = dataset_inventory(active_df, self.qc_df, self.meta_df, mapping)
        role_sum = role_summary(mapping)
        design = design_overview(active_df, mapping)
        family = feature_family_overview(active_df, feature_cols, self.registry_df)
        groups = group_counts(active_df)
        dist = feature_distribution_summary(active_df, feature_cols)
        outlier_flags = robust_outlier_flags(active_df, feature_cols, self.registry_df)
        range_flags = expected_range_flags(active_df, feature_cols, self.registry_df)
        dist_review = distribution_review_summary(dist, range_flags)
        shape_audit = distribution_shape_audit(dist, range_flags)
        row_outlier_burden = row_outlier_burden_summary(outlier_flags, len(feature_cols))
        feature_missing = missingness_feature_summary(active_df, feature_cols, self.registry_df)
        row_missing = missingness_row_summary(active_df, feature_cols)
        group_missing = missingness_group_summary(active_df, feature_cols)
        family_missing = missingness_family_summary(feature_missing)
        comissing = missingness_comissing_pairs(active_df, feature_cols)
        quality_landscape = overview_feature_quality_landscape(dist, self.registry_df)
        readiness = overview_readiness_summary(active_df, self.qc_df, self.meta_df, mapping, dist, feature_missing, groups)
        qc_corr = feature_qc_correlations(active_df, self.qc_df, feature_cols)
        qc_catalog = qc_metric_catalog(self.qc_df)
        qc_family = qc_family_burden_summary(self.qc_df)
        qc_row_burden = qc_row_burden_summary(self.qc_df)
        qc_family_assoc = feature_qc_family_association(qc_corr)
        qc_missing_assoc = qc_missingness_associations(active_df, self.qc_df, feature_cols)
        qc_outlier_assoc = qc_outlier_associations(outlier_flags, self.qc_df)
        qc_summary = qc_integration_summary(self.qc_df, qc_corr, qc_family, qc_missing_assoc, qc_outlier_assoc)
        rel_summary = feature_relationship_summary(active_df, feature_cols, self.registry_df)
        rel_corr_long = feature_correlation_long_table(active_df, feature_cols)
        rel_redundant = redundant_feature_pairs(rel_corr_long, self.registry_df)
        rel_modules = feature_relationship_modules(rel_corr_long, self.registry_df)
        rel_family_matrix = feature_family_correlation_matrix(rel_corr_long, self.registry_df)
        rel_pca_summary = feature_pca_summary(active_df, feature_cols)
        rel_pca_loadings = feature_pca_loadings(active_df, feature_cols, self.registry_df)
        rel_pca_scores = feature_pca_scores(active_df, feature_cols)
        screening = build_group_outcome_screening(active_df, feature_cols, mapping)
        reliability = reliability_screen(dist, qc_corr)
        reliability_design = reliability_design_summary(active_df, feature_cols, mapping)
        reliability_subjects = reliability_subject_record_counts(active_df, mapping)
        reliability_repeatability = feature_repeatability_summary(active_df, feature_cols, mapping, self.registry_df)
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
            "metadata_context": pd.DataFrame([{"strategy": getattr(self, "metadata_join_strategy", "feature_table_only")}]),
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
        active_df = self.analysis_df if self.analysis_df is not None else self.feature_df
        paths["overview_readiness_scorecard"] = str(plot_overview_readiness_scorecard(outputs.get("overview_readiness_summary", pd.DataFrame()), plots_dir / "overview_readiness_scorecard.png"))
        paths["overview_design_tiles"] = str(plot_dataset_design_tiles(outputs.get("dataset_design_overview", pd.DataFrame()), plots_dir / "overview_design_tiles.png"))
        paths["role_counts"] = str(plot_role_counts(outputs.get("feature_role_summary", pd.DataFrame()), plots_dir / "overview_role_counts.png"))
        paths["group_counts"] = str(plot_group_counts(outputs.get("group_counts", pd.DataFrame()), plots_dir / "overview_group_counts.png"))
        paths["overview_subject_task_matrix"] = str(plot_subject_task_matrix(active_df, plots_dir / "overview_subject_task_matrix.png"))
        paths["feature_family_counts"] = str(plot_feature_family_counts(outputs.get("feature_family_overview", pd.DataFrame()), plots_dir / "overview_feature_family_counts.png"))
        paths["overview_feature_family_quality"] = str(plot_feature_family_quality(outputs.get("feature_family_overview", pd.DataFrame()), plots_dir / "overview_feature_family_quality.png"))
        paths["overview_feature_quality_landscape"] = str(plot_feature_quality_landscape(outputs.get("overview_feature_quality_landscape", pd.DataFrame()), plots_dir / "overview_feature_quality_landscape.png"))
        paths["missingness_top_features"] = str(plot_missingness(outputs.get("feature_distribution_summary", pd.DataFrame()), plots_dir / "missingness_top_features.png"))
        paths["feature_availability_heatmap"] = str(plot_feature_availability_heatmap(active_df, feature_cols, plots_dir / "feature_availability_heatmap.png"))
        # Missingness-specific plots are generated at the same time so that the Missingness page is immediately usable.
        paths["missingness_row_distribution"] = str(plot_row_missingness_distribution(outputs.get("missingness_by_row", pd.DataFrame()), plots_dir / "missingness_row_distribution.png"))
        paths["missingness_by_group"] = str(plot_missingness_by_group(outputs.get("missingness_by_group", pd.DataFrame()), plots_dir / "missingness_by_group.png"))
        paths["missingness_by_family"] = str(plot_missingness_family_summary(outputs.get("missingness_by_family", pd.DataFrame()), plots_dir / "missingness_by_family.png"))
        paths["missingness_comissing_heatmap"] = str(plot_comissing_heatmap(active_df, feature_cols, plots_dir / "missingness_comissing_heatmap.png"))
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
            focus_cols = [c for c in shape_for_grid["feature"].astype(str).tolist() if c in active_df.columns] or focus_cols
        paths["feature_distribution_grid"] = str(plot_distribution_grid(active_df, focus_cols, plots_dir / "feature_distribution_grid.png"))
        first_feature = feature_cols[0] if feature_cols else None
        if first_feature:
            paths["selected_feature_distribution"] = str(plot_selected_feature_diagnostic(active_df, first_feature, plots_dir / "selected_feature_distribution.png"))
            paths["selected_feature_by_group"] = str(plot_group_feature_boxplot(active_df, first_feature, plots_dir / "selected_feature_by_group.png"))
        paths["qc_artifact_model"] = str(plot_qc_artifact_model(plots_dir / "qc_artifact_model.png"))
        paths["qc_family_burden"] = str(plot_qc_family_burden(outputs.get("qc_family_burden_summary", pd.DataFrame()), plots_dir / "qc_family_burden.png"))
        paths["qc_metric_distributions"] = str(plot_qc_metric_distributions(self.qc_df, outputs.get("qc_metric_catalog", pd.DataFrame()), plots_dir / "qc_metric_distributions.png"))
        paths["qc_feature_association_heatmap"] = str(plot_qc_feature_association_heatmap(outputs.get("feature_qc_family_association", pd.DataFrame()), plots_dir / "qc_feature_association_heatmap.png"))
        paths["qc_top_feature_associations"] = str(plot_qc_top_feature_associations(outputs.get("feature_qc_spearman_correlation", pd.DataFrame()), plots_dir / "qc_top_feature_associations.png"))
        paths["qc_missingness_associations"] = str(plot_qc_missingness_associations(outputs.get("qc_missingness_associations", pd.DataFrame()), plots_dir / "qc_missingness_associations.png"))
        paths["qc_row_burden"] = str(plot_qc_row_burden(outputs.get("qc_row_burden_summary", pd.DataFrame()), plots_dir / "qc_row_burden.png"))
        qcols = outputs.get("qc_metric_catalog", pd.DataFrame()).get("qc_variable", pd.Series(dtype=str)).astype(str).tolist()
        if first_feature and qcols:
            paths["selected_feature_qc_scatter"] = str(plot_selected_feature_qc_scatter(active_df, self.qc_df, first_feature, qcols[0], plots_dir / "selected_feature_qc_scatter.png"))
        paths["relationship_correlation_heatmap"] = str(plot_relationship_correlation_heatmap(outputs.get("feature_correlation_long", pd.DataFrame()), plots_dir / "relationship_correlation_heatmap.png"))
        paths["relationship_redundant_pairs"] = str(plot_relationship_redundant_pairs(outputs.get("feature_redundant_pairs", pd.DataFrame()), plots_dir / "relationship_redundant_pairs.png"))
        paths["relationship_family_matrix"] = str(plot_relationship_family_matrix(outputs.get("feature_family_correlation_matrix", pd.DataFrame()), plots_dir / "relationship_family_matrix.png"))
        paths["relationship_pca_scree"] = str(plot_relationship_pca_scree(outputs.get("feature_pca_summary", pd.DataFrame()), plots_dir / "relationship_pca_scree.png"))
        paths["relationship_pca_scores"] = str(plot_relationship_pca_scores(outputs.get("feature_pca_scores", pd.DataFrame()), active_df, plots_dir / "relationship_pca_scores.png"))
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
            paths["selected_feature_outcome"] = str(plot_selected_feature_outcome(active_df, first_feature, screen_var_list[0], plots_dir / "selected_feature_outcome.png"))
        paths["reliability_status_counts"] = str(plot_reliability_status_counts(outputs.get("feature_repeatability_summary", pd.DataFrame()), plots_dir / "reliability_status_counts.png"))
        paths["reliability_icc_ranking"] = str(plot_reliability_icc_ranking(outputs.get("feature_repeatability_summary", pd.DataFrame()), plots_dir / "reliability_icc_ranking.png"))
        paths["reliability_variance_landscape"] = str(plot_reliability_variance_landscape(outputs.get("feature_repeatability_summary", pd.DataFrame()), plots_dir / "reliability_variance_landscape.png"))
        paths["reliability_family_summary"] = str(plot_reliability_family_summary(outputs.get("reliability_family_summary", pd.DataFrame()), plots_dir / "reliability_family_summary.png"))
        paths["reliability_subject_counts"] = str(plot_reliability_subject_counts(outputs.get("reliability_subject_record_counts", pd.DataFrame()), plots_dir / "reliability_subject_counts.png"))
        if first_feature:
            paths["selected_feature_reliability"] = str(plot_selected_feature_reliability(active_df, first_feature, plots_dir / "selected_feature_reliability.png"))
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
            "Design context": "overview_design_tiles",
            "Role mapping summary": "role_counts",
            "Feature-family coverage": "overview_feature_family_quality",
            "Feature-quality landscape": "overview_feature_quality_landscape",
            "Subject x task coverage": "overview_subject_task_matrix",
        }
        self.preview_plot(key_map.get(label, "overview_readiness_scorecard"))

    def _overview_plot_caption_text(self, key: str) -> str:
        captions = {
            "overview_readiness_scorecard": "Readiness scorecard: a compact orientation panel across completeness, numeric analyzability, metadata context, QC context, design richness, row depth, and feature breadth. This is an audit readiness view, not an ML performance score.",
            "overview_design_tiles": "Design context: shows whether subject, session/visit, task, diagnosis/severity, sex/gender, and device/context variables are detected. Metadata-derived variables are included when metadata can be joined safely.",
            "role_counts": "Role mapping summary: verifies that columns are classified as features, identifiers, targets, covariates, QC variables, audit/status fields, or ignored fields before downstream analysis.",
            "overview_subject_task_matrix": "Subject x task coverage: shows repeated-measures and task coverage when subject/task fields are available. Empty blocks reveal missing task coverage or incomplete alignment.",
            "overview_feature_family_quality": "Feature-family coverage: summarizes predictor coverage by subsystem/family. This is the high-level family view; detailed missingness and QC are handled in their own menus.",
            "overview_feature_quality_landscape": "Feature-quality landscape: positions features by missingness and robust outlier burden. Use it only to identify features needing review; use the Distribution and QC menus for detailed diagnosis.",
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
            self.update_task_review_dashboard(outputs)
            self.update_longitudinal_dashboard(outputs)
            self.update_screening_dashboard(outputs)
            self.update_reliability_dashboard(outputs)
            self.update_recommendations_dashboard(outputs)
            self.update_export_dashboard(outputs)
            self.log(f"Overview/missingness plots generated in: {plots_dir}")
            self.preview_plot("overview_readiness_scorecard", generate_if_missing=False)
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
        self._display_plot_image(self.overview_plot_preview, path)

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




    def _display_plot_image(self, label: QLabel, path: Path) -> bool:
        """Render a plot preview without allowing repeated Show clicks to resize/zoom the widget."""
        pix = QPixmap(str(path))
        if pix.isNull():
            label.clear()
            label.setText(f"Could not load plot:\n{path}")
            return False
        target = label.contentsRect().size()
        if target.width() < 320 or target.height() < 260:
            target = QSize(1000, 560)
        scaled = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.clear()
        label.setPixmap(scaled)
        label.setToolTip(str(path))
        return True

    def _add_standard_plot_gallery(
        self,
        parent_layout: QVBoxLayout,
        title: str,
        caption: str,
        combo_attr: str,
        items: list[tuple[str, str]],
        preview_attr: str,
        interpretation_attr: str | None,
        preview_callback,
        open_callback,
        placeholder: str,
        min_height: int = 500,
    ) -> None:
        """Add the shared plot-first gallery used across Feature GUI review menus."""
        plot_panel = QFrame()
        plot_panel.setStyleSheet(f"QFrame {{ background:#F8FBFE; border:1px solid {LINE}; border-radius:12px; }}")
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(14, 14, 14, 14)
        plot_layout.setSpacing(10)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)
        label = QLabel(title)
        label.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        toolbar.addWidget(label)

        combo = QComboBox()
        combo.setMinimumWidth(360)
        for text, key in items:
            combo.addItem(text, key)
        combo.setToolTip("Choose one focused plot for this menu. Detailed tables stay below the plot area.")
        setattr(self, combo_attr, combo)
        toolbar.addWidget(combo, 1)

        show_btn = QPushButton("Show")
        show_btn.setProperty("secondary", True)
        show_btn.clicked.connect(lambda _=False, c=combo: preview_callback(c.currentData()))
        toolbar.addWidget(show_btn)

        regen = QPushButton("Regenerate")
        regen.clicked.connect(self.regenerate_overview_plots)
        toolbar.addWidget(regen)

        toolbar.addStretch(1)
        open_btn = QPushButton("Open current plot")
        open_btn.setProperty("secondary", True)
        open_btn.clicked.connect(open_callback)
        toolbar.addWidget(open_btn)
        plot_layout.addLayout(toolbar)

        caption_label = QLabel(caption)
        caption_label.setWordWrap(True)
        caption_label.setStyleSheet(f"color:{MUTED}; background:#FFFFFF; border:1px solid {LINE}; border-radius:8px; padding:9px;")
        plot_layout.addWidget(caption_label)

        preview = QLabel(placeholder)
        preview.setAlignment(Qt.AlignCenter)
        preview.setMinimumHeight(min_height)
        preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview.setStyleSheet(f"QLabel {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; color:{MUTED}; padding:16px; }}")
        setattr(self, preview_attr, preview)
        plot_layout.addWidget(preview, 1)

        if interpretation_attr:
            interp = QLabel("Select a plot to see structured interpretation guidance.")
            interp.setWordWrap(True)
            interp.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            interp.setStyleSheet(f"QLabel {{ background:#FFFFFF; color:{INK}; border:1px solid {LINE}; border-radius:10px; padding:12px; font-size:12px; line-height:140%; }}")
            setattr(self, interpretation_attr, interp)
            plot_layout.addWidget(interp)

        parent_layout.addWidget(plot_panel, 1)

    def _missingness_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        card = Card(
            "Missingness",
            "Audit feature and row availability. This page is limited to missing-data burden and structure; distribution shape, QC sensitivity, and outcome screening stay in their own menus."
        )

        self.missing_note = QLabel("Run Feature Analysis to populate missingness plots and tables. Use this page to decide whether missingness is isolated, feature-family specific, row/recording specific, or related to metadata groups before any imputation or exclusion.")
        self.missing_note.setWordWrap(True)
        self.missing_note.setStyleSheet(f"color:{MUTED}; background:#F7FAFD; border:1px solid {LINE}; border-radius:8px; padding:10px;")
        card.layout.addWidget(self.missing_note)

        missing_split = QHBoxLayout()
        missing_split.setSpacing(14)

        # Main visual area follows the same pattern as Overview: compact toolbar,
        # explanatory caption, large plot preview, and detailed tables below.
        plot_panel = QFrame()
        plot_panel.setStyleSheet(f"QFrame {{ background:#F8FBFE; border:1px solid {LINE}; border-radius:12px; }}")
        plot_panel_layout = QVBoxLayout(plot_panel)
        plot_panel_layout.setContentsMargins(14, 14, 14, 14)
        plot_panel_layout.setSpacing(10)

        plot_header = QHBoxLayout()
        plot_header.setSpacing(10)
        plot_title = QLabel("Missingness plot")
        plot_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        plot_header.addWidget(plot_title)

        self.missing_plot_combo = QComboBox()
        self.missing_plot_combo.setMinimumWidth(360)
        self.missing_plot_combo.addItem("Feature missingness burden", "missingness_top_features")
        self.missing_plot_combo.addItem("Recording missingness burden", "missingness_row_distribution")
        self.missing_plot_combo.addItem("Missingness by metadata group", "missingness_by_group")
        self.missing_plot_combo.addItem("Missingness by feature family", "missingness_by_family")
        self.missing_plot_combo.addItem("Feature availability heatmap", "feature_availability_heatmap")
        self.missing_plot_combo.addItem("Co-missingness clusters", "missingness_comissing_heatmap")
        plot_header.addWidget(self.missing_plot_combo, 1)

        show_btn = QPushButton("Show")
        show_btn.setProperty("secondary", True)
        show_btn.clicked.connect(lambda: self.preview_missingness_plot(self.missing_plot_combo.currentData()))
        plot_header.addWidget(show_btn)

        regen = QPushButton("Regenerate")
        regen.clicked.connect(self.regenerate_missingness_scope_plots)
        plot_header.addWidget(regen)

        plot_header.addStretch(1)
        open_btn = QPushButton("Open current plot")
        open_btn.setProperty("secondary", True)
        open_btn.clicked.connect(self.open_current_missingness_plot)
        plot_header.addWidget(open_btn)
        plot_panel_layout.addLayout(plot_header)

        self.missing_plot_caption = QLabel(
            "Missingness uses only availability plots: feature burden, recording burden, metadata-group structure, feature-family structure, availability heatmap, and co-missingness clusters. It does not duplicate distribution, QC, relationship, screening, or ML-export plots."
        )
        self.missing_plot_caption.setWordWrap(True)
        self.missing_plot_caption.setStyleSheet(f"color:{MUTED}; background:#FFFFFF; border:1px solid {LINE}; border-radius:8px; padding:9px;")
        plot_panel_layout.addWidget(self.missing_plot_caption)

        scope_row = QHBoxLayout()
        scope_row.setSpacing(10)
        scope_row.addWidget(QLabel("Task focus:"))
        self.missing_task_combo = QComboBox()
        self.missing_task_combo.setMinimumWidth(320)
        self.missing_task_combo.addItem("All tasks / not available")
        self.missing_task_combo.currentIndexChanged.connect(lambda _=0: self.preview_missingness_plot(self.missing_plot_combo.currentData() if hasattr(self, "missing_plot_combo") else "missingness_top_features"))
        scope_row.addWidget(self.missing_task_combo, 1)
        scope_row.addStretch(1)
        plot_panel_layout.addLayout(scope_row)

        self.missing_plot_preview = QLabel("Run Feature Analysis, then choose one missingness plot.")
        self.missing_plot_preview.setAlignment(Qt.AlignCenter)
        self.missing_plot_preview.setMinimumHeight(520)
        self.missing_plot_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.missing_plot_preview.setStyleSheet(f"QLabel {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; color:{MUTED}; padding:16px; }}")
        plot_panel_layout.addWidget(self.missing_plot_preview, 1)
        missing_split.addWidget(plot_panel, 1)

        # Side summary keeps availability metrics visible without occupying the top
        # of the page, matching the Overview page pattern.
        side_panel = QFrame()
        side_panel.setStyleSheet(f"QFrame {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:12px; }}")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        side_title = QLabel("Availability snapshot")
        side_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        side_layout.addWidget(side_title)
        side_note = QLabel("Compact availability metrics. Use these as navigation cues; detailed feature, row, group, family, and co-missing tables remain below the plot.")
        side_note.setWordWrap(True)
        side_note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent; font-size:12px;")
        side_layout.addWidget(side_note)
        self.missing_metric_grid = QGridLayout()
        self.missing_metric_grid.setHorizontalSpacing(8)
        self.missing_metric_grid.setVerticalSpacing(8)
        side_layout.addLayout(self.missing_metric_grid)
        side_layout.addStretch(1)
        side_panel.setFixedWidth(300)
        missing_split.addWidget(side_panel)
        card.layout.addLayout(missing_split)

        tables_header = QLabel("Detailed missingness tables")
        tables_header.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; padding-top:8px;")
        card.layout.addWidget(tables_header)

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
            t.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
            t.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        tabs.addTab(self.missing_feature_table, "By feature")
        tabs.addTab(self.missing_row_table, "By row / recording")
        tabs.addTab(self.missing_group_table, "By group")
        tabs.addTab(self.missing_family_table, "By family")
        tabs.addTab(self.missing_comissing_table, "Co-missing pairs")
        card.layout.addWidget(tabs)

        layout.addWidget(card)
        return self._wrap_scroll(body)


    def _refresh_missingness_task_combo(self, df: pd.DataFrame) -> None:
        if not hasattr(self, "missing_task_combo"):
            return
        task_col = self._task_col(df) if hasattr(self, "_task_col") else None
        current = self.missing_task_combo.currentText()
        self.missing_task_combo.blockSignals(True)
        self.missing_task_combo.clear()
        if task_col and task_col in df.columns:
            values = sorted([str(x) for x in df[task_col].dropna().unique().tolist() if str(x).strip()])
            self.missing_task_combo.addItem("All tasks")
            for v in values[:500]:
                self.missing_task_combo.addItem(v)
        else:
            self.missing_task_combo.addItem("All tasks / not available")
        ix = self.missing_task_combo.findText(current)
        if ix >= 0:
            self.missing_task_combo.setCurrentIndex(ix)
        self.missing_task_combo.blockSignals(False)

    def _missingness_scope_table(self) -> tuple[pd.DataFrame, str]:
        """Return task-scoped table for Missingness plots only.

        This does not alter analysis outputs or tables. It is deliberately local
        so missingness can be inspected per task without rerunning the whole GUI.
        """
        df = self._active_analysis_table() if hasattr(self, "_active_analysis_table") else (self.analysis_df if self.analysis_df is not None else self.feature_df)
        if df is None:
            return pd.DataFrame(), "All tasks"
        task_col = self._task_col(df) if hasattr(self, "_task_col") else None
        task_value = self.missing_task_combo.currentText() if hasattr(self, "missing_task_combo") else "All tasks"
        if task_col and task_col in df.columns and task_value not in {"", "All tasks", "All tasks / not available"}:
            scoped = df[df[task_col].astype(str).eq(str(task_value))].copy()
            return scoped, f"Task = {task_value}"
        return df.copy(), "All tasks"

    def _missingness_feature_cols(self, df: pd.DataFrame) -> list[str]:
        if getattr(self, "mapping_df", None) is not None and not self.mapping_df.empty:
            roles = role_lists(self.mapping_df)
            return [c for c in roles.get("Feature", []) if c in df.columns]
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    def generate_missingness_scope_plots(self) -> None:
        df, scope_label = self._missingness_scope_table()
        self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
        feature_cols = self._missingness_feature_cols(df)
        safe_scope = re.sub(r"[^A-Za-z0-9_.-]+", "_", scope_label).strip("_")[:80] or "all_tasks"
        if not hasattr(self, "plot_paths"):
            self.plot_paths = {}
        if df.empty or not feature_cols:
            # Generate clear empty plots rather than failing.
            missing_feature = pd.DataFrame()
            row_missing = pd.DataFrame()
            group_missing = pd.DataFrame()
            family_missing = pd.DataFrame()
        else:
            missing_feature = missingness_feature_summary(df, feature_cols, self.registry_df)
            row_missing = missingness_row_summary(df, feature_cols)
            group_missing = missingness_group_summary(df, feature_cols)
            family_missing = missingness_family_summary(missing_feature)
        self.plot_paths["missingness_top_features"] = str(plot_missingness(missing_feature, plots_dir / f"missingness_top_features_{safe_scope}.png"))
        self.plot_paths["missingness_row_distribution"] = str(plot_row_missingness_distribution(row_missing, plots_dir / f"missingness_row_distribution_{safe_scope}.png"))
        self.plot_paths["missingness_by_group"] = str(plot_missingness_by_group(group_missing, plots_dir / f"missingness_by_group_{safe_scope}.png"))
        self.plot_paths["missingness_by_family"] = str(plot_missingness_family_summary(family_missing, plots_dir / f"missingness_by_family_{safe_scope}.png"))
        self.plot_paths["feature_availability_heatmap"] = str(plot_feature_availability_heatmap(df, feature_cols, plots_dir / f"feature_availability_heatmap_{safe_scope}.png", max_features=None, max_rows=None))
        self.plot_paths["missingness_comissing_heatmap"] = str(plot_comissing_heatmap(df, feature_cols, plots_dir / f"missingness_comissing_heatmap_{safe_scope}.png", max_features=None))
        if hasattr(self, "missing_plot_caption"):
            self.missing_plot_caption.setText(f"{self._missingness_plot_caption_text(self.missing_plot_combo.currentData())}\n\nCurrent scope: {scope_label}. Feature heatmaps include all selected feature columns when feasible.")

    def regenerate_missingness_scope_plots(self) -> None:
        self.generate_missingness_scope_plots()
        self.preview_missingness_plot(self.missing_plot_combo.currentData() if hasattr(self, "missing_plot_combo") else "missingness_top_features", regenerate=False)

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
        mean_miss = "-"
        high_features = "-"
        high_rows = "-"
        groups = "-"
        if feat is not None and not feat.empty and "missing_fraction" in feat.columns:
            mean_miss = f"{pd.to_numeric(feat['missing_fraction'], errors='coerce').mean():.3f}"
            high_features = int(feat.get("missingness_status", pd.Series(dtype=str)).isin(["review", "high_review"]).sum())
        if row is not None and not row.empty and "missing_fraction_feature_columns" in row.columns:
            high_rows = int((pd.to_numeric(row["missing_fraction_feature_columns"], errors="coerce") >= 0.50).sum())
        if group is not None and not group.empty and "group_variable" in group.columns:
            groups = int(group["group_variable"].nunique())
        tiles = [
            ("Features", n_features, "selected predictors"),
            ("Rows", len(row) if row is not None else "-", "recordings / files"),
            ("Mean missingness", mean_miss, "across selected predictors"),
            ("High-missing features", high_features, ">=50% missing or worse"),
            ("Metadata groups", groups, "available strata"),
            ("Default policy", "do not auto-impute", "review mechanism first"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.missing_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx, 0)
        self._refresh_missingness_task_combo(self._active_analysis_table() if hasattr(self, "_active_analysis_table") else pd.DataFrame())
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

    def preview_missingness_plot(self, key: str, regenerate: bool = True) -> None:
        if regenerate:
            self.generate_missingness_scope_plots()
        elif not hasattr(self, "plot_paths") or key not in self.plot_paths or not Path(self.plot_paths.get(key, "")).exists():
            self.generate_missingness_scope_plots()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths:
            QMessageBox.information(self, "Plot unavailable", "Run Feature Analysis first, or this plot was not generated for the current dataset.")
            return
        path = Path(self.plot_paths[key])
        if not path.exists():
            QMessageBox.information(self, "Plot unavailable", f"Plot file not found:\n{path}")
            return
        self.current_missingness_plot = path
        if hasattr(self, "missing_plot_caption"):
            _, scope_label = self._missingness_scope_table()
            self.missing_plot_caption.setText(
                f"{self._missingness_plot_caption_text(key)}\n\nCurrent scope: {scope_label}. Feature heatmaps include all selected feature columns when feasible."
            )
        self._display_plot_image(self.missing_plot_preview, path)

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
            "Inspect feature distributions, robust outliers, expected-range flags, and group/context overlays. This menu is limited to value shape and plausibility; missingness, QC sensitivity, relationships, and outcome screening stay in their own menus."
        )

        self.dist_note = QLabel("Run Feature Analysis to populate distribution diagnostics. Review outliers in context: task, QC, device, diagnosis/severity, and implementation status can all affect feature values.")
        self.dist_note.setWordWrap(True)
        self.dist_note.setStyleSheet(f"color:{MUTED}; background:#F7FAFD; border:1px solid {LINE}; border-radius:8px; padding:10px;")
        card.layout.addWidget(self.dist_note)

        dist_split = QHBoxLayout()
        dist_split.setSpacing(14)

        # Main visual area matches Overview and Missingness: one large plot panel,
        # one compact toolbar, and detailed tables below.
        plot_panel = QFrame()
        plot_panel.setStyleSheet(f"QFrame {{ background:#F8FBFE; border:1px solid {LINE}; border-radius:12px; }}")
        plot_panel_layout = QVBoxLayout(plot_panel)
        plot_panel_layout.setContentsMargins(14, 14, 14, 14)
        plot_panel_layout.setSpacing(10)

        plot_header = QHBoxLayout()
        plot_header.setSpacing(10)
        plot_title = QLabel("Distribution plot")
        plot_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        plot_header.addWidget(plot_title)

        self.dist_plot_combo = QComboBox()
        self.dist_plot_combo.setMinimumWidth(360)
        self.dist_plot_combo.addItem("Review status", "distribution_review_status")
        self.dist_plot_combo.addItem("Shape priority", "distribution_shape_summary")
        self.dist_plot_combo.addItem("Shape landscape", "distribution_shape_landscape")
        self.dist_plot_combo.addItem("Expected-range flags", "expected_range_flags")
        self.dist_plot_combo.addItem("Outlier counts", "outlier_counts")
        self.dist_plot_combo.addItem("Row outlier burden", "row_outlier_burden")
        self.dist_plot_combo.addItem("Variance screen", "variance_screen")
        self.dist_plot_combo.addItem("Selected feature diagnostic", "selected_feature_distribution")
        self.dist_plot_combo.addItem("Selected feature by group", "selected_feature_by_group")
        plot_header.addWidget(self.dist_plot_combo, 1)

        show_btn = QPushButton("Show")
        show_btn.setProperty("secondary", True)
        show_btn.clicked.connect(lambda: self.preview_distribution_plot(self.dist_plot_combo.currentData()))
        plot_header.addWidget(show_btn)

        regen = QPushButton("Regenerate")
        regen.clicked.connect(self.regenerate_distribution_scope_plots)
        plot_header.addWidget(regen)

        plot_header.addStretch(1)
        open_btn = QPushButton("Open current plot")
        open_btn.setProperty("secondary", True)
        open_btn.clicked.connect(self.open_current_distribution_plot)
        plot_header.addWidget(open_btn)
        plot_panel_layout.addLayout(plot_header)

        self.dist_plot_caption = QLabel(
            "Distributions uses only value-shape and plausibility plots. Task focus filters all distribution plots locally. Clinical context/value filters the distribution scope and also supplies grouping for selected-feature plots. There is no separate group overlay control."
        )
        self.dist_plot_caption.setWordWrap(True)
        self.dist_plot_caption.setStyleSheet(f"color:{MUTED}; background:#FFFFFF; border:1px solid {LINE}; border-radius:8px; padding:9px;")
        plot_panel_layout.addWidget(self.dist_plot_caption)

        scope_row = QHBoxLayout()
        scope_row.setSpacing(10)
        scope_row.addWidget(QLabel("Task focus:"))
        self.dist_task_combo = QComboBox()
        self.dist_task_combo.setMinimumWidth(260)
        self.dist_task_combo.addItem("All tasks / not available")
        self.dist_task_combo.currentIndexChanged.connect(lambda _=0: self.preview_distribution_plot(self.dist_plot_combo.currentData() if hasattr(self, "dist_plot_combo") else "distribution_review_status"))
        scope_row.addWidget(self.dist_task_combo, 1)
        scope_row.addWidget(QLabel("Clinical context:"))
        self.dist_context_combo = QComboBox()
        self.dist_context_combo.setMinimumWidth(250)
        self.dist_context_combo.addItem("Auto / not available")
        self.dist_context_combo.currentIndexChanged.connect(lambda _=0: (self._refresh_dist_context_value_combo(self._active_analysis_table()), self.preview_distribution_plot(self.dist_plot_combo.currentData() if hasattr(self, "dist_plot_combo") else "distribution_review_status")))
        scope_row.addWidget(self.dist_context_combo, 1)
        scope_row.addWidget(QLabel("Value:"))
        self.dist_context_value_combo = QComboBox()
        self.dist_context_value_combo.setMinimumWidth(220)
        self.dist_context_value_combo.addItem("All values")
        self.dist_context_value_combo.currentIndexChanged.connect(lambda _=0: self.preview_distribution_plot(self.dist_plot_combo.currentData() if hasattr(self, "dist_plot_combo") else "distribution_review_status"))
        scope_row.addWidget(self.dist_context_value_combo, 1)
        plot_panel_layout.addLayout(scope_row)

        selectors = QHBoxLayout()
        selectors.setSpacing(10)
        selectors.addWidget(QLabel("Feature to inspect:"))
        self.dist_feature_combo = QComboBox()
        self.dist_feature_combo.setMinimumWidth(360)
        self.dist_feature_combo.currentTextChanged.connect(lambda _: self.generate_selected_distribution_plot())
        selectors.addWidget(self.dist_feature_combo, 1)
        selectors.addWidget(QLabel("Grouping for selected-feature plots comes from Clinical context above."))
        selectors.addStretch(1)
        plot_panel_layout.addLayout(selectors)

        self.dist_plot_preview = QLabel("Run Feature Analysis, then choose one distribution plot.")
        self.dist_plot_preview.setAlignment(Qt.AlignCenter)
        self.dist_plot_preview.setMinimumHeight(520)
        self.dist_plot_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dist_plot_preview.setStyleSheet(f"QLabel {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; color:{MUTED}; padding:16px; }}")
        plot_panel_layout.addWidget(self.dist_plot_preview, 1)

        self.dist_interpretation_label = QLabel("Select a plot to see structured interpretation guidance.")
        self.dist_interpretation_label.setWordWrap(True)
        self.dist_interpretation_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.dist_interpretation_label.setStyleSheet(f"QLabel {{ background:#FFFFFF; color:{INK}; border:1px solid {LINE}; border-radius:10px; padding:12px; font-size:12px; line-height:140%; }}")
        plot_panel_layout.addWidget(self.dist_interpretation_label)

        dist_split.addWidget(plot_panel, 1)

        # Side summary keeps distribution metrics visible without occupying the
        # top of the page, matching Overview and Missingness.
        side_panel = QFrame()
        side_panel.setStyleSheet(f"QFrame {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:12px; }}")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        side_title = QLabel("Distribution snapshot")
        side_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        side_layout.addWidget(side_title)
        side_note = QLabel("Compact value-shape metrics. Use these as navigation cues; detailed review, outlier, range, shape, and row-burden tables remain below the plot.")
        side_note.setWordWrap(True)
        side_note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent; font-size:12px;")
        side_layout.addWidget(side_note)
        self.dist_metric_grid = QGridLayout()
        self.dist_metric_grid.setHorizontalSpacing(8)
        self.dist_metric_grid.setVerticalSpacing(8)
        side_layout.addLayout(self.dist_metric_grid)
        side_layout.addStretch(1)
        side_panel.setFixedWidth(300)
        dist_split.addWidget(side_panel)
        card.layout.addLayout(dist_split)

        tables_header = QLabel("Detailed distribution tables")
        tables_header.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; padding-top:8px;")
        card.layout.addWidget(tables_header)

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
            t.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
            t.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        tabs.addTab(self.dist_summary_table, "Distribution summary")
        tabs.addTab(self.dist_review_table, "Review summary")
        tabs.addTab(self.dist_outlier_table, "Row-level outliers")
        tabs.addTab(self.dist_range_table, "Expected ranges")
        tabs.addTab(self.dist_shape_table, "Shape audit")
        tabs.addTab(self.dist_row_burden_table, "Row burden")
        card.layout.addWidget(tabs)

        layout.addWidget(card)
        return self._wrap_scroll(body)


    def _refresh_dist_task_combo(self, df: pd.DataFrame) -> None:
        if not hasattr(self, "dist_task_combo"):
            return
        task_col = self._task_col(df) if hasattr(self, "_task_col") else None
        current = self.dist_task_combo.currentText()
        self.dist_task_combo.blockSignals(True)
        self.dist_task_combo.clear()
        if task_col and task_col in df.columns:
            values = sorted([str(x) for x in df[task_col].dropna().unique().tolist() if str(x).strip()])
            self.dist_task_combo.addItem("All tasks")
            for v in values[:500]:
                self.dist_task_combo.addItem(v)
        else:
            self.dist_task_combo.addItem("All tasks / not available")
        ix = self.dist_task_combo.findText(current)
        if ix >= 0:
            self.dist_task_combo.setCurrentIndex(ix)
        self.dist_task_combo.blockSignals(False)

    def _dist_context_series(self, df: pd.DataFrame) -> tuple[pd.Series | None, str]:
        if df is None or df.empty or not hasattr(self, "dist_context_combo"):
            return None, "clinical context"
        label = self.dist_context_combo.currentText()
        candidates = self._safe_context_candidates(df) if hasattr(self, "_safe_context_candidates") else {}
        if label in {"", "Auto / not available"}:
            if "Diagnosis" in candidates:
                label = "Diagnosis"
            elif "ALSFRS bulbar" in candidates:
                label = "ALSFRS bulbar severity"
            else:
                return None, "clinical context"
        if label == "ALSFRS bulbar severity":
            col = candidates.get("ALSFRS bulbar")
            if not col:
                return None, label
            return df[col].map(self._bulbar_severity_local).astype("string"), label
        if label == "ALSFRS total severity":
            col = candidates.get("ALSFRS total")
            if not col:
                return None, label
            return df[col].map(self._alsfrs_total_severity_local).astype("string"), label
        if label == "ALSBDI severity":
            col = candidates.get("ALSBDI")
            if not col:
                return None, label
            score = pd.to_numeric(df[col], errors="coerce")
            out = pd.Series("score_detected_cutoffs_pending", index=df.index, dtype="string")
            out[score.isna()] = "no_score_or_control"
            return out, label
        col = candidates.get(label)
        if col:
            return df[col].astype("string"), label
        return None, label

    def _refresh_dist_context_controls(self, df: pd.DataFrame) -> None:
        if not hasattr(self, "dist_context_combo"):
            return
        current = self.dist_context_combo.currentText()
        candidates = self._safe_context_candidates(df) if hasattr(self, "_safe_context_candidates") else {}
        options = ["Auto / not available"]
        for label in ["Diagnosis", "ALSFRS bulbar severity", "ALSFRS total severity", "ALSBDI severity", "Sex / gender", "Session / visit", "Iteration"]:
            if label == "ALSFRS bulbar severity" and "ALSFRS bulbar" not in candidates:
                continue
            if label == "ALSFRS total severity" and "ALSFRS total" not in candidates:
                continue
            if label == "ALSBDI severity" and "ALSBDI" not in candidates:
                continue
            if label in {"Diagnosis", "Sex / gender", "Session / visit", "Iteration"} and label not in candidates:
                continue
            options.append(label)
        self.dist_context_combo.blockSignals(True)
        self.dist_context_combo.clear()
        for opt in options:
            self.dist_context_combo.addItem(opt)
        ix = self.dist_context_combo.findText(current)
        if ix >= 0:
            self.dist_context_combo.setCurrentIndex(ix)
        self.dist_context_combo.blockSignals(False)
        self._refresh_dist_context_value_combo(df)

    def _refresh_dist_context_value_combo(self, df: pd.DataFrame) -> None:
        if not hasattr(self, "dist_context_value_combo"):
            return
        current = self.dist_context_value_combo.currentText()
        series, label = self._dist_context_series(df)
        self.dist_context_value_combo.blockSignals(True)
        self.dist_context_value_combo.clear()
        self.dist_context_value_combo.addItem("All values")
        if series is not None:
            for v in sorted([str(x) for x in series.dropna().unique().tolist() if str(x).strip()])[:200]:
                self.dist_context_value_combo.addItem(v)
        ix = self.dist_context_value_combo.findText(current)
        if ix >= 0:
            self.dist_context_value_combo.setCurrentIndex(ix)
        self.dist_context_value_combo.blockSignals(False)

    def _distribution_scope_table(self) -> tuple[pd.DataFrame, str]:
        df = self._active_analysis_table() if hasattr(self, "_active_analysis_table") else (self.analysis_df if self.analysis_df is not None else self.feature_df)
        if df is None:
            return pd.DataFrame(), "All tasks / all clinical context"
        out = df.copy()
        parts = []
        task_col = self._task_col(out) if hasattr(self, "_task_col") else None
        task_value = self.dist_task_combo.currentText() if hasattr(self, "dist_task_combo") else "All tasks"
        if task_col and task_col in out.columns and task_value not in {"", "All tasks", "All tasks / not available"}:
            out = out[out[task_col].astype(str).eq(str(task_value))].copy()
            parts.append(f"Task = {task_value}")
        else:
            parts.append("All tasks")
        series, label = self._dist_context_series(out)
        value = self.dist_context_value_combo.currentText() if hasattr(self, "dist_context_value_combo") else "All values"
        if series is not None:
            out["__local_clinical_context__"] = series.values
            if value not in {"", "All values", "Auto / not available"}:
                out = out[out["__local_clinical_context__"].astype(str).eq(str(value))].copy()
                parts.append(f"{label} = {value}")
            else:
                parts.append(f"{label} = all")
        else:
            parts.append("clinical context = unavailable")
        return out, " | ".join(parts)

    def _distribution_feature_cols(self, df: pd.DataFrame) -> list[str]:
        if getattr(self, "mapping_df", None) is not None and not self.mapping_df.empty:
            roles = role_lists(self.mapping_df)
            return [c for c in roles.get("Feature", []) if c in df.columns]
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    def generate_distribution_scope_plots(self) -> None:
        df, scope_label = self._distribution_scope_table()
        self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
        feature_cols = self._distribution_feature_cols(df)
        safe_scope = re.sub(r"[^A-Za-z0-9_.-]+", "_", scope_label).strip("_")[:90] or "distribution_scope"
        if not hasattr(self, "plot_paths"):
            self.plot_paths = {}
        if df.empty or not feature_cols:
            dist = pd.DataFrame()
            expected = pd.DataFrame()
            review = pd.DataFrame()
            shape = pd.DataFrame()
            outliers = pd.DataFrame()
            row_burden = pd.DataFrame()
        else:
            dist = feature_distribution_summary(df, feature_cols)
            expected = expected_range_flags(df, feature_cols, self.registry_df)
            review = distribution_review_summary(dist, expected)
            shape = distribution_shape_audit(dist, expected)
            outliers = robust_outlier_flags(df, feature_cols, self.registry_df)
            row_burden = row_outlier_burden_summary(outliers, len(feature_cols))
        self.plot_paths["distribution_review_status"] = str(plot_distribution_review_summary(review, plots_dir / f"distribution_review_status_{safe_scope}.png"))
        self.plot_paths["distribution_shape_summary"] = str(plot_distribution_shape_summary(shape, plots_dir / f"distribution_shape_summary_{safe_scope}.png"))
        self.plot_paths["distribution_shape_landscape"] = str(plot_distribution_shape_landscape(shape, plots_dir / f"distribution_shape_landscape_{safe_scope}.png"))
        self.plot_paths["expected_range_flags"] = str(plot_expected_range_flags(expected, plots_dir / f"feature_expected_range_flags_{safe_scope}.png"))
        self.plot_paths["outlier_counts"] = str(plot_outlier_counts(outliers, plots_dir / f"outlier_counts_{safe_scope}.png"))
        self.plot_paths["row_outlier_burden"] = str(plot_row_outlier_burden(row_burden, plots_dir / f"row_outlier_burden_{safe_scope}.png"))
        self.plot_paths["variance_screen"] = str(plot_variance_screen(dist, plots_dir / f"variance_screen_{safe_scope}.png"))
        if hasattr(self, "dist_plot_caption"):
            self.dist_plot_caption.setText(
                "Distributions uses value-shape and plausibility plots only. Current scope: "
                f"{scope_label}. The base analysis table is not modified."
            )

    def regenerate_distribution_scope_plots(self) -> None:
        self.generate_distribution_scope_plots()
        self.preview_distribution_plot(self.dist_plot_combo.currentData() if hasattr(self, "dist_plot_combo") else "distribution_review_status", regenerate=False)

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
            ("Features", n_features, "numeric predictors"),
            ("Monitor", int(monitor), "moderate issues"),
            ("Review", int(review_n), "high-risk issues"),
            ("Outlier rows", out_n, "row-level flags"),
            ("Range flags", range_n, "outside expected ranges"),
            ("Zero variance", zero_n, "not useful for ML"),
            ("Shape review", shape_review, "shape needs review"),
            ("Row burden", row_review, "recordings with many flags"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.dist_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx, 0)
        active_for_scope = self._active_analysis_table() if hasattr(self, "_active_analysis_table") else (self.analysis_df if self.analysis_df is not None else self.feature_df)
        self._refresh_dist_task_combo(active_for_scope)
        self._refresh_dist_context_controls(active_for_scope)
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
        self.dist_note.setText("Distribution audit generated. Review statistical outliers together with expected ranges, QC artifacts, task compatibility, and clinical/context metadata before excluding or transforming features.")

    def _selected_distribution_feature(self) -> str | None:
        if not hasattr(self, "dist_feature_combo"):
            return None
        val = self.dist_feature_combo.currentText().strip()
        return val or None

    def _selected_distribution_group(self) -> str | None:
        """Grouping variable for selected-feature distribution plots.

        The old separate "Group overlay" control was removed because it was
        ambiguous. Distributions now uses the local Clinical context selector as
        the single grouping source. If no clinical context is available, fallback
        to task when present.
        """
        df, _scope = self._distribution_scope_table()
        if "__local_clinical_context__" in df.columns and df["__local_clinical_context__"].notna().sum() > 0:
            return "__local_clinical_context__"
        task_col = self._task_col(df) if hasattr(self, "_task_col") else None
        if task_col and task_col in df.columns and df[task_col].notna().sum() > 0:
            return task_col
        return None

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
            active_df, scope_label = self._distribution_scope_table()
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            low, high = self._selected_expected_bounds(feature)
            safe_scope = re.sub(r"[^A-Za-z0-9_.-]+", "_", scope_label).strip("_")[:90] or "distribution_scope"
            path = plot_selected_feature_diagnostic(active_df, feature, plots_dir / f"selected_feature_distribution_{safe_scope}.png", low, high, self._selected_distribution_group())
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
            active_df, scope_label = self._distribution_scope_table()
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            safe_scope = re.sub(r"[^A-Za-z0-9_.-]+", "_", scope_label).strip("_")[:90] or "distribution_scope"
            path = plot_group_feature_boxplot(active_df, feature, plots_dir / f"selected_feature_by_group_{safe_scope}.png", self._selected_distribution_group())
            if not hasattr(self, "plot_paths"):
                self.plot_paths = {}
            self.plot_paths["selected_feature_by_group"] = str(path)
        except Exception:
            pass

    def preview_distribution_plot(self, key: str, regenerate: bool = True) -> None:
        if key == "selected_feature_distribution":
            self.generate_selected_distribution_plot()
        elif key == "selected_feature_by_group":
            self.generate_selected_group_plot()
        elif regenerate:
            self.generate_distribution_scope_plots()
        elif not hasattr(self, "plot_paths") or key not in self.plot_paths or not Path(self.plot_paths.get(key, "")).exists():
            self.generate_distribution_scope_plots()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths:
            QMessageBox.information(self, "Plot unavailable", "Run Feature Analysis first, or this plot was not generated for the current dataset.")
            return
        path = Path(self.plot_paths[key])
        if not path.exists():
            QMessageBox.information(self, "Plot unavailable", f"Plot file not found:\n{path}")
            return
        self.current_distribution_plot = path
        self.update_distribution_interpretation(key)
        self._display_plot_image(self.dist_plot_preview, path)

    def update_distribution_interpretation(self, key: str) -> None:
        if not hasattr(self, "dist_interpretation_label"):
            return
        feature = self._selected_distribution_feature() or "selected feature"
        group = self.dist_context_combo.currentText() if hasattr(self, "dist_context_combo") and self.dist_context_combo.currentText() not in {"", "Auto / not available"} else "task or detected context"
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

        self.qc_note = QLabel(
            "Load an optional QC table in Project, accept Column Mapping, and run Feature Analysis. QC is interpreted as a multidimensional profile: additive interference, gain/level dynamics, reverberation/echo, channel/device/platform, nonlinear distortion, and temporal discontinuities."
        )
        self.qc_note.setWordWrap(True)
        self.qc_note.setStyleSheet(f"color:{INK}; background:#F7FAFD; border:1px solid {LINE}; border-radius:10px; padding:10px;")
        card.layout.addWidget(self.qc_note)

        qc_split = QHBoxLayout()
        qc_split.setSpacing(14)

        # Main visual area matches Overview, Missingness, and Distributions:
        # one large plot panel, one compact toolbar, selected controls inside the
        # plot context, and detailed tables below.
        plot_panel = QFrame()
        plot_panel.setStyleSheet(f"QFrame {{ background:#F8FBFE; border:1px solid {LINE}; border-radius:12px; }}")
        plot_panel_layout = QVBoxLayout(plot_panel)
        plot_panel_layout.setContentsMargins(14, 14, 14, 14)
        plot_panel_layout.setSpacing(10)

        plot_header = QHBoxLayout()
        plot_header.setSpacing(10)
        plot_title = QLabel("QC plot")
        plot_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        plot_header.addWidget(plot_title)

        self.qc_plot_combo = QComboBox()
        self.qc_plot_combo.setMinimumWidth(360)
        self.qc_plot_combo.addItem("QC framework", "qc_artifact_model")
        self.qc_plot_combo.addItem("Family burden", "qc_family_burden")
        self.qc_plot_combo.addItem("QC metric distributions", "qc_metric_distributions")
        self.qc_plot_combo.addItem("Feature x QC-family heatmap", "qc_feature_association_heatmap")
        self.qc_plot_combo.addItem("Top feature-QC associations", "qc_top_feature_associations")
        self.qc_plot_combo.addItem("Missingness linked to QC", "qc_missingness_associations")
        self.qc_plot_combo.addItem("Row QC burden", "qc_row_burden")
        self.qc_plot_combo.addItem("Selected feature x selected QC", "selected_feature_qc_scatter")
        plot_header.addWidget(self.qc_plot_combo, 1)

        show_btn = QPushButton("Show")
        show_btn.setProperty("secondary", True)
        show_btn.clicked.connect(lambda: self.preview_qc_plot(self.qc_plot_combo.currentData()))
        plot_header.addWidget(show_btn)

        regen = QPushButton("Regenerate")
        regen.clicked.connect(self.regenerate_overview_plots)
        plot_header.addWidget(regen)

        plot_header.addStretch(1)
        open_btn = QPushButton("Open current plot")
        open_btn.setProperty("secondary", True)
        open_btn.clicked.connect(self.open_current_qc_plot)
        plot_header.addWidget(open_btn)
        plot_panel_layout.addLayout(plot_header)

        self.qc_plot_caption = QLabel(
            "QC Integration uses only acquisition-sensitivity plots: artifact framework, QC burden, feature-QC associations, missingness-QC links, row burden, and one selected feature x QC metric scatter. It does not duplicate Missingness, Distributions, Relationships, Screening, or ML Export."
        )
        self.qc_plot_caption.setWordWrap(True)
        self.qc_plot_caption.setStyleSheet(f"color:{MUTED}; background:#FFFFFF; border:1px solid {LINE}; border-radius:8px; padding:9px;")
        plot_panel_layout.addWidget(self.qc_plot_caption)

        selectors = QHBoxLayout()
        selectors.setSpacing(10)
        selectors.addWidget(QLabel("Feature:"))
        self.qc_feature_combo = QComboBox()
        self.qc_feature_combo.setMinimumWidth(320)
        selectors.addWidget(self.qc_feature_combo, 1)
        selectors.addWidget(QLabel("QC metric:"))
        self.qc_metric_combo = QComboBox()
        self.qc_metric_combo.setMinimumWidth(220)
        selectors.addWidget(self.qc_metric_combo)
        plot_panel_layout.addLayout(selectors)

        self.qc_plot_preview = QLabel("Run Feature Analysis, then choose one QC plot.")
        self.qc_plot_preview.setAlignment(Qt.AlignCenter)
        self.qc_plot_preview.setMinimumHeight(520)
        self.qc_plot_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.qc_plot_preview.setStyleSheet(f"QLabel {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; color:{MUTED}; padding:16px; }}")
        plot_panel_layout.addWidget(self.qc_plot_preview, 1)

        self.qc_interpretation_label = QLabel("Select a plot to see structured interpretation guidance.")
        self.qc_interpretation_label.setWordWrap(True)
        self.qc_interpretation_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.qc_interpretation_label.setStyleSheet(f"QLabel {{ background:#FFFFFF; color:{INK}; border:1px solid {LINE}; border-radius:10px; padding:12px; font-size:12px; line-height:140%; }}")
        plot_panel_layout.addWidget(self.qc_interpretation_label)

        qc_split.addWidget(plot_panel, 1)

        # Side summary keeps QC metrics visible without occupying the top of the
        # page, matching Overview, Missingness, and Distributions.
        side_panel = QFrame()
        side_panel.setStyleSheet(f"QFrame {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:12px; }}")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        side_title = QLabel("QC snapshot")
        side_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        side_layout.addWidget(side_title)
        side_note = QLabel("Compact acquisition-context metrics. Use these as navigation cues; detailed QC summary, metric catalog, feature-QC associations, missingness-QC links, and row-burden tables remain below the plot.")
        side_note.setWordWrap(True)
        side_note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent; font-size:12px;")
        side_layout.addWidget(side_note)
        self.qc_metric_grid = QGridLayout()
        self.qc_metric_grid.setHorizontalSpacing(8)
        self.qc_metric_grid.setVerticalSpacing(8)
        side_layout.addLayout(self.qc_metric_grid)
        side_layout.addStretch(1)
        side_panel.setFixedWidth(300)
        qc_split.addWidget(side_panel)
        card.layout.addLayout(qc_split)

        tables_header = QLabel("Detailed QC integration tables")
        tables_header.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; padding-top:8px;")
        card.layout.addWidget(tables_header)

        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {LINE}; border-radius: 8px; background: #FFFFFF; }}
            QTabBar::tab {{ background:#F7FAFD; color:{NAVY}; border:1px solid {LINE}; border-bottom:none; padding:8px 14px; min-height:22px; }}
            QTabBar::tab:selected {{ background:#FFFFFF; color:{NAVY}; border-top:2px solid {TEAL}; font-weight:700; }}
            QTabBar::tab:hover {{ background:#F3FAF9; color:{NAVY}; }}
        """)
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
        tabs.addTab(self.qc_assoc_table, "Feature x QC")
        tabs.addTab(self.qc_family_assoc_table, "Feature x family")
        tabs.addTab(self.qc_missing_table, "Missingness x QC")
        tabs.addTab(self.qc_outlier_table, "Outliers x QC")
        tabs.addTab(self.qc_row_table, "Row QC burden")
        card.layout.addWidget(tabs)

        layout.addWidget(card)
        return self._wrap_scroll(body)

    def update_qc_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "qc_metric_grid"):
            return
        while self.qc_metric_grid.count():
            item = self.qc_metric_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        summary = outputs.get("qc_integration_summary", pd.DataFrame())
        def metric_value(name, default="-"):
            if summary is None or summary.empty or "metric" not in summary.columns:
                return default
            hit = summary[summary["metric"].astype(str).eq(name)]
            return hit["value"].iloc[0] if not hit.empty else default
        tiles = [
            ("QC table", "yes" if str(metric_value("qc_table_loaded", False)).lower() == "true" else "no", "artifact context"),
            ("QC rows", metric_value("qc_rows"), "recordings with QC"),
            ("QC metrics", metric_value("numeric_qc_metrics"), "numeric indicators"),
            ("Families", metric_value("artifact_families_detected"), "artifact domains"),
            ("Monitor pairs", metric_value("feature_qc_pairs_abs_rho_ge_0_30", 0), "abs rho >= .30"),
            ("Review pairs", metric_value("feature_qc_pairs_abs_rho_ge_0_50", 0), "abs rho >= .50"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.qc_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx, 0)

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
            active_df = self.analysis_df if self.analysis_df is not None else self.feature_df
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            path = plot_selected_feature_qc_scatter(active_df, self.qc_df, feature, metric, plots_dir / "selected_feature_qc_scatter.png")
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
                "<b>Next check</b><br>Inspect family burden, feature-QC associations, and selected feature x QC scatter plots."
            ),
            "qc_family_burden": (
                "<b>What it shows</b><br>Which artifact families have elevated QC metrics across recordings.<br><br>"
                "<b>Concerning pattern</b><br>High burden in additive, gain, channel, reverberation, distortion, or temporal families indicates acquisition effects that may bias feature interpretation.<br><br>"
                "<b>Do not overinterpret</b><br>High family burden does not automatically mean recordings are unusable; it means downstream features need artifact-aware interpretation.<br><br>"
                "<b>Next check</b><br>Use Feature x QC-family heatmap to see which acoustic features are sensitive to that family."
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
                "<b>Next check</b><br>Inspect top feature-QC associations and selected feature x selected QC scatter."
            ),
            "qc_top_feature_associations": (
                "<b>What it shows</b><br>The strongest individual feature-QC metric associations ranked by |Spearman rho|.<br><br>"
                "<b>Concerning pattern</b><br>|rho| >= .30 warrants monitoring; |rho| >= .50 should be reviewed before using the feature in ML or clinical interpretation.<br><br>"
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

        self.relationship_note = QLabel("Run Feature Analysis to populate relationship plots and tables. Use this menu for feature-feature structure only; outcome screening, task review, and QC sensitivity are handled separately.")
        self.relationship_note.setWordWrap(True)
        self.relationship_note.setStyleSheet(f"color:{MUTED}; background:#F7FAFD; border:1px solid {LINE}; border-radius:8px; padding:10px;")
        card.layout.addWidget(self.relationship_note)

        rel_split = QHBoxLayout()
        rel_split.setSpacing(14)

        plot_panel = QFrame()
        plot_panel.setStyleSheet(f"QFrame {{ background:#F8FBFE; border:1px solid {LINE}; border-radius:12px; }}")
        plot_panel_layout = QVBoxLayout(plot_panel)
        plot_panel_layout.setContentsMargins(14, 14, 14, 14)
        plot_panel_layout.setSpacing(10)

        plot_header = QHBoxLayout()
        plot_header.setSpacing(10)
        plot_title = QLabel("Relationship plot")
        plot_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        plot_header.addWidget(plot_title)

        self.relationship_plot_combo = QComboBox()
        self.relationship_plot_combo.setMinimumWidth(360)
        self.relationship_plot_combo.addItem("Correlation heatmap", "relationship_correlation_heatmap")
        self.relationship_plot_combo.addItem("Top redundant pairs", "relationship_redundant_pairs")
        self.relationship_plot_combo.addItem("Family/block matrix", "relationship_family_matrix")
        self.relationship_plot_combo.addItem("PCA scree", "relationship_pca_scree")
        self.relationship_plot_combo.addItem("PCA recording map", "relationship_pca_scores")
        self.relationship_plot_combo.addItem("PCA top loadings", "relationship_pca_loadings")
        self.relationship_plot_combo.addItem("Selected feature links", "selected_feature_correlations")
        plot_header.addWidget(self.relationship_plot_combo, 1)

        show_btn = QPushButton("Show")
        show_btn.setProperty("secondary", True)
        show_btn.clicked.connect(lambda: self.preview_relationship_plot(self.relationship_plot_combo.currentData()))
        plot_header.addWidget(show_btn)

        regen = QPushButton("Regenerate")
        regen.clicked.connect(self.regenerate_overview_plots)
        plot_header.addWidget(regen)

        plot_header.addStretch(1)
        open_btn = QPushButton("Open current plot")
        open_btn.setProperty("secondary", True)
        open_btn.clicked.connect(self.open_current_relationship_plot)
        plot_header.addWidget(open_btn)
        plot_panel_layout.addLayout(plot_header)

        self.relationship_plot_caption = QLabel(
            "Feature Relationships uses only feature-feature structure plots: correlation heatmap, redundant pairs, family/block structure, PCA summaries, and one selected-feature link plot. It does not duplicate Task Review, Outcome Screening, QC Integration, Missingness, or ML Export."
        )
        self.relationship_plot_caption.setWordWrap(True)
        self.relationship_plot_caption.setStyleSheet(f"color:{MUTED}; background:#FFFFFF; border:1px solid {LINE}; border-radius:8px; padding:9px;")
        plot_panel_layout.addWidget(self.relationship_plot_caption)

        selectors = QHBoxLayout()
        selectors.setSpacing(10)
        selectors.addWidget(QLabel("Selected feature:"))
        self.relationship_feature_combo = QComboBox()
        self.relationship_feature_combo.setMinimumWidth(320)
        self.relationship_feature_combo.currentIndexChanged.connect(lambda _=0: self.preview_relationship_plot("selected_feature_correlations"))
        selectors.addWidget(self.relationship_feature_combo, 1)
        selectors.addWidget(QLabel("Task focus:"))
        self.relationship_task_combo = QComboBox()
        self.relationship_task_combo.setMinimumWidth(220)
        self.relationship_task_combo.addItem("All tasks / not available")
        selectors.addWidget(self.relationship_task_combo)
        plot_panel_layout.addLayout(selectors)

        self.relationship_plot_preview = QLabel("Run Feature Analysis, then choose one relationship plot.")
        self.relationship_plot_preview.setAlignment(Qt.AlignCenter)
        self.relationship_plot_preview.setMinimumHeight(520)
        self.relationship_plot_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.relationship_plot_preview.setStyleSheet(f"QLabel {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; color:{MUTED}; padding:16px; }}")
        plot_panel_layout.addWidget(self.relationship_plot_preview, 1)

        self.relationship_interpretation_label = QLabel("Select a plot to see structured interpretation guidance.")
        self.relationship_interpretation_label.setWordWrap(True)
        self.relationship_interpretation_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.relationship_interpretation_label.setStyleSheet(f"QLabel {{ background:#FFFFFF; color:{INK}; border:1px solid {LINE}; border-radius:10px; padding:12px; font-size:12px; line-height:140%; }}")
        plot_panel_layout.addWidget(self.relationship_interpretation_label)

        rel_split.addWidget(plot_panel, 1)

        side_panel = QFrame()
        side_panel.setStyleSheet(f"QFrame {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:12px; }}")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        side_title = QLabel("Relationship snapshot")
        side_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        side_layout.addWidget(side_title)
        side_note = QLabel("Compact redundancy and PCA metrics. Use these as navigation cues; detailed correlation, module, family-matrix, PCA, and loading tables remain below the plot.")
        side_note.setWordWrap(True)
        side_note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent; font-size:12px;")
        side_layout.addWidget(side_note)
        self.relationship_metric_grid = QGridLayout()
        self.relationship_metric_grid.setHorizontalSpacing(8)
        self.relationship_metric_grid.setVerticalSpacing(8)
        side_layout.addLayout(self.relationship_metric_grid)
        side_layout.addStretch(1)
        side_panel.setFixedWidth(300)
        rel_split.addWidget(side_panel)
        card.layout.addLayout(rel_split)

        tables_header = QLabel("Detailed relationship tables")
        tables_header.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; padding-top:8px;")
        card.layout.addWidget(tables_header)

        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {LINE}; border-radius: 8px; background: #FFFFFF; }}
            QTabBar::tab {{ background:#F7FAFD; color:{NAVY}; border:1px solid {LINE}; border-bottom:none; padding:8px 14px; min-height:22px; }}
            QTabBar::tab:selected {{ background:#FFFFFF; color:{NAVY}; border-top:2px solid {TEAL}; font-weight:700; }}
            QTabBar::tab:hover {{ background:#F3FAF9; color:{NAVY}; }}
        """)
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
        card.layout.addWidget(tabs)

        layout.addWidget(card)
        return self._wrap_scroll(body)

    def update_relationships_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "relationship_summary_table"):
            return
        while self.relationship_metric_grid.count():
            item = self.relationship_metric_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        summary = outputs.get("feature_relationship_summary", pd.DataFrame())
        def metric(name: str, default: object = "-") -> object:
            if summary is None or summary.empty or "metric" not in summary.columns:
                return default
            row = summary.loc[summary["metric"].astype(str).eq(name)]
            return row["value"].iloc[0] if not row.empty else default
        tiles = [
            ("Numeric features", metric("numeric_features"), "usable for relationships"),
            ("Pairwise links", metric("feature_pairs_evaluated"), "Spearman pairs"),
            ("|rho| >= .80", metric("redundant_pairs_abs_rho_ge_0_80"), "strong redundancy"),
            ("Modules", metric("correlation_modules_abs_rho_ge_0_70"), "connected blocks"),
            ("PC1 variance", metric("pc1_variance_percent"), "% if PCA available"),
            ("PC1-PC3", metric("pc1_pc3_cumulative_percent"), "cumulative variance"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.relationship_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx, 0)
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
                "<b>Concerning pattern</b><br>|rho| >= .80 suggests near-duplicate information. |rho| >= .90 often indicates variables should not all enter the same small-sample ML model.<br><br>"
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


    def _task_review_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        card = Card(
            "Task Review",
            "Task-aware dataset inspection. This menu summarizes task availability, task x subject coverage, task x diagnosis/severity context, and task-specific feature support when task metadata is available."
        )

        self.task_note = QLabel("Run Feature Analysis with task metadata available. If no task column can be detected or merged, this menu will report that task analysis is not available.")
        self.task_note.setWordWrap(True)
        self.task_note.setStyleSheet(f"color:{MUTED}; background:#F7FAFD; border:1px solid {LINE}; border-radius:8px; padding:10px;")
        card.layout.addWidget(self.task_note)

        task_split = QHBoxLayout()
        task_split.setSpacing(14)

        plot_panel = QFrame()
        plot_panel.setStyleSheet(f"QFrame {{ background:#F8FBFE; border:1px solid {LINE}; border-radius:12px; }}")
        plot_panel_layout = QVBoxLayout(plot_panel)
        plot_panel_layout.setContentsMargins(14, 14, 14, 14)
        plot_panel_layout.setSpacing(10)

        plot_header = QHBoxLayout()
        plot_header.setSpacing(10)
        plot_title = QLabel("Task review")
        plot_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        plot_header.addWidget(plot_title)
        self.task_review_combo = QComboBox()
        self.task_review_combo.setMinimumWidth(360)
        self.task_review_combo.addItem("Task coverage", "task_counts")
        self.task_review_combo.addItem("Task x subject coverage", "task_subject_matrix")
        self.task_review_combo.addItem("Task x clinical context", "task_clinical_context")
        self.task_review_combo.addItem("Task counts within clinical value", "task_clinical_filtered_counts")
        self.task_review_combo.addItem("Task x diagnosis/severity", "task_label_context")
        self.task_review_combo.addItem("Task feature support", "task_feature_support")
        plot_header.addWidget(self.task_review_combo, 1)
        show_btn = QPushButton("Show")
        show_btn.setProperty("secondary", True)
        show_btn.clicked.connect(lambda: self.preview_task_review_plot(self.task_review_combo.currentData()))
        plot_header.addWidget(show_btn)
        regen = QPushButton("Regenerate")
        regen.clicked.connect(lambda: self.update_task_review_dashboard(getattr(self, "outputs", {})))
        plot_header.addWidget(regen)
        plot_header.addStretch(1)
        open_btn = QPushButton("Open current plot")
        open_btn.setProperty("secondary", True)
        open_btn.clicked.connect(self.open_current_task_review_plot)
        plot_header.addWidget(open_btn)
        plot_panel_layout.addLayout(plot_header)

        self.task_review_caption = QLabel("Task Review is the dedicated place for task axes and task selection. Downstream menus can use task focus controls, but detailed task interpretation belongs here.")
        self.task_review_caption.setWordWrap(True)
        self.task_review_caption.setStyleSheet(f"color:{MUTED}; background:#FFFFFF; border:1px solid {LINE}; border-radius:8px; padding:9px;")
        plot_panel_layout.addWidget(self.task_review_caption)

        selectors = QHBoxLayout()
        selectors.addWidget(QLabel("Task focus:"))
        self.task_focus_combo = QComboBox()
        self.task_focus_combo.setMinimumWidth(320)
        self.task_focus_combo.addItem("All tasks / not available")
        self.task_focus_combo.currentIndexChanged.connect(lambda _=0: self.preview_task_review_plot(self.task_review_combo.currentData() if hasattr(self, 'task_review_combo') else 'task_counts'))
        selectors.addWidget(self.task_focus_combo, 1)
        selectors.addWidget(QLabel("Clinical context:"))
        self.clinical_context_combo = QComboBox()
        self.clinical_context_combo.setMinimumWidth(260)
        self.clinical_context_combo.addItem("Auto / not available")
        self.clinical_context_combo.currentIndexChanged.connect(lambda _=0: self._refresh_clinical_value_combo(self._active_analysis_table()))
        selectors.addWidget(self.clinical_context_combo, 1)
        selectors.addWidget(QLabel("Value:"))
        self.clinical_value_combo = QComboBox()
        self.clinical_value_combo.setMinimumWidth(220)
        self.clinical_value_combo.addItem("All values")
        selectors.addWidget(self.clinical_value_combo, 1)
        selectors.addStretch(1)
        plot_panel_layout.addLayout(selectors)

        self.task_review_preview = QLabel("Run Feature Analysis, then choose one task review plot.")
        self.task_review_preview.setAlignment(Qt.AlignCenter)
        self.task_review_preview.setMinimumHeight(520)
        self.task_review_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.task_review_preview.setStyleSheet(f"QLabel {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; color:{MUTED}; padding:16px; }}")
        plot_panel_layout.addWidget(self.task_review_preview, 1)

        self.task_review_interpretation = QLabel("Select a task plot to see interpretation guidance.")
        self.task_review_interpretation.setWordWrap(True)
        self.task_review_interpretation.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.task_review_interpretation.setStyleSheet(f"QLabel {{ background:#FFFFFF; color:{INK}; border:1px solid {LINE}; border-radius:10px; padding:12px; font-size:12px; line-height:140%; }}")
        plot_panel_layout.addWidget(self.task_review_interpretation)
        task_split.addWidget(plot_panel, 1)

        side_panel = QFrame()
        side_panel.setStyleSheet(f"QFrame {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:12px; }}")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        side_title = QLabel("Task snapshot")
        side_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        side_layout.addWidget(side_title)
        side_note = QLabel("Compact task-context metrics. If task metadata is unavailable, this panel reports that explicitly.")
        side_note.setWordWrap(True)
        side_note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent; font-size:12px;")
        side_layout.addWidget(side_note)
        self.task_metric_grid = QGridLayout()
        self.task_metric_grid.setHorizontalSpacing(8)
        self.task_metric_grid.setVerticalSpacing(8)
        side_layout.addLayout(self.task_metric_grid)
        side_layout.addStretch(1)
        side_panel.setFixedWidth(300)
        task_split.addWidget(side_panel)
        card.layout.addLayout(task_split)

        tables_header = QLabel("Detailed task tables")
        tables_header.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; padding-top:8px;")
        card.layout.addWidget(tables_header)

        tabs = QTabWidget()
        self.task_summary_table = self._simple_table()
        self.task_subject_table = self._simple_table()
        self.task_diagnosis_table = self._simple_table()
        self.task_feature_support_table = self._simple_table()
        self.local_context_detection_table = self._simple_table()
        self.local_severity_preset_table = self._simple_table()
        tabs.addTab(self.task_summary_table, "Task summary")
        tabs.addTab(self.task_subject_table, "Task x subject")
        tabs.addTab(self.task_diagnosis_table, "Task x clinical group")
        tabs.addTab(self.task_feature_support_table, "Task feature support")
        tabs.addTab(self.local_context_detection_table, "Context detection")
        tabs.addTab(self.local_severity_preset_table, "Severity presets")
        card.layout.addWidget(tabs)

        layout.addWidget(card)
        return self._wrap_scroll(body)

    def _longitudinal_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        card = Card(
            "Longitudinal / Iterations",
            "Repeated-measure and visit-context inspection. This menu checks whether subjects have multiple sessions, visits, iterations, or recording dates that can support longitudinal or within-subject comparisons."
        )

        self.longitudinal_note = QLabel("Run Feature Analysis. If subject, session/visit, iteration, or date fields are unavailable, this menu reports what is missing instead of guessing.")
        self.longitudinal_note.setWordWrap(True)
        self.longitudinal_note.setStyleSheet(f"color:{MUTED}; background:#F7FAFD; border:1px solid {LINE}; border-radius:8px; padding:10px;")
        card.layout.addWidget(self.longitudinal_note)

        long_split = QHBoxLayout()
        long_split.setSpacing(14)

        plot_panel = QFrame()
        plot_panel.setStyleSheet(f"QFrame {{ background:#F8FBFE; border:1px solid {LINE}; border-radius:12px; }}")
        plot_panel_layout = QVBoxLayout(plot_panel)
        plot_panel_layout.setContentsMargins(14, 14, 14, 14)
        plot_panel_layout.setSpacing(10)

        plot_header = QHBoxLayout()
        plot_title = QLabel("Longitudinal review")
        plot_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        plot_header.addWidget(plot_title)
        self.longitudinal_view_combo = QComboBox()
        self.longitudinal_view_combo.setMinimumWidth(360)
        self.longitudinal_view_combo.addItem("Subject repeats", "long_subject_records")
        self.longitudinal_view_combo.addItem("Session / visit structure", "long_session_matrix")
        self.longitudinal_view_combo.addItem("Iteration coverage", "long_iteration_counts")
        self.longitudinal_view_combo.addItem("Visit-date coverage", "long_date_timeline")
        plot_header.addWidget(self.longitudinal_view_combo, 1)
        show_btn = QPushButton("Show")
        show_btn.setProperty("secondary", True)
        show_btn.clicked.connect(lambda: self.preview_longitudinal_plot(self.longitudinal_view_combo.currentData()))
        plot_header.addWidget(show_btn)
        regen = QPushButton("Regenerate")
        regen.clicked.connect(lambda: self.update_longitudinal_dashboard(getattr(self, "outputs", {})))
        plot_header.addWidget(regen)
        plot_header.addStretch(1)
        open_btn = QPushButton("Open current plot")
        open_btn.setProperty("secondary", True)
        open_btn.clicked.connect(self.open_current_longitudinal_plot)
        plot_header.addWidget(open_btn)
        plot_panel_layout.addLayout(plot_header)

        self.longitudinal_caption = QLabel("This menu separates repeated iterations, visits/sessions, and dates from generic reliability. It is a readiness screen for future longitudinal models, not a model fitting page.")
        self.longitudinal_caption.setWordWrap(True)
        self.longitudinal_caption.setStyleSheet(f"color:{MUTED}; background:#FFFFFF; border:1px solid {LINE}; border-radius:8px; padding:9px;")
        plot_panel_layout.addWidget(self.longitudinal_caption)

        selectors = QHBoxLayout()
        selectors.addWidget(QLabel("Subject focus:"))
        self.subject_focus_combo = QComboBox()
        self.subject_focus_combo.setMinimumWidth(320)
        self.subject_focus_combo.addItem("All subjects / not available")
        self.subject_focus_combo.currentIndexChanged.connect(lambda _=0: self.preview_longitudinal_plot(self.longitudinal_view_combo.currentData() if hasattr(self, 'longitudinal_view_combo') else 'long_subject_records'))
        selectors.addWidget(self.subject_focus_combo, 1)
        selectors.addStretch(1)
        plot_panel_layout.addLayout(selectors)

        self.longitudinal_preview = QLabel("Run Feature Analysis, then choose one longitudinal review plot.")
        self.longitudinal_preview.setAlignment(Qt.AlignCenter)
        self.longitudinal_preview.setMinimumHeight(520)
        self.longitudinal_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.longitudinal_preview.setStyleSheet(f"QLabel {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; color:{MUTED}; padding:16px; }}")
        plot_panel_layout.addWidget(self.longitudinal_preview, 1)

        self.longitudinal_interpretation = QLabel("Select a longitudinal plot to see interpretation guidance.")
        self.longitudinal_interpretation.setWordWrap(True)
        self.longitudinal_interpretation.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.longitudinal_interpretation.setStyleSheet(f"QLabel {{ background:#FFFFFF; color:{INK}; border:1px solid {LINE}; border-radius:10px; padding:12px; font-size:12px; line-height:140%; }}")
        plot_panel_layout.addWidget(self.longitudinal_interpretation)
        long_split.addWidget(plot_panel, 1)

        side_panel = QFrame()
        side_panel.setStyleSheet(f"QFrame {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:12px; }}")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        side_title = QLabel("Longitudinal snapshot")
        side_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        side_layout.addWidget(side_title)
        side_note = QLabel("Compact repeated-measure metrics. If dates, visits, sessions, or iterations are unavailable, this panel reports that explicitly.")
        side_note.setWordWrap(True)
        side_note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent; font-size:12px;")
        side_layout.addWidget(side_note)
        self.longitudinal_metric_grid = QGridLayout()
        self.longitudinal_metric_grid.setHorizontalSpacing(8)
        self.longitudinal_metric_grid.setVerticalSpacing(8)
        side_layout.addLayout(self.longitudinal_metric_grid)
        side_layout.addStretch(1)
        side_panel.setFixedWidth(300)
        long_split.addWidget(side_panel)
        card.layout.addLayout(long_split)

        tables_header = QLabel("Detailed longitudinal / iteration tables")
        tables_header.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; padding-top:8px;")
        card.layout.addWidget(tables_header)

        tabs = QTabWidget()
        self.long_subject_table = self._simple_table()
        self.long_session_table = self._simple_table()
        self.long_iteration_table = self._simple_table()
        self.long_date_table = self._simple_table()
        tabs.addTab(self.long_subject_table, "Subject repeats")
        tabs.addTab(self.long_session_table, "Session / visit")
        tabs.addTab(self.long_iteration_table, "Iterations")
        tabs.addTab(self.long_date_table, "Dates")
        card.layout.addWidget(tabs)

        layout.addWidget(card)
        return self._wrap_scroll(body)


    def _active_analysis_table(self) -> pd.DataFrame:
        if getattr(self, "analysis_df", None) is not None and not self.analysis_df.empty:
            return self.analysis_df
        if getattr(self, "feature_df", None) is not None:
            return self.feature_df
        return pd.DataFrame()

    def _first_existing_col(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        if df is None or df.empty:
            return None
        lookup = {str(c).lower(): str(c) for c in df.columns}
        for c in candidates:
            if c in df.columns:
                return c
            if c.lower() in lookup:
                return lookup[c.lower()]
        return None

    def _task_col(self, df: pd.DataFrame) -> str | None:
        return self._first_existing_col(df, ["task", "task_name", "metadata__task", "metadata__task_name"])

    def _subject_col(self, df: pd.DataFrame) -> str | None:
        return self._first_existing_col(df, ["subject_id", "SubjectID", "subject", "participant_id", "metadata__subject_id", "metadata__SubjectID"])

    def _session_col(self, df: pd.DataFrame) -> str | None:
        return self._first_existing_col(df, ["session_id", "visit_id", "Clinical Visit ID", "metadata__session_id", "metadata__visit_id", "metadata__Clinical Visit ID"])

    def _iteration_col(self, df: pd.DataFrame) -> str | None:
        return self._first_existing_col(df, ["iteration", "Iteration", "metadata__iteration", "metadata__Iteration"])

    def _date_col(self, df: pd.DataFrame) -> str | None:
        return self._first_existing_col(df, ["recording_date", "visit_date", "assessment_date", "Recording date", "metadata__recording_date", "metadata__Recording date"])

    def _refresh_focus_combos(self) -> None:
        df = self._active_analysis_table()
        task_col = self._task_col(df)
        task_values = []
        if task_col:
            task_values = sorted([str(x) for x in df[task_col].dropna().unique().tolist() if str(x).strip()])
        for attr in ["task_focus_combo", "relationship_task_combo"]:
            combo = getattr(self, attr, None)
            if combo is None:
                continue
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All tasks" if task_values else "All tasks / not available")
            for v in task_values[:300]:
                combo.addItem(v)
            if current:
                ix = combo.findText(current)
                if ix >= 0:
                    combo.setCurrentIndex(ix)
            combo.blockSignals(False)

        subj_col = self._subject_col(df)
        subj_values = []
        if subj_col:
            subj_values = sorted([str(x) for x in df[subj_col].dropna().unique().tolist() if str(x).strip()])
        combo = getattr(self, "subject_focus_combo", None)
        if combo is not None:
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All subjects" if subj_values else "All subjects / not available")
            for v in subj_values[:300]:
                combo.addItem(v)
            if current:
                ix = combo.findText(current)
                if ix >= 0:
                    combo.setCurrentIndex(ix)
            combo.blockSignals(False)


    def _safe_context_candidates(self, df: pd.DataFrame) -> dict[str, str]:
        """Detect optional metadata columns without modifying analysis data.

        This helper is intentionally permissive and never raises. Missing metadata
        simply produces fewer options in local plot controls.
        """
        candidates: dict[str, str] = {}
        if df is None or df.empty:
            return candidates
        def add(label: str, aliases: list[str]) -> None:
            col = self._first_existing_col(df, aliases)
            if col and col in df.columns and df[col].notna().sum() > 0:
                candidates[label] = col

        add("Diagnosis", ["diagnosis", "Diagnosis", "diagnostic_group", "metadata__diagnosis", "metadata__Diagnosis"])
        add("ALSFRS total", ["ALSFRS total score", "ALSFRS-R total score", "ALSFRS total", "ALSFRS-R total", "alsfrs_total", "alsfrsr_total", "metadata__ALSFRS total score"])
        add("ALSFRS bulbar", ["ALSFRS bulbar", "ALSFRS-R bulbar", "ALSFRS bulbar score", "ALSFRS-R bulbar score", "bulbar", "bulbar_score", "alsfrs_bulbar", "alsfrsr_bulbar", "metadata__ALSFRS bulbar", "metadata__ALSFRS-R bulbar"])
        add("ALSBDI", ["ALSBDI", "ALSBDI score", "ALS Bulbar Dysfunction Index", "alsbdi", "alsbdi_score", "metadata__ALSBDI", "metadata__ALSBDI score"])
        add("Sex / gender", ["sex", "Sex", "gender", "Gender", "metadata__Sex", "metadata__sex"])
        add("Session / visit", ["session_id", "visit_id", "Clinical Visit ID", "metadata__session_id", "metadata__visit_id", "metadata__Clinical Visit ID"])
        add("Iteration", ["iteration", "Iteration", "metadata__iteration", "metadata__Iteration"])
        return candidates

    def _bulbar_severity_local(self, score) -> str:
        try:
            val = pd.to_numeric(pd.Series([score]), errors="coerce").iloc[0]
        except Exception:
            val = pd.NA
        if pd.isna(val):
            return "no_score_or_control"
        if val > 12:
            return "invalid_above_12"
        if val < 0:
            return "invalid_below_0"
        if val >= 11:
            return "near_normal_11_12"
        if val >= 9:
            return "mild_9_10"
        if val >= 6:
            return "moderate_6_8"
        return "severe_0_5"

    def _alsfrs_total_severity_local(self, score) -> str:
        val = pd.to_numeric(pd.Series([score]), errors="coerce").iloc[0]
        if pd.isna(val):
            return "no_score_or_control"
        if val > 48:
            return "invalid_above_48"
        if val < 0:
            return "invalid_below_0"
        if val >= 37:
            return "mild_or_high_function_37_48"
        if val >= 25:
            return "moderate_25_36"
        return "severe_0_24"

    def _clinical_context_series(self, df: pd.DataFrame) -> tuple[pd.Series | None, str]:
        """Return a local clinical grouping series for plotting only."""
        if df is None or df.empty or not hasattr(self, "clinical_context_combo"):
            return None, "clinical context"
        label = self.clinical_context_combo.currentText()
        candidates = self._safe_context_candidates(df)
        if label in {"", "Auto / not available"}:
            if "Diagnosis" in candidates:
                label = "Diagnosis"
            elif "ALSFRS bulbar" in candidates:
                label = "ALSFRS bulbar severity"
            else:
                return None, "clinical context"
        if label == "ALSFRS bulbar severity":
            col = candidates.get("ALSFRS bulbar")
            if not col:
                return None, label
            return df[col].map(self._bulbar_severity_local).astype("string"), label
        if label == "ALSFRS total severity":
            col = candidates.get("ALSFRS total")
            if not col:
                return None, label
            return df[col].map(self._alsfrs_total_severity_local).astype("string"), label
        if label == "ALSBDI severity":
            col = candidates.get("ALSBDI")
            if not col:
                return None, label
            score = pd.to_numeric(df[col], errors="coerce")
            out = pd.Series("score_detected_cutoffs_pending", index=df.index, dtype="string")
            out[score.isna()] = "no_score_or_control"
            return out, label
        col = candidates.get(label)
        if col:
            return df[col].astype("string"), label
        return None, label

    def _refresh_clinical_context_controls(self, df: pd.DataFrame) -> None:
        if not hasattr(self, "clinical_context_combo"):
            return
        current = self.clinical_context_combo.currentText()
        candidates = self._safe_context_candidates(df)
        options = ["Auto / not available"]
        for label in ["Diagnosis", "ALSFRS bulbar severity", "ALSFRS total severity", "ALSBDI severity", "Sex / gender", "Session / visit", "Iteration"]:
            if label == "ALSFRS bulbar severity" and "ALSFRS bulbar" not in candidates:
                continue
            if label == "ALSFRS total severity" and "ALSFRS total" not in candidates:
                continue
            if label == "ALSBDI severity" and "ALSBDI" not in candidates:
                continue
            if label in {"Diagnosis", "Sex / gender", "Session / visit", "Iteration"} and label not in candidates:
                continue
            options.append(label)
        self.clinical_context_combo.blockSignals(True)
        self.clinical_context_combo.clear()
        for opt in options:
            self.clinical_context_combo.addItem(opt)
        ix = self.clinical_context_combo.findText(current)
        if ix >= 0:
            self.clinical_context_combo.setCurrentIndex(ix)
        self.clinical_context_combo.blockSignals(False)
        self._refresh_clinical_value_combo(df)

    def _refresh_clinical_value_combo(self, df: pd.DataFrame) -> None:
        if not hasattr(self, "clinical_value_combo"):
            return
        current = self.clinical_value_combo.currentText()
        series, label = self._clinical_context_series(df)
        self.clinical_value_combo.blockSignals(True)
        self.clinical_value_combo.clear()
        self.clinical_value_combo.addItem("All values")
        if series is not None:
            for v in sorted([str(x) for x in series.dropna().unique().tolist() if str(x).strip()])[:200]:
                self.clinical_value_combo.addItem(v)
        ix = self.clinical_value_combo.findText(current)
        if ix >= 0:
            self.clinical_value_combo.setCurrentIndex(ix)
        self.clinical_value_combo.blockSignals(False)

    def _local_context_detection_table(self, df: pd.DataFrame) -> pd.DataFrame:
        candidates = self._safe_context_candidates(df)
        rows = []
        for label in ["Diagnosis", "ALSFRS total", "ALSFRS bulbar", "ALSBDI", "Sex / gender", "Session / visit", "Iteration"]:
            col = candidates.get(label, "")
            if col:
                s = df[col]
                rows.append({"context": label, "source_column": col, "n_nonmissing": int(s.notna().sum()), "n_unique": int(s.dropna().nunique()), "status": "detected"})
            else:
                rows.append({"context": label, "source_column": "", "n_nonmissing": 0, "n_unique": 0, "status": "not_detected"})
        return pd.DataFrame(rows)

    def _local_severity_preset_table(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"preset": "ALSFRS bulbar project preset", "source_score": "ALSFRS bulbar", "valid_range": "0-12", "bin": "near_normal_11_12", "rule": "score >= 11"},
            {"preset": "ALSFRS bulbar project preset", "source_score": "ALSFRS bulbar", "valid_range": "0-12", "bin": "mild_9_10", "rule": "score >= 9 and <= 10"},
            {"preset": "ALSFRS bulbar project preset", "source_score": "ALSFRS bulbar", "valid_range": "0-12", "bin": "moderate_6_8", "rule": "score >= 6 and <= 8"},
            {"preset": "ALSFRS bulbar project preset", "source_score": "ALSFRS bulbar", "valid_range": "0-12", "bin": "severe_0_5", "rule": "score <= 5"},
            {"preset": "ALSFRS bulbar validity", "source_score": "ALSFRS bulbar", "valid_range": "0-12", "bin": "invalid_above_12", "rule": "score > 12"},
            {"preset": "ALSFRS total configurable preset", "source_score": "ALSFRS total", "valid_range": "0-48", "bin": "mild_or_high_function_37_48 / moderate_25_36 / severe_0_24", "rule": "temporary preset; revise later if needed"},
            {"preset": "ALSBDI pending preset", "source_score": "ALSBDI", "valid_range": "pending", "bin": "score_detected_cutoffs_pending", "rule": "score shown; cutoffs not hard-coded yet"},
        ])

    def update_task_review_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "task_metric_grid"):
            return
        while self.task_metric_grid.count():
            item = self.task_metric_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        df = self._active_analysis_table()
        task_col = self._task_col(df)
        subj_col = self._subject_col(df)
        self._refresh_clinical_context_controls(df)
        clinical_series, clinical_label = self._clinical_context_series(df)
        if df.empty or not task_col:
            tiles = [
                ("Task column", "not available", "load metadata or infer tasks"),
                ("Tasks", 0, "detected task values"),
                ("Subjects", 0, "with task context"),
                ("Rows", len(df), "records / files"),
            ]
            for idx, (title, value, subtitle) in enumerate(tiles):
                self.task_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx, 0)
            empty = pd.DataFrame([{"status": "task metadata not available", "next_step": "load metadata with task labels or run filename/task inference"}])
            self._fill_table(self.task_summary_table, empty)
            self._fill_table(self.task_subject_table, pd.DataFrame())
            self._fill_table(self.task_diagnosis_table, pd.DataFrame())
            self._fill_table(self.task_feature_support_table, pd.DataFrame())
            if hasattr(self, "local_context_detection_table"):
                self._fill_table(self.local_context_detection_table, self._local_context_detection_table(df))
            if hasattr(self, "local_severity_preset_table"):
                self._fill_table(self.local_severity_preset_table, self._local_severity_preset_table())
            self.task_note.setText("Task metadata is not available. Load metadata with a task column, or use filename/task parsing later. Optional clinical metadata detection is still shown below when available.")
            self._task_feature_support_df = pd.DataFrame()
            self.generate_task_review_plots()
            self.preview_task_review_plot("task_counts")
            return

        task_series = df[task_col].astype(str).replace({"nan": ""})
        task_summary = (
            df.assign(_task=task_series)
              .loc[lambda x: x["_task"].str.len() > 0]
              .groupby("_task", dropna=False)
              .size()
              .reset_index(name="n_rows")
              .rename(columns={"_task": "task"})
              .sort_values("n_rows", ascending=False)
        )
        if subj_col:
            task_subject = pd.crosstab(df[subj_col].astype(str), task_series).reset_index().rename(columns={subj_col: "subject_id"})
        else:
            task_subject = pd.DataFrame([{"status": "subject_id not available; task x subject matrix cannot be built"}])
        if clinical_series is not None:
            task_diag = pd.crosstab(task_series, clinical_series.astype(str).fillna("missing")).reset_index().rename(columns={"row_0": "task"})
            task_diag.insert(0, "clinical_context", clinical_label)
        else:
            task_diag = pd.DataFrame([{"status": "diagnosis/severity metadata not available for task context"}])

        feature_cols = []
        if getattr(self, "mapping_df", None) is not None and not self.mapping_df.empty:
            feature_cols = self.mapping_df.loc[self.mapping_df["role"].astype(str).str.contains("Feature", case=False, na=False), "column"].astype(str).tolist()
        feature_cols = [c for c in feature_cols if c in df.columns]
        rows = []
        for task, subdf in df.groupby(task_series):
            if not str(task).strip():
                continue
            n_feats = len(feature_cols)
            mean_missing = float(subdf[feature_cols].isna().mean().mean()) if feature_cols else float("nan")
            rows.append({"task": task, "n_rows": len(subdf), "n_features": n_feats, "mean_feature_missingness": mean_missing})
        task_feature_support = pd.DataFrame(rows).sort_values("n_rows", ascending=False) if rows else pd.DataFrame()

        tiles = [
            ("Tasks", int(task_summary["task"].nunique()), "detected task values"),
            ("Rows", len(df), "records / files"),
            ("Subjects", int(df[subj_col].nunique()) if subj_col else "-", "with task context"),
            ("Largest task", task_summary.iloc[0]["n_rows"] if not task_summary.empty else "-", task_summary.iloc[0]["task"] if not task_summary.empty else "not available"),
            ("Clinical context", "yes" if clinical_series is not None else "no", clinical_label),
            ("Severity presets", "available", "bulbar / total / ALSBDI"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.task_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx, 0)
        self._task_feature_support_df = task_feature_support
        self._fill_table(self.task_summary_table, task_summary)
        self._fill_table(self.task_subject_table, task_subject)
        self._fill_table(self.task_diagnosis_table, task_diag)
        self._fill_table(self.task_feature_support_table, task_feature_support)
        if hasattr(self, "local_context_detection_table"):
            self._fill_table(self.local_context_detection_table, self._local_context_detection_table(df))
        if hasattr(self, "local_severity_preset_table"):
            self._fill_table(self.local_severity_preset_table, self._local_severity_preset_table())
        self.task_note.setText("Task review generated. Clinical context controls are local to this menu and only affect Task Review plots, not the base analysis.")
        self._refresh_focus_combos()
        self.generate_task_review_plots()
        self.preview_task_review_plot(self.task_review_combo.currentData() if hasattr(self, "task_review_combo") else "task_counts")

    def generate_task_review_plots(self) -> None:
        df = self._active_analysis_table()
        self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
        task_col = self._task_col(df)
        subj_col = self._subject_col(df)
        self._refresh_clinical_context_controls(df)
        clinical_series, clinical_label = self._clinical_context_series(df)
        selected_level = self.clinical_value_combo.currentText() if hasattr(self, "clinical_value_combo") else "All values"
        label_col = None
        if clinical_series is not None:
            df = df.copy()
            label_col = "__local_clinical_context__"
            df[label_col] = clinical_series.values
        support = pd.DataFrame()
        if hasattr(self, "task_feature_support_table"):
            # Prefer the table just filled in the GUI when available.
            support = getattr(self, "_task_feature_support_df", pd.DataFrame())
        if not hasattr(self, "plot_paths"):
            self.plot_paths = {}
        self.plot_paths["task_counts"] = str(plot_task_counts(df, task_col, plots_dir / "task_counts.png"))
        self.plot_paths["task_subject_matrix"] = str(plot_task_subject_matrix(df, task_col, subj_col, plots_dir / "task_subject_matrix.png"))
        self.plot_paths["task_clinical_context"] = str(plot_task_clinical_context(df, task_col, clinical_series, clinical_label, plots_dir / "task_clinical_context.png"))
        self.plot_paths["task_clinical_filtered_counts"] = str(plot_task_clinical_filtered_counts(df, task_col, clinical_series, clinical_label, selected_level, plots_dir / "task_clinical_filtered_counts.png"))
        self.plot_paths["task_label_context"] = str(plot_task_label_context(df, task_col, label_col, plots_dir / "task_label_context.png"))
        self.plot_paths["task_feature_support"] = str(plot_task_feature_support(support, plots_dir / "task_feature_support.png"))

    def preview_task_review_plot(self, key: str | None = None) -> None:
        key = key or (self.task_review_combo.currentData() if hasattr(self, "task_review_combo") else "task_counts")
        if not hasattr(self, "plot_paths") or key not in self.plot_paths or not Path(self.plot_paths.get(key, "")).exists():
            self.generate_task_review_plots()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths:
            QMessageBox.information(self, "Plot unavailable", "Run Feature Analysis first, or this task plot could not be generated.")
            return
        path = Path(self.plot_paths[key])
        if not path.exists():
            QMessageBox.information(self, "Plot unavailable", f"Plot file not found:\n{path}")
            return
        self.current_task_review_plot = path
        captions = {
            "task_counts": "Rows by task. Use this to decide whether pooled analysis is dominated by one task or whether task-specific review is required.",
            "task_subject_matrix": "Task x subject coverage. Sparse blocks indicate that task comparisons may be confounded by subject availability.",
            "task_clinical_context": "Task x clinical context. Use this to inspect diagnosis, ALSFRS bulbar severity, ALSFRS total severity, ALSBDI placeholder bins, or other detected grouping variables across tasks.",
            "task_clinical_filtered_counts": "Task counts within a selected clinical value. Use the Clinical context and Value controls to inspect one diagnosis/severity group without changing the base analysis.",
            "task_label_context": "Task x diagnosis/severity context. This legacy view uses the currently selected local clinical context.",
            "task_feature_support": "Task-level feature support. High missingness in one task suggests task-specific feature incompatibility or acquisition issues.",
        }
        if hasattr(self, "task_review_interpretation"):
            self.task_review_interpretation.setText(captions.get(key, "Task review plot."))
        pix = QPixmap(str(path))
        if pix.isNull():
            self.task_review_preview.setText(f"Could not load plot:\n{path}")
            return
        scaled = pix.scaled(self.task_review_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.task_review_preview.setPixmap(scaled)
        self.task_review_preview.setToolTip(str(path))

    def open_current_task_review_plot(self) -> None:
        path = getattr(self, "current_task_review_plot", None)
        if not path:
            QMessageBox.information(self, "No current plot", "Preview a task plot first, then open the full-resolution file.")
            return
        self.open_file(Path(path))

    def update_longitudinal_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "longitudinal_metric_grid"):
            return
        while self.longitudinal_metric_grid.count():
            item = self.longitudinal_metric_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        df = self._active_analysis_table()
        subj_col = self._subject_col(df)
        session_col = self._session_col(df)
        iter_col = self._iteration_col(df)
        date_col = self._date_col(df)

        if df.empty or not subj_col:
            tiles = [
                ("Subject ID", "not available", "required for repeats"),
                ("Repeated subjects", 0, "cannot evaluate"),
                ("Session / visit", "not available", "load or infer"),
                ("Dates", "not available", "load or infer"),
            ]
            for idx, (title, value, subtitle) in enumerate(tiles):
                self.longitudinal_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx, 0)
            self._fill_table(self.long_subject_table, pd.DataFrame([{"status": "subject_id not available; longitudinal review cannot be built"}]))
            self._fill_table(self.long_session_table, pd.DataFrame())
            self._fill_table(self.long_iteration_table, pd.DataFrame())
            self._fill_table(self.long_date_table, pd.DataFrame())
            self.generate_longitudinal_plots()
            self.preview_longitudinal_plot("long_subject_records")
            return

        subj_counts = df.groupby(subj_col).size().reset_index(name="n_records").sort_values("n_records", ascending=False)
        repeated = int((subj_counts["n_records"] > 1).sum())
        if session_col:
            session_table = df.groupby([subj_col, session_col]).size().reset_index(name="n_records").sort_values([subj_col, session_col])
        else:
            session_table = pd.DataFrame([{"status": "session_id/visit_id not available"}])
        if iter_col:
            iter_table = df.groupby([subj_col, iter_col]).size().reset_index(name="n_records").sort_values([subj_col, iter_col])
        else:
            iter_table = pd.DataFrame([{"status": "iteration not available"}])
        if date_col:
            date_table = df[[subj_col, date_col]].dropna().drop_duplicates().sort_values([subj_col, date_col])
        else:
            date_table = pd.DataFrame([{"status": "recording_date/visit_date not available"}])

        tiles = [
            ("Subjects", int(df[subj_col].nunique()), "detected IDs"),
            ("Repeated subjects", repeated, "n_records > 1"),
            ("Session / visit", "yes" if session_col else "no", "available field"),
            ("Iterations", "yes" if iter_col else "no", "available field"),
            ("Dates", "yes" if date_col else "no", "visit/date field"),
            ("Rows", len(df), "records / files"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.longitudinal_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx, 0)
        self._fill_table(self.long_subject_table, subj_counts)
        self._fill_table(self.long_session_table, session_table)
        self._fill_table(self.long_iteration_table, iter_table)
        self._fill_table(self.long_date_table, date_table)
        self.longitudinal_note.setText("Longitudinal / iteration review generated. Use this menu to decide whether repeated-measure, iteration-specific, or date-aware analysis is supported.")
        self._refresh_focus_combos()
        self.generate_longitudinal_plots()
        self.preview_longitudinal_plot(self.longitudinal_view_combo.currentData() if hasattr(self, "longitudinal_view_combo") else "long_subject_records")

    def generate_longitudinal_plots(self) -> None:
        df = self._active_analysis_table()
        self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
        subj_col = self._subject_col(df)
        session_col = self._session_col(df)
        iter_col = self._iteration_col(df)
        date_col = self._date_col(df)
        if not hasattr(self, "plot_paths"):
            self.plot_paths = {}
        self.plot_paths["long_subject_records"] = str(plot_longitudinal_subject_records(df, subj_col, plots_dir / "long_subject_records.png"))
        self.plot_paths["long_session_matrix"] = str(plot_longitudinal_session_matrix(df, subj_col, session_col, plots_dir / "long_session_matrix.png"))
        self.plot_paths["long_iteration_counts"] = str(plot_longitudinal_iteration_counts(df, subj_col, iter_col, plots_dir / "long_iteration_counts.png"))
        self.plot_paths["long_date_timeline"] = str(plot_longitudinal_date_timeline(df, subj_col, date_col, plots_dir / "long_date_timeline.png"))

    def preview_longitudinal_plot(self, key: str | None = None) -> None:
        key = key or (self.longitudinal_view_combo.currentData() if hasattr(self, "longitudinal_view_combo") else "long_subject_records")
        if not hasattr(self, "plot_paths") or key not in self.plot_paths or not Path(self.plot_paths.get(key, "")).exists():
            self.generate_longitudinal_plots()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths:
            QMessageBox.information(self, "Plot unavailable", "Run Feature Analysis first, or this longitudinal plot could not be generated.")
            return
        path = Path(self.plot_paths[key])
        if not path.exists():
            QMessageBox.information(self, "Plot unavailable", f"Plot file not found:\n{path}")
            return
        self.current_longitudinal_plot = path
        captions = {
            "long_subject_records": "Records per subject. Subjects with more than one row can support repeated-measure review, but dates/sessions are still needed for true longitudinal interpretation.",
            "long_session_matrix": "Subject x session/visit coverage. Sparse coverage indicates uneven visit structure and limits direct longitudinal comparisons.",
            "long_iteration_counts": "Subject x iteration counts. Use this to inspect repeated attempts or iterations within subjects.",
            "long_date_timeline": "Visit or recording dates by subject. This is the most direct visual check for longitudinal timing when dates are available.",
        }
        if hasattr(self, "longitudinal_interpretation"):
            self.longitudinal_interpretation.setText(captions.get(key, "Longitudinal review plot."))
        pix = QPixmap(str(path))
        if pix.isNull():
            self.longitudinal_preview.setText(f"Could not load plot:\n{path}")
            return
        scaled = pix.scaled(self.longitudinal_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.longitudinal_preview.setPixmap(scaled)
        self.longitudinal_preview.setToolTip(str(path))

    def open_current_longitudinal_plot(self) -> None:
        path = getattr(self, "current_longitudinal_plot", None)
        if not path:
            QMessageBox.information(self, "No current plot", "Preview a longitudinal plot first, then open the full-resolution file.")
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
        self._add_standard_plot_gallery(
            plot_card.layout,
            "Screening plot",
            "Group/Outcome Screening keeps balance, effect ranking, group/continuous heatmaps, effect-size landscape, and one selected feature x outcome view. These are descriptive screens only.",
            "screening_plot_combo",
            [
                ("Group balance", "screening_group_balance"),
                ("Top screening effects", "screening_effect_ranking"),
                ("Feature x continuous outcome heatmap", "screening_continuous_heatmap"),
                ("Feature x categorical group heatmap", "screening_group_heatmap"),
                ("Effect-size landscape", "screening_effect_landscape"),
                ("Selected feature vs selected outcome/group", "selected_feature_outcome"),
            ],
            "screening_plot_preview",
            "screening_interpretation_label",
            self.preview_screening_plot,
            self.open_current_screening_plot,
            "Run Feature Analysis, then choose one screening plot.",
            500,
        )
        layout.addWidget(plot_card, 1)
        layout.addWidget(card)
        return self._wrap_scroll(body)

    def update_screening_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "screening_summary_table"):
            return
        while self.screening_metric_grid.count():
            item = self.screening_metric_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        summary = outputs.get("screening_summary", pd.DataFrame())
        def metric(name: str, default: object = "-") -> object:
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
            ("Continuous |effect| >= .30", metric("continuous_abs_effect_ge_0_30"), "monitor/review"),
            ("Group |effect| >= .30", metric("group_abs_effect_ge_0_30"), "monitor/review"),
            ("Top effect", "-" if (cont is None or cont.empty) and (cat is None or cat.empty) else round(float(pd.concat([d for d in [cont.get('abs_effect') if cont is not None and not cont.empty else pd.Series(dtype=float), cat.get('abs_effect') if cat is not None and not cat.empty else pd.Series(dtype=float)]], ignore_index=True).max()), 3), "descriptive only"),
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
            active_df = self.analysis_df if self.analysis_df is not None else self.feature_df
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            path = plot_selected_feature_outcome(active_df, feature, variable, plots_dir / "selected_feature_outcome.png")
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
        self._add_standard_plot_gallery(
            plot_card.layout,
            "Reliability plot",
            "Reliability keeps repeated-measure support, ICC-style ranking, within/between variance, family summary, subject record counts, and one selected-feature trajectory.",
            "reliability_plot_combo",
            [
                ("Reliability status counts", "reliability_status_counts"),
                ("ICC ranking", "reliability_icc_ranking"),
                ("Within vs between variance", "reliability_variance_landscape"),
                ("Reliability by family", "reliability_family_summary"),
                ("Subject record counts", "reliability_subject_counts"),
                ("Selected feature trajectory", "selected_feature_reliability"),
            ],
            "reliability_plot_preview",
            "reliability_interpretation_label",
            self.preview_reliability_plot,
            self.open_current_reliability_plot,
            "Run Feature Analysis, then choose one reliability plot.",
            500,
        )
        layout.addWidget(plot_card, 1)
        layout.addWidget(card)
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
        def metric(name: str, default: object = "-") -> object:
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
            ("Stable features", stable, "ICC proxy >= .75"),
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
            active_df = self.analysis_df if self.analysis_df is not None else self.feature_df
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            path = plot_selected_feature_reliability(active_df, feature, plots_dir / "selected_feature_reliability.png")
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
        plot_card = Card(
            "Recommendation plot board",
            "These plots explain the integrated readiness decision. Use them to document why features are exported, held for review, or excluded by default."
        )
        self._add_standard_plot_gallery(
            plot_card.layout,
            "Recommendation plot",
            "Feature Recommendation keeps only integrated readiness plots: category counts, readiness landscape, reason counts, family summary, and ML export manifest summary.",
            "recommendation_plot_combo",
            [
                ("Readiness category counts", "recommendation_counts"),
                ("Readiness landscape", "recommendation_score_landscape"),
                ("Review reason counts", "recommendation_reason_counts"),
                ("Readiness by family", "recommendation_family_summary"),
                ("ML export manifest", "ml_export_manifest_summary"),
            ],
            "recommendation_plot_preview",
            "recommendation_interpretation_label",
            self.preview_recommendation_plot,
            self.open_current_recommendation_plot,
            "Run Feature Analysis, then choose one recommendation plot.",
            500,
        )
        layout.addWidget(plot_card, 1)
        layout.addWidget(card)
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
            "- acoustic_ml_ready.csv and acoustic_features_only.csv\n"
            "- kinematic_ml_ready.csv and kinematic_features_only.csv\n"
            "- multimodal_early_fusion_ml_ready.csv when a safe shared key exists\n"
            "- feature_manifest_unified.csv\n"
            "- row_alignment_report.csv, row_exclusions.csv, ml_export_manifest.json"
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
            {"export_profile": "recommended_only", "n_total_features": n_total, "n_included_features": int((recs.get("readiness_recommendation", pd.Series(dtype=str)).astype(str) == "recommended").sum()) if not recs.empty else 0, "n_held_or_excluded_features": "-", "purpose": "Strictest clean set."},
            {"export_profile": "recommended_plus_caution", "n_total_features": n_total, "n_included_features": int(recs.get("readiness_recommendation", pd.Series(dtype=str)).astype(str).isin(["recommended", "recommended_with_caution"]).sum()) if not recs.empty else 0, "n_held_or_excluded_features": "-", "purpose": "Default starting set for ML."},
            {"export_profile": "review_set", "n_total_features": n_total, "n_included_features": int(recs.get("readiness_recommendation", pd.Series(dtype=str)).astype(str).isin(["recommended", "recommended_with_caution", "review_before_use"]).sum()) if not recs.empty else 0, "n_held_or_excluded_features": "-", "purpose": "Broad sensitivity/review set."},
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
        active_df = self.analysis_df if self.analysis_df is not None else self.feature_df
        profile = self._export_profile_key()
        profile_slug = profile.replace("_", "-")
        base_dir = (self.output_dir if getattr(self, "output_dir", None) else Path(self.output_edit.text().strip()) / "feature_analysis")
        export_dir = base_dir / "exports" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{profile_slug}"
        export_dir.mkdir(parents=True, exist_ok=True)
        manifest, summary, profile_table = self._build_export_preview_tables(outputs)
        mapping = self.collect_mapping_from_table()
        roles = role_lists(mapping)
        id_cols = [c for c in roles.get(ROLE_IDENTIFIER, []) if c in active_df.columns]
        target_cols = [c for c in roles.get(ROLE_TARGET, []) if c in active_df.columns]
        cov_cols = [c for c in roles.get(ROLE_COVARIATE, []) if c in active_df.columns]
        qc_cols = [c for c in roles.get(ROLE_QC, []) if c in active_df.columns]
        include_features = manifest.loc[manifest.get("final_include", False).astype(bool), "feature"].astype(str).tolist() if not manifest.empty and "feature" in manifest.columns else []
        include_features = [c for c in include_features if c in active_df.columns]
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
        write_df("ml_ready_feature_matrix.csv", active_df[matrix_cols].copy() if matrix_cols else pd.DataFrame())
        write_df("ml_target_table.csv", active_df[id_cols + target_cols].copy() if target_cols else pd.DataFrame(columns=id_cols))
        write_df("ml_covariate_table.csv", active_df[id_cols + cov_cols].copy() if cov_cols else pd.DataFrame(columns=id_cols))
        write_df("ml_qc_covariate_table_from_feature_table.csv", active_df[id_cols + qc_cols].copy() if qc_cols else pd.DataFrame(columns=id_cols))
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
        self.log(f"ERROR | {context}: {exc}")

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
            feature_mapping = classify_columns(self.feature_df, table_kind=kind, registry=self.registry_df)
            self.analysis_df, self.mapping_df, self.metadata_join_strategy = self._merge_metadata_context(self.feature_df, feature_mapping)
            self.proposed_mapping_df = self.mapping_df.copy()
            self.mapping_modified = False
            self.mapping_accepted = False
            self.refresh_mapping_table()
            roles = summarize_roles(self.mapping_df)
            self.log(f"Loaded feature table: {self.feature_df.shape[0]} rows x {self.feature_df.shape[1]} columns")
            if self.qc_df is not None:
                self.log(f"Loaded QC table: {self.qc_df.shape[0]} rows x {self.qc_df.shape[1]} columns")
            if self.meta_df is not None:
                self.log(f"Loaded metadata table: {self.meta_df.shape[0]} rows x {self.meta_df.shape[1]} columns")
                self.log(f"Metadata context strategy: {self.metadata_join_strategy}")
            if self.registry_df is not None:
                self.log(f"Loaded registry/policy table: {self.registry_df.shape[0]} rows x {self.registry_df.shape[1]} columns")
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
            combo = NoWheelComboBox()
            combo.setStyleSheet("QComboBox { min-width: 128px; max-width: 145px; padding: 4px 6px; }")
            combo.addItems(ROLE_OPTIONS)
            combo.setCurrentText(str(r["role"]))
            combo.currentTextChanged.connect(self.mark_mapping_modified)
            self.mapping_table.setCellWidget(i, 1, combo)
            for col_idx, value in [
                (2, f"{float(r['confidence']):.2f}"),
                (3, str(r["reason"])),
                (4, str(r["dtype"])),
                (5, f"{float(r['missing_fraction']):.3f}"),
                (6, str(r["unique_values"])),
            ]:
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.mapping_table.setItem(i, col_idx, item)
        self.mapping_table.resizeRowsToContents()
        for row in range(self.mapping_table.rowCount()):
            self.mapping_table.setRowHeight(row, min(max(self.mapping_table.rowHeight(row), 28), 38))
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
        feature_mapping = classify_columns(self.feature_df, table_kind="feature", registry=self.registry_df)
        self.analysis_df, self.mapping_df, self.metadata_join_strategy = self._merge_metadata_context(self.feature_df, feature_mapping)
        self.proposed_mapping_df = self.mapping_df.copy()
        self.mapping_modified = False
        self.mapping_accepted = False
        self.refresh_mapping_table()
        self.log(f"Proposed mapping refreshed. Metadata context strategy: {self.metadata_join_strategy}")

    def accept_mapping_and_continue(self) -> None:
        self.collect_mapping_from_table()
        self.update_mapping_summary()
        roles = role_lists(self.mapping_df)
        n_features = len(roles.get(ROLE_FEATURE, []))
        if n_features == 0:
            QMessageBox.warning(self, "No feature columns selected", "At least one column must be assigned the role 'Feature' before analysis.")
            return
        if not self.output_edit.text().strip():
            QMessageBox.warning(self, "Output folder required", "Please select an output folder before accepting mapping so the accepted mapping can be saved.")
            self.show_page("project")
            return
        n_changed = self._n_mapping_changes_from_proposal()
        if n_changed > 0:
            reply = QMessageBox.question(
                self,
                "Confirm modified column mapping",
                f"You changed {n_changed} column role(s) from the proposed mapping.\n\nSave this accepted mapping and continue to Overview?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                self.log("Column mapping confirmation cancelled; edits remain visible on the mapping page.")
                return
        path = self.save_accepted_mapping()
        self.mapping_modified = False
        self.mapping_accepted = True
        if path:
            self.log(f"Column mapping accepted and saved: {path}")
        self.log(f"Column mapping accepted: {n_features} feature columns selected.")
        self.show_page("overview")

    def update_mapping_summary(self) -> None:
        if self.mapping_df.empty or not hasattr(self, "mapping_summary_label"):
            return
        summary = summarize_roles(self.mapping_df)
        parts = [f"{r['role']}: {int(r['n_columns'])}" for _, r in summary.iterrows()]
        self.mapping_summary_label.setText("Current mapping | " + " | ".join(parts))

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
            self.update_task_review_dashboard(outputs)
            self.update_longitudinal_dashboard(outputs)
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
        pairs = [
            ("current_overview_plot", "overview_plot_preview"),
            ("current_missingness_plot", "missing_plot_preview"),
            ("current_distribution_plot", "dist_plot_preview"),
            ("current_qc_plot", "qc_plot_preview"),
            ("current_relationship_plot", "relationship_plot_preview"),
            ("current_screening_plot", "screening_plot_preview"),
            ("current_reliability_plot", "reliability_plot_preview"),
            ("current_recommendation_plot", "recommendation_plot_preview"),
            ("current_export_plot", "export_plot_preview"),
        ]
        for path_attr, label_attr in pairs:
            path = getattr(self, path_attr, None)
            label = getattr(self, label_attr, None)
            if path and label is not None:
                self._display_plot_image(label, Path(path))

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
