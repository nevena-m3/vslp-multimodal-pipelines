"""VSLP Feature Analysis GUI.

Professional, modality-neutral feature audit interface for acoustic, kinematic,
multimodal, and generic feature tables.
"""
from __future__ import annotations

import sys
import json
import shutil
import re
import traceback
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
        QTabWidget, QProgressBar, QListView
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Feature Analysis GUI requires PySide6. Install with pip install -e '.[gui]'.") from exc

from vslp.analysis.features.column_mapping import (
    ROLE_OPTIONS, ROLE_FEATURE, ROLE_TARGET, ROLE_IDENTIFIER, ROLE_QC, ROLE_COVARIATE, ROLE_IGNORE, classify_columns, infer_table_kind, role_lists, summarize_roles, normalize_name
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
    feature_recommendation_reason_counts, feature_recommendation_family_summary,
    qc_family_from_name
)
from vslp.analysis.features.ml_export_builder import build_ml_export_package
from vslp.analysis.features.plots import (
    plot_role_counts, plot_group_counts, plot_feature_family_counts,
    plot_missingness, plot_feature_availability_heatmap,
    plot_row_missingness_distribution, plot_missingness_by_group,
    plot_missingness_family_summary, plot_comissing_heatmap,
    plot_feature_availability_bars, plot_comissing_pair_bars,
    plot_distribution_review_summary, plot_expected_range_flags,
    plot_distribution_shape_story, plot_distribution_outlier_range_story, plot_row_outlier_story,
    plot_selected_feature_distribution, plot_selected_feature_diagnostic, plot_group_feature_boxplot, plot_distribution_grid, plot_outlier_counts,
    plot_distribution_shape_summary, plot_distribution_shape_landscape, plot_row_outlier_burden, plot_variance_screen,
    plot_overview_readiness_scorecard, plot_dataset_design_tiles,
    plot_overview_dataset_structure, plot_role_mapping_summary,
    plot_feature_quality_landscape, plot_feature_family_quality, plot_subject_task_matrix,
    plot_overview_design_from_tables, plot_subject_task_summary_bars,
    plot_qc_family_burden, plot_qc_metric_distributions, plot_qc_feature_association_heatmap,
    plot_qc_top_feature_associations, plot_qc_missingness_associations,
    plot_qc_row_burden, plot_selected_feature_qc_scatter, plot_qc_artifact_model,
    plot_relationship_correlation_heatmap, plot_relationship_redundant_pairs,
    plot_relationship_family_matrix, plot_relationship_pca_scree,
    plot_relationship_pca_scores, plot_relationship_pca_loadings,
    plot_relationship_dimensionality_profile, plot_selected_feature_correlations,
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
    plot_task_feature_support, plot_task_clinical_context, plot_task_clinical_filtered_counts,
    plot_task_readiness_dashboard, plot_task_clinical_balance_bars,
    plot_task_subject_coverage_summary, plot_task_feature_profile,
    plot_longitudinal_subject_records, plot_longitudinal_readiness,
    plot_longitudinal_visit_timeline, plot_longitudinal_feature_family_trajectory,
    plot_longitudinal_session_matrix, plot_longitudinal_iteration_counts,
    plot_longitudinal_date_timeline
)

APP_VERSION = "v0.122.0"

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
            ("mapping", "o  Feature Mapping"),
            ("metadata_mapping", "o  Metadata Mapping"),
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
        lab.setStyleSheet(f"color:{NAVY}; font-weight:800; background:transparent; border:none; padding:0px;")
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
        self.metadata_mapping_df = pd.DataFrame()
        self.metadata_mapping_accepted = False
        self.analysis_df: Optional[pd.DataFrame] = None
        self.metadata_join_strategy = "feature_table_only"
        self.mapping_modified = False
        self.mapping_accepted = False
        self.outputs: dict[str, pd.DataFrame] = {}
        self.analysis_ready = False
        self.output_dir: Optional[Path] = None
        self.page_keys = ["project", "mapping", "metadata_mapping", "overview", "missing", "dist", "qc", "relationships", "task_review", "longitudinal", "screening", "reliability", "recommendations", "ml_export", "export"]

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
            "metadata_mapping": self._metadata_mapping_page(),
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

    def _analysis_required_pages(self) -> set[str]:
        return {
            "missing", "dist", "qc", "relationships", "task_review",
            "longitudinal", "screening", "reliability", "recommendations",
            "ml_export", "export",
        }

    def _has_full_analysis_outputs(self) -> bool:
        if not bool(getattr(self, "analysis_ready", False)):
            return False
        outputs = getattr(self, "outputs", {}) or {}
        required_any = [
            "feature_distribution_summary", "missingness_by_feature",
            "feature_qc_spearman_correlation", "feature_relationship_summary",
        ]
        return any(k in outputs and isinstance(outputs.get(k), pd.DataFrame) and not outputs.get(k).empty for k in required_any)

    def _prompt_run_analysis_before_menu(self, key: str) -> bool:
        if key not in self._analysis_required_pages():
            return True
        if self._has_full_analysis_outputs():
            return True
        page_label = {
            "missing": "Missingness",
            "dist": "Distributions / Outliers",
            "qc": "QC Integration",
            "relationships": "Feature Relationships",
            "task_review": "Task Review",
            "longitudinal": "Longitudinal / Iterations",
            "screening": "Group / Outcome Screening",
            "reliability": "Reliability",
            "recommendations": "Recommendations",
            "ml_export": "ML Export Builder",
            "export": "Export / Report",
        }.get(key, key)
        reply = QMessageBox.question(
            self,
            "Run Feature Analysis first",
            f"{page_label} uses derived analysis tables and scoped plots.\n\nRun Feature Analysis now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            self.run_analysis()
        else:
            self.log(f"Navigation to {page_label} cancelled because Feature Analysis has not been run yet.")
        return False

    def show_page(self, key: str) -> None:
        if key not in self.page_keys:
            return
        if not self._prompt_run_analysis_before_menu(key):
            return
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
        widget.setMinimumWidth(860)
        return sc

    def _project_step_label(self, number: int, title: str, subtitle: str) -> QFrame:
        step = QFrame()
        step.setStyleSheet(
            f"QFrame {{ background:#F8FBFE; border:1px solid {LINE}; border-radius:14px; }}"
            f"QLabel#StepNum {{ color:#FFFFFF; background:{TEAL}; border-radius:13px; font-weight:900; }}"
            f"QLabel#StepTitle {{ color:{NAVY}; font-size:13px; font-weight:900; border:none; background:transparent; }}"
            f"QLabel#StepSub {{ color:{MUTED}; font-size:11px; border:none; background:transparent; }}"
        )
        lay = QHBoxLayout(step)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(10)
        num = QLabel(str(number)); num.setObjectName("StepNum"); num.setAlignment(Qt.AlignCenter); num.setFixedSize(26, 26)
        texts = QVBoxLayout(); texts.setSpacing(1)
        t = QLabel(title); t.setObjectName("StepTitle")
        s = QLabel(subtitle); s.setObjectName("StepSub"); s.setWordWrap(True)
        texts.addWidget(t); texts.addWidget(s)
        lay.addWidget(num); lay.addLayout(texts, 1)
        return step

    def _project_section_title(self, title: str, subtitle: str | None = None) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color:{NAVY}; font-size:14px; font-weight:900; border:none; background:transparent;")
        lay.addWidget(title_label)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setWordWrap(True)
            sub.setStyleSheet(f"color:{MUTED}; font-size:11px; border:none; background:transparent;")
            lay.addWidget(sub)
        return box

    def _project_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 22, 24, 24)
        layout.setSpacing(14)

        intro = Card("Project setup", "Load source tables and choose analysis output. Filename-derived context is configured in Metadata Mapping when no metadata table is provided.")
        intro.layout.setSpacing(16)

        step_row = QHBoxLayout()
        step_row.setSpacing(12)
        step_row.addWidget(self._project_step_label(1, "Tables", "Feature table required; QC, metadata, registry optional."))
        step_row.addWidget(self._project_step_label(2, "Output", "Select modality and analysis folder."))
        step_row.addWidget(self._project_step_label(3, "Map context", "Use Feature Mapping and Metadata Mapping before analysis."))
        intro.layout.addLayout(step_row)

        intro.layout.addWidget(self._project_section_title("Source tables"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(12)
        self.feature_picker = FilePicker("Primary feature table", optional=False)
        self.qc_picker = FilePicker("QC table", optional=True)
        self.meta_picker = FilePicker("Metadata table", optional=True)
        self.registry_picker = FilePicker("Feature registry / policy", optional=True)
        grid.addWidget(self.feature_picker, 0, 0)
        grid.addWidget(self.qc_picker, 0, 1)
        grid.addWidget(self.meta_picker, 1, 0)
        grid.addWidget(self.registry_picker, 1, 1)
        intro.layout.addLayout(grid)

        intro.layout.addWidget(self._project_section_title("Analysis settings"))
        settings = QGridLayout()
        settings.setHorizontalSpacing(14)
        settings.setVerticalSpacing(8)
        modality_label = QLabel("Modality")
        modality_label.setStyleSheet(f"color:{NAVY}; font-weight:800; border:none; background:transparent;")
        self.modality_combo = QComboBox()
        self.modality_combo.addItems(["Auto-detect", "Acoustic", "Kinematic", "Mixed acoustic + kinematic", "Generic"])
        self.modality_combo.setMinimumWidth(290)
        output_label = QLabel("Output folder")
        output_label.setStyleSheet(f"color:{NAVY}; font-weight:800; border:none; background:transparent;")
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Required output folder for analysis results")
        out_button = QPushButton("Browse")
        out_button.setProperty("secondary", True)
        out_button.clicked.connect(self.pick_output_folder)
        settings.addWidget(modality_label, 0, 0)
        settings.addWidget(self.modality_combo, 1, 0)
        settings.addWidget(output_label, 0, 1)
        settings.addWidget(self.output_edit, 1, 1)
        settings.addWidget(out_button, 1, 2)
        settings.setColumnStretch(1, 1)
        intro.layout.addLayout(settings)

        actions = QHBoxLayout()
        actions.setSpacing(10)
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
        self.project_status.setMinimumHeight(145)
        self.project_status.setPlaceholderText("Run messages will appear here.")
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



    def _metadata_mapping_roles(self) -> list[str]:
        return [
            "Ignore",

            "-- File / recording identity --",
            "File name",
            "File extension",
            "Subject ID",
            "Protocol ID",
            "Iteration",
            "Recording date",
            "Task code",
            "Task name",
            "Visit/session ID",
            "Timepoint",

            "-- Core clinical grouping --",
            "Diagnosis",
            "Disease group",
            "Group label",

            "-- Priority clinical scores --",
            "ALSFRS bulbar score",
            "ALSBDI total score",
            "ALSFRS total score",
            "Functional score",
            "Functional total score",
            "Bulbar score",
            "Bulbar total score",
            "Disease severity score",
            "Severity class/bin",

            "-- Other clinical outcomes / scores --",
            "Speech intelligibility outcome",
            "Speaking rate outcome",
            "Swallowing score",
            "Sialorrhea score",
            "Cognitive score",
            "Behavioral score",
            "Mood / depression score",
            "Clinical score 1",
            "Clinical score 2",
            "Clinical score 3",
            "Primary target",
            "Secondary target",
            "Target 1",
            "Target 2",
            "Target 3",
            "Outcome 1",
            "Outcome 2",
            "Outcome 3",

            "-- Demographics --",
            "Sex / gender",
            "Age",
            "Date of birth",
            "Race / ethnicity",
            "Education",
            "Language background",
            "Demographic covariate",

            "-- Disease history / clinical covariates --",
            "Onset presentation",
            "Date of first symptom",
            "Date of diagnosis",
            "Site of onset",
            "Disease duration source",
            "Hearing status",
            "Vision status",
            "Clinical covariate",

            "-- Media / acquisition context --",
            "Duration",
            "Frame rate",
            "Sampling rate",
            "Frame width",
            "Frame height",
            "Site / batch",
            "Device",
            "Media technical metadata",

            "-- Manual metadata QC / validity flags --",
            "Task validity flag",
            "Parsing-needed flag",
            "Manual audio QC flag",
            "Manual video QC flag",
            "Manual face/visibility QC flag",
            "Manual acquisition QC flag",
            "Appearance/accessory flag",

            "-- Administrative / governance --",
            "Governance / sharing flag",
            "Data-use permission",
            "Administrative metadata",

            "-- Fallback --",
            "Other covariate",
            "Unlabeled / malformed metadata column",
        ]

    def _metadata_role_to_canonical(self, role: str) -> str | None:
        if role.startswith("--"):
            return None
        return {
            "File name": "file_name",
            "File extension": "file_extension",
            "Subject ID": "subject_id",
            "Protocol ID": "protocol_id",
            "Iteration": "iteration",
            "Recording date": "recording_date",
            "Task code": "task_code",
            "Task name": "task",
            "Visit/session ID": "visit_id",
            "Timepoint": "timepoint",

            "Diagnosis": "diagnosis",
            "Disease group": "diagnosis",
            "Group label": "group_label",

            "ALSFRS bulbar score": "alsfrs_bulbar",
            "ALSBDI total score": "alsbdi_total",
            "ALSFRS total score": "alsfrs_total",
            "Functional score": "functional_score",
            "Functional total score": "functional_score",
            "Bulbar score": "bulbar_score",
            "Bulbar total score": "bulbar_total_score",
            "Disease severity score": "severity_score",
            "Severity class/bin": "severity_bin",

            "Speech intelligibility outcome": "speech_intelligibility",
            "Speaking rate outcome": "speaking_rate",
            "Swallowing score": "swallowing_score",
            "Sialorrhea score": "sialorrhea_score",
            "Cognitive score": "cognitive_score",
            "Behavioral score": "behavioral_score",
            "Mood / depression score": "mood_score",
            "Clinical score 1": "clinical_score_1",
            "Clinical score 2": "clinical_score_2",
            "Clinical score 3": "clinical_score_3",
            "Primary target": "target_primary",
            "Secondary target": "target_secondary",
            "Target 1": "target_1",
            "Target 2": "target_2",
            "Target 3": "target_3",
            "Outcome 1": "outcome_1",
            "Outcome 2": "outcome_2",
            "Outcome 3": "outcome_3",

            "Sex / gender": "sex_or_gender",
            "Age": "age",
            "Date of birth": "date_of_birth",
            "Race / ethnicity": "race_ethnicity",
            "Education": "education",
            "Language background": "language_background",
            "Demographic covariate": None,

            "Onset presentation": "onset_presentation",
            "Date of first symptom": "date_first_symptom",
            "Date of diagnosis": "date_diagnosis",
            "Site of onset": "site_of_onset",
            "Disease duration source": "disease_duration_source",
            "Hearing status": "hearing_status",
            "Vision status": "vision_status",
            "Clinical covariate": None,

            "Duration": "duration",
            "Frame rate": "frame_rate",
            "Sampling rate": "sampling_rate",
            "Frame width": "frame_width",
            "Frame height": "frame_height",
            "Site / batch": "site",
            "Device": "device",
            "Media technical metadata": None,

            "Task validity flag": "task_validity_flag",
            "Parsing-needed flag": "parsing_needed_flag",
            "Manual audio QC flag": "manual_audio_qc_flag",
            "Manual video QC flag": "manual_video_qc_flag",
            "Manual face/visibility QC flag": "manual_face_visibility_qc_flag",
            "Manual acquisition QC flag": "manual_acquisition_qc_flag",
            "Appearance/accessory flag": "appearance_accessory_flag",

            "Governance / sharing flag": "governance_flag",
            "Data-use permission": "data_use_permission",
            "Administrative metadata": None,

            "Other covariate": None,
            "Unlabeled / malformed metadata column": None,
            "Ignore": None,
        }.get(role)

    def _accepted_metadata_role_map(self) -> dict[str, str]:
        """Return user-accepted metadata source-column -> canonical-field mappings.

        This is the authoritative bridge from the Metadata Mapping UI to all
        downstream menus. It lets manually assigned roles propagate even when the
        original metadata headers are dataset-specific, verbose, or malformed.
        """
        df = getattr(self, "metadata_mapping_df", pd.DataFrame())
        if df is None or df.empty:
            return {}
        role_map: dict[str, str] = {}
        for _, row in df.iterrows():
            source = str(row.get("column", "")).strip()
            role = str(row.get("role", "Ignore")).strip()
            canonical = str(row.get("canonical_field", "")).strip()
            if not canonical or canonical.lower() in {"none", "nan", "<na>"}:
                canonical = self._metadata_role_to_canonical(role) or ""
            if not source or not canonical or role == "Ignore" or role.startswith("--"):
                continue
            role_map[source] = canonical
        return role_map

    def _apply_accepted_metadata_roles_to_columns(self, meta_df: pd.DataFrame) -> pd.DataFrame:
        """Create/fill canonical metadata columns from the accepted role mapping.

        The original metadata columns are preserved for auditability. Canonical
        columns such as diagnosis, alsfrs_bulbar, sex_or_gender, recording_date,
        iteration, task_validity_flag, and manual_*_qc_flag are added or filled so
        Overview, Missingness, Distributions, QC, Task Review, Longitudinal, and
        ML export all query the same stable names.
        """
        if meta_df is None or meta_df.empty:
            return meta_df
        role_map = self._accepted_metadata_role_map()
        if not role_map:
            return meta_df
        out = self._ensure_unique_columns(meta_df.copy(), "Metadata table")
        exact_lookup = {str(c): str(c) for c in out.columns}
        norm_lookup: dict[str, str] = {}
        for c in out.columns:
            norm_lookup.setdefault(normalize_name(c), str(c))
        for source, canonical in role_map.items():
            source_col = exact_lookup.get(source) or norm_lookup.get(normalize_name(source))
            if source_col is None or source_col not in out.columns:
                continue
            if canonical == source_col:
                continue
            values = out.loc[:, source_col]
            if isinstance(values, pd.DataFrame):
                values = values.iloc[:, 0]
            if canonical not in out.columns:
                out[canonical] = values
                exact_lookup[canonical] = canonical
                norm_lookup.setdefault(normalize_name(canonical), canonical)
            else:
                existing = out.loc[:, canonical]
                if isinstance(existing, pd.DataFrame):
                    existing = existing.iloc[:, 0]
                empty = self._is_effectively_empty(existing)
                if bool(empty.any()):
                    out[canonical] = existing.astype("object")
                    out.loc[empty, canonical] = values.loc[empty].astype("object")
        return self._ensure_unique_columns(out, "Metadata table")

    def _infer_metadata_role(self, column: str) -> tuple[str, str]:
        n = normalize_name(column)
        raw = str(column).strip()
        if raw.lower().startswith("unnamed"):
            return "Unlabeled / malformed metadata column", "Column has no usable header; inspect examples before mapping."

        alias = {
            "raw_media_file_name": "File name",
            "media_file_name": "File name",
            "filename": "File name",
            "file_name": "File name",
            "file": "File name",
            "extension": "File extension",
            "subjectid": "Subject ID",
            "subject_id": "Subject ID",
            "participant_id": "Subject ID",
            "patient_id": "Subject ID",
            "protocol_id": "Protocol ID",
            "iteration": "Iteration",
            "recording_date": "Recording date",
            "assessment_date": "Recording date",
            "visit_date": "Recording date",
            "task_name": "Task name",
            "task": "Task name",
            "task_code": "Task code",
            "clinical_visit_id": "Visit/session ID",
            "visit_id": "Visit/session ID",
            "session_id": "Visit/session ID",
            "timepoint": "Timepoint",

            "diagnosis": "Diagnosis",
            "dx": "Diagnosis",
            "diagnostic_group": "Disease group",
            "disease_group": "Disease group",
            "group": "Group label",
            "group_label": "Group label",

            "alsfrs_bulbar_subscore": "ALSFRS bulbar score",
            "alsfrs_r_bulbar_subscore": "ALSFRS bulbar score",
            "alsfrs_bulbar": "ALSFRS bulbar score",
            "alsbdi_total_score": "ALSBDI total score",
            "alsbdi": "ALSBDI total score",
            "alsfrs_total_score": "ALSFRS total score",
            "alsfrs_r_total_score": "ALSFRS total score",
            "alsfrs_total": "ALSFRS total score",
            "plsfrs_total_score": "Functional total score",
            "plsfrs_bulbar_subscore": "Bulbar score",
            "sbmafrs_total_score": "Functional total score",
            "sbmafrs_bulbar_score": "Bulbar score",
            "mg_ii_total_score": "Disease severity score",
            "mg_ii_bulbar_subscore": "Bulbar score",
            "eat10_total_score": "Swallowing score",
            "cnsbfs_sialorrhea_subscore": "Sialorrhea score",
            "cnsbfs_speech_subscore": "Bulbar score",
            "cnsbfs_swallowing_subscore": "Swallowing score",
            "moca_total_score": "Cognitive score",
            "alstcbs_total_score": "Behavioral score",
            "alscbs_total_score": "Behavioral score",
            "ecas_total_score": "Cognitive score",
            "beck_depression_inventory_total_score": "Mood / depression score",
            "sentence_intelligibility_percent": "Speech intelligibility outcome",
            "speaking_rate": "Speaking rate outcome",
            "severity_score": "Disease severity score",
            "severity": "Disease severity score",
            "severity_bin": "Severity class/bin",
            "severity_class": "Severity class/bin",

            "date_of_birth": "Date of birth",
            "sex": "Sex / gender",
            "gender": "Sex / gender",
            "race_ethnicity": "Race / ethnicity",
            "level_of_education": "Education",
            "is_english_the_first_language": "Language background",
            "other_languages_spoken": "Language background",

            "presentation_at_onset": "Onset presentation",
            "date_of_first_symptom": "Date of first symptom",
            "date_of_diagnosis": "Date of diagnosis",
            "site_of_disease_onset": "Site of onset",
            "hearing_status": "Hearing status",
            "vision_status": "Vision status",

            "organization_name": "Site / batch",
            "site": "Site / batch",
            "batch": "Site / batch",
            "device": "Device",
            "microphone": "Device",
            "platform": "Device",
            "frame_rate": "Frame rate",
            "sampling_rate": "Sampling rate",
            "frame_width": "Frame width",
            "frame_height": "Frame height",
            "duration_s": "Duration",
            "duration": "Duration",

            "task_completed_as_instructed": "Manual acquisition QC flag",
            "needs_parsing": "Manual acquisition QC flag",
            "another_person_in_frame": "Manual video QC flag",
            "another_person_speaks": "Manual audio QC flag",
            "background_noise": "Manual audio QC flag",
            "volume_is_unstable": "Manual audio QC flag",
            "poor_audio_quality": "Manual audio QC flag",
            "frozen_video": "Manual video QC flag",
            "video_is_unstable": "Manual video QC flag",
            "subject_looks_away": "Manual face/visibility QC flag",
            "poor_light": "Manual acquisition QC flag",
            "blurry_image": "Manual acquisition QC flag",
            "wearing_glasses": "Manual face/visibility QC flag",
            "facial_hair_present": "Manual face/visibility QC flag",

            "shared_externally": "Governance / sharing flag",
            "type_of_data_to_be_shared": "Data-use permission",
            "audio_for_academic_purposes": "Data-use permission",
            "video_for_academic_purposes": "Data-use permission",
        }
        role = alias.get(n)
        if role is None:
            if any(x in n for x in ["diagnosis", "disease", "dx"]):
                role = "Diagnosis"
            elif any(x in n for x in ["bulbar"]):
                role = "Bulbar score"
            elif any(x in n for x in ["alsbdi"]):
                role = "ALSBDI total score"
            elif any(x in n for x in ["alsfrs"]):
                role = "Functional total score"
            elif any(x in n for x in ["intelligibility"]):
                role = "Speech intelligibility outcome"
            elif any(x in n for x in ["speaking_rate", "speech_rate"]):
                role = "Speaking rate outcome"
            elif any(x in n for x in ["target", "label", "outcome"]):
                role = "Primary target"
            elif any(x in n for x in ["severity", "stage"]):
                role = "Disease severity score"
            elif any(x in n for x in ["score", "scale", "rating"]):
                role = "Clinical score 1"
            elif any(x in n for x in ["sex", "gender"]):
                role = "Sex / gender"
            elif any(x in n for x in ["age", "birth", "race", "education", "language"]):
                role = "Demographic covariate"
            elif any(x in n for x in ["noise", "quality", "unstable", "blurry", "frozen", "light", "glasses", "facial_hair"]):
                role = "Manual acquisition QC flag"
            elif any(x in n for x in ["shared", "consent", "academic", "permission"]):
                role = "Administrative metadata"
            elif any(x in n for x in ["visit", "session", "timepoint"]):
                role = "Visit/session ID"
            else:
                role = "Ignore"
        return role, "Matched metadata naming pattern." if role != "Ignore" else "No clinical/demographic role inferred."


    def _metadata_role_group(self, role: str) -> str:
        if role.startswith("--"):
            return "Group heading"
        if role in {"File name", "File extension", "Subject ID", "Protocol ID", "Iteration", "Recording date", "Task code", "Task name", "Visit/session ID", "Timepoint"}:
            return "File / recording identity"
        if role in {"Diagnosis", "Disease group", "Group label"}:
            return "Core clinical grouping"
        if role in {"ALSFRS bulbar score", "ALSBDI total score", "ALSFRS total score", "Functional score", "Functional total score", "Bulbar score", "Bulbar total score", "Disease severity score", "Severity class/bin"}:
            return "Priority clinical scores"
        if role in {"Speech intelligibility outcome", "Speaking rate outcome", "Swallowing score", "Sialorrhea score", "Cognitive score", "Behavioral score", "Mood / depression score", "Clinical score 1", "Clinical score 2", "Clinical score 3", "Primary target", "Secondary target", "Target 1", "Target 2", "Target 3", "Outcome 1", "Outcome 2", "Outcome 3"}:
            return "Other outcomes / scores"
        if role in {"Sex / gender", "Age", "Date of birth", "Race / ethnicity", "Education", "Language background", "Demographic covariate"}:
            return "Demographics"
        if role in {"Onset presentation", "Date of first symptom", "Date of diagnosis", "Site of onset", "Disease duration source", "Hearing status", "Vision status", "Clinical covariate"}:
            return "Disease history / covariates"
        if role in {"Duration", "Frame rate", "Sampling rate", "Frame width", "Frame height", "Site / batch", "Device", "Media technical metadata"}:
            return "Media / acquisition context"
        if role in {"Task validity flag", "Parsing-needed flag", "Manual audio QC flag", "Manual video QC flag", "Manual face/visibility QC flag", "Manual acquisition QC flag", "Appearance/accessory flag"}:
            return "Manual metadata QC / validity"
        if role in {"Governance / sharing flag", "Data-use permission", "Administrative metadata"}:
            return "Administrative / governance"
        if role in {"Other covariate", "Unlabeled / malformed metadata column", "Ignore"}:
            return "Fallback / ignored"
        return "Other"

    def _metadata_example_values(self, series: pd.Series, max_values: int = 4) -> str:
        if series is None:
            return ""
        vals = []
        for v in series.dropna().astype(str).tolist():
            v = v.strip()
            if not v or v.lower() in {"nan", "none", "<na>", "nat"}:
                continue
            if v not in vals:
                vals.append(v)
            if len(vals) >= max_values:
                break
        return "; ".join(vals)

    def _build_metadata_mapping_df(self) -> pd.DataFrame:
        if self.meta_df is None or self.meta_df.empty:
            return pd.DataFrame(columns=["column", "role_group", "role", "canonical_field", "confidence", "reason", "dtype", "missing", "unique", "examples"])
        rows = []
        for c in self.meta_df.columns:
            role, reason = self._infer_metadata_role(str(c))
            canon = self._metadata_role_to_canonical(role)
            s = self.meta_df[c]
            rows.append({
                "column": str(c),
                "role_group": self._metadata_role_group(role),
                "role": role,
                "canonical_field": canon or "",
                "confidence": 0.9 if role != "Ignore" and not role.startswith("--") else 0.4,
                "reason": reason,
                "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "unique": int(s.nunique(dropna=True)),
                "examples": self._metadata_example_values(s),
            })
        return pd.DataFrame(rows)

    def _metadata_mapping_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignTop)

        card = Card("Metadata Mapping", "Assign clinical, demographic, manual-QC, and administrative roles. If no metadata table is provided, derive basic recording context from a filename column.")
        card.layout.setContentsMargins(14, 10, 14, 12)
        card.layout.setSpacing(8)
        card.layout.setAlignment(Qt.AlignTop)

        self.metadata_toolbar_widget = QWidget()
        toolbar = QHBoxLayout(self.metadata_toolbar_widget)
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        toolbar.addWidget(QLabel("Show:"))
        self.metadata_mapping_filter_combo = QComboBox()
        self.metadata_mapping_filter_combo.addItems(["All roles", "Mapped only", "Unmapped / ignored", "Recording identity", "Core clinical", "Priority clinical scores", "Demographics", "Disease history / covariates", "Media / device", "Manual metadata QC", "Administrative", "Unlabeled columns"])
        self.metadata_mapping_filter_combo.currentIndexChanged.connect(lambda _=0: self.refresh_metadata_mapping_table(rebuild=False))
        toolbar.addWidget(self.metadata_mapping_filter_combo)
        toolbar.addStretch(1)
        refresh = QPushButton("Refresh")
        refresh.setProperty("secondary", True)
        refresh.clicked.connect(self.refresh_metadata_mapping_table)
        accept = QPushButton("Accept Metadata Mapping")
        accept.clicked.connect(self.accept_metadata_mapping)
        toolbar.addWidget(refresh)
        toolbar.addWidget(accept)
        card.layout.addWidget(self.metadata_toolbar_widget)

        self.filename_metadata_fallback_frame = QFrame()
        self.filename_metadata_fallback_frame.setObjectName("FilenameFallbackFrame")
        self.filename_metadata_fallback_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.filename_metadata_fallback_frame.setStyleSheet(
            f"QFrame#FilenameFallbackFrame {{ background:#F8FBFE; border:1px solid {LINE}; border-radius:14px; }}"
            f"QFrame#FilenameFallbackFrame QLabel {{ color:{INK}; background:transparent; border:none; padding:0px; }}"
            f"QLabel#ExampleBox {{ color:{INK}; background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; padding:10px; font-family:Consolas, 'Courier New'; font-size:11px; }}"
        )
        fallback_layout = QVBoxLayout(self.filename_metadata_fallback_frame)
        fallback_layout.setContentsMargins(14, 10, 14, 12)
        fallback_layout.setSpacing(8)
        fallback_layout.setAlignment(Qt.AlignTop)
        fallback_title = QLabel("Filename-derived metadata fallback")
        fallback_title.setStyleSheet(f"color:{NAVY}; font-size:14px; font-weight:900; border:none; background:transparent;")
        fallback_layout.addWidget(fallback_title)
        fallback_note = QLabel("No metadata table is loaded. Use this section to derive subject, protocol, iteration, date, task code, and task name from a filename-like column in the feature table.")
        fallback_note.setWordWrap(True)
        fallback_note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent;")
        fallback_layout.addWidget(fallback_note)

        source_row = QGridLayout()
        source_row.setHorizontalSpacing(12)
        source_row.setVerticalSpacing(6)
        context_source_label = QLabel("Context source")
        context_source_label.setStyleSheet(f"color:{MUTED}; font-size:11px; font-weight:800; background:transparent; border:none; padding:0px;")
        filename_column_label = QLabel("Filename column")
        filename_column_label.setStyleSheet(f"color:{MUTED}; font-size:11px; font-weight:800; background:transparent; border:none; padding:0px;")
        source_row.addWidget(context_source_label, 0, 0)
        source_row.addWidget(filename_column_label, 0, 1)
        self.filename_parser_combo = QComboBox()
        self.filename_parser_combo.addItems([
            "Selected filename column + metadata fallback",
            "Auto-detect filename column + metadata fallback",
            "Metadata only (no filename parsing)",
        ])
        self.filename_parser_combo.setCurrentText("Selected filename column + metadata fallback")
        self.filename_parser_combo.setMinimumWidth(340)
        self.filename_source_combo = QComboBox()
        self.filename_source_combo.addItem("Auto-detect")
        self.filename_source_combo.setMinimumWidth(340)
        self.filename_source_combo.currentIndexChanged.connect(lambda _=0: self.refresh_filename_template_ui())
        source_row.addWidget(self.filename_parser_combo, 1, 0)
        source_row.addWidget(self.filename_source_combo, 1, 1)
        source_row.setColumnStretch(1, 1)
        fallback_layout.addLayout(source_row)

        self.filename_example_label = QLabel("Choose/load a feature table, then click Inspect filename example.")
        self.filename_example_label.setObjectName("ExampleBox")
        self.filename_example_label.setWordWrap(True)
        fallback_layout.addWidget(self.filename_example_label)

        self.filename_fallback_status_label = QLabel("No filename context has been applied yet.")
        self.filename_fallback_status_label.setWordWrap(True)
        self.filename_fallback_status_label.setStyleSheet(f"color:{MUTED}; background:transparent; border:none; padding:0px;")
        fallback_layout.addWidget(self.filename_fallback_status_label)

        token_grid = QGridLayout()
        token_grid.setHorizontalSpacing(10)
        token_grid.setVerticalSpacing(8)
        self.filename_token_combos = {}
        token_roles = [
            ("subject_id", "Subject ID"),
            ("protocol_id", "Protocol ID"),
            ("iteration", "Iteration"),
            ("duration", "Duration"),
            ("recording_date", "Recording date"),
            ("task_code", "Task code"),
            ("task", "Task start"),
            ("task_end", "Task end"),
        ]
        for row_i, (role_key, role_label) in enumerate(token_roles):
            label = QLabel(role_label)
            label.setStyleSheet(f"color:{MUTED}; font-size:11px; font-weight:800; border:none; background:transparent;")
            combo = QComboBox()
            combo.addItem("Auto")
            combo.setMinimumWidth(170)
            self.filename_token_combos[role_key] = combo
            token_grid.addWidget(label, row_i // 4 * 2, row_i % 4)
            token_grid.addWidget(combo, row_i // 4 * 2 + 1, row_i % 4)
        fallback_layout.addLayout(token_grid)

        fallback_actions = QHBoxLayout()
        inspect = QPushButton("Inspect filename example")
        inspect.setProperty("secondary", True)
        inspect.clicked.connect(self.refresh_filename_template_ui)
        apply = QPushButton("Apply filename context")
        apply.clicked.connect(self.apply_filename_context_from_metadata_mapping)
        fallback_actions.addWidget(inspect)
        fallback_actions.addWidget(apply)
        fallback_actions.addStretch(1)
        fallback_layout.addLayout(fallback_actions)
        card.layout.addWidget(self.filename_metadata_fallback_frame)

        self.metadata_mapping_table = QTableWidget(0, 10)
        self.metadata_mapping_table.setHorizontalHeaderLabels(["Column", "Role group", "Assigned role", "Canonical field", "Confidence", "Missing", "Unique", "Examples", "Reason", "dtype"])
        self.metadata_mapping_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.metadata_mapping_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.metadata_mapping_table.setAlternatingRowColors(True)
        self.metadata_mapping_table.setWordWrap(False)
        self.metadata_mapping_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.metadata_mapping_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.metadata_mapping_table.verticalHeader().setDefaultSectionSize(32)
        self.metadata_mapping_table.verticalHeader().setMinimumSectionSize(30)
        self.metadata_mapping_table.setMinimumHeight(620)
        header = self.metadata_mapping_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        self.metadata_mapping_table.setColumnWidth(0, 260)
        self.metadata_mapping_table.setColumnWidth(1, 190)
        self.metadata_mapping_table.setColumnWidth(2, 240)
        self.metadata_mapping_table.setColumnWidth(3, 160)
        self.metadata_mapping_table.setColumnWidth(4, 80)
        self.metadata_mapping_table.setColumnWidth(5, 75)
        self.metadata_mapping_table.setColumnWidth(6, 75)
        self.metadata_mapping_table.setColumnWidth(7, 320)
        self.metadata_mapping_table.setColumnWidth(8, 320)
        self.metadata_mapping_table.setColumnWidth(9, 110)
        card.layout.addWidget(self.metadata_mapping_table, 1)

        self.metadata_quick_role_widget = QWidget()
        self.metadata_quick_role_widget.setVisible(False)
        quick = QHBoxLayout(self.metadata_quick_role_widget)
        quick.setContentsMargins(0, 0, 0, 0)
        quick.setSpacing(8)
        for label, role in [
            ("Diagnosis", "Diagnosis"),
            ("ALSFRS bulbar", "ALSFRS bulbar score"),
            ("ALSBDI", "ALSBDI total score"),
            ("ALSFRS total", "ALSFRS total score"),
            ("Sex", "Sex / gender"),
            ("Manual audio QC", "Manual audio QC flag"),
            ("Manual video QC", "Manual video QC flag"),
            ("Ignore", "Ignore"),
        ]:
            b = QPushButton(label)
            b.setProperty("secondary", True)
            b.clicked.connect(lambda _=False, r=role: self.set_selected_metadata_role(r))
            quick.addWidget(b)
        quick.addStretch(1)
        card.layout.addWidget(self.metadata_quick_role_widget)

        self.metadata_mapping_summary_label = QLabel("Load a metadata table to inspect clinical/demographic roles. If no metadata exists, use filename-derived metadata fallback above.")
        self.metadata_mapping_summary_label.setWordWrap(True)
        self.metadata_mapping_summary_label.setStyleSheet(f"color:{MUTED}; background:#F7FAFD; border:1px solid {LINE}; border-radius:8px; padding:8px;")
        card.layout.addWidget(self.metadata_mapping_summary_label)

        layout.addWidget(card, 0, Qt.AlignTop)
        layout.addStretch(1)
        self.update_metadata_mapping_mode_visibility()
        return self._wrap_scroll(body)


    def update_metadata_mapping_mode_visibility(self) -> None:
        """Switch Metadata Mapping between metadata-role mode and filename-fallback mode."""
        has_metadata = self.meta_df is not None and not self.meta_df.empty
        if hasattr(self, "filename_metadata_fallback_frame"):
            self.filename_metadata_fallback_frame.setVisible(not has_metadata)
        if hasattr(self, "metadata_toolbar_widget"):
            self.metadata_toolbar_widget.setVisible(has_metadata)
        if hasattr(self, "metadata_mapping_table"):
            self.metadata_mapping_table.setVisible(has_metadata)
        if hasattr(self, "metadata_mapping_filter_combo"):
            self.metadata_mapping_filter_combo.setEnabled(has_metadata)
        if hasattr(self, "metadata_quick_role_widget"):
            self.metadata_quick_role_widget.setVisible(has_metadata)
        if hasattr(self, "metadata_mapping_summary_label"):
            self.metadata_mapping_summary_label.setVisible(has_metadata)
        if not has_metadata and hasattr(self, "filename_source_combo"):
            self.refresh_filename_source_combo(self.feature_df)

    def apply_filename_context_from_metadata_mapping(self) -> None:
        """Apply filename-derived context directly from the Metadata Mapping fallback UI.

        This deliberately bypasses the metadata merge path. In no-metadata mode,
        the selected feature-table filename column and token template are the
        source of truth, and the parsed canonical fields must be written into
        analysis_df so downstream menus can immediately see task/subject/date.
        """
        if self.feature_df is None or self.feature_df.empty:
            QMessageBox.information(self, "No feature table", "Load a primary feature table first.")
            return

        if hasattr(self, "filename_source_combo") and self.filename_source_combo.count() <= 1:
            self.refresh_filename_source_combo(self.feature_df)

        selected_col = self._selected_filename_source_column(self.feature_df)
        if selected_col is None or selected_col not in self.feature_df.columns:
            msg = "No usable filename/source column is selected. Choose the feature-table column containing filenames or stems."
            if hasattr(self, "filename_fallback_status_label"):
                self.filename_fallback_status_label.setText(msg)
            QMessageBox.warning(self, "Filename context not applied", msg)
            return

        self.log(f"Applying filename context from Metadata Mapping fallback using selected column: {selected_col}")
        non_empty = int(self.feature_df[selected_col].dropna().astype(str).str.strip().ne("").sum())
        first_example = ""
        sample = self.feature_df[selected_col].dropna().astype(str).str.strip()
        if not sample.empty:
            first_example = str(sample.iloc[0])

        base_df = self._standardize_feature_match_helpers(self.feature_df)
        parsed_context = self._parse_filename_context_frame(base_df, source_col=selected_col)
        self.filename_context_parse_df = parsed_context

        out = base_df.copy()
        canonical_pairs = [
            ("subject_id", "parsed_subject_id"),
            ("protocol_id", "parsed_protocol_id"),
            ("iteration", "parsed_iteration"),
            ("duration", "parsed_duration"),
            ("recording_date", "parsed_recording_date"),
            ("task_code", "parsed_task_code"),
            ("task", "parsed_task"),
        ]

        def fill_or_create(canonical: str, parsed_col: str) -> None:
            if parsed_col not in parsed_context.columns:
                return
            values = parsed_context[parsed_col].reindex(out.index)
            valid = values.notna()
            if canonical not in out.columns:
                out[canonical] = pd.Series(pd.NA, index=out.index, dtype="object")
            if canonical != "recording_date":
                out[canonical] = out[canonical].astype("object")
                out.loc[valid, canonical] = values.loc[valid].astype("object")
            else:
                out.loc[valid, canonical] = values.loc[valid]

        for canonical, parsed_col in canonical_pairs:
            fill_or_create(canonical, parsed_col)
        for c in parsed_context.columns:
            if c not in out.columns:
                out[c] = parsed_context[c]

        helper_cols = [c for c in ["_match_file_basename", "_match_file_stem"] if c in out.columns]
        if helper_cols:
            out = out.drop(columns=helper_cols)

        self.analysis_df = out
        self.filename_context_applied = True
        self.metadata_join_strategy = "feature_table_only:metadata_mapping_filename_fallback_direct"
        # Re-classify the enriched analysis table so newly-created context columns
        # are visible to mapping/export code without disturbing feature values.
        try:
            self.mapping_df = classify_columns(self.analysis_df, table_kind="feature", registry=self.registry_df)
            self.proposed_mapping_df = self.mapping_df.copy()
            self.refresh_mapping_table()
        except Exception as exc:
            self.log_error("Filename context mapping refresh failed", exc)

        parsed_ok = 0
        status_counts = {}
        if parsed_context is not None and not parsed_context.empty:
            if "parsed_context_status" in parsed_context.columns:
                status_series = parsed_context["parsed_context_status"].astype(str)
                status_counts = status_series.value_counts(dropna=False).to_dict()
                parsed_ok = int(status_series.isin(["parsed", "parsed_by_user_template"]).sum())
            if parsed_ok == 0:
                context_cols = [c for c in ["parsed_subject_id", "parsed_protocol_id", "parsed_iteration", "parsed_recording_date", "parsed_task_code", "parsed_task"] if c in parsed_context.columns]
                if context_cols:
                    parsed_ok = int(parsed_context[context_cols].notna().any(axis=1).sum())

        mapping = self._filename_template_mapping()
        status = (
            f"Filename-derived context applied. Parsed rows: {parsed_ok} / {len(parsed_context)}. "
            f"Column: {selected_col}. Non-empty values: {non_empty}. "
            f"Example: {first_example or 'none'}. Template: {mapping}. Status counts: {status_counts or 'none'}."
        )
        if hasattr(self, "filename_fallback_status_label"):
            self.filename_fallback_status_label.setText(status)
        self.log(status)
        if hasattr(self, "_refresh_missingness_task_combo"):
            self._refresh_missingness_task_combo(self.analysis_df)
        if hasattr(self, "_refresh_dist_task_combo"):
            self._refresh_dist_task_combo(self.analysis_df)
        if hasattr(self, "_refresh_qc_task_combo"):
            self._refresh_qc_task_combo(self.analysis_df)
        if hasattr(self, "_refresh_focus_combos"):
            self._refresh_focus_combos()
        if parsed_context is not None and not parsed_context.empty:
            try:
                out_dir = self._output_dir() / "feature_analysis" / "tables"
                out_dir.mkdir(parents=True, exist_ok=True)
                parsed_context.to_csv(out_dir / "filename_context_parse_preview.csv", index=False)
                self.log(f"Filename context preview saved: {out_dir / 'filename_context_parse_preview.csv'}")
            except Exception as exc:
                self.log_error("Could not save filename context preview", exc)
        try:
            if hasattr(self, "regenerate_overview_plots"):
                self.regenerate_overview_plots()
        except Exception as exc:
            self.log_error("Overview refresh after filename context apply failed", exc)
        QMessageBox.information(self, "Filename context applied", f"Filename-derived context applied. Parsed rows: {parsed_ok} / {len(parsed_context)}. Downstream menus now use these parsed context fields.")

    def _style_metadata_role_combo(self, combo: QComboBox) -> None:
        """Make Assigned role selectors compact and readable inside table rows."""
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(18)
        combo.setMinimumWidth(190)
        combo.setMaximumWidth(260)
        combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        combo.setFixedHeight(28)
        combo.setView(QListView())
        combo.view().setMinimumWidth(300)
        combo.view().setTextElideMode(Qt.ElideRight)
        combo.setStyleSheet(
            f"QComboBox {{ background:#FFFFFF; color:{INK}; border:1px solid {LINE}; "
            "border-radius:8px; padding:2px 22px 2px 8px; min-height:22px; }}"
            f"QComboBox:hover {{ border:1px solid {TEAL}; }}"
            "QComboBox::drop-down { width:20px; border:none; }"
            f"QComboBox QAbstractItemView {{ background:#FFFFFF; color:{INK}; "
            f"border:1px solid {LINE}; selection-background-color:#E9F7F6; "
            f"selection-color:{INK}; outline:0px; }}"
        )

    def refresh_metadata_mapping_table(self, rebuild: bool = True) -> None:
        if not hasattr(self, "metadata_mapping_table"):
            return
        self.update_metadata_mapping_mode_visibility()
        if self.meta_df is None or self.meta_df.empty:
            self.metadata_mapping_df = pd.DataFrame()
            self.metadata_mapping_table.setRowCount(0)
            if hasattr(self, "metadata_mapping_summary_label"):
                self.metadata_mapping_summary_label.setText("No metadata table loaded. Filename-derived fields applied here are included in the analysis table and become available to Overview, Missingness, Distributions, QC, Task Review, and Longitudinal menus.")
            self.refresh_filename_source_combo(self.feature_df)
            self.refresh_filename_template_ui()
            return
        if rebuild or self.metadata_mapping_df.empty:
            self.metadata_mapping_df = self._build_metadata_mapping_df()

        df = self.metadata_mapping_df.copy()
        filt = self.metadata_mapping_filter_combo.currentText() if hasattr(self, "metadata_mapping_filter_combo") else "All roles"
        if filt == "Recording identity":
            df = df[df["role_group"].astype(str).eq("File / recording identity")]
        elif filt == "Core clinical":
            df = df[df["role_group"].astype(str).eq("Core clinical grouping")]
        elif filt == "Demographics":
            df = df[df["role_group"].astype(str).eq("Demographics")]
        elif filt == "Disease history / covariates":
            df = df[df["role_group"].astype(str).eq("Disease history / covariates")]
        elif filt == "Media / device":
            df = df[df["role_group"].astype(str).eq("Media / acquisition context")]
        elif filt == "Administrative":
            df = df[df["role_group"].astype(str).eq("Administrative / governance")]
        elif filt == "Mapped only":
            df = df[df["role"].astype(str).ne("Ignore")]
        elif filt == "Unmapped / ignored":
            df = df[df["role"].astype(str).eq("Ignore")]
        elif filt == "Recording identity":
            df = df[df["role_group"].astype(str).eq("File / recording identity")]
        elif filt == "Core clinical":
            df = df[df["role_group"].astype(str).eq("Core clinical grouping")]
        elif filt == "Priority clinical scores":
            df = df[df["role_group"].astype(str).eq("Priority clinical scores")]
        elif filt == "Demographics":
            df = df[df["role_group"].astype(str).eq("Demographics")]
        elif filt == "Disease history / covariates":
            df = df[df["role_group"].astype(str).eq("Disease history / covariates")]
        elif filt == "Media / device":
            df = df[df["role_group"].astype(str).eq("Media / acquisition context")]
        elif filt == "Manual metadata QC":
            df = df[df["role_group"].astype(str).eq("Manual metadata QC / validity")]
        elif filt == "Administrative":
            df = df[df["role_group"].astype(str).eq("Administrative / governance")]
        elif filt == "Unlabeled columns":
            df = df[df["role"].astype(str).eq("Unlabeled / malformed metadata column")]

        self.metadata_mapping_table.setRowCount(len(df))
        self.metadata_mapping_table.setColumnCount(10)
        roles = self._metadata_mapping_roles()
        source_indices = df.index.tolist()
        for visual_i, source_i in enumerate(source_indices):
            r = self.metadata_mapping_df.loc[source_i]
            self.metadata_mapping_table.setItem(visual_i, 0, QTableWidgetItem(str(r["column"])))
            self.metadata_mapping_table.setItem(visual_i, 1, QTableWidgetItem(str(r.get("role_group", ""))))
            combo = NoWheelComboBox()
            self._style_metadata_role_combo(combo)
            for role in roles:
                combo.addItem(role)
                if role.startswith("--"):
                    idx = combo.count() - 1
                    combo.model().item(idx).setEnabled(False)
            combo.setCurrentText(str(r.get("role", "Ignore")))
            combo.setProperty("source_index", int(source_i))
            combo.currentTextChanged.connect(lambda _=None: self.collect_metadata_mapping_from_table())
            self.metadata_mapping_table.setCellWidget(visual_i, 2, combo)
            for j, col in enumerate(["canonical_field", "confidence", "missing", "unique", "examples", "reason", "dtype"], start=3):
                item = QTableWidgetItem(str(r.get(col, "")))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.metadata_mapping_table.setItem(visual_i, j, item)
        self.update_metadata_mapping_summary()
        self.update_metadata_mapping_mode_visibility()

    def collect_metadata_mapping_from_table(self) -> pd.DataFrame:
        if self.metadata_mapping_df.empty or not hasattr(self, "metadata_mapping_table"):
            return self.metadata_mapping_df
        df = self.metadata_mapping_df.copy()
        for i in range(self.metadata_mapping_table.rowCount()):
            widget = self.metadata_mapping_table.cellWidget(i, 2)
            if not isinstance(widget, QComboBox):
                continue
            source_i = widget.property("source_index")
            if source_i is None or int(source_i) not in df.index:
                continue
            role = widget.currentText()
            df.loc[int(source_i), "role"] = role
            df.loc[int(source_i), "role_group"] = self._metadata_role_group(role)
            df.loc[int(source_i), "canonical_field"] = self._metadata_role_to_canonical(role) or ""
        self.metadata_mapping_df = df
        self.update_metadata_mapping_summary()
        return df

    def update_metadata_mapping_summary(self) -> None:
        if self.metadata_mapping_df.empty or not hasattr(self, "metadata_mapping_summary_label"):
            return
        roles = self.metadata_mapping_df["role"].astype(str)
        groups = self.metadata_mapping_df["role_group"].astype(str)
        mapped = int(roles.ne("Ignore").sum())
        ignored = int(roles.eq("Ignore").sum())
        priority = int(groups.eq("Priority clinical scores").sum())
        manual_qc = int(groups.eq("Manual metadata QC / validity").sum())
        unlabeled = int(roles.eq("Unlabeled / malformed metadata column").sum())
        self.metadata_mapping_summary_label.setText(
            f"Mapped: {mapped} | Ignored: {ignored} | Priority clinical scores: {priority} | "
            f"Manual metadata QC flags: {manual_qc} | Unlabeled/malformed: {unlabeled}. "
            "Manual metadata QC flags are yes/no acquisition observations; numeric QC-feature analysis remains in QC Integration."
        )

    def set_selected_metadata_role(self, role: str) -> None:
        if not hasattr(self, "metadata_mapping_table"):
            return
        rows = sorted({idx.row() for idx in self.metadata_mapping_table.selectedIndexes()})
        if not rows:
            QMessageBox.information(self, "No rows selected", "Select one or more metadata rows first.")
            return
        for row in rows:
            widget = self.metadata_mapping_table.cellWidget(row, 2)
            if isinstance(widget, QComboBox):
                widget.setCurrentText(role)
        self.collect_metadata_mapping_from_table()

    def _metadata_mapping_path(self) -> Path | None:
        out = self.output_edit.text().strip() if hasattr(self, "output_edit") else ""
        if not out:
            return None
        tables_dir = Path(out) / "feature_analysis" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        return tables_dir / "accepted_metadata_mapping.csv"

    def accept_metadata_mapping(self) -> None:
        if self.meta_df is None or self.meta_df.empty:
            QMessageBox.information(self, "No metadata table", "Load a metadata table first.")
            return
        self.collect_metadata_mapping_from_table()
        self.metadata_mapping_accepted = True
        path = self._metadata_mapping_path()
        if path is not None and not self.metadata_mapping_df.empty:
            self.metadata_mapping_df.to_csv(path, index=False)
            self.log(f"Metadata mapping accepted and saved: {path}")
        if self.feature_df is not None and not self.mapping_df.empty:
            self.analysis_df, self.mapping_df, self.metadata_join_strategy = self._merge_metadata_context(self.feature_df, self.mapping_df)
            self.log(f"Metadata context refreshed after metadata mapping: {self.metadata_join_strategy}")
            if hasattr(self, "_refresh_missingness_task_combo"):
                self._refresh_missingness_task_combo(self.analysis_df)
            if hasattr(self, "_refresh_dist_task_combo"):
                self._refresh_dist_task_combo(self.analysis_df)
            if hasattr(self, "_refresh_qc_task_combo"):
                self._refresh_qc_task_combo(self.analysis_df)
            if hasattr(self, "_refresh_clinical_context_controls"):
                self._refresh_clinical_context_controls(self.analysis_df)
            if hasattr(self, "_refresh_focus_combos"):
                self._refresh_focus_combos()
        self.show_page("overview")

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
            "Design context",
            "Role mapping summary",
            "Feature-family coverage",
            "Feature-quality landscape",
            "Subject x task coverage",
        ])
        plot_header.addWidget(self.overview_plot_combo, 1)

        task_lab = QLabel("Task focus")
        task_lab.setStyleSheet(f"font-weight:800; color:{NAVY}; border:none; background:transparent;")
        plot_header.addWidget(task_lab)
        self.overview_task_combo = QComboBox()
        self.overview_task_combo.setMinimumWidth(220)
        self.overview_task_combo.addItem("All tasks")
        self.overview_task_combo.currentIndexChanged.connect(self.preview_selected_overview_plot)
        plot_header.addWidget(self.overview_task_combo)

        show_btn = QPushButton("Show")
        show_btn.setProperty("secondary", True)
        show_btn.clicked.connect(self.preview_selected_overview_plot)
        plot_header.addWidget(show_btn)

        regen = QPushButton("Regenerate")
        regen.clicked.connect(self.regenerate_relationships)
        plot_header.addWidget(regen)

        plot_header.addStretch(1)

        open_current = QPushButton("Open current plot")
        open_current.setProperty("secondary", True)
        open_current.clicked.connect(self.open_current_overview_plot)
        plot_header.addWidget(open_current)
        plot_panel_layout.addLayout(plot_header)

        self.overview_plot_caption = QLabel("Overview is the dataset-orientation layer: use it to understand subject counts, task structure, diagnosis/sex balance, repeated-measures depth, mapped roles, feature-family coverage, and first-pass quality before opening deeper menus.")
        self.overview_plot_caption.setWordWrap(True)
        self.overview_plot_caption.setStyleSheet(f"color:{INK}; background:#FFFFFF; border:1px solid {LINE}; border-left:5px solid {TEAL}; border-radius:10px; padding:12px; font-size:12px;")
        plot_panel_layout.addWidget(self.overview_plot_caption)

        self.overview_plot_preview = QLabel("Run Feature Analysis, then choose one overview plot.")
        self.overview_plot_preview.setAlignment(Qt.AlignCenter)
        self.overview_plot_preview.setMinimumHeight(560)
        self.overview_plot_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.overview_plot_preview.setStyleSheet(f"QLabel {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:14px; color:{MUTED}; padding:18px; }}")
        plot_panel_layout.addWidget(self.overview_plot_preview, 1)
        overview_split.addWidget(plot_panel, 1)

        # Side summary keeps metric tiles visible without occupying the top of the page.
        side_panel = QFrame()
        side_panel.setStyleSheet(f"QFrame {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:14px; }}")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        side_title = QLabel("Dataset snapshot")
        side_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:15px; border:none; background:transparent;")
        side_layout.addWidget(side_title)
        side_note = QLabel("Compact orientation metrics from the accepted mapping and current analysis table.")
        side_note.setWordWrap(True)
        side_note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent; font-size:12px;")
        side_layout.addWidget(side_note)
        self.overview_metric_grid = QGridLayout()
        self.overview_metric_grid.setHorizontalSpacing(10)
        self.overview_metric_grid.setVerticalSpacing(10)
        side_layout.addLayout(self.overview_metric_grid)
        side_layout.addStretch(1)
        side_panel.setMinimumWidth(280)
        side_panel.setMaximumWidth(360)
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

        self.overview_design_table = QTableWidget(0, 0)
        self.overview_roles_table = QTableWidget(0, 0)
        self.overview_family_table = QTableWidget(0, 0)
        self.overview_quality_table = QTableWidget(0, 0)
        self.overview_subject_task_table = QTableWidget(0, 0)
        for t in [
            self.overview_design_table, self.overview_roles_table,
            self.overview_family_table, self.overview_quality_table, self.overview_subject_task_table,
        ]:
            t.setAlternatingRowColors(True)
            t.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
            t.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

        tabs.addTab(self.overview_design_table, "Design context")
        tabs.addTab(self.overview_roles_table, "Role mapping")
        tabs.addTab(self.overview_family_table, "Feature families")
        tabs.addTab(self.overview_quality_table, "Feature quality")
        tabs.addTab(self.overview_subject_task_table, "Subject x task")
        card.layout.addWidget(tabs)

        layout.addWidget(card)
        return self._wrap_scroll(body)

    def _metric_tile(self, title: str, value: object, subtitle: str = "") -> QFrame:
        tile = QFrame()
        tile.setMinimumHeight(86)
        tile.setStyleSheet(
            f"QFrame {{ background:#F8FBFE; border:1px solid {LINE}; border-radius:14px; }}"
            f"QFrame:hover {{ border:1px solid {TEAL}; background:#FFFFFF; }}"
            f"QLabel#MetricTitle {{ color:{MUTED}; font-size:10px; font-weight:800; letter-spacing:0.5px; text-transform:uppercase; border:none; background:transparent; }}"
            f"QLabel#MetricValue {{ color:{NAVY}; font-size:25px; font-weight:950; border:none; background:transparent; }}"
            f"QLabel#MetricSub {{ color:{MUTED}; font-size:10px; border:none; background:transparent; }}"
        )
        lay = QVBoxLayout(tile)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(3)
        lab = QLabel(str(title)); lab.setObjectName("MetricTitle")
        val = QLabel(str(value)); val.setObjectName("MetricValue")
        sub = QLabel(str(subtitle)); sub.setObjectName("MetricSub"); sub.setWordWrap(True)
        lay.addWidget(lab)
        lay.addWidget(val)
        lay.addWidget(sub)
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
            ("Subjects", metric_value("unique_subjects"), "unique IDs"),
            ("Tasks", metric_value("unique_tasks"), "task levels"),
            ("Complete", score_value("Feature completeness"), "0-100"),
            ("Design", score_value("Design richness"), "context score"),
            ("Review", n_review, f"monitor {n_monitor}"),
            ("QC", "yes" if str(metric_value("qc_table_loaded", False)).lower() == "true" else "no", "artifact table"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.overview_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx // 2, idx % 2)

        self._fill_table(self.overview_roles_table, outputs.get("feature_role_summary", pd.DataFrame()))
        self._fill_table(self.overview_design_table, outputs.get("dataset_design_overview", pd.DataFrame()))
        self._fill_table(self.overview_family_table, outputs.get("feature_family_overview", pd.DataFrame()))
        self._fill_table(self.overview_quality_table, outputs.get("overview_feature_quality_landscape", pd.DataFrame()))
        self._fill_table(self.overview_subject_task_table, outputs.get("overview_subject_task_coverage", pd.DataFrame()))
        self._refresh_overview_task_focus_from_outputs(outputs)
        self.overview_note.setText(
            "Overview generated. Use this page to understand dataset structure: subjects, sex/gender balance, diagnosis balance, task coverage, repeated-measures depth, mapped roles, feature-family coverage, and feature-quality risk."
        )



    def _refresh_overview_task_focus_from_outputs(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "overview_task_combo"):
            return
        current = self.overview_task_combo.currentText()
        self.overview_task_combo.blockSignals(True)
        self.overview_task_combo.clear()
        self.overview_task_combo.addItem("All tasks")
        subject_task = outputs.get("overview_subject_task_coverage", pd.DataFrame())
        if subject_task is not None and not subject_task.empty and "task" in subject_task.columns:
            for task in subject_task["task"].astype(str).tolist()[:300]:
                if task.strip():
                    self.overview_task_combo.addItem(task)
        idx = self.overview_task_combo.findText(current)
        self.overview_task_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.overview_task_combo.blockSignals(False)

    def _refresh_overview_task_focus(self, df: pd.DataFrame) -> None:
        if not hasattr(self, "overview_task_combo"):
            return
        current = self.overview_task_combo.currentText()
        self.overview_task_combo.blockSignals(True)
        self.overview_task_combo.clear()
        self.overview_task_combo.addItem("All tasks")
        task_col = self._first_context_column(df, "task") if df is not None else None
        if task_col and task_col in df.columns:
            vals = df[task_col].astype("string").fillna("").str.strip()
            vals = vals.loc[vals.ne("") & ~vals.str.lower().isin(["nan", "none", "<na>"])]
            for task in vals.value_counts().index.tolist():
                self.overview_task_combo.addItem(str(task))
        idx = self.overview_task_combo.findText(current)
        self.overview_task_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.overview_task_combo.blockSignals(False)

    def _overview_filtered_by_task(self) -> pd.DataFrame:
        df = self._active_analysis_table() if hasattr(self, "_active_analysis_table") else (self.analysis_df if self.analysis_df is not None else self.feature_df)
        if df is None:
            return pd.DataFrame()
        task_value = self.overview_task_combo.currentText() if hasattr(self, "overview_task_combo") else "All tasks"
        if not task_value or task_value == "All tasks":
            return df
        task_col = self._first_context_column(df, "task")
        if not task_col or task_col not in df.columns:
            return df
        mask = df[task_col].astype("string").fillna("").str.strip().eq(str(task_value))
        return df.loc[mask].copy()

    def _overview_subject_task_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["task", "records", "subjects", "repeated_subjects"])
        subject_col = self._first_context_column(df, "subject")
        task_col = self._first_context_column(df, "task")
        iteration_col = self._first_context_column(df, "iteration")
        if not subject_col or not task_col:
            return pd.DataFrame(columns=["task", "records", "subjects", "repeated_subjects"])
        cols = [subject_col, task_col] + ([iteration_col] if iteration_col else [])
        work = df[cols].copy()
        for c in cols:
            work[c] = work[c].astype("string").fillna("").str.strip()
        work = work.loc[work[subject_col].ne("") & work[task_col].ne("")]
        if work.empty:
            return pd.DataFrame(columns=["task", "records", "subjects", "repeated_subjects"])
        records = work[task_col].value_counts()
        subjects = work.groupby(task_col)[subject_col].nunique()
        if iteration_col:
            rep = work.loc[work[iteration_col].ne("")].groupby([subject_col, task_col])[iteration_col].nunique().reset_index(name="n_iterations")
            repeated = rep.loc[rep["n_iterations"].ge(2)].groupby(task_col)[subject_col].nunique()
        else:
            rep = work.groupby([subject_col, task_col]).size().reset_index(name="n_records")
            repeated = rep.loc[rep["n_records"].ge(2)].groupby(task_col)[subject_col].nunique()
        out = pd.DataFrame({"task": records.index.astype(str), "records": records.values})
        out["subjects"] = out["task"].map(subjects).fillna(0).astype(int)
        out["repeated_subjects"] = out["task"].map(repeated).fillna(0).astype(int)
        return out.sort_values(["subjects", "records"], ascending=False).reset_index(drop=True)

    def _safe_task_slug(self, value: str) -> str:
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
        return text[:80] if text else "all_tasks"

    def _context_aliases(self) -> dict[str, list[str]]:
        return {
            "subject": ["metadata__subject_id", "metadata__SubjectID", "subject_id", "SubjectID", "participant_id", "patient_id"],
            "session": ["session_id", "visit_id", "clinical_visit_id", "Clinical Visit ID", "metadata__session_id", "metadata__visit_id", "metadata__Clinical Visit ID"],
            "task": ["metadata__task", "metadata__task_name", "metadata__Task Name", "task", "task_name", "Task Name", "prompt", "parsed_task"],
            "task_code": ["metadata__task_code", "metadata__Task Code", "task_code", "Task Code", "protocol_task_code", "parsed_task_code"],
            "diagnosis": ["metadata__diagnosis", "metadata__Diagnosis", "metadata__diagnostic_group", "diagnosis", "Diagnosis", "dx", "Dx", "diagnostic_group", "disease_group", "group", "Group", "group_label"],
            "severity_bin": ["severity_bin", "severity_class", "alsfrs_bulbar_severity", "ALSFRS bulbar severity", "metadata__severity_bin", "metadata__severity_class"],
            "severity_score": ["severity_score", "functional_score", "bulbar_score", "bulbar_total_score", "alsfrs_total", "alsfrs_bulbar", "alsbdi_total", "clinical_score_1", "clinical_score_2", "clinical_score_3", "target_primary", "target_1", "outcome_1", "ALSFRS total score", "ALSFRS-R total score", "ALSFRS bulbar", "ALSFRS-R bulbar", "ALSBDI", "metadata__severity_score", "metadata__alsfrs_total", "metadata__alsfrs_bulbar", "metadata__alsbdi_total", "metadata__ALSFRS total score", "metadata__ALSFRS bulbar", "metadata__ALSBDI"],
            "sex_or_gender": ["metadata__sex_or_gender", "metadata__Sex", "metadata__Gender", "sex_or_gender", "sex", "Sex", "gender", "Gender"],
            "device": ["device", "microphone", "site", "platform", "recording_device", "metadata__device", "metadata__microphone", "metadata__site"],
            "recording_date": ["recording_date", "Recording date", "assessment_date", "visit_date", "metadata__recording_date", "metadata__Recording date"],
            "iteration": ["metadata__iteration", "metadata__Iteration", "iteration", "Iteration", "parsed_iteration"],
        }

    def _first_context_column(self, df: pd.DataFrame, role: str) -> str | None:
        if df is None or df.empty:
            return None
        aliases = self._context_aliases().get(role, [])
        norm_lookup = {normalize_name(c): c for c in df.columns}
        for a in aliases:
            hit = norm_lookup.get(normalize_name(a))
            if hit in df.columns:
                s = df[hit]
                if not self._is_effectively_empty(s).all():
                    return hit
        return None

    def _overview_context_detection_table(self, df: pd.DataFrame) -> pd.DataFrame:
        roles = [
            ("subject", "Subject"),
            ("session", "Session"),
            ("task", "Task"),
            ("task_code", "Task code"),
            ("diagnosis", "Diagnosis"),
            ("severity_score", "Severity score"),
            ("severity_bin", "Severity bin"),
            ("sex_or_gender", "Sex or gender"),
            ("recording_date", "Recording date"),
            ("iteration", "Iteration"),
            ("device", "Device"),
        ]
        rows = []
        n = len(df) if df is not None else 0
        if df is None:
            df = pd.DataFrame()
        for role, label in roles:
            col = self._first_context_column(df, role)
            if not col:
                rows.append({"variable_type": label, "column": "not_detected", "status": "missing", "n_unique": 0, "n_missing": n, "top_values": ""})
                continue
            s = df[col]
            empty = self._is_effectively_empty(s)
            valid = s.loc[~empty].astype(str)
            vc = valid.value_counts().head(8)
            rows.append({
                "variable_type": label,
                "column": col,
                "status": "detected" if len(valid) else "empty_column",
                "n_unique": int(valid.nunique()) if len(valid) else 0,
                "n_missing": int(empty.sum()),
                "top_values": "; ".join([f"{idx}: {int(val)}" for idx, val in vc.items()]),
            })
        return pd.DataFrame(rows)

    def _align_context_series_to_frame(self, series: pd.Series | None, frame: pd.DataFrame) -> pd.Series | None:
        """Return a context series safely aligned to a local/scoped frame index.

        Several menus build task- or context-scoped DataFrames locally. If a
        combo selection changes while a scoped table is being regenerated, a
        context helper can return a Series from a different frame/index. Assigning
        raw ``series.values`` then raises pandas length-mismatch errors. This
        helper always returns a Series with exactly ``frame.index``.
        """
        if series is None or frame is None:
            return None
        if len(series) == len(frame) and series.index.equals(frame.index):
            return series.astype("string")
        try:
            aligned = series.reindex(frame.index)
            if len(aligned) == len(frame):
                return aligned.astype("string")
        except Exception:
            pass
        if len(series) == len(frame):
            return pd.Series(series.to_numpy(), index=frame.index, dtype="string")
        return pd.Series(pd.NA, index=frame.index, dtype="string")

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
        lookup: dict[str, str] = {}
        for c in df.columns:
            key = normalize_name(c)
            if key and key not in lookup:
                lookup[key] = str(c)
        return lookup

    def _unique_column_name(self, base: str, used: set[str]) -> str:
        """Return a stable unique column name without disturbing the first occurrence."""
        name = str(base)
        if name not in used:
            return name
        i = 2
        while f"{name}__dup{i}" in used:
            i += 1
        return f"{name}__dup{i}"

    def _ensure_unique_columns(self, df: pd.DataFrame, table_label: str = "table") -> pd.DataFrame:
        """Guarantee unique DataFrame columns before pandas assignment/merge operations.

        Acoustic metadata exports can contain repeated REDCap/blank-derived fields, and
        canonical role propagation can map several source fields to the same stable
        context name. Pandas raises "Setting with non-unique columns is not allowed"
        when later assigning into such a frame, so keep the first name and suffix later
        duplicates deterministically.
        """
        if df is None or df.empty:
            return df
        cols = [str(c) for c in df.columns]
        if len(cols) == len(set(cols)):
            return df
        used: set[str] = set()
        new_cols: list[str] = []
        renamed: list[str] = []
        for c in cols:
            if c in used:
                unique = self._unique_column_name(c, used)
                new_cols.append(unique)
                used.add(unique)
                renamed.append(f"{c}->{unique}")
            else:
                new_cols.append(c)
                used.add(c)
        out = df.copy()
        out.columns = new_cols
        try:
            self.log(f"{table_label}: renamed duplicate columns: " + "; ".join(renamed[:12]) + (" ..." if len(renamed) > 12 else ""))
        except Exception:
            pass
        return out

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

    def _file_source_column(self, df: pd.DataFrame) -> str | None:
        if df is None or df.empty:
            return None
        if hasattr(self, "filename_source_combo"):
            selected = self.filename_source_combo.currentText().strip()
            if selected and selected != "Auto-detect" and selected in df.columns:
                return selected
        preferred = [
            "file_name", "source_file_path", "raw_media_file_name", "segmentation_wav_path",
            "video_id", "input_timeseries_csv", "Raw Media File name", "filename", "file", "recording",
            "recording_id", "record_key"
        ]
        for c in preferred:
            if c in df.columns:
                return c
        # Fallback: choose the first text-like column with filename-looking values.
        best_col = None
        best_score = -1
        for c in df.columns:
            s = df[c].dropna().astype(str)
            if s.empty:
                continue
            sample = s.head(50)
            score = 0
            score += int(any(x in str(c).lower() for x in ["file", "filename", "path", "media", "video", "audio", "wav", "webm", "record"]))
            score += int(sample.str.contains(r"\\.(wav|webm|mp4|avi|mov|csv)$", case=False, regex=True).mean() * 5)
            score += int(sample.str.contains(r"[_\\-]", regex=True).mean() * 3)
            score += int(sample.str.contains(r"[A-Za-z]", regex=True).mean() * 2)
            score += int(sample.str.contains(r"\\d", regex=True).mean() * 2)
            if score > best_score:
                best_col = c
                best_score = score
        return best_col if best_score >= 4 else None

    def _add_file_match_helpers(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        source = self._file_source_column(out)
        if source is not None:
            out["_match_file_basename"] = self._basename_series(out[source])
            out["_match_file_stem"] = out["_match_file_basename"].str.replace(r"\.[a-z0-9]+$", "", regex=True)
        return out

    def _filename_parser_mode(self) -> str:
        if hasattr(self, "filename_parser_combo"):
            return self.filename_parser_combo.currentText()
        return "Selected filename column + metadata fallback"

    def _filename_parsing_enabled(self) -> bool:
        mode = self._filename_parser_mode()
        return mode not in {"Metadata only (no filename parsing)", "Off"}



    def _filename_source_candidates(self, df: pd.DataFrame | None = None) -> list[str]:
        """Return all primary-table columns as user-selectable filename sources.

        The GUI must not hide columns behind heuristics. It may rank likely
        filename columns first, but every column remains available because users
        know their own naming convention.
        """
        source_df = df if df is not None else self.feature_df
        if source_df is None or source_df.empty:
            path = self.feature_picker.path if hasattr(self, "feature_picker") else ""
            if path:
                try:
                    source_df = read_table(path)
                except Exception:
                    source_df = None
        if source_df is None or source_df.empty:
            return []

        def score_col(c: object) -> tuple[float, str]:
            s = source_df[c].dropna().astype(str)
            if s.empty:
                return (0.0, str(c))
            sample = s.head(50)
            cname = str(c).lower()
            score = 0.0
            score += 20.0 if any(x in cname for x in ["file", "filename", "path", "media", "raw", "source"]) else 0.0
            score += 14.0 if any(x in cname for x in ["video", "audio", "record", "recording", "wav", "webm"]) else 0.0
            score += float(sample.str.contains(r"\.(?:wav|webm|mp4|avi|mov|csv)$", case=False, regex=True).mean()) * 8.0
            score += float(sample.str.contains(r"[_\-]", regex=True).mean()) * 4.0
            score += float(sample.str.contains(r"[A-Za-z]", regex=True).mean()) * 2.0
            score += float(sample.str.contains(r"\d", regex=True).mean()) * 2.0
            return (score, str(c))

        scored = sorted([score_col(c) for c in source_df.columns], key=lambda x: (-x[0], x[1]))
        return [c for _score, c in scored]

    def refresh_filename_source_combo(self, df: pd.DataFrame | None = None) -> None:
        """Populate filename column dropdown with every primary-table column."""
        if not hasattr(self, "filename_source_combo"):
            return
        current = self.filename_source_combo.currentText()
        candidates = self._filename_source_candidates(df)
        self.filename_source_combo.blockSignals(True)
        self.filename_source_combo.clear()
        self.filename_source_combo.addItem("Auto-detect")
        for c in candidates:
            self.filename_source_combo.addItem(str(c))
        ix = self.filename_source_combo.findText(current)
        if ix >= 0:
            self.filename_source_combo.setCurrentIndex(ix)
        elif candidates:
            # Pick the most likely filename column by default, but keep every
            # column visible for manual correction.
            self.filename_source_combo.setCurrentIndex(1)
        self.filename_source_combo.blockSignals(False)

    def _selected_filename_source_column(self, df: pd.DataFrame | None = None) -> str | None:
        """Return the exact filename/source column selected by the user, if usable.

        The filename fallback page must honor the dropdown selection. Earlier
        versions could silently fall back to the auto-detected column during
        Apply, which made the button appear to do nothing when Auto-detect
        chose the wrong field.
        """
        source_df = df if df is not None else self.feature_df
        if source_df is None or source_df.empty:
            return None
        if hasattr(self, "filename_source_combo") and self.filename_source_combo.count() <= 1:
            self.refresh_filename_source_combo(source_df)
        if hasattr(self, "filename_source_combo"):
            selected = self.filename_source_combo.currentText().strip()
            if selected and selected != "Auto-detect" and selected in source_df.columns:
                return selected
        detected = self._file_source_column(source_df)
        return detected if detected in source_df.columns else None

    def _filename_tokens_from_stem(self, stem_value: object) -> list[str]:
        raw = str(stem_value).strip()
        raw = re.split(r"[\\/]", raw)[-1]
        # Strip one or more trailing file extensions so task tokens never retain file extensions.
        while re.search(r"\.[A-Za-z0-9]{1,8}$", raw):
            raw = re.sub(r"\.[A-Za-z0-9]{1,8}$", "", raw)
        # Preserve token text exactly apart from surrounding punctuation. Task
        # reconstruction joins inclusive token spans, so multi-word task names
        # such as NSM_PUFF or DDK_PA_TA_KA remain complete.
        cleaned = []
        for tok in re.split(r"[_\-\s]+", raw):
            tok = tok.strip().strip("'\".,;:()[]{}")
            if tok:
                cleaned.append(tok)
        return cleaned

    def _filename_template_mapping(self) -> dict[str, int | None]:
        mapping: dict[str, int | None] = {}
        combos = getattr(self, "filename_token_combos", {})
        for role, combo in combos.items():
            val = combo.currentText().strip() if combo is not None else "Auto"
            if not val or val == "Auto":
                mapping[role] = None
                continue
            m = re.match(r"Token\s+(\d+)\b", val)
            if m:
                mapping[role] = int(m.group(1))
                continue
            m_final = re.match(r"Final token:\s+(\d+)\b", val)
            mapping[role] = int(m_final.group(1)) if m_final else None
        return mapping

    def _filename_template_is_active(self) -> bool:
        mapping = self._filename_template_mapping()
        return any(v is not None for v in mapping.values())

    def _example_filename_value(self, df: pd.DataFrame | None = None) -> str | None:
        """Pull one example directly from the user-selected filename column."""
        source_df = df if df is not None else self.feature_df
        if source_df is None or source_df.empty:
            path = self.feature_picker.path if hasattr(self, "feature_picker") else ""
            if path:
                try:
                    source_df = read_table(path)
                    self.refresh_filename_source_combo(source_df)
                except Exception:
                    source_df = None
        if source_df is None or source_df.empty:
            return None

        if hasattr(self, "filename_source_combo") and self.filename_source_combo.count() <= 1:
            self.refresh_filename_source_combo(source_df)

        source_col = self._selected_filename_source_column(source_df)
        if source_col is None or source_col not in source_df.columns:
            return None

        s = source_df[source_col].dropna().astype(str)
        if s.empty:
            return None
        return str(s.iloc[0])

    def refresh_filename_template_ui(self) -> None:
        """Show one real example filename and populate token-position dropdowns."""
        if not hasattr(self, "filename_example_label") or not hasattr(self, "filename_token_combos"):
            return
        example = self._example_filename_value()
        if not example:
            self.filename_example_label.setText("Example filename: no filename-like column detected yet.")
            for combo in self.filename_token_combos.values():
                current = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                combo.addItem("Auto")
                combo.blockSignals(False)
            return
        stem = self._filename_tokens_from_stem(example)
        tokens = stem
        token_text = " | ".join([f"{i}: {tok}" for i, tok in enumerate(tokens)])
        self.filename_example_label.setText(f"Example filename/stem: {example}\\nTokens: {token_text}")
        for role, combo in self.filename_token_combos.items():
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Auto")
            for i, tok in enumerate(tokens):
                combo.addItem(f"Token {i}: {tok}")
            if role == "task_end" and tokens:
                combo.addItem(f"Final token: {len(tokens) - 1}: {tokens[-1]}")
            ix = combo.findText(current)
            if ix >= 0:
                combo.setCurrentIndex(ix)
            combo.blockSignals(False)

    def _filename_task_from_span(self, tokens: list[str], start_idx: int | None, end_idx: int | None) -> str | None:
        """Reconstruct a task label from an inclusive token span without dropping words.

        If a user selects a task start but leaves task end as Auto, the intended
        clinical task is usually the rest of the filename stem. Using the final
        token by default prevents NSM_TNG_LATERAL_NORMAL from collapsing to only
        NSM. Tokens are joined with underscores to create a stable downstream
        task key while preserving the full token text.
        """
        if not tokens or start_idx is None or start_idx < 0 or start_idx >= len(tokens):
            return None
        if end_idx is None or end_idx < 0 or end_idx >= len(tokens):
            end_idx = len(tokens) - 1
        lo, hi = sorted([int(start_idx), int(end_idx)])
        span = [str(t).strip() for t in tokens[lo:hi + 1] if str(t).strip()]
        return "_".join(span) if span else None

    def _parse_filename_context_by_template(self, stem_value: object) -> dict[str, object]:
        tokens = self._filename_tokens_from_stem(stem_value)
        mapping = self._filename_template_mapping()
        result: dict[str, object] = {
            "parsed_subject_id": pd.NA,
            "parsed_protocol_id": pd.NA,
            "parsed_iteration": pd.NA,
            "parsed_duration": pd.NA,
            "parsed_recording_date": pd.NaT,
            "parsed_task_code": pd.NA,
            "parsed_task": pd.NA,
            "parsed_context_status": "unparsed",
            "parsed_template_mode": "user_template",
        }
        def token_at(role: str) -> str | None:
            idx = mapping.get(role)
            if idx is None or idx < 0 or idx >= len(tokens):
                return None
            return tokens[idx]

        subj = token_at("subject_id")
        proto = token_at("protocol_id")
        iteration = token_at("iteration")
        duration = token_at("duration")
        date = token_at("recording_date")
        task_code = token_at("task_code")
        task_start_idx = mapping.get("task")
        task_end_idx = mapping.get("task_end")
        task = self._filename_task_from_span(tokens, task_start_idx, task_end_idx)

        if subj:
            result["parsed_subject_id"] = subj
        if proto:
            result["parsed_protocol_id"] = proto
        if iteration:
            result["parsed_iteration"] = iteration
        if duration:
            result["parsed_duration"] = duration
        if date:
            parsed_date = pd.to_datetime(date, format="%Y%m%d", errors="coerce")
            if pd.isna(parsed_date):
                parsed_date = pd.to_datetime(date, errors="coerce")
            result["parsed_recording_date"] = parsed_date
        if task_code:
            result["parsed_task_code"] = task_code
        if task:
            result["parsed_task"] = str(task).strip()
        if any(pd.notna(result.get(k)) for k in ["parsed_subject_id", "parsed_protocol_id", "parsed_iteration", "parsed_duration", "parsed_recording_date", "parsed_task_code", "parsed_task"]):
            result["parsed_context_status"] = "parsed_by_user_template"
        return result

    def _parse_filename_context_frame(self, df: pd.DataFrame, source_col: str | None = None) -> pd.DataFrame:
        """Parse common VSLP filename context without requiring metadata.

        Expected robust pattern:
        SUBJECT_PROTOCOL_ITERATION_YYYYMMDD_RECORD_TASK...
        The task fallback is the final nonnumeric textual portion after date and
        record/index tokens. This is intentionally heuristic and never raises.
        """
        if df is None or df.empty:
            return pd.DataFrame(index=df.index if df is not None else None)
        source = source_col if source_col in df.columns else self._selected_filename_source_column(df)
        out = pd.DataFrame(index=df.index)
        if source is None or source not in df.columns:
            out["parsed_context_status"] = "no_filename_column"
            return out
        base = self._basename_series(df[source])
        stem = base.apply(lambda x: "_".join(self._filename_tokens_from_stem(x)))
        out["parsed_file_stem"] = stem

        use_template = self._filename_template_is_active()

        def parse_one(s: str) -> dict[str, object]:
            if use_template:
                return self._parse_filename_context_by_template(s)
            tokens = self._filename_tokens_from_stem(s)
            result: dict[str, object] = {
                "parsed_subject_id": pd.NA,
                "parsed_protocol_id": pd.NA,
                "parsed_iteration": pd.NA,
                "parsed_duration": pd.NA,
                "parsed_recording_date": pd.NaT,
                "parsed_task_code": pd.NA,
                "parsed_task": pd.NA,
                "parsed_context_status": "unparsed",
                "parsed_template_mode": "auto",
            }
            if not tokens:
                return result
            result["parsed_subject_id"] = tokens[0]
            if len(tokens) > 1 and re.fullmatch(r"\d+", tokens[1]):
                result["parsed_protocol_id"] = tokens[1]
            if len(tokens) > 2 and re.fullmatch(r"\d+", tokens[2]):
                result["parsed_iteration"] = tokens[2]
            date_idx = None
            for i, tok in enumerate(tokens):
                if re.fullmatch(r"(19|20)\d{6}", tok):
                    date_idx = i
                    try:
                        result["parsed_recording_date"] = pd.to_datetime(tok, format="%Y%m%d", errors="coerce")
                    except Exception:
                        result["parsed_recording_date"] = pd.NaT
                    break
            task_tokens: list[str] = []
            if date_idx is not None:
                tail = tokens[date_idx + 1:]
                if tail and re.fullmatch(r"\d+", tail[0]):
                    result["parsed_task_code"] = tail[0]
                    tail = tail[1:]
                task_tokens = [t for t in tail if re.search(r"[A-Za-z]", t)]
            if not task_tokens:
                # Last textual run anywhere in the filename, used as final fallback.
                text_positions = [i for i, t in enumerate(tokens) if re.search(r"[A-Za-z]", t)]
                if text_positions:
                    last = text_positions[-1]
                    first = last
                    while first - 1 >= 0 and re.search(r"[A-Za-z]", tokens[first - 1]):
                        first -= 1
                    task_tokens = tokens[first:last + 1]
            if task_tokens:
                result["parsed_task"] = "_".join(task_tokens).upper()
                result["parsed_context_status"] = "parsed"
            return result

        parsed = [parse_one(s) for s in stem.tolist()]
        parsed_df = pd.DataFrame(parsed, index=df.index)
        return pd.concat([out, parsed_df], axis=1)

    def _fill_empty_context_from_filename(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fill empty canonical context columns from filename parsing, preserving metadata."""
        mode = self._filename_parser_mode()
        out = df.copy()
        parsed = self._parse_filename_context_frame(out)
        if parsed.empty or not self._filename_parsing_enabled():
            return out, parsed

        def fill_col(canonical: str, parsed_col: str) -> None:
            if parsed_col not in parsed.columns:
                return
            values = parsed[parsed_col].reindex(out.index)
            if canonical not in out.columns:
                out[canonical] = values
                return
            empty = self._is_effectively_empty(out[canonical])
            if mode == "Filename only when metadata is absent":
                # Only fill if the whole field is unavailable/empty.
                if empty.all():
                    out[canonical] = values
                return
            # Use object dtype during partial fills so string filename tokens can
            # safely fill columns that pandas initially inferred as numeric.
            safe_existing = out[canonical].astype("object")
            safe_existing.loc[empty] = values.loc[empty].astype("object")
            out[canonical] = safe_existing

        fill_col("subject_id", "parsed_subject_id")
        fill_col("protocol_id", "parsed_protocol_id")
        fill_col("iteration", "parsed_iteration")
        fill_col("duration", "parsed_duration")
        fill_col("recording_date", "parsed_recording_date")
        fill_col("task_code", "parsed_task_code")
        fill_col("task", "parsed_task")
        for c in parsed.columns:
            if c not in out.columns:
                out[c] = parsed[c]
        return out, parsed


    def _apply_metadata_mapping_to_table(self, meta_df: pd.DataFrame) -> pd.DataFrame:
        """Rename user-mapped metadata columns to canonical context names.

        Multiple columns mapped to the same canonical role are preserved with
        deterministic suffixes; no duplicate DataFrame columns are allowed.
        """
        if meta_df is None or meta_df.empty or self.metadata_mapping_df.empty:
            return meta_df
        out = self._ensure_unique_columns(meta_df.copy(), "Metadata table")
        rename: dict[str, str] = {}
        used = set(str(c) for c in out.columns)
        for _, row in self.metadata_mapping_df.iterrows():
            col = str(row.get("column", ""))
            canon = str(row.get("canonical_field", "")).strip()
            role = str(row.get("role", "Ignore"))
            if not col or col not in out.columns or not canon or role == "Ignore":
                continue
            if canon == col:
                continue
            target = canon if canon not in used else self._unique_column_name(f"metadata__{canon}", used)
            rename[col] = target
            used.add(target)
        if rename:
            out = out.rename(columns=rename)
        return self._ensure_unique_columns(out, "Metadata table")

    def _standardize_metadata_table(self, meta_df: pd.DataFrame) -> pd.DataFrame:
        """Rename/map metadata columns and add match helpers without discarding originals."""
        if meta_df is None or meta_df.empty:
            return meta_df
        out = self._ensure_unique_columns(meta_df.copy(), "Metadata table")
        # First honor explicit user role assignments from Metadata Mapping. This
        # is more reliable than column-name heuristics for clinical scores,
        # diagnosis, demographics, manual QC flags, dates, and iterations.
        out = self._apply_accepted_metadata_roles_to_columns(out)
        rename: dict[str, str] = {}
        used = set(str(c) for c in out.columns)
        for col in out.columns:
            col_s = str(col)
            canon = self._canonical_metadata_name(col_s)
            if canon != col_s:
                if canon not in used:
                    target = canon
                else:
                    target = self._unique_column_name(f"metadata__{canon}", used)
                rename[col_s] = target
                used.add(target)
        if rename:
            out = out.rename(columns=rename)
        out = self._ensure_unique_columns(out, "Metadata table")
        # Apply explicit roles again after heuristic renaming, so a mapped source
        # column still wins if a rename changed the source-column spelling.
        out = self._apply_accepted_metadata_roles_to_columns(out)
        out = self._add_file_match_helpers(out)
        return self._ensure_unique_columns(out, "Metadata table")

    def _standardize_feature_match_helpers(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        return self._add_file_match_helpers(feature_df)

    def _metadata_join_candidates(self) -> list[list[str]]:
        return [
            ["record_key"],
            ["recording_id"],
            ["file_name"],
            ["_match_file_basename"],
            ["_match_file_stem"],
            ["subject_id", "protocol_id", "iteration", "recording_date", "task_code"],
            ["subject_id", "protocol_id", "iteration", "recording_date", "task"],
            ["subject_id", "protocol_id", "iteration", "recording_date"],
            ["subject_id", "protocol_id", "iteration"],
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

    def _metadata_collision_rename_map(
        self,
        metadata_cols: list[str],
        existing_cols: list[str] | set[str],
    ) -> tuple[dict[str, str], list[tuple[str, str]]]:
        """Rename metadata columns that collide with feature columns without creating duplicates.

        Acoustic feature tables already carry context columns such as file_name,
        subject_id, iteration, task, recording_date, diagnosis, severity_score,
        and severity_bin. Metadata standardization can also create canonical
        columns with those same names plus audit columns like metadata__diagnosis.
        A naive collision rename from diagnosis -> metadata__diagnosis can collide
        with an existing metadata__diagnosis column and later make pandas raise
        "Setting with non-unique columns is not allowed". This helper chooses a
        target name unique across the feature frame, the metadata frame, and all
        planned renames.
        """
        existing = {str(c) for c in existing_cols}
        metadata = [str(c) for c in metadata_cols]
        used = set(existing) | set(metadata)
        rename_map: dict[str, str] = {}
        duplicate_canonical_cols: list[tuple[str, str]] = []
        for c in metadata:
            if c not in existing:
                continue
            base = f"metadata__{c}"
            target = base if base not in used else self._unique_column_name(base, used)
            rename_map[c] = target
            used.add(target)
            duplicate_canonical_cols.append((c, target))
        return rename_map, duplicate_canonical_cols

    def _metadata_priority_context_fields(self) -> list[str]:
        """Canonical context fields where mapped metadata is authoritative."""
        return [
            "subject_id", "protocol_id", "iteration", "duration", "recording_date",
            "task_code", "task", "session_id", "visit_id", "timepoint",
            "diagnosis", "disease_group", "group_label", "sex_or_gender", "age",
            "alsfrs_bulbar", "alsbdi_total", "alsfrs_total", "severity_score",
            "severity_bin", "manual_audio_qc_flag", "manual_video_qc_flag",
            "manual_face_visibility_qc_flag", "manual_acquisition_qc_flag",
            "task_validity_flag", "parsing_needed_flag",
        ]

    def _promote_metadata_context_columns(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Promote joined metadata context into canonical columns safely.

        When metadata is loaded, accepted Metadata Mapping is the source of truth
        for clinical, demographic, date/session, task, and manual-QC context.
        Filename-derived context is only a fallback in the no-metadata workflow.
        """
        if merged is None or merged.empty:
            return merged
        out = self._ensure_unique_columns(merged.copy(), "Analysis table before metadata context promotion")
        for canonical in self._metadata_priority_context_fields():
            candidates = [f"metadata__{canonical}"]
            candidates.extend([str(c) for c in out.columns if str(c).startswith(f"metadata__{canonical}_")])
            source_col = next((c for c in candidates if c in out.columns), None)
            if source_col is None:
                continue
            source = out.loc[:, source_col]
            if isinstance(source, pd.DataFrame):
                source = source.iloc[:, 0]
            source = source.reindex(out.index)
            source_nonempty = ~self._is_effectively_empty(source)
            if not bool(source_nonempty.any()):
                continue
            if canonical not in out.columns:
                out[canonical] = pd.Series(pd.NA, index=out.index, dtype="object")
            target = out.loc[:, canonical]
            if isinstance(target, pd.DataFrame):
                target = target.iloc[:, 0]
            target = target.astype("object").reindex(out.index)
            target.loc[source_nonempty] = source.loc[source_nonempty].astype("object")
            out[canonical] = target
        return self._ensure_unique_columns(out, "Analysis table after metadata context promotion")

    def _merge_metadata_context(self, feature_df: pd.DataFrame, feature_mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        """Return feature table enriched with safe metadata labels/covariates when available.

        The Feature GUI must search both the feature table and metadata table for
        subject/session/task/diagnosis/context variables. Metadata is joined only
        when a defensible key is available. When metadata fills an empty canonical
        feature-table column, the canonical column is populated so all downstream
        menus see the same information.
        """
        if self.meta_df is None or self.meta_df.empty:
            feature_base = self._standardize_feature_match_helpers(feature_df)
            feature_base, parsed_context = self._fill_empty_context_from_filename(feature_base)
            self.filename_context_parse_df = parsed_context
            parsed_ok = 0
            if parsed_context is not None and not parsed_context.empty and "parsed_context_status" in parsed_context.columns:
                parsed_ok = int(parsed_context["parsed_context_status"].astype(str).isin(["parsed", "parsed_by_user_template"]).sum())
            return feature_base, feature_mapping.copy(), f"feature_table_only:filename_context_fallback_{parsed_ok}_rows"

        feature_base = self._standardize_feature_match_helpers(feature_df)
        # Metadata is present, so filename parsing is audit-only here. File basename/stem
        # helpers are still available for joining, but canonical context values should
        # come from accepted Metadata Mapping after the metadata merge.
        self.filename_context_parse_df = self._parse_filename_context_frame(feature_base)
        meta_df = self._standardize_metadata_table(self.meta_df.copy())
        feature_lookup = self._normal_col_lookup(feature_base)
        meta_lookup = self._normal_col_lookup(meta_df)

        chosen_keys: list[str] = []
        strategy = "metadata_loaded_not_merged"
        merged = feature_base.copy()
        duplicate_canonical_cols: list[tuple[str, str]] = []

        # Select the best actual metadata join, not simply the first column pair
        # that exists. This matters for kinematic tables where `video_id` has no
        # extension while metadata has .webm/.wav names: basename exists but has
        # zero matches; stem is the correct key.
        best_join: dict[str, object] | None = None
        for candidate in self._metadata_join_candidates():
            if not all(k in feature_lookup and k in meta_lookup for k in candidate):
                continue
            left_keys = [feature_lookup[k] for k in candidate]
            right_keys_orig = [meta_lookup[k] for k in candidate]
            right = meta_df.copy()
            if left_keys != right_keys_orig:
                right = right.rename(columns={rk: lk for lk, rk in zip(left_keys, right_keys_orig)})

            deduped = False
            if right.duplicated(subset=left_keys).any():
                # File-stem metadata can legitimately duplicate audio/video rows
                # for the same recording. Keep one row for context merge; do not
                # allow broad subject-only duplicate joins.
                if any(k in {"_match_file_stem", "_match_file_basename", "file_name", "record_key", "recording_id"} for k in candidate):
                    right = right.drop_duplicates(subset=left_keys, keep="first")
                    deduped = True
                else:
                    continue

            left_frame = merged[left_keys].astype(str).fillna("")
            right_frame = right[left_keys].astype(str).fillna("")
            if len(left_keys) == 1:
                right_values = set(right_frame[left_keys[0]].tolist())
                match_mask = left_frame[left_keys[0]].isin(right_values)
            else:
                right_values = set(map(tuple, right_frame[left_keys].to_numpy()))
                match_mask = left_frame[left_keys].apply(lambda r: tuple(r.values) in right_values, axis=1)
            n_matches = int(match_mask.sum())
            if n_matches <= 0:
                continue
            score = (n_matches / max(1, len(merged)), len(left_keys), -int(deduped))
            if best_join is None or score > best_join["score"]:
                best_join = {
                    "score": score,
                    "candidate": candidate,
                    "left_keys": left_keys,
                    "right": right,
                    "deduped": deduped,
                    "n_matches": n_matches,
                }

        if best_join is not None:
            left_keys = list(best_join["left_keys"])
            right = best_join["right"]
            extra_cols = self._metadata_extra_columns(merged, right, left_keys)
            rename_map, duplicate_canonical_cols = self._metadata_collision_rename_map(
                extra_cols,
                set(str(c) for c in merged.columns),
            )
            right = right[left_keys + extra_cols].rename(columns=rename_map)
            right = self._ensure_unique_columns(right, "Metadata table merge slice")
            merged = merged.merge(right, on=left_keys, how="left", validate="m:1")
            merged = self._ensure_unique_columns(merged, "Analysis table after metadata merge")
            chosen_keys = left_keys
            strategy = (
                "metadata_key_join:" + "+".join(left_keys)
                + f":matched_{int(best_join['n_matches'])}_of_{len(feature_base)}"
                + (":dedup_file_metadata" if best_join.get("deduped") else "")
            )

        if not chosen_keys and len(meta_df) == len(feature_df):
            add = meta_df.reset_index(drop=True).copy()
            helper_renames: dict[str, str] = {}
            helper_used = set(str(c) for c in merged.columns) | set(str(c) for c in add.columns)
            for c in ["_match_file_basename", "_match_file_stem"]:
                if c in add.columns:
                    target = f"metadata__{c}"
                    if target in helper_used:
                        target = self._unique_column_name(target, helper_used)
                    helper_renames[c] = target
                    helper_used.add(target)
            collision_cols = [str(c) for c in add.columns if str(c) not in helper_renames]
            collision_map, duplicate_canonical_cols = self._metadata_collision_rename_map(
                collision_cols,
                set(str(c) for c in merged.columns) | set(helper_renames.values()),
            )
            rename_map = {**helper_renames, **collision_map}
            add = self._ensure_unique_columns(add.rename(columns=rename_map), "Metadata row-order slice")
            add = add[[c for c in add.columns if c not in merged.columns]]
            merged = pd.concat([merged.reset_index(drop=True), add], axis=1)
            merged = self._ensure_unique_columns(merged, "Analysis table after row-order metadata merge")
            strategy = "metadata_row_order_join:same_row_count"

        for canonical, metadata_col in duplicate_canonical_cols:
            if canonical in merged.columns and metadata_col in merged.columns:
                empty_mask = self._is_effectively_empty(merged[canonical])
                # Metadata may contain strings for canonical fields that pandas read as
                # all-missing float columns in the feature table. Cast before assignment
                # so filling subject/session/task/diagnosis never raises a dtype error.
                if bool(empty_mask.any()):
                    merged[canonical] = merged[canonical].astype("object")
                    fill_values = merged.loc[empty_mask, metadata_col]
                    if isinstance(fill_values, pd.DataFrame):
                        fill_values = fill_values.iloc[:, 0]
                    merged.loc[empty_mask, canonical] = fill_values.astype("object")

        # With metadata loaded, promote accepted metadata roles into canonical context
        # columns and do not use filename-derived context to fill/overwrite them.
        # Filename fallback remains active only in the no-metadata branch above.
        merged = self._promote_metadata_context_columns(merged)

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
            "metadata_context": pd.DataFrame([{"strategy": getattr(self, "metadata_join_strategy", "feature_table_only"), "filename_parser_mode": self._filename_parser_mode()}]),
            "filename_context_parse": getattr(self, "filename_context_parse_df", pd.DataFrame()),
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
        """Generate only Overview plots.

        Earlier builds generated every downstream menu plot from the Overview
        refresh path.  That made the dataset-orientation page fail when an
        unrelated deep diagnostic, such as PCA/QC/screening, encountered a
        row/feature dimension mismatch.  Overview should answer only: what is
        in the dataset, what roles are mapped, what feature families are
        covered, what first-pass quality looks like, and how subjects/tasks are
        represented.
        """
        paths: dict[str, str] = {}
        plot_jobs = [
            ("overview_design_context", lambda: plot_overview_design_from_tables(
                outputs.get("overview_design_metrics", pd.DataFrame()),
                outputs.get("overview_design_counts", pd.DataFrame()),
                plots_dir / "overview_design_context.png",
            )),
            ("overview_role_mapping_summary", lambda: plot_role_mapping_summary(outputs.get("feature_role_summary", pd.DataFrame()), plots_dir / "overview_role_mapping_summary.png")),
            ("overview_subject_task_counts", lambda: plot_subject_task_summary_bars(outputs.get("overview_subject_task_coverage", pd.DataFrame()), plots_dir / "overview_subject_task_counts.png")),
            ("overview_feature_family_quality", lambda: plot_feature_family_quality(outputs.get("feature_family_overview", pd.DataFrame()), plots_dir / "overview_feature_family_quality.png")),
            ("overview_feature_quality_landscape", lambda: plot_feature_quality_landscape(outputs.get("overview_feature_quality_landscape", pd.DataFrame()), plots_dir / "overview_feature_quality_landscape.png")),
        ]
        for key, job in plot_jobs:
            try:
                paths[key] = str(job())
            except Exception as exc:
                self.log(f"WARN | Overview plot {key} failed: {exc}")
                self.log(traceback.format_exc())
        return paths

    def preview_selected_overview_plot(self) -> None:
        label = self.overview_plot_combo.currentText() if hasattr(self, "overview_plot_combo") else ""
        key_map = {
            "Design context": "overview_design_context",
            "Role mapping summary": "overview_role_mapping_summary",
            "Feature-family coverage": "overview_feature_family_quality",
            "Feature-quality landscape": "overview_feature_quality_landscape",
            "Subject x task coverage": "overview_subject_task_counts",
        }
        key = key_map.get(label, "overview_design_context")
        if label in {"Feature-family coverage", "Feature-quality landscape"} and hasattr(self, "overview_task_combo") and self.overview_task_combo.currentText() != "All tasks":
            key = self._generate_task_scoped_overview_plot(label)
        self.preview_plot(key)

    def _overview_plot_caption_text(self, key: str) -> str:
        captions = {
            "overview_design_context": "Design context: counts subjects, sex/gender balance, diagnosis balance, tasks, sessions per subject, and repeated-analysis subjects using accepted metadata and filename-derived context.",
            "overview_design_tiles": "Legacy design-variable detection table retained for detailed context audit.",
            "overview_role_mapping_summary": "Role mapping summary: shows how accepted mapping partitions columns into predictors, identifiers, outcomes, covariates, QC variables, and ignored fields.",
            "overview_subject_task_counts": "Subject x task coverage: numeric bar view of records, subjects, and repeated-analysis subjects per task. This replaces the old heatmap.",
            "overview_feature_family_quality": "Feature-family coverage: summarizes predictor coverage by subsystem/family. Use Task focus to inspect coverage for a selected task.",
            "overview_feature_quality_landscape": "Feature-quality landscape: each point is a feature; x-axis is missingness and y-axis is robust outlier burden. Use Task focus to see whether quality issues are task-specific.",
        }
        return captions.get(key, "Overview plot.")


    def _generate_task_scoped_overview_plot(self, label: str) -> str:
        task = self.overview_task_combo.currentText() if hasattr(self, "overview_task_combo") else "All tasks"
        base_key = "overview_feature_family_quality" if label == "Feature-family coverage" else "overview_feature_quality_landscape"
        if not task or task == "All tasks":
            return base_key
        try:
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            df = self._overview_context_source_frame()
            task_series = self._overview_series_for_role(df, "task").astype("string").fillna("").str.strip()
            scoped = df.loc[task_series.eq(str(task))].copy()
            mapping = self.collect_mapping_from_table()
            feature_cols = self._overview_safe_feature_cols(scoped, mapping)
            slug = self._safe_task_slug(task)
            if not hasattr(self, "plot_paths"):
                self.plot_paths = {}
            if label == "Feature-family coverage":
                family = self._overview_feature_family_table(scoped, feature_cols)
                key = f"overview_feature_family_quality__task__{slug}"
                self.plot_paths[key] = str(plot_feature_family_quality(family, plots_dir / f"overview_feature_family_quality__task__{slug}.png"))
            else:
                quality = self._overview_quality_table(scoped, feature_cols)
                key = f"overview_feature_quality_landscape__task__{slug}"
                self.plot_paths[key] = str(plot_feature_quality_landscape(quality, plots_dir / f"overview_feature_quality_landscape__task__{slug}.png"))
            return key
        except Exception as exc:
            self.log(f"Overview task-scoped plot failed for {task}: {exc}")
            self.log(traceback.format_exc())
            return base_key

    def _safe_dashboard_update(self, label: str, updater, outputs: dict[str, pd.DataFrame]) -> bool:
        """Update one dashboard without letting scoped-table bugs abort analysis.

        Overview is the dataset-orientation page and must remain available even
        if a deeper menu has a task/QC/context alignment issue.  Each dashboard
        gets the same outputs object, but failures are isolated and logged with
        traceback for the next menu-specific patch.
        """
        try:
            updater(outputs)
            return True
        except Exception as exc:
            self.log(f"WARN | {label} dashboard update skipped: {exc}")
            self.log(traceback.format_exc())
            return False

    def _safe_refresh_analysis_dashboards(self, outputs: dict[str, pd.DataFrame]) -> None:
        """Refresh dashboards in dependency order, isolating failures per menu."""
        # Overview first: this is the top-level dataset inspection page and
        # should not be blocked by Missingness/Distributions/QC/Task Review.
        dashboard_updates = [
            ("Overview", self.update_overview_dashboard),
            ("Missingness", self.update_missingness_dashboard),
            ("Distributions", self.update_distribution_dashboard),
            ("QC Integration", self.update_qc_dashboard),
            ("Feature Relationships", self.update_relationships_dashboard),
            ("Task Review", self.update_task_review_dashboard),
            ("Longitudinal / Iterations", self.update_longitudinal_dashboard),
            ("Group / Outcome Screening", self.update_screening_dashboard),
            ("Reliability", self.update_reliability_dashboard),
            ("Recommendations", self.update_recommendations_dashboard),
            ("ML Export", self.update_export_dashboard),
        ]
        failed = []
        for label, updater in dashboard_updates:
            if not self._safe_dashboard_update(label, updater, outputs):
                failed.append(label)
        if failed:
            self.log("WARN | Some downstream dashboards need menu-specific review: " + "; ".join(failed))

    def _overview_context_source_frame(self) -> pd.DataFrame:
        """Return a duplicate-safe, metadata-first frame for Overview only.

        Overview is a dataset-orientation menu. It must never depend on deep
        diagnostics or feature-matrix side effects. If metadata is loaded,
        canonical context columns already promoted from accepted Metadata Mapping
        are the source of truth. Filename-derived fields are used only when no
        metadata table exists.
        """
        base = self.analysis_df if getattr(self, "analysis_df", None) is not None and not self.analysis_df.empty else self.feature_df
        if base is None:
            return pd.DataFrame()
        out = self._ensure_unique_columns(base.copy(), "Overview source table").reset_index(drop=True)
        return out

    def _overview_series_for_role(self, df: pd.DataFrame, role: str) -> pd.Series:
        n = len(df) if df is not None else 0
        if df is None or df.empty:
            return pd.Series(pd.NA, index=range(n), dtype="object")
        aliases = self._context_aliases().get(role, [])
        # Metadata-loaded policy: metadata-derived canonical fields are preferred.
        metadata_loaded = getattr(self, "meta_df", None) is not None and not self.meta_df.empty
        preferred: list[str] = []
        canonical_by_role = {
            "subject": "subject_id", "session": "session_id", "task": "task", "task_code": "task_code",
            "diagnosis": "diagnosis", "severity_score": "severity_score", "severity_bin": "severity_bin",
            "sex_or_gender": "sex_or_gender", "recording_date": "recording_date", "iteration": "iteration", "device": "device",
        }
        canonical = canonical_by_role.get(role)
        if metadata_loaded and canonical:
            preferred.extend([f"metadata__{canonical}", canonical])
        elif canonical:
            preferred.extend([canonical, f"metadata__{canonical}"])
        preferred.extend(aliases)
        lookup: dict[str, str] = {}
        for c in df.columns:
            lookup.setdefault(normalize_name(c), str(c))
        for name in preferred:
            col = lookup.get(normalize_name(name))
            if col is None or col not in df.columns:
                continue
            values = df.loc[:, col]
            if isinstance(values, pd.DataFrame):
                values = values.iloc[:, 0]
            values = values.reindex(df.index)
            empty = self._is_effectively_empty(values)
            if not bool((~empty).any()):
                continue
            return values.astype("object")
        return pd.Series(pd.NA, index=df.index, dtype="object")

    def _overview_clean_series(self, series: pd.Series) -> pd.Series:
        if series is None:
            return pd.Series(dtype="object")
        s = series.astype("string").fillna("").str.strip()
        return s.loc[s.ne("") & ~s.str.lower().isin(["nan", "none", "<na>", "nat", "null"])]

    def _overview_safe_feature_cols(self, df: pd.DataFrame, mapping: pd.DataFrame) -> list[str]:
        roles = role_lists(mapping) if mapping is not None and not mapping.empty else {}
        candidates = [str(c) for c in roles.get("Feature", []) if str(c) in df.columns]
        seen: set[str] = set()
        out: list[str] = []
        for c in candidates:
            if c in seen:
                continue
            values = df.loc[:, c]
            if isinstance(values, pd.DataFrame):
                values = values.iloc[:, 0]
            if pd.api.types.is_numeric_dtype(values):
                out.append(c)
                seen.add(c)
        return out

    def _overview_feature_family_table(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        registry = self.registry_df if getattr(self, "registry_df", None) is not None else None
        subsystem_lookup: dict[str, str] = {}
        if registry is not None and not registry.empty:
            fcol = next((c for c in ["feature", "feature_name", "name", "column"] if c in registry.columns), None)
            scol = next((c for c in ["subsystem", "family", "feature_family", "group", "domain"] if c in registry.columns), None)
            if fcol and scol:
                for _, r in registry[[fcol, scol]].dropna().iterrows():
                    subsystem_lookup[str(r[fcol])] = str(r[scol])
        rows = []
        for c in feature_cols:
            values = df.loc[:, c]
            if isinstance(values, pd.DataFrame):
                values = values.iloc[:, 0]
            x = pd.to_numeric(values, errors="coerce")
            rows.append({
                "feature": c,
                "family_or_subsystem": subsystem_lookup.get(c, "unclassified"),
                "n_features": 1,
                "n_numeric_features": 1,
                "missing_fraction": float(x.isna().mean()) if len(x) else 0.0,
                "n_nonmissing": int(x.notna().sum()),
            })
        if not rows:
            return pd.DataFrame(columns=["family_or_subsystem", "n_features", "n_numeric_features", "mean_missing_fraction", "median_nonmissing"])
        detail = pd.DataFrame(rows)
        return detail.groupby("family_or_subsystem", dropna=False).agg(
            n_features=("feature", "count"),
            n_numeric_features=("n_numeric_features", "sum"),
            mean_missing_fraction=("missing_fraction", "mean"),
            median_nonmissing=("n_nonmissing", "median"),
        ).reset_index().sort_values("n_features", ascending=False)

    def _overview_quality_table(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        registry = self.registry_df if getattr(self, "registry_df", None) is not None else None
        subsystem_lookup: dict[str, str] = {}
        if registry is not None and not registry.empty:
            fcol = next((c for c in ["feature", "feature_name", "name", "column"] if c in registry.columns), None)
            scol = next((c for c in ["subsystem", "family", "feature_family", "group", "domain"] if c in registry.columns), None)
            if fcol and scol:
                for _, r in registry[[fcol, scol]].dropna().iterrows():
                    subsystem_lookup[str(r[fcol])] = str(r[scol])
        rows = []
        for c in feature_cols:
            values = df.loc[:, c]
            if isinstance(values, pd.DataFrame):
                values = values.iloc[:, 0]
            x = pd.to_numeric(values, errors="coerce")
            non = x.dropna()
            miss = float(x.isna().mean()) if len(x) else 1.0
            out_frac = 0.0
            zero = True
            if len(non) >= 3:
                zero = bool(non.nunique(dropna=True) <= 1)
                med = float(non.median())
                mad = float((non - med).abs().median())
                if mad > 0:
                    rz = 0.6745 * (non - med) / mad
                    out_frac = float((rz.abs() > 3.5).mean())
            n_valid = int(non.size)
            score = max(0.0, min(100.0, 100.0 - 70.0 * miss - 125.0 * out_frac - (40.0 if zero else 0.0) - (30.0 if n_valid < 3 else 0.0)))
            status = "review" if zero or n_valid < 3 or miss >= 0.50 or out_frac >= 0.20 else ("monitor" if miss >= 0.20 or out_frac >= 0.05 else "ok")
            rows.append({
                "feature": c,
                "family_or_subsystem": subsystem_lookup.get(c, "unclassified"),
                "missing_fraction": miss,
                "robust_outlier_fraction": out_frac,
                "n_valid": n_valid,
                "zero_variance": zero,
                "quality_status": status,
                "quality_score": round(score, 1),
            })
        return pd.DataFrame(rows).sort_values(["quality_status", "quality_score", "feature"], ascending=[False, True, True]) if rows else pd.DataFrame(columns=["feature", "family_or_subsystem", "missing_fraction", "robust_outlier_fraction", "n_valid", "quality_status", "quality_score"])

    def _overview_design_tables(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        n_rows = len(df) if df is not None else 0
        subject = self._overview_clean_series(self._overview_series_for_role(df, "subject"))
        task = self._overview_clean_series(self._overview_series_for_role(df, "task"))
        sex = self._overview_clean_series(self._overview_series_for_role(df, "sex_or_gender"))
        diagnosis = self._overview_clean_series(self._overview_series_for_role(df, "diagnosis"))
        session = self._overview_clean_series(self._overview_series_for_role(df, "session"))
        if session.empty:
            session = self._overview_clean_series(self._overview_series_for_role(df, "recording_date"))
        iteration = self._overview_clean_series(self._overview_series_for_role(df, "iteration"))
        subj_raw = self._overview_series_for_role(df, "subject").astype("string").fillna("").str.strip()
        task_raw = self._overview_series_for_role(df, "task").astype("string").fillna("").str.strip()
        iter_raw = self._overview_series_for_role(df, "iteration").astype("string").fillna("").str.strip()
        repeated_subjects = 0
        tmp = pd.DataFrame({"subject": subj_raw, "task": task_raw, "iteration": iter_raw}, index=df.index if df is not None else None)
        tmp = tmp.loc[tmp["subject"].ne("") & tmp["task"].ne("")]
        if not tmp.empty:
            if tmp["iteration"].ne("").any():
                rep = tmp.loc[tmp["iteration"].ne("")].groupby(["subject", "task"])["iteration"].nunique().reset_index(name="n_iterations")
                repeated_subjects = int(rep.loc[rep["n_iterations"].ge(2), "subject"].nunique())
            else:
                rep = tmp.groupby(["subject", "task"]).size().reset_index(name="n_records")
                repeated_subjects = int(rep.loc[rep["n_records"].ge(2), "subject"].nunique())
        metrics = pd.DataFrame([
            {"metric": "rows/files", "value": n_rows},
            {"metric": "subjects", "value": int(subject.nunique()) if len(subject) else 0},
            {"metric": "sex/gender levels", "value": int(sex.nunique()) if len(sex) else 0},
            {"metric": "diagnosis levels", "value": int(diagnosis.nunique()) if len(diagnosis) else 0},
            {"metric": "tasks", "value": int(task.nunique()) if len(task) else 0},
            {"metric": "subjects eligible for repeated analysis", "value": repeated_subjects},
        ])
        rows = []
        for section, ser in [("sex_or_gender", sex), ("diagnosis", diagnosis), ("task", task)]:
            vc = ser.value_counts().head(40) if len(ser) else pd.Series(dtype=int)
            for level, count in vc.items():
                rows.append({"section": section, "level": str(level), "count": int(count)})
        if len(subject):
            if len(session):
                stmp = pd.DataFrame({"subject": subj_raw, "session": self._overview_series_for_role(df, "session").astype("string").fillna("").str.strip()})
                stmp = stmp.loc[stmp["subject"].ne("") & stmp["session"].ne("")]
                sessions_per_subject = stmp.groupby("subject")["session"].nunique() if not stmp.empty else subj_raw.value_counts()
            else:
                sessions_per_subject = subj_raw.loc[subj_raw.ne("")].value_counts()
            for level, count in sessions_per_subject.value_counts().sort_index().items():
                rows.append({"section": "sessions_per_subject", "level": str(level), "count": int(count)})
        counts = pd.DataFrame(rows, columns=["section", "level", "count"])
        return metrics, counts

    def _overview_subject_task_table(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["task", "records", "subjects", "repeated_subjects"])
        subject = self._overview_series_for_role(df, "subject").astype("string").fillna("").str.strip()
        task = self._overview_series_for_role(df, "task").astype("string").fillna("").str.strip()
        iteration = self._overview_series_for_role(df, "iteration").astype("string").fillna("").str.strip()
        work = pd.DataFrame({"subject": subject, "task": task, "iteration": iteration}, index=df.index)
        work = work.loc[work["subject"].ne("") & work["task"].ne("")]
        if work.empty:
            return pd.DataFrame(columns=["task", "records", "subjects", "repeated_subjects"])
        records = work["task"].value_counts()
        subjects = work.groupby("task")["subject"].nunique()
        if work["iteration"].ne("").any():
            rep = work.loc[work["iteration"].ne("")].groupby(["subject", "task"])["iteration"].nunique().reset_index(name="n_iterations")
            repeated = rep.loc[rep["n_iterations"].ge(2)].groupby("task")["subject"].nunique()
        else:
            rep = work.groupby(["subject", "task"]).size().reset_index(name="n_records")
            repeated = rep.loc[rep["n_records"].ge(2)].groupby("task")["subject"].nunique()
        out = pd.DataFrame({"task": records.index.astype(str), "records": records.astype(int).values})
        out["subjects"] = out["task"].map(subjects).fillna(0).astype(int)
        out["repeated_subjects"] = out["task"].map(repeated).fillna(0).astype(int)
        return out.sort_values(["subjects", "records"], ascending=False).reset_index(drop=True)

    def _build_overview_outputs(self) -> tuple[dict[str, pd.DataFrame], list[str]]:
        """Build Overview tables only, without deep feature diagnostics.

        This path is intentionally row-safe and metadata-first. It does not call
        PCA, QC integration, screening, reliability, or generic distribution
        audit functions, because those may mix row-indexed and feature-indexed
        vectors in menu-specific code.
        """
        if self.feature_df is None:
            self.load_and_map()
        if self.feature_df is None:
            raise RuntimeError("Load a primary feature table first.")
        mapping = self.collect_mapping_from_table()
        active_df = self._overview_context_source_frame()
        feature_cols = self._overview_safe_feature_cols(active_df, mapping)
        design_metrics, design_counts = self._overview_design_tables(active_df)
        subject_task = self._overview_subject_task_table(active_df)
        family = self._overview_feature_family_table(active_df, feature_cols)
        quality = self._overview_quality_table(active_df, feature_cols)
        role_sum = role_summary(mapping)
        inventory = pd.DataFrame([
            {"metric": "feature_table_rows", "value": len(active_df)},
            {"metric": "detected_feature_columns", "value": len(feature_cols)},
            {"metric": "unique_subjects", "value": int(design_metrics.loc[design_metrics["metric"].eq("subjects"), "value"].iloc[0]) if not design_metrics.empty else 0},
            {"metric": "unique_tasks", "value": int(design_metrics.loc[design_metrics["metric"].eq("tasks"), "value"].iloc[0]) if not design_metrics.empty else 0},
            {"metric": "qc_table_loaded", "value": bool(getattr(self, "qc_df", None) is not None and not self.qc_df.empty)},
            {"metric": "metadata_table_loaded", "value": bool(getattr(self, "meta_df", None) is not None and not self.meta_df.empty)},
        ])
        outputs = {
            "dataset_inventory": inventory,
            "feature_role_summary": role_sum,
            "overview_design_metrics": design_metrics,
            "overview_design_counts": design_counts,
            "dataset_design_overview": pd.concat([
                design_metrics.assign(section="metrics", level=design_metrics["metric"], count=design_metrics["value"])[["section", "level", "count"]],
                design_counts
            ], ignore_index=True),
            "metadata_context": pd.DataFrame([{"strategy": getattr(self, "metadata_join_strategy", "feature_table_only"), "filename_parser_mode": self._filename_parser_mode()}]),
            "filename_context_parse": getattr(self, "filename_context_parse_df", pd.DataFrame()),
            "feature_family_overview": family,
            "feature_distribution_summary": pd.DataFrame(),
            "overview_feature_quality_landscape": quality,
            "overview_subject_task_coverage": subject_task,
            "overview_readiness_summary": pd.DataFrame(),
            "group_counts": pd.DataFrame(),
            "feature_column_mapping": mapping,
            "missingness_by_feature": pd.DataFrame(),
            "missingness_by_row": pd.DataFrame(),
            "missingness_by_group": pd.DataFrame(),
            "missingness_by_family": pd.DataFrame(),
        }
        return outputs, feature_cols

    def _overview_emergency_outputs(self) -> tuple[dict[str, pd.DataFrame], list[str]]:
        """Last-resort Overview builder that cannot assign feature-length values to row tables.

        This is intentionally simple and metadata-first.  It uses only scalar
        counts and already existing columns in the current analysis table.  When
        a metadata table is present, metadata/promoted metadata columns are
        preferred for clinical/context fields; filename-derived columns are not
        used unless metadata is absent.
        """
        src = None
        if getattr(self, "analysis_df", None) is not None and not self.analysis_df.empty:
            src = self.analysis_df.copy()
        elif getattr(self, "feature_df", None) is not None and not self.feature_df.empty:
            src = self.feature_df.copy()
        else:
            src = pd.DataFrame()
        if src is None:
            src = pd.DataFrame()
        src = self._ensure_unique_columns(src.copy(), "Overview emergency source").reset_index(drop=True)
        metadata_loaded = bool(getattr(self, "meta_df", None) is not None and not self.meta_df.empty)

        def pick(role: str) -> pd.Series:
            n = len(src)
            if n == 0:
                return pd.Series(dtype="object")
            aliases = self._context_aliases().get(role, []) if hasattr(self, "_context_aliases") else []
            canonical = {
                "subject": "subject_id", "session": "session_id", "task": "task", "task_code": "task_code",
                "diagnosis": "diagnosis", "severity_score": "severity_score", "severity_bin": "severity_bin",
                "sex_or_gender": "sex_or_gender", "recording_date": "recording_date", "iteration": "iteration", "device": "device",
            }.get(role, role)
            names: list[str] = []
            if metadata_loaded:
                names.extend([f"metadata__{canonical}", canonical])
                names.extend([a for a in aliases if str(a).startswith("metadata__")])
                names.extend([a for a in aliases if not str(a).startswith("metadata__")])
            else:
                names.extend([canonical, f"metadata__{canonical}"] + aliases)
            lookup: dict[str, str] = {}
            for c in src.columns:
                lookup.setdefault(normalize_name(c), str(c))
            for name in names:
                col = lookup.get(normalize_name(name))
                if not col or col not in src.columns:
                    continue
                ser = src.loc[:, col]
                if isinstance(ser, pd.DataFrame):
                    ser = ser.iloc[:, 0]
                ser = pd.Series(ser.to_numpy(dtype=object), index=src.index, dtype="object")
                nonempty = ser.astype("string").fillna("").str.strip()
                nonempty = nonempty.loc[nonempty.ne("") & ~nonempty.str.lower().isin(["nan", "none", "<na>", "nat", "null"])]
                if len(nonempty):
                    return ser
            return pd.Series(pd.NA, index=src.index, dtype="object")

        def clean(ser: pd.Series) -> pd.Series:
            if ser is None or len(ser) == 0:
                return pd.Series(dtype="object")
            x = ser.astype("string").fillna("").str.strip()
            return x.loc[x.ne("") & ~x.str.lower().isin(["nan", "none", "<na>", "nat", "null"])]

        subject = clean(pick("subject"))
        task = clean(pick("task"))
        sex = clean(pick("sex_or_gender"))
        diagnosis = clean(pick("diagnosis"))
        session = clean(pick("session"))
        if session.empty:
            session = clean(pick("recording_date"))
        subj_raw = pick("subject").astype("string").fillna("").str.strip()
        task_raw = pick("task").astype("string").fillna("").str.strip()
        iter_raw = pick("iteration").astype("string").fillna("").str.strip()
        repeated_subjects = 0
        if len(src):
            tmp = pd.DataFrame({"subject": subj_raw.to_numpy(), "task": task_raw.to_numpy(), "iteration": iter_raw.to_numpy()})
            tmp = tmp.loc[tmp["subject"].ne("") & tmp["task"].ne("")]
            if not tmp.empty:
                if tmp["iteration"].ne("").any():
                    rep = tmp.loc[tmp["iteration"].ne("")].groupby(["subject", "task"])["iteration"].nunique().reset_index(name="n_iterations")
                    repeated_subjects = int(rep.loc[rep["n_iterations"].ge(2), "subject"].nunique())
                else:
                    rep = tmp.groupby(["subject", "task"]).size().reset_index(name="n_records")
                    repeated_subjects = int(rep.loc[rep["n_records"].ge(2), "subject"].nunique())

        design_metrics = pd.DataFrame([
            {"metric": "rows/files", "value": int(len(src))},
            {"metric": "subjects", "value": int(subject.nunique()) if len(subject) else 0},
            {"metric": "sex/gender levels", "value": int(sex.nunique()) if len(sex) else 0},
            {"metric": "diagnosis levels", "value": int(diagnosis.nunique()) if len(diagnosis) else 0},
            {"metric": "tasks", "value": int(task.nunique()) if len(task) else 0},
            {"metric": "subjects eligible for repeated analysis", "value": int(repeated_subjects)},
        ])
        count_rows = []
        for section, ser in [("sex_or_gender", sex), ("diagnosis", diagnosis), ("task", task)]:
            for level, count in ser.value_counts().head(40).items() if len(ser) else []:
                count_rows.append({"section": section, "level": str(level), "count": int(count)})
        if len(subject):
            subj_full = subj_raw.loc[subj_raw.ne("")]
            sess_full = pick("session").astype("string").fillna("").str.strip()
            if sess_full.ne("").any():
                stmp = pd.DataFrame({"subject": subj_raw.to_numpy(), "session": sess_full.to_numpy()})
                stmp = stmp.loc[stmp["subject"].ne("") & stmp["session"].ne("")]
                sessions_per_subject = stmp.groupby("subject")["session"].nunique() if not stmp.empty else subj_full.value_counts()
            else:
                sessions_per_subject = subj_full.value_counts()
            for level, count in sessions_per_subject.value_counts().sort_index().items():
                count_rows.append({"section": "sessions_per_subject", "level": str(level), "count": int(count)})
        design_counts = pd.DataFrame(count_rows, columns=["section", "level", "count"])

        if len(src):
            work = pd.DataFrame({"subject": subj_raw.to_numpy(), "task": task_raw.to_numpy(), "iteration": iter_raw.to_numpy()})
            work = work.loc[work["subject"].ne("") & work["task"].ne("")]
        else:
            work = pd.DataFrame()
        if work.empty:
            subject_task = pd.DataFrame(columns=["task", "records", "subjects", "repeated_subjects"])
        else:
            records = work["task"].value_counts()
            subjects = work.groupby("task")["subject"].nunique()
            if work["iteration"].ne("").any():
                rep = work.loc[work["iteration"].ne("")].groupby(["subject", "task"])["iteration"].nunique().reset_index(name="n_iterations")
                repeated = rep.loc[rep["n_iterations"].ge(2)].groupby("task")["subject"].nunique()
            else:
                rep = work.groupby(["subject", "task"]).size().reset_index(name="n_records")
                repeated = rep.loc[rep["n_records"].ge(2)].groupby("task")["subject"].nunique()
            subject_task = pd.DataFrame({"task": records.index.astype(str), "records": records.astype(int).to_numpy()})
            subject_task["subjects"] = subject_task["task"].map(subjects).fillna(0).astype(int)
            subject_task["repeated_subjects"] = subject_task["task"].map(repeated).fillna(0).astype(int)
            subject_task = subject_task.sort_values(["subjects", "records"], ascending=False).reset_index(drop=True)

        mapping = self.collect_mapping_from_table() if hasattr(self, "collect_mapping_from_table") else pd.DataFrame()
        try:
            role_sum = role_summary(mapping)
        except Exception:
            role_sum = summarize_roles(mapping) if mapping is not None and not mapping.empty else pd.DataFrame()
        try:
            feature_cols = self._overview_safe_feature_cols(src, mapping) if hasattr(self, "_overview_safe_feature_cols") else []
        except Exception:
            feature_cols = []
        if not feature_cols:
            context_norms = {normalize_name(x) for x in ["file_name", "source_file_path", "subject_id", "session_id", "iteration", "task", "recording_date", "diagnosis", "severity_score", "severity_bin", "record_key"]}
            for c in src.columns:
                if normalize_name(c) in context_norms or str(c).startswith("metadata__"):
                    continue
                ser = src.loc[:, c]
                if isinstance(ser, pd.DataFrame):
                    ser = ser.iloc[:, 0]
                if pd.to_numeric(ser, errors="coerce").notna().any():
                    feature_cols.append(str(c))

        family_rows = []
        quality_rows = []
        registry = getattr(self, "registry_df", None)
        fam_lookup: dict[str, str] = {}
        if registry is not None and not registry.empty:
            fcol = next((c for c in ["feature", "feature_name", "name", "column"] if c in registry.columns), None)
            scol = next((c for c in ["subsystem", "family", "feature_family", "group", "domain"] if c in registry.columns), None)
            if fcol and scol:
                fam_lookup = dict(zip(registry[fcol].astype(str), registry[scol].astype(str)))
        for c in feature_cols:
            ser = src.loc[:, c]
            if isinstance(ser, pd.DataFrame):
                ser = ser.iloc[:, 0]
            x = pd.to_numeric(ser, errors="coerce")
            non = x.dropna()
            miss = float(x.isna().mean()) if len(x) else 1.0
            out_frac = 0.0
            zero = bool(non.nunique(dropna=True) <= 1) if len(non) else True
            if len(non) >= 3:
                med = float(non.median()); mad = float((non - med).abs().median())
                if mad > 0:
                    rz = 0.6745 * (non - med) / mad
                    out_frac = float((rz.abs() > 3.5).mean())
            fam = fam_lookup.get(str(c), "unclassified")
            family_rows.append({"feature": str(c), "family_or_subsystem": fam, "n_features": 1, "n_numeric_features": 1, "missing_fraction": miss, "n_nonmissing": int(non.size)})
            score = max(0.0, min(100.0, 100.0 - 70.0 * miss - 125.0 * out_frac - (40.0 if zero else 0.0) - (30.0 if len(non) < 3 else 0.0)))
            status = "review" if zero or len(non) < 3 or miss >= 0.50 or out_frac >= 0.20 else ("monitor" if miss >= 0.20 or out_frac >= 0.05 else "ok")
            quality_rows.append({"feature": str(c), "family_or_subsystem": fam, "missing_fraction": miss, "robust_outlier_fraction": out_frac, "n_valid": int(non.size), "zero_variance": zero, "quality_status": status, "quality_score": round(score, 1)})
        family_detail = pd.DataFrame(family_rows)
        if family_detail.empty:
            family = pd.DataFrame(columns=["family_or_subsystem", "n_features", "n_numeric_features", "mean_missing_fraction", "median_nonmissing"])
        else:
            family = family_detail.groupby("family_or_subsystem", dropna=False).agg(n_features=("feature", "count"), n_numeric_features=("n_numeric_features", "sum"), mean_missing_fraction=("missing_fraction", "mean"), median_nonmissing=("n_nonmissing", "median")).reset_index().sort_values("n_features", ascending=False)
        quality = pd.DataFrame(quality_rows, columns=["feature", "family_or_subsystem", "missing_fraction", "robust_outlier_fraction", "n_valid", "zero_variance", "quality_status", "quality_score"])
        inventory = pd.DataFrame([
            {"metric": "feature_table_rows", "value": int(len(src))},
            {"metric": "detected_feature_columns", "value": int(len(feature_cols))},
            {"metric": "unique_subjects", "value": int(subject.nunique()) if len(subject) else 0},
            {"metric": "unique_tasks", "value": int(task.nunique()) if len(task) else 0},
            {"metric": "qc_table_loaded", "value": bool(getattr(self, "qc_df", None) is not None and not self.qc_df.empty)},
            {"metric": "metadata_table_loaded", "value": metadata_loaded},
        ])
        dataset_design = pd.concat([
            design_metrics.assign(section="metrics", level=design_metrics["metric"], count=design_metrics["value"])[["section", "level", "count"]],
            design_counts,
        ], ignore_index=True)
        outputs = {
            "dataset_inventory": inventory,
            "feature_role_summary": role_sum,
            "overview_design_metrics": design_metrics,
            "overview_design_counts": design_counts,
            "dataset_design_overview": dataset_design,
            "metadata_context": pd.DataFrame([{"strategy": getattr(self, "metadata_join_strategy", "feature_table_only"), "metadata_first": metadata_loaded, "filename_parser_mode": self._filename_parser_mode()}]),
            "filename_context_parse": getattr(self, "filename_context_parse_df", pd.DataFrame()),
            "feature_family_overview": family,
            "overview_feature_quality_landscape": quality,
            "overview_subject_task_coverage": subject_task,
            "overview_readiness_summary": pd.DataFrame(),
            "group_counts": pd.DataFrame(),
            "feature_column_mapping": mapping,
        }
        return outputs, feature_cols

    def _write_overview_placeholder_plot(self, path: Path, title: str, message: str) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=16, fontweight="bold", color="#071A33")
        ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10.5, color="#607089", wrap=True)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    def _generate_overview_plots_guaranteed(self, outputs: dict[str, pd.DataFrame], plots_dir: Path) -> dict[str, str]:
        jobs = [
            ("overview_design_context", "Dataset design context", lambda p: plot_overview_design_from_tables(outputs.get("overview_design_metrics", pd.DataFrame()), outputs.get("overview_design_counts", pd.DataFrame()), p)),
            ("overview_role_mapping_summary", "Role mapping summary", lambda p: plot_role_mapping_summary(outputs.get("feature_role_summary", pd.DataFrame()), p)),
            ("overview_subject_task_counts", "Subject x task coverage", lambda p: plot_subject_task_summary_bars(outputs.get("overview_subject_task_coverage", pd.DataFrame()), p)),
            ("overview_feature_family_quality", "Feature-family coverage", lambda p: plot_feature_family_quality(outputs.get("feature_family_overview", pd.DataFrame()), p)),
            ("overview_feature_quality_landscape", "Feature-quality landscape", lambda p: plot_feature_quality_landscape(outputs.get("overview_feature_quality_landscape", pd.DataFrame()), p)),
        ]
        paths: dict[str, str] = {}
        for key, title, fn in jobs:
            path = plots_dir / f"{key}.png"
            try:
                paths[key] = str(fn(path))
            except Exception as exc:
                self.log(f"WARN | Overview safe plot {key} failed and was replaced by placeholder: {exc}")
                self.log(traceback.format_exc())
                paths[key] = str(self._write_overview_placeholder_plot(path, title, str(exc)))
        return paths

    def regenerate_overview_plots(self) -> None:
        """Regenerate the Overview page through a guaranteed row-safe path.

        This callback must never run deep feature diagnostics. If metadata is
        loaded, metadata/promoted metadata columns are used first. Filename
        context is only a no-metadata fallback configured in Metadata Mapping.
        """
        try:
            if self.feature_df is None:
                self.load_and_map()
            if self.feature_df is None:
                return
            if not self.output_edit.text().strip():
                QMessageBox.warning(self, "Missing output folder", "Please select an output folder before generating plots.")
                return
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            try:
                outputs, feature_cols = self._overview_emergency_outputs()
            except Exception as exc:
                self.log(f"ERROR | Emergency Overview builder failed: {exc}")
                self.log(traceback.format_exc())
                outputs, feature_cols = {"dataset_inventory": pd.DataFrame(), "feature_role_summary": pd.DataFrame(), "overview_design_metrics": pd.DataFrame(), "overview_design_counts": pd.DataFrame(), "dataset_design_overview": pd.DataFrame(), "feature_family_overview": pd.DataFrame(), "overview_feature_quality_landscape": pd.DataFrame(), "overview_subject_task_coverage": pd.DataFrame()}, []
            self.outputs = outputs
            try:
                self._write_outputs(outputs, tables_dir)
            except Exception as exc:
                self.log(f"WARN | Could not write some Overview tables: {exc}")
            self.plot_paths = self._generate_overview_plots_guaranteed(outputs, plots_dir)
            try:
                self.update_overview_dashboard(outputs)
            except Exception as exc:
                self.log(f"WARN | Overview dashboard table update failed but plots were generated: {exc}")
                self.log(traceback.format_exc())
            self.log(f"Overview plots generated in safe metadata-first mode: {plots_dir}")
            self.preview_plot("overview_design_context", generate_if_missing=False)
        except Exception as exc:
            self.log_error("Could not generate overview plots", exc)
            try:
                QMessageBox.critical(self, "Could not generate overview plots", str(exc))
            except Exception:
                pass

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
        regen.clicked.connect(self.regenerate_relationships)
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

        self.missing_note = QLabel("Use Task focus and Context/Value to inspect missingness within the relevant analysis subset. Tables below mirror the current scope.")
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
        self.missing_plot_combo.addItem("Feature availability summary", "feature_availability_summary")
        self.missing_plot_combo.addItem("Co-missingness pair summary", "missingness_comissing_pairs_plot")
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
            "Missingness summarizes feature availability by task and context. Use Task focus and Context/Value to localize missingness before making exclusion or imputation decisions."
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
        self.missing_task_combo.currentIndexChanged.connect(lambda _=0: (self._refresh_missingness_context_value_combo(self._missingness_scope_table()[0] if hasattr(self, "_missingness_scope_table") else self._active_analysis_table()), self.preview_missingness_plot(self.missing_plot_combo.currentData() if hasattr(self, "missing_plot_combo") else "missingness_top_features")))
        scope_row.addWidget(self.missing_task_combo, 1)
        scope_row.addWidget(QLabel("Context:"))
        self.missing_context_combo = QComboBox()
        self.missing_context_combo.setMinimumWidth(240)
        self.missing_context_combo.addItem("All context / not available")
        self.missing_context_combo.currentIndexChanged.connect(lambda _=0: (self._refresh_missingness_context_value_combo(self._active_analysis_table()), self.preview_missingness_plot(self.missing_plot_combo.currentData() if hasattr(self, "missing_plot_combo") else "missingness_top_features")))
        scope_row.addWidget(self.missing_context_combo, 1)
        scope_row.addWidget(QLabel("Value:"))
        self.missing_context_value_combo = QComboBox()
        self.missing_context_value_combo.setMinimumWidth(220)
        self.missing_context_value_combo.addItem("All values")
        self.missing_context_value_combo.currentIndexChanged.connect(lambda _=0: self.preview_missingness_plot(self.missing_plot_combo.currentData() if hasattr(self, "missing_plot_combo") else "missingness_top_features"))
        scope_row.addWidget(self.missing_context_value_combo, 1)
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
        side_panel.setStyleSheet(f"QFrame {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:14px; }}")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        side_title = QLabel("Availability snapshot")
        side_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        side_layout.addWidget(side_title)
        side_note = QLabel("Current-scope missingness metrics. Updates with Task focus and Context/Value.")
        side_note.setWordWrap(True)
        side_note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent; font-size:12px;")
        side_layout.addWidget(side_note)
        self.missing_metric_grid = QGridLayout()
        self.missing_metric_grid.setHorizontalSpacing(8)
        self.missing_metric_grid.setVerticalSpacing(8)
        side_layout.addLayout(self.missing_metric_grid)
        side_layout.addStretch(1)
        side_panel.setMinimumWidth(280)
        side_panel.setMaximumWidth(360)
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



    def _missingness_task_column(self, df: pd.DataFrame) -> str | None:
        """Return the task column for Missingness, preferring metadata-derived task when available.

        Metadata Mapping is the source of truth when metadata is loaded. Filename-derived
        task is used only when metadata task columns are absent or empty.
        """
        if df is None or df.empty:
            return None
        candidates = [
            "metadata__task", "metadata__task_name", "metadata__Task Name",
            "metadata__task_code",
            "task", "task_name", "Task Name", "parsed_task", "parsed_task_code", "task_code",
        ]
        for c in candidates:
            if c in df.columns:
                try:
                    if not self._is_effectively_empty(df[c]).all():
                        return c
                except Exception:
                    if df[c].notna().any():
                        return c
        try:
            return self._first_context_column(df, "task")
        except Exception:
            return None




    def _missingness_context_series(self, df: pd.DataFrame) -> tuple[pd.Series | None, str]:
        if df is None or df.empty or not hasattr(self, "missing_context_combo"):
            return None, "context"
        label = self.missing_context_combo.currentText()
        if label in {"", "All context / not available"}:
            return None, "context"
        role_map = {
            "Diagnosis": "diagnosis",
            "Severity score": "severity_score",
            "Severity bin": "severity_bin",
            "Sex or gender": "sex_or_gender",
            "Session": "session",
            "Subject": "subject",
            "Task code": "task_code",
            "Iteration": "iteration",
        }
        role = role_map.get(label)
        col = self._first_context_column(df, role) if role else None
        if not col:
            return None, label
        return df[col].astype("string"), label

    def _refresh_missingness_context_controls(self, df: pd.DataFrame) -> None:
        if not hasattr(self, "missing_context_combo"):
            return
        current = self.missing_context_combo.currentText()
        options = ["All context / not available"]
        for label, role in [
            ("Diagnosis", "diagnosis"),
            ("Severity score", "severity_score"),
            ("Severity bin", "severity_bin"),
            ("Sex or gender", "sex_or_gender"),
            ("Session", "session"),
            ("Subject", "subject"),
            ("Task code", "task_code"),
            ("Iteration", "iteration"),
        ]:
            if self._first_context_column(df, role):
                options.append(label)
        self.missing_context_combo.blockSignals(True)
        self.missing_context_combo.clear()
        for opt in options:
            self.missing_context_combo.addItem(opt)
        ix = self.missing_context_combo.findText(current)
        if ix >= 0:
            self.missing_context_combo.setCurrentIndex(ix)
        self.missing_context_combo.blockSignals(False)
        self._refresh_missingness_context_value_combo(df)

    def _refresh_missingness_context_value_combo(self, df: pd.DataFrame) -> None:
        if not hasattr(self, "missing_context_value_combo"):
            return
        current = self.missing_context_value_combo.currentText()
        series, label = self._missingness_context_series(df)
        self.missing_context_value_combo.blockSignals(True)
        self.missing_context_value_combo.clear()
        self.missing_context_value_combo.addItem("All values")
        if series is not None:
            vals = sorted([str(x) for x in series.dropna().astype(str).unique().tolist() if str(x).strip() and str(x).lower() not in {"nan", "none", "<na>"}])
            for v in vals[:250]:
                self.missing_context_value_combo.addItem(v)
        ix = self.missing_context_value_combo.findText(current)
        if ix >= 0:
            self.missing_context_value_combo.setCurrentIndex(ix)
        self.missing_context_value_combo.blockSignals(False)

    def _refresh_missingness_task_combo(self, df: pd.DataFrame) -> None:
        if not hasattr(self, "missing_task_combo"):
            return
        task_col = self._missingness_task_column(df) if hasattr(self, "_missingness_task_column") else (self._task_col(df) if hasattr(self, "_task_col") else None)
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
        """Return task/context-scoped table for Missingness plots only.

        This does not alter analysis outputs or tables. It is deliberately local
        so missingness can be inspected by task and clinical/context variables
        without rerunning the whole GUI.
        """
        df = self._active_analysis_table() if hasattr(self, "_active_analysis_table") else (self.analysis_df if self.analysis_df is not None else self.feature_df)
        if df is None:
            return pd.DataFrame(), "All tasks | all context"
        scoped = df.copy()
        parts = []
        task_col = self._missingness_task_column(scoped) if hasattr(self, "_missingness_task_column") else (self._task_col(scoped) if hasattr(self, "_task_col") else None)
        task_value = self.missing_task_combo.currentText() if hasattr(self, "missing_task_combo") else "All tasks"
        if task_col and task_col in scoped.columns and task_value not in {"", "All tasks", "All tasks / not available"}:
            scoped = scoped[scoped[task_col].astype(str).eq(str(task_value))].copy()
            parts.append(f"Task = {task_value}")
        else:
            parts.append("All tasks")

        series, label = self._missingness_context_series(scoped)
        value = self.missing_context_value_combo.currentText() if hasattr(self, "missing_context_value_combo") else "All values"
        if series is not None:
            aligned_series = self._align_context_series_to_frame(series, scoped) if hasattr(self, "_align_context_series_to_frame") else pd.Series(series.to_numpy() if len(series) == len(scoped) else pd.NA, index=scoped.index)
            scoped["__local_missingness_context__"] = aligned_series
            if value not in {"", "All values", "All context / not available"}:
                scoped = scoped[scoped["__local_missingness_context__"].astype(str).eq(str(value))].copy()
                parts.append(f"{label} = {value}")
            else:
                parts.append(f"{label} = all")
        else:
            parts.append("context = all/unavailable")
        return scoped, " | ".join(parts)

    def _missingness_feature_cols(self, df: pd.DataFrame) -> list[str]:
        if getattr(self, "mapping_df", None) is not None and not self.mapping_df.empty:
            roles = role_lists(self.mapping_df)
            return [c for c in roles.get("Feature", []) if c in df.columns]
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    def _missingness_scope_outputs(self) -> tuple[dict[str, pd.DataFrame], list[str], str]:
        """Build Missingness tables for the currently selected task/context scope.

        These tables are local to the Missingness menu. They are recomputed from
        the same scoped DataFrame used for plots so the detailed tabs and side
        snapshot always match Task focus and Context/Value. This avoids relying
        on stale full-analysis outputs that may not have been regenerated after
        the user changes scope.
        """
        df, scope_label = self._missingness_scope_table()
        feature_cols = self._missingness_feature_cols(df)
        outputs: dict[str, pd.DataFrame] = {}
        if df is None or df.empty or not feature_cols:
            outputs["missingness_by_feature"] = pd.DataFrame(columns=["feature", "missing_fraction", "n_missing", "n_total"])
            outputs["missingness_by_row"] = pd.DataFrame(columns=["row_index", "missing_fraction_feature_columns", "n_missing_feature_columns"])
            outputs["missingness_by_group"] = pd.DataFrame(columns=["group_variable", "column", "level", "n_rows", "mean_feature_missing_fraction"])
            outputs["missingness_by_family"] = pd.DataFrame(columns=["family_or_subsystem", "n_features", "mean_missing_fraction"])
            outputs["missingness_comissing_pairs"] = pd.DataFrame(columns=["feature_a", "feature_b", "co_missing_fraction", "n_co_missing"])
            outputs["missingness_scope_summary"] = pd.DataFrame([{
                "scope": scope_label,
                "n_rows": 0 if df is None else int(len(df)),
                "n_feature_columns": 0,
                "mean_missingness": pd.NA,
            }])
            return outputs, feature_cols, scope_label

        missing_feature = missingness_feature_summary(df, feature_cols, self.registry_df)
        row_missing = missingness_row_summary(df, feature_cols)
        group_missing = missingness_group_summary(df, feature_cols)
        family_missing = missingness_family_summary(missing_feature)
        comissing_pairs = missingness_comissing_pairs(df, feature_cols)
        mean_missing = pd.NA
        if missing_feature is not None and not missing_feature.empty and "missing_fraction" in missing_feature.columns:
            mean_missing = float(pd.to_numeric(missing_feature["missing_fraction"], errors="coerce").mean())
        outputs["missingness_by_feature"] = missing_feature
        outputs["missingness_by_row"] = row_missing
        outputs["missingness_by_group"] = group_missing
        outputs["missingness_by_family"] = family_missing
        outputs["missingness_comissing_pairs"] = comissing_pairs
        outputs["missingness_scope_summary"] = pd.DataFrame([{
            "scope": scope_label,
            "n_rows": int(len(df)),
            "n_feature_columns": int(len(feature_cols)),
            "mean_missingness": mean_missing,
            "task_focus": self.missing_task_combo.currentText() if hasattr(self, "missing_task_combo") else "All tasks",
            "context": self.missing_context_combo.currentText() if hasattr(self, "missing_context_combo") else "All context / not available",
            "value": self.missing_context_value_combo.currentText() if hasattr(self, "missing_context_value_combo") else "All values",
        }])
        return outputs, feature_cols, scope_label

    def _update_missingness_snapshot(self, outputs: dict[str, pd.DataFrame], scope_label: str | None = None) -> None:
        if not hasattr(self, "missing_metric_grid"):
            return
        while self.missing_metric_grid.count():
            item = self.missing_metric_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        feat = outputs.get("missingness_by_feature", pd.DataFrame()) if isinstance(outputs, dict) else pd.DataFrame()
        row = outputs.get("missingness_by_row", pd.DataFrame()) if isinstance(outputs, dict) else pd.DataFrame()
        group = outputs.get("missingness_by_group", pd.DataFrame()) if isinstance(outputs, dict) else pd.DataFrame()
        summary = outputs.get("missingness_scope_summary", pd.DataFrame()) if isinstance(outputs, dict) else pd.DataFrame()
        n_features = len(feat) if feat is not None else 0
        n_rows = len(row) if row is not None else 0
        if summary is not None and not summary.empty:
            try:
                n_rows = int(summary.iloc[0].get("n_rows", n_rows))
            except Exception:
                pass
        mean_miss = "-"
        high_features = "-"
        high_rows = "-"
        groups = "-"
        if feat is not None and not feat.empty and "missing_fraction" in feat.columns:
            miss = pd.to_numeric(feat["missing_fraction"], errors="coerce")
            mean_miss = f"{miss.mean():.3f}" if miss.notna().any() else "-"
            high_features = int((miss >= 0.50).sum())
        if row is not None and not row.empty and "missing_fraction_feature_columns" in row.columns:
            high_rows = int((pd.to_numeric(row["missing_fraction_feature_columns"], errors="coerce") >= 0.50).sum())
        if group is not None and not group.empty and "group_variable" in group.columns:
            groups = int(group["group_variable"].nunique())
        scope_text = scope_label or "current scope"
        if len(str(scope_text)) > 42:
            scope_text = str(scope_text)[:39] + "..."
        tiles = [
            ("Scope", scope_text, "task/context filter"),
            ("Rows", n_rows, "recordings in scope"),
            ("Features", n_features, "mapped feature columns"),
            ("Mean missingness", mean_miss, "across features"),
            ("High-missing features", high_features, ">=50% missing"),
            ("High-missing rows", high_rows, ">=50% feature missingness"),
            ("Metadata groups", groups, "detected strata"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.missing_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx, 0)

    def _update_missingness_tables(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "missing_feature_table"):
            return
        self._fill_table(self.missing_feature_table, outputs.get("missingness_by_feature", pd.DataFrame()))
        self._fill_table(self.missing_row_table, outputs.get("missingness_by_row", pd.DataFrame()))
        self._fill_table(self.missing_group_table, outputs.get("missingness_by_group", pd.DataFrame()))
        self._fill_table(self.missing_family_table, outputs.get("missingness_by_family", pd.DataFrame()))
        self._fill_table(self.missing_comissing_table, outputs.get("missingness_comissing_pairs", pd.DataFrame()))

    def generate_missingness_scope_plots(self) -> None:
        self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
        outputs, feature_cols, scope_label = self._missingness_scope_outputs()
        safe_scope = re.sub(r"[^A-Za-z0-9_.-]+", "_", scope_label).strip("_")[:80] or "all_tasks"
        if not hasattr(self, "plot_paths"):
            self.plot_paths = {}
        missing_feature = outputs.get("missingness_by_feature", pd.DataFrame())
        row_missing = outputs.get("missingness_by_row", pd.DataFrame())
        group_missing = outputs.get("missingness_by_group", pd.DataFrame())
        family_missing = outputs.get("missingness_by_family", pd.DataFrame())
        comissing_pairs = outputs.get("missingness_comissing_pairs", pd.DataFrame())
        self.plot_paths["missingness_top_features"] = str(plot_missingness(missing_feature, plots_dir / f"missingness_top_features_{safe_scope}.png"))
        self.plot_paths["missingness_row_distribution"] = str(plot_row_missingness_distribution(row_missing, plots_dir / f"missingness_row_distribution_{safe_scope}.png"))
        self.plot_paths["missingness_by_group"] = str(plot_missingness_by_group(group_missing, plots_dir / f"missingness_by_group_{safe_scope}.png"))
        self.plot_paths["missingness_by_family"] = str(plot_missingness_family_summary(family_missing, plots_dir / f"missingness_by_family_{safe_scope}.png"))
        self.plot_paths["feature_availability_summary"] = str(plot_feature_availability_bars(missing_feature, plots_dir / f"feature_availability_summary_{safe_scope}.png"))
        self.plot_paths["missingness_comissing_pairs_plot"] = str(plot_comissing_pair_bars(comissing_pairs, plots_dir / f"missingness_comissing_pairs_{safe_scope}.png"))
        # Backward-compatible aliases for older saved reports/buttons. These now
        # point to readable bar summaries rather than dense heatmaps.
        self.plot_paths["feature_availability_heatmap"] = self.plot_paths["feature_availability_summary"]
        self.plot_paths["missingness_comissing_heatmap"] = self.plot_paths["missingness_comissing_pairs_plot"]
        self.current_missingness_outputs = outputs
        self._update_missingness_snapshot(outputs, scope_label)
        self._update_missingness_tables(outputs)
        if hasattr(self, "missing_plot_caption"):
            self.missing_plot_caption.setText(f"{self._missingness_plot_caption_text(self.missing_plot_combo.currentData())}\n\nCurrent scope: {scope_label}.")
        if hasattr(self, "missing_note"):
            self.missing_note.setText("Missingness audit generated for the current scope. Tables and snapshot update with Task focus and Context/Value.")

    def regenerate_missingness_scope_plots(self) -> None:
        self.generate_missingness_scope_plots()
        self.preview_missingness_plot(self.missing_plot_combo.currentData() if hasattr(self, "missing_plot_combo") else "missingness_top_features", regenerate=False)

    def update_missingness_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "missing_metric_grid"):
            return
        active_missing_scope = self._active_analysis_table() if hasattr(self, "_active_analysis_table") else pd.DataFrame()
        self._refresh_missingness_task_combo(active_missing_scope)
        self._refresh_missingness_context_controls(active_missing_scope)
        # Build scoped outputs from the current canonical analysis table so the
        # detailed tabs and snapshot are never empty/stale after task/context changes.
        try:
            scoped_outputs, _feature_cols, scope_label = self._missingness_scope_outputs()
        except Exception as exc:
            self.log(f"WARN | Missingness scoped table update failed; using full-analysis outputs. {exc}") if hasattr(self, "log") else None
            scoped_outputs = outputs or {}
            scope_label = "All tasks | context = all/unavailable"
        self.current_missingness_outputs = scoped_outputs
        self._update_missingness_snapshot(scoped_outputs, scope_label)
        self._update_missingness_tables(scoped_outputs)
        try:
            self.generate_missingness_scope_plots()
        except Exception as exc:
            self.log(f"WARN | Missingness scoped plot refresh failed: {exc}") if hasattr(self, "log") else None
        self.missing_note.setText("Missingness audit generated for the current scope. Tables and snapshot update with Task focus and Context/Value.")

    def _missingness_plot_caption_text(self, key: str) -> str:
        captions = {
            "missingness_top_features": "Top missing features: ranks feature columns by missing fraction. Features near the top may be unsupported for some tasks, sensitive to signal quality, or computationally unstable. Do not exclude automatically; first check whether missingness is task-, group-, or QC-linked.",
            "missingness_row_distribution": "Row-level missingness: shows how much feature information is lost per recording/row. A right-shifted distribution means many recordings have broad feature failure, which can reduce usable sample size and bias ML training.",
            "missingness_by_group": "Missingness by group: compares average feature missingness across detected task, diagnosis, severity, device, session, or similar groups. Group differences suggest missingness may be non-random and should not be handled by naive complete-case analysis.",
            "missingness_by_family": "Missingness by family: summarizes failure by feature subsystem. A high family-level value suggests a systematic issue, such as formant tracking, voicing detection, segmentation support, or missing registry labels, rather than isolated bad features.",
            "feature_availability_summary": "Feature availability summary: shows available versus missing counts for the least available features in the current task/context scope. This replaces the dense heatmap with a readable bar summary.",
            "missingness_comissing_pairs_plot": "Co-missingness pair summary: shows the feature pairs that are most often missing together in the current task/context scope. Review these as shared algorithmic or task-support failure modes.",
            "feature_availability_heatmap": "Feature availability summary: shows available versus missing counts for the least available features in the current task/context scope.",
            "missingness_comissing_heatmap": "Co-missingness pair summary: shows feature pairs that are most often missing together in the current task/context scope.",
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
                f"{self._missingness_plot_caption_text(key)}\n\nCurrent scope: {scope_label}."
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
            "Clinician-facing distribution and outlier triage: review feature value shape, expected-range violations, robust outliers, row-level burden, and selected-feature diagnostics within the active task scope."
        )

        self.dist_note = QLabel("Run Feature Analysis to populate distribution/outlier diagnostics. Use Task focus and Feature to inspect to move from global triage to selected-feature review.")
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
        self.dist_plot_combo.setMinimumWidth(260)
        self.dist_plot_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dist_plot_combo.addItem("Review status", "distribution_review_status")
        self.dist_plot_combo.addItem("Shape and tail audit", "distribution_shape_story")
        self.dist_plot_combo.addItem("Range and outlier triage", "distribution_outlier_range_story")
        self.dist_plot_combo.addItem("Recording outlier burden", "row_outlier_story")
        self.dist_plot_combo.addItem("Variance screen", "variance_screen")
        self.dist_plot_combo.addItem("Selected feature diagnostic", "selected_feature_distribution")
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
            "Task focus filters every plot. Review status gives the high-level triage; Shape and tail audit summarizes distribution shape risk; Range and outlier triage links expected-range violations with robust-outlier burden; Selected feature diagnostic provides the detailed feature-level review."
        )
        self.dist_plot_caption.setWordWrap(True)
        self.dist_plot_caption.setStyleSheet(f"color:{MUTED}; background:#FFFFFF; border:1px solid {LINE}; border-radius:8px; padding:9px;")
        plot_panel_layout.addWidget(self.dist_plot_caption)

        scope_row = QHBoxLayout()
        scope_row.setSpacing(10)
        scope_row.addWidget(QLabel("Task focus:"))
        self.dist_task_combo = QComboBox()
        self.dist_task_combo.setMinimumWidth(200)
        self.dist_task_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dist_task_combo.addItem("All tasks / not available")
        self.dist_task_combo.currentIndexChanged.connect(lambda _=0: self.preview_distribution_plot(self.dist_plot_combo.currentData() if hasattr(self, "dist_plot_combo") else "distribution_review_status"))
        scope_row.addWidget(self.dist_task_combo, 1)
        plot_panel_layout.addLayout(scope_row)

        selectors = QHBoxLayout()
        selectors.setSpacing(10)
        selectors.addWidget(QLabel("Feature to inspect:"))
        self.dist_feature_combo = QComboBox()
        self.dist_feature_combo.setMinimumWidth(240)
        self.dist_feature_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dist_feature_combo.currentTextChanged.connect(lambda _: self.generate_selected_distribution_plot())
        selectors.addWidget(self.dist_feature_combo, 1)
        selectors.addStretch(1)
        plot_panel_layout.addLayout(selectors)

        self.dist_plot_preview = QLabel("Run Feature Analysis, then choose one distribution plot.")
        self.dist_plot_preview.setAlignment(Qt.AlignCenter)
        self.dist_plot_preview.setMinimumHeight(440)
        self.dist_plot_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dist_plot_preview.setStyleSheet(f"QLabel {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; color:{MUTED}; padding:16px; }}")
        plot_panel_layout.addWidget(self.dist_plot_preview, 1)

        self.dist_interpretation_label = QLabel("Select a plot to see structured interpretation guidance.")
        self.dist_interpretation_label.setWordWrap(True)
        self.dist_interpretation_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.dist_interpretation_label.setStyleSheet(f"QLabel {{ background:#FFFFFF; color:{INK}; border:1px solid {LINE}; border-radius:10px; padding:12px; font-size:12px; line-height:140%; }}")
        plot_panel_layout.addWidget(self.dist_interpretation_label)

        plot_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        dist_split.addWidget(plot_panel, 4)

        # Side summary keeps distribution metrics visible without occupying the
        # top of the page, matching Overview and Missingness.
        side_panel = QFrame()
        side_panel.setStyleSheet(f"QFrame {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:14px; }}")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        side_title = QLabel("Distribution snapshot")
        side_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        side_layout.addWidget(side_title)
        side_note = QLabel("Compact current-scope triage metrics. These update with Task focus and point to the detailed tables below.")
        side_note.setWordWrap(True)
        side_note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent; font-size:12px;")
        side_layout.addWidget(side_note)
        self.dist_metric_grid = QGridLayout()
        self.dist_metric_grid.setHorizontalSpacing(8)
        self.dist_metric_grid.setVerticalSpacing(8)
        side_layout.addLayout(self.dist_metric_grid)
        side_layout.addStretch(1)
        side_panel.setMinimumWidth(260)
        side_panel.setMaximumWidth(340)
        side_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        dist_split.addWidget(side_panel, 0)
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
        tabs.addTab(self.dist_summary_table, "Feature diagnostics")
        tabs.addTab(self.dist_review_table, "Review summary")
        tabs.addTab(self.dist_outlier_table, "Row-level outliers")
        tabs.addTab(self.dist_range_table, "Expected ranges")
        tabs.addTab(self.dist_shape_table, "Shape audit")
        tabs.addTab(self.dist_row_burden_table, "Recording burden")
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
        return out, " | ".join(parts)

    def _distribution_feature_cols(self, df: pd.DataFrame) -> list[str]:
        if getattr(self, "mapping_df", None) is not None and not self.mapping_df.empty:
            roles = role_lists(self.mapping_df)
            return [c for c in roles.get("Feature", []) if c in df.columns]
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    def _distribution_scope_outputs(self) -> tuple[dict[str, pd.DataFrame], list[str], str]:
        """Build row-safe distribution outputs for the current Task focus.

        This mirrors the Missingness menu pattern: all plots, snapshot tiles, and
        detailed tables are built from the same scoped DataFrame. It never
        mutates analysis_df and never relies on stale full-analysis outputs.
        """
        df, scope_label = self._distribution_scope_table()
        feature_cols = self._distribution_feature_cols(df)
        outputs: dict[str, pd.DataFrame] = {}
        if df is None or df.empty or not feature_cols:
            outputs["feature_distribution_summary"] = pd.DataFrame(columns=["feature", "n", "missing_fraction", "median", "iqr", "robust_outlier_fraction", "zero_variance", "near_zero_variance"])
            outputs["feature_expected_range_flags"] = pd.DataFrame(columns=["feature", "fraction_outside_expected", "n_outside_expected"])
            outputs["distribution_review_summary"] = pd.DataFrame(columns=["feature", "distribution_status"])
            outputs["distribution_shape_audit"] = pd.DataFrame(columns=["feature", "priority", "skew_proxy", "tail_ratio"])
            outputs["robust_outlier_flags"] = pd.DataFrame(columns=["row_index", "feature", "value", "robust_z"])
            outputs["row_outlier_burden_summary"] = pd.DataFrame(columns=["row_index", "n_flagged_features", "fraction_flagged_features", "review_level"])
            outputs["distribution_scope_summary"] = pd.DataFrame([{"scope": scope_label, "n_rows": 0 if df is None else int(len(df)), "n_feature_columns": 0}])
            return outputs, feature_cols, scope_label
        dist = feature_distribution_summary(df, feature_cols)
        expected = expected_range_flags(df, feature_cols, self.registry_df)
        review = distribution_review_summary(dist, expected)
        shape = distribution_shape_audit(dist, expected)
        outliers = robust_outlier_flags(df, feature_cols, self.registry_df)
        row_burden = row_outlier_burden_summary(outliers, len(feature_cols))
        outputs["feature_distribution_summary"] = dist
        outputs["feature_expected_range_flags"] = expected
        outputs["distribution_review_summary"] = review
        outputs["distribution_shape_audit"] = shape
        outputs["robust_outlier_flags"] = outliers
        outputs["row_outlier_burden_summary"] = row_burden
        outputs["distribution_scope_summary"] = pd.DataFrame([{
            "scope": scope_label,
            "n_rows": int(len(df)),
            "n_feature_columns": int(len(feature_cols)),
            "task_focus": self.dist_task_combo.currentText() if hasattr(self, "dist_task_combo") else "All tasks",
        }])
        return outputs, feature_cols, scope_label

    def _update_distribution_snapshot(self, outputs: dict[str, pd.DataFrame], scope_label: str | None = None) -> None:
        if not hasattr(self, "dist_metric_grid"):
            return
        while self.dist_metric_grid.count():
            item = self.dist_metric_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        dist = outputs.get("feature_distribution_summary", pd.DataFrame()) if isinstance(outputs, dict) else pd.DataFrame()
        review = outputs.get("distribution_review_summary", pd.DataFrame()) if isinstance(outputs, dict) else pd.DataFrame()
        outliers = outputs.get("robust_outlier_flags", pd.DataFrame()) if isinstance(outputs, dict) else pd.DataFrame()
        ranges = outputs.get("feature_expected_range_flags", pd.DataFrame()) if isinstance(outputs, dict) else pd.DataFrame()
        shape = outputs.get("distribution_shape_audit", pd.DataFrame()) if isinstance(outputs, dict) else pd.DataFrame()
        row_burden = outputs.get("row_outlier_burden_summary", pd.DataFrame()) if isinstance(outputs, dict) else pd.DataFrame()
        summary = outputs.get("distribution_scope_summary", pd.DataFrame()) if isinstance(outputs, dict) else pd.DataFrame()
        n_features = len(dist) if dist is not None else 0
        n_rows = 0
        if summary is not None and not summary.empty:
            try:
                n_rows = int(summary.iloc[0].get("n_rows", 0))
            except Exception:
                n_rows = 0
        status = review.get("distribution_status", pd.Series(dtype=str)).astype(str) if review is not None and not review.empty else pd.Series(dtype=str)
        monitor = int(status.eq("monitor").sum()) if not status.empty else 0
        review_n = int(status.eq("review").sum()) if not status.empty else 0
        range_n = int(pd.to_numeric(ranges.get("fraction_outside_expected", pd.Series(dtype=float)), errors="coerce").gt(0).sum()) if ranges is not None and not ranges.empty else 0
        out_features = int(outliers["feature"].nunique()) if outliers is not None and not outliers.empty and "feature" in outliers.columns else 0
        zero_n = int(dist.get("zero_variance", pd.Series(dtype=bool)).fillna(False).sum()) if dist is not None and not dist.empty else 0
        row_review = int(row_burden.get("review_level", pd.Series(dtype=str)).astype(str).eq("review").sum()) if row_burden is not None and not row_burden.empty else 0
        scope_text = scope_label or "current scope"
        if len(str(scope_text)) > 42:
            scope_text = str(scope_text)[:39] + "..."
        tiles = [
            ("Scope", scope_text, "task filter"),
            ("Rows", n_rows, "recordings in scope"),
            ("Features", n_features, "mapped numeric features"),
            ("Monitor", monitor, "moderate diagnostics"),
            ("Review", review_n, "highest-priority features"),
            ("Range flags", range_n, "features outside policy range"),
            ("Outlier features", out_features, "robust outlier evidence"),
            ("Row burden", row_review, "recordings needing review"),
            ("Zero variance", zero_n, "not informative in scope"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.dist_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx, 0)

    def _update_distribution_tables(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "dist_summary_table"):
            return
        self._fill_table(self.dist_summary_table, outputs.get("feature_distribution_summary", pd.DataFrame()))
        self._fill_table(self.dist_review_table, outputs.get("distribution_review_summary", pd.DataFrame()))
        self._fill_table(self.dist_outlier_table, outputs.get("robust_outlier_flags", pd.DataFrame()))
        self._fill_table(self.dist_range_table, outputs.get("feature_expected_range_flags", pd.DataFrame()))
        self._fill_table(self.dist_shape_table, outputs.get("distribution_shape_audit", pd.DataFrame()))
        self._fill_table(self.dist_row_burden_table, outputs.get("row_outlier_burden_summary", pd.DataFrame()))

    def _update_distribution_feature_selector(self, dist: pd.DataFrame) -> None:
        if not hasattr(self, "dist_feature_combo"):
            return
        current = self.dist_feature_combo.currentText()
        self.dist_feature_combo.blockSignals(True)
        self.dist_feature_combo.clear()
        if dist is not None and not dist.empty and "feature" in dist.columns:
            # Prioritize review/monitor features when available.
            ordered = dist.copy()
            if "robust_outlier_fraction" in ordered.columns:
                ordered["__sort__"] = pd.to_numeric(ordered["robust_outlier_fraction"], errors="coerce").fillna(0)
                ordered = ordered.sort_values("__sort__", ascending=False)
            self.dist_feature_combo.addItems([str(x) for x in ordered["feature"].dropna().tolist()])
        if current:
            ix = self.dist_feature_combo.findText(current)
            if ix >= 0:
                self.dist_feature_combo.setCurrentIndex(ix)
        self.dist_feature_combo.blockSignals(False)

    def generate_distribution_scope_plots(self) -> None:
        self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
        outputs, feature_cols, scope_label = self._distribution_scope_outputs()
        safe_scope = re.sub(r"[^A-Za-z0-9_.-]+", "_", scope_label).strip("_")[:90] or "distribution_scope"
        if not hasattr(self, "plot_paths"):
            self.plot_paths = {}
        dist = outputs.get("feature_distribution_summary", pd.DataFrame())
        expected = outputs.get("feature_expected_range_flags", pd.DataFrame())
        review = outputs.get("distribution_review_summary", pd.DataFrame())
        shape = outputs.get("distribution_shape_audit", pd.DataFrame())
        outliers = outputs.get("robust_outlier_flags", pd.DataFrame())
        row_burden = outputs.get("row_outlier_burden_summary", pd.DataFrame())
        self.plot_paths["distribution_review_status"] = str(plot_distribution_review_summary(review, plots_dir / f"distribution_review_status_{safe_scope}.png"))
        self.plot_paths["distribution_shape_story"] = str(plot_distribution_shape_story(shape, plots_dir / f"distribution_shape_story_{safe_scope}.png"))
        self.plot_paths["distribution_outlier_range_story"] = str(plot_distribution_outlier_range_story(expected, outliers, review, plots_dir / f"distribution_outlier_range_story_{safe_scope}.png"))
        self.plot_paths["row_outlier_story"] = str(plot_row_outlier_story(row_burden, plots_dir / f"row_outlier_story_{safe_scope}.png"))
        self.plot_paths["variance_screen"] = str(plot_variance_screen(dist, plots_dir / f"variance_screen_{safe_scope}.png"))
        # Backward-compatible keys retained for reports/tests. These now point to
        # the consolidated diagnostic panels where appropriate.
        self.plot_paths["distribution_shape_summary"] = self.plot_paths["distribution_shape_story"]
        self.plot_paths["distribution_shape_landscape"] = self.plot_paths["distribution_shape_story"]
        self.plot_paths["expected_range_flags"] = self.plot_paths["distribution_outlier_range_story"]
        self.plot_paths["outlier_counts"] = self.plot_paths["distribution_outlier_range_story"]
        self.plot_paths["row_outlier_burden"] = self.plot_paths["row_outlier_story"]
        self.current_distribution_outputs = outputs
        self._update_distribution_snapshot(outputs, scope_label)
        self._update_distribution_tables(outputs)
        self._update_distribution_feature_selector(dist)
        if hasattr(self, "dist_plot_caption"):
            self.dist_plot_caption.setText(f"{self._distribution_plot_caption_text(self.dist_plot_combo.currentData())}\n\nCurrent scope: {scope_label}.")
        if hasattr(self, "dist_note"):
            self.dist_note.setText("Distribution/outlier audit generated for the current task scope. Snapshot, detailed tables, and plots all use the same scoped rows.")

    def regenerate_distribution_scope_plots(self) -> None:
        self.generate_distribution_scope_plots()
        self.preview_distribution_plot(self.dist_plot_combo.currentData() if hasattr(self, "dist_plot_combo") else "distribution_review_status", regenerate=False)

    def update_distribution_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "dist_metric_grid"):
            return
        active_for_scope = self._active_analysis_table() if hasattr(self, "_active_analysis_table") else (self.analysis_df if self.analysis_df is not None else self.feature_df)
        self._refresh_dist_task_combo(active_for_scope)
        try:
            scoped_outputs, _feature_cols, scope_label = self._distribution_scope_outputs()
        except Exception as exc:
            self.log(f"WARN | Distribution scoped table update failed; using full-analysis outputs. {exc}") if hasattr(self, "log") else None
            scoped_outputs = outputs or {}
            scope_label = "All tasks"
        self.current_distribution_outputs = scoped_outputs
        self._update_distribution_snapshot(scoped_outputs, scope_label)
        self._update_distribution_tables(scoped_outputs)
        self._update_distribution_feature_selector(scoped_outputs.get("feature_distribution_summary", pd.DataFrame()))
        try:
            self.generate_distribution_scope_plots()
        except Exception as exc:
            self.log(f"WARN | Distribution scoped plot refresh failed: {exc}") if hasattr(self, "log") else None
        self.dist_note.setText("Distribution/outlier audit generated for the current task scope. Use the selected-feature diagnostic for the final feature-level review.")

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

    def _distribution_plot_caption_text(self, key: str) -> str:
        captions = {
            "distribution_review_status": "Review status: one high-level triage bar showing how many features are ok, monitor, or review in the current task scope.",
            "distribution_shape_story": "Shape and tail audit: combines shape priority with skew/tail burden so you can see whether distribution risk is broad or concentrated in a small feature group.",
            "distribution_outlier_range_story": "Range and outlier triage: compares expected-range violations and robust-outlier burden to identify features needing unit, QC, or task-compatibility review.",
            "row_outlier_story": "Recording outlier burden: shows whether outlier flags are concentrated in a few recordings or spread across the dataset.",
            "variance_screen": "Variance screen: identifies zero or near-zero variance features that are unlikely to help downstream models in the current task scope.",
            "selected_feature_distribution": "Selected feature diagnostic: detailed single-feature distribution, expected bounds, and task grouping. Use this before excluding or transforming a feature.",
            "distribution_shape_summary": "Shape and tail audit: combined priority and shape landscape.",
            "distribution_shape_landscape": "Shape and tail audit: combined priority and shape landscape.",
            "expected_range_flags": "Range and outlier triage: expected range plus outlier evidence.",
            "outlier_counts": "Range and outlier triage: expected range plus outlier evidence.",
            "row_outlier_burden": "Recording outlier burden: row-level burden summary.",
        }
        return captions.get(str(key), "Distribution/outlier plot for the current task scope.")

    def preview_distribution_plot(self, key: str, regenerate: bool = True) -> None:
        if key == "selected_feature_distribution":
            self.generate_selected_distribution_plot()
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
        group = "task focus"
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
            "Choose Manual QC to review human metadata flags, or Automated QC to review the uploaded QC table. Manual QC is filtered by the Project modality setting so acoustic projects use acquisition/audio flags and kinematic projects use acquisition/video/face-visibility flags."
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
        self.qc_plot_combo.setMinimumWidth(300)
        self.qc_plot_combo.addItem("QC source overview", "qc_artifact_model")
        self.qc_plot_combo.addItem("QC flag/metric burden", "qc_family_burden")
        self.qc_plot_combo.addItem("QC value distributions", "qc_metric_distributions")
        self.qc_plot_combo.addItem("Feature × QC-family heatmap", "qc_feature_association_heatmap")
        self.qc_plot_combo.addItem("Top feature-QC associations", "qc_top_feature_associations")
        self.qc_plot_combo.addItem("Missingness linked to QC", "qc_missingness_associations")
        self.qc_plot_combo.addItem("Recording QC burden", "qc_row_burden")
        self.qc_plot_combo.addItem("Selected feature × selected QC", "selected_feature_qc_scatter")
        plot_header.addWidget(self.qc_plot_combo, 1)

        show_btn = QPushButton("Show")
        show_btn.setProperty("secondary", True)
        show_btn.clicked.connect(lambda: self.preview_qc_plot(self.qc_plot_combo.currentData()))
        plot_header.addWidget(show_btn)

        regen = QPushButton("Regenerate")
        regen.clicked.connect(self.regenerate_qc_scope_plots)
        plot_header.addWidget(regen)

        plot_header.addStretch(1)
        open_btn = QPushButton("Open current plot")
        open_btn.setProperty("secondary", True)
        open_btn.clicked.connect(self.open_current_qc_plot)
        plot_header.addWidget(open_btn)
        plot_panel_layout.addLayout(plot_header)

        self.qc_plot_caption = QLabel(
            "QC Integration asks whether feature values, missingness, or row outliers are linked to quality-control flags/metrics rather than clinical signal. Manual QC uses metadata flags; Automated QC uses the uploaded QC table only when selected."
        )
        self.qc_plot_caption.setWordWrap(True)
        self.qc_plot_caption.setStyleSheet(f"color:{MUTED}; background:#FFFFFF; border:1px solid {LINE}; border-radius:8px; padding:9px;")
        plot_panel_layout.addWidget(self.qc_plot_caption)

        scope_row = QHBoxLayout()
        scope_row.setSpacing(10)
        scope_row.addWidget(QLabel("QC source:"))
        self.qc_source_combo = QComboBox()
        self.qc_source_combo.setMinimumWidth(180)
        self.qc_source_combo.addItems(["Manual QC", "Automated QC"])
        self.qc_source_combo.currentIndexChanged.connect(lambda _=0: (self._refresh_qc_selector_combos(), self.preview_qc_plot(self.qc_plot_combo.currentData() if hasattr(self, "qc_plot_combo") else "qc_artifact_model")))
        scope_row.addWidget(self.qc_source_combo, 0)
        scope_row.addWidget(QLabel("Task focus:"))
        self.qc_task_combo = QComboBox()
        self.qc_task_combo.setMinimumWidth(220)
        self.qc_task_combo.addItem("All tasks / not available")
        self.qc_task_combo.currentIndexChanged.connect(lambda _=0: (self._refresh_qc_selector_combos(), self.preview_qc_plot(self.qc_plot_combo.currentData() if hasattr(self, "qc_plot_combo") else "qc_artifact_model")))
        scope_row.addWidget(self.qc_task_combo, 1)
        plot_panel_layout.addLayout(scope_row)

        selectors = QHBoxLayout()
        selectors.setSpacing(10)
        selectors.addWidget(QLabel("Feature:"))
        self.qc_feature_combo = QComboBox()
        self.qc_feature_combo.setMinimumWidth(320)
        self.qc_feature_combo.currentIndexChanged.connect(lambda _=0: self.preview_qc_plot("selected_feature_qc_scatter") if hasattr(self, "qc_plot_combo") and self.qc_plot_combo.currentData() == "selected_feature_qc_scatter" else None)
        selectors.addWidget(self.qc_feature_combo, 1)
        selectors.addWidget(QLabel("QC metric:"))
        self.qc_metric_combo = QComboBox()
        self.qc_metric_combo.setMinimumWidth(220)
        self.qc_metric_combo.currentIndexChanged.connect(lambda _=0: self.preview_qc_plot("selected_feature_qc_scatter") if hasattr(self, "qc_plot_combo") and self.qc_plot_combo.currentData() == "selected_feature_qc_scatter" else None)
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
        side_panel.setStyleSheet(f"QFrame {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:14px; }}")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        side_title = QLabel("QC snapshot")
        side_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        side_layout.addWidget(side_title)
        side_note = QLabel("Compact metrics for the selected QC source and task. Manual QC is modality-aware; Automated QC is used only when selected.")
        side_note.setWordWrap(True)
        side_note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent; font-size:12px;")
        side_layout.addWidget(side_note)
        self.qc_metric_grid = QGridLayout()
        self.qc_metric_grid.setHorizontalSpacing(8)
        self.qc_metric_grid.setVerticalSpacing(8)
        side_layout.addLayout(self.qc_metric_grid)
        side_layout.addStretch(1)
        side_panel.setMinimumWidth(280)
        side_panel.setMaximumWidth(360)
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
        self.qc_source_table = self._simple_table()
        tabs.addTab(self.qc_summary_table, "Summary")
        tabs.addTab(self.qc_source_table, "QC source variables")
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


    def _project_modality_mode(self) -> str:
        """Return the modality selected in Project setup as a stable key."""
        text = ""
        if hasattr(self, "modality_combo"):
            text = str(self.modality_combo.currentText() or "")
        n = normalize_name(text)
        if "acoustic" in n and "kinematic" not in n:
            return "acoustic"
        if "kinematic" in n and "acoustic" not in n:
            return "kinematic"
        if "mixed" in n or ("acoustic" in n and "kinematic" in n):
            return "mixed"
        if "generic" in n:
            return "generic"
        return "auto"

    def _selected_qc_source_mode(self) -> str:
        if hasattr(self, "qc_source_combo"):
            val = self.qc_source_combo.currentText().strip()
            if val in {"Manual QC", "Automated QC"}:
                return val
        return "Manual QC"

    def _manual_qc_family_for_role(self, role: str, col: str) -> str | None:
        """Map accepted metadata-QC roles into manual QC families.

        Manual QC must be driven by the accepted Metadata Mapping role, not by
        loose header keywords over the joined analysis table.  The previous
        header fallback was too broad: feature/clinical names containing words
        such as ``speech`` or ``audio`` were pulled into the manual-QC panel.
        Here, a non-empty non-manual role is treated as an explicit exclusion.
        Header inference is kept only for pre-mapping/legacy states where no
        accepted role exists at all.
        """
        role = str(role or "").strip()
        n = normalize_name(col)
        if role in {"Manual acquisition QC flag", "Task validity flag", "Parsing-needed flag"}:
            return "Manual acquisition QC"
        if role == "Manual audio QC flag":
            return "Manual audio QC"
        if role == "Manual video QC flag":
            return "Manual video QC"
        if role in {"Manual face/visibility QC flag", "Appearance/accessory flag"}:
            return "Manual face/visibility QC"
        if role and not role.startswith("--"):
            return None
        # Legacy/header fallback only when no assigned metadata role is known.
        if any(t in n for t in ["task_completed", "completed_as_instructed", "needs_parsing", "parsing_needed", "poor_light", "blurry", "acquisition"]):
            return "Manual acquisition QC"
        if any(t in n for t in ["another_person_speaks", "background_noise", "poor_audio", "audio_quality", "volume_is_unstable", "microphone"]):
            return "Manual audio QC"
        if any(t in n for t in ["another_person_in_frame", "frozen_video", "video_is_unstable"]):
            return "Manual video QC"
        if any(t in n for t in ["face_visibility", "subject_looks_away", "wearing_glasses", "facial_hair", "mouth_visible", "occlusion"]):
            return "Manual face/visibility QC"
        return None

    def _manual_qc_family_allowed(self, family: str | None) -> bool:
        modality = self._project_modality_mode()
        if family is None:
            return False
        if modality == "acoustic":
            return family in {"Manual acquisition QC", "Manual audio QC"}
        if modality == "kinematic":
            return family in {"Manual acquisition QC", "Manual video QC", "Manual face/visibility QC"}
        # Auto, mixed, and generic keep all manual QC families because the user has
        # not chosen a single acquisition channel.
        return family in {"Manual acquisition QC", "Manual audio QC", "Manual video QC", "Manual face/visibility QC"}

    def _find_analysis_column(self, df: pd.DataFrame, *names: str) -> str | None:
        if df is None or df.empty:
            return None
        exact = {str(c): str(c) for c in df.columns}
        norm = {}
        for c in df.columns:
            norm.setdefault(normalize_name(c), str(c))
        for name in names:
            if not name:
                continue
            candidates = [str(name), f"metadata__{name}"]
            for cand in candidates:
                if cand in exact:
                    return exact[cand]
                hit = norm.get(normalize_name(cand))
                if hit:
                    return hit
        return None

    def _manual_qc_badness(self, s: pd.Series, col: str) -> pd.Series:
        """Convert yes/no human QC annotations into 0=not flagged, 1=flagged."""
        n = normalize_name(col)
        raw = s.astype("object")
        num = pd.to_numeric(raw, errors="coerce")
        text = raw.astype(str).str.strip().str.lower()
        out = pd.Series(pd.NA, index=s.index, dtype="Float64")
        negative = text.isin(["", "nan", "none", "<na>", "nat"])
        # Most manual QC columns are issue-present flags: yes/present/poor -> 1.
        yes = text.isin(["1", "yes", "y", "true", "t", "present", "detected", "poor", "bad", "fail", "failed", "unstable", "needs", "needed"])
        no = text.isin(["0", "no", "n", "false", "f", "absent", "not present", "good", "ok", "okay", "pass", "passed", "valid", "normal"])
        out.loc[yes] = 1.0
        out.loc[no] = 0.0
        if num.notna().any():
            out.loc[num.notna()] = (num.loc[num.notna()] > 0).astype(float)
        # Completion/validity columns are inverted: completed/valid/pass is good.
        if any(t in n for t in ["completed_as_instructed", "task_completed", "task_validity", "validity"]):
            out.loc[yes] = 0.0
            out.loc[no] = 1.0
            if num.notna().any():
                out.loc[num.notna()] = (num.loc[num.notna()] <= 0).astype(float)
        # Parsing columns are direct flags: needs parsing = bad.
        if any(t in n for t in ["needs_parsing", "parsing_needed"]):
            out.loc[yes] = 1.0
            out.loc[no] = 0.0
        out.loc[negative] = pd.NA
        return out

    def _manual_qc_columns_from_mapping(self, df: pd.DataFrame) -> list[tuple[str, str, str]]:
        """Return (analysis column, family, original role) for accepted manual QC metadata flags.

        This intentionally uses assigned Metadata Mapping roles as the source of
        truth.  Feature columns, clinical scores, permissions, and generic media
        metadata are never admitted just because their names contain words like
        ``speech`` or ``audio``.
        """
        rows: list[tuple[str, str, str]] = []
        used: set[str] = set()
        mapping = getattr(self, "metadata_mapping_df", pd.DataFrame())
        if mapping is not None and not mapping.empty:
            for _, r in mapping.iterrows():
                source = str(r.get("column", "")).strip()
                role = str(r.get("role", "")).strip()
                canonical = str(r.get("canonical_field", "")).strip()
                family = self._manual_qc_family_for_role(role, source)
                if not self._manual_qc_family_allowed(family):
                    continue
                candidates = [source, f"metadata__{source}", canonical, f"metadata__{canonical}"]
                if canonical:
                    candidates += [c for c in df.columns if str(c).startswith(f"metadata__{canonical}_")]
                col = self._find_analysis_column(df, *[c for c in candidates if c])
                if col and col not in used:
                    rows.append((col, family or "Manual QC", role))
                    used.add(col)
        if rows:
            return rows
        # Legacy fallback only if there is no accepted metadata mapping at all.
        if mapping is None or mapping.empty:
            for c in df.columns:
                family = self._manual_qc_family_for_role("", str(c))
                if self._manual_qc_family_allowed(family) and str(c) not in used:
                    rows.append((str(c), family or "Manual QC", "inferred_from_header_no_mapping"))
                    used.add(str(c))
        return rows

    def _manual_qc_dataframe(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        if feature_df is None or feature_df.empty:
            return pd.DataFrame()
        id_cols = [c for c in ["record_key", "file_name", "filename", "source_file", "subject_id", "participant_id", "task", "task_name"] if c in feature_df.columns]
        out = feature_df[id_cols].copy() if id_cols else pd.DataFrame(index=feature_df.index)
        col_rows = self._manual_qc_columns_from_mapping(feature_df)
        used_names = set(str(c) for c in out.columns)
        family_lookup = {}
        family_prefix = {
            "Manual acquisition QC": "manual_acquisition_qc",
            "Manual audio QC": "manual_audio_qc",
            "Manual video QC": "manual_video_qc",
            "Manual face/visibility QC": "manual_face_visibility_qc",
        }
        for col, family, role in col_rows:
            base_name = normalize_name(col).replace("metadata__", "") or "flag"
            metric = f"{family_prefix.get(family, 'manual_qc')}__{base_name}"
            if metric in used_names:
                metric = self._unique_column_name(metric, used_names)
            used_names.add(metric)
            out[metric] = self._manual_qc_badness(feature_df[col], col).values
            family_lookup[metric] = family
        self._current_manual_qc_family_lookup = family_lookup
        return out

    def _automated_qc_dataframe(self, row_mask: pd.Series | None = None) -> pd.DataFrame | None:
        q = self.qc_df.copy() if getattr(self, "qc_df", None) is not None else None
        if q is None or q.empty:
            return q
        if row_mask is not None and len(q) == len(row_mask):
            q = q.loc[row_mask.values].copy()
        return q

    def _limit_qc_table_for_speed(self, qc_df: pd.DataFrame | None, source_mode: str) -> pd.DataFrame | None:
        """Limit expensive QC metrics while preserving automated artifact-family coverage.

        The prior speed guard selected only the highest-variance QC variables.
        That made automated heatmaps faster, but it could silently drop low-variance
        artifact families, which is why the automated family heatmap sometimes
        showed only four of the six acoustic QC families.  This guard is now
        stratified by QC family: keep a small quota from every detected family,
        then fill remaining slots by information content.
        """
        if qc_df is None or qc_df.empty:
            return qc_df
        q = qc_df.copy()
        id_cols = [c for c in ["record_key", "file_name", "filename", "source_file", "subject_id", "participant_id", "task", "task_name"] if c in q.columns]
        numeric: list[str] = []
        for c in q.columns:
            if c in id_cols:
                continue
            x = pd.to_numeric(q[c], errors="coerce")
            if x.notna().any() and x.nunique(dropna=True) > 1:
                numeric.append(c)
        if not numeric:
            return q.loc[:, id_cols].copy() if id_cols else q.iloc[:, 0:0].copy()

        max_metrics = 60 if source_mode == "Manual QC" else 48
        selected = self._selected_qc_metric() if hasattr(self, "qc_metric_combo") else None
        if len(numeric) <= max_metrics:
            return q

        scored = []
        for c in numeric:
            x = pd.to_numeric(q[c], errors="coerce")
            family = qc_family_from_name(c)
            scored.append((c, family, float(x.var(skipna=True) or 0.0), int(x.notna().sum()), int(x.nunique(dropna=True))))

        keep: list[str] = []
        if source_mode == "Automated QC":
            family_order = [
                "Additive interference",
                "Gain / level dynamics",
                "Reverberation / echo",
                "Channel / device / platform",
                "Nonlinear distortion",
                "Temporal discontinuities",
            ]
            per_family = max(3, min(8, max_metrics // max(1, len(family_order))))
            for fam in family_order:
                fam_rows = [r for r in scored if r[1] == fam]
                fam_rows.sort(key=lambda t: (t[2], t[3], t[4]), reverse=True)
                for c, *_ in fam_rows[:per_family]:
                    if c not in keep:
                        keep.append(c)

        # Fill remaining capacity from globally informative variables.
        for c, *_ in sorted(scored, key=lambda t: (t[2], t[3], t[4]), reverse=True):
            if len(keep) >= max_metrics:
                break
            if c not in keep:
                keep.append(c)
        if selected and selected in numeric and selected not in keep:
            keep.append(selected)
        return q.loc[:, id_cols + [c for c in keep if c not in id_cols]].copy()

    def _qc_source_variable_table(self, qc_df: pd.DataFrame | None) -> pd.DataFrame:
        source = self._selected_qc_source_mode()
        modality = self._project_modality_mode()
        if qc_df is None or qc_df.empty:
            return pd.DataFrame([{"qc_source": source, "project_modality": modality, "status": "no_usable_qc_variables", "n_variables": 0}])
        catalog = qc_metric_catalog(qc_df)
        if catalog.empty:
            return pd.DataFrame([{"qc_source": source, "project_modality": modality, "status": "no_numeric_or_binary_qc_variables", "n_variables": 0}])
        out = catalog[["qc_variable", "artifact_family", "n_valid", "missing_fraction", "n_unique", "interpretation"]].copy()
        out.insert(0, "qc_source", source)
        out.insert(1, "project_modality", modality)
        return out

    # Backward-compatible name used by the existing table update path.
    def _qc_source_table(self, qc_df: pd.DataFrame | None) -> pd.DataFrame:
        return self._qc_source_variable_table(qc_df)

    def _qc_scope_tables(self) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
        feature_df = self._active_analysis_table() if hasattr(self, "_active_analysis_table") else (self.analysis_df if self.analysis_df is not None else self.feature_df)
        if feature_df is None:
            return pd.DataFrame(), pd.DataFrame(), f"{self._selected_qc_source_mode()} | no feature table"
        base = feature_df.copy()
        row_mask = pd.Series(True, index=base.index)
        task_col = self._task_col(base) if hasattr(self, "_task_col") else None
        task_value = self.qc_task_combo.currentText() if hasattr(self, "qc_task_combo") else "All tasks"
        parts = []
        if task_col and task_col in base.columns and task_value not in {"", "All tasks", "All tasks / not available"}:
            row_mask = base[task_col].astype(str).eq(str(task_value))
            f = base.loc[row_mask].copy()
            parts.append(f"Task = {task_value}")
        else:
            f = base.copy()
            parts.append("All tasks")
        source = self._selected_qc_source_mode()
        if source == "Manual QC":
            q = self._manual_qc_dataframe(f)
            if q is not None and not q.empty and hasattr(self, "log"):
                manual_cols = [c for c in q.columns if str(c).startswith("manual_")]
                self.log(f"Manual QC source: {len(manual_cols)} assigned metadata QC flags after modality filter ({self._project_modality_mode()}).")
        else:
            q = self._automated_qc_dataframe(row_mask)
        q = self._limit_qc_table_for_speed(q, source)
        parts.insert(0, source)
        parts.append(f"Project modality = {self._project_modality_mode()}")
        return f, q, " | ".join(parts)

    def _refresh_qc_task_combo(self, df: pd.DataFrame) -> None:
        if not hasattr(self, "qc_task_combo"):
            return
        task_col = self._task_col(df) if hasattr(self, "_task_col") else None
        current = self.qc_task_combo.currentText()
        self.qc_task_combo.blockSignals(True)
        self.qc_task_combo.clear()
        if task_col and task_col in df.columns:
            values = sorted([str(x) for x in df[task_col].dropna().unique().tolist() if str(x).strip()])
            self.qc_task_combo.addItem("All tasks")
            for v in values[:500]:
                self.qc_task_combo.addItem(v)
        else:
            self.qc_task_combo.addItem("All tasks / not available")
        ix = self.qc_task_combo.findText(current)
        if ix >= 0:
            self.qc_task_combo.setCurrentIndex(ix)
        self.qc_task_combo.blockSignals(False)

    def _qc_feature_cols(self, df: pd.DataFrame) -> list[str]:
        if df is None or df.empty:
            return []
        if getattr(self, "mapping_df", None) is not None and not self.mapping_df.empty:
            roles = role_lists(self.mapping_df)
            cols = [c for c in roles.get("Feature", []) if c in df.columns]
        else:
            cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        numeric = [c for c in cols if pd.to_numeric(df[c], errors="coerce").notna().any()]
        max_features = 70
        selected = self._selected_qc_feature() if hasattr(self, "qc_feature_combo") else None
        if len(numeric) > max_features:
            score = []
            for c in numeric:
                x = pd.to_numeric(df[c], errors="coerce")
                score.append((c, float(x.var(skipna=True) or 0.0), int(x.notna().sum()), int(x.nunique(dropna=True))))
            numeric = [c for c, *_ in sorted(score, key=lambda t: (t[1], t[2], t[3]), reverse=True)[:max_features]]
            if selected and selected in cols and selected not in numeric:
                numeric.append(selected)
        return numeric

    def _refresh_qc_source_combo(self) -> None:
        if not hasattr(self, "qc_source_combo"):
            return
        current = self.qc_source_combo.currentText()
        self.qc_source_combo.blockSignals(True)
        self.qc_source_combo.clear()
        for item in ["Manual QC", "Automated QC"]:
            self.qc_source_combo.addItem(item)
        ix = self.qc_source_combo.findText(current)
        if ix >= 0:
            self.qc_source_combo.setCurrentIndex(ix)
        self.qc_source_combo.blockSignals(False)

    # Compatibility alias for older call sites/tests.
    def _refresh_qc_framework_combo(self) -> None:
        self._refresh_qc_source_combo()


    def _refresh_qc_selector_combos(self) -> None:
        """Populate Feature and QC metric selectors from current local QC scope."""
        if not hasattr(self, "qc_feature_combo") or not hasattr(self, "qc_metric_combo"):
            return
        feature_df, qc_df, _scope_label = self._qc_scope_tables()
        feature_cols = self._qc_feature_cols(feature_df)
        current_feature = self.qc_feature_combo.currentText()
        current_metric = self.qc_metric_combo.currentText()

        self.qc_feature_combo.blockSignals(True)
        self.qc_feature_combo.clear()
        for c in feature_cols:
            self.qc_feature_combo.addItem(str(c))
        ix = self.qc_feature_combo.findText(current_feature)
        if ix >= 0:
            self.qc_feature_combo.setCurrentIndex(ix)
        self.qc_feature_combo.blockSignals(False)

        catalog = qc_metric_catalog(qc_df)
        self.qc_metric_combo.blockSignals(True)
        self.qc_metric_combo.clear()
        if catalog is not None and not catalog.empty and "qc_variable" in catalog.columns:
            for q in catalog["qc_variable"].astype(str).tolist():
                self.qc_metric_combo.addItem(q)
        ix = self.qc_metric_combo.findText(current_metric)
        if ix >= 0:
            self.qc_metric_combo.setCurrentIndex(ix)
        self.qc_metric_combo.blockSignals(False)

    def generate_qc_scope_plots(self) -> None:
        feature_df, qc_df, scope_label = self._qc_scope_tables()
        self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
        feature_cols = self._qc_feature_cols(feature_df)
        safe_scope = re.sub(r"[^A-Za-z0-9_.-]+", "_", scope_label).strip("_")[:90] or "qc_scope"
        if not hasattr(self, "plot_paths"):
            self.plot_paths = {}
        catalog = qc_metric_catalog(qc_df)
        family = qc_family_burden_summary(qc_df)
        qc_corr = feature_qc_correlations(feature_df, qc_df, feature_cols)
        family_assoc = feature_qc_family_association(qc_corr)
        missing_assoc = qc_missingness_associations(feature_df, qc_df, feature_cols)
        row_burden = qc_row_burden_summary(qc_df)

        self.plot_paths["qc_artifact_model"] = str(plot_qc_artifact_model(plots_dir / f"qc_artifact_model_{safe_scope}.png", self._selected_qc_source_mode()))
        self.plot_paths["qc_family_burden"] = str(plot_qc_family_burden(family, plots_dir / f"qc_family_burden_{safe_scope}.png"))
        self.plot_paths["qc_metric_distributions"] = str(plot_qc_metric_distributions(qc_df, catalog, plots_dir / f"qc_metric_distributions_{safe_scope}.png", max_metrics=18))
        self.plot_paths["qc_feature_association_heatmap"] = str(plot_qc_feature_association_heatmap(family_assoc, plots_dir / f"qc_feature_association_heatmap_{safe_scope}.png", top_features=80))
        self.plot_paths["qc_top_feature_associations"] = str(plot_qc_top_feature_associations(qc_corr, plots_dir / f"qc_top_feature_associations_{safe_scope}.png", top_n=35))
        self.plot_paths["qc_missingness_associations"] = str(plot_qc_missingness_associations(missing_assoc, plots_dir / f"qc_missingness_associations_{safe_scope}.png", top_n=35))
        self.plot_paths["qc_row_burden"] = str(plot_qc_row_burden(row_burden, plots_dir / f"qc_row_burden_{safe_scope}.png", top_n=50))
        if hasattr(self, "qc_plot_caption"):
            n_qc = len(catalog) if catalog is not None else 0
            self.qc_plot_caption.setText(
                f"Current QC scope: {scope_label}. "
                f"Usable QC flags/metrics in this source: {n_qc}. Automated QC tables are used only when Automated QC is selected."
            )

    def regenerate_qc_scope_plots(self) -> None:
        self.generate_qc_scope_plots()
        self.preview_qc_plot(self.qc_plot_combo.currentData() if hasattr(self, "qc_plot_combo") else "qc_artifact_model", regenerate=False)

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

        active_for_qc_scope = self._active_analysis_table() if hasattr(self, "_active_analysis_table") else (self.analysis_df if self.analysis_df is not None else self.feature_df)
        self._refresh_qc_task_combo(active_for_qc_scope)
        self._refresh_qc_source_combo()
        self._fill_table(self.qc_summary_table, summary)
        self._fill_table(self.qc_catalog_table, outputs.get("qc_metric_catalog", pd.DataFrame()))
        self._fill_table(self.qc_family_table, outputs.get("qc_family_burden_summary", pd.DataFrame()))
        self._fill_table(self.qc_assoc_table, outputs.get("feature_qc_spearman_correlation", pd.DataFrame()))
        self._fill_table(self.qc_family_assoc_table, outputs.get("feature_qc_family_association", pd.DataFrame()))
        self._fill_table(self.qc_missing_table, outputs.get("qc_missingness_associations", pd.DataFrame()))
        self._fill_table(self.qc_outlier_table, outputs.get("qc_outlier_associations", pd.DataFrame()))
        self._fill_table(self.qc_row_table, outputs.get("qc_row_burden_summary", pd.DataFrame()))
        if hasattr(self, "qc_source_table"):
            _f_scope, _q_scope, _scope_label = self._qc_scope_tables()
            self._fill_table(self.qc_source_table, self._qc_source_table(_q_scope))

        self._refresh_qc_selector_combos()
        self.qc_note.setText(
            "QC Integration generated. Choose Manual QC to use modality-aware metadata flags, or Automated QC to use the uploaded QC table. Associations are descriptive screening signals, not automatic exclusion rules."
        )

    def _selected_qc_feature(self) -> str | None:
        if hasattr(self, "qc_feature_combo") and self.qc_feature_combo.count() > 0:
            val = self.qc_feature_combo.currentText().strip()
            return val or None
        return None

    def _selected_qc_metric(self) -> str | None:
        if hasattr(self, "qc_metric_combo") and self.qc_metric_combo.count() > 0:
            val = self.qc_metric_combo.currentText().strip()
            return val or None
        return None

    def generate_selected_qc_scatter(self) -> None:
        feature = self._selected_qc_feature()
        metric = self._selected_qc_metric()
        if not feature or not metric:
            return
        try:
            active_df, qc_df, scope_label = self._qc_scope_tables()
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            safe_scope = re.sub(r"[^A-Za-z0-9_.-]+", "_", scope_label).strip("_")[:90] or "qc_scope"
            path = plot_selected_feature_qc_scatter(active_df, qc_df, feature, metric, plots_dir / f"selected_feature_qc_scatter_{safe_scope}.png")
            if not hasattr(self, "plot_paths"):
                self.plot_paths = {}
            self.plot_paths["selected_feature_qc_scatter"] = str(path)
        except Exception as exc:
            QMessageBox.warning(self, "Selected QC scatter failed", str(exc))

    def preview_qc_plot(self, key: str, regenerate: bool = True) -> None:
        if key == "selected_feature_qc_scatter":
            self.generate_selected_qc_scatter()
        if key != "selected_feature_qc_scatter" and regenerate:
            self.generate_qc_scope_plots()
        elif not hasattr(self, "plot_paths") or key not in self.plot_paths or not Path(self.plot_paths.get(key, "")).exists():
            self.generate_qc_scope_plots()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths:
            QMessageBox.information(self, "Plot unavailable", "Run Feature Analysis first, or this QC plot could not be generated for the current dataset.")
            return
        path = Path(self.plot_paths[key])
        if not path.exists():
            QMessageBox.information(self, "Plot unavailable", f"Plot file not found:\n{path}")
            return
        self.current_qc_plot = path
        self.update_qc_interpretation(key)
        self._display_plot_image(self.qc_plot_preview, path)

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
        self.relationship_plot_combo.addItem("Correlation structure", "relationship_correlation_heatmap")
        self.relationship_plot_combo.addItem("Redundancy triage", "relationship_redundant_pairs")
        self.relationship_plot_combo.addItem("Feature-family structure", "relationship_family_matrix")
        self.relationship_plot_combo.addItem("Dimensionality profile", "relationship_dimensionality_profile")
        self.relationship_plot_combo.addItem("Recording similarity map", "relationship_pca_scores")
        self.relationship_plot_combo.addItem("Selected feature links", "selected_feature_correlations")
        plot_header.addWidget(self.relationship_plot_combo, 1)

        show_btn = QPushButton("Show")
        show_btn.setProperty("secondary", True)
        show_btn.clicked.connect(lambda: self.preview_relationship_plot(self.relationship_plot_combo.currentData()))
        plot_header.addWidget(show_btn)

        regen = QPushButton("Regenerate")
        regen.clicked.connect(self.regenerate_relationships)
        plot_header.addWidget(regen)

        plot_header.addStretch(1)
        open_btn = QPushButton("Open current plot")
        open_btn.setProperty("secondary", True)
        open_btn.clicked.connect(self.open_current_relationship_plot)
        plot_header.addWidget(open_btn)
        plot_panel_layout.addLayout(plot_header)

        self.relationship_plot_caption = QLabel(
            "Feature Relationships is a feature-structure layer. It asks which features move together, which are redundant, whether families form coherent blocks, and which selected feature links deserve follow-up. Task focus rebuilds the relationship tables and plots for the selected task."
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
        self.relationship_task_combo.currentIndexChanged.connect(lambda _=0: self.regenerate_relationships(silent=True))
        selectors.addWidget(self.relationship_task_combo)
        plot_panel_layout.addLayout(selectors)

        self.relationship_plot_preview = QLabel("Run Feature Analysis, then choose one relationship plot. Task focus rebuilds this menu from the current task scope.")
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
        side_panel.setStyleSheet(f"QFrame {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:14px; }}")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        side_title = QLabel("Relationship snapshot")
        side_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        side_layout.addWidget(side_title)
        side_note = QLabel("Compact feature-structure metrics for the current task scope. Use this to decide whether to inspect redundancy, family coupling, PCA, or a selected feature.")
        side_note.setWordWrap(True)
        side_note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent; font-size:12px;")
        side_layout.addWidget(side_note)
        self.relationship_metric_grid = QGridLayout()
        self.relationship_metric_grid.setHorizontalSpacing(8)
        self.relationship_metric_grid.setVerticalSpacing(8)
        side_layout.addLayout(self.relationship_metric_grid)
        side_layout.addStretch(1)
        side_panel.setMinimumWidth(280)
        side_panel.setMaximumWidth(360)
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
        scope = outputs.get("feature_relationship_scope", pd.DataFrame())
        task_scope = scope["task_scope"].iloc[0] if scope is not None and not scope.empty and "task_scope" in scope.columns else "All tasks"
        rows_scope = scope["rows_in_scope"].iloc[0] if scope is not None and not scope.empty and "rows_in_scope" in scope.columns else "-"
        tiles = [
            ("Task scope", task_scope, "current relationship scope"),
            ("Rows", rows_scope, "recordings in scope"),
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

    def _relationship_current_task(self) -> str | None:
        combo = getattr(self, "relationship_task_combo", None)
        if combo is None or combo.count() == 0:
            return None
        txt = combo.currentText().strip()
        if not txt or txt.startswith("All tasks"):
            return None
        return txt

    def _relationship_scoped_frame(self) -> pd.DataFrame:
        df = self._active_analysis_table().copy()
        task = self._relationship_current_task()
        if task and not df.empty:
            task_col = self._task_col(df)
            if task_col and task_col in df.columns:
                mask = df[task_col].astype("string").fillna("").str.strip().eq(str(task).strip())
                df = df.loc[mask].copy()
        return df

    def _relationship_feature_cols(self, df: pd.DataFrame) -> list[str]:
        if df is None or df.empty:
            return []
        try:
            mapping = self.collect_mapping_from_table()
        except Exception:
            mapping = getattr(self, "mapping_df", pd.DataFrame())
        roles = role_lists(mapping) if mapping is not None and not mapping.empty else {}
        mapped = [c for c in roles.get("Feature", []) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if mapped:
            return mapped
        base = self.feature_df if getattr(self, "feature_df", None) is not None else df
        numeric = [c for c in base.columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        excluded = {self._task_col(df), self._subject_col(df), self._session_col(df), self._iteration_col(df), self._date_col(df)}
        return [c for c in numeric if c not in excluded][:220]

    def _build_relationship_outputs(self) -> tuple[dict[str, pd.DataFrame], list[str]]:
        df = self._relationship_scoped_frame()
        feature_cols = self._relationship_feature_cols(df)
        registry = getattr(self, "registry_df", pd.DataFrame())
        outputs = {
            "feature_relationship_summary": feature_relationship_summary(df, feature_cols, registry),
            "feature_correlation_long": feature_correlation_long_table(df, feature_cols, max_features=180),
        }
        outputs["feature_redundant_pairs"] = redundant_feature_pairs(outputs["feature_correlation_long"], registry)
        outputs["feature_relationship_modules"] = feature_relationship_modules(outputs["feature_correlation_long"], registry)
        outputs["feature_family_correlation_matrix"] = feature_family_correlation_matrix(outputs["feature_correlation_long"], registry)
        outputs["feature_pca_summary"] = feature_pca_summary(df, feature_cols)
        outputs["feature_pca_loadings"] = feature_pca_loadings(df, feature_cols, registry)
        outputs["feature_pca_scores"] = feature_pca_scores(df, feature_cols)
        task = self._relationship_current_task() or "All tasks"
        outputs["feature_relationship_scope"] = pd.DataFrame([{
            "task_scope": task,
            "rows_in_scope": int(len(df)),
            "mapped_numeric_features": int(len(feature_cols)),
        }])
        return outputs, feature_cols

    def _generate_relationship_plots(self, outputs: dict[str, pd.DataFrame], plots_dir: Path) -> dict[str, str]:
        paths: dict[str, str] = {}
        jobs = [
            ("relationship_correlation_heatmap", "Correlation structure", lambda p: plot_relationship_correlation_heatmap(outputs.get("feature_correlation_long", pd.DataFrame()), p, top_n=55)),
            ("relationship_redundant_pairs", "Redundancy triage", lambda p: plot_relationship_redundant_pairs(outputs.get("feature_redundant_pairs", pd.DataFrame()), p, top_n=28)),
            ("relationship_family_matrix", "Feature-family structure", lambda p: plot_relationship_family_matrix(outputs.get("feature_family_correlation_matrix", pd.DataFrame()), p)),
            ("relationship_dimensionality_profile", "Dimensionality profile", lambda p: plot_relationship_dimensionality_profile(outputs.get("feature_pca_summary", pd.DataFrame()), outputs.get("feature_pca_loadings", pd.DataFrame()), p)),
            ("relationship_pca_scores", "Recording similarity map", lambda p: plot_relationship_pca_scores(outputs.get("feature_pca_scores", pd.DataFrame()), self._relationship_scoped_frame(), p)),
        ]
        task = self._relationship_current_task()
        suffix = f"__task__{self._safe_task_slug(task)}" if task else ""
        for key, title, fn in jobs:
            path = plots_dir / f"{key}{suffix}.png"
            try:
                paths[key] = str(fn(path))
            except Exception as exc:
                self.log(f"WARN | Relationship plot {key} failed and was replaced by placeholder: {exc}")
                self.log(traceback.format_exc())
                paths[key] = str(self._write_overview_placeholder_plot(path, title, str(exc)))
        return paths

    def regenerate_relationships(self, silent: bool = False) -> None:
        try:
            if self.feature_df is None:
                if not silent:
                    QMessageBox.information(self, "Load features first", "Load and map feature tables, then run Feature Analysis before reviewing relationships.")
                return
            if not self.output_edit.text().strip():
                if not silent:
                    QMessageBox.warning(self, "Missing output folder", "Please select an output folder before generating relationship plots.")
                return
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            outputs, feature_cols = self._build_relationship_outputs()
            merged = dict(getattr(self, "outputs", {}) or {})
            merged.update(outputs)
            self.outputs = merged
            try:
                self._write_outputs(outputs, tables_dir)
            except Exception as exc:
                self.log(f"WARN | Could not write relationship tables: {exc}")
            new_paths = self._generate_relationship_plots(outputs, plots_dir)
            if not hasattr(self, "plot_paths") or not isinstance(self.plot_paths, dict):
                self.plot_paths = {}
            self.plot_paths.update(new_paths)
            self.update_relationships_dashboard(outputs)
            if not silent:
                task = self._relationship_current_task() or "All tasks"
                self.log(f"Feature Relationships regenerated for task scope: {task} | rows={outputs.get('feature_relationship_scope', pd.DataFrame()).get('rows_in_scope', pd.Series(['-'])).iloc[0]} | features={len(feature_cols)}")
                self.preview_relationship_plot(self.relationship_plot_combo.currentData())
        except Exception as exc:
            self.log_error("Feature Relationships regeneration failed", exc)
            if not silent:
                QMessageBox.critical(self, "Feature Relationships failed", str(exc))

    def generate_selected_feature_correlations(self) -> None:
        feature = self._selected_relationship_feature()
        if not feature:
            return
        try:
            self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
            if not getattr(self, "outputs", None) or "feature_correlation_long" not in self.outputs:
                self.regenerate_relationships(silent=True)
            task = self._relationship_current_task()
            suffix = f"__task__{self._safe_task_slug(task)}" if task else ""
            safe_feature = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(feature))[:80]
            path = plot_selected_feature_correlations(self.outputs.get("feature_correlation_long", pd.DataFrame()), feature, plots_dir / f"selected_feature_correlations__{safe_feature}{suffix}.png")
            if not hasattr(self, "plot_paths"):
                self.plot_paths = {}
            self.plot_paths["selected_feature_correlations"] = str(path)
        except Exception as exc:
            self.log(f"WARN | Selected feature relationship plot failed: {exc}")
            self.log(traceback.format_exc())

    def preview_relationship_plot(self, key: str) -> None:
        if key == "selected_feature_correlations":
            self.generate_selected_feature_correlations()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths or not Path(self.plot_paths.get(key, "")).exists():
            self.regenerate_relationships(silent=True)
            if key == "selected_feature_correlations":
                self.generate_selected_feature_correlations()
        if not hasattr(self, "plot_paths") or key not in self.plot_paths:
            QMessageBox.information(self, "Plot unavailable", "Run Feature Analysis first, or this relationship plot could not be generated for the current dataset.")
            return
        path = Path(self.plot_paths[key])
        if not path.exists():
            QMessageBox.information(self, "Plot unavailable", f"Plot file not found:\n{path}")
            return
        self.current_relationship_plot = path
        self.update_relationship_interpretation(key)
        self._display_plot_image(self.relationship_plot_preview, path)

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
            "relationship_dimensionality_profile": (
                "<b>What it shows</b><br>How concentrated the feature space is, and which features drive the first few unsupervised axes.<br><br>"
                "<b>Concerning pattern</b><br>A dominant PC1 plus one narrow feature family in the loadings suggests a global technical/task axis or redundant feature block.<br><br>"
                "<b>Do not overinterpret</b><br>PCA is unsupervised. It is not a diagnostic classifier or biomarker result.<br><br>"
                "<b>Next check</b><br>Compare dominant loading features with QC Integration, Missingness, and Distribution/Outliers."
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
            "Essential task-level readiness checks: coverage, clinical balance, and feature completeness."
        )

        self.task_note = QLabel("Run Feature Analysis with task metadata available. Task Review intentionally keeps only the essential plots needed to decide which tasks are clinically usable downstream.")
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
        self.task_review_combo.setMinimumWidth(300)
        self.task_review_combo.addItem("Task readiness summary", "task_readiness_overview")
        self.task_review_combo.addItem("Clinical balance by task", "task_clinical_balance")
        self.task_review_combo.addItem("Subject coverage by task", "task_subject_coverage")
        self.task_review_combo.addItem("Feature completeness by task", "task_feature_completeness")
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

        self.task_review_caption = QLabel("Task Review answers four practical questions: which tasks have enough recordings, which have enough subjects, whether clinical groups are balanced, and whether feature extraction is complete enough to interpret.")
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
        self.clinical_value_combo.currentIndexChanged.connect(lambda _=0: self.preview_task_review_plot(self.task_review_combo.currentData() if hasattr(self, 'task_review_combo') else 'task_readiness_overview'))
        selectors.addWidget(self.clinical_value_combo, 1)
        selectors.addStretch(1)
        plot_panel_layout.addLayout(selectors)

        self.task_review_preview = QLabel("Run Feature Analysis, then choose a task review plot.")
        self.task_review_preview.setAlignment(Qt.AlignCenter)
        self.task_review_preview.setMinimumHeight(440)
        self.task_review_preview.setScaledContents(False)
        self.task_review_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.task_review_preview.setStyleSheet(f"QLabel {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; color:{MUTED}; padding:16px; }}")
        plot_panel_layout.addWidget(self.task_review_preview, 1)

        self.task_review_interpretation = QLabel("Select a task plot to see interpretation guidance.")
        self.task_review_interpretation.setWordWrap(True)
        self.task_review_interpretation.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.task_review_interpretation.setStyleSheet(f"QLabel {{ background:#FFFFFF; color:{INK}; border:1px solid {LINE}; border-radius:10px; padding:12px; font-size:12px; line-height:140%; }}")
        plot_panel_layout.addWidget(self.task_review_interpretation)
        task_split.addWidget(plot_panel, 1)

        side_panel = QFrame()
        side_panel.setStyleSheet(f"QFrame {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:14px; }}")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        side_title = QLabel("Task snapshot")
        side_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        side_layout.addWidget(side_title)
        side_note = QLabel("Compact task-readiness metrics for clinical interpretation and downstream modeling.")
        side_note.setWordWrap(True)
        side_note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent; font-size:12px;")
        side_layout.addWidget(side_note)
        self.task_metric_grid = QGridLayout()
        self.task_metric_grid.setHorizontalSpacing(8)
        self.task_metric_grid.setVerticalSpacing(8)
        side_layout.addLayout(self.task_metric_grid)
        side_layout.addStretch(1)
        side_panel.setMinimumWidth(280)
        side_panel.setMaximumWidth(360)
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
        tabs.addTab(self.task_summary_table, "Readiness summary")
        tabs.addTab(self.task_subject_table, "Subject coverage")
        tabs.addTab(self.task_diagnosis_table, "Clinical balance")
        tabs.addTab(self.task_feature_support_table, "Feature completeness")
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
        self.longitudinal_view_combo.setMinimumWidth(300)
        # v0.117: expose one safe longitudinal view only. Later trajectory plots will be added one at a time.
        self.longitudinal_view_combo.addItem("Repeated-record cohort summary", "longitudinal_readiness")
        self.longitudinal_view_combo.addItem("Selected subject-task visit timeline", "longitudinal_visit_timeline")
        self.longitudinal_view_combo.addItem("Selected subject-task feature change audit", "longitudinal_feature_family_trajectory")
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

        self.longitudinal_caption = QLabel("Step 1: quantify the repeated-record cohort. Step 2: verify one subject-task visit timeline. Step 3: audit individual feature changes within the same task; direction is numeric only unless a feature registry defines clinical direction.")
        self.longitudinal_caption.setWordWrap(True)
        self.longitudinal_caption.setStyleSheet(f"color:{MUTED}; background:#FFFFFF; border:1px solid {LINE}; border-radius:8px; padding:9px;")
        plot_panel_layout.addWidget(self.longitudinal_caption)

        selectors = QHBoxLayout()
        selectors.addWidget(QLabel("Task focus:"))
        self.longitudinal_task_combo = QComboBox()
        self.longitudinal_task_combo.setMinimumWidth(260)
        self.longitudinal_task_combo.addItem("All tasks", None)
        self.longitudinal_task_combo.currentIndexChanged.connect(lambda _=0: self.update_longitudinal_dashboard(getattr(self, "outputs", {})))
        selectors.addWidget(self.longitudinal_task_combo, 1)
        selectors.addWidget(QLabel("Subject focus:"))
        self.subject_focus_combo = QComboBox()
        self.subject_focus_combo.setMinimumWidth(260)
        self.subject_focus_combo.addItem("All subjects / not available")
        self.subject_focus_combo.currentIndexChanged.connect(lambda _=0: self._on_longitudinal_subject_changed())
        selectors.addWidget(self.subject_focus_combo, 1)
        selectors.addWidget(QLabel("Feature family:"))
        self.longitudinal_family_combo = QComboBox()
        self.longitudinal_family_combo.setMinimumWidth(260)
        self.longitudinal_family_combo.addItem("All mapped features", "__all__")
        self.longitudinal_family_combo.currentIndexChanged.connect(lambda _=0: self.preview_longitudinal_plot(self.longitudinal_view_combo.currentData() if hasattr(self, "longitudinal_view_combo") else "longitudinal_readiness"))
        selectors.addWidget(self.longitudinal_family_combo, 1)
        selectors.addStretch(1)
        plot_panel_layout.addLayout(selectors)

        self.longitudinal_preview = QLabel("Run Feature Analysis, then choose one longitudinal review plot.")
        self.longitudinal_preview.setAlignment(Qt.AlignCenter)
        self.longitudinal_preview.setScaledContents(False)
        self.longitudinal_preview.setMinimumHeight(460)
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
        side_panel.setStyleSheet(f"QFrame {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:14px; }}")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        side_title = QLabel("Longitudinal snapshot")
        side_title.setStyleSheet(f"font-weight:900; color:{NAVY}; font-size:14px; border:none; background:transparent;")
        side_layout.addWidget(side_title)
        side_note = QLabel("Compact repeated-record metrics. Later longitudinal plots will use only subject-task units with at least two recordings of the same task.")
        side_note.setWordWrap(True)
        side_note.setStyleSheet(f"color:{MUTED}; border:none; background:transparent; font-size:12px;")
        side_layout.addWidget(side_note)
        self.longitudinal_metric_grid = QGridLayout()
        self.longitudinal_metric_grid.setHorizontalSpacing(8)
        self.longitudinal_metric_grid.setVerticalSpacing(8)
        side_layout.addLayout(self.longitudinal_metric_grid)
        side_layout.addStretch(1)
        side_panel.setMinimumWidth(280)
        side_panel.setMaximumWidth(360)
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
        self.long_feature_change_table = self._simple_table()
        tabs.addTab(self.long_subject_table, "Repeated subject-task units")
        tabs.addTab(self.long_session_table, "Visit / session support")
        tabs.addTab(self.long_iteration_table, "Iteration support")
        tabs.addTab(self.long_date_table, "Date support")
        tabs.addTab(self.long_feature_change_table, "Feature change audit")
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
        try:
            hit = self._first_context_column(df, "task")
            if hit:
                return hit
        except Exception:
            pass
        return self._first_existing_col(df, ["metadata__task", "metadata__task_name", "task", "task_name", "parsed_task"])

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
        add("ALSFRS total", ["ALSFRS total score", "ALSFRS-R total score", "ALSFRS total", "ALSFRS-R total", "alsfrs_total", "alsfrsr_total", "metadata__alsfrs_total", "metadata__ALSFRS total score"])
        add("ALSFRS bulbar", ["ALSFRS bulbar", "ALSFRS-R bulbar", "ALSFRS bulbar score", "ALSFRS-R bulbar score", "bulbar", "bulbar_score", "alsfrs_bulbar", "alsfrsr_bulbar", "metadata__alsfrs_bulbar", "metadata__ALSFRS bulbar", "metadata__ALSFRS-R bulbar"])
        add("ALSBDI", ["ALSBDI", "ALSBDI score", "ALS Bulbar Dysfunction Index", "alsbdi", "alsbdi_score", "alsbdi_total", "metadata__alsbdi_total", "metadata__ALSBDI", "metadata__ALSBDI score"])
        add("Sex / gender", ["sex_or_gender", "sex", "Sex", "gender", "Gender", "metadata__sex_or_gender", "metadata__Sex", "metadata__sex"])
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
    def _task_review_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Mapped numeric feature columns for Task Review only."""
        if df is None or df.empty:
            return []
        cols: list[str] = []
        if getattr(self, "mapping_df", None) is not None and not self.mapping_df.empty:
            col_key = "column" if "column" in self.mapping_df.columns else "source_column" if "source_column" in self.mapping_df.columns else None
            if col_key and "role" in self.mapping_df.columns:
                cols = self.mapping_df.loc[
                    self.mapping_df["role"].astype(str).str.contains("Feature", case=False, na=False), col_key
                ].astype(str).tolist()
        if not cols:
            excluded = {self._task_col(df), self._subject_col(df), self._session_col(df), self._iteration_col(df), self._date_col(df)}
            cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
        clean = []
        for c in cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() >= 3:
                clean.append(c)
        return clean[:200]

    def _task_readiness_table(
        self,
        df: pd.DataFrame,
        task_col: str | None,
        subj_col: str | None,
        feature_cols: list[str],
        clinical_series: pd.Series | None,
    ) -> pd.DataFrame:
        if df is None or df.empty or not task_col or task_col not in df.columns:
            return pd.DataFrame()
        tmp = df.copy()
        tmp["__task__"] = tmp[task_col].astype("string").fillna("").astype(str)
        tmp = tmp[tmp["__task__"].str.strip().ne("")]
        rows = []
        aligned_clinical = None
        if clinical_series is not None and len(clinical_series) == len(df):
            aligned_clinical = pd.Series(clinical_series.to_numpy(), index=df.index).astype("string")
        for task, subdf in tmp.groupby("__task__", dropna=False):
            if not str(task).strip():
                continue
            n_rows = int(len(subdf))
            n_subjects = int(subdf[subj_col].astype(str).nunique()) if subj_col and subj_col in subdf.columns else 0
            mean_missing = float(subdf[feature_cols].isna().mean().mean()) if feature_cols else float("nan")
            median_available = float(subdf[feature_cols].notna().sum(axis=1).median()) if feature_cols else float("nan")
            n_clinical_groups = 0
            smallest_group_n = 0
            if aligned_clinical is not None:
                counts = aligned_clinical.loc[subdf.index].astype("string").fillna("missing").value_counts()
                counts = counts[counts.index.astype(str).str.len() > 0]
                n_clinical_groups = int(len(counts))
                smallest_group_n = int(counts.min()) if len(counts) else 0
            if n_rows < 5 or n_subjects < 3 or (pd.notna(mean_missing) and mean_missing >= .40):
                status = "review_before_modeling"
            elif n_rows < 10 or n_subjects < 5 or (pd.notna(mean_missing) and mean_missing >= .15):
                status = "usable_with_caution"
            else:
                status = "well_supported"
            rows.append({
                "task": task,
                "n_rows": n_rows,
                "n_subjects": n_subjects,
                "records_per_subject": round(n_rows / max(n_subjects, 1), 2) if n_subjects else pd.NA,
                "n_feature_columns": len(feature_cols),
                "median_available_features_per_record": median_available,
                "mean_feature_missingness": mean_missing,
                "clinical_groups_detected": n_clinical_groups,
                "smallest_clinical_group_n": smallest_group_n,
                "readiness_status": status,
            })
        return pd.DataFrame(rows).sort_values(["readiness_status", "n_rows"], ascending=[True, False]) if rows else pd.DataFrame()


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

        feature_cols = self._task_review_feature_cols(df)
        task_feature_support = self._task_readiness_table(df, task_col, subj_col, feature_cols, clinical_series)

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
        self._fill_table(self.task_summary_table, task_feature_support if not task_feature_support.empty else task_summary)
        self._fill_table(self.task_subject_table, task_subject)
        self._fill_table(self.task_diagnosis_table, task_diag)
        self._fill_table(self.task_feature_support_table, task_feature_support)
        if hasattr(self, "local_context_detection_table"):
            self._fill_table(self.local_context_detection_table, self._local_context_detection_table(df))
        if hasattr(self, "local_severity_preset_table"):
            self._fill_table(self.local_severity_preset_table, self._local_severity_preset_table())
        self.task_note.setText("Task Review generated from accepted task context. The menu is intentionally limited to essential task-readiness checks: coverage, clinical balance, and feature completeness.")
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
            aligned_series = self._align_context_series_to_frame(clinical_series, df) if hasattr(self, "_align_context_series_to_frame") else pd.Series(clinical_series.to_numpy() if len(clinical_series) == len(df) else pd.NA, index=df.index)
            df[label_col] = aligned_series
        feature_cols = self._task_review_feature_cols(df)
        support = getattr(self, "_task_feature_support_df", pd.DataFrame())
        if support is None or support.empty:
            support = self._task_readiness_table(df, task_col, subj_col, feature_cols, clinical_series)
        selected_task = self.task_focus_combo.currentText() if hasattr(self, "task_focus_combo") else "All tasks"
        if not hasattr(self, "plot_paths"):
            self.plot_paths = {}
        self.plot_paths["task_readiness_overview"] = str(plot_task_readiness_dashboard(df, task_col, subj_col, clinical_series, clinical_label, feature_cols, support, plots_dir / "task_readiness_overview.png"))
        self.plot_paths["task_clinical_balance"] = str(plot_task_clinical_balance_bars(df, task_col, clinical_series, clinical_label, selected_level, plots_dir / "task_clinical_balance.png"))
        self.plot_paths["task_subject_coverage"] = str(plot_task_subject_coverage_summary(df, task_col, subj_col, plots_dir / "task_subject_coverage.png"))
        self.plot_paths["task_feature_completeness"] = str(plot_task_feature_support(support, plots_dir / "task_feature_completeness.png"))
        # The older task feature profile and subject-by-task heatmap are intentionally not exposed in v0.111.
        # They were visually noisy and redundant with the compact coverage/completeness views.
        self.plot_paths["task_feature_profile"] = self.plot_paths.get("task_feature_completeness", "")
        self.plot_paths["task_subject_matrix"] = self.plot_paths.get("task_subject_coverage", "")
        # Legacy keys retained for compatibility with older saved sessions/tests.
        self.plot_paths["task_counts"] = str(plot_task_counts(df, task_col, plots_dir / "task_counts.png"))
        self.plot_paths["task_clinical_context"] = str(plot_task_clinical_context(df, task_col, clinical_series, clinical_label, plots_dir / "task_clinical_context.png"))
        self.plot_paths["task_clinical_filtered_counts"] = str(plot_task_clinical_filtered_counts(df, task_col, clinical_series, clinical_label, selected_level, plots_dir / "task_clinical_filtered_counts.png"))
        self.plot_paths["task_label_context"] = str(plot_task_label_context(df, task_col, label_col, plots_dir / "task_label_context.png"))
        self.plot_paths["task_feature_support"] = self.plot_paths["task_feature_completeness"]

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
            "task_readiness_overview": "Task readiness summary: compact view of recording count, subject breadth, feature completeness, and clinical balance. Use it to decide which tasks are safe to carry forward.",
            "task_clinical_balance": "Clinical balance by task: shows whether diagnosis/severity or another selected clinical context is unevenly distributed across tasks.",
            "task_subject_coverage": "Subject coverage by task: compact replacement for the unreadable subject-by-task heatmap. It separates broad cohort coverage from repeated recordings.",
            "task_feature_completeness": "Feature completeness by task: identifies tasks where feature extraction is incomplete enough to limit interpretation.",
            "task_feature_profile": "Deprecated in this simplified Task Review view; use Feature Relationships or Distributions for feature-level structure.",
            "task_subject_matrix": "Deprecated in this simplified Task Review view; use Subject coverage by task instead.",
            "task_counts": "Rows by task. Use this to decide whether pooled analysis is dominated by one task or whether task-specific review is required.",
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
        target_size = self.task_review_preview.contentsRect().size()
        if target_size.width() < 50 or target_size.height() < 50:
            target_size = QSize(900, 440)
        scaled = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.task_review_preview.clear()
        self.task_review_preview.setPixmap(scaled)
        self.task_review_preview.setToolTip(str(path))

    def open_current_task_review_plot(self) -> None:
        path = getattr(self, "current_task_review_plot", None)
        if not path:
            QMessageBox.information(self, "No current plot", "Preview a task plot first, then open the full-resolution file.")
            return
        self.open_file(Path(path))


    def _longitudinal_selected_task(self) -> str | None:
        combo = getattr(self, "longitudinal_task_combo", None)
        if combo is None:
            return None
        value = combo.currentData()
        return str(value) if value not in (None, "", "All tasks") else None

    def _longitudinal_selected_subject(self) -> str | None:
        combo = getattr(self, "subject_focus_combo", None)
        if combo is None:
            return None
        value = combo.currentData()
        if value in (None, "", "All subjects", "Select repeated subject"):
            return None
        return str(value)

    def _refresh_longitudinal_subject_combo(self, readiness: pd.DataFrame | None = None) -> None:
        combo = getattr(self, "subject_focus_combo", None)
        if combo is None:
            return
        current = combo.currentData()
        subjects: list[str] = []
        if isinstance(readiness, pd.DataFrame) and not readiness.empty and "subject_id" in readiness.columns and "n_records" in readiness.columns:
            d = readiness.copy()
            d["n_records"] = pd.to_numeric(d["n_records"], errors="coerce")
            d = d.loc[d["n_records"].fillna(0) >= 2]
            if not d.empty:
                d = d.sort_values(["n_records", "elapsed_days"], ascending=[False, False], na_position="last")
                subjects = [str(x) for x in d["subject_id"].dropna().astype(str).unique().tolist() if str(x).strip()]
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Select repeated subject" if subjects else "No repeated subjects", None)
        for subject in subjects[:500]:
            combo.addItem(subject, subject)
        if current is not None:
            ix = combo.findData(current)
            if ix >= 0:
                combo.setCurrentIndex(ix)
        combo.blockSignals(False)

    def _longitudinal_scope_df(self) -> pd.DataFrame:
        df = self._active_analysis_table().copy()
        if df.empty:
            return df
        task_col = self._task_col(df)
        selected = self._longitudinal_selected_task()
        if selected and task_col and task_col in df.columns:
            mask = df[task_col].astype(str).eq(selected)
            df = df.loc[mask].copy()
        return df

    def _refresh_longitudinal_task_combo(self, df: pd.DataFrame | None = None) -> None:
        combo = getattr(self, "longitudinal_task_combo", None)
        if combo is None:
            return
        df = self._active_analysis_table() if df is None else df
        task_col = self._task_col(df)
        current = combo.currentData()
        values = []
        if task_col and task_col in df.columns:
            values = sorted([str(x) for x in df[task_col].dropna().unique().tolist() if str(x).strip()])
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("All tasks", None)
        for value in values:
            combo.addItem(value, value)
        if current is not None:
            idx = combo.findData(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _on_longitudinal_subject_changed(self) -> None:
        self._refresh_longitudinal_family_combo()
        self.preview_longitudinal_plot(self.longitudinal_view_combo.currentData() if hasattr(self, "longitudinal_view_combo") else "longitudinal_readiness")

    def _longitudinal_feature_cols(self, df: pd.DataFrame) -> list[str]:
        if df is None or df.empty:
            return []
        mapping = getattr(self, "mapping_df", pd.DataFrame())
        try:
            if mapping is None or mapping.empty:
                mapping = self.collect_mapping_from_table()
        except Exception:
            mapping = getattr(self, "mapping_df", pd.DataFrame())
        roles = role_lists(mapping) if mapping is not None and not mapping.empty else {}
        mapped = [c for c in roles.get("Feature", []) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if mapped:
            return mapped
        base = self.feature_df if getattr(self, "feature_df", None) is not None else df
        excluded = {self._subject_col(df), self._task_col(df), self._session_col(df), self._iteration_col(df), self._date_col(df)}
        return [c for c in base.columns if c in df.columns and c not in excluded and pd.api.types.is_numeric_dtype(df[c])][:250]

    def _longitudinal_feature_family_lookup(self, feature_cols: list[str]) -> dict[str, str]:
        lookup = {str(c): "All mapped features" for c in feature_cols}
        registry = getattr(self, "registry_df", None)
        if registry is not None and not registry.empty:
            feature_col = next((c for c in ["feature", "feature_name", "name", "column"] if c in registry.columns), None)
            family_col = next((c for c in ["subsystem", "family", "feature_family", "group", "domain"] if c in registry.columns), None)
            if feature_col and family_col:
                valid = set(map(str, feature_cols))
                for _, row in registry[[feature_col, family_col]].dropna().iterrows():
                    f = str(row[feature_col])
                    if f in valid:
                        lookup[f] = str(row[family_col])
        return lookup

    def _longitudinal_selected_family(self) -> str:
        combo = getattr(self, "longitudinal_family_combo", None)
        if combo is None:
            return "All mapped features"
        value = combo.currentData()
        if value in (None, "", "__all__"):
            return "All mapped features"
        return str(value)

    def _refresh_longitudinal_family_combo(self) -> None:
        combo = getattr(self, "longitudinal_family_combo", None)
        if combo is None:
            return
        df = self._longitudinal_scope_df()
        feature_cols = self._longitudinal_feature_cols(df)
        fam_lookup = self._longitudinal_feature_family_lookup(feature_cols)
        families = sorted({v for v in fam_lookup.values() if str(v).strip() and str(v) != "All mapped features"})
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("All mapped features", "__all__")
        for fam in families[:80]:
            combo.addItem(str(fam), str(fam))
        if current is not None:
            ix = combo.findData(current)
            if ix >= 0:
                combo.setCurrentIndex(ix)
        combo.blockSignals(False)

    def _longitudinal_readiness_table(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame([{"status": "No analysis table available. Run Feature Analysis first."}])
        subj_col = self._subject_col(df)
        task_col = self._task_col(df)
        session_col = self._session_col(df)
        iter_col = self._iteration_col(df)
        date_col = self._date_col(df)
        if not subj_col or subj_col not in df.columns:
            return pd.DataFrame([{"status": "Subject ID field not available; longitudinal readiness cannot be evaluated."}])
        group_cols = [subj_col]
        if task_col and task_col in df.columns:
            group_cols.append(task_col)
        rows = []
        for keys, g in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {
                "subject_id": keys[0],
                "task": keys[1] if len(keys) > 1 else "all_tasks",
                "n_records": int(len(g)),
                "n_sessions": int(g[session_col].nunique(dropna=True)) if session_col and session_col in g.columns else 0,
                "n_iterations": int(g[iter_col].nunique(dropna=True)) if iter_col and iter_col in g.columns else 0,
                "n_dates": 0,
                "first_date": "",
                "last_date": "",
                "elapsed_days": pd.NA,
            }
            if date_col and date_col in g.columns:
                dates = pd.to_datetime(g[date_col], errors="coerce").dropna().sort_values()
                row["n_dates"] = int(dates.nunique())
                if not dates.empty:
                    row["first_date"] = dates.iloc[0].date().isoformat()
                    row["last_date"] = dates.iloc[-1].date().isoformat()
                    row["elapsed_days"] = int((dates.iloc[-1] - dates.iloc[0]).days)
            row["ready_for_longitudinal_review"] = bool(row["n_records"] >= 2 and max(row["n_sessions"], row["n_iterations"], row["n_dates"]) >= 2)
            rows.append(row)
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values(["ready_for_longitudinal_review", "n_records", "elapsed_days"], ascending=[False, False, False], na_position="last")
        return out

    def _longitudinal_visit_records_table(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame([{"status": "No analysis table available. Run Feature Analysis first."}])
        subj_col = self._subject_col(df)
        task_col = self._task_col(df)
        session_col = self._session_col(df)
        iter_col = self._iteration_col(df)
        date_col = self._date_col(df)
        selected_subject = self._longitudinal_selected_subject()
        if not subj_col or subj_col not in df.columns:
            return pd.DataFrame([{"status": "Subject ID field not available."}])
        if not selected_subject:
            return pd.DataFrame([{"status": "Select a repeated subject to show the visit timeline."}])
        g = df.loc[df[subj_col].astype(str).eq(selected_subject)].copy()
        if g.empty:
            return pd.DataFrame([{"status": f"No rows found for selected subject: {selected_subject}"}])
        if date_col and date_col in g.columns:
            g["__date"] = pd.to_datetime(g[date_col], errors="coerce")
        else:
            g["__date"] = pd.NaT
        if iter_col and iter_col in g.columns:
            g["__iteration_num"] = pd.to_numeric(g[iter_col], errors="coerce")
        else:
            g["__iteration_num"] = pd.NA
        sort_cols = [c for c in ["__date", "__iteration_num", session_col] if c and c in g.columns]
        if sort_cols:
            g = g.sort_values(sort_cols, na_position="last")
        rows = []
        first_date = g["__date"].dropna().min() if "__date" in g.columns else pd.NaT
        prev_date = pd.NaT
        for order, (_, row) in enumerate(g.iterrows(), start=1):
            date_value = row.get("__date", pd.NaT)
            days_since_first = pd.NA
            interval_days = pd.NA
            if pd.notna(date_value) and pd.notna(first_date):
                days_since_first = int((date_value - first_date).days)
            if pd.notna(date_value) and pd.notna(prev_date):
                interval_days = int((date_value - prev_date).days)
            if pd.notna(date_value):
                prev_date = date_value
            rows.append({
                "subject_id": selected_subject,
                "task": str(row.get(task_col, self._longitudinal_selected_task() or "all_tasks")) if task_col and task_col in g.columns else (self._longitudinal_selected_task() or "all_tasks"),
                "record_order": order,
                "date": date_value.date().isoformat() if pd.notna(date_value) else "",
                "days_since_first": days_since_first,
                "interval_from_previous_days": interval_days,
                "session_or_visit": str(row.get(session_col, "")) if session_col and session_col in g.columns else "",
                "iteration": str(row.get(iter_col, "")) if iter_col and iter_col in g.columns else "",
            })
        out = pd.DataFrame(rows)
        if len(out) < 2:
            return pd.DataFrame([{"status": "Selected subject has fewer than two records in the current task scope."}])
        return out

    def _longitudinal_feature_direction_lookup(self, feature_cols: list[str]) -> dict[str, str]:
        """Optional registry lookup for clinical direction of change.

        Direction is not inferred from feature names. If the registry contains a
        direction-like column, values are carried into the audit table; otherwise
        features are marked as unspecified so the plot does not imply that an
        increase or decrease is clinically better/worse.
        """
        lookup = {str(c): "not specified" for c in feature_cols}
        registry = getattr(self, "registry_df", None)
        if registry is None or registry.empty:
            return lookup
        feature_col = next((c for c in ["feature", "feature_name", "name", "column"] if c in registry.columns), None)
        direction_col = next((c for c in [
            "direction", "clinical_direction", "expected_direction", "deterioration_direction",
            "higher_is_better", "higher_is_worse", "interpretation_direction", "change_direction"
        ] if c in registry.columns), None)
        if not feature_col or not direction_col:
            return lookup
        valid = set(map(str, feature_cols))
        for _, row in registry[[feature_col, direction_col]].dropna().iterrows():
            f = str(row[feature_col])
            if f in valid:
                val = str(row[direction_col]).strip()
                lookup[f] = val if val else "not specified"
        return lookup

    def _longitudinal_feature_family_change_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selected subject-task feature-level change audit table.

        This table is intentionally strict: it requires a selected subject and
        one selected task, then analyzes only that subject's repeated recordings
        within the same task. Features are kept separate rather than collapsed
        into one family mean because scales, semantics, and clinical direction
        can differ within a family.
        """
        if df is None or df.empty:
            return pd.DataFrame([{"status": "No analysis table available. Run Feature Analysis first."}])
        selected_task = self._longitudinal_selected_task()
        selected_subject = self._longitudinal_selected_subject()
        if not selected_task:
            return pd.DataFrame([{"status": "Select one task first. Longitudinal feature change is only interpreted within the same task."}])
        if not selected_subject:
            return pd.DataFrame([{"status": "Select one repeated subject first."}])
        subj_col = self._subject_col(df)
        task_col = self._task_col(df)
        session_col = self._session_col(df)
        iter_col = self._iteration_col(df)
        date_col = self._date_col(df)
        if not subj_col or subj_col not in df.columns:
            return pd.DataFrame([{"status": "Subject ID field not available."}])
        if not task_col or task_col not in df.columns:
            return pd.DataFrame([{"status": "Task field not available; same-task change cannot be enforced."}])
        task_df = df.loc[df[task_col].astype(str).eq(str(selected_task))].copy()
        g = task_df.loc[task_df[subj_col].astype(str).eq(str(selected_subject))].copy()
        if len(g) < 2:
            return pd.DataFrame([{"status": "Selected subject has fewer than two recordings for the selected task."}])
        feature_cols = self._longitudinal_feature_cols(task_df)
        if not feature_cols:
            return pd.DataFrame([{"status": "No mapped numeric feature columns available for longitudinal change."}])
        fam_lookup = self._longitudinal_feature_family_lookup(feature_cols)
        direction_lookup = self._longitudinal_feature_direction_lookup(feature_cols)
        selected_family = self._longitudinal_selected_family()
        if selected_family and selected_family != "All mapped features":
            selected_features = [f for f in feature_cols if fam_lookup.get(f, "All mapped features") == selected_family]
        else:
            selected_features = list(feature_cols)
            selected_family = "All mapped features"
        selected_features = [f for f in selected_features if f in g.columns and pd.api.types.is_numeric_dtype(task_df[f])]
        if not selected_features:
            return pd.DataFrame([{"status": f"No numeric features found for selected family: {selected_family}"}])
        if date_col and date_col in g.columns:
            g["__date"] = pd.to_datetime(g[date_col], errors="coerce")
        else:
            g["__date"] = pd.NaT
        if iter_col and iter_col in g.columns:
            g["__iteration_num"] = pd.to_numeric(g[iter_col], errors="coerce")
        else:
            g["__iteration_num"] = pd.NA
        sort_cols = [c for c in ["__date", "__iteration_num", session_col] if c and c in g.columns]
        if sort_cols:
            g = g.sort_values(sort_cols, na_position="last")
        cohort = task_df[selected_features].apply(pd.to_numeric, errors="coerce")
        scales = cohort.std(axis=0, ddof=0).replace(0, pd.NA)
        q75 = cohort.quantile(0.75)
        q25 = cohort.quantile(0.25)
        iqr = (q75 - q25).replace(0, pd.NA)
        scales = scales.fillna(iqr).replace(0, pd.NA)
        usable = [f for f in selected_features if pd.notna(scales.get(f, pd.NA))]
        if not usable:
            return pd.DataFrame([{"status": "Selected feature family has no varying features in the selected task; standardized change cannot be computed."}])
        values = g[usable].apply(pd.to_numeric, errors="coerce")
        baseline = values.iloc[0]
        first_date = g["__date"].dropna().min() if "__date" in g.columns else pd.NaT
        rows = []
        for order, (idx, row) in enumerate(g.iterrows(), start=1):
            vals = values.loc[idx]
            date_value = row.get("__date", pd.NaT)
            days_since_first = pd.NA
            if pd.notna(date_value) and pd.notna(first_date):
                days_since_first = int((date_value - first_date).days)
            for feat in usable:
                raw_value = vals.get(feat, pd.NA)
                base_value = baseline.get(feat, pd.NA)
                scale = scales.get(feat, pd.NA)
                raw_change = pd.NA
                pct_change = pd.NA
                z_change = pd.NA
                if pd.notna(raw_value) and pd.notna(base_value):
                    raw_change = float(raw_value - base_value)
                    if float(base_value) != 0:
                        pct_change = float(100.0 * raw_change / abs(float(base_value)))
                if pd.notna(raw_change) and pd.notna(scale) and float(scale) != 0:
                    z_change = float(raw_change / float(scale))
                rows.append({
                    "subject_id": selected_subject,
                    "task": selected_task,
                    "feature_family": fam_lookup.get(feat, selected_family),
                    "selected_family": selected_family,
                    "feature": feat,
                    "record_order": order,
                    "date": date_value.date().isoformat() if pd.notna(date_value) else "",
                    "days_since_first": days_since_first,
                    "session_or_visit": str(row.get(session_col, "")) if session_col and session_col in g.columns else "",
                    "iteration": str(row.get(iter_col, "")) if iter_col and iter_col in g.columns else "",
                    "raw_value": float(raw_value) if pd.notna(raw_value) else pd.NA,
                    "baseline_value": float(base_value) if pd.notna(base_value) else pd.NA,
                    "raw_change_from_baseline": raw_change,
                    "percent_change_from_baseline": pct_change,
                    "cohort_scale_within_task": float(scale) if pd.notna(scale) else pd.NA,
                    "standardized_change_from_baseline": z_change,
                    "absolute_standardized_change": abs(z_change) if pd.notna(z_change) else pd.NA,
                    "clinical_direction_from_registry": direction_lookup.get(feat, "not specified"),
                    "direction_interpretation": "numeric direction only; clinical meaning not inferred" if direction_lookup.get(feat, "not specified") == "not specified" else "use registry direction",
                })
        out = pd.DataFrame(rows)
        if out.empty or out["standardized_change_from_baseline"].notna().sum() < 2:
            return pd.DataFrame([{"status": "Selected subject-task feature audit has fewer than two usable feature-change values."}])
        return out

    def update_longitudinal_dashboard(self, outputs: dict[str, pd.DataFrame]) -> None:
        if not hasattr(self, "longitudinal_metric_grid"):
            return
        while self.longitudinal_metric_grid.count():
            item = self.longitudinal_metric_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        full_df = self._active_analysis_table()
        self._refresh_longitudinal_task_combo(full_df)
        df = self._longitudinal_scope_df()
        subj_col = self._subject_col(df)
        task_col = self._task_col(df)
        session_col = self._session_col(df)
        iter_col = self._iteration_col(df)
        date_col = self._date_col(df)
        readiness = self._longitudinal_readiness_table(df)
        self._refresh_longitudinal_subject_combo(readiness)
        self._refresh_longitudinal_family_combo()
        total_units = len(readiness) if not readiness.empty and "status" not in readiness.columns else 0
        repeated_df = readiness.loc[pd.to_numeric(readiness.get("n_records", pd.Series(dtype=float)), errors="coerce").fillna(0) >= 2].copy() if total_units else pd.DataFrame()
        repeated_units = int(len(repeated_df))
        repeated_subjects = int(repeated_df["subject_id"].astype(str).nunique()) if repeated_units and "subject_id" in repeated_df.columns else 0
        dated_repeated = repeated_df.loc[pd.to_numeric(repeated_df.get("elapsed_days", pd.Series(dtype=float)), errors="coerce").notna()].copy() if repeated_units else pd.DataFrame()
        median_elapsed = pd.to_numeric(dated_repeated.get("elapsed_days", pd.Series(dtype=float)), errors="coerce").median() if not dated_repeated.empty else pd.NA
        mean_elapsed = pd.to_numeric(dated_repeated.get("elapsed_days", pd.Series(dtype=float)), errors="coerce").mean() if not dated_repeated.empty else pd.NA
        median_records = pd.to_numeric(repeated_df.get("n_records", pd.Series(dtype=float)), errors="coerce").median() if repeated_units else pd.NA
        mean_records = pd.to_numeric(repeated_df.get("n_records", pd.Series(dtype=float)), errors="coerce").mean() if repeated_units else pd.NA
        tiles = [
            ("Task scope", self._longitudinal_selected_task() or "All tasks", "current filter"),
            ("Repeated subjects", repeated_subjects, "≥2 same-task recordings"),
            ("Repeated subject-task units", repeated_units, f"of {total_units} evaluated"),
            ("Median / mean records", "n/a" if pd.isna(median_records) else f"{median_records:.1f} / {mean_records:.1f}", "per repeated unit"),
            ("Median / mean span", "n/a" if pd.isna(median_elapsed) else f"{median_elapsed:.0f} / {mean_elapsed:.0f} d", "dated repeated units"),
            ("Fields", f"S:{'yes' if subj_col else 'no'} T:{'yes' if task_col else 'no'} D:{'yes' if date_col else 'no'}", "subject/task/date"),
        ]
        for idx, (title, value, subtitle) in enumerate(tiles):
            self.longitudinal_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx, 0)

        if readiness.empty:
            readiness = pd.DataFrame([{"status": "No subject-task units could be evaluated."}])
        repeated_for_tables = readiness.loc[pd.to_numeric(readiness.get("n_records", pd.Series(dtype=float)), errors="coerce").fillna(0) >= 2].copy() if "n_records" in readiness.columns else pd.DataFrame()
        if repeated_for_tables.empty:
            repeated_for_tables = pd.DataFrame([{"status": "No subject-task unit has at least two recordings of the same task in the current scope."}])
        self._fill_table(self.long_subject_table, repeated_for_tables)
        support_rows = [{"field": "subject", "column": subj_col or "not detected", "required": True},
                        {"field": "task", "column": task_col or "not detected", "required": False},
                        {"field": "session/visit", "column": session_col or "not detected", "required": False},
                        {"field": "iteration", "column": iter_col or "not detected", "required": False},
                        {"field": "recording/visit date", "column": date_col or "not detected", "required": False}]
        self._fill_table(self.long_session_table, pd.DataFrame(support_rows))
        repeated_source = repeated_for_tables if isinstance(repeated_for_tables, pd.DataFrame) and "status" not in repeated_for_tables.columns else readiness
        iter_summary = repeated_source[[c for c in ["subject_id", "task", "n_records", "n_sessions", "n_iterations", "ready_for_longitudinal_review"] if c in repeated_source.columns]].copy() if not repeated_source.empty else pd.DataFrame()
        date_summary = repeated_source[[c for c in ["subject_id", "task", "n_dates", "first_date", "last_date", "elapsed_days", "ready_for_longitudinal_review"] if c in repeated_source.columns]].copy() if not repeated_source.empty else pd.DataFrame()
        self._fill_table(self.long_iteration_table, iter_summary)
        self._fill_table(self.long_date_table, date_summary)
        self._fill_table(self.long_feature_change_table, self._longitudinal_feature_family_change_table(df))
        self.longitudinal_note.setText("Repeated-record cohort summary generated. Later longitudinal plots will analyze only subject-task units with at least two recordings of the same task.")
        self.generate_longitudinal_plots()
        self.preview_longitudinal_plot("longitudinal_readiness")

    def generate_longitudinal_plots(self) -> None:
        df = self._longitudinal_scope_df()
        self.output_dir, tables_dir, reports_dir, plots_dir = self._analysis_dirs()
        subj_col = self._subject_col(df)
        task_col = self._task_col(df)
        session_col = self._session_col(df)
        iter_col = self._iteration_col(df)
        date_col = self._date_col(df)
        readiness = self._longitudinal_readiness_table(df)
        if not hasattr(self, "plot_paths"):
            self.plot_paths = {}
        self.plot_paths["longitudinal_readiness"] = str(plot_longitudinal_readiness(readiness, plots_dir / "longitudinal_readiness.png"))
        visit_records = self._longitudinal_visit_records_table(df)
        self.plot_paths["longitudinal_visit_timeline"] = str(plot_longitudinal_visit_timeline(visit_records, plots_dir / "longitudinal_visit_timeline.png"))
        feature_family_change = self._longitudinal_feature_family_change_table(df)
        if hasattr(self, "long_feature_change_table"):
            self._fill_table(self.long_feature_change_table, feature_family_change)
        self.plot_paths["longitudinal_feature_family_trajectory"] = str(plot_longitudinal_feature_family_trajectory(feature_family_change, plots_dir / "longitudinal_feature_family_trajectory.png"))

    def preview_longitudinal_plot(self, key: str | None = None) -> None:
        key = key or (self.longitudinal_view_combo.currentData() if hasattr(self, "longitudinal_view_combo") else "long_subject_records")
        # Always regenerate the selected longitudinal plot from the current task/subject/family controls.
        # This prevents stale paths and removes the need to click Regenerate before Show.
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
            "longitudinal_readiness": "Repeated-record cohort summary. Use this first to quantify how many subjects have repeated same-task recordings, typical recording depth, and typical dated follow-up span before reviewing feature change.",
            "longitudinal_visit_timeline": "Selected subject-task visit timeline. Use this to verify visit ordering, days since first recording, and spacing between repeated recordings before interpreting feature change.",
            "longitudinal_feature_family_trajectory": "Selected subject-task feature change audit. This analyzes only repeated recordings of the same task for the selected subject. Each feature is shown separately as standardized change from that subject's first same-task recording; positive/negative sign is numeric direction, not clinical improvement/worsening unless the registry defines direction."
        }
        if hasattr(self, "longitudinal_interpretation"):
            self.longitudinal_interpretation.setText(captions.get(key, "Longitudinal review plot."))
        pix = QPixmap(str(path))
        if pix.isNull():
            self.longitudinal_preview.setText(f"Could not load plot:\n{path}")
            return
        target_size = self.longitudinal_preview.contentsRect().size()
        if target_size.width() < 50 or target_size.height() < 50:
            target_size = QSize(900, 460)
        scaled = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.longitudinal_preview.clear()
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
            self.feature_df = self._ensure_unique_columns(read_table(self.feature_picker.path), "Primary feature table")
            self.refresh_filename_source_combo(self.feature_df)
            self.qc_df = self._ensure_unique_columns(read_table(self.qc_picker.path), "QC table") if self.qc_picker.path else None
            self.meta_df = self._ensure_unique_columns(read_table(self.meta_picker.path), "Metadata table") if self.meta_picker.path else None
            if self.meta_df is not None:
                self.metadata_mapping_df = self._build_metadata_mapping_df()
                self.metadata_mapping_accepted = False
                self.refresh_metadata_mapping_table()
            else:
                self.metadata_mapping_df = pd.DataFrame()
                self.metadata_mapping_accepted = False
                self.refresh_metadata_mapping_table()
            self.registry_df = self._ensure_unique_columns(read_table(self.registry_picker.path), "Feature registry") if self.registry_picker.path else None
            kind = infer_table_kind(self.feature_picker.path, explicit="feature")
            feature_mapping = classify_columns(self.feature_df, table_kind=kind, registry=self.registry_df)
            self.refresh_filename_template_ui()
            self.analysis_df, self.mapping_df, self.metadata_join_strategy = self._merge_metadata_context(self.feature_df, feature_mapping)
            self.proposed_mapping_df = self.mapping_df.copy()
            self.mapping_modified = False
            self.mapping_accepted = False
            self.analysis_ready = False
            self.outputs = {}
            self.refresh_mapping_table()
            roles = summarize_roles(self.mapping_df)
            self.log(f"Loaded feature table: {self.feature_df.shape[0]} rows x {self.feature_df.shape[1]} columns")
            if self.qc_df is not None:
                self.log(f"Loaded QC table: {self.qc_df.shape[0]} rows x {self.qc_df.shape[1]} columns")
            if self.meta_df is not None:
                self.log(f"Loaded metadata table: {self.meta_df.shape[0]} rows x {self.meta_df.shape[1]} columns")
                self.log(f"Metadata context strategy: {self.metadata_join_strategy}")
                self.log("Review Metadata Mapping if clinical/demographic columns need manual assignment.")
            else:
                self.log("No metadata table loaded; use Metadata Mapping > filename-derived metadata fallback to parse subject/iteration/date/task from feature-table filenames.")
            selected_filename_col = self._file_source_column(self.feature_df) if self.feature_df is not None else None
            self.log(f"Filename context source: {self._filename_parser_mode()} | filename column: {selected_filename_col or 'none'}")
            parsed_df = getattr(self, "filename_context_parse_df", pd.DataFrame())
            if parsed_df is not None and not parsed_df.empty and "parsed_context_status" in parsed_df.columns:
                parsed_ok = int(parsed_df["parsed_context_status"].astype(str).isin(["parsed", "parsed_by_user_template"]).sum())
                self.log(f"Filename context parser mode: {self._filename_parser_mode()} | parsed task context rows: {parsed_ok} / {len(parsed_df)}")
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
        for visual_i, source_i in enumerate(df.index.tolist()):
            r = df.loc[source_i]
            self.mapping_table.setItem(visual_i, 0, QTableWidgetItem(str(r["column"])))
            combo = NoWheelComboBox()
            combo.setStyleSheet("QComboBox { min-width: 128px; max-width: 145px; padding: 4px 6px; }")
            combo.addItems(ROLE_OPTIONS)
            combo.setCurrentText(str(r["role"]))
            combo.setProperty("source_index", int(source_i) if isinstance(source_i, int) or str(type(source_i)).endswith("int64'>") else str(source_i))
            combo.setProperty("source_column", str(r["column"]))
            combo.currentTextChanged.connect(self.mark_mapping_modified)
            self.mapping_table.setCellWidget(visual_i, 1, combo)
            for col_idx, value in [
                (2, f"{float(r['confidence']):.2f}"),
                (3, str(r["reason"])),
                (4, str(r["dtype"])),
                (5, f"{float(r['missing_fraction']):.3f}"),
                (6, str(r["unique_values"])),
            ]:
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.mapping_table.setItem(visual_i, col_idx, item)
        self.mapping_table.resizeRowsToContents()
        for row in range(self.mapping_table.rowCount()):
            self.mapping_table.setRowHeight(row, min(max(self.mapping_table.rowHeight(row), 28), 38))
        self.update_mapping_summary()

    def collect_mapping_from_table(self) -> pd.DataFrame:
        """Collect Feature Mapping roles without assuming table rows equal mapping rows.

        The mapping table can be rebuilt or partially visible after metadata joins,
        and the underlying mapping_df can include feature + metadata columns. Older
        code assigned a raw roles list to df["role"], which fails whenever the
        widget row count differs from mapping_df length, e.g. 275 visible rows vs
        451 mapped columns. Update by stored source_index/source_column instead.
        """
        if self.mapping_df.empty:
            return self.mapping_df
        df = self.mapping_df.copy()
        if not hasattr(self, "mapping_table"):
            return df
        if "role" not in df.columns:
            df["role"] = ROLE_IGNORE
        if "column" not in df.columns:
            self.mapping_df = df
            return df

        table_rows = int(self.mapping_table.rowCount())
        if table_rows != len(df) and hasattr(self, "log"):
            self.log(f"WARN | Feature Mapping table has {table_rows} visible rows but mapping has {len(df)} rows; syncing roles by source index/column.")

        # Track duplicate column labels so fallback column-name matching is only
        # used when unambiguous. Source-index matching is preferred.
        col_counts = df["column"].astype(str).value_counts().to_dict()
        for visual_i in range(table_rows):
            widget = self.mapping_table.cellWidget(visual_i, 1)
            if not isinstance(widget, QComboBox):
                continue
            role = widget.currentText()
            source_i = widget.property("source_index")
            updated = False
            if source_i is not None:
                try:
                    if source_i in df.index:
                        df.loc[source_i, "role"] = role
                        updated = True
                    else:
                        source_i_int = int(source_i)
                        if source_i_int in df.index:
                            df.loc[source_i_int, "role"] = role
                            updated = True
                except Exception:
                    updated = False
            if updated:
                continue
            source_col = widget.property("source_column")
            if source_col is None:
                item = self.mapping_table.item(visual_i, 0)
                source_col = item.text() if item is not None else None
            if source_col is None:
                continue
            source_col = str(source_col)
            if col_counts.get(source_col, 0) == 1:
                df.loc[df["column"].astype(str).eq(source_col), "role"] = role

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
        self.refresh_filename_template_ui()
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
                f"You changed {n_changed} column role(s) from the proposed mapping.\n\nSave this accepted mapping and continue to Metadata Mapping?",
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
        self.log("Next step: review Metadata Mapping. If no metadata table is loaded, apply filename-derived context there.")
        self.show_page("metadata_mapping")

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

            try:
                outputs, feature_cols = self._build_analysis_outputs()
                full_analysis_ok = True
            except Exception as exc:
                self.log_error("Full feature analysis tables failed; falling back to Overview-only outputs", exc)
                outputs, feature_cols = self._build_overview_outputs()
                full_analysis_ok = False
            self.outputs = outputs
            self._write_outputs(outputs, tables_dir)
            self.plot_paths = self._generate_overview_plots(outputs, feature_cols, plots_dir)

            if full_analysis_ok:
                self.write_report(reports_dir / "vslp_feature_analysis_report.html", outputs)
                self.populate_output_tables(outputs)
                self._safe_refresh_analysis_dashboards(outputs)
                self.analysis_ready = True
                self.log(f"Analysis complete. Outputs written to: {self.output_dir}")
            else:
                self.analysis_ready = False
                self.update_overview_dashboard(outputs)
                self.log(f"Overview-only analysis complete. Deep menu tables were skipped after a dimension mismatch; Overview outputs written to: {self.output_dir}")
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
