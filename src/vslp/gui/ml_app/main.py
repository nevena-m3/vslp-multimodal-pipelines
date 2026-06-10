"""VSLP ML Modeling GUI — v0.1 data-contract workstation.

The first ML GUI milestone deliberately builds datasets and validates multimodal
inputs before any model training is exposed. This prevents subject leakage,
feature-role confusion, and fragile acoustic/kinematic joins.
"""

from __future__ import annotations

from pathlib import Path
import json

try:
    import pandas as pd
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QComboBox,
    )
    from vslp.gui.common.theme import VSLP_DARK_QSS
    from vslp.gui.common.utils import open_path
    from vslp.ml.dataset_contract import MLDatasetContractConfig, build_ml_datasets
except Exception:  # pragma: no cover - allows import without GUI deps
    QApplication = None

APP_VERSION = "v0.1"


def _set_table(table: QTableWidget, df: pd.DataFrame | None, max_rows: int = 200) -> None:
    if df is None or df.empty:
        table.setRowCount(0)
        table.setColumnCount(0)
        return
    show = df.head(max_rows).copy()
    table.setRowCount(len(show))
    table.setColumnCount(len(show.columns))
    table.setHorizontalHeaderLabels([str(c) for c in show.columns])
    for r, (_, row) in enumerate(show.iterrows()):
        for c, value in enumerate(row):
            item = QTableWidgetItem("" if pd.isna(value) else str(value))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(r, c, item)
    table.resizeColumnsToContents()


class PathRow(QWidget):
    def __init__(self, label: str, file_filter: str = "CSV files (*.csv);;All files (*.*)"):
        super().__init__()
        self.file_filter = file_filter
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(label)
        self.label.setMinimumWidth(175)
        self.edit = QLineEdit()
        self.button = QPushButton("Browse")
        self.button.clicked.connect(self._browse)
        layout.addWidget(self.label)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", str(Path.home()), self.file_filter)
        if path:
            self.edit.setText(path)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, text: str) -> None:
        self.edit.setText(text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"VSLP | ML Modeling {APP_VERSION}")
        self.resize(1500, 950)
        self.output_root = QLineEdit(str(Path.home() / "VSLP_ML_Output"))
        self.status_cards: dict[str, QLabel] = {}
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        main = QVBoxLayout(root)

        header = QLabel("ML Modeling Workstation — Data Contract First")
        header.setStyleSheet("font-size: 22px; font-weight: 700; padding: 8px 0;")
        sub = QLabel(
            "Load acoustic/kinematic ML-ready tables and metadata, validate row keys, build unimodal and early-fusion datasets. "
            "Model training is intentionally disabled in v0.1 until the dataset contract is proven."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #B7C7D6; padding-bottom: 8px;")
        main.addWidget(header)
        main.addWidget(sub)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("ML output root"))
        out_row.addWidget(self.output_root, 1)
        out_browse = QPushButton("Browse")
        out_browse.clicked.connect(self._browse_output_root)
        out_row.addWidget(out_browse)
        main.addLayout(out_row)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._data_sources_tab(), "Data Sources")
        self.tabs.addTab(self._dataset_builder_tab(), "Dataset Builder")
        self.tabs.addTab(self._feature_space_placeholder_tab(), "Modality & Feature Space")
        self.tabs.addTab(self._validation_placeholder_tab(), "Splits & Validation")
        self.tabs.addTab(self._future_tabs_placeholder(), "Training / Evaluation / Registry")
        main.addWidget(self.tabs, 1)
        self.setCentralWidget(root)

    def _data_sources_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        sources = QGroupBox("Input sources")
        s_layout = QVBoxLayout(sources)
        self.acoustic_csv = PathRow("Acoustic ML-ready CSV")
        self.kinematic_csv = PathRow("Kinematic ML-ready CSV")
        self.metadata_csv = PathRow("Metadata / labels CSV")
        self.acoustic_manifest_csv = PathRow("Acoustic feature manifest")
        self.kinematic_manifest_csv = PathRow("Kinematic feature manifest")
        for row in [self.acoustic_csv, self.kinematic_csv, self.metadata_csv, self.acoustic_manifest_csv, self.kinematic_manifest_csv]:
            s_layout.addWidget(row)
        layout.addWidget(sources)

        actions = QHBoxLayout()
        load_btn = QPushButton("Inspect Sources")
        load_btn.clicked.connect(self.inspect_sources)
        build_btn = QPushButton("Build ML Dataset Contract")
        build_btn.clicked.connect(self.build_contract)
        open_btn = QPushButton("Open ML Output Folder")
        open_btn.clicked.connect(lambda: open_path(self.output_root.text()))
        actions.addWidget(load_btn)
        actions.addWidget(build_btn)
        actions.addWidget(open_btn)
        actions.addStretch(1)
        layout.addLayout(actions)

        cards = QGroupBox("Source status")
        grid = QGridLayout(cards)
        for i, name in enumerate(["Acoustic rows", "Kinematic rows", "Metadata rows", "Acoustic features", "Kinematic features", "Matched rows"]):
            lab = QLabel("0")
            lab.setStyleSheet("font-size: 18px; font-weight: 700; color: #7BD88F;")
            self.status_cards[name] = lab
            grid.addWidget(QLabel(name), i // 3 * 2, i % 3)
            grid.addWidget(lab, i // 3 * 2 + 1, i % 3)
        layout.addWidget(cards)

        self.source_notes = QTextEdit()
        self.source_notes.setReadOnly(True)
        self.source_notes.setMinimumHeight(130)
        self.source_notes.setText(
            "Expected inputs:\n"
            "- acoustic_features_ml_ready.csv + acoustic_feature_manifest.csv\n"
            "- kinematic_features_ml_ready.csv + kinematic_feature_manifest.csv\n"
            "- metadata/labels CSV with subject_id and target labels\n\n"
            "The ML GUI will build acoustic-only, kinematic-only, and early-fusion datasets."
        )
        layout.addWidget(self.source_notes)
        return page

    def _dataset_builder_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        cfg = QGroupBox("Dataset contract settings")
        grid = QGridLayout(cfg)
        self.target_column = QLineEdit()
        self.key_columns = QLineEdit()
        self.qc_policy = QComboBox()
        self.qc_policy.addItems(["pass_review", "pass_only", "include_all"])
        self.comparison_mode = QComboBox()
        self.comparison_mode.addItems(["maximum_available", "matched_subjects"])
        grid.addWidget(QLabel("Target column"), 0, 0)
        grid.addWidget(self.target_column, 0, 1)
        grid.addWidget(QLabel("Optional key columns, comma-separated"), 1, 0)
        grid.addWidget(self.key_columns, 1, 1)
        grid.addWidget(QLabel("QC policy"), 2, 0)
        grid.addWidget(self.qc_policy, 2, 1)
        grid.addWidget(QLabel("Comparison mode"), 3, 0)
        grid.addWidget(self.comparison_mode, 3, 1)
        layout.addWidget(cfg)

        self.output_tabs = QTabWidget()
        self.overlap_table = QTableWidget(0, 0)
        self.manifest_table = QTableWidget(0, 0)
        self.acoustic_table = QTableWidget(0, 0)
        self.kinematic_table = QTableWidget(0, 0)
        self.early_table = QTableWidget(0, 0)
        self.contract_json = QTextEdit()
        self.contract_json.setReadOnly(True)
        self.output_tabs.addTab(self.overlap_table, "Modality overlap")
        self.output_tabs.addTab(self.acoustic_table, "Acoustic-only dataset")
        self.output_tabs.addTab(self.kinematic_table, "Kinematic-only dataset")
        self.output_tabs.addTab(self.early_table, "Early-fusion dataset")
        self.output_tabs.addTab(self.manifest_table, "ML feature manifest")
        self.output_tabs.addTab(self.contract_json, "Dataset manifest JSON")
        layout.addWidget(self.output_tabs, 1)
        return page

    def _feature_space_placeholder_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setText(
            "Feature-space controls will be built after v0.1 confirms the dataset contract.\n\n"
            "Planned controls:\n"
            "- acoustic-only, kinematic-only, early-fusion, late-fusion modes\n"
            "- canonical vs ML-safe vs custom feature sets\n"
            "- family filters: voice quality, timing, rhythm, lip aperture, jaw, symmetry, velocity, etc.\n"
            "- matched-subject comparison mode for fair modality comparisons\n"
            "- feature-role guardrails to exclude metadata, QC, provenance, and target leakage."
        )
        layout.addWidget(text)
        return page

    def _validation_placeholder_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setText(
            "Splits and validation will be implemented after the v0.1 dataset builder.\n\n"
            "Required defaults:\n"
            "- subject-grouped train/test and CV\n"
            "- StratifiedGroupKFold for classification when feasible\n"
            "- GroupKFold for regression\n"
            "- leakage checks: same subject must never appear in train and test\n"
            "- longitudinal time-aware split scaffold when visit dates/days_from_baseline are available."
        )
        layout.addWidget(text)
        return page

    def _future_tabs_placeholder(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setText(
            "Future ML tabs will be built only after data assembly and leakage-safe splitting are stable.\n\n"
            "Planned stages:\n"
            "1. Baselines: dummy, logistic/ridge.\n"
            "2. Training: SVM, elastic net, random forest, gradient boosting.\n"
            "3. Optimization: fold-internal imputation/scaling/selection and nested CV.\n"
            "4. Evaluation: ROC/PR/confusion/residuals/calibration.\n"
            "5. Interpretation: feature/family/modality importance.\n"
            "6. Registry: import/export sklearn pipelines with model cards and expected feature schema."
        )
        layout.addWidget(text)
        return page

    def _browse_output_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select ML output folder", str(Path.home()))
        if path:
            self.output_root.setText(path)

    def _load_optional(self, path: str) -> pd.DataFrame | None:
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        return pd.read_csv(p)

    def inspect_sources(self):
        acoustic = self._load_optional(self.acoustic_csv.text())
        kinematic = self._load_optional(self.kinematic_csv.text())
        metadata = self._load_optional(self.metadata_csv.text())
        aman = self._load_optional(self.acoustic_manifest_csv.text())
        kman = self._load_optional(self.kinematic_manifest_csv.text())
        self.status_cards["Acoustic rows"].setText(str(0 if acoustic is None else len(acoustic)))
        self.status_cards["Kinematic rows"].setText(str(0 if kinematic is None else len(kinematic)))
        self.status_cards["Metadata rows"].setText(str(0 if metadata is None else len(metadata)))
        self.status_cards["Acoustic features"].setText(str(0 if aman is None else int(aman.get("include_in_ml_default", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1", "yes"]).sum())))
        self.status_cards["Kinematic features"].setText(str(0 if kman is None else int(kman.get("include_in_ml_default", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1", "yes"]).sum())))
        notes = ["Source inspection complete."]
        for name, df in [("acoustic", acoustic), ("kinematic", kinematic), ("metadata", metadata)]:
            notes.append(f"{name}: {'missing' if df is None else str(df.shape)}")
        self.source_notes.setText("\n".join(notes))

    def _config(self) -> MLDatasetContractConfig:
        keys = [k.strip() for k in self.key_columns.text().split(",") if k.strip()]
        return MLDatasetContractConfig(
            acoustic_features_csv=self.acoustic_csv.text() or None,
            kinematic_features_csv=self.kinematic_csv.text() or None,
            metadata_csv=self.metadata_csv.text() or None,
            acoustic_manifest_csv=self.acoustic_manifest_csv.text() or None,
            kinematic_manifest_csv=self.kinematic_manifest_csv.text() or None,
            target_column=self.target_column.text().strip() or None,
            key_columns=keys,
            qc_policy=self.qc_policy.currentText(),
            comparison_mode=self.comparison_mode.currentText(),
        )

    def build_contract(self):
        out = build_ml_datasets(self.output_root.text(), self._config())
        _set_table(self.overlap_table, pd.read_csv(out["modality_overlap_csv"]))
        _set_table(self.manifest_table, pd.read_csv(out["feature_manifest_csv"]), max_rows=300)
        _set_table(self.acoustic_table, pd.read_csv(out["acoustic_only_csv"]), max_rows=50)
        _set_table(self.kinematic_table, pd.read_csv(out["kinematic_only_csv"]), max_rows=50)
        _set_table(self.early_table, pd.read_csv(out["early_fusion_csv"]), max_rows=50)
        payload = json.loads(Path(out["dataset_manifest_json"]).read_text(encoding="utf-8"))
        self.contract_json.setText(json.dumps(payload, indent=2))
        self.status_cards["Matched rows"].setText(str(payload.get("n_rows", {}).get("early_fusion", 0)))
        self.source_notes.setText("ML dataset contract built. Model training remains disabled until splits/validation are implemented.")
        self.tabs.setCurrentIndex(1)


def main():
    if QApplication is None:
        raise ImportError("Install GUI dependencies with `pip install -e .[gui]`.")
    app = QApplication([])
    app.setStyleSheet(VSLP_DARK_QSS)
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
