"""Acoustic Pipeline GUI scaffold."""

from __future__ import annotations

try:
    from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QTabWidget, QWidget, QVBoxLayout
    from vslp.gui.common.theme import VSLP_DARK_QSS
except Exception:  # Allows backend tests without GUI extras.
    QApplication = None


class AcousticPipelineWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VSLP | Acoustic Pipeline")
        self.resize(1440, 900)
        tabs = QTabWidget()
        for name in ["Project", "Ingest", "Preprocess", "Segmentation", "QC", "Features", "Aggregation", "Reports"]:
            w = QWidget()
            layout = QVBoxLayout(w)
            layout.addWidget(QLabel(f"{name} module scaffold. Backend contracts are implemented first."))
            tabs.addTab(w, name)
        self.setCentralWidget(tabs)


def main():
    if QApplication is None:
        raise ImportError("Install GUI dependencies with `pip install -e .[gui]`.")
    app = QApplication([])
    app.setStyleSheet(VSLP_DARK_QSS)
    win = AcousticPipelineWindow()
    win.show()
    app.exec()
