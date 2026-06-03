"""ML Modeling GUI scaffold."""

from __future__ import annotations

try:
    from PySide6.QtWidgets import QApplication, QLabel, QMainWindow
    from vslp.gui.common.theme import VSLP_DARK_QSS
except Exception:
    QApplication = None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VSLP | ML Modeling")
        self.resize(1440, 900)
        self.setCentralWidget(QLabel("ML Modeling scaffold. Backend contracts will drive this GUI."))

def main():
    if QApplication is None:
        raise ImportError("Install GUI dependencies with `pip install -e .[gui]`.")
    app = QApplication([])
    app.setStyleSheet(VSLP_DARK_QSS)
    win = MainWindow()
    win.show()
    app.exec()
