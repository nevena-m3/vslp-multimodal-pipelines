"""Independent Alignment GUI launcher."""

from __future__ import annotations

import sys


def launch_alignment_gui() -> int:
    from PySide6.QtWidgets import QApplication

    from vslp.gui.alignment_app.main_window import AlignmentWindow
    from vslp.gui.theme import build_dark_stylesheet

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("VSLP Alignment")
    app.setStyleSheet(build_dark_stylesheet())
    window = AlignmentWindow()
    window.show()
    return app.exec()
