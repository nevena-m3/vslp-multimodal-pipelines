"""Acoustic GUI launcher."""

from __future__ import annotations

import sys


def launch_acoustic_gui() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "The acoustic GUI requires PySide6. Activate your VSLP environment and run: "
            "pip install -e '.[gui]'"
        ) from exc

    from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow
    from vslp.gui.theme import build_dark_stylesheet

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("VSLP")
    app.setStyleSheet(build_dark_stylesheet())

    window = AcousticPipelineWindow()
    window.show()
    return app.exec()
