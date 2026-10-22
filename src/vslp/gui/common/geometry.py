"""Screen-aware initial sizing for the two desktop applications."""

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget


def fit_initial_window(window: QWidget, preferred: tuple[int, int],
                       minimum: tuple[int, int]) -> None:
    """Use Qt logical pixels and leave room for the desktop window frame."""
    screen = window.screen() or QGuiApplication.primaryScreen()
    if screen is None:
        window.setMinimumSize(*minimum)
        window.resize(*preferred)
        return
    available = screen.availableGeometry()
    width = min(preferred[0], max(1, available.width() - 32))
    height = min(preferred[1], max(1, available.height() - 32))
    window.setMinimumSize(min(minimum[0], width), min(minimum[1], height))
    window.resize(width, height)
