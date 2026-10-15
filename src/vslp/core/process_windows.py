"""Cross-platform subprocess options for GUI-triggered console tools."""

from __future__ import annotations

import subprocess
import sys


def hidden_console_kwargs() -> dict:
    """Capture output normally without opening a child console on Windows."""
    if sys.platform != "win32":
        return {}
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    return {"creationflags": subprocess.CREATE_NO_WINDOW, "startupinfo": startup}
