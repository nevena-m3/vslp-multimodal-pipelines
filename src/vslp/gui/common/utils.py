"""GUI helper utilities."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import webbrowser


def open_path(path: str | Path) -> None:
    target = str(Path(path).expanduser().resolve())
    if sys.platform.startswith("darwin"):
        subprocess.Popen(["open", target])
    elif os.name == "nt":
        os.startfile(target)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", target])


def open_in_browser(path: str | Path) -> None:
    webbrowser.open(Path(path).expanduser().resolve().as_uri())
