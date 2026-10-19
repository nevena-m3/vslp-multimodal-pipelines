"""Shared background stage runner for the two VSLP desktop applications."""

from __future__ import annotations

import inspect
import traceback
from typing import Callable

from PySide6.QtCore import QObject, Signal

from vslp.core.project import prune_empty_acoustic_directories


class Worker(QObject):
    started = Signal(str)
    finished = Signal(str, object)
    failed = Signal(str, str)
    message = Signal(str)
    progress = Signal(int, int, str)

    def __init__(self, name: str, func: Callable, kwargs: dict):
        super().__init__()
        self.name = name
        self.func = func
        self.kwargs = kwargs
        self.output_root = kwargs.get("output_root")

    def run(self) -> None:
        label = "Acoustic Features" if self.name == "features" else self.name
        self.started.emit(self.name)
        self.message.emit(f"Starting {label}...")
        try:
            kwargs = dict(self.kwargs)
            if "progress_callback" in inspect.signature(self.func).parameters:
                kwargs["progress_callback"] = self.progress.emit
            result = self.func(**kwargs)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            if self.output_root is not None:
                prune_empty_acoustic_directories(self.output_root)
            self.failed.emit(self.name, f"{exc}\n\n{tb}")
            return
        if self.output_root is not None:
            prune_empty_acoustic_directories(self.output_root)
        self.message.emit(f"Finished {label}: {getattr(result, 'status', 'ok')}")
        self.finished.emit(self.name, result)
