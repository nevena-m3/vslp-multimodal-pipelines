"""Bounded, cancellable subprocess runner for one MFA corpus operation."""

from __future__ import annotations

import os
import subprocess
from threading import Event, Lock
from time import monotonic


class AlignmentCancelled(RuntimeError):
    """The user cancelled an external alignment command."""


class CancellableMfaRunner:
    def __init__(self) -> None:
        self._cancel = Event()
        self._lock = Lock()
        self._process: subprocess.Popen[str] | None = None

    def cancel(self) -> None:
        self._cancel.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            self._terminate_tree(process)

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @staticmethod
    def _terminate_tree(process: subprocess.Popen[str]) -> None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, check=False,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if process.poll() is None:
            process.kill()

    def __call__(self, command: list[str], *, capture_output: bool = True,
                 text: bool = True, encoding: str = "utf-8", errors: str = "replace",
                 check: bool = False, timeout: float | None = None,
                 env: dict[str, str] | None = None, creationflags: int = 0,
                 **_ignored) -> subprocess.CompletedProcess[str]:
        if self._cancel.is_set():
            raise AlignmentCancelled("alignment_cancelled")
        process = subprocess.Popen(command, stdout=subprocess.PIPE if capture_output else None,
                                   stderr=subprocess.PIPE if capture_output else None,
                                   text=text, encoding=encoding, errors=errors, env=env,
                                   creationflags=creationflags)
        with self._lock:
            self._process = process
        started = monotonic()
        try:
            while True:
                if self._cancel.is_set():
                    self._terminate_tree(process)
                    raise AlignmentCancelled("alignment_cancelled")
                remaining = None if timeout is None else timeout - (monotonic() - started)
                if remaining is not None and remaining <= 0:
                    self._terminate_tree(process)
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    stdout, stderr = process.communicate(timeout=min(.25, remaining)
                                                         if remaining is not None else .25)
                    break
                except subprocess.TimeoutExpired:
                    continue
            result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            if self._cancel.is_set():
                raise AlignmentCancelled("alignment_cancelled")
            if check:
                result.check_returncode()
            return result
        finally:
            with self._lock:
                self._process = None
