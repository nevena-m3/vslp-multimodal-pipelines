"""Provenance helpers: hashes, versions, command capture, and environment records."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 for a file without loading the full file into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def run_command(command: Iterable[str], timeout: int = 30) -> dict[str, str | int | bool]:
    """Run a diagnostic command and capture stdout/stderr without raising."""
    command = [str(c) for c in command]
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
            "command": " ".join(command),
        }
    except Exception as exc:  # noqa: BLE001 - provenance must never crash the run
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": str(exc), "command": " ".join(command)}


def python_environment() -> dict[str, str]:
    """Minimal reproducibility environment snapshot."""
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def tool_versions(ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict[str, dict[str, str | int | bool]]:
    """Capture FFmpeg/FFprobe version diagnostics."""
    return {
        "ffmpeg": run_command([ffmpeg, "-version"]),
        "ffprobe": run_command([ffprobe, "-version"]),
    }
