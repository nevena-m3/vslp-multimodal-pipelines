"""Environment diagnostics for VSLP.

The goal is to give non-expert users a clear yes/no setup check before running
pipeline stages.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def _cmd_version(cmd: str, args: list[str]) -> CheckResult:
    exe = shutil.which(cmd)
    if exe is None:
        return CheckResult(
            name=cmd,
            ok=False,
            detail=f"{cmd} not found on PATH",
            fix="Install FFmpeg, then reopen your terminal. On macOS: brew install ffmpeg",
        )
    try:
        p = subprocess.run([exe, *args], capture_output=True, text=True, timeout=10, check=False)
        first = (p.stdout or p.stderr).splitlines()[0] if (p.stdout or p.stderr) else exe
        return CheckResult(name=cmd, ok=True, detail=first)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name=cmd, ok=False, detail=str(exc), fix="Check that FFmpeg is installed correctly.")


def _module(name: str, install_hint: str) -> CheckResult:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return CheckResult(name=name, ok=False, detail=f"Python module '{name}' not installed", fix=install_hint)
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "installed")
        return CheckResult(name=name, ok=True, detail=str(version))
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name=name, ok=False, detail=f"Module found but import failed: {exc}", fix=install_hint)


def run_doctor(include_silero: bool = True) -> list[CheckResult]:
    checks: list[CheckResult] = []
    py_ok = sys.version_info[:2] == (3, 11)
    checks.append(
        CheckResult(
            name="python",
            ok=py_ok,
            detail=f"{platform.python_version()} at {sys.executable}",
            fix="Use Python 3.11 for this project.",
        )
    )
    checks.append(_cmd_version("ffmpeg", ["-version"]))
    checks.append(_cmd_version("ffprobe", ["-version"]))

    required = ["numpy", "pandas", "scipy", "soundfile", "matplotlib", "pydantic", "typer"]
    for name in required:
        checks.append(_module(name, "Run: pip install -e ."))

    if include_silero:
        checks.append(_module("torch", "Run: pip install -e '.[silero]'"))
        checks.append(_module("torchaudio", "Run: pip install -e '.[silero]'"))

    return checks
