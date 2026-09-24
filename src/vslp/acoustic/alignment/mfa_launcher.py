"""Launch MFA outside the VSLP interpreter, including isolated Conda environments."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess


class LauncherError(RuntimeError):
    """A configured external MFA environment cannot be launched."""


def find_conda(configured: str = "") -> str | None:
    candidates = [configured, os.environ.get("CONDA_EXE", ""), shutil.which("conda") or ""]
    if os.name == "nt":
        for root in (os.environ.get("LOCALAPPDATA", ""), os.environ.get("USERPROFILE", "")):
            for folder in ("miniconda3", "anaconda3", "Miniconda3", "Anaconda3"):
                if root:
                    candidates.append(str(Path(root) / folder / "Scripts" / "conda.exe"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    return None


class MfaLauncher:
    def __init__(self, launcher_type: str = "CURRENT_PATH", conda_environment: str = "",
                 conda_executable: str = "", mfa_executable: str = "",
                 runner=subprocess.run) -> None:
        self.launcher_type = launcher_type.upper()
        self.conda_environment = conda_environment
        self.conda_executable = conda_executable
        self.mfa_executable = mfa_executable
        self.runner = runner
        self.environment_path = ""
        self.resolved_conda = ""
        self.resolved_mfa = ""

    def resolve(self) -> list[str]:
        if self.launcher_type == "CONDA_ENV":
            conda = find_conda(self.conda_executable)
            if not conda:
                raise LauncherError("CONDA_NOT_FOUND")
            self.resolved_conda = conda
            result = self.runner([conda, "env", "list", "--json"], capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", check=False,
                                 timeout=60,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode:
                raise LauncherError("CONDA_ENVIRONMENT_QUERY_FAILED")
            paths = json.loads(result.stdout).get("envs", [])
            selected = next((Path(path) for path in paths
                             if Path(path).name.casefold() == self.conda_environment.casefold()), None)
            if selected is None:
                raise LauncherError("MFA_ENVIRONMENT_NOT_FOUND")
            self.environment_path = str(selected)
            executable = selected / ("Scripts/mfa.exe" if os.name == "nt" else "bin/mfa")
            if not executable.is_file():
                raise LauncherError("MFA_NOT_INSTALLED_IN_ENVIRONMENT")
            self.resolved_mfa = str(executable)
            return [conda, "run", "--no-capture-output", "-n", self.conda_environment, "mfa"]
        if self.launcher_type == "EXPLICIT_EXECUTABLE":
            if not self.mfa_executable:
                raise LauncherError("MFA_NOT_INSTALLED")
            self.resolved_mfa = self.mfa_executable
            return [self.mfa_executable]
        if self.launcher_type == "CURRENT_PATH":
            executable = self.mfa_executable or shutil.which("mfa")
            if not executable:
                raise LauncherError("MFA_NOT_INSTALLED")
            self.resolved_mfa = executable
            return [executable]
        raise LauncherError("INVALID_MFA_LAUNCHER")
