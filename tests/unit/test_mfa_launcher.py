"""External MFA launcher never imports MFA into the VSLP interpreter."""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from vslp.acoustic.alignment.mfa_launcher import LauncherError, MfaLauncher
from vslp.acoustic.alignment.mfa_provider import MfaProfile
from vslp.acoustic.alignment.profiles import default_profile_path


def test_conda_environment_launch_prefix(tmp_path):
    conda = tmp_path / "conda.exe"
    conda.write_bytes(b"stub")
    environment = tmp_path / "envs" / "vslp-mfa-334"
    mfa = environment / ("Scripts/mfa.exe" if os.name == "nt" else "bin/mfa")
    mfa.parent.mkdir(parents=True)
    mfa.write_bytes(b"stub")
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps({"envs": [str(environment)]}), "")

    launcher = MfaLauncher("CONDA_ENV", "vslp-mfa-334", str(conda), runner=runner)
    assert launcher.resolve() == [str(conda), "run", "--no-capture-output", "-n", "vslp-mfa-334", "mfa"]
    assert calls[0][-3:] == ["env", "list", "--json"]


def test_missing_conda_and_missing_environment(tmp_path, monkeypatch):
    monkeypatch.setattr("vslp.acoustic.alignment.mfa_launcher.find_conda", lambda _configured: None)
    with pytest.raises(LauncherError, match="CONDA_NOT_FOUND"):
        MfaLauncher("CONDA_ENV", "absent").resolve()
    monkeypatch.undo()
    conda = tmp_path / "conda.exe"
    conda.write_bytes(b"stub")
    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, '{"envs": []}', "")
    with pytest.raises(LauncherError, match="MFA_ENVIRONMENT_NOT_FOUND"):
        MfaLauncher("CONDA_ENV", "absent", str(conda), runner=runner).resolve()


def test_packaged_engineering_profile_uses_isolated_environment():
    profile = MfaProfile.load(default_profile_path())
    assert profile.launcher_type == "CONDA_ENV"
    assert profile.conda_environment == "vslp-mfa-334"
    assert profile.supported_mfa_version == "3.3.4"
    assert profile.acoustic_model == profile.dictionary == "english_us_arpa"
