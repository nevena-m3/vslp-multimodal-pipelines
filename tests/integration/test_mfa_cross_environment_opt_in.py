"""Opt-in real VSLP Python -> Conda -> MFA engineering check."""

from __future__ import annotations

import os

import pytest

from vslp.acoustic.alignment.mfa_provider import MfaProfile, MfaProvider
from vslp.acoustic.alignment.profiles import default_profile_path
from vslp.acoustic.alignment.self_test import run_mfa_self_test


def test_real_cross_environment_self_test():
    if os.environ.get("VSLP_RUN_REAL_MFA") != "1":
        pytest.skip("Set VSLP_RUN_REAL_MFA=1 to use installed Conda MFA")
    profile = MfaProfile.load(default_profile_path())
    state = MfaProvider().inspect_environment(profile)
    assert state["status"] == "AVAILABLE"
    assert state["version"] == "3.3.4"
    assert state["launcher_type"] == "CONDA_ENV"
    assert state["resources"]["acoustic"]["identity"] == "english_us_arpa"
    assert state["resources"]["dictionary"]["identity"] == "english_us_arpa"
    result = run_mfa_self_test(default_profile_path())
    assert result.result == "PASS", result.failure_message
