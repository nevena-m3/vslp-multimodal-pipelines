"""Real external MFA self-test runs only with explicit local configuration."""

from __future__ import annotations

import os

import pytest

from vslp.acoustic.alignment.self_test import run_mfa_self_test


def test_real_mfa_self_test_opt_in():
    if os.environ.get("VSLP_RUN_REAL_MFA_SELF_TEST") != "1":
        pytest.skip("Set VSLP_RUN_REAL_MFA_SELF_TEST=1 to execute real MFA")
    profile = os.environ.get("VSLP_REAL_MFA_PROFILE", "")
    assert profile, "VSLP_REAL_MFA_PROFILE is required for the opt-in real test"
    result = run_mfa_self_test(profile, os.environ.get("VSLP_NONCLINICAL_TEST_WAV") or None)
    assert result.result == "PASS", result.failure_message
