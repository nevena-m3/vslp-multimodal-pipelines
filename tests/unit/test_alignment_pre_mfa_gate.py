"""Synthetic admission checks before any MFA corpus is created."""

import json

import pandas as pd
import pytest

from vslp.acoustic.alignment.stage import AlignmentConfig, _trial_units


@pytest.mark.parametrize("state", ["PROPOSED", "NEEDS_TRIAL_REVIEW", "STALE", "OMITTED"])
def test_unready_trial_cannot_enter_mfa(state, tmp_path):
    manifest = tmp_path / "trials.json"
    manifest.write_text(json.dumps({"schema_version": "1", "trials": [{
        "recording_id": "synthetic_recording", "trial_id": "trial_001",
        "start_sec": 1.0, "end_sec": 2.0, "trial_status": state,
    }]}), encoding="utf-8")
    kept = pd.DataFrame([{"recording_id": "synthetic_recording",
                          "analysis_start_sec": 0.0, "analysis_end_sec": 3.0}])
    with pytest.raises(ValueError, match="TRIAL_NOT_READY_FOR_MFA"):
        _trial_units(AlignmentConfig(source="mfa", trial_manifest_path=str(manifest)), kept)


def test_ready_trial_can_enter_mfa(tmp_path):
    manifest = tmp_path / "trials.json"
    manifest.write_text(json.dumps({"schema_version": "1", "trials": [{
        "recording_id": "synthetic_recording", "trial_id": "trial_001",
        "start_sec": 1.0, "end_sec": 2.0, "trial_status": "READY_FOR_ALIGNMENT",
        "trial_review_source": "HUMAN",
    }]}), encoding="utf-8")
    kept = pd.DataFrame([{"recording_id": "synthetic_recording",
                          "analysis_start_sec": 0.0, "analysis_end_sec": 3.0}])
    units = _trial_units(AlignmentConfig(source="mfa", trial_manifest_path=str(manifest)), kept)
    assert units["synthetic_recording"][0]["trial_id"] == "trial_001"


def test_mfa_requires_explicit_ready_trial_manifest():
    kept = pd.DataFrame([{"recording_id": "synthetic_recording",
                          "analysis_start_sec": 0.0, "analysis_end_sec": 3.0}])
    with pytest.raises(ValueError, match="READY_TRIAL_MANIFEST_REQUIRED"):
        _trial_units(AlignmentConfig(source="mfa"), kept)


@pytest.mark.parametrize("legacy,admitted", [
    ("HUMAN_CONFIRMED", True), ("HUMAN_ACCEPTED", True),
    ("AUTO_ACCEPTED_STRUCTURAL", False),
])
def test_legacy_pre_mfa_state_mapping_is_explicit(tmp_path, legacy, admitted):
    manifest = tmp_path / "trials.json"
    manifest.write_text(json.dumps({"schema_version": "1", "trials": [{
        "recording_id": "synthetic_recording", "trial_id": "trial_001",
        "start_sec": 1.0, "end_sec": 2.0, "review_status": legacy,
    }]}), encoding="utf-8")
    kept = pd.DataFrame([{"recording_id": "synthetic_recording",
                          "analysis_start_sec": 0.0, "analysis_end_sec": 3.0}])
    config = AlignmentConfig(source="mfa", trial_manifest_path=str(manifest))
    if admitted:
        assert _trial_units(config, kept)["synthetic_recording"][0]["trial_id"] == "trial_001"
    else:
        with pytest.raises(ValueError, match="TRIAL_NOT_READY_FOR_MFA"):
            _trial_units(config, kept)
