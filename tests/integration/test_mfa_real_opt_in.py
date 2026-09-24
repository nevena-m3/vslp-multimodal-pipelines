"""Opt-in nonclinical MFA corpus smoke test; never downloads models or data."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import json

import pandas as pd
import pytest

from vslp.acoustic.alignment import (
    AlignmentConfig, freeze_alignment, list_alignment_runs, load_final_alignment,
    run_acoustic_alignment,
)
from vslp.acoustic.alignment.mfa_provider import MfaProfile, MfaProvider
from vslp.acoustic.features.stage import FeatureExtractionConfig, run_acoustic_feature_extraction


def test_real_mfa_corpus_freeze_reload_opt_in(tmp_path):
    fixture = os.environ.get("VSLP_REAL_ALIGNMENT_ROOT")
    profile_path = os.environ.get("VSLP_REAL_MFA_PROFILE")
    if not fixture or not profile_path:
        pytest.skip("Set VSLP_REAL_ALIGNMENT_ROOT and VSLP_REAL_MFA_PROFILE explicitly")
    profile = MfaProfile.load(profile_path)
    provider = MfaProvider()
    state = provider.inspect_environment(profile)
    if state["status"] != "AVAILABLE":
        pytest.skip(f"MFA environment unavailable: {state['status']}")
    root = tmp_path / "nonclinical_fixture"
    shutil.copytree(Path(fixture), root)
    decisions_path = (root / "acoustic" / "003_segmentation_review" / "final" /
                      "final_segmentation_decisions.csv")
    decisions = pd.read_csv(decisions_path)
    if len(decisions) < 2:
        pytest.fail("Real corpus fixture must contain at least two recordings")
    fixture_root = Path(fixture).resolve()
    relocated = []
    for original in decisions.analysis_wav_path:
        source = Path(original).resolve()
        if not source.is_relative_to(fixture_root):
            pytest.fail("Real fixture audio must be inside its nonclinical fixture root")
        relocated.append(str(root / source.relative_to(fixture_root)))
    decisions["analysis_wav_path"] = relocated
    decisions.to_csv(decisions_path, index=False)
    speakers = json.loads((root / "speakers.json").read_text(encoding="utf-8"))
    if len({item["speaker_id"] for item in speakers["recordings"]}) < 2:
        pytest.fail("Real corpus fixture must contain at least two logical speakers")
    result = run_acoustic_alignment(root, AlignmentConfig(
        source="mfa", prompt_manifest_path=str(root / "prompt.json"),
        speaker_manifest_path=str(root / "speakers.json"),
        mfa_profile_path=profile_path))
    assert result.status in {"completed", "completed_with_warnings"}
    run_id = list_alignment_runs(root)[-1]["alignment_run_id"]
    freeze_alignment(root, run_id)
    store, issue = load_final_alignment(root)
    assert issue == "" and store is not None
    assert len(store.words) > 0 and len(store.phones) > 0
    result = run_acoustic_feature_extraction(
        decisions_path, root,
        FeatureExtractionConfig(selected_features=["mean_word_duration_s", "f1_token_hz"]),
        root / "acoustic" / "003_segmentation_review" / "final" /
        "final_segmentation_intervals.csv")
    status_path = result.summary_table.parent / "acoustic_feature_status_long.csv"
    statuses = pd.read_csv(status_path)
    assert "mean_word_duration_s" in set(statuses.feature_id)
    token_path = (root / "acoustic" / "005_features" / "tables" /
                  "native_measurements" / "formant_token_measurements.csv")
    assert token_path.is_file()
