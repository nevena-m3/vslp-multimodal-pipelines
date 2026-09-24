"""Alignment contracts do not depend on an installed MFA executable."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from vslp.acoustic.alignment.stage import (
    PHONE_SET, AlignmentConfig, MFAProvider, PromptManifest, TimePiece,
    _coverage, _parse_textgrid, _validate_order,
    map_interval, normalize_phone,
)


def test_prompt_must_be_explicit_and_versioned(tmp_path):
    path = tmp_path / "prompt.json"
    path.write_text(json.dumps({"manifest_version": "1", "task_id": "bamboo",
                                "prompt_version": "v1", "language": "en",
                                "transcript": "Buy a puppy", "expected_words": ["Buy", "a", "puppy"],
                                "phone_set": PHONE_SET}), encoding="utf-8")
    assert PromptManifest.load(path).transcript == "Buy a puppy"
    path.write_text(json.dumps({"manifest_version": "1", "task_id": "bamboo",
                                "prompt_version": "v1", "language": "en",
                                "transcript": "Buy a puppy", "expected_words": ["Wrong"],
                                "phone_set": PHONE_SET}), encoding="utf-8")
    with pytest.raises(ValueError, match="PROMPT_MISMATCH"):
        PromptManifest.load(path)
    path.write_text(json.dumps({"task_id": "bamboo"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing_prompt_fields"):
        PromptManifest.load(path)


def test_phone_classification_is_phone_set_specific():
    assert normalize_phone("AA1") == ("AA", "1", True)
    assert normalize_phone("T") == ("T", "", False)
    with pytest.raises(ValueError, match="unsupported_phone_set"):
        normalize_phone("AA1", "unknown")
    with pytest.raises(ValueError, match="PHONESET_MAPPING_REQUIRED"):
        normalize_phone("XYZ")


def test_time_map_rejects_token_bridge_over_contamination():
    pieces = [TimePiece(0, 1, 2, 3, 1), TimePiece(1, 2, 4, 5, 2)]
    assert map_interval(1.2, 1.4, pieces) == pytest.approx((4.2, 4.4, 2))
    with pytest.raises(ValueError, match="token_crosses_excluded_gap"):
        map_interval(.9, 1.1, pieces)


def test_coverage_handles_missing_middle_word_and_repetition():
    words = [{"word": "the"}, {"word": "cat"}, {"word": "the"}]
    matched, missing, extra = _coverage(words, ("the", "small", "cat", "the"))
    assert (matched, missing, extra) == (3, ["small"], 0)


def test_mfa_long_textgrid_parser(tmp_path):
    path = tmp_path / "x.TextGrid"
    path.write_text('''File type = "ooTextFile"
item [1]:
    name = "words"
    intervals [1]:
        xmin = 0.1
        xmax = 0.4
        text = "buy"
item [2]:
    name = "phones"
    intervals [1]:
        xmin = 0.1
        xmax = 0.4
        text = "AY1"
''', encoding="utf-8")
    words, phones = _parse_textgrid(path)
    assert isinstance(words, pd.DataFrame)
    assert words.iloc[0].label == "buy"
    assert phones.iloc[0].label == "AY1"


def test_mfa_unavailable_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr("vslp.acoustic.alignment.mfa_provider.shutil.which", lambda _: None)
    assert MFAProvider().detect()["status"] == "MFA_NOT_INSTALLED"


def test_overlap_and_order_diagnostics():
    assert _validate_order([{"start_sec": 1, "end_sec": 2},
                            {"start_sec": 1.5, "end_sec": 3}]) == (1, 0)
    assert _validate_order([{"start_sec": 2, "end_sec": 3},
                            {"start_sec": 1, "end_sec": 1.5}]) == (0, 1)
