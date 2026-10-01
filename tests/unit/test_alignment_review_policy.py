"""Engineering checks for structural triage, without boundary-accuracy claims."""

import json

import pandas as pd

from vslp.acoustic.alignment.review_policy import trial_exception_flags


def _case():
    trial = {"start_sec": 1.0, "end_sec": 2.0}
    diagnostic = {"status": "ALIGNED", "word_coverage": 1.0,
                  "extra_tokens": 0, "missing_words": "[]"}
    links = pd.DataFrame([
        {"recording_id": "r", "token_type": "word", "token_index": 1},
        {"recording_id": "r", "token_type": "phone", "token_index": 1},
    ])
    words = pd.DataFrame([{"recording_id": "r", "word_index": 1,
                           "start_sec": 1.1, "end_sec": 1.8}])
    phones = pd.DataFrame([{"recording_id": "r", "phone_index": 1,
                            "start_sec": 1.2, "end_sec": 1.7}])
    return diagnostic, trial, links, words, phones


def _flags(expected=3, observed=3, override=False, mutate=None):
    diagnostic, trial, links, words, phones = _case()
    if mutate:
        mutate(diagnostic, trial, links, words, phones)
    return trial_exception_flags(
        diagnostic, trial, expected_count=expected, observed_count=observed,
        transcript_overridden=override, links=links, words=words, phones=phones,
        working_sample_rate_hz=16000)


def test_clear_case_has_no_structural_exception():
    assert _flags() == []


def test_expected_count_flags_both_missing_and_extra_without_fabrication():
    assert _flags(observed=2) == ["PROTOCOL_TRIAL_COUNT_MISMATCH"]
    assert _flags(observed=5) == ["PROTOCOL_TRIAL_COUNT_MISMATCH"]


def test_transcript_coverage_and_correction_require_review():
    assert _flags(override=True) == ["TRANSCRIPT_CORRECTION_REQUIRED"]
    assert _flags(mutate=lambda d, *_: d.update(
        {"word_coverage": .75, "missing_words": json.dumps(["GEESE"])}
    )) == ["INCOMPLETE_TRANSCRIPT_COVERAGE"]


def test_token_outside_trial_requires_review():
    assert _flags(mutate=lambda _d, _t, _l, words, _p:
                  words.__setitem__("start_sec", [.999])) == [
                      "TRIAL_TOKEN_BOUNDARY_MISMATCH"]


def test_large_batch_requires_review_only_for_structural_exceptions():
    flags = [_flags(observed=2 if index < 66 else 3) for index in range(2000)]
    assert sum(not item for item in flags) == 1934
    assert sum(bool(item) for item in flags) == 66
