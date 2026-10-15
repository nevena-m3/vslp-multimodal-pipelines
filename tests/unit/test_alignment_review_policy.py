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
                           "word": "WE", "start_sec": 1.1, "end_sec": 1.8}])
    phones = pd.DataFrame([{"recording_id": "r", "phone_index": 1,
                            "word_index": 1, "start_sec": 1.2, "end_sec": 1.7}])
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
    )) == ["INCOMPLETE_TRANSCRIPT_COVERAGE", "MISSING_REQUIRED_TOKEN"]


def test_token_outside_trial_requires_review():
    assert _flags(mutate=lambda _d, _t, _l, words, _p:
                  words.__setitem__("start_sec", [.999])) == [
                      "TRIAL_TOKEN_BOUNDARY_MISMATCH"]


def test_large_batch_requires_review_only_for_structural_exceptions():
    flags = [_flags(observed=2 if index < 66 else 3) for index in range(2000)]
    assert sum(not item for item in flags) == 1934
    assert sum(bool(item) for item in flags) == 66


def test_execution_failure_and_empty_tokens_require_review():
    assert "MFA_EXECUTION_FAILED" in _flags(mutate=lambda d, *_: d.update(
        {"status": "ALIGNMENT_FAILED"}))
    assert "MISSING_WORD_TOKENS" in _flags(mutate=lambda _d, _t, links, *_:
        links.drop(links.index[0], inplace=True))


def test_invalid_word_and_phone_boundaries_are_flagged():
    assert "INVALID_BOUNDARY" in _flags(mutate=lambda _d, _t, _l, words, _p:
        words.__setitem__("end_sec", [1.1]))
    assert "PHONE_WORD_RELATIONSHIP_INVALID" in _flags(
        mutate=lambda _d, _t, _l, _w, phones:
        phones.__setitem__("end_sec", [1.9]))


def test_nonmonotonic_words_and_unidentified_token_are_flagged():
    def reverse(_d, _t, links, words, _p):
        links.loc[len(links)] = ["r", "word", 2]
        words.loc[len(words)] = ["r", 2, "SEE", 1.0, 1.05]
    assert "NONMONOTONIC_WORDS" in _flags(mutate=reverse)
    assert "UNRESOLVED_OOV" in _flags(mutate=lambda _d, _t, _l, words, _p:
        words.__setitem__("word", ["<unk>"]))


def test_stale_prompt_trial_exclusion_and_oov_block_auto_acceptance():
    diagnostic, trial, links, words, phones = _case()
    exclusions = pd.DataFrame([{"start_sec": 1.3, "end_sec": 1.4}])
    flags = trial_exception_flags(diagnostic, {**trial, "prompt_id": "old"},
        expected_count=1, observed_count=1, transcript_overridden=False,
        links=links, words=words, phones=phones, working_sample_rate_hz=16000,
        trial_metadata={"trial_status": "STALE", "prompt_id": "old",
                        "segmentation_review_hash": "old"},
        expected_prompt_id="registered", current_review_hash="new",
        exclusions=exclusions, oov_words=["MISSING"])
    assert {"STALE_TRIAL", "STALE_PROMPT", "TRIAL_CROSSES_EXCLUSION",
            "UNRESOLVED_OOV"} <= set(flags)


def test_bamboo_scientific_word_count_is_separate_from_mfa_tokens():
    from hashlib import sha256

    from vslp.acoustic.alignment.stage import normalize_transcript
    from vslp.acoustic.alignment.task_workflow import task_entry

    prompt = task_entry("bamboo_passage")["prompts"][0]
    canonical = prompt["exact_expected_text"]
    assert prompt["prompt_id"] == "bamboo_passage_v1"
    assert prompt["prompt_version"] == "v1"
    assert prompt["expected_repetitions"] == 1
    assert prompt["scientific_word_count"] == 97
    assert prompt["scientific_word_count_rule"] == "whitespace_orthographic_good-looking_is_one_v1"
    assert prompt["prompt_sha256"] == sha256(canonical.encode("utf-8")).hexdigest()
    assert len(canonical.split()) == 97
    assert "good-looking" in canonical
    assert len(normalize_transcript(canonical).split()) == 98
