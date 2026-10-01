"""Deterministic structural triage; never a claim of boundary accuracy."""

from __future__ import annotations

import json

import pandas as pd


ALIGNMENT_REVIEW_POLICY_VERSION = "alignment_structural_review_v1"


def trial_exception_flags(diagnostic: dict, trial: dict, *, expected_count: int | None,
                          observed_count: int, transcript_overridden: bool,
                          links: pd.DataFrame, words: pd.DataFrame, phones: pd.DataFrame,
                          working_sample_rate_hz: int) -> list[str]:
    """Flag every deterministic condition that prevents structural auto-acceptance."""
    flags: list[str] = []
    if diagnostic.get("status") != "ALIGNED":
        flags.append(str(diagnostic.get("status") or "ALIGNMENT_FAILED"))
    if expected_count is not None and observed_count != expected_count:
        flags.append("PROTOCOL_TRIAL_COUNT_MISMATCH")
    if transcript_overridden:
        flags.append("TRANSCRIPT_CORRECTION_REQUIRED")
    if (float(diagnostic.get("word_coverage") or 0) < 1 or
            int(diagnostic.get("extra_tokens") or 0) or
            json.loads(diagnostic.get("missing_words") or "[]")):
        flags.append("INCOMPLETE_TRANSCRIPT_COVERAGE")
    tolerance = 0.5 / working_sample_rate_hz
    bounds = (float(trial["start_sec"]), float(trial["end_sec"]))
    for kind, table, index_name in (("word", words, "word_index"),
                                    ("phone", phones, "phone_index")):
        tier = links.loc[links.token_type.eq(kind)]
        if tier.empty:
            flags.append(f"MISSING_{kind.upper()}_TOKENS")
            continue
        source = table.set_index(["recording_id", index_name])
        for item in tier.itertuples():
            key = (str(item.recording_id), int(item.token_index))
            if key not in source.index:
                flags.append("TOKEN_IDENTITY_MISMATCH")
                continue
            token = source.loc[key]
            start, end = float(token.start_sec), float(token.end_sec)
            if not bounds[0] - tolerance <= start < end <= bounds[1] + tolerance:
                flags.append("TRIAL_TOKEN_BOUNDARY_MISMATCH")
    trial_words = links.loc[links.token_type.eq("word")]
    trial_phones = links.loc[links.token_type.eq("phone")]
    if not trial_words.empty and not trial_phones.empty and "word_index" in phones:
        identity = str(trial_words.recording_id.iloc[0])
        owned_words = words.loc[
            words.recording_id.astype(str).eq(identity)
            & words.word_index.isin(trial_words.token_index)]
        owned_phones = phones.loc[
            phones.recording_id.astype(str).eq(identity)
            & phones.phone_index.isin(trial_phones.token_index)]
        by_word = owned_words.set_index("word_index")
        for phone in owned_phones.itertuples():
            if int(phone.word_index) not in by_word.index:
                flags.append("PHONE_WORD_RELATIONSHIP_INVALID")
                continue
            word = by_word.loc[int(phone.word_index)]
            if not (float(word.start_sec) - tolerance <= float(phone.start_sec)
                    < float(phone.end_sec) <= float(word.end_sec) + tolerance):
                flags.append("PHONE_WORD_RELATIONSHIP_INVALID")
    return sorted(set(flags))
