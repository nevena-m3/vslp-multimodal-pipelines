"""Deterministic structural triage; never a claim of boundary accuracy."""

from __future__ import annotations

import json
import math

import pandas as pd


ALIGNMENT_REVIEW_POLICY_VERSION = "alignment_structural_review_v2"


def trial_exception_flags(diagnostic: dict, trial: dict, *, expected_count: int | None,
                          observed_count: int, transcript_overridden: bool,
                          links: pd.DataFrame, words: pd.DataFrame, phones: pd.DataFrame,
                          working_sample_rate_hz: int,
                          trial_metadata: dict | None = None,
                          expected_prompt_id: str = "",
                          expected_prompt_version: str = "",
                          exclusions: pd.DataFrame | None = None,
                          oov_words: list[str] | None = None,
                          current_review_hash: str = "") -> list[str]:
    """Structural coherence only; never a claim of dysarthric boundary accuracy."""
    flags: list[str] = []
    if diagnostic.get("status") != "ALIGNED":
        flags.append("MFA_EXECUTION_FAILED" if diagnostic.get("status") in {
            None, "", "ALIGNMENT_FAILED"} else str(diagnostic["status"]))
    metadata = trial_metadata or {}
    if metadata and metadata.get("trial_status") != "READY_FOR_ALIGNMENT":
        flags.append("STALE_TRIAL")
    if (metadata.get("segmentation_review_hash") and current_review_hash and
            metadata["segmentation_review_hash"] != current_review_hash):
        flags.append("STALE_TRIAL")
    if (expected_prompt_id and (trial.get("prompt_id") != expected_prompt_id or
            metadata.get("prompt_id", expected_prompt_id) != expected_prompt_id)):
        flags.append("STALE_PROMPT")
    if (expected_prompt_version and metadata.get("prompt_version", expected_prompt_version)
            != expected_prompt_version):
        flags.append("STALE_PROMPT")
    if oov_words:
        flags.append("UNRESOLVED_OOV")
    if expected_count is not None and observed_count != expected_count:
        flags.append("PROTOCOL_TRIAL_COUNT_MISMATCH")
    if transcript_overridden:
        flags.append("TRANSCRIPT_CORRECTION_REQUIRED")
    if (float(diagnostic.get("word_coverage") or 0) < 1 or
            int(diagnostic.get("extra_tokens") or 0) or
            json.loads(diagnostic.get("missing_words") or "[]")):
        flags.append("INCOMPLETE_TRANSCRIPT_COVERAGE")
    if json.loads(diagnostic.get("missing_words") or "[]"):
        flags.append("MISSING_REQUIRED_TOKEN")
    if int(diagnostic.get("extra_tokens") or 0):
        flags.append("UNEXPECTED_TOKEN")
    tolerance = 0.5 / working_sample_rate_hz
    bounds = (float(trial["start_sec"]), float(trial["end_sec"]))
    if not all(map(math.isfinite, bounds)) or bounds[0] >= bounds[1]:
        flags.append("INVALID_BOUNDARY")
    if exclusions is not None and not exclusions.empty and any(
            float(row.start_sec) < bounds[1] and float(row.end_sec) > bounds[0]
            for row in exclusions.itertuples()):
        flags.append("TRIAL_CROSSES_EXCLUSION")
    for kind, table, index_name in (("word", words, "word_index"),
                                    ("phone", phones, "phone_index")):
        tier = links.loc[links.token_type.eq(kind)]
        if tier.empty:
            flags.append(f"MISSING_{kind.upper()}_TOKENS")
            if kind == "word":
                flags.append("EMPTY_ALIGNMENT")
            continue
        source = table.set_index(["recording_id", index_name])
        for item in tier.itertuples():
            key = (str(item.recording_id), int(item.token_index))
            if key not in source.index:
                flags.append("TOKEN_IDENTITY_MISMATCH")
                continue
            token = source.loc[key]
            start, end = float(token.start_sec), float(token.end_sec)
            if not all(map(math.isfinite, (start, end))) or end <= start:
                flags.append("INVALID_BOUNDARY")
                continue
            if not bounds[0] - tolerance <= start < end <= bounds[1] + tolerance:
                flags.append("TRIAL_TOKEN_BOUNDARY_MISMATCH")
        ordered = tier.sort_values("token_index")
        times = []
        for item in ordered.itertuples():
            key = (str(item.recording_id), int(item.token_index))
            if key in source.index:
                token = source.loc[key]
                times.append((float(token.start_sec), float(token.end_sec)))
        if any(math.isfinite(left[0]) and math.isfinite(right[0]) and
               (right[0] < left[0] - tolerance or right[0] < left[1] - tolerance)
               for left, right in zip(times, times[1:])):
            flags.append(f"NONMONOTONIC_{kind.upper()}S")
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
    if "word" in words:
        owned = words.loc[words.recording_id.astype(str).isin(
            trial_words.recording_id.astype(str)) &
            words.word_index.isin(trial_words.token_index)]
        if owned.word.astype(str).str.casefold().isin({"<unk>", "unk", "<unknown>"}).any():
            flags.append("UNRESOLVED_OOV")
    return sorted(set(flags))
