"""Provider-independent, original-time linguistic token contract.

The legacy frozen word and phone CSVs contain provider tokens.  This module
creates separate scientific lexical units and never guesses an unmatched word.
"""

from __future__ import annotations

from hashlib import sha256
import json

import pandas as pd


FINAL_ALIGNMENT_CONTRACT_VERSION = "FINAL_ALIGNMENT_CONTRACT_V1"


def canonical_units(text: str) -> list[str]:
    """Keep hyphenated lexical words together; punctuation is not a word."""
    import re

    return re.findall(r"[\w]+(?:[-’'][\w]+)*", text)


def build_feature_tables(
    trials: pd.DataFrame,
    links: pd.DataFrame,
    provider_words: pd.DataFrame,
    provider_phones: pd.DataFrame,
    reviews: pd.DataFrame,
    prompts: dict[str, dict],
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map final provider tokens to canonical words, or require review.

    A canonical word may normalize to several provider tokens.  The mapping
    must match every provider token exactly and in order within each trial.
    """
    from .stage import normalize_transcript

    word_rows: list[dict] = []
    phone_rows: list[dict] = []
    trial_rows: list[dict] = []
    review_by_key = {(str(r.recording_id), str(r.trial_id)): r
                     for r in reviews.itertuples()}
    for trial in trials.itertuples():
        recording_id, trial_id = str(trial.recording_id), str(trial.trial_id)
        key = recording_id, trial_id
        if key not in review_by_key:
            raise ValueError("final_alignment_review_missing")
        review = review_by_key[key]
        status = str(getattr(review, "alignment_review_status", ""))
        if status not in {"AUTO_ACCEPTED_STRUCTURAL", "HUMAN_ACCEPTED"}:
            raise ValueError("final_alignment_unresolved")
        prompt_id = str(trial.prompt_id)
        prompt = prompts.get(prompt_id) or prompts.get(recording_id)
        if prompt is None or str(prompt["prompt_version"]) != str(getattr(trial, "prompt_version", prompt["prompt_version"])):
            raise ValueError("final_alignment_prompt_mismatch")
        text = str(prompt.get("exact_expected_text", prompt.get("transcript", "")))
        if not text:
            raise ValueError("final_alignment_prompt_missing_text")
        units = canonical_units(text)
        sequence_id = f"{recording_id}:{trial_id}"
        common = {"recording_id": recording_id, "trial_id": trial_id,
                  "sequence_id": sequence_id, "task_id": str(prompt["task_id"]),
                  "task_item": str(getattr(trial, "task_item", "")),
                  "prompt_id": prompt_id, "prompt_version": str(prompt["prompt_version"]),
                  "alignment_review_status": status, "alignment_run_id": run_id,
                  "alignment_contract_version": FINAL_ALIGNMENT_CONTRACT_VERSION}
        trial_rows.append({**common, "trial_index": len(trial_rows) + 1,
                           "start_sec": float(trial.start_sec),
                           "end_sec": float(trial.end_sec),
                           "prompt_sha256": sha256(text.encode("utf-8")).hexdigest(),
                           "production_structure": str(getattr(trial, "production_structure", "")),
                           "protocol_expected_repetitions": getattr(
                               trial, "protocol_expected_repetitions", ""),
                           "observed_trial_count": getattr(trial, "observed_candidate_count", ""),
                           "alignment_source_summary": str(getattr(review, "final_status", "")),
                           "qc_flags": str(getattr(review, "flags_triggered", "")),
                           "segmentation_review_hash": str(getattr(trial, "segmentation_review_hash", "")),
                           "trial_manifest_hash": ""})
        word_links = links.loc[links.recording_id.astype(str).eq(recording_id)
                               & links.trial_id.astype(str).eq(trial_id)
                               & links.token_type.eq("word")].sort_values("token_index")
        selected_words = provider_words.loc[
            provider_words.recording_id.astype(str).eq(recording_id)
            & provider_words.word_index.isin(word_links.token_index)]
        tokens = selected_words.set_index("word_index")
        ordered = [tokens.loc[int(index)] for index in word_links.token_index]
        expected = [part for unit in units for part in normalize_transcript(unit).split()]
        actual = [normalize_transcript(str(row.word)) for row in ordered]
        if expected != actual or len(ordered) != len(word_links):
            raise ValueError(f"canonical_provider_token_mapping_requires_review:{recording_id}:{trial_id}")
        cursor = 0
        trial_phone_index = 0
        for word_index, unit in enumerate(units, 1):
            count = len(normalize_transcript(unit).split())
            parts = ordered[cursor:cursor + count]
            provider_ids = [int(index) for index in word_links.token_index.iloc[
                cursor:cursor + count]]
            word_id = f"{sequence_id}:word_{word_index:03d}"
            source = ("REVIEWED_MANUAL_CORRECTION" if any(
                str(row.alignment_source) == "reviewed_manual_correction" for row in parts)
                else str(parts[0].alignment_source).upper())
            word_rows.append({**common, "word_id": word_id,
                              "canonical_word_index": word_index,
                              "canonical_word_text": unit,
                              "word_normalized": normalize_transcript(unit),
                              "provider_token_ids": json.dumps(provider_ids),
                              "provider_token_count": count,
                              "word_index": word_index, "word": unit,
                              "start_sec": float(parts[0].start_sec),
                              "end_sec": float(parts[-1].end_sec),
                              "duration_sec": float(parts[-1].end_sec) - float(parts[0].start_sec),
                              "boundary_source": source,
                              "alignment_source": source,
                              "qc_flags": str(getattr(review, "flags_triggered", ""))})
            phones = provider_phones.loc[
                provider_phones.recording_id.astype(str).eq(recording_id)
                & provider_phones.word_index.isin(provider_ids)].sort_values("phone_index")
            for local_index, phone in enumerate(phones.itertuples(), 1):
                trial_phone_index += 1
                phone_source = ("REVIEWED_MANUAL_CORRECTION" if
                                str(phone.alignment_source) == "reviewed_manual_correction" else
                                str(phone.alignment_source).upper())
                phone_rows.append({**common,
                                   "phone_id": f"{sequence_id}:phone_{int(phone.phone_index):04d}",
                                   "phone_index_in_trial": trial_phone_index,
                                   "phone_index_in_word": local_index,
                                   "parent_word_id": word_id,
                                   "parent_word_index": word_index,
                                   "parent_word_text": unit,
                                   "word_index": word_index,
                                   "phone_index": int(phone.phone_index),
                                   "word": unit,
                                   "phone_raw": str(phone.phone_raw),
                                   "phone": str(phone.phone_raw),
                                   "phone_normalized": str(phone.phone_normalized),
                                   "stress": str(phone.stress),
                                   "is_vowel": str(phone.is_vowel).lower() in {"true", "1"},
                                   "start_sec": float(phone.start_sec),
                                   "end_sec": float(phone.end_sec),
                                   "duration_sec": float(phone.end_sec) - float(phone.start_sec),
                                   "boundary_source": phone_source,
                                   "alignment_source": phone_source,
                                   "qc_flags": str(getattr(review, "flags_triggered", ""))})
            cursor += count
    return pd.DataFrame(trial_rows), pd.DataFrame(word_rows), pd.DataFrame(phone_rows)
