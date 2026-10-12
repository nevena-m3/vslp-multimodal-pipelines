"""Frozen F07 measurements from reviewed timing and Final Alignment V1.

No provider audio, TextGrid, or MFA process is used here. All times are
original-recording seconds. A result is a value and an explicit reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vslp.acoustic.alignment.reviewed_trials import load_confirmed_trials


VERSION = "family07-reviewed-v2.0.0"
MIN_VOWEL_SEC = 0.030
SHORT_VOWEL_SEC = 0.050
MIN_BAMBOO_VOWELS = 30
S3_PROMPT_ID = "sentence_s3_blue_spot_key"
# Highest-weight english_us_arpa variants for the/blue/spot/is/on/the/key/again.
S3_EXPECTED_VOWELS = ("AH", "UW", "AA", "IH", "AA", "AH", "IY", "AH", "EH")
S3_EXPECTED_WORD_VOWELS = (("the", "AH"), ("blue", "UW"), ("spot", "AA"),
                           ("is", "IH"), ("on", "AA"), ("the", "AH"),
                           ("key", "IY"), ("again", "AH"), ("again", "EH"))
ARPABET_VOWELS = frozenset({"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
                            "IH", "IY", "OW", "OY", "UH", "UW"})
TIMING_IDS = frozenset({"bamboo_speech_duration_s", "bamboo_total_duration_s",
                        "s3_sentence_duration_s"})
ALIGNMENT_IDS = frozenset({"s3_vocalic_duration_mean_s", "s3_vocalic_duration_median_s",
                           "s3_vocalic_duration_sd_s", "s3_vocalic_duration_iqr_s",
                           "delta_v_s", "varco_v_pct", "npvi_v_pct",
                           "s3_vocalic_proportion_pct"})
FAMILY07_IDS = TIMING_IDS | ALIGNMENT_IDS
TOKEN_COLUMNS = ("recording_id", "trial_id", "sequence_id", "task_id", "word_id",
                 "word", "phone_id", "phone_raw", "phone_normalized",
                 "phone_index_in_trial", "start_sec", "end_sec", "duration_sec",
                 "short_vowel_review_flag", "alignment_review_status", "boundary_source",
                 "valid", "rejection_reason")


def _result(value: float = np.nan, reason: str = "") -> tuple[float, str]:
    return float(value), reason


def _bounds(frame: pd.DataFrame, role: str) -> list[tuple[float, float]]:
    return [(float(row.start_sec), float(row.end_sec)) for row in
            frame.loc[frame.segment_role.astype(str).eq(role)].itertuples()]


def _eligible_speech(speech: list[tuple[float, float]],
                     exclusions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Subtract hard exclusions without joining the audio on either side."""
    pieces = []
    for start, end in speech:
        parts = [(start, end)]
        for excluded_start, excluded_end in sorted(exclusions):
            next_parts = []
            for left, right in parts:
                if excluded_end <= left or excluded_start >= right:
                    next_parts.append((left, right))
                else:
                    if left < excluded_start:
                        next_parts.append((left, excluded_start))
                    if excluded_end < right:
                        next_parts.append((excluded_end, right))
            parts = next_parts
        pieces.extend(parts)
    return pieces


def reviewed_timing(intervals: pd.DataFrame, task_id: str) -> tuple[dict[str, tuple[float, str]], pd.DataFrame]:
    """C044 sums eligible speech; C045 spans its first through last boundary."""
    required = {"segment_role", "start_sec", "end_sec"}
    if not required <= set(intervals) or intervals.empty:
        return {}, pd.DataFrame()
    frame = (intervals.loc[intervals["view"].astype(str).eq("authoritative")].copy()
             if "view" in intervals else intervals.copy())
    audit = frame.copy()
    audit["duration_sec"] = pd.to_numeric(audit.end_sec, errors="coerce") - pd.to_numeric(
        audit.start_sec, errors="coerce")
    audit["boundary_source"] = audit.get("boundary_source", "FINAL_REVIEWED_SEGMENTATION")
    if task_id != "bamboo_passage":
        return {}, audit
    if (not np.isfinite(audit[["start_sec", "end_sec"]].to_numpy(dtype=float)).all()
            or (audit.duration_sec <= 0).any()):
        return {feature: _result(reason="invalid_speech_interval") for feature in TIMING_IDS
                if feature.startswith("bamboo_")}, audit
    # Frozen review is authoritative; hard exclusions subtract from speech.
    speech = _bounds(frame, "speech")
    exclusions = _bounds(frame, "manual_exclusion")
    if not speech:
        return {feature: _result(reason="no_valid_speech_intervals") for feature in TIMING_IDS
                if feature.startswith("bamboo_")}, audit
    ordered_speech = sorted(speech)
    if any(right > next_left for (_left, right), (next_left, _next_right) in
           zip(ordered_speech, ordered_speech[1:], strict=False)):
        return {feature: _result(reason="invalid_speech_interval") for feature in TIMING_IDS
                if feature.startswith("bamboo_")}, audit
    eligible = _eligible_speech(speech, exclusions)
    if not eligible:
        return {feature: _result(reason="no_valid_speech_intervals") for feature in TIMING_IDS
                if feature.startswith("bamboo_")}, audit
    eligible_audit = pd.DataFrame([{"segment_role": "eligible_speech", "start_sec": a,
                                    "end_sec": b, "duration_sec": b - a,
                                    "boundary_source": "FINAL_REVIEWED_SEGMENTATION_MINUS_EXCLUSIONS"}
                                   for a, b in eligible])
    audit = pd.concat([audit, eligible_audit], ignore_index=True)
    speech_duration = sum(b - a for a, b in eligible)
    total = max(b for _, b in eligible) - min(a for a, _ in eligible)
    reason = "speech_duration_exceeds_total" if speech_duration > total + 1e-8 else ""
    return {"bamboo_speech_duration_s": _result(speech_duration if not reason else np.nan, reason),
            "bamboo_total_duration_s": _result(total if not reason else np.nan, reason)}, audit


def confirmed_trials(root: Path, recording_id: str, task_id: str,
                     decisions_path: Path, intervals_path: Path) -> tuple[list[dict], str]:
    """Use human-confirmed, current-review trial choices without loading Alignment."""
    _ = decisions_path, intervals_path  # Frozen paths are resolved by the V1 backend.
    trials, issue = load_confirmed_trials(root, recording_id, task_id)
    return ([{**trial, "boundary_source": "HUMAN_CONFIRMED_TRIAL"}
             for trial in trials], issue)


def s3_sentence_trials(trials: list[dict], intervals: pd.DataFrame) -> list[dict]:
    exclusions = _bounds(intervals, "manual_exclusion")
    result = []
    for trial in trials:
        a, b = trial["start_sec"], trial["end_sec"]
        reason = ("sentence_contains_hard_exclusion" if any(
            a < right and b > left for left, right in exclusions) else "")
        result.append({**trial, "duration_sec": b - a if not reason else np.nan,
                       "failure_reason": reason})
    return result


def vowel_tokens(store, recording_id: str, trial, speech: list[tuple[float, float]],
                 exclusions: list[tuple[float, float]]) -> pd.DataFrame:
    """Audit every candidate vowel; never repair or zero-fill a bad boundary."""
    rows = []
    for phone in store.phones_for_trial(recording_id, str(trial.trial_id)).itertuples():
        normalized = str(phone.phone_normalized).upper().rstrip("012")
        if normalized not in ARPABET_VOWELS:
            continue
        a, b = float(phone.start_sec), float(phone.end_sec)
        reason = ("invalid_vowel_boundary" if not np.isfinite([a, b]).all() or b <= a else
                  "vowel_outside_trial" if a < float(trial.start_sec) - 1e-8 or
                  b > float(trial.end_sec) + 1e-8 else
                  "vowel_overlaps_exclusion" if any(a < right and b > left
                                                      for left, right in exclusions) else
                  "vowel_outside_eligible_audio" if not any(left <= a < b <= right
                                                               for left, right in speech) else
                  "vowel_too_short" if b - a < MIN_VOWEL_SEC else
                  "unaccepted_alignment_token" if str(phone.alignment_review_status)
                  not in {"AUTO_ACCEPTED_STRUCTURAL", "HUMAN_ACCEPTED"} else "")
        rows.append({"recording_id": recording_id, "trial_id": str(trial.trial_id),
                     "sequence_id": str(phone.sequence_id), "task_id": str(trial.task_id),
                     "word_id": str(phone.parent_word_id), "word": str(phone.parent_word_text),
                     "phone_id": str(phone.phone_id), "phone_raw": str(phone.phone_raw),
                     "phone_normalized": normalized,
                     "phone_index_in_trial": int(phone.phone_index_in_trial),
                     "start_sec": a, "end_sec": b, "duration_sec": b - a,
                     "short_vowel_review_flag": bool(not reason and b - a < SHORT_VOWEL_SEC),
                     "alignment_review_status": str(phone.alignment_review_status),
                     "boundary_source": str(phone.boundary_source), "valid": not reason,
                     "rejection_reason": reason})
    return pd.DataFrame(rows, columns=TOKEN_COLUMNS)


def _vowel_coverage(tokens: pd.DataFrame, task_id: str) -> tuple[bool, str, float, float]:
    valid = tokens.loc[tokens.valid] if not tokens.empty else tokens
    if task_id == "sentence_s3":
        ordered = valid.sort_values("phone_index_in_trial")
        observed = tuple(zip(ordered.word.astype(str).str.lower(),
                             ordered.phone_normalized.astype(str), strict=True))
        coverage = len(observed) / len(S3_EXPECTED_VOWELS)
        return (observed == S3_EXPECTED_WORD_VOWELS,
                "" if observed == S3_EXPECTED_WORD_VOWELS else "incomplete_vowel_coverage",
                len(S3_EXPECTED_VOWELS), coverage)
    count = len(valid)
    # Thirty is a minimum sample size, not a fabricated expected passage count.
    return (count >= MIN_BAMBOO_VOWELS,
            "" if count >= MIN_BAMBOO_VOWELS else "insufficient_bamboo_vowels",
            np.nan, np.nan)


def vowel_trial_values(tokens: pd.DataFrame, task_id: str,
                       sentence_duration: float = np.nan) -> dict[str, tuple[float, str]]:
    """Token → one trial; nPVI pairs never cross sequence boundaries."""
    ids = (ALIGNMENT_IDS - {"delta_v_s", "varco_v_pct", "npvi_v_pct"}
           if task_id == "sentence_s3" else
           {"delta_v_s", "varco_v_pct", "npvi_v_pct"})
    complete, issue, _expected, _fraction = _vowel_coverage(tokens, task_id)
    if not complete:
        return {feature: _result(reason=issue) for feature in ids}
    valid = tokens.loc[tokens.valid].sort_values("phone_index_in_trial")
    d = valid.duration_sec.to_numpy(dtype=float)
    if task_id == "sentence_s3":
        values = {"s3_vocalic_duration_mean_s": _result(np.mean(d)),
                  "s3_vocalic_duration_median_s": _result(np.median(d)),
                  "s3_vocalic_duration_sd_s": _result(np.std(d, ddof=1)) if len(d) >= 2
                  else _result(reason="insufficient_vowels_for_sd"),
                  "s3_vocalic_duration_iqr_s": _result(np.quantile(d, .75) - np.quantile(d, .25))
                  if len(d) >= 2 else _result(reason="insufficient_vowels_for_iqr")}
        total = float(d.sum())
        if not np.isfinite(sentence_duration) or sentence_duration <= 0:
            values["s3_vocalic_proportion_pct"] = _result(reason="sentence_duration_nonpositive")
        elif total > sentence_duration + 1e-8:
            values["s3_vocalic_proportion_pct"] = _result(
                reason="vocalic_duration_exceeds_sentence_duration")
        else:
            values["s3_vocalic_proportion_pct"] = _result(100 * total / sentence_duration)
        return values
    if len(d) < 2:
        return {feature: _result(reason="insufficient_vowels_for_sd") for feature in ids}
    sd = float(np.std(d, ddof=1))
    mean = float(np.mean(d))
    pairs = []
    # A rejected vowel is an adjacency break, even when the sequence ID is shared.
    previous = None
    for row in tokens.sort_values("phone_index_in_trial").itertuples():
        if not row.valid:
            previous = None
            continue
        if previous is not None and row.sequence_id == previous.sequence_id:
            pairs.append(abs(row.duration_sec - previous.duration_sec) /
                         ((row.duration_sec + previous.duration_sec) / 2))
        previous = row
    return {"delta_v_s": _result(sd),
            "varco_v_pct": _result(100 * sd / mean) if mean > 0 else
            _result(reason="zero_mean_vowel_duration"),
            "npvi_v_pct": _result(100 * np.mean(pairs)) if pairs else
            _result(reason="insufficient_vowels_for_npvi")}
