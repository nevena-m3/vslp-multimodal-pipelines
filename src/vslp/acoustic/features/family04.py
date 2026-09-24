"""Family 04 formants from frozen aligned vowel tokens on native-rate audio."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth

from vslp.acoustic.alignment.stage import ARPABET_VOWELS, PHONE_SET
from vslp.core.provenance import sha256_file


ALGORITHM_VERSION = "family04-parselmouth-burg-1.0.0"
DEFAULT_PROFILE_ID = "family04_burg_native_5500_v1"
SENSITIVITY_PROFILE_ID = "family04_burg_native_5000_v1"
TOKEN_IDS = frozenset({"f1_token_hz", "f2_token_hz", "f3_token_hz"})
VOWEL_IDS = frozenset({"f1_vowel_median_hz", "f2_vowel_median_hz", "f3_vowel_median_hz"})
DERIVED_IDS = frozenset({"vsa3_iau_hz2", "fri_iau", "sfri_iau", "vai_iau",
                         "fcr_iau", "f2_distance_i_a_hz"})
FAMILY04_IDS = TOKEN_IDS | VOWEL_IDS | DERIVED_IDS
FORMANT_BOUNDS = {"f1": (150.0, 1200.0), "f2": (500.0, 3500.0),
                  "f3": (1200.0, 5000.0)}
TOKEN_COLUMNS = (
    "recording_id", "file_name", "task_id", "task_type", "alignment_run_id",
    "word_index", "phone_index", "phone_raw", "phone_normalized", "vowel_category",
    "token_start_sec", "token_end_sec", "token_duration_sec", "measurement_start_sec",
    "measurement_end_sec", "f1_token_hz", "f2_token_hz", "f3_token_hz",
    "f1_iqr_hz", "f2_iqr_hz", "f3_iqr_hz", "n_valid_frames", "n_total_frames",
    "n_nonordered_frames", "formant_profile_id", "tracking_status", "review_flags",
)
VOWEL_COLUMNS = (
    "recording_id", "file_name", "task_id", "alignment_run_id", "target_vowel",
    "f1_vowel_median_hz", "f2_vowel_median_hz", "f3_vowel_median_hz",
    "f1_token_iqr_hz", "f2_token_iqr_hz", "f3_token_iqr_hz", "n_valid_tokens",
    "n_tracking_failures", "review_flags", "formant_profile_id",
)
FRAME_COLUMNS = ("token_index", "time_sec", "f1_hz", "f2_hz", "f3_hz",
                 "valid_ordered", "in_middle_50_percent")


@dataclass(frozen=True)
class FormantProfile:
    profile_id: str = DEFAULT_PROFILE_ID
    window_length_sec: float = 0.025
    time_step_sec: float = 0.005
    max_formants: int = 5
    pre_emphasis_from_hz: float = 50.0
    ceiling_hz: float = 5500.0
    minimum_token_duration_sec: float = 0.050
    minimum_valid_frames: int = 3


def profile_by_id(profile_id: str = DEFAULT_PROFILE_ID) -> FormantProfile:
    if profile_id == DEFAULT_PROFILE_ID:
        return FormantProfile()
    if profile_id == SENSITIVITY_PROFILE_ID:
        return FormantProfile(profile_id=profile_id, ceiling_hz=5000.0)
    raise ValueError("unknown_formant_profile")


def load_vowel_category_mapping(path: str | Path | None, task_id: str,
                                phone_set: str = PHONE_SET) -> tuple[dict[str, str] | None, str, str]:
    """Require explicit versioned phone→canonical vowel assignments; never infer them."""
    if not path or not Path(path).is_file():
        return None, "missing_vowel_category_mapping", ""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (data.get("manifest_version") != "1" or data.get("task_id") != task_id
                or data.get("phone_set") != phone_set):
            raise ValueError("vowel_mapping_manifest_mismatch")
        mapping: dict[str, str] = {}
        categories = data.get("categories")
        if not isinstance(categories, list) or not categories:
            raise ValueError("vowel_mapping_categories_missing")
        for entry in categories:
            category = str(entry["canonical_vowel"]).strip().lower()
            labels = entry["accepted_phone_labels"]
            if not category or not isinstance(labels, list) or not labels:
                raise ValueError("vowel_mapping_category_invalid")
            for label in labels:
                normalized = str(label).strip().upper()
                if normalized not in ARPABET_VOWELS or normalized in mapping:
                    raise ValueError("vowel_mapping_phone_invalid_or_duplicated")
                mapping[normalized] = category
        return mapping, "", sha256_file(path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return None, str(exc) or "invalid_vowel_category_mapping", ""


def _valid_formants(values: tuple[float, float, float]) -> bool:
    return bool(np.isfinite(values).all() and 0 < values[0] < values[1] < values[2])


def summarize_formant_token(frame_table: pd.DataFrame, token_start: float, token_end: float,
                            profile: FormantProfile) -> dict[str, object]:
    """Median and IQR of ordered F1/F2/F3 frames from the middle 50% only."""
    duration = token_end - token_start
    middle_start, middle_end = token_start + .25 * duration, token_end - .25 * duration
    result: dict[str, object] = {
        "token_start_sec": token_start, "token_end_sec": token_end,
        "token_duration_sec": duration, "measurement_start_sec": middle_start,
        "measurement_end_sec": middle_end, "n_total_frames": 0,
        "n_valid_frames": 0, "n_nonordered_frames": 0,
        "tracking_status": "OK", "review_flags": "",
    }
    for number in (1, 2, 3):
        result[f"f{number}_token_hz"] = np.nan
        result[f"f{number}_iqr_hz"] = np.nan
    if duration < profile.minimum_token_duration_sec:
        result["tracking_status"] = "TOKEN_TOO_SHORT"
        return result
    middle = frame_table.loc[frame_table.time_sec.between(middle_start, middle_end,
                                                          inclusive="both")].copy()
    result["n_total_frames"] = len(middle)
    if middle.empty:
        result["tracking_status"] = "INSUFFICIENT_VALID_FRAMES"
        return result
    result["n_nonordered_frames"] = int((~middle.valid_ordered.astype(bool)).sum())
    valid = middle.loc[middle.valid_ordered.astype(bool)]
    result["n_valid_frames"] = len(valid)
    if len(valid) < profile.minimum_valid_frames:
        result["tracking_status"] = "INSUFFICIENT_VALID_FRAMES"
        return result
    flags = []
    if result["n_nonordered_frames"]:
        flags.append("nonordered_formant_frames")
    for number in (1, 2, 3):
        key = f"f{number}"
        values = valid[f"{key}_hz"].to_numpy(dtype=float)
        result[f"{key}_token_hz"] = float(np.median(values))
        result[f"{key}_iqr_hz"] = float(np.percentile(values, 75) - np.percentile(values, 25))
        low, high = FORMANT_BOUNDS[key]
        if np.any((values < low) | (values > high)):
            flags.append(f"{key}_outside_plausibility_range")
    result["review_flags"] = ";".join(flags)
    return result


def track_aligned_vowels(audio: np.ndarray, sample_rate: int, vowel_tokens: pd.DataFrame,
                         profile: FormantProfile, *, recording_id: str, file_name: str,
                         task_id: str, task_type: str, alignment_run_id: str,
                         category_mapping: dict[str, str] | None = None,
                         formant_object=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One Burg track per recording, shared by Families 04 and 05."""
    if sample_rate <= 2 * profile.ceiling_hz:
        raise ValueError("incompatible_native_sample_rate_formant_ceiling")
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1 or not signal.size or not np.isfinite(signal).all():
        raise ValueError("invalid_canonical_audio")
    formant = formant_object or parselmouth.Sound(
        signal, sampling_frequency=sample_rate).to_formant_burg(
            time_step=profile.time_step_sec, max_number_of_formants=profile.max_formants,
            maximum_formant=profile.ceiling_hz, window_length=profile.window_length_sec,
            pre_emphasis_from=profile.pre_emphasis_from_hz)
    times = np.asarray(formant.xs(), dtype=float)
    token_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    for token_index, token in enumerate(vowel_tokens.sort_values("start_sec").itertuples(), 1):
        start, end = float(token.start_sec), float(token.end_sec)
        if not 0 <= start < end <= len(signal) / sample_rate + 1e-7:
            raise ValueError("aligned_vowel_outside_canonical_audio")
        duration = end - start
        measurement_start, measurement_end = start + .25 * duration, end - .25 * duration
        local = []
        for time in times[(times >= start) & (times <= end)]:
            values = tuple(float(formant.get_value_at_time(index, float(time))) for index in (1, 2, 3))
            valid = _valid_formants(values)
            frame = {"token_index": token_index, "time_sec": float(time),
                     "f1_hz": values[0], "f2_hz": values[1], "f3_hz": values[2],
                     "valid_ordered": valid,
                     "in_middle_50_percent": measurement_start <= time <= measurement_end}
            frame_rows.append(frame)
            local.append(frame)
        summary = summarize_formant_token(pd.DataFrame(local, columns=FRAME_COLUMNS),
                                          start, end, profile)
        normalized = str(token.phone_normalized).upper()
        token_rows.append({"recording_id": recording_id, "file_name": file_name,
                           "task_id": task_id, "task_type": task_type,
                           "alignment_run_id": alignment_run_id,
                           "word_index": token.word_index, "phone_index": token.phone_index,
                           "phone_raw": token.phone_raw, "phone_normalized": normalized,
                           "vowel_category": (category_mapping or {}).get(normalized, ""),
                           "formant_profile_id": profile.profile_id, **summary})
    return (pd.DataFrame(token_rows, columns=TOKEN_COLUMNS),
            pd.DataFrame(frame_rows, columns=FRAME_COLUMNS))


def aggregate_vowel_centroids(tokens: pd.DataFrame) -> pd.DataFrame:
    """Recording median of valid token medians, separately for each named vowel."""
    if tokens.empty:
        return pd.DataFrame(columns=VOWEL_COLUMNS)
    rows = []
    for category, group in tokens.loc[tokens.vowel_category.astype(str).ne("")].groupby("vowel_category"):
        valid = group.loc[group.tracking_status.eq("OK")]
        row = {field: group.iloc[0][field] for field in
               ("recording_id", "file_name", "task_id", "alignment_run_id", "formant_profile_id")}
        row.update({"target_vowel": category, "n_valid_tokens": len(valid),
                    "n_tracking_failures": len(group) - len(valid),
                    "review_flags": "low_n_vowel_tokens" if len(valid) == 1 else ""})
        for index in (1, 2, 3):
            data = valid[f"f{index}_token_hz"].to_numpy(dtype=float)
            row[f"f{index}_vowel_median_hz"] = float(np.median(data)) if len(data) else np.nan
            row[f"f{index}_token_iqr_hz"] = (float(np.percentile(data, 75) -
                                                  np.percentile(data, 25)) if len(data) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows, columns=VOWEL_COLUMNS)


def derive_iau_features(vowels: pd.DataFrame,
                        mapping_available: bool) -> dict[str, tuple[float, str]]:
    """Exact source-lineage /i,a,u/ quantities from named Hz centroids."""
    if not mapping_available:
        return {feature: (np.nan, "missing_vowel_category_mapping") for feature in DERIVED_IDS}
    by_vowel = {str(row.target_vowel): row for row in vowels.itertuples()}

    def needed(*categories: str) -> bool:
        return all(category in by_vowel and int(by_vowel[category].n_valid_tokens) > 0
                   for category in categories)

    results = {feature: (np.nan, "missing_required_vowel") for feature in DERIVED_IDS}
    if needed("i", "a"):
        results["f2_distance_i_a_hz"] = (abs(float(by_vowel["i"].f2_vowel_median_hz) -
                                              float(by_vowel["a"].f2_vowel_median_hz)), "")
    if not needed("i", "a", "u"):
        return results
    i, a, u = (by_vowel[key] for key in ("i", "a", "u"))
    f1i, f1a, f1u = (float(item.f1_vowel_median_hz) for item in (i, a, u))
    f2i, f2a, f2u = (float(item.f2_vowel_median_hz) for item in (i, a, u))
    values = (f1i, f1a, f1u, f2i, f2a, f2u)
    if not np.isfinite(values).all():
        return {feature: (np.nan, "invalid_required_vowel_centroid") for feature in DERIVED_IDS}
    results["vsa3_iau_hz2"] = (.5 * abs(f1i * (f2a - f2u) + f1a * (f2u - f2i) +
                                      f1u * (f2i - f2a)), "")
    formulas = {
        "fri_iau": (f1a + f1i + f1u + f2a + f2i, f2u),
        "sfri_iau": (f2a + f2i, f2u),
        "vai_iau": (f2i + f1a, f1i + f1u + f2u + f2a),
        "fcr_iau": (f2u + f2a + f1i + f1u, f2i + f1a),
    }
    for feature, (numerator, denominator) in formulas.items():
        results[feature] = ((numerator / denominator, "") if denominator > 0 else
                            (np.nan, "invalid_centroid_denominator"))
    return results


def vowel_distance(x: tuple[float, float], y: tuple[float, float]) -> float:
    """Internal calculator; no public parameterized ID without frozen inventory."""
    return float(np.hypot(x[0] - y[0], x[1] - y[1]))


def vowel_dispersion(centroids: list[tuple[float, float]]) -> float:
    """Internal mean-pairwise distance; no public variant without frozen inventory."""
    if len(centroids) < 2:
        return np.nan
    distances = [vowel_distance(left, right) for index, left in enumerate(centroids)
                 for right in centroids[index + 1:]]
    return float(np.mean(distances))
