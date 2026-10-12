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


ALGORITHM_VERSION = "family04-parselmouth-burg-2.0.0"
DEFAULT_PROFILE_ID = "family04_burg_native_5500_v1"
SENSITIVITY_PROFILE_ID = "family04_burg_native_5000_v1"
TOKEN_IDS = frozenset({"f1_token_hz", "f2_token_hz", "f3_token_hz"})
VOWEL_IDS = frozenset({"f1_vowel_median_hz", "f2_vowel_median_hz", "f3_vowel_median_hz"})
GENERIC_IDS = frozenset({"f1_median_hz", "f2_median_hz", "f3_median_hz"})
S3_IDS = frozenset({"vowel_distance_ai_hz", "vowel_distance_au_hz",
                    "vowel_distance_iu_hz", "f2_distance_i_a_hz",
                    "vowel_envelope_distance_ai_source"})
DISPERSION_IDS = frozenset({"vowel_dispersion_ai_hz", "vowel_dispersion_au_hz",
                            "vowel_dispersion_iu_hz", "vowel_dispersion_mean_3corner_hz",
                            "vowel_dispersion_a_to_global_hz", "vowel_dispersion_i_to_global_hz",
                            "vowel_dispersion_u_to_global_hz", "vowel_dispersion_global_mean_hz"})
CORNER_IDS = frozenset({"vsa3_iau_hz2", "fri_iau", "sfri_iau", "vai_iau",
                        "fcr_iau"}) | DISPERSION_IDS
DERIVED_IDS = CORNER_IDS | S3_IDS
FAMILY04_IDS = TOKEN_IDS | VOWEL_IDS | GENERIC_IDS | DERIVED_IDS
FORMANT_BOUNDS = {"f1": (150.0, 1200.0), "f2": (500.0, 3500.0),
                  "f3": (1200.0, 5000.0)}
TOKEN_COLUMNS = (
    "recording_id", "file_name", "task_id", "task_type", "task_item", "trial_id",
    "alignment_run_id", "sequence_id", "word_id", "phone_id",
    "word_index", "word", "phone_index_in_trial", "phone_index_in_word",
    "boundary_source", "alignment_review_status", "phone_raw",
    "phone_normalized", "vowel_category",
    "token_start_sec", "token_end_sec", "token_duration_sec", "measurement_start_sec",
    "measurement_end_sec", "f1_token_hz", "f2_token_hz", "f3_token_hz",
    "f1_iqr_hz", "f2_iqr_hz", "f3_iqr_hz", "b1_hz", "b2_hz", "b3_hz",
    "n_valid_frames", "n_total_frames", "native_sample_rate_hz",
    "n_nonordered_frames", "formant_profile_id", "tracking_status", "review_flags",
)
VOWEL_COLUMNS = (
    "recording_id", "file_name", "task_id", "task_item", "trial_id",
    "alignment_run_id", "target_vowel",
    "f1_vowel_median_hz", "f2_vowel_median_hz", "f3_vowel_median_hz",
    "f1_token_iqr_hz", "f2_token_iqr_hz", "f3_token_iqr_hz", "n_valid_tokens",
    "n_tracking_failures", "review_flags", "formant_profile_id",
)
FRAME_COLUMNS = ("token_index", "time_sec", "f1_hz", "f2_hz", "f3_hz",
                 "b1_hz", "b2_hz", "b3_hz",
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
    """Use the frozen task inventory, with explicit manifests kept for compatibility."""
    if not path:
        if task_id in {"bamboo_passage", "sentence_s3", "sentence_s5", "wstg"}:
            return {"AA": "a", "IY": "i", "UW": "u"}, "", "f04_iau_arpabet_v1"
        return None, "missing_vowel_category_mapping", ""
    if not Path(path).is_file():
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
                            profile: FormantProfile,
                            analysis_window: tuple[float, float] = (.25, .75)) -> dict[str, object]:
    """Median/IQR from the chosen central window (F04 default is middle 50%)."""
    duration = token_end - token_start
    middle_start = token_start + analysis_window[0] * duration
    middle_end = token_start + analysis_window[1] * duration
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
        result[f"b{number}_hz"] = np.nan
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
    in_range = np.logical_and.reduce([
        middle[f"f{number}_hz"].between(*FORMANT_BOUNDS[f"f{number}"]).to_numpy()
        for number in (1, 2, 3)])
    valid = middle.loc[middle.valid_ordered.astype(bool) & in_range]
    result["n_valid_frames"] = len(valid)
    if len(valid) < profile.minimum_valid_frames:
        result["tracking_status"] = "INSUFFICIENT_VALID_FRAMES"
        return result
    flags = []
    if result["n_nonordered_frames"]:
        flags.append("nonordered_formant_frames")
    if not np.all(in_range):
        flags.append("out_of_plausibility_range")
    for number in (1, 2, 3):
        key = f"f{number}"
        values = valid[f"{key}_hz"].to_numpy(dtype=float)
        result[f"{key}_token_hz"] = float(np.median(values))
        result[f"{key}_iqr_hz"] = float(np.percentile(values, 75) - np.percentile(values, 25))
        low, high = FORMANT_BOUNDS[key]
        all_values = middle[f"{key}_hz"].to_numpy(dtype=float)
        if np.any((all_values < low) | (all_values > high)):
            flags.append(f"{key}_outside_plausibility_range")
        bandwidth = f"b{number}_hz"
        if bandwidth in valid:
            widths = valid[bandwidth].to_numpy(dtype=float)
            widths = widths[np.isfinite(widths) & (widths > 0)]
            if widths.size:
                result[bandwidth] = float(np.median(widths))
    result["review_flags"] = ";".join(flags)
    return result


def track_aligned_vowels(audio: np.ndarray, sample_rate: int, vowel_tokens: pd.DataFrame,
                         profile: FormantProfile, *, recording_id: str, file_name: str,
                         task_id: str, task_type: str, alignment_run_id: str,
                         task_item: str = "", trial_id: str = "",
                         category_mapping: dict[str, str] | None = None,
                         hard_exclusions: list[tuple[float, float]] | None = None,
                         formant_object=None,
                         analysis_window: tuple[float, float] = (.25, .75)) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        measurement_start = start + analysis_window[0] * duration
        measurement_end = start + analysis_window[1] * duration
        trial_start = float(getattr(token, "trial_start_sec", start))
        trial_end = float(getattr(token, "trial_end_sec", end))
        if not np.isfinite([trial_start, trial_end]).all() or not trial_start <= start < end <= trial_end:
            raise ValueError("aligned_vowel_outside_trial")
        local = []
        half_window = profile.window_length_sec / 2.0
        for time in times[(times >= measurement_start) & (times <= measurement_end)]:
            support_start, support_end = time - half_window, time + half_window
            if (support_start < max(start, trial_start, 0.0) - 1e-9 or
                    support_end > min(end, trial_end, len(signal) / sample_rate) + 1e-9 or
                    any(support_start < excluded_end and support_end > excluded_start
                        for excluded_start, excluded_end in hard_exclusions or ())):
                continue
            values = tuple(float(formant.get_value_at_time(index, float(time))) for index in (1, 2, 3))
            valid = _valid_formants(values)
            widths = tuple(float(formant.get_bandwidth_at_time(index, float(time)))
                           for index in (1, 2, 3)) if hasattr(formant, "get_bandwidth_at_time") else (np.nan,) * 3
            frame = {"token_index": token_index, "time_sec": float(time),
                     "f1_hz": values[0], "f2_hz": values[1], "f3_hz": values[2],
                     "b1_hz": widths[0], "b2_hz": widths[1], "b3_hz": widths[2],
                     "valid_ordered": valid,
                     "in_middle_50_percent": start + .25 * duration <= time <= end - .25 * duration}
            frame_rows.append(frame)
            local.append(frame)
        summary = summarize_formant_token(pd.DataFrame(local, columns=FRAME_COLUMNS),
                                          start, end, profile, analysis_window)
        normalized = str(token.phone_normalized).upper()
        token_rows.append({"recording_id": recording_id, "file_name": file_name,
                           "task_id": task_id, "task_type": task_type,
                           "task_item": task_item,
                           "trial_id": getattr(token, "trial_id", trial_id),
                           "alignment_run_id": alignment_run_id,
                           "sequence_id": getattr(token, "sequence_id", ""),
                           "word_id": getattr(token, "parent_word_id", ""),
                           "phone_id": getattr(token, "phone_id", ""),
                           "word_index": getattr(token, "parent_word_index", np.nan),
                           "phone_index_in_trial": getattr(token, "phone_index_in_trial", np.nan),
                           "phone_index_in_word": getattr(token, "phone_index_in_word", np.nan),
                           "boundary_source": getattr(token, "boundary_source", ""),
                           "alignment_review_status": getattr(token, "alignment_review_status", ""),
                           "word": getattr(token, "parent_word_text", getattr(token, "word", "")),
                           "phone_raw": token.phone_raw, "phone_normalized": normalized,
                           "vowel_category": (category_mapping or {}).get(normalized, ""),
                           "native_sample_rate_hz": sample_rate,
                           "formant_profile_id": profile.profile_id, **summary})
    return (pd.DataFrame(token_rows, columns=TOKEN_COLUMNS),
            pd.DataFrame(frame_rows, columns=FRAME_COLUMNS))


def aggregate_vowel_centroids(tokens: pd.DataFrame) -> pd.DataFrame:
    """Recording median of valid token medians, separately for each named vowel."""
    if tokens.empty:
        return pd.DataFrame(columns=VOWEL_COLUMNS)
    rows = []
    candidates = tokens.loc[tokens.vowel_category.astype(str).ne("")].copy()
    if "trial_id" not in candidates:
        candidates["trial_id"] = "legacy_whole_recording"
    for (trial_id, category), group in candidates.groupby(["trial_id", "vowel_category"]):
        valid = group.loc[group.tracking_status.eq("OK")]
        row = {field: group.iloc[0].get(field, "") for field in
               ("recording_id", "file_name", "task_id", "task_item", "trial_id",
                "alignment_run_id", "formant_profile_id")}
        row.update({"trial_id": trial_id, "target_vowel": category,
                    "n_valid_tokens": len(valid),
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
        return {feature: (np.nan, "missing_vowel_category_mapping") for feature in CORNER_IDS}
    by_vowel = {str(row.target_vowel): row for row in vowels.itertuples()}

    def needed(*categories: str) -> bool:
        return all(category in by_vowel and int(by_vowel[category].n_valid_tokens) > 0
                   for category in categories)

    results = {feature: (np.nan, "missing_required_vowel") for feature in CORNER_IDS}
    if not needed("i", "a", "u"):
        return results
    i, a, u = (by_vowel[key] for key in ("i", "a", "u"))
    f1i, f1a, f1u = (float(item.f1_vowel_median_hz) for item in (i, a, u))
    f2i, f2a, f2u = (float(item.f2_vowel_median_hz) for item in (i, a, u))
    values = (f1i, f1a, f1u, f2i, f2a, f2u)
    if not np.isfinite(values).all():
        return {feature: (np.nan, "invalid_required_vowel_centroid") for feature in CORNER_IDS}
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
    points = {"a": (f1a, f2a), "i": (f1i, f2i), "u": (f1u, f2u)}
    pair_values = {pair: vowel_distance(points[pair[0]], points[pair[1]])
                   for pair in ("ai", "au", "iu")}
    for pair, distance in pair_values.items():
        results[f"vowel_dispersion_{pair}_hz"] = (distance, "")
    results["vowel_dispersion_mean_3corner_hz"] = (float(np.mean(list(pair_values.values()))), "")
    global_center = (float(np.mean([f1a, f1i, f1u])),
                     float(np.mean([f2a, f2i, f2u])))
    radial = {key: vowel_distance(point, global_center) for key, point in points.items()}
    for key, distance in radial.items():
        results[f"vowel_dispersion_{key}_to_global_hz"] = (distance, "")
    results["vowel_dispersion_global_mean_hz"] = (float(np.mean(list(radial.values()))), "")
    return results


S3_TARGETS = {"a": ("spot", "AA"), "i": ("key", "IY"), "u": ("blue", "UW")}


def derive_s3_features(tokens: pd.DataFrame) -> dict[str, tuple[float, str]]:
    """Use only the three approved word/phone anchors, within one trial."""
    results = {key: (np.nan, "missing_required_lexical_vowel") for key in S3_IDS}
    points = {}
    ambiguous = set()
    for vowel, (word, phone) in S3_TARGETS.items():
        group = tokens.loc[tokens.word.astype(str).str.casefold().eq(word)
                           & tokens.phone_normalized.astype(str).eq(phone)
                           & tokens.tracking_status.eq("OK")]
        if len(group) > 1:
            ambiguous.add(vowel)
        elif not group.empty:
            points[vowel] = (float(group.f1_token_hz.median()),
                             float(group.f2_token_hz.median()))
    for pair in ("ai", "au", "iu"):
        if pair[0] in ambiguous or pair[1] in ambiguous:
            results[f"vowel_distance_{pair}_hz"] = (np.nan, "AMBIGUOUS_CANONICAL_TARGET")
        elif pair[0] in points and pair[1] in points:
            results[f"vowel_distance_{pair}_hz"] = (
                vowel_distance(points[pair[0]], points[pair[1]]), "")
    if "a" in ambiguous or "i" in ambiguous:
        results["f2_distance_i_a_hz"] = (np.nan, "AMBIGUOUS_CANONICAL_TARGET")
    elif "a" in points and "i" in points:
        results["f2_distance_i_a_hz"] = (abs(points["i"][1] - points["a"][1]), "")
    return results


def derive_trial_features(tokens: pd.DataFrame, centroids: pd.DataFrame,
                          task_id: str, mapping_available: bool,
                          audio: np.ndarray | None = None,
                          sample_rate: int | None = None) -> dict[str, tuple[float, str]]:
    """One trial's scalar outputs from the shared authoritative token table."""
    valid = tokens.loc[tokens.tracking_status.eq("OK")] if not tokens.empty else tokens
    results = {}
    for number in (1, 2, 3):
        field = f"f{number}_token_hz"
        vals = valid[field].to_numpy(dtype=float) if field in valid else np.empty(0)
        vals = vals[np.isfinite(vals)]
        results[f"f{number}_median_hz"] = (
            (float(np.median(vals)), "") if vals.size else
            (np.nan, "insufficient_valid_formant_tokens"))
    if task_id == "bamboo_passage":
        results.update(derive_iau_features(centroids, mapping_available))
    else:
        results.update({key: (np.nan, "requires_bamboo_corner_vowels")
                        for key in CORNER_IDS})
    if task_id == "sentence_s3":
        results.update(derive_s3_features(valid))
        if audio is not None and sample_rate is not None:
            from vslp.acoustic.features.family04_envelope import (
                allpole_token_envelope, envelope_l1_distance,
            )
            envelopes = {}
            duplicate_envelope_target = False
            for vowel in ("a", "i"):
                word, phone = S3_TARGETS[vowel]
                targets = valid.loc[valid.word.astype(str).str.casefold().eq(word)
                                    & valid.phone_normalized.astype(str).eq(phone)]
                if len(targets) > 1:
                    duplicate_envelope_target = True
                    continue
                measured = []
                for token in targets.itertuples():
                    try:
                        measured.append(allpole_token_envelope(
                            audio, sample_rate, float(token.token_start_sec),
                            float(token.token_end_sec)))
                    except (ValueError, RuntimeError):
                        continue
                if measured:
                    envelopes[vowel] = np.median(measured, axis=0)
            if duplicate_envelope_target:
                results["vowel_envelope_distance_ai_source"] = (
                    np.nan, "AMBIGUOUS_CANONICAL_TARGET")
            elif set(envelopes) == {"a", "i"}:
                results["vowel_envelope_distance_ai_source"] = (
                    envelope_l1_distance(envelopes["i"], envelopes["a"]), "")
            else:
                results["vowel_envelope_distance_ai_source"] = (
                    np.nan, "insufficient_valid_envelope_tokens")
    else:
        results.update({key: (np.nan, "requires_s3_lexical_targets") for key in S3_IDS})
    return results


def track_sustained_midpoints(audio: np.ndarray, sample_rate: int,
                              regions: list[tuple[str, float, float]],
                              profile: FormantProfile, *, recording_id: str,
                              file_name: str, task_id: str) -> pd.DataFrame:
    """Evaluate the same Burg object at each reviewed stable-/a/ trial midpoint."""
    if sample_rate <= 2 * profile.ceiling_hz:
        raise ValueError("incompatible_native_sample_rate_formant_ceiling")
    formant = parselmouth.Sound(np.asarray(audio, dtype=float),
                                sampling_frequency=sample_rate).to_formant_burg(
        time_step=profile.time_step_sec, max_number_of_formants=profile.max_formants,
        maximum_formant=profile.ceiling_hz, window_length=profile.window_length_sec,
        pre_emphasis_from=profile.pre_emphasis_from_hz)
    rows = []
    for trial_id, start, end in regions:
        midpoint = (start + end) / 2
        vals = [float(formant.get_value_at_time(i, midpoint)) for i in (1, 2, 3)]
        valid = _valid_formants(tuple(vals)) and all(
            FORMANT_BOUNDS[f"f{i}"][0] <= val <= FORMANT_BOUNDS[f"f{i}"][1]
            for i, val in enumerate(vals, 1))
        widths = [float(formant.get_bandwidth_at_time(i, midpoint))
                  for i in (1, 2, 3)]
        rows.append({"recording_id": recording_id, "file_name": file_name,
                     "task_id": task_id, "task_type": "Sustained phonation",
                     "task_item": "a", "trial_id": trial_id, "word": "",
                     "phone_raw": "AA", "phone_normalized": "AA",
                     "vowel_category": "a", "token_start_sec": start,
                     "token_end_sec": end, "token_duration_sec": end - start,
                     "measurement_start_sec": midpoint, "measurement_end_sec": midpoint,
                     "native_sample_rate_hz": sample_rate,
                     "formant_profile_id": profile.profile_id,
                     "tracking_status": "OK" if valid else "INVALID_MIDPOINT_FORMANTS",
                     "review_flags": "" if valid else "invalid_midpoint_formants",
                     "n_valid_frames": int(valid), "n_total_frames": 1,
                     **{f"f{i}_token_hz": val if valid else np.nan
                        for i, val in enumerate(vals, 1)},
                     **{f"b{i}_hz": width if np.isfinite(width) and width > 0 else np.nan
                        for i, width in enumerate(widths, 1)}})
    return pd.DataFrame(rows, columns=TOKEN_COLUMNS)


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
