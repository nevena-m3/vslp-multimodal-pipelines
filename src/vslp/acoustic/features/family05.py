"""Family 05 target-defined calculations over Family 04's shared formant frames.

No public Family 05 feature ID exists until a versioned target manifest freezes
its linguistic selector, analysis window, estimator, and exact output ID.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from vslp.acoustic.features.family04 import FRAME_COLUMNS


ALGORITHM_VERSION = "family05-formant-trajectory-1.0.0"
PARAMETER_SET_ID = "family05_targeted_formant_v1"
SLOPE_MIN_FRAMES = 5
SLOPE_MIN_TRANSITION_SEC = 0.040
DEFAULT_TRANSITION_WINDOW = (0.20, 0.80)
FEATURE_ID_TEMPLATES = (
    "f1_slope_<target>_hz_s", "f2_slope_<target>_hz_s",
    "f1_range_<diphthong>_hz", "f2_range_<diphthong>_hz",
)


@dataclass(frozen=True)
class FormantTarget:
    target_id: str
    task_id: str
    target_type: str
    expected_word: str
    expected_phone: str
    alignment_selection_rule: str
    window_start_fraction: float
    window_end_fraction: float
    aggregation: str
    f1_feature_id: str
    f2_feature_id: str
    fitted_trajectory_source: str


def load_formant_target_manifest(path: str | Path, task_id: str) -> tuple[FormantTarget, ...]:
    """Validate explicit targets; never derive target names from text or audio."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("manifest_version") != "1" or data.get("task_id") != task_id:
        raise ValueError("formant_target_manifest_mismatch")
    if not isinstance(data.get("targets"), list) or not data["targets"]:
        raise ValueError("formant_targets_missing")
    targets = []
    seen = set()
    for item in data["targets"]:
        target_id = str(item["target_id"])
        kind = str(item["target_type"])
        if not re.fullmatch(r"[a-z][a-z0-9_]*", target_id) or target_id in seen:
            raise ValueError("invalid_or_duplicate_formant_target_id")
        if kind not in {"transition", "diphthong"}:
            raise ValueError("invalid_formant_target_type")
        rule = str(item["alignment_selection_rule"])
        if rule not in {"exact_word", "exact_phone", "exact_word_and_phone"}:
            raise ValueError("alignment_selection_rule_not_frozen")
        word, phone = str(item.get("expected_word", "")), str(item.get("expected_phone", ""))
        if ((rule in {"exact_word", "exact_word_and_phone"} and not word)
                or (rule in {"exact_phone", "exact_word_and_phone"} and not phone)):
            raise ValueError("target_linguistic_label_missing")
        start, end = float(item["window_start_fraction"]), float(item["window_end_fraction"])
        if not 0 <= start < end <= 1:
            raise ValueError("invalid_formant_target_window")
        aggregation = str(item["aggregation"])
        if aggregation != "median_across_equivalent_tokens":
            raise ValueError("unsupported_formant_target_aggregation")
        f1_id, f2_id = str(item["f1_feature_id"]), str(item["f2_feature_id"])
        infix = "slope" if kind == "transition" else "range"
        suffix = "hz_s" if kind == "transition" else "hz"
        id_pattern = rf"f[12]_{infix}_[a-z][a-z0-9_]*_{suffix}"
        if (not re.fullmatch(id_pattern, f1_id)
                or not re.fullmatch(id_pattern, f2_id)
                or f1_id[3:] != f2_id[3:]):
            raise ValueError("formant_feature_ids_not_frozen")
        fit_source = str(item.get("fitted_trajectory_source", ""))
        if kind == "diphthong" and not fit_source:
            raise ValueError("diphthong_fitted_trajectory_not_frozen")
        targets.append(FormantTarget(target_id, task_id, kind, word, phone, rule,
                                     start, end, aggregation, f1_id, f2_id, fit_source))
        seen.add(target_id)
    return tuple(targets)


def load_shared_formant_frames(path: str | Path, *, profile_id: str,
                               alignment_run_id: str) -> pd.DataFrame:
    """Load the audit track produced by Family 04; never call Praat again."""
    with np.load(path, allow_pickle=False) as artifact:
        if (str(artifact["formant_profile_id"]) != profile_id
                or str(artifact["alignment_run_id"]) != alignment_run_id):
            raise ValueError("shared_formant_track_provenance_mismatch")
        return pd.DataFrame({field: artifact[field] for field in FRAME_COLUMNS})


def select_target_tokens(target: FormantTarget, token_measurements: pd.DataFrame,
                         aligned_words: pd.DataFrame) -> pd.DataFrame:
    """Exact label selection over frozen aligned tokens and their parent words."""
    tokens = token_measurements.copy()
    if target.alignment_selection_rule in {"exact_phone", "exact_word_and_phone"}:
        tokens = tokens.loc[tokens.phone_normalized.astype(str).eq(target.expected_phone)]
    if target.alignment_selection_rule in {"exact_word", "exact_word_and_phone"}:
        indices = aligned_words.loc[aligned_words.word.astype(str).eq(target.expected_word),
                                    "word_index"]
        tokens = tokens.loc[tokens.word_index.isin(indices)]
    return tokens


def regression_formant_slope(frames: pd.DataFrame, *, token_index: int,
                             token_start_sec: float, token_end_sec: float,
                             formant_number: int,
                             window: tuple[float, float] = DEFAULT_TRANSITION_WINDOW
                             ) -> tuple[float, str, int]:
    """Frozen generic OLS slope of Hz against original-clock seconds."""
    if formant_number not in {1, 2}:
        raise ValueError("unsupported_formant_number")
    duration = token_end_sec - token_start_sec
    if duration < SLOPE_MIN_TRANSITION_SEC:
        return np.nan, "transition_too_short", 0
    left, right = window
    if not 0 <= left < right <= 1:
        raise ValueError("invalid_transition_window")
    start, end = token_start_sec + duration * left, token_start_sec + duration * right
    selected = frames.loc[frames.token_index.eq(token_index)
                          & frames.time_sec.between(start, end, inclusive="both")
                          & frames.valid_ordered.astype(bool)].copy()
    selected = selected.loc[np.isfinite(selected[f"f{formant_number}_hz"].to_numpy(dtype=float))]
    if len(selected) < SLOPE_MIN_FRAMES:
        return np.nan, "insufficient_valid_trajectory_frames", len(selected)
    x = selected.time_sec.to_numpy(dtype=float)
    y = selected[f"f{formant_number}_hz"].to_numpy(dtype=float)
    centered = x - np.mean(x)
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        return np.nan, "degenerate_trajectory_time_axis", len(selected)
    slope = float(np.dot(centered, y - np.mean(y)) / denominator)
    return slope, "", len(selected)


def range_from_fitted_trajectory(fitted_hz: np.ndarray) -> tuple[float, str]:
    """Exact max-min operation; source fit and glide guards must be supplied upstream."""
    values = np.asarray(fitted_hz, dtype=float)
    if values.size < SLOPE_MIN_FRAMES or not np.isfinite(values).all():
        return np.nan, "insufficient_fitted_trajectory_frames"
    return float(np.max(values) - np.min(values)), ""


def aggregate_equivalent_targets(values: list[float]) -> tuple[float, str]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return (float(np.median(finite)), "") if finite.size else (np.nan, "no_valid_target_tokens")
