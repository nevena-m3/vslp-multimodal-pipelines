"""Family 05 target-defined dynamics on final Alignment V1 and native audio.

Structural implementation and source lineage are distinct from numerical source
parity or biological boundary accuracy. No provider-specific audio or tokens enter.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from vslp.acoustic.features.family04 import FRAME_COLUMNS, FORMANT_BOUNDS


ALGORITHM_VERSION = "family05-bisquare-irls-1.0.0"
PARAMETER_SET_ID = "family05_targeted_formant_v2"
SLOPE_MIN_FRAMES = 5
SLOPE_MIN_TRANSITION_SEC = 0.040
DEFAULT_TRANSITION_WINDOW = (0.20, 0.80)
ROBUST_METHOD = "bisquare_irls_mad_4.685_v1"
TARGETS = {
    "buy_ay": ("sentence_s2", "buy", "AY"),
    "away_ey": ("sentence_s5", "away", "EY"),
    "ago_ow": ("sentence_s5", "ago", "OW"),
}
TARGET_PROMPTS = {"sentence_s2": "sentence_s2_buy_bobby_a_puppy",
                  "sentence_s5": "sentence_s5_we_were_away_a_year_ago"}
FAMILY05_IDS = frozenset({
    f"f{number}_transition_slope_{target}_hz_s"
    for target in TARGETS for number in (1, 2)
}) | frozenset({"f1_range_buy_ay_hz", "f2_range_buy_ay_hz"})
AUDIT_COLUMNS = (
    "recording_id", "trial_id", "sequence_id", "task_id", "task_item",
    "word_id", "phone_id", "word", "canonical_phone", "target_id",
    "token_start_sec", "token_end_sec", "token_duration_sec",
    "analysis_start_sec", "analysis_end_sec", "analysis_duration_sec",
    "native_sample_rate_hz", "n_candidate_frames", "n_valid_frames",
    "f1_slope_hz_s", "f2_slope_hz_s", "f1_fitted_range_hz",
    "f2_fitted_range_hz", "f1_intercept_hz", "f2_intercept_hz",
    "f1_residual_mad_hz", "f2_residual_mad_hz", "fit_method",
    "fit_convergence_status", "formant_parameter_set_id", "boundary_source",
    "alignment_review_status", "alignment_run_id", "source_low_level_parity",
    "qc_status", "qc_flags",
)
FEATURE_ID_TEMPLATES = (
    "f1_slope_<target>_hz_s", "f2_slope_<target>_hz_s",
    "f1_range_<diphthong>_hz", "f2_range_<diphthong>_hz",
)
# Historical research helpers below remain callable for compatibility. The
# production F05 executor uses TARGETS, measure_target_frames and bisquare IRLS.


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
    """Legacy generic OLS helper; production F05 uses robust_bisquare_fit."""
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


def transition_window(start: float, end: float) -> tuple[float, float]:
    """Central 20–80% of an original-time final phone interval."""
    duration = end - start
    if not np.isfinite([start, end]).all() or duration < SLOPE_MIN_TRANSITION_SEC:
        raise ValueError("target_too_short")
    return start + .20 * duration, start + .80 * duration


def robust_bisquare_fit(times: np.ndarray, values: np.ndarray) -> dict[str, float | str]:
    """MAD-scaled Tukey bisquare IRLS, c=4.685, max 50, relative beta tol=1e-8.

    Time is centered for numerical stability; returned intercept is at the first
    input time. A zero-residual line is accepted without an unnecessary iteration.
    """
    x, y = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    if len(x) < SLOPE_MIN_FRAMES or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("robust_fit_failed")
    origin = float(x[0])
    design = np.column_stack((np.ones(len(x)), x - origin))
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError("robust_fit_failed")
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    status = "converged"
    for _ in range(50):
        residual = y - design @ beta
        mad = float(np.median(np.abs(residual - np.median(residual))))
        scale = mad / .6744897501960817
        if scale <= 1e-10:
            break
        u = residual / (4.685 * scale)
        weights = np.square(1 - np.square(u)) * (np.abs(u) < 1)
        if np.count_nonzero(weights > 0) < 2:
            raise ValueError("robust_fit_failed")
        weighted = design * np.sqrt(weights)[:, None]
        candidate = np.linalg.lstsq(weighted, y * np.sqrt(weights), rcond=None)[0]
        if not np.isfinite(candidate).all():
            raise ValueError("robust_fit_failed")
        if np.linalg.norm(candidate - beta) <= 1e-8 * (1 + np.linalg.norm(beta)):
            beta = candidate
            break
        beta = candidate
    else:
        status = "not_converged"
    residual = y - design @ beta
    return {"intercept": float(beta[0]), "slope": float(beta[1]),
            "residual_mad": float(np.median(np.abs(residual - np.median(residual)))),
            "status": status}


def resolve_target_phones(store, recording_id: str, trial_id: str,
                          target_id: str) -> tuple[pd.DataFrame, str]:
    """Select by canonical parent word and normalized phone within one V1 trial."""
    task_id, word, phone = TARGETS[target_id]
    trial = store.trials.loc[store.trials.recording_id.astype(str).eq(recording_id)
                             & store.trials.trial_id.astype(str).eq(trial_id)]
    if len(trial) != 1 or str(trial.iloc[0].task_id) != task_id:
        return pd.DataFrame(), "wrong_task"
    if str(trial.iloc[0].prompt_id) != TARGET_PROMPTS[task_id]:
        return pd.DataFrame(), "wrong_prompt"
    words = store.words_for_trial(recording_id, trial_id)
    parents = words.loc[words.canonical_word_text.astype(str).str.casefold().eq(word)]
    if parents.empty:
        return pd.DataFrame(), "target_word_missing"
    phones = store.phones_for_trial(recording_id, trial_id)
    candidate = phones.loc[phones.phone_normalized.astype(str).str.upper().eq(phone)]
    if candidate.empty:
        return pd.DataFrame(), "target_phone_missing"
    matched = candidate.loc[candidate.parent_word_id.astype(str).isin(
        parents.word_id.astype(str))]
    if matched.empty:
        return pd.DataFrame(), "wrong_parent_word"
    if not matched.parent_word_text.astype(str).str.casefold().eq(word).all():
        return pd.DataFrame(), "wrong_parent_word"
    return matched.copy(), ""


def measure_target_frames(frames: pd.DataFrame, *, start: float, end: float,
                          token_index: int) -> dict[str, float | int | str]:
    """Fit each ordered/plausible Burg formant track in the target's 20–80% span."""
    left, right = transition_window(start, end)
    selected = frames.loc[frames.token_index.eq(token_index)
                          & frames.time_sec.between(left, right, inclusive="both")].copy()
    result: dict[str, float | int | str] = {
        "n_candidate_frames": len(selected), "n_valid_frames": 0,
        "qc_status": "FAILED", "qc_flags": "insufficient_formant_frames",
        "fit_convergence_status": "not_run",
    }
    for number in (1, 2):
        result[f"f{number}_slope_hz_s"] = np.nan
        result[f"f{number}_fitted_range_hz"] = np.nan
        result[f"f{number}_intercept_hz"] = np.nan
        result[f"f{number}_residual_mad_hz"] = np.nan
    plausible = np.logical_and.reduce([
        selected[f"f{number}_hz"].between(*FORMANT_BOUNDS[f"f{number}"]).to_numpy()
        for number in (1, 2, 3)
    ]) if not selected.empty else np.empty(0, dtype=bool)
    valid = selected.loc[selected.valid_ordered.astype(bool) & plausible]
    result["n_valid_frames"] = len(valid)
    if len(valid) < SLOPE_MIN_FRAMES:
        if len(selected) >= SLOPE_MIN_FRAMES and len(valid) < len(selected):
            result["qc_flags"] = "formant_track_invalid;insufficient_formant_frames"
        return result
    try:
        fits = {number: robust_bisquare_fit(valid.time_sec.to_numpy(dtype=float),
                                             valid[f"f{number}_hz"].to_numpy(dtype=float))
                for number in (1, 2)}
    except ValueError:
        result["qc_flags"] = "robust_fit_failed"
        return result
    if any(fit["status"] != "converged" for fit in fits.values()):
        result["qc_flags"] = "robust_fit_failed"
        result["fit_convergence_status"] = "not_converged"
        return result
    for number, fit in fits.items():
        slope = float(fit["slope"])
        fitted_range = abs(slope) * (right - left)
        if not np.isfinite(slope):
            result["qc_flags"] = "nonfinite_slope"
            return result
        if not np.isfinite(fitted_range):
            result["qc_flags"] = "nonfinite_range"
            return result
        result[f"f{number}_slope_hz_s"] = slope
        result[f"f{number}_fitted_range_hz"] = fitted_range
        result[f"f{number}_intercept_hz"] = float(fit["intercept"])
        result[f"f{number}_residual_mad_hz"] = float(fit["residual_mad"])
    result["fit_convergence_status"] = "converged"
    result["qc_status"] = "OK"
    result["qc_flags"] = ""
    return result
