"""Manual-reference alignment and downstream sensitivity comparisons.

These are descriptive engineering reports, not clinical acceptance gates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SENSITIVITY_FEATURES = frozenset({
    "mean_word_duration_s", "delta_v_s", "varco_v_pct", "npvi_v_pct",
    "f1_token_hz", "f2_token_hz", "f3_token_hz",
    "f1_vowel_median_hz", "f2_vowel_median_hz", "f3_vowel_median_hz",
    "vsa3_iau_hz2", "fri_iau", "sfri_iau", "vai_iau", "fcr_iau",
})


def _summary(values: np.ndarray, tolerances: tuple[float, ...]) -> dict:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"n": 0, "median": np.nan, "iqr": np.nan, "p95": np.nan,
                **{f"within_{t:g}": np.nan for t in tolerances}}
    return {"n": len(finite), "median": float(np.median(finite)),
            "iqr": float(np.percentile(finite, 75) - np.percentile(finite, 25)),
            "p95": float(np.percentile(finite, 95)),
            **{f"within_{t:g}": float(np.mean(finite <= t)) for t in tolerances}}


def compare_boundaries(aligned: pd.DataFrame, manual: pd.DataFrame, *,
                       token_type: str, tolerances_sec: tuple[float, ...] = ()) -> pd.DataFrame:
    """Pair by explicit recording/token keys; unmatched tokens are reported."""
    index = "word_index" if token_type == "word" else "phone_index"
    keys = ["recording_id", index]
    if token_type == "phone":
        keys.insert(1, "word_index")
    if aligned.duplicated(keys).any() or manual.duplicated(keys).any():
        raise ValueError("duplicate_reference_token_key")
    pair = aligned.merge(manual, on=keys, how="outer", suffixes=("_mfa", "_manual"), indicator=True)
    rows = [{"metric": "matched_tokens", "n": int((pair._merge == "both").sum()),
             "unmatched_mfa": int((pair._merge == "left_only").sum()),
             "unmatched_manual": int((pair._merge == "right_only").sum())}]
    matched = pair.loc[pair._merge.eq("both")]
    for name, first, second in (
        ("onset_absolute_error_sec", "start_sec_mfa", "start_sec_manual"),
        ("offset_absolute_error_sec", "end_sec_mfa", "end_sec_manual"),
        ("duration_absolute_error_sec", "duration_sec_mfa", "duration_sec_manual"),
    ):
        errors = np.abs(matched[first].to_numpy(dtype=float) - matched[second].to_numpy(dtype=float))
        rows.append({"metric": name, **_summary(errors, tolerances_sec)})
    return pd.DataFrame(rows)


def compare_feature_sensitivity(mfa_values: pd.DataFrame, manual_values: pd.DataFrame,
                                *, tolerances: tuple[float, ...] = ()) -> pd.DataFrame:
    """Compare the same feature and optional target dimensions across boundaries."""
    keys = ["recording_id", "feature_id"]
    for dimension in ("target_vowel", "word_index", "phone_index"):
        if dimension in mfa_values and dimension in manual_values:
            keys.append(dimension)
    for table in (mfa_values, manual_values):
        if table.duplicated(keys).any():
            raise ValueError("duplicate_feature_sensitivity_key")
    pair = mfa_values.merge(manual_values, on=keys, how="outer", suffixes=("_mfa", "_manual"),
                            indicator=True)
    rows = []
    for feature_id, group in pair.groupby("feature_id"):
        if feature_id not in SENSITIVITY_FEATURES:
            continue
        matched = group.loc[group._merge.eq("both")]
        delta = np.abs(matched.value_mfa.to_numpy(dtype=float)
                       - matched.value_manual.to_numpy(dtype=float))
        rows.append({"feature_id": feature_id,
                     "unmatched_mfa": int((group._merge == "left_only").sum()),
                     "unmatched_manual": int((group._merge == "right_only").sum()),
                     **_summary(delta, tolerances)})
    return pd.DataFrame(rows)
