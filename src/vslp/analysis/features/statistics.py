from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


def numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in columns:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def robust_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    med = x.median(skipna=True)
    mad = (x - med).abs().median(skipna=True)
    if pd.isna(mad) or mad == 0:
        return pd.Series(np.nan, index=x.index)
    return 0.67448975 * (x - med) / mad


def distribution_summary(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        valid = x.dropna()
        if valid.empty:
            rows.append({"feature": col, "n_valid": 0, "missing_fraction": 1.0})
            continue
        rz = robust_z(x)
        rows.append({
            "feature": col,
            "n_valid": int(valid.shape[0]),
            "missing_fraction": float(x.isna().mean()),
            "mean": float(valid.mean()),
            "median": float(valid.median()),
            "sd": float(valid.std(ddof=1)) if valid.shape[0] > 1 else np.nan,
            "iqr": float(valid.quantile(0.75) - valid.quantile(0.25)),
            "q05": float(valid.quantile(0.05)),
            "q25": float(valid.quantile(0.25)),
            "q75": float(valid.quantile(0.75)),
            "q95": float(valid.quantile(0.95)),
            "min": float(valid.min()),
            "max": float(valid.max()),
            "robust_outlier_fraction_abs_z_gt_3_5": float((rz.abs() > 3.5).mean(skipna=True)),
            "zero_variance": bool(valid.nunique(dropna=True) <= 1),
        })
    return pd.DataFrame(rows)


def missingness_by_row(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    if not feature_cols:
        return pd.DataFrame({"row_index": df.index, "feature_missing_fraction": []})
    missing = df[list(feature_cols)].isna().mean(axis=1)
    return pd.DataFrame({"row_index": df.index, "feature_missing_fraction": missing})


def outlier_flags(df: pd.DataFrame, feature_cols: Sequence[str], threshold: float = 3.5) -> pd.DataFrame:
    rows = []
    id_cols = [c for c in ["record_key", "file_name", "subject_id", "session_id", "task", "iteration"] if c in df.columns]
    for col in feature_cols:
        rz = robust_z(df[col])
        mask = rz.abs() > threshold
        for idx in df.index[mask.fillna(False)]:
            row = {"row_index": int(idx), "feature": col, "value": pd.to_numeric(df.loc[idx, col], errors="coerce"), "robust_z": rz.loc[idx]}
            for idc in id_cols:
                row[idc] = df.loc[idx, idc]
            rows.append(row)
    return pd.DataFrame(rows)


def spearman_correlation(df: pd.DataFrame, cols: Sequence[str], max_cols: int = 80) -> pd.DataFrame:
    cols = list(cols)[:max_cols]
    if len(cols) < 2:
        return pd.DataFrame()
    num = numeric_frame(df, cols)
    return num.corr(method="spearman", min_periods=max(3, min(8, len(df)//3)))


def qc_feature_correlations(feature_df: pd.DataFrame, qc_df: pd.DataFrame, feature_cols: Sequence[str], qc_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for f in feature_cols:
        xf = pd.to_numeric(feature_df[f], errors="coerce")
        for q in qc_cols:
            if q not in qc_df.columns:
                continue
            xq = pd.to_numeric(qc_df[q], errors="coerce")
            both = pd.concat([xf, xq], axis=1).dropna()
            if both.shape[0] < 4:
                rho = np.nan
            else:
                rho = both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman")
            rows.append({"feature": f, "qc_feature": q, "spearman_rho": rho, "n_pairwise": int(both.shape[0])})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_spearman_rho"] = out["spearman_rho"].abs()
        out = out.sort_values("abs_spearman_rho", ascending=False, na_position="last")
    return out
