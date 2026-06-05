"""Feature Analysis backend utilities for VSLP."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .column_mapping import ROLE_FEATURE, ROLE_QC, classify_columns, role_lists


def read_table(path: str | Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(p, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported table format: {suffix}")


def numeric_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def dataset_inventory(feature_df: pd.DataFrame, qc_df: pd.DataFrame | None, meta_df: pd.DataFrame | None, mapping: pd.DataFrame) -> pd.DataFrame:
    roles = role_lists(mapping)
    ids = roles.get("Identifier", [])
    features = roles.get("Feature", [])
    targets = roles.get("Target / label", [])
    covars = roles.get("Covariate", [])
    rows = [
        {"metric": "feature_table_rows", "value": len(feature_df)},
        {"metric": "feature_table_columns", "value": feature_df.shape[1]},
        {"metric": "detected_feature_columns", "value": len(features)},
        {"metric": "detected_identifier_columns", "value": len(ids)},
        {"metric": "detected_target_label_columns", "value": len(targets)},
        {"metric": "detected_covariate_columns", "value": len(covars)},
        {"metric": "qc_table_loaded", "value": qc_df is not None},
        {"metric": "qc_table_rows", "value": 0 if qc_df is None else len(qc_df)},
        {"metric": "metadata_table_loaded", "value": meta_df is not None},
        {"metric": "metadata_rows", "value": 0 if meta_df is None else len(meta_df)},
    ]
    if "subject_id" in feature_df.columns:
        rows.append({"metric": "unique_subjects", "value": int(feature_df["subject_id"].nunique(dropna=True))})
    if "task" in feature_df.columns:
        rows.append({"metric": "unique_tasks", "value": int(feature_df["task"].nunique(dropna=True))})
    return pd.DataFrame(rows)


def feature_distribution_summary(feature_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in numeric_columns(feature_df, feature_cols):
        x = pd.to_numeric(feature_df[c], errors="coerce")
        non = x.dropna()
        if non.empty:
            rows.append({"feature": c, "n": 0, "missing_fraction": float(x.isna().mean()), "mean": np.nan, "median": np.nan, "sd": np.nan, "iqr": np.nan, "min": np.nan, "max": np.nan, "robust_outlier_fraction": np.nan, "zero_variance": True})
            continue
        q1, q3 = np.nanpercentile(non, [25, 75])
        med = float(np.nanmedian(non))
        iqr = float(q3 - q1)
        mad = float(np.nanmedian(np.abs(non - med)))
        if mad > 0:
            rz = 0.6745 * (non - med) / mad
            out_frac = float((np.abs(rz) > 3.5).mean())
        elif iqr > 0:
            out_frac = float(((non < q1 - 1.5 * iqr) | (non > q3 + 1.5 * iqr)).mean())
        else:
            out_frac = 0.0
        rows.append({
            "feature": c,
            "n": int(non.size),
            "missing_fraction": float(x.isna().mean()),
            "mean": float(np.nanmean(non)),
            "median": med,
            "sd": float(np.nanstd(non, ddof=1)) if non.size > 1 else np.nan,
            "iqr": iqr,
            "min": float(np.nanmin(non)),
            "max": float(np.nanmax(non)),
            "robust_outlier_fraction": out_frac,
            "zero_variance": bool(non.nunique(dropna=True) <= 1),
        })
    return pd.DataFrame(rows)


def feature_qc_correlations(feature_df: pd.DataFrame, qc_df: pd.DataFrame | None, feature_cols: list[str]) -> pd.DataFrame:
    if qc_df is None or qc_df.empty:
        return pd.DataFrame(columns=["feature", "qc_variable", "spearman_rho", "n_pairwise"])
    # Merge if possible; otherwise row-align only when same length.
    join_keys = [k for k in ["record_key", "file_name", "subject_id"] if k in feature_df.columns and k in qc_df.columns]
    if join_keys:
        merged = feature_df.merge(qc_df, on=join_keys, how="inner", suffixes=("", "__qc"))
    elif len(feature_df) == len(qc_df):
        merged = pd.concat([feature_df.reset_index(drop=True), qc_df.reset_index(drop=True).add_suffix("__qc")], axis=1)
    else:
        return pd.DataFrame(columns=["feature", "qc_variable", "spearman_rho", "n_pairwise"])

    qc_mapping = classify_columns(qc_df, table_kind="qc")
    qc_cols_original = role_lists(qc_mapping).get("QC feature", [])
    qc_cols = []
    for q in qc_cols_original:
        qc_cols.append(q if q in merged.columns else f"{q}__qc")
    qc_cols = [q for q in qc_cols if q in merged.columns and pd.api.types.is_numeric_dtype(merged[q])]
    feat_cols = numeric_columns(merged, [c for c in feature_cols if c in merged.columns])
    rows = []
    for f in feat_cols:
        for q in qc_cols:
            pair = merged[[f, q]].dropna()
            if len(pair) < 4 or pair[f].nunique() <= 1 or pair[q].nunique() <= 1:
                rho = np.nan
            else:
                rho = float(pair[f].corr(pair[q], method="spearman"))
            rows.append({"feature": f, "qc_variable": q.replace("__qc", ""), "spearman_rho": rho, "n_pairwise": int(len(pair))})
    return pd.DataFrame(rows)


def reliability_screen(dist: pd.DataFrame, qc_corr: pd.DataFrame | None = None) -> pd.DataFrame:
    max_qc = {}
    if qc_corr is not None and not qc_corr.empty:
        tmp = qc_corr.copy()
        tmp["abs_rho"] = tmp["spearman_rho"].abs()
        max_qc = tmp.groupby("feature")["abs_rho"].max().to_dict()
    rows = []
    for _, r in dist.iterrows():
        reasons = []
        recommendation = "use"
        miss = float(r.get("missing_fraction", 0) or 0)
        out = float(r.get("robust_outlier_fraction", 0) or 0)
        zero = bool(r.get("zero_variance", False))
        qc = float(max_qc.get(r["feature"], np.nan))
        if zero:
            recommendation = "exclude_or_review"
            reasons.append("zero variance")
        if miss >= 0.50:
            recommendation = "exclude_or_review"
            reasons.append("high missingness")
        elif miss >= 0.20 and recommendation == "use":
            recommendation = "use_with_caution"
            reasons.append("moderate missingness")
        if out >= 0.20 and recommendation == "use":
            recommendation = "use_with_caution"
            reasons.append("many robust outliers")
        if not np.isnan(qc) and qc >= 0.70:
            recommendation = "exclude_or_review" if recommendation != "exclude_or_review" else recommendation
            reasons.append("strong QC association")
        elif not np.isnan(qc) and qc >= 0.50 and recommendation == "use":
            recommendation = "use_with_caution"
            reasons.append("moderate QC association")
        rows.append({
            "feature": r["feature"],
            "recommendation": recommendation,
            "missing_fraction": miss,
            "robust_outlier_fraction": out,
            "zero_variance": zero,
            "max_abs_qc_spearman": qc,
            "review_reason": "; ".join(reasons) if reasons else "no major screening issue detected",
        })
    return pd.DataFrame(rows)
