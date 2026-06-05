"""Feature Analysis backend utilities for VSLP.

These functions are deliberately descriptive and conservative. They are designed
for feature audit, QC screening, and dataset orientation before any ML modeling.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .column_mapping import (
    ROLE_FEATURE, ROLE_QC, ROLE_IDENTIFIER, ROLE_TARGET, ROLE_COVARIATE,
    ROLE_TASK, ROLE_TIME, classify_columns, role_lists, normalize_name
)


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


def _first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    norm = {normalize_name(c): c for c in df.columns}
    for n in names:
        if n in norm:
            return norm[n]
    return None


def dataset_inventory(feature_df: pd.DataFrame, qc_df: pd.DataFrame | None, meta_df: pd.DataFrame | None, mapping: pd.DataFrame) -> pd.DataFrame:
    roles = role_lists(mapping)
    ids = roles.get(ROLE_IDENTIFIER, [])
    features = roles.get(ROLE_FEATURE, [])
    targets = roles.get(ROLE_TARGET, [])
    covars = roles.get(ROLE_COVARIATE, [])
    tasks = roles.get(ROLE_TASK, [])
    times = roles.get(ROLE_TIME, [])
    numeric_features = numeric_columns(feature_df, features)
    rows = [
        {"section": "input", "metric": "feature_table_rows", "value": len(feature_df), "interpretation": "Number of records available for analysis."},
        {"section": "input", "metric": "feature_table_columns", "value": feature_df.shape[1], "interpretation": "All columns in the primary feature table."},
        {"section": "roles", "metric": "detected_feature_columns", "value": len(features), "interpretation": "Columns currently assigned as analyzable features."},
        {"section": "roles", "metric": "numeric_feature_columns", "value": len(numeric_features), "interpretation": "Feature columns usable for numeric summaries and correlations."},
        {"section": "roles", "metric": "detected_identifier_columns", "value": len(ids), "interpretation": "Identifier columns used for traceability and table linking."},
        {"section": "roles", "metric": "detected_target_label_columns", "value": len(targets), "interpretation": "Clinical labels or outcomes; descriptive only in this GUI."},
        {"section": "roles", "metric": "detected_covariate_columns", "value": len(covars), "interpretation": "Potential grouping or adjustment variables."},
        {"section": "roles", "metric": "detected_task_columns", "value": len(tasks), "interpretation": "Task/prompt descriptors available for stratified analysis."},
        {"section": "roles", "metric": "detected_time_columns", "value": len(times), "interpretation": "Session/date/time variables available for longitudinal views."},
        {"section": "linked_tables", "metric": "qc_table_loaded", "value": bool(qc_df is not None), "interpretation": "QC table available for feature-quality association screening."},
        {"section": "linked_tables", "metric": "qc_table_rows", "value": 0 if qc_df is None else len(qc_df), "interpretation": "Rows in optional QC table."},
        {"section": "linked_tables", "metric": "metadata_table_loaded", "value": bool(meta_df is not None), "interpretation": "Metadata table available for labels/covariates."},
        {"section": "linked_tables", "metric": "metadata_rows", "value": 0 if meta_df is None else len(meta_df), "interpretation": "Rows in optional metadata table."},
    ]
    subject_col = _first_present(feature_df, ["subject_id", "participant_id", "patient_id"])
    session_col = _first_present(feature_df, ["session_id", "visit_id", "clinical_visit_id"])
    task_col = _first_present(feature_df, ["task", "task_name", "prompt"])
    target_col = _first_present(feature_df, ["diagnosis", "severity_bin", "severity_score", "alsfrs_total", "alsfrs_bulbar"])
    if subject_col:
        rows.append({"section": "design", "metric": "unique_subjects", "value": int(feature_df[subject_col].nunique(dropna=True)), "interpretation": f"Unique values in {subject_col}."})
    if session_col:
        rows.append({"section": "design", "metric": "unique_sessions", "value": int(feature_df[session_col].nunique(dropna=True)), "interpretation": f"Unique values in {session_col}."})
    if task_col:
        rows.append({"section": "design", "metric": "unique_tasks", "value": int(feature_df[task_col].nunique(dropna=True)), "interpretation": f"Unique values in {task_col}."})
    if target_col:
        rows.append({"section": "design", "metric": "primary_label_detected", "value": target_col, "interpretation": "First detected label column; used for descriptive counts only."})
    if numeric_features:
        miss = feature_df[numeric_features].isna().mean().mean()
        rows.append({"section": "data_quality", "metric": "mean_feature_missingness", "value": round(float(miss), 4), "interpretation": "Average missingness across numeric feature columns."})
    return pd.DataFrame(rows)


def role_summary(mapping: pd.DataFrame) -> pd.DataFrame:
    if mapping.empty:
        return pd.DataFrame(columns=["role", "n_columns", "example_columns"])
    rows = []
    for role, sub in mapping.groupby("role", dropna=False):
        examples = ", ".join(sub["column"].astype(str).head(8).tolist())
        rows.append({"role": role, "n_columns": int(len(sub)), "example_columns": examples})
    return pd.DataFrame(rows).sort_values("role")


def design_overview(feature_df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    cols = [
        ("subject", ["subject_id", "participant_id", "patient_id"]),
        ("session", ["session_id", "visit_id", "clinical_visit_id"]),
        ("task", ["task", "task_name", "prompt"]),
        ("diagnosis", ["diagnosis", "dx", "group"]),
        ("severity_bin", ["severity_bin", "severity_class"]),
        ("sex_or_gender", ["sex", "gender"]),
        ("device", ["device", "microphone", "site"]),
    ]
    rows = []
    for label, names in cols:
        c = _first_present(feature_df, names)
        if not c:
            rows.append({"variable_type": label, "column": "not_detected", "n_unique": 0, "n_missing": len(feature_df), "top_values": ""})
            continue
        vc = feature_df[c].astype(str).replace("nan", np.nan).value_counts(dropna=True).head(8)
        rows.append({
            "variable_type": label,
            "column": c,
            "n_unique": int(feature_df[c].nunique(dropna=True)),
            "n_missing": int(feature_df[c].isna().sum()),
            "top_values": "; ".join([f"{idx}: {val}" for idx, val in vc.items()]),
        })
    return pd.DataFrame(rows)


def feature_family_overview(feature_df: pd.DataFrame, feature_cols: list[str], registry: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    subsystem_lookup: dict[str, str] = {}
    if registry is not None and not registry.empty:
        feature_col = next((c for c in ["feature", "feature_name", "name", "column"] if c in registry.columns), None)
        subsystem_col = next((c for c in ["subsystem", "family", "feature_family", "group"] if c in registry.columns), None)
        if feature_col and subsystem_col:
            for _, r in registry[[feature_col, subsystem_col]].dropna().iterrows():
                subsystem_lookup[str(r[feature_col])] = str(r[subsystem_col])
    for c in feature_cols:
        fam = subsystem_lookup.get(c, "unclassified")
        x = pd.to_numeric(feature_df[c], errors="coerce") if c in feature_df.columns else pd.Series(dtype=float)
        rows.append({
            "feature": c,
            "family_or_subsystem": fam,
            "numeric": bool(c in feature_df.columns and pd.api.types.is_numeric_dtype(feature_df[c])),
            "missing_fraction": float(x.isna().mean()) if len(x) else np.nan,
            "n_nonmissing": int(x.notna().sum()) if len(x) else 0,
            "n_unique": int(x.nunique(dropna=True)) if len(x) else 0,
        })
    if not rows:
        return pd.DataFrame(columns=["family_or_subsystem", "n_features", "mean_missing_fraction", "n_numeric_features"])
    df = pd.DataFrame(rows)
    return df.groupby("family_or_subsystem", dropna=False).agg(
        n_features=("feature", "count"),
        n_numeric_features=("numeric", "sum"),
        mean_missing_fraction=("missing_fraction", "mean"),
        median_nonmissing=("n_nonmissing", "median"),
    ).reset_index().sort_values("n_features", ascending=False)


def group_counts(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, candidates in [
        ("task", ["task", "task_name", "prompt"]),
        ("diagnosis", ["diagnosis", "dx", "group"]),
        ("severity_bin", ["severity_bin", "severity_class"]),
        ("sex_or_gender", ["sex", "gender"]),
        ("session", ["session_id", "visit_id", "clinical_visit_id"]),
    ]:
        c = _first_present(feature_df, candidates)
        if not c:
            continue
        vc = feature_df[c].astype(str).replace("nan", np.nan).value_counts(dropna=True).reset_index()
        vc.columns = ["level", "n_rows"]
        vc.insert(0, "group_variable", group_name)
        vc.insert(1, "column", c)
        rows.append(vc)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["group_variable", "column", "level", "n_rows"])


def feature_distribution_summary(feature_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in numeric_columns(feature_df, feature_cols):
        x = pd.to_numeric(feature_df[c], errors="coerce")
        non = x.dropna()
        if non.empty:
            rows.append({"feature": c, "n": 0, "missing_fraction": float(x.isna().mean()), "mean": np.nan, "median": np.nan, "sd": np.nan, "iqr": np.nan, "q05": np.nan, "q95": np.nan, "min": np.nan, "max": np.nan, "robust_outlier_fraction": np.nan, "zero_variance": True})
            continue
        q05, q1, q3, q95 = np.nanpercentile(non, [5, 25, 75, 95])
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
            "q05": float(q05),
            "q95": float(q95),
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
    qc_cols_original = role_lists(qc_mapping).get(ROLE_QC, [])
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
