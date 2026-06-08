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
        q01, q05, q1, q3, q95, q99 = np.nanpercentile(non, [1, 5, 25, 75, 95, 99])
        med = float(np.nanmedian(non))
        mean = float(np.nanmean(non))
        sd = float(np.nanstd(non, ddof=1)) if non.size > 1 else np.nan
        iqr = float(q3 - q1)
        mad = float(np.nanmedian(np.abs(non - med)))
        unique_n = int(non.nunique(dropna=True))
        zero_var = bool(unique_n <= 1)
        near_zero_var = bool((iqr == 0 and unique_n <= max(2, int(0.03 * non.size))) or (sd == 0 if pd.notna(sd) else False))
        if mad > 0:
            rz = 0.6745 * (non - med) / mad
            out_frac = float((np.abs(rz) > 3.5).mean())
            robust_method = "median_mad"
        elif iqr > 0:
            rz = (non - med) / (iqr / 1.349)
            out_frac = float(((non < q1 - 1.5 * iqr) | (non > q3 + 1.5 * iqr)).mean())
            robust_method = "iqr_scaled"
        else:
            out_frac = 0.0
            robust_method = "zero_variance"
        skew_proxy = float((mean - med) / sd) if pd.notna(sd) and sd > 0 else np.nan
        tail_ratio = float((q95 - q05) / iqr) if iqr > 0 else np.nan
        robust_cv = float(iqr / abs(med)) if med != 0 and iqr > 0 else np.nan
        floor_frac = float((non == np.nanmin(non)).mean())
        ceiling_frac = float((non == np.nanmax(non)).mean())
        rows.append({
            "feature": c,
            "n": int(non.size),
            "missing_fraction": float(x.isna().mean()),
            "mean": mean,
            "median": med,
            "sd": sd,
            "iqr": iqr,
            "q01": float(q01),
            "q05": float(q05),
            "q25": float(q1),
            "q75": float(q3),
            "q95": float(q95),
            "q99": float(q99),
            "min": float(np.nanmin(non)),
            "max": float(np.nanmax(non)),
            "unique_values": unique_n,
            "robust_outlier_fraction": out_frac,
            "robust_outlier_method": robust_method,
            "skew_proxy_mean_minus_median_over_sd": skew_proxy,
            "tail_ratio_q95_q05_over_iqr": tail_ratio,
            "robust_cv_iqr_over_abs_median": robust_cv,
            "floor_fraction": floor_frac,
            "ceiling_fraction": ceiling_frac,
            "zero_variance": zero_var,
            "near_zero_variance": near_zero_var,
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



QC_FAMILY_PREFIXES = [
    ("qadd", "Additive interference"),
    ("qgain", "Gain / level dynamics"),
    ("qrev", "Reverberation / echo"),
    ("qchan", "Channel / device / platform"),
    ("qdist", "Nonlinear distortion"),
    ("qtemp", "Temporal discontinuities"),
    ("qdrop", "Temporal discontinuities"),
]

QC_FAMILY_MECHANISMS = {
    "Additive interference": "External acoustic energy superimposed on speech: background noise, competing speech, hum, transient pause noise.",
    "Gain / level dynamics": "Time-varying amplitude scaling: microphone distance, input level, automatic gain control, level drift.",
    "Reverberation / echo": "Convolutional room effects: delayed reflections, echo tails, boundary smearing, room acoustics.",
    "Channel / device / platform": "Recording-chain spectral transformation: microphone response, bandwidth limitation, codec/browser/platform processing.",
    "Nonlinear distortion": "Amplitude-dependent deformation: clipping, saturation, overload, nonlinear compression.",
    "Temporal discontinuities": "Disruption of time structure: dropouts, skips, glitches, missing or repeated segments, abrupt energy jumps.",
    "unclassified": "QC-like variable not recognized from naming conventions; interpret manually.",
}


def qc_family_from_name(name: str) -> str:
    n = normalize_name(str(name))
    for prefix, family in QC_FAMILY_PREFIXES:
        if n.startswith(prefix):
            return family
    if any(tok in n for tok in ["snr", "noise", "pause_rms", "hum", "interference"]):
        return "Additive interference"
    if any(tok in n for tok in ["gain", "rms", "level", "agc", "peak_to_rms"]):
        return "Gain / level dynamics"
    if any(tok in n for tok in ["reverb", "echo", "tail", "decay", "blur"]):
        return "Reverberation / echo"
    if any(tok in n for tok in ["channel", "device", "codec", "band", "centroid", "rolloff", "tilt", "crest"]):
        return "Channel / device / platform"
    if any(tok in n for tok in ["clip", "dist", "saturat", "overload"]):
        return "Nonlinear distortion"
    if any(tok in n for tok in ["drop", "jump", "glitch", "skip", "discontinu", "zero_run", "temp"]):
        return "Temporal discontinuities"
    return "unclassified"


def _qc_numeric_columns(qc_df: pd.DataFrame | None) -> list[str]:
    if qc_df is None or qc_df.empty:
        return []
    mapping = classify_columns(qc_df, table_kind="qc")
    proposed = role_lists(mapping).get(ROLE_QC, [])
    if proposed:
        cols = [c for c in proposed if c in qc_df.columns and pd.api.types.is_numeric_dtype(qc_df[c])]
    else:
        cols = [c for c in qc_df.columns if pd.api.types.is_numeric_dtype(qc_df[c])]
    # Exclude obvious identifiers accidentally numeric.
    drop_tokens = ["id", "index", "subject", "participant", "file", "record"]
    clean = []
    for c in cols:
        n = normalize_name(c)
        if any(n == tok or n.endswith("_" + tok) for tok in drop_tokens):
            continue
        clean.append(c)
    return clean


def align_feature_qc(feature_df: pd.DataFrame, qc_df: pd.DataFrame | None) -> tuple[pd.DataFrame | None, str]:
    """Return feature rows aligned with optional QC columns and a readable alignment note."""
    if qc_df is None or qc_df.empty:
        return None, "No QC table supplied."
    join_keys = [k for k in ["record_key", "file_name", "filename", "source_file", "audio_file", "subject_id", "participant_id"] if k in feature_df.columns and k in qc_df.columns]
    if join_keys:
        key = join_keys[0]
        merged = feature_df.merge(qc_df, on=key, how="left", suffixes=("", "__qc"))
        return merged, f"QC table aligned to feature table using shared key: {key}."
    if len(feature_df) == len(qc_df):
        merged = pd.concat([feature_df.reset_index(drop=True), qc_df.reset_index(drop=True).add_suffix("__qc")], axis=1)
        return merged, "QC table aligned by row order because no shared key was detected and row counts matched. Verify this assumption."
    return None, "QC table could not be aligned to feature rows: no shared key and row counts differ."


def qc_metric_catalog(qc_df: pd.DataFrame | None) -> pd.DataFrame:
    cols = _qc_numeric_columns(qc_df)
    if qc_df is None or qc_df.empty or not cols:
        return pd.DataFrame(columns=["qc_variable", "artifact_family", "mechanism", "n_valid", "missing_fraction", "median", "iqr", "min", "max", "n_unique", "interpretation"])
    rows = []
    for c in cols:
        x = pd.to_numeric(qc_df[c], errors="coerce")
        q25 = x.quantile(0.25); q75 = x.quantile(0.75)
        fam = qc_family_from_name(c)
        rows.append({
            "qc_variable": c,
            "artifact_family": fam,
            "mechanism": QC_FAMILY_MECHANISMS.get(fam, ""),
            "n_valid": int(x.notna().sum()),
            "missing_fraction": float(x.isna().mean()),
            "median": float(x.median()) if x.notna().any() else np.nan,
            "iqr": float(q75 - q25) if pd.notna(q25) and pd.notna(q75) else np.nan,
            "min": float(x.min()) if x.notna().any() else np.nan,
            "max": float(x.max()) if x.notna().any() else np.nan,
            "n_unique": int(x.nunique(dropna=True)),
            "interpretation": _qc_metric_interpretation(c, fam),
        })
    return pd.DataFrame(rows).sort_values(["artifact_family", "qc_variable"])


def _qc_metric_interpretation(name: str, family: str) -> str:
    n = normalize_name(name)
    if family == "Additive interference":
        if "speech_pause_level_diff" in n or "snr" in n:
            return "Lower contrast usually indicates more background interference relative to speech."
        return "Higher or more variable pause energy/noise structure suggests additive contamination."
    if family == "Gain / level dynamics":
        return "Large variability, drift, or extreme peaks suggest unstable recording level, AGC, distance change, or headroom risk."
    if family == "Reverberation / echo":
        return "Large tail, slow decay, or boundary blur suggests room reflections that can smear timing and spectral features."
    if family == "Channel / device / platform":
        return "Extreme spectral centroid/rolloff/high-band/tilt values may indicate device filtering, bandwidth limitation, or platform processing."
    if family == "Nonlinear distortion":
        return "Nonzero clipping or near-clipping fractions indicate overload or amplitude distortion. Sparse events can still be important."
    if family == "Temporal discontinuities":
        return "Energy jumps, dropouts, or zero runs suggest skipped audio, glitches, or recording/transmission discontinuities."
    return "Interpret with the QC documentation and feature definition."


def _robust_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    med = x.median(skipna=True)
    mad = (x - med).abs().median(skipna=True)
    if pd.isna(med):
        return pd.Series(np.nan, index=x.index)
    if pd.notna(mad) and mad > 0:
        return 0.6745 * (x - med) / mad
    q25, q75 = x.quantile(0.25), x.quantile(0.75)
    iqr = q75 - q25
    if pd.notna(iqr) and iqr > 0:
        return (x - med) / (iqr / 1.349)
    return pd.Series(0.0, index=x.index)


def qc_row_burden_summary(qc_df: pd.DataFrame | None) -> pd.DataFrame:
    cols = _qc_numeric_columns(qc_df)
    if qc_df is None or qc_df.empty or not cols:
        return pd.DataFrame(columns=["row_index", "total_qc_flags", "qc_flag_fraction", "max_abs_qc_z", "elevated_qc_families", "top_qc_variable"])
    flags = pd.DataFrame(index=qc_df.index)
    zvals = pd.DataFrame(index=qc_df.index)
    for c in cols:
        z = _robust_z(qc_df[c]).abs()
        zvals[c] = z
        flags[c] = z >= 3.5
    out = pd.DataFrame({
        "row_index": qc_df.index.astype(int),
        "total_qc_flags": flags.sum(axis=1).astype(int),
        "qc_flag_fraction": flags.mean(axis=1).astype(float),
        "max_abs_qc_z": zvals.max(axis=1, skipna=True),
    })
    elevated = []
    topvar = []
    for idx in qc_df.index:
        fams = sorted({qc_family_from_name(c) for c in cols if bool(flags.loc[idx, c])})
        elevated.append("; ".join(fams))
        if zvals.loc[idx].notna().any():
            topvar.append(str(zvals.loc[idx].idxmax()))
        else:
            topvar.append("")
    out["elevated_qc_families"] = elevated
    out["top_qc_variable"] = topvar
    id_candidates = [c for c in ["record_key", "file_name", "filename", "source_file", "subject_id", "participant_id", "task", "task_name"] if c in qc_df.columns]
    for c in reversed(id_candidates):
        out.insert(1, c, qc_df[c].astype(str).values)
    return out.sort_values(["total_qc_flags", "max_abs_qc_z"], ascending=[False, False])


def qc_family_burden_summary(qc_df: pd.DataFrame | None) -> pd.DataFrame:
    catalog = qc_metric_catalog(qc_df)
    if qc_df is None or qc_df.empty or catalog.empty:
        return pd.DataFrame(columns=["artifact_family", "n_qc_metrics", "mean_missing_fraction", "median_metric_iqr", "median_row_flag_fraction", "mechanism", "interpretation"])
    row_flags = pd.DataFrame(index=qc_df.index)
    rows = []
    for family, sub in catalog.groupby("artifact_family"):
        cols = [c for c in sub["qc_variable"].tolist() if c in qc_df.columns]
        if not cols:
            continue
        fam_flags = pd.DataFrame({c: (_robust_z(qc_df[c]).abs() >= 3.5) for c in cols}, index=qc_df.index)
        row_flags[family] = fam_flags.mean(axis=1)
        rows.append({
            "artifact_family": family,
            "n_qc_metrics": int(len(cols)),
            "mean_missing_fraction": float(sub["missing_fraction"].mean()),
            "median_metric_iqr": float(pd.to_numeric(sub["iqr"], errors="coerce").median()),
            "median_row_flag_fraction": float(row_flags[family].median(skipna=True)),
            "mechanism": QC_FAMILY_MECHANISMS.get(family, ""),
            "interpretation": _qc_family_interpretation(family),
        })
    return pd.DataFrame(rows).sort_values("artifact_family")


def _qc_family_interpretation(family: str) -> str:
    if family == "Additive interference":
        return "Review whether background noise or competing speech could alter spectral, cepstral, segmentation, and pause-based features."
    if family == "Gain / level dynamics":
        return "Review whether level instability, AGC, or microphone distance changes could alter intensity, perturbation, CPP/HNR, or clipping-sensitive features."
    if family == "Reverberation / echo":
        return "Review whether room reflections could smear boundaries, inflate pause/tail energy, and bias spectral or timing features."
    if family == "Channel / device / platform":
        return "Review whether recording-chain filtering or codec/browser effects could bias formants, spectral tilt, high-band ratios, and voice-quality measures."
    if family == "Nonlinear distortion":
        return "Review any nonzero clipping/near-clipping carefully; sparse clipping can disproportionately affect perturbation and spectral measures."
    if family == "Temporal discontinuities":
        return "Review whether dropouts or glitches could distort duration, pauses, rhythm, and segment-level summaries."
    return "Review manually."


def feature_qc_family_association(qc_corr: pd.DataFrame | None) -> pd.DataFrame:
    if qc_corr is None or qc_corr.empty or not {"feature", "qc_variable", "spearman_rho"}.issubset(qc_corr.columns):
        return pd.DataFrame(columns=["feature", "artifact_family", "max_abs_spearman", "median_abs_spearman", "n_qc_metrics", "strongest_qc_variable", "strongest_direction", "review_priority", "interpretation"])
    df = qc_corr.copy()
    df["abs_spearman"] = pd.to_numeric(df["spearman_rho"], errors="coerce").abs()
    df["artifact_family"] = df["qc_variable"].apply(qc_family_from_name)
    rows = []
    for (feature, family), sub in df.dropna(subset=["abs_spearman"]).groupby(["feature", "artifact_family"]):
        if sub.empty:
            continue
        imax = sub["abs_spearman"].idxmax()
        rho = float(sub.loc[imax, "spearman_rho"])
        max_abs = float(sub.loc[imax, "abs_spearman"])
        priority = "review" if max_abs >= 0.50 else "monitor" if max_abs >= 0.30 else "context"
        rows.append({
            "feature": feature,
            "artifact_family": family,
            "max_abs_spearman": max_abs,
            "median_abs_spearman": float(sub["abs_spearman"].median()),
            "n_qc_metrics": int(sub["qc_variable"].nunique()),
            "strongest_qc_variable": sub.loc[imax, "qc_variable"],
            "strongest_direction": "positive" if rho > 0 else "negative" if rho < 0 else "zero",
            "review_priority": priority,
            "interpretation": f"{feature} varies monotonically with {family} metrics; check whether this reflects physiology, task structure, or acquisition artifact before ML.",
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["review_priority", "max_abs_spearman"], ascending=[False, False])


def qc_missingness_associations(feature_df: pd.DataFrame, qc_df: pd.DataFrame | None, feature_cols: list[str]) -> pd.DataFrame:
    aligned, note = align_feature_qc(feature_df, qc_df)
    if aligned is None:
        return pd.DataFrame(columns=["feature", "qc_variable", "artifact_family", "spearman_rho", "abs_spearman", "n_pairwise", "alignment_note"])
    qc_cols = _qc_numeric_columns(qc_df)
    qc_lookup = {q: q if q in aligned.columns else f"{q}__qc" for q in qc_cols}
    rows = []
    for f in numeric_columns(feature_df, feature_cols):
        if f not in aligned.columns:
            continue
        miss = aligned[f].isna().astype(float)
        if miss.nunique(dropna=True) <= 1:
            continue
        for q_orig, q in qc_lookup.items():
            if q not in aligned.columns:
                continue
            pair = pd.DataFrame({"missing": miss, "qc": pd.to_numeric(aligned[q], errors="coerce")}).dropna()
            if len(pair) < 8 or pair["qc"].nunique() <= 1:
                continue
            rho = float(pair["missing"].corr(pair["qc"], method="spearman"))
            rows.append({"feature": f, "qc_variable": q_orig, "artifact_family": qc_family_from_name(q_orig), "spearman_rho": rho, "abs_spearman": abs(rho), "n_pairwise": int(len(pair)), "alignment_note": note})
    out = pd.DataFrame(rows)
    return out.sort_values("abs_spearman", ascending=False) if not out.empty else out


def qc_outlier_associations(outlier_flags: pd.DataFrame, qc_df: pd.DataFrame | None) -> pd.DataFrame:
    if qc_df is None or qc_df.empty or outlier_flags is None or outlier_flags.empty or "row_index" not in outlier_flags.columns:
        return pd.DataFrame(columns=["qc_variable", "artifact_family", "spearman_rho", "abs_spearman", "n_pairwise", "interpretation"])
    row_burden = outlier_flags.groupby("row_index").size().rename("feature_outlier_count").reset_index()
    tmp = pd.DataFrame({"row_index": qc_df.index.astype(int)}).merge(row_burden, on="row_index", how="left")
    tmp["feature_outlier_count"] = tmp["feature_outlier_count"].fillna(0)
    qc_cols = _qc_numeric_columns(qc_df)
    rows = []
    for q in qc_cols:
        pair = pd.DataFrame({"outliers": tmp["feature_outlier_count"], "qc": pd.to_numeric(qc_df[q], errors="coerce")}).dropna()
        if len(pair) < 8 or pair["outliers"].nunique() <= 1 or pair["qc"].nunique() <= 1:
            continue
        rho = float(pair["outliers"].corr(pair["qc"], method="spearman"))
        rows.append({"qc_variable": q, "artifact_family": qc_family_from_name(q), "spearman_rho": rho, "abs_spearman": abs(rho), "n_pairwise": int(len(pair)), "interpretation": "Positive values indicate rows with higher QC burden also tend to accumulate more feature outlier flags."})
    out = pd.DataFrame(rows)
    return out.sort_values("abs_spearman", ascending=False) if not out.empty else out


def qc_integration_summary(qc_df: pd.DataFrame | None, qc_corr: pd.DataFrame | None, family_summary: pd.DataFrame | None, missing_assoc: pd.DataFrame | None, outlier_assoc: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    loaded = qc_df is not None and not qc_df.empty
    qc_cols = _qc_numeric_columns(qc_df)
    rows.append({"metric": "qc_table_loaded", "value": bool(loaded), "interpretation": "Whether objective recording-quality metrics were supplied for artifact-aware interpretation."})
    rows.append({"metric": "qc_rows", "value": 0 if qc_df is None else int(len(qc_df)), "interpretation": "Number of QC records available."})
    rows.append({"metric": "numeric_qc_metrics", "value": int(len(qc_cols)), "interpretation": "Number of numeric QC indicators usable for correlation and burden summaries."})
    rows.append({"metric": "artifact_families_detected", "value": int(len({qc_family_from_name(c) for c in qc_cols if qc_family_from_name(c) != 'unclassified'})), "interpretation": "Number of recognized QC artifact families represented by supplied metrics."})
    if qc_corr is not None and not qc_corr.empty and "spearman_rho" in qc_corr.columns:
        absrho = pd.to_numeric(qc_corr["spearman_rho"], errors="coerce").abs()
        rows.append({"metric": "feature_qc_pairs_abs_rho_ge_0_30", "value": int((absrho >= 0.30).sum()), "interpretation": "Feature-QC monotonic associations requiring contextual review."})
        rows.append({"metric": "feature_qc_pairs_abs_rho_ge_0_50", "value": int((absrho >= 0.50).sum()), "interpretation": "Stronger feature-QC associations; prioritize for sensitivity checks."})
    else:
        rows.append({"metric": "feature_qc_pairs_abs_rho_ge_0_30", "value": 0, "interpretation": "No usable feature-QC association table was available."})
    if missing_assoc is not None and not missing_assoc.empty and "abs_spearman" in missing_assoc.columns:
        rows.append({"metric": "missingness_qc_pairs_abs_rho_ge_0_30", "value": int((missing_assoc["abs_spearman"] >= 0.30).sum()), "interpretation": "Cases where feature absence may be linked to artifact burden."})
    if outlier_assoc is not None and not outlier_assoc.empty and "abs_spearman" in outlier_assoc.columns:
        rows.append({"metric": "row_outlier_qc_pairs_abs_rho_ge_0_30", "value": int((outlier_assoc["abs_spearman"] >= 0.30).sum()), "interpretation": "Cases where row-level feature outlier burden co-varies with QC metrics."})
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



def missingness_feature_summary(feature_df: pd.DataFrame, feature_cols: list[str], registry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature-level missingness audit.

    This is intentionally descriptive. High missingness is not automatically bad:
    in clinical speech/kinematic datasets it can reflect task incompatibility,
    physiologic inability to produce valid support, segmentation failure, or a
    real data-quality problem. The output therefore gives review categories
    rather than deleting columns.
    """
    if not feature_cols:
        return pd.DataFrame(columns=["feature", "family_or_subsystem", "n_rows", "n_missing", "n_valid", "missing_fraction", "n_unique_valid", "numeric", "missingness_status", "interpretation"])
    subsystem_lookup: dict[str, str] = {}
    if registry is not None and not registry.empty:
        fcol = next((c for c in ["feature", "feature_name", "name", "column"] if c in registry.columns), None)
        scol = next((c for c in ["subsystem", "family", "feature_family", "group"] if c in registry.columns), None)
        if fcol and scol:
            for _, r in registry[[fcol, scol]].dropna().iterrows():
                subsystem_lookup[str(r[fcol])] = str(r[scol])
    rows = []
    n = len(feature_df)
    for c in feature_cols:
        if c not in feature_df.columns:
            continue
        x = feature_df[c]
        nmiss = int(x.isna().sum())
        nvalid = int(x.notna().sum())
        frac = float(nmiss / n) if n else np.nan
        if frac >= 0.80:
            status = "high_review"
            interp = "Very high missingness. Review task compatibility, feature implementation status, segmentation/QC support, and whether this feature should enter downstream analysis."
        elif frac >= 0.50:
            status = "review"
            interp = "Substantial missingness. Do not impute automatically; inspect pattern by task, group, subject/session, and QC burden."
        elif frac >= 0.20:
            status = "monitor"
            interp = "Moderate missingness. Usually usable with transparent reporting and sensitivity checks."
        else:
            status = "low"
            interp = "Low missingness. Still inspect outliers and QC sensitivity before modeling."
        rows.append({
            "feature": c,
            "family_or_subsystem": subsystem_lookup.get(c, "unclassified"),
            "n_rows": int(n),
            "n_missing": nmiss,
            "n_valid": nvalid,
            "missing_fraction": frac,
            "n_unique_valid": int(x.nunique(dropna=True)),
            "numeric": bool(pd.api.types.is_numeric_dtype(x)),
            "missingness_status": status,
            "interpretation": interp,
        })
    return pd.DataFrame(rows).sort_values(["missing_fraction", "feature"], ascending=[False, True])


def missingness_row_summary(feature_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Row/recording-level missingness with useful identifiers preserved."""
    id_candidates = ["record_key", "file_name", "subject_id", "session_id", "iteration", "task", "diagnosis", "severity_bin"]
    id_cols = [c for c in id_candidates if c in feature_df.columns]
    base = feature_df[id_cols].copy() if id_cols else pd.DataFrame(index=feature_df.index)
    base.insert(0, "row_index", range(len(feature_df)))
    base["missing_fraction_all_columns"] = feature_df.isna().mean(axis=1).values
    if feature_cols:
        base["missing_fraction_feature_columns"] = feature_df[feature_cols].isna().mean(axis=1).values
        base["n_missing_features"] = feature_df[feature_cols].isna().sum(axis=1).values
        base["n_valid_features"] = feature_df[feature_cols].notna().sum(axis=1).values
    else:
        base["missing_fraction_feature_columns"] = np.nan
        base["n_missing_features"] = np.nan
        base["n_valid_features"] = np.nan
    def status(frac: float) -> str:
        if pd.isna(frac): return "not_available"
        if frac >= 0.80: return "high_review"
        if frac >= 0.50: return "review"
        if frac >= 0.20: return "monitor"
        return "low"
    base["row_missingness_status"] = base["missing_fraction_feature_columns"].apply(status)
    return base.sort_values("missing_fraction_feature_columns", ascending=False, na_position="last")


def missingness_group_summary(feature_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Descriptive missingness by task/diagnosis/severity/sex/session/device.

    This is for bias detection. It does not run hypothesis tests because small-N
    and group imbalance are expected in clinical datasets.
    """
    if not feature_cols:
        return pd.DataFrame(columns=["group_variable", "column", "level", "n_rows", "mean_feature_missing_fraction", "median_feature_missing_fraction", "n_high_review_rows"])
    row_miss = feature_df[feature_cols].isna().mean(axis=1)
    candidates = [
        ("task", ["task", "task_name", "prompt"]),
        ("diagnosis", ["diagnosis", "dx", "group"]),
        ("severity_bin", ["severity_bin", "severity_class"]),
        ("sex_or_gender", ["sex", "gender"]),
        ("session", ["session_id", "visit_id", "clinical_visit_id"]),
        ("subject", ["subject_id", "participant_id", "patient_id"]),
        ("device", ["device", "microphone", "site"]),
        ("modality", ["modality"]),
    ]
    rows = []
    tmp = feature_df.copy()
    tmp["__feature_missing_fraction"] = row_miss
    for group_name, names in candidates:
        c = _first_present(tmp, names)
        if not c:
            continue
        grouped = tmp.groupby(c, dropna=True)["__feature_missing_fraction"]
        for level, vals in grouped:
            vals = pd.to_numeric(vals, errors="coerce").dropna()
            if vals.empty:
                continue
            rows.append({
                "group_variable": group_name,
                "column": c,
                "level": str(level),
                "n_rows": int(vals.shape[0]),
                "mean_feature_missing_fraction": float(vals.mean()),
                "median_feature_missing_fraction": float(vals.median()),
                "q75_feature_missing_fraction": float(vals.quantile(0.75)),
                "n_high_review_rows": int((vals >= 0.80).sum()),
                "n_review_or_higher_rows": int((vals >= 0.50).sum()),
                "interpretation": "Descriptive only. Compare groups cautiously; missingness can be confounded by disease severity, task, device, and repeated sessions.",
            })
    return pd.DataFrame(rows).sort_values(["group_variable", "mean_feature_missing_fraction"], ascending=[True, False]) if rows else pd.DataFrame(columns=["group_variable", "column", "level", "n_rows", "mean_feature_missing_fraction"])


def missingness_family_summary(feature_missing: pd.DataFrame) -> pd.DataFrame:
    if feature_missing is None or feature_missing.empty or "family_or_subsystem" not in feature_missing.columns:
        return pd.DataFrame(columns=["family_or_subsystem", "n_features", "mean_missing_fraction", "median_missing_fraction", "n_review_features"])
    df = feature_missing.copy()
    df["missing_fraction"] = pd.to_numeric(df["missing_fraction"], errors="coerce")
    return df.groupby("family_or_subsystem", dropna=False).agg(
        n_features=("feature", "count"),
        mean_missing_fraction=("missing_fraction", "mean"),
        median_missing_fraction=("missing_fraction", "median"),
        max_missing_fraction=("missing_fraction", "max"),
        n_review_features=("missingness_status", lambda x: int(pd.Series(x).isin(["review", "high_review"]).sum())),
    ).reset_index().sort_values("mean_missing_fraction", ascending=False)


def missingness_comissing_pairs(feature_df: pd.DataFrame, feature_cols: list[str], top_n: int = 40) -> pd.DataFrame:
    """Pairwise co-missingness among most-missing features."""
    if not feature_cols:
        return pd.DataFrame(columns=["feature_a", "feature_b", "co_missing_fraction", "n_co_missing"])
    cols = [c for c in feature_cols if c in feature_df.columns]
    miss_fr = feature_df[cols].isna().mean().sort_values(ascending=False)
    cols = miss_fr.head(top_n).index.tolist()
    rows = []
    n = len(feature_df)
    for i, a in enumerate(cols):
        ma = feature_df[a].isna()
        for b in cols[i+1:]:
            mb = feature_df[b].isna()
            nco = int((ma & mb).sum())
            rows.append({"feature_a": a, "feature_b": b, "co_missing_fraction": float(nco / n) if n else np.nan, "n_co_missing": nco})
    return pd.DataFrame(rows).sort_values("co_missing_fraction", ascending=False) if rows else pd.DataFrame(columns=["feature_a", "feature_b", "co_missing_fraction", "n_co_missing"])



def _registry_feature_metadata(registry: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    """Return registry metadata keyed by feature name when available."""
    if registry is None or registry.empty:
        return {}
    fcol = next((c for c in ["feature", "feature_name", "name", "column"] if c in registry.columns), None)
    if not fcol:
        return {}
    out: dict[str, dict[str, object]] = {}
    for _, row in registry.iterrows():
        feat = row.get(fcol)
        if pd.isna(feat):
            continue
        out[str(feat)] = {str(k): row[k] for k in registry.columns}
    return out


def _expected_bounds_from_meta(meta: dict[str, object]) -> tuple[float, float]:
    low_cols = ["expected_low", "expected_min", "physiologic_low", "range_low", "orientation_low"]
    high_cols = ["expected_high", "expected_max", "physiologic_high", "range_high", "orientation_high"]
    low = np.nan
    high = np.nan
    for c in low_cols:
        if c in meta:
            low = pd.to_numeric(pd.Series([meta[c]]), errors="coerce").iloc[0]
            if pd.notna(low):
                break
    for c in high_cols:
        if c in meta:
            high = pd.to_numeric(pd.Series([meta[c]]), errors="coerce").iloc[0]
            if pd.notna(high):
                break
    return float(low) if pd.notna(low) else np.nan, float(high) if pd.notna(high) else np.nan


def robust_outlier_flags(feature_df: pd.DataFrame, feature_cols: list[str], registry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Row-level robust outlier and expected-range flags.

    Uses robust z-scores based on median/MAD when possible, with IQR fallback.
    Expected ranges are only applied when a registry/policy table supplies usable
    low/high limits. These are review flags, not automatic exclusions.
    """
    id_candidates = ["file_name", "record_key", "subject_id", "session_id", "task", "iteration", "recording_date"]
    id_cols = [c for c in id_candidates if c in feature_df.columns]
    meta_lookup = _registry_feature_metadata(registry)
    rows = []
    for c in numeric_columns(feature_df, feature_cols):
        x = pd.to_numeric(feature_df[c], errors="coerce")
        valid = x.dropna()
        if valid.empty:
            continue
        med = float(np.nanmedian(valid))
        q1, q3 = np.nanpercentile(valid, [25, 75])
        iqr = float(q3 - q1)
        mad = float(np.nanmedian(np.abs(valid - med)))
        if mad > 0:
            rz = 0.6745 * (x - med) / mad
            robust_method = "median_mad"
        elif iqr > 0:
            rz = (x - med) / (iqr / 1.349)
            robust_method = "iqr_scaled"
        else:
            rz = pd.Series(np.nan, index=x.index)
            robust_method = "zero_variance"
        low, high = _expected_bounds_from_meta(meta_lookup.get(c, {}))
        outside_low = pd.Series(False, index=x.index)
        outside_high = pd.Series(False, index=x.index)
        if not np.isnan(low):
            outside_low = x < low
        if not np.isnan(high):
            outside_high = x > high
        robust_flag = rz.abs() > 3.5
        impossible_flag = outside_low | outside_high
        flag_any = robust_flag | impossible_flag
        for idx in x.index[flag_any.fillna(False)]:
            reasons = []
            if bool(robust_flag.loc[idx]):
                reasons.append("robust_z_abs_gt_3_5")
            if bool(outside_low.loc[idx]):
                reasons.append("below_expected_range")
            if bool(outside_high.loc[idx]):
                reasons.append("above_expected_range")
            review_level = "range_review" if bool(impossible_flag.loc[idx]) else "statistical_review"
            record = {
                "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "feature": c,
                "value": float(x.loc[idx]) if pd.notna(x.loc[idx]) else np.nan,
                "median": med,
                "iqr": iqr,
                "robust_z": float(rz.loc[idx]) if pd.notna(rz.loc[idx]) else np.nan,
                "expected_low": low,
                "expected_high": high,
                "flag_type": "; ".join(reasons),
                "review_level": review_level,
                "robust_method": robust_method,
            }
            for idc in id_cols:
                record[idc] = feature_df.loc[idx, idc]
            rows.append(record)
    return pd.DataFrame(rows)


def expected_range_flags(feature_df: pd.DataFrame, feature_cols: list[str], registry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature-level expected-range audit where registry bounds are available."""
    meta_lookup = _registry_feature_metadata(registry)
    rows = []
    for c in numeric_columns(feature_df, feature_cols):
        x = pd.to_numeric(feature_df[c], errors="coerce")
        valid = x.dropna()
        low, high = _expected_bounds_from_meta(meta_lookup.get(c, {}))
        has_range = not (np.isnan(low) and np.isnan(high))
        if valid.empty:
            n_below = n_above = 0
            frac = np.nan
            minv = maxv = np.nan
        else:
            below = (valid < low) if not np.isnan(low) else pd.Series(False, index=valid.index)
            above = (valid > high) if not np.isnan(high) else pd.Series(False, index=valid.index)
            n_below = int(below.sum())
            n_above = int(above.sum())
            frac = float((below | above).mean()) if has_range else np.nan
            minv = float(valid.min())
            maxv = float(valid.max())
        if not has_range:
            status = "no_registry_range"
            interp = "No expected range was supplied; evaluate using distribution/outlier and domain review."
        elif frac == 0:
            status = "within_range"
            interp = "All valid values are within the supplied expected range."
        elif frac < 0.05:
            status = "minor_review"
            interp = "Small fraction outside expected range; inspect rows and QC context."
        elif frac < 0.20:
            status = "review"
            interp = "Meaningful fraction outside expected range; inspect computation, task compatibility, and QC."
        else:
            status = "high_review"
            interp = "Large fraction outside expected range; do not use blindly for modeling."
        rows.append({
            "feature": c,
            "n_valid": int(valid.size),
            "expected_low": low,
            "expected_high": high,
            "min": minv,
            "max": maxv,
            "n_below_expected": n_below,
            "n_above_expected": n_above,
            "fraction_outside_expected": frac,
            "range_status": status,
            "interpretation": interp,
        })
    return pd.DataFrame(rows)


def distribution_review_summary(dist: pd.DataFrame, expected: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature-level distribution/outlier review summary."""
    if dist is None or dist.empty:
        return pd.DataFrame(columns=["feature", "distribution_status", "review_reason"])
    df = dist.copy()
    if expected is not None and not expected.empty and "feature" in expected.columns:
        keep = [c for c in ["feature", "fraction_outside_expected", "range_status"] if c in expected.columns]
        df = df.merge(expected[keep], on="feature", how="left")
    else:
        df["fraction_outside_expected"] = np.nan
        df["range_status"] = "no_registry_range"
    rows = []
    for _, r in df.iterrows():
        reasons = []
        status = "ok"
        miss = float(r.get("missing_fraction", 0) or 0)
        out = float(r.get("robust_outlier_fraction", 0) or 0)
        zero = bool(r.get("zero_variance", False))
        n = int(r.get("n", 0) or 0)
        range_status = str(r.get("range_status", "no_registry_range"))
        if n < 3:
            status = "review"
            reasons.append("too few valid observations")
        if zero:
            status = "review"
            reasons.append("zero variance")
        if miss >= 0.50:
            status = "review"
            reasons.append("high missingness")
        elif miss >= 0.20 and status == "ok":
            status = "monitor"
            reasons.append("moderate missingness")
        if out >= 0.20:
            status = "review"
            reasons.append("many robust outliers")
        elif out >= 0.05 and status == "ok":
            status = "monitor"
            reasons.append("some robust outliers")
        if range_status in ["review", "high_review"]:
            status = "review"
            reasons.append("expected-range violations")
        elif range_status == "minor_review" and status == "ok":
            status = "monitor"
            reasons.append("minor expected-range violations")
        rows.append({
            "feature": r.get("feature"),
            "n_valid": n,
            "missing_fraction": miss,
            "median": r.get("median"),
            "iqr": r.get("iqr"),
            "q05": r.get("q05"),
            "q95": r.get("q95"),
            "min": r.get("min"),
            "max": r.get("max"),
            "robust_outlier_fraction": out,
            "fraction_outside_expected": r.get("fraction_outside_expected"),
            "zero_variance": zero,
            "distribution_status": status,
            "review_reason": "; ".join(reasons) if reasons else "no major distribution issue detected",
        })
    return pd.DataFrame(rows)




def distribution_shape_audit(dist: pd.DataFrame, expected: pd.DataFrame | None = None) -> pd.DataFrame:
    """Human-readable distribution diagnostics for each feature.

    This table is descriptive. It does not transform, exclude, impute, or select
    features. It explains why a feature looks easy to interpret, requires
    monitoring, or needs review before downstream modelling.
    """
    if dist is None or dist.empty:
        return pd.DataFrame(columns=[
            "feature", "shape_class", "priority", "why_it_matters",
            "recommended_review", "possible_transform_for_ml", "do_not_conclude"
        ])
    df = dist.copy()
    if expected is not None and not expected.empty and "feature" in expected.columns:
        keep = [c for c in ["feature", "range_status", "fraction_outside_expected"] if c in expected.columns]
        df = df.merge(expected[keep], on="feature", how="left")
    rows = []
    for _, r in df.iterrows():
        reasons = []
        transform = []
        shape = []
        priority = "ok"
        n = int(r.get("n", 0) or 0)
        miss = float(r.get("missing_fraction", 0) or 0)
        out = float(r.get("robust_outlier_fraction", 0) or 0)
        zero = bool(r.get("zero_variance", False))
        nzv = bool(r.get("near_zero_variance", False))
        skew = r.get("skew_proxy_mean_minus_median_over_sd", np.nan)
        tail = r.get("tail_ratio_q95_q05_over_iqr", np.nan)
        floor = float(r.get("floor_fraction", 0) or 0)
        ceiling = float(r.get("ceiling_fraction", 0) or 0)
        range_status = str(r.get("range_status", "no_registry_range"))
        if n < 10:
            priority = "review"
            reasons.append("very small valid n")
            shape.append("insufficient_n")
        if zero:
            priority = "review"
            reasons.append("zero variance; cannot separate records")
            shape.append("zero_variance")
        elif nzv:
            priority = "monitor" if priority == "ok" else priority
            reasons.append("near-zero variance / sparse spread")
            shape.append("near_zero_variance")
        if pd.notna(skew) and abs(float(skew)) >= 0.75:
            priority = "monitor" if priority == "ok" else priority
            direction = "right-skewed" if float(skew) > 0 else "left-skewed"
            reasons.append(direction)
            shape.append(direction)
            transform.append("consider log/robust scaling in ML only if scientifically compatible")
        if pd.notna(tail) and float(tail) >= 4.5:
            priority = "monitor" if priority == "ok" else priority
            reasons.append("heavy-tailed distribution")
            shape.append("heavy_tailed")
            transform.append("inspect extreme rows; robust scaling may be preferable in ML")
        if floor >= 0.20:
            priority = "monitor" if priority == "ok" else priority
            reasons.append("possible floor effect")
            shape.append("floor_effect")
        if ceiling >= 0.20:
            priority = "monitor" if priority == "ok" else priority
            reasons.append("possible ceiling effect")
            shape.append("ceiling_effect")
        if miss >= 0.50:
            priority = "review"
            reasons.append("high missingness")
        elif miss >= 0.20 and priority == "ok":
            priority = "monitor"
            reasons.append("moderate missingness")
        if out >= 0.20:
            priority = "review"
            reasons.append("large robust outlier burden")
        elif out >= 0.05 and priority == "ok":
            priority = "monitor"
            reasons.append("some robust outliers")
        if range_status in ["review", "high_review"]:
            priority = "review"
            reasons.append("expected-range violations")
        elif range_status == "minor_review" and priority == "ok":
            priority = "monitor"
            reasons.append("minor expected-range violations")
        shape_class = ", ".join(dict.fromkeys(shape)) if shape else "compact_or_regular"
        if not transform:
            transform.append("no transformation suggested at feature-analysis stage")
        if priority == "review":
            rec = "Inspect raw rows, task/QC context, and computation validity before ML export."
        elif priority == "monitor":
            rec = "Keep visible in downstream review; consider robust ML preprocessing inside CV."
        else:
            rec = "Distribution is descriptively acceptable; continue to QC/reliability review."
        rows.append({
            "feature": r.get("feature"),
            "n_valid": n,
            "missing_fraction": miss,
            "robust_outlier_fraction": out,
            "skew_proxy": r.get("skew_proxy_mean_minus_median_over_sd"),
            "tail_ratio": r.get("tail_ratio_q95_q05_over_iqr"),
            "floor_fraction": floor,
            "ceiling_fraction": ceiling,
            "shape_class": shape_class,
            "priority": priority,
            "why_it_matters": "; ".join(reasons) if reasons else "no major distribution-shape concern detected",
            "recommended_review": rec,
            "possible_transform_for_ml": "; ".join(dict.fromkeys(transform)),
            "do_not_conclude": "Do not treat statistical non-normality or an outlier as invalid physiology without raw/QC/context review.",
        })
    priority_order = {"review": 0, "monitor": 1, "ok": 2}
    out = pd.DataFrame(rows)
    if not out.empty:
        out["priority_rank"] = out["priority"].map(priority_order).fillna(9)
        out = out.sort_values(["priority_rank", "robust_outlier_fraction", "missing_fraction"], ascending=[True, False, False]).drop(columns=["priority_rank"])
    return out


def row_outlier_burden_summary(outliers: pd.DataFrame, n_features: int) -> pd.DataFrame:
    """Summarize how many feature-level flags accumulate on each row/recording."""
    base_cols = ["row_index", "n_flagged_features", "fraction_flagged_features", "range_flag_count", "robust_flag_count", "review_level", "flagged_features"]
    if outliers is None or outliers.empty or "row_index" not in outliers.columns:
        return pd.DataFrame(columns=base_cols)
    rows = []
    id_cols = [c for c in ["file_name", "record_key", "subject_id", "session_id", "task", "iteration", "recording_date"] if c in outliers.columns]
    denom = max(1, int(n_features or 1))
    for row_index, g in outliers.groupby("row_index", dropna=False):
        flags = g.get("flag_type", pd.Series(dtype=str)).astype(str)
        n_range = int(flags.str.contains("expected_range", case=False, na=False).sum())
        n_robust = int(flags.str.contains("robust_z", case=False, na=False).sum())
        n_feat = int(g["feature"].nunique()) if "feature" in g.columns else int(len(g))
        frac = n_feat / denom
        if n_range > 0 or frac >= 0.20:
            level = "review"
        elif frac >= 0.05:
            level = "monitor"
        else:
            level = "ok"
        rec = {
            "row_index": row_index,
            "n_flagged_features": n_feat,
            "fraction_flagged_features": frac,
            "range_flag_count": n_range,
            "robust_flag_count": n_robust,
            "review_level": level,
            "flagged_features": ", ".join(g["feature"].astype(str).drop_duplicates().head(30)) if "feature" in g.columns else "",
        }
        for c in id_cols:
            rec[c] = g[c].iloc[0]
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["review_level", "n_flagged_features"], ascending=[False, False])

def overview_readiness_summary(
    feature_df: pd.DataFrame,
    qc_df: pd.DataFrame | None,
    meta_df: pd.DataFrame | None,
    mapping: pd.DataFrame,
    dist: pd.DataFrame | None = None,
    feature_missing: pd.DataFrame | None = None,
    group_counts_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """High-level readiness dimensions for the Overview page.

    These are descriptive orientation scores, not ML performance scores. They are
    intended to help the user see what kinds of downstream analyses are supported
    by the current table: feature completeness, metadata context, QC context,
    numeric analyzability, and design richness.
    """
    roles = role_lists(mapping)
    feature_cols = roles.get(ROLE_FEATURE, [])
    target_cols = roles.get(ROLE_TARGET, [])
    covar_cols = roles.get(ROLE_COVARIATE, [])
    id_cols = roles.get(ROLE_IDENTIFIER, [])
    numeric_feats = numeric_columns(feature_df, feature_cols)
    n_rows = len(feature_df)
    n_features = len(feature_cols)
    mean_missing = np.nan
    if feature_missing is not None and not feature_missing.empty and "missing_fraction" in feature_missing.columns:
        mean_missing = float(pd.to_numeric(feature_missing["missing_fraction"], errors="coerce").mean())
    elif numeric_feats:
        mean_missing = float(feature_df[numeric_feats].isna().mean().mean())
    feature_completeness = 100.0 * (1.0 - mean_missing) if pd.notna(mean_missing) else 0.0
    numeric_coverage = 100.0 * (len(numeric_feats) / n_features) if n_features else 0.0
    qc_context = 100.0 if qc_df is not None and not qc_df.empty else 0.0
    metadata_context = 0.0
    if meta_df is not None and not meta_df.empty:
        metadata_context = 100.0
    elif target_cols or covar_cols:
        metadata_context = 70.0
    elif id_cols:
        metadata_context = 35.0
    design_components = 0
    max_components = 5
    for g in ["task", "diagnosis", "severity_bin", "sex_or_gender", "session"]:
        if group_counts_df is not None and not group_counts_df.empty and "group_variable" in group_counts_df.columns and g in set(group_counts_df["group_variable"].astype(str)):
            design_components += 1
    design_richness = 100.0 * design_components / max_components
    row_depth = min(100.0, 100.0 * n_rows / 100.0) if n_rows else 0.0
    feature_depth = min(100.0, 100.0 * len(numeric_feats) / 50.0) if numeric_feats else 0.0
    readiness = [
        ("Feature completeness", feature_completeness, "Average availability across selected feature columns."),
        ("Numeric analyzability", numeric_coverage, "Share of mapped features that are numeric and can enter quantitative audits."),
        ("Metadata context", metadata_context, "Availability of labels, covariates, tasks, or a linked metadata table."),
        ("QC context", qc_context, "Availability of QC metrics for artifact-sensitivity screening."),
        ("Design richness", design_richness, "Availability of task, diagnosis/severity, sex/gender, session, or related grouping structure."),
        ("Row depth", row_depth, "Whether enough rows are present for stable descriptive summaries."),
        ("Feature breadth", feature_depth, "Whether enough numeric features are present for meaningful feature-space review."),
    ]
    rows = []
    for dimension, score, interp in readiness:
        score = float(np.clip(score, 0, 100))
        if score >= 80:
            status = "strong"
        elif score >= 50:
            status = "adequate"
        elif score > 0:
            status = "limited"
        else:
            status = "absent"
        rows.append({"dimension": dimension, "score_0_100": round(score, 1), "status": status, "interpretation": interp})
    return pd.DataFrame(rows)


def overview_feature_quality_landscape(dist: pd.DataFrame | None, registry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compact feature quality table for Overview plots.

    Combines missingness, outlier burden, variance, and optional subsystem labels.
    """
    if dist is None or dist.empty:
        return pd.DataFrame(columns=["feature", "family_or_subsystem", "missing_fraction", "robust_outlier_fraction", "n_valid", "quality_status", "quality_score"])
    subsystem_lookup: dict[str, str] = {}
    if registry is not None and not registry.empty:
        fcol = next((c for c in ["feature", "feature_name", "name", "column"] if c in registry.columns), None)
        scol = next((c for c in ["subsystem", "family", "feature_family", "group"] if c in registry.columns), None)
        if fcol and scol:
            for _, r in registry[[fcol, scol]].dropna().iterrows():
                subsystem_lookup[str(r[fcol])] = str(r[scol])
    rows = []
    for _, r in dist.iterrows():
        feature = str(r.get("feature", ""))
        miss = float(r.get("missing_fraction", np.nan)) if pd.notna(r.get("missing_fraction", np.nan)) else np.nan
        out = float(r.get("robust_outlier_fraction", np.nan)) if pd.notna(r.get("robust_outlier_fraction", np.nan)) else np.nan
        n_valid = int(r.get("n", 0) or 0)
        zero = bool(r.get("zero_variance", False))
        penalty = 0.0
        if pd.notna(miss):
            penalty += min(70.0, 70.0 * miss)
        if pd.notna(out):
            penalty += min(25.0, 125.0 * out)
        if zero:
            penalty += 40.0
        if n_valid < 3:
            penalty += 30.0
        score = float(np.clip(100.0 - penalty, 0, 100))
        if zero or n_valid < 3 or (pd.notna(miss) and miss >= 0.50) or (pd.notna(out) and out >= 0.20):
            status = "review"
        elif (pd.notna(miss) and miss >= 0.20) or (pd.notna(out) and out >= 0.05):
            status = "monitor"
        else:
            status = "ok"
        rows.append({
            "feature": feature,
            "family_or_subsystem": subsystem_lookup.get(feature, "unclassified"),
            "missing_fraction": miss,
            "robust_outlier_fraction": out,
            "n_valid": n_valid,
            "zero_variance": zero,
            "quality_status": status,
            "quality_score": round(score, 1),
        })
    return pd.DataFrame(rows).sort_values(["quality_status", "quality_score", "feature"], ascending=[False, True, True])
