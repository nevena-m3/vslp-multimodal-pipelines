from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .column_mapping import infer_column_roles, columns_by_role, choose_join_key
from .loaders import read_table, normalize_column_names, safe_to_csv
from .plots import (
    plot_missingness,
    plot_feature_availability_heatmap,
    plot_distribution_grid,
    plot_corr_heatmap,
    plot_outlier_counts,
)
from .reports import write_report
from .schemas import AnalysisInputs, AnalysisResult, ColumnRole
from .statistics import distribution_summary, missingness_by_row, outlier_flags, spearman_correlation, qc_feature_correlations


def _stage_dirs(root: Path) -> Dict[str, Path]:
    base = Path(root) / "feature_analysis"
    dirs = {
        "base": base,
        "tables": base / "tables",
        "plots": base / "plots",
        "reports": base / "reports",
        "exports": base / "exports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _merge_optional(feature_df: pd.DataFrame, other_df: Optional[pd.DataFrame], preferred_key: str = "auto") -> tuple[pd.DataFrame, Optional[str], str]:
    if other_df is None:
        return feature_df, None, "No optional table supplied."
    key = None if preferred_key == "auto" else preferred_key
    if key and (key not in feature_df.columns or key not in other_df.columns):
        key = None
    if key is None:
        key = choose_join_key(feature_df, other_df)
    if key is None:
        return feature_df, None, "Optional table could not be merged: no shared join key found."
    merged = feature_df.merge(other_df, on=key, how="left", suffixes=("", "_optional"))
    return merged, key, f"Merged optional table using key: {key}."


def run_feature_analysis(inputs: AnalysisInputs) -> AnalysisResult:
    dirs = _stage_dirs(inputs.output_root)
    tables: Dict[str, Path] = {}
    plots: Dict[str, Path] = {}
    messages = []

    feat = normalize_column_names(read_table(inputs.feature_table))
    tables["input_feature_preview"] = safe_to_csv(feat.head(200), dirs["tables"] / "input_feature_preview.csv")

    feature_mapping = infer_column_roles(feat, "features")
    tables["feature_column_mapping"] = safe_to_csv(feature_mapping, dirs["tables"] / "feature_column_mapping.csv")
    feature_cols = columns_by_role(feature_mapping, ColumnRole.FEATURE)
    id_cols = columns_by_role(feature_mapping, ColumnRole.IDENTIFIER)
    target_cols = columns_by_role(feature_mapping, ColumnRole.TARGET)
    task_cols = columns_by_role(feature_mapping, ColumnRole.TASK)

    qc = None
    qc_mapping = pd.DataFrame()
    qc_cols = []
    if inputs.qc_table:
        qc = normalize_column_names(read_table(inputs.qc_table))
        qc_mapping = infer_column_roles(qc, "qc")
        tables["qc_column_mapping"] = safe_to_csv(qc_mapping, dirs["tables"] / "qc_column_mapping.csv")
        qc_cols = columns_by_role(qc_mapping, ColumnRole.QC_FEATURE)
        merged, join_key, msg = _merge_optional(feat, qc, inputs.join_key)
        messages.append(msg)
        if join_key:
            # align qc to feature rows for QC correlations
            qc_aligned = merged[[c for c in merged.columns if c in qc.columns or c.endswith("_optional")]].copy()
        else:
            qc_aligned = qc.reindex(feat.index)
    else:
        qc_aligned = None
        messages.append("No QC table supplied. QC-feature association analyses were skipped.")

    metadata = None
    if inputs.metadata_table:
        metadata = normalize_column_names(read_table(inputs.metadata_table))
        meta_mapping = infer_column_roles(metadata, "metadata")
        tables["metadata_column_mapping"] = safe_to_csv(meta_mapping, dirs["tables"] / "metadata_column_mapping.csv")
        merged, join_key, msg = _merge_optional(feat, metadata, inputs.join_key)
        messages.append(msg)
    else:
        messages.append("No metadata table supplied. Group/longitudinal views are limited.")

    registry = None
    if inputs.feature_registry:
        registry = normalize_column_names(read_table(inputs.feature_registry))
        tables["feature_registry_preview"] = safe_to_csv(registry.head(500), dirs["tables"] / "feature_registry_preview.csv")

    inventory = pd.DataFrame([
        {"item": "rows", "value": int(feat.shape[0])},
        {"item": "columns", "value": int(feat.shape[1])},
        {"item": "detected_feature_columns", "value": int(len(feature_cols))},
        {"item": "detected_identifier_columns", "value": int(len(id_cols))},
        {"item": "detected_target_columns", "value": int(len(target_cols))},
        {"item": "detected_task_columns", "value": int(len(task_cols))},
        {"item": "detected_qc_columns", "value": int(len(qc_cols))},
    ])
    tables["dataset_inventory"] = safe_to_csv(inventory, dirs["tables"] / "dataset_inventory.csv")

    dist = distribution_summary(feat, feature_cols)
    tables["feature_distribution_summary"] = safe_to_csv(dist, dirs["tables"] / "feature_distribution_summary.csv")
    miss_row = missingness_by_row(feat, feature_cols)
    tables["missingness_by_row"] = safe_to_csv(miss_row, dirs["tables"] / "missingness_by_row.csv")
    outliers = outlier_flags(feat, feature_cols)
    tables["robust_outlier_flags"] = safe_to_csv(outliers, dirs["tables"] / "robust_outlier_flags.csv")
    corr = spearman_correlation(feat, feature_cols, max_cols=inputs.max_corr_features)
    if not corr.empty:
        tables["feature_spearman_correlation"] = safe_to_csv(corr.reset_index(names="feature"), dirs["tables"] / "feature_spearman_correlation.csv")

    if qc is not None and qc_cols:
        join_key = choose_join_key(feat, qc)
        if join_key:
            aligned = feat[[join_key] + feature_cols].merge(qc[[join_key] + qc_cols], on=join_key, how="left")
            qc_corr = qc_feature_correlations(aligned, aligned, feature_cols, qc_cols)
        else:
            aligned = pd.concat([feat[feature_cols].reset_index(drop=True), qc[qc_cols].reset_index(drop=True)], axis=1)
            qc_corr = qc_feature_correlations(aligned, aligned, feature_cols, qc_cols)
        tables["feature_qc_spearman_correlation"] = safe_to_csv(qc_corr, dirs["tables"] / "feature_qc_spearman_correlation.csv")
        if not qc_corr.empty:
            max_qc = qc_corr.dropna(subset=["abs_spearman_rho"]).groupby("feature", as_index=False)["abs_spearman_rho"].max()
        else:
            max_qc = pd.DataFrame(columns=["feature", "abs_spearman_rho"])
    else:
        max_qc = pd.DataFrame(columns=["feature", "abs_spearman_rho"])

    # Initial reliability screen: descriptive, not an automatic modelling decision.
    reliability = dist[["feature", "missing_fraction", "n_valid", "robust_outlier_fraction_abs_z_gt_3_5", "zero_variance"]].copy() if not dist.empty else pd.DataFrame(columns=["feature"])
    reliability = reliability.merge(max_qc.rename(columns={"abs_spearman_rho": "max_abs_qc_spearman"}), on="feature", how="left")
    def _rec(row):
        reasons = []
        if row.get("zero_variance") is True:
            reasons.append("zero variance")
        if row.get("missing_fraction", 0) >= 0.50:
            reasons.append("high missingness")
        if row.get("robust_outlier_fraction_abs_z_gt_3_5", 0) >= 0.10:
            reasons.append("many robust outliers")
        if pd.notna(row.get("max_abs_qc_spearman", np.nan)) and row.get("max_abs_qc_spearman", 0) >= 0.70:
            reasons.append("strong QC association")
        if row.get("zero_variance") is True or row.get("missing_fraction", 0) >= 0.80:
            rec = "exclude_or_review"
        elif reasons:
            rec = "use_with_caution"
        else:
            rec = "use"
        return pd.Series({"recommended_use": rec, "review_reason": "; ".join(reasons) if reasons else "none"})
    if not reliability.empty:
        reliability = pd.concat([reliability, reliability.apply(_rec, axis=1)], axis=1)
    tables["feature_reliability_screen"] = safe_to_csv(reliability, dirs["tables"] / "feature_reliability_screen.csv")

    plots["missingness_top_features"] = plot_missingness(dist, dirs["plots"] / "missingness_top_features.png")
    plots["feature_availability_heatmap"] = plot_feature_availability_heatmap(feat, feature_cols, dirs["plots"] / "feature_availability_heatmap.png")
    plots["feature_distribution_grid"] = plot_distribution_grid(feat, feature_cols, dirs["plots"] / "feature_distribution_grid.png")
    plots["feature_correlation_heatmap"] = plot_corr_heatmap(corr, dirs["plots"] / "feature_correlation_heatmap.png")
    plots["outlier_counts"] = plot_outlier_counts(outliers, dirs["plots"] / "outlier_counts.png")

    report = write_report(dirs["base"], "VSLP Feature Analysis Report", tables, plots, messages)
    return AnalysisResult(output_root=dirs["base"], report_path=report, tables=tables, plots=plots, messages=messages)
