"""Acoustic feature aggregation stage.

This stage converts per-file acoustic features into analysis-ready tables for
feature analysis and ML. It preserves metadata keys, makes missingness explicit,
and writes audit-grade outputs rather than silently dropping problematic values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vslp.core.project import ensure_stage_folders
from vslp.core.provenance import python_environment
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult
from vslp.acoustic.features.registry import build_acoustic_feature_registry
import json

DEFAULT_GROUP_COLUMNS = ["subject_id", "session_id", "iteration", "task"]
NON_FEATURE_COLUMNS = {
    "file_name", "source_file_path", "segmentation_wav_path", "subject_id", "session_id", "iteration",
    "task", "recording_date", "diagnosis", "severity_score", "severity_bin", "record_key", "duration_sec",
    "feature_extraction_status",
}

ACOUSTIC_ML_ID_COLUMNS: tuple[str, ...] = (
    "subject_id", "session_id", "visit_id", "iteration", "task", "record_key", "file_name"
)
ACOUSTIC_ML_CONTEXT_COLUMNS: tuple[str, ...] = (
    "diagnosis", "severity_score", "severity_bin", "recording_date", "n_files_aggregated"
)


def acoustic_feature_manifest_dataframe(columns: list[str] | None = None) -> pd.DataFrame:
    """Return a column-level manifest for acoustic ML-ready exports.

    This mirrors the kinematic feature manifest contract: predictors, metadata,
    QC/provenance fields, and dense engineering summaries are separated so the
    ML GUI can avoid leakage and expose feature families cleanly.
    """
    registry = build_acoustic_feature_registry()
    by_name = {str(row["feature"]): row for _, row in registry.iterrows()}
    column_list = list(columns) if columns is not None else registry["feature"].astype(str).tolist()
    rows: list[dict[str, object]] = []
    metadata_cols = set(ACOUSTIC_ML_ID_COLUMNS) | {"source_file_path", "segmentation_wav_path", "recording_date"}
    label_cols = {"diagnosis", "severity_score", "severity_bin", "target", "label", "disease_group"}
    qc_cols = {"feature_extraction_status", "qc_status", "quality_status", "duration_sec", "n_files_aggregated"}
    for col in column_list:
        base_col = str(col)
        if base_col in by_name:
            rec = by_name[base_col]
            role = "canonical_feature"
            family = str(rec.get("subsystem", "acoustic"))
            primitive = str(rec.get("subsystem", "acoustic"))
            unit = str(rec.get("unit", "varies"))
            model_role = "candidate_predictor"
            include_gui = True
            include_ml = str(rec.get("implementation_status", "")).lower() in {"implemented", "computed_proxy"}
            interpretation = str(rec.get("meaning", "Acoustic feature."))
            caution = str(rec.get("computation_note", "Use after acoustic QC and task validation."))
            summary_stat = "per-row acoustic scalar"
            evidence_tier = str(rec.get("evidence_tier", ""))
        elif base_col in metadata_cols:
            role = "metadata"
            family = "metadata"
            primitive = "identifier"
            unit = "n/a"
            model_role = "join_key_or_provenance"
            include_gui = False
            include_ml = False
            interpretation = "Identifier/provenance field; do not use as a model predictor."
            caution = "May leak subject, task, file, or source identity if used as a predictor."
            summary_stat = "n/a"
            evidence_tier = "n/a"
        elif base_col in label_cols:
            role = "label_or_outcome"
            family = "target_metadata"
            primitive = "outcome"
            unit = "varies"
            model_role = "target_or_stratification"
            include_gui = True
            include_ml = False
            interpretation = "Outcome or label column; may be selected as target, not predictor."
            caution = "Never include target/outcome columns as predictors."
            summary_stat = "n/a"
            evidence_tier = "n/a"
        elif base_col in qc_cols or "qc" in base_col.lower() or "status" in base_col.lower():
            role = "qc_metric"
            family = "quality_control"
            primitive = "qc/provenance"
            unit = "varies"
            model_role = "filter_or_stratify"
            include_gui = True
            include_ml = False
            interpretation = "Acoustic quality-control or processing-readiness metric."
            caution = "Use to filter/stratify data; do not treat as a disease biomarker."
            summary_stat = "n/a"
            evidence_tier = "n/a"
        elif any(base_col.endswith(suffix) for suffix in ("__mean", "__median", "__sd", "__iqr", "__q05", "__q95", "__min", "__max", "__n", "__missing_fraction")):
            stem = base_col.split("__", 1)[0]
            rec = by_name.get(stem)
            role = "dense_engineering_summary"
            family = str(rec.get("subsystem", "expanded_acoustic_summary")) if rec is not None else "expanded_acoustic_summary"
            primitive = "summary_statistic"
            unit = str(rec.get("unit", "varies")) if rec is not None else "varies"
            model_role = "optional_predictor"
            include_gui = True
            include_ml = False
            interpretation = "Expanded acoustic summary retained for research review."
            caution = "Can inflate dimensionality and redundancy; not part of default ML-ready acoustic set."
            summary_stat = base_col.split("__", 1)[1]
            evidence_tier = str(rec.get("evidence_tier", "")) if rec is not None else "review"
        else:
            role = "other"
            family = "uncategorized"
            primitive = "unknown"
            unit = "varies"
            model_role = "review_before_modeling"
            include_gui = False
            include_ml = False
            interpretation = "Uncategorized acoustic output column."
            caution = "Review manually before use in analysis or ML."
            summary_stat = "unknown"
            evidence_tier = "review"
        rows.append({
            "column_name": base_col,
            "column_role": role,
            "modality": "acoustic",
            "family": family,
            "subsystem": family,
            "primitive": primitive,
            "summary_statistic": summary_stat,
            "unit": unit,
            "model_role": model_role,
            "include_in_feature_gui_default": bool(include_gui),
            "include_in_ml_default": bool(include_ml),
            "evidence_tier": evidence_tier,
            "interpretation": interpretation,
            "caution": caution,
        })
    return pd.DataFrame(rows)


def write_acoustic_ml_ready_exports(features_df: pd.DataFrame, output_root: Path | str) -> dict[str, Path | int]:
    """Write acoustic ML-ready and manifest exports parallel to kinematics.

    The full aggregation outputs remain unchanged. These derivative tables expose
    a clean predictor matrix and a column manifest for the ML GUI.
    """
    root = Path(output_root).expanduser().resolve()
    tables = root / "acoustic" / "005_aggregation" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    manifest = acoustic_feature_manifest_dataframe(list(features_df.columns))
    feature_cols = manifest.loc[manifest["include_in_ml_default"].astype(bool), "column_name"].astype(str).tolist()
    feature_cols = [c for c in feature_cols if c in features_df.columns]
    id_cols = [c for c in ACOUSTIC_ML_ID_COLUMNS if c in features_df.columns]
    context_cols = [c for c in ACOUSTIC_ML_CONTEXT_COLUMNS if c in features_df.columns and c not in id_cols]

    ml_ready = features_df[id_cols + feature_cols].copy() if feature_cols else pd.DataFrame(columns=id_cols)
    context = features_df[id_cols + context_cols + feature_cols].copy() if feature_cols else features_df[id_cols + context_cols].copy()
    features_only = features_df[feature_cols].copy() if feature_cols else pd.DataFrame()

    ml_ready_csv = tables / "acoustic_features_ml_ready.csv"
    context_csv = tables / "acoustic_features_canonical.csv"
    features_only_csv = tables / "acoustic_features_only.csv"
    manifest_csv = tables / "acoustic_feature_manifest.csv"
    manifest_json = tables / "acoustic_feature_manifest.json"

    ml_ready.to_csv(ml_ready_csv, index=False)
    context.to_csv(context_csv, index=False)
    features_only.to_csv(features_only_csv, index=False)
    manifest.to_csv(manifest_csv, index=False)
    payload = {
        "status": "ACOUSTIC_FEATURE_EXPORT_MANIFEST",
        "ml_ready_csv": str(ml_ready_csv),
        "features_only_csv": str(features_only_csv),
        "feature_manifest_csv": str(manifest_csv),
        "n_rows": int(len(features_df)),
        "n_full_columns": int(len(features_df.columns)),
        "n_ml_default_features": int(len(feature_cols)),
        "note": "acoustic_features_ml_ready.csv keeps row IDs plus default acoustic predictors. acoustic_feature_manifest.csv labels all columns for ML safety.",
    }
    manifest_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return {
        "ml_ready_csv": ml_ready_csv,
        "features_only_csv": features_only_csv,
        "canonical_csv": context_csv,
        "feature_manifest_csv": manifest_csv,
        "feature_manifest_json": manifest_json,
        "n_ml_default_features": len(feature_cols),
    }


@dataclass(frozen=True)
class AggregationConfig:
    """Configuration for acoustic aggregation."""

    group_columns: list[str] = field(default_factory=lambda: DEFAULT_GROUP_COLUMNS.copy())
    numeric_policy: str = "mean"  # mean, median
    missing_policy: str = "preserve"  # preserve, drop_features_over_threshold, impute_group_median
    max_missing_fraction: float = 0.40
    min_non_missing_per_feature: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            out.append(col)
    return out


def run_acoustic_aggregation(
    features_csv: str | Path,
    output_root: str | Path,
    config: AggregationConfig | None = None,
) -> StageResult:
    """Aggregate acoustic feature rows by subject/session/task or custom groups."""
    cfg = config or AggregationConfig()
    features_csv = Path(features_csv)
    stage_dir = Path(output_root) / "acoustic" / "005_aggregation"
    folders = ensure_stage_folders(stage_dir)

    df = pd.read_csv(features_csv)
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []

    group_cols = [c for c in cfg.group_columns if c in df.columns]
    if not group_cols:
        group_cols = ["file_name"] if "file_name" in df.columns else []
        warnings.append("Requested group columns were missing; fell back to file_name aggregation.")
    if not group_cols:
        raise ValueError("No valid group columns available for aggregation.")

    feature_cols = _numeric_feature_columns(df)
    working = df.copy()
    for col in feature_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    feature_missing = working[feature_cols].isna().mean().sort_values(ascending=False) if feature_cols else pd.Series(dtype=float)
    dropped_features: list[str] = []
    if cfg.missing_policy == "drop_features_over_threshold":
        dropped_features = feature_missing.loc[feature_missing > cfg.max_missing_fraction].index.astype(str).tolist()
        feature_cols = [c for c in feature_cols if c not in dropped_features]
        warnings.append(f"Dropped {len(dropped_features)} features over missingness threshold {cfg.max_missing_fraction}.")
    elif cfg.missing_policy == "impute_group_median":
        for col in feature_cols:
            median = working[col].median(skipna=True)
            if np.isfinite(median):
                working[col] = working[col].fillna(median)
    elif cfg.missing_policy != "preserve":
        errors.append({"aggregation_error": f"Unknown missing_policy: {cfg.missing_policy}"})
        warnings.append(f"Unknown missing_policy {cfg.missing_policy}; preserved missing values.")

    if cfg.numeric_policy == "median":
        agg_numeric = working.groupby(group_cols, dropna=False)[feature_cols].median().reset_index()
    elif cfg.numeric_policy == "mean":
        agg_numeric = working.groupby(group_cols, dropna=False)[feature_cols].mean().reset_index()
    else:
        errors.append({"aggregation_error": f"Unknown numeric_policy: {cfg.numeric_policy}"})
        agg_numeric = working.groupby(group_cols, dropna=False)[feature_cols].mean().reset_index()
        warnings.append(f"Unknown numeric_policy {cfg.numeric_policy}; used mean.")

    # Preserve key descriptive metadata by first non-null within each group.
    meta_cols = [c for c in ["diagnosis", "severity_score", "severity_bin", "recording_date"] if c in working.columns and c not in group_cols]
    if meta_cols:
        meta = working.groupby(group_cols, dropna=False)[meta_cols].agg(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan).reset_index()
        out = agg_numeric.merge(meta, on=group_cols, how="left")
    else:
        out = agg_numeric

    counts = working.groupby(group_cols, dropna=False).size().reset_index(name="n_files_aggregated")
    out = out.merge(counts, on=group_cols, how="left")

    aggregated_path = folders["tables"] / "acoustic_features_aggregated.csv"
    multistat_path = folders["tables"] / "acoustic_features_aggregated_multistat.csv"
    strategy_path = folders["tables"] / "acoustic_aggregation_strategy.csv"
    missingness_path = folders["tables"] / "aggregation_missingness.csv"
    dropped_path = folders["tables"] / "aggregation_dropped_features.csv"
    errors_path = folders["errors"] / "acoustic_aggregation_errors.csv"

    out.to_csv(aggregated_path, index=False)
    _build_multistat_aggregation(working, group_cols, feature_cols).to_csv(multistat_path, index=False)
    _build_aggregation_strategy(Path(output_root), feature_cols).to_csv(strategy_path, index=False)
    feature_missing.rename("missing_fraction").rename_axis("feature").reset_index().to_csv(missingness_path, index=False)
    pd.DataFrame({"feature": dropped_features}).to_csv(dropped_path, index=False)
    pd.DataFrame(errors).to_csv(errors_path, index=False)
    ml_exports = write_acoustic_ml_ready_exports(out, output_root)

    missing_plot = folders["plots"] / "aggregation_missingness_top30.png"
    group_plot = folders["plots"] / "aggregation_group_counts.png"
    _plot_missingness(missingness_path, missing_plot)
    _plot_group_counts(out, group_cols, group_plot)

    report_path = folders["reports"] / "acoustic_aggregation_report.html"
    _write_aggregation_report(report_path, out, cfg, warnings, errors, missing_plot, group_plot)

    manifest = StageManifest(
        stage_name="acoustic_feature_aggregation",
        stage_version="0.12.0",
        status="completed_with_warnings" if warnings or errors else "completed",
        input_artifacts=[ArtifactRef(path=str(features_csv), role="features_per_file", media_type="text/csv")],
        output_artifacts=[
            ArtifactRef(path=str(aggregated_path), role="aggregated_features", media_type="text/csv"),
            ArtifactRef(path=str(multistat_path), role="aggregated_features_multistat", media_type="text/csv"),
            ArtifactRef(path=str(ml_exports["ml_ready_csv"]), role="acoustic_ml_ready_features", media_type="text/csv"),
            ArtifactRef(path=str(ml_exports["feature_manifest_csv"]), role="acoustic_feature_manifest", media_type="text/csv"),
            ArtifactRef(path=str(strategy_path), role="aggregation_strategy", media_type="text/csv"),
            ArtifactRef(path=str(missingness_path), role="aggregation_missingness", media_type="text/csv"),
            ArtifactRef(path=str(report_path), role="aggregation_html_report", media_type="text/html"),
        ],
        config=cfg.to_dict(),
        environment={"python": python_environment()},
        warnings=warnings,
        errors=errors,
        notes=["Aggregation is explicit and reproducible; missingness is not silently hidden.", "v0.32 adds multistat aggregation so mean/median are not the only ML-ready summary choices.", "v0.88 writes acoustic_features_ml_ready.csv and acoustic_feature_manifest.csv for multimodal ML GUI input."],
    )
    manifest_path = folders["logs"] / "stage_manifest.json"
    manifest.write_json(manifest_path)

    return StageResult(status=manifest.status, manifest_path=manifest_path, summary_table=aggregated_path, error_table=errors_path, report_path=report_path)


def _build_multistat_aggregation(working: pd.DataFrame, group_cols: list[str], feature_cols: list[str]) -> pd.DataFrame:
    """Create distribution-preserving group summaries for ML and analysis.

    This table intentionally keeps multiple summaries per feature. It prevents the
    aggregation layer from collapsing all within-group information to a single
    mean or median. For small groups, some statistics are naturally NaN.
    """
    if not feature_cols:
        return working[group_cols].drop_duplicates().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for key, grp in working.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        out = {col: val for col, val in zip(group_cols, key_tuple, strict=False)}
        out["n_files_aggregated"] = int(len(grp))
        for feat in feature_cols:
            vals = pd.to_numeric(grp[feat], errors="coerce").dropna()
            prefix = feat
            out[f"{prefix}__n"] = int(vals.size)
            out[f"{prefix}__missing_fraction"] = float(1.0 - vals.size / len(grp)) if len(grp) else np.nan
            if vals.size:
                out[f"{prefix}__mean"] = float(vals.mean())
                out[f"{prefix}__median"] = float(vals.median())
                out[f"{prefix}__sd"] = float(vals.std(ddof=0)) if vals.size >= 2 else np.nan
                out[f"{prefix}__iqr"] = float(vals.quantile(0.75) - vals.quantile(0.25)) if vals.size >= 2 else np.nan
                out[f"{prefix}__q05"] = float(vals.quantile(0.05))
                out[f"{prefix}__q25"] = float(vals.quantile(0.25))
                out[f"{prefix}__q75"] = float(vals.quantile(0.75))
                out[f"{prefix}__q95"] = float(vals.quantile(0.95))
                out[f"{prefix}__min"] = float(vals.min())
                out[f"{prefix}__max"] = float(vals.max())
            else:
                for stat in ["mean", "median", "sd", "iqr", "q05", "q25", "q75", "q95", "min", "max"]:
                    out[f"{prefix}__{stat}"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def _build_aggregation_strategy(output_root: Path, feature_cols: list[str]) -> pd.DataFrame:
    """Merge feature scale metadata into an aggregation strategy table if available."""
    scale_path = output_root / "acoustic" / "004_features" / "tables" / "acoustic_feature_measurement_scale_registry.csv"
    if scale_path.exists():
        try:
            scale = pd.read_csv(scale_path)
            scale = scale.loc[scale["feature"].astype(str).isin([str(c) for c in feature_cols])].copy()
            if not scale.empty:
                return scale[[c for c in ["feature", "subsystem", "native_scale", "physiologic_unit", "recommended_file_reducers", "recommended_group_reducers", "ml_recommendation", "interpretation_note"] if c in scale.columns]]
        except Exception:
            pass
    return pd.DataFrame({"feature": feature_cols, "recommended_group_reducers": "median,iqr,q05,q95,n_nonmissing,missing_fraction"})


def aggregate_features(input_csv: str | Path, output_csv: str | Path, group_columns: list[str], numeric_policy: str = "mean") -> Path:
    """Backward-compatible helper retained for early scripts."""
    df = pd.read_csv(input_csv)
    numeric = df.select_dtypes(include="number").columns.tolist()
    if numeric_policy == "median":
        out = df.groupby(group_columns, dropna=False)[numeric].median().reset_index()
    elif numeric_policy == "mean":
        out = df.groupby(group_columns, dropna=False)[numeric].mean().reset_index()
    else:
        raise NotImplementedError(f"Aggregation policy not implemented yet: {numeric_policy}")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return output_csv


def _plot_missingness(missingness_csv: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(missingness_csv)
    if df.empty:
        return
    df = df.sort_values("missing_fraction", ascending=True).tail(30)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(df) + 2)))
    ax.barh(df["feature"].astype(str), df["missing_fraction"].astype(float))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Missing fraction")
    ax.set_title("Top feature missingness before aggregation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_group_counts(out: pd.DataFrame, group_cols: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if out.empty or "n_files_aggregated" not in out.columns:
        return
    labels = out[group_cols].astype(str).agg(" | ".join, axis=1).head(30)
    vals = out["n_files_aggregated"].head(30)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.28 * len(labels) + 2)))
    ax.barh(labels, vals)
    ax.set_xlabel("Files per aggregate row")
    ax.set_title("Aggregation group counts")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _write_aggregation_report(path: Path, out: pd.DataFrame, cfg: AggregationConfig, warnings: list[str], errors: list[dict[str, Any]], missing_plot: Path, group_plot: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def img(title: str, p: Path) -> str:
        if not p.exists():
            return ""
        return f"<div class='card'><h2>{title}</h2><img src='../plots/{p.name}'></div>"

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>VSLP Acoustic Aggregation Report</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif; background:#071A2D; color:#EAF2F8; margin:32px; }}
.card {{ background:#0B253D; border:1px solid #315D7C; border-radius:12px; padding:18px; margin:14px 0; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:6px; background:#0E3A5B; margin-right:8px; margin-top:4px; }}
pre {{ white-space:pre-wrap; background:#102A43; padding:12px; border-radius:8px; }}
img {{ max-width:100%; border-radius:10px; border:1px solid #315D7C; background:white; }}
</style></head><body>
<h1>VSLP Acoustic Aggregation Report</h1>
<div class='card'><span class='badge'>Aggregated rows: {len(out)}</span><span class='badge'>Columns: {len(out.columns)}</span><span class='badge'>Warnings: {len(warnings)}</span><span class='badge'>Errors: {len(errors)}</span></div>
<div class='card'><h2>Configuration</h2><pre>{cfg.to_dict()}</pre></div>
<div class='card'><h2>Warnings</h2><pre>{warnings}</pre></div>
{img('Feature missingness before aggregation', missing_plot)}
{img('Aggregation group counts', group_plot)}
</body></html>"""
    path.write_text(html, encoding="utf-8")
