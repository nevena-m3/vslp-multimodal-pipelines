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

DEFAULT_GROUP_COLUMNS = ["subject_id", "session_id", "iteration", "task"]
NON_FEATURE_COLUMNS = {
    "file_name", "source_file_path", "segmentation_wav_path", "subject_id", "session_id", "iteration",
    "task", "recording_date", "diagnosis", "severity_score", "severity_bin", "record_key", "duration_sec",
    "feature_extraction_status",
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
    missingness_path = folders["tables"] / "aggregation_missingness.csv"
    dropped_path = folders["tables"] / "aggregation_dropped_features.csv"
    errors_path = folders["errors"] / "acoustic_aggregation_errors.csv"

    out.to_csv(aggregated_path, index=False)
    feature_missing.rename("missing_fraction").rename_axis("feature").reset_index().to_csv(missingness_path, index=False)
    pd.DataFrame({"feature": dropped_features}).to_csv(dropped_path, index=False)
    pd.DataFrame(errors).to_csv(errors_path, index=False)

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
            ArtifactRef(path=str(missingness_path), role="aggregation_missingness", media_type="text/csv"),
            ArtifactRef(path=str(report_path), role="aggregation_html_report", media_type="text/html"),
        ],
        config=cfg.to_dict(),
        environment={"python": python_environment()},
        warnings=warnings,
        errors=errors,
        notes=["Aggregation is explicit and reproducible; missingness is not silently hidden."],
    )
    manifest_path = folders["logs"] / "stage_manifest.json"
    manifest.write_json(manifest_path)

    return StageResult(status=manifest.status, manifest_path=manifest_path, summary_table=aggregated_path, error_table=errors_path, report_path=report_path)


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
