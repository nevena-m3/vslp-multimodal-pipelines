"""Acoustic feature extraction stage.

V0.7/V0.8 introduces a plugin architecture and computes the first validated-safe
feature families:
- respiratory/timing features from Silero segments;
- rhythm/envelope modulation features from canonical segmentation WAVs;
- phonatory engineering proxies for F0 and CPP;
- global RMS amplitude.

Features still requiring formula-level validation against the uploaded notebook remain
registered, written as NaN, and marked `not_implemented_yet`. This avoids silent
approximation of clinical features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vslp.acoustic.features.plugins import build_default_plugins, implemented_feature_names
from vslp.acoustic.features.plugins.base import FeatureContext, FeatureValue
from vslp.acoustic.features.registry import build_acoustic_feature_registry
from vslp.core.project import ensure_stage_folders
from vslp.core.provenance import python_environment
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult

IMPLEMENTED_FEATURES = implemented_feature_names()
PROXY_FEATURES = {"f0_mean", "f0_std", "CPP_mean"}


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Configuration for acoustic feature extraction.

    selected_subsystems
        Optional subsystem whitelist. Empty means all registry subsystems.
    selected_features
        Optional feature whitelist. Empty means all features from selected subsystems.
    task_word_counts
        Optional map from task name to known word count. Used only for speech_rate.
        If absent, speech_rate is emitted as NaN rather than guessed.
    metadata_csv
        Optional metadata/file index CSV with file_name as key.
    minimum_pause_duration_sec
        Internal nonspeech runs shorter than this are ignored for pause summary features.
    """

    selected_subsystems: list[str] = field(default_factory=list)
    selected_features: list[str] = field(default_factory=list)
    task_word_counts: dict[str, float] = field(default_factory=dict)
    metadata_csv: str | None = None
    minimum_pause_duration_sec: float = 0.15
    acoustic_region_policy: str = "speech_only"  # speech_only, effective_task, full_file

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _select_registry(cfg: FeatureExtractionConfig) -> pd.DataFrame:
    registry = build_acoustic_feature_registry()
    if cfg.selected_subsystems:
        registry = registry.loc[registry["subsystem"].isin(cfg.selected_subsystems)].copy()
    if cfg.selected_features:
        registry = registry.loc[registry["feature"].isin(cfg.selected_features)].copy()
    return registry.reset_index(drop=True)


def _merge_metadata_if_available(seg_summary: pd.DataFrame, metadata_csv: str | None) -> pd.DataFrame:
    if not metadata_csv:
        return seg_summary
    metadata_path = Path(metadata_csv).expanduser()
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata_csv was provided but does not exist: {metadata_path}")
    meta = pd.read_csv(metadata_path)
    if "file_name" not in meta.columns:
        raise ValueError("metadata_csv must contain a file_name column")
    keep_cols = [
        c
        for c in [
            "file_name",
            "subject_id",
            "session_id",
            "iteration",
            "task",
            "recording_date",
            "diagnosis",
            "severity_score",
            "severity_bin",
            "record_key",
        ]
        if c in meta.columns
    ]
    out = seg_summary.merge(meta[keep_cols], on="file_name", how="left", suffixes=("", "_metadata"))
    for col in ["subject_id", "session_id", "iteration", "task", "recording_date", "diagnosis", "severity_score", "severity_bin", "record_key"]:
        mcol = f"{col}_metadata"
        if mcol in out.columns:
            if col in out.columns:
                out[col] = out[col].combine_first(out[mcol])
            else:
                out[col] = out[mcol]
            out = out.drop(columns=[mcol])
    return out


def _make_context(row: pd.Series, cfg: FeatureExtractionConfig) -> FeatureContext:
    segments_raw = row.get("segments_csv_path", "")
    wav_raw = row.get("segmentation_wav_path", "")
    segments_path = Path(str(segments_raw)) if str(segments_raw) and str(segments_raw) != "nan" else None
    wav_path = Path(str(wav_raw)) if str(wav_raw) and str(wav_raw) != "nan" else None
    duration = row.get("duration_sec", np.nan)
    try:
        duration_float = float(duration)
    except Exception:
        duration_float = np.nan
    return FeatureContext(
        file_name=str(row.get("file_name", "")),
        row=row,
        segments_csv=segments_path,
        segmentation_wav_path=wav_path,
        task=str(row.get("task")) if pd.notna(row.get("task", np.nan)) else None,
        duration_sec=duration_float,
        config=cfg,
        analysis_region=str(getattr(cfg, "acoustic_region_policy", "speech_only")),
    )


def _metadata_prefix(row: pd.Series) -> dict[str, Any]:
    return {
        "file_name": row.get("file_name", ""),
        "source_file_path": row.get("source_file_path", row.get("file_path")),
        "segmentation_wav_path": row.get("segmentation_wav_path"),
        "subject_id": row.get("subject_id", np.nan),
        "session_id": row.get("session_id", np.nan),
        "iteration": row.get("iteration", np.nan),
        "task": row.get("task", np.nan),
        "recording_date": row.get("recording_date", np.nan),
        "diagnosis": row.get("diagnosis", np.nan),
        "severity_score": row.get("severity_score", np.nan),
        "severity_bin": row.get("severity_bin", np.nan),
        "record_key": row.get("record_key", np.nan),
        "duration_sec": row.get("duration_sec", np.nan),
    }


def run_acoustic_feature_extraction(
    segmentation_summary_csv: str | Path,
    output_root: str | Path,
    config: FeatureExtractionConfig | None = None,
) -> StageResult:
    """Extract acoustic features from segmentation outputs."""
    cfg = config or FeatureExtractionConfig()
    segmentation_summary_csv = Path(segmentation_summary_csv)
    stage_dir = Path(output_root) / "acoustic" / "004_features"
    folders = ensure_stage_folders(stage_dir)

    registry = _select_registry(cfg)
    selected_names = set(registry["feature"].astype(str).tolist())
    seg_summary = pd.read_csv(segmentation_summary_csv)
    seg_summary = _merge_metadata_if_available(seg_summary, cfg.metadata_csv)

    plugins = [p for p in build_default_plugins() if selected_names.intersection(set(p.feature_names))]

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    long_status_rows: list[dict[str, Any]] = []

    for _, row in seg_summary.iterrows():
        file_name = str(row.get("file_name", ""))
        out_row: dict[str, Any] = _metadata_prefix(row)
        feature_results: dict[str, FeatureValue] = {}
        context = _make_context(row, cfg)
        try:
            for plugin in plugins:
                try:
                    feature_results.update(plugin.compute(context))
                except Exception as exc:  # noqa: BLE001 - one plugin should not kill the file
                    for name in plugin.feature_names:
                        feature_results[name] = FeatureValue(name, np.nan, "failed", f"plugin_failed: {exc}")

            for _, feat in registry.iterrows():
                name = str(feat["feature"])
                subsystem = str(feat["subsystem"])
                result = feature_results.get(name)
                if result is not None:
                    out_row[name] = result.value
                    status = result.status
                    note = result.note
                elif name in IMPLEMENTED_FEATURES:
                    out_row[name] = np.nan
                    status = "not_selected_or_missing_input"
                    note = "feature_has_plugin_but_required_input_was_unavailable"
                else:
                    out_row[name] = np.nan
                    status = "not_implemented_yet"
                    note = "registered_from_uploaded_feature_notebook_pending_validated_implementation"
                long_status_rows.append(
                    {
                        "file_name": file_name,
                        "feature": name,
                        "subsystem": subsystem,
                        "status": status,
                        "note": note,
                    }
                )
            out_row["feature_extraction_status"] = "ok"
            rows.append(out_row)
        except Exception as exc:  # noqa: BLE001
            out_row["feature_extraction_status"] = "failed"
            rows.append(out_row)
            errors.append({"file_name": file_name, "status": "failed", "error": str(exc)})

    features_path = folders["tables"] / "acoustic_features_per_file.csv"
    status_path = folders["tables"] / "acoustic_feature_status_long.csv"
    registry_path = folders["tables"] / "selected_acoustic_feature_registry.csv"
    errors_path = folders["errors"] / "acoustic_feature_errors.csv"

    pd.DataFrame(rows).to_csv(features_path, index=False)
    pd.DataFrame(long_status_rows).to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    pd.DataFrame(errors).to_csv(errors_path, index=False)

    missingness_plot = folders["plots"] / "feature_missingness.png"
    subsystem_plot = folders["plots"] / "feature_subsystem_implementation_status.png"
    distribution_plot = folders["plots"] / "implemented_feature_distributions.png"
    task_plot = folders["plots"] / "task_feature_overview.png"
    _plot_feature_missingness(features_path, registry, missingness_plot)
    _plot_subsystem_status(status_path, subsystem_plot)
    _plot_implemented_feature_distributions(features_path, registry, distribution_plot)
    _plot_task_feature_overview(features_path, task_plot)

    report_path = folders["reports"] / "acoustic_feature_report.html"
    _write_feature_html_report(
        report_path,
        rows,
        long_status_rows,
        errors,
        cfg,
        missingness_plot,
        subsystem_plot,
        distribution_plot,
        task_plot,
    )

    computed_statuses = {"computed", "computed_proxy"}
    status_df = pd.DataFrame(long_status_rows)
    computed_features = sorted(status_df.loc[status_df["status"].isin(computed_statuses), "feature"].unique().tolist()) if not status_df.empty else []
    proxy_features = sorted(status_df.loc[status_df["status"].eq("computed_proxy"), "feature"].unique().tolist()) if not status_df.empty else []

    warnings: list[str] = []
    if proxy_features:
        warnings.append(f"Proxy features require reference validation before clinical interpretation: {proxy_features}")
    if errors:
        warnings.append(f"{len(errors)} files failed feature extraction")

    manifest = StageManifest(
        stage_name="acoustic_feature_extraction",
        stage_version="0.12.0",
        status="completed_with_warnings" if warnings else "completed",
        input_artifacts=[ArtifactRef(path=str(segmentation_summary_csv), role="segmentation_summary", media_type="text/csv")],
        output_artifacts=[
            ArtifactRef(path=str(features_path), role="features_per_file", media_type="text/csv"),
            ArtifactRef(path=str(status_path), role="feature_status_long", media_type="text/csv"),
            ArtifactRef(path=str(registry_path), role="selected_feature_registry", media_type="text/csv"),
            ArtifactRef(path=str(report_path), role="feature_html_report", media_type="text/html"),
        ],
        config=cfg.to_dict(),
        environment={"python": python_environment()},
        warnings=warnings,
        errors=errors,
        notes=[
            "Feature extraction now uses subsystem plugins and a region-aware signal policy.",
            f"Computed feature families in this pass: {computed_features}",
            "Registered-but-not-yet-implemented features remain explicit NaN placeholders.",
        ],
    )
    manifest_path = folders["logs"] / "stage_manifest.json"
    manifest.write_json(manifest_path)

    return StageResult(
        status=manifest.status,
        manifest_path=manifest_path,
        summary_table=features_path,
        error_table=errors_path,
        report_path=report_path,
    )


def _feature_names_in_table(features_csv: Path, registry: pd.DataFrame) -> list[str]:
    df_cols = pd.read_csv(features_csv, nrows=0).columns
    return [f for f in registry["feature"].tolist() if f in df_cols]


def _plot_feature_missingness(features_csv: Path, registry: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_csv)
    feature_names = [f for f in registry["feature"].tolist() if f in df.columns]
    if not feature_names:
        return
    miss = df[feature_names].isna().mean().sort_values(ascending=True)
    height = max(5.0, min(18.0, 0.18 * len(miss) + 2.0))
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(miss.index, miss.values)
    ax.set_xlabel("Fraction missing / not implemented")
    ax.set_xlim(0, 1)
    ax.set_title("VSLP Acoustic Feature Missingness")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_subsystem_status(status_csv: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(status_csv)
    if df.empty:
        return
    tmp = df.drop_duplicates(["feature", "subsystem", "status"])
    counts = tmp.groupby(["subsystem", "status"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Number of selected features")
    ax.set_title("Feature implementation status by subsystem")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_implemented_feature_distributions(features_csv: Path, registry: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_csv)
    names = [f for f in registry["feature"].astype(str).tolist() if f in df.columns and f in IMPLEMENTED_FEATURES]
    numeric_names = [n for n in names if pd.to_numeric(df[n], errors="coerce").notna().any()]
    if not numeric_names:
        return
    chosen = numeric_names[:16]
    n = len(chosen)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, max(3, 2.8 * nrows)))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, name in zip(axes_arr, chosen, strict=False):
        vals = pd.to_numeric(df[name], errors="coerce").dropna().values
        if vals.size:
            ax.hist(vals, bins=min(20, max(5, int(np.sqrt(vals.size)))))
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
    for ax in axes_arr[len(chosen):]:
        ax.axis("off")
    fig.suptitle("Implemented acoustic feature distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_task_feature_overview(features_csv: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_csv)
    if "task" not in df.columns or df.empty:
        return
    candidate_features = [f for f in ["percent_pause", "speech_dur", "f0_mean", "intensity_CV", "fft_peaks1", "RMSamp"] if f in df.columns]
    if not candidate_features:
        return
    task_counts = df["task"].fillna("unknown").astype(str).value_counts().head(12)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(task_counts.index, task_counts.values)
    ax.set_ylabel("Files")
    ax.set_title("Files by task in feature table")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_feature_html_report(
    path: Path,
    rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    cfg: FeatureExtractionConfig,
    missingness_plot: Path,
    subsystem_plot: Path,
    distribution_plot: Path,
    task_plot: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status_df = pd.DataFrame(status_rows)
    computed = int(status_df["status"].isin(["computed", "computed_proxy"]).sum()) if not status_df.empty else 0
    proxies = int((status_df["status"] == "computed_proxy").sum()) if not status_df.empty else 0
    pending = int((status_df["status"] == "not_implemented_yet").sum()) if not status_df.empty else 0
    files_ok = int(sum(1 for r in rows if r.get("feature_extraction_status") == "ok"))
    failed = len(errors)
    implemented = sorted(status_df.loc[status_df["status"].isin(["computed", "computed_proxy"]), "feature"].unique().tolist()) if not status_df.empty else []

    def img_block(title: str, image_path: Path) -> str:
        if not image_path.exists():
            return ""
        rel = Path("../plots") / image_path.name
        return f"<div class='card'><h2>{title}</h2><img src='{rel.as_posix()}'></div>"

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>VSLP Acoustic Feature Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; background:#071A2D; color:#EAF2F8; margin:32px; }}
.card {{ background:#0B253D; border:1px solid #315D7C; border-radius:12px; padding:18px; margin:14px 0; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:6px; background:#0E3A5B; margin-right:8px; margin-top:4px; }}
pre {{ white-space:pre-wrap; background:#102A43; padding:12px; border-radius:8px; }}
code {{ background:#102A43; padding:2px 5px; border-radius:4px; }}
img {{ max-width:100%; border-radius:10px; border:1px solid #315D7C; background:white; }}
.warning {{ color:#FFDFA8; }}
</style></head><body>
<h1>VSLP Acoustic Feature Extraction Report</h1>
<div class='card'><span class='badge'>Files OK: {files_ok}</span><span class='badge'>Files failed: {failed}</span><span class='badge'>Computed feature values: {computed}</span><span class='badge'>Proxy values: {proxies}</span><span class='badge'>Pending placeholders: {pending}</span></div>
<div class='card'><h2>Current implementation scope</h2><p>V0.12 uses a region-aware plugin architecture. Timing and rhythm features are computed from validated segmentation/preprocessed audio outputs. F0 and CPP are currently local engineering proxies and must be validated against the reference notebook/Praat-style definitions before clinical interpretation.</p><p class='warning'>Registered features that are not yet implemented remain explicit <code>NaN</code> placeholders with status <code>not_implemented_yet</code>.</p></div>
<div class='card'><h2>Computed feature names</h2><pre>{json.dumps(implemented, indent=2)}</pre></div>
{img_block('Feature missingness', missingness_plot)}
{img_block('Subsystem implementation status', subsystem_plot)}
{img_block('Implemented feature distributions', distribution_plot)}
{img_block('Task overview', task_plot)}
<div class='card'><h2>Configuration</h2><pre>{json.dumps(cfg.to_dict(), indent=2)}</pre></div>
</body></html>"""
    path.write_text(html, encoding="utf-8")
