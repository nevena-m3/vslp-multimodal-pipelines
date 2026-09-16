"""Acoustic QC dashboard stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class AcousticQCConfig:
    """QC thresholds used to flag files.

    These defaults are conservative research-screening thresholds. They are not
    clinical acceptance criteria and should be tuned once enough real data are reviewed.
    """

    min_duration_sec: float = 0.5
    min_snr_db: float = 10.0
    max_clipping_fraction: float = 0.001
    min_speech_fraction: float = 0.05
    max_speech_fraction: float = 0.98
    min_speech_segments: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_acoustic_qc_dashboard(output_root: str | Path, config: AcousticQCConfig | None = None) -> StageResult:
    cfg = config or AcousticQCConfig()
    output_root = Path(output_root)
    stage_dir = output_root / "acoustic" / "006_qc_dashboard"
    folders = ensure_stage_folders(stage_dir)

    preprocess_csv = output_root / "acoustic" / "001_preprocess" / "tables" / "acoustic_preprocess_summary.csv"
    segmentation_csv = output_root / "acoustic" / "002_segmentation" / "tables" / "acoustic_segmentation_summary.csv"
    feature_status_csv = output_root / "acoustic" / "005_features" / "tables" / "acoustic_feature_status_long.csv"

    warnings: list[str] = []
    errors: list[dict[str, Any]] = []
    if not preprocess_csv.exists():
        raise FileNotFoundError(f"Missing preprocess summary: {preprocess_csv}")

    pre = pd.read_csv(preprocess_csv)
    seg = pd.read_csv(segmentation_csv) if segmentation_csv.exists() else pd.DataFrame()
    if seg.empty:
        warnings.append("Segmentation summary missing or empty; segmentation QC fields will be unavailable.")
    status = pd.read_csv(feature_status_csv) if feature_status_csv.exists() else pd.DataFrame()

    qc = pre.copy()
    if not seg.empty and "file_name" in seg.columns:
        keep = [c for c in ["file_name", "speech_fraction", "n_speech_segments", "n_internal_nonspeech_segments", "leading_nonspeech_sec", "trailing_nonspeech_sec"] if c in seg.columns]
        qc = qc.merge(seg[keep], on="file_name", how="left")

    flags: list[dict[str, Any]] = []
    for _, row in qc.iterrows():
        file_name = str(row.get("file_name", ""))
        file_flags: list[str] = []
        dur = _to_float(row.get("raw_duration_sec", row.get("duration_sec", np.nan)))
        snr = _to_float(row.get("snr_db_estimate", np.nan))
        clip = _to_float(row.get("clipping_fraction_near_full_scale", 0.0))
        speech_fraction = _to_float(row.get("speech_fraction", np.nan))
        n_speech = _to_float(row.get("n_speech_segments", np.nan))
        if np.isfinite(dur) and dur < cfg.min_duration_sec:
            file_flags.append("too_short")
        if np.isfinite(snr) and snr < cfg.min_snr_db:
            file_flags.append("low_snr")
        if np.isfinite(clip) and clip > cfg.max_clipping_fraction:
            file_flags.append("clipping")
        if bool(row.get("powerline_50hz_flag", False)):
            file_flags.append("50hz_powerline")
        if bool(row.get("powerline_60hz_flag", False)):
            file_flags.append("60hz_powerline")
        if np.isfinite(speech_fraction) and speech_fraction < cfg.min_speech_fraction:
            file_flags.append("very_low_speech_fraction")
        if np.isfinite(speech_fraction) and speech_fraction > cfg.max_speech_fraction:
            file_flags.append("very_high_speech_fraction")
        if np.isfinite(n_speech) and n_speech < cfg.min_speech_segments:
            file_flags.append("no_or_too_few_speech_segments")
        flags.append({
            "file_name": file_name,
            "qc_status": "review" if file_flags else "pass",
            "qc_flags": ";".join(file_flags),
            "n_qc_flags": len(file_flags),
        })

    flags_df = pd.DataFrame(flags)
    qc = qc.merge(flags_df, on="file_name", how="left") if "file_name" in qc.columns else qc

    # Feature completeness by file.
    if not status.empty and "file_name" in status.columns:
        comp = status.assign(is_computed=status["status"].isin(["computed", "computed_proxy"]))
        comp = comp.groupby("file_name").agg(
            selected_feature_values=("feature", "count"),
            computed_feature_values=("is_computed", "sum"),
        ).reset_index()
        comp["feature_completion_fraction"] = comp["computed_feature_values"] / comp["selected_feature_values"].replace(0, np.nan)
        qc = qc.merge(comp, on="file_name", how="left")

    qc_csv = folders["tables"] / "acoustic_qc_dashboard.csv"
    flag_counts_csv = folders["tables"] / "qc_flag_counts.csv"
    errors_path = folders["errors"] / "acoustic_qc_errors.csv"
    qc.to_csv(qc_csv, index=False)
    if "qc_flags" in qc.columns:
        all_flags = []
        for txt in qc["qc_flags"].fillna("").astype(str):
            all_flags.extend([x for x in txt.split(";") if x])
        flag_counts = pd.Series(all_flags).value_counts().rename_axis("flag").reset_index(name="count")
    else:
        flag_counts = pd.DataFrame(columns=["flag", "count"])
    flag_counts.to_csv(flag_counts_csv, index=False)
    pd.DataFrame(errors).to_csv(errors_path, index=False)

    qc_bar = folders["plots"] / "qc_pass_review_counts.png"
    snr_plot = folders["plots"] / "qc_snr_distribution.png"
    speech_plot = folders["plots"] / "qc_speech_fraction_distribution.png"
    flag_plot = folders["plots"] / "qc_flag_counts.png"
    _plot_pass_review(qc, qc_bar)
    _plot_hist(qc, "snr_db_estimate", snr_plot, "Estimated SNR distribution", "SNR estimate (dB)")
    _plot_hist(qc, "speech_fraction", speech_plot, "Speech fraction distribution", "Speech fraction")
    _plot_flags(flag_counts, flag_plot)

    report_path = folders["reports"] / "acoustic_qc_dashboard.html"
    _write_qc_report(report_path, qc, flag_counts, cfg, warnings, qc_bar, snr_plot, speech_plot, flag_plot)

    manifest = StageManifest(
        stage_name="acoustic_qc_dashboard",
        stage_version="0.12.0",
        status="completed_with_warnings" if warnings or errors else "completed",
        input_artifacts=[
            ArtifactRef(path=str(preprocess_csv), role="preprocess_summary", media_type="text/csv"),
            ArtifactRef(path=str(segmentation_csv), role="segmentation_summary", media_type="text/csv") if segmentation_csv.exists() else ArtifactRef(path=str(segmentation_csv), role="segmentation_summary_missing", media_type="text/csv"),
        ],
        output_artifacts=[
            ArtifactRef(path=str(qc_csv), role="qc_dashboard", media_type="text/csv"),
            ArtifactRef(path=str(report_path), role="qc_html_report", media_type="text/html"),
        ],
        config=cfg.to_dict(),
        environment={"python": python_environment()},
        warnings=warnings,
        errors=errors,
        notes=["QC flags are research-screening indicators; thresholds should be tuned with real cohort data."],
    )
    manifest_path = folders["logs"] / "stage_manifest.json"
    manifest.write_json(manifest_path)

    return StageResult(status=manifest.status, manifest_path=manifest_path, summary_table=qc_csv, error_table=errors_path, report_path=report_path)


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return np.nan


def _plot_pass_review(qc: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = qc.get("qc_status", pd.Series(dtype=str)).fillna("unknown").value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_ylabel("Files")
    ax.set_title("QC pass/review counts")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_hist(qc: pd.DataFrame, col: str, path: Path, title: str, xlabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if col not in qc.columns:
        return
    vals = pd.to_numeric(qc[col], errors="coerce").dropna()
    if vals.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals.values, bins=min(30, max(5, int(np.sqrt(len(vals))))))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Files")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_flags(flag_counts: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if flag_counts.empty:
        return
    df = flag_counts.sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df) + 2)))
    ax.barh(df["flag"].astype(str), df["count"].astype(int))
    ax.set_xlabel("Files")
    ax.set_title("QC flag counts")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_qc_report(path: Path, qc: pd.DataFrame, flag_counts: pd.DataFrame, cfg: AcousticQCConfig, warnings: list[str], qc_bar: Path, snr_plot: Path, speech_plot: Path, flag_plot: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(qc)
    n_review = int((qc.get("qc_status", pd.Series(dtype=str)) == "review").sum()) if n else 0
    def img(title: str, p: Path) -> str:
        if not p.exists():
            return ""
        return f"<div class='card'><h2>{title}</h2><img src='../plots/{p.name}'></div>"
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>VSLP Acoustic QC Dashboard</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif; background:#071A2D; color:#EAF2F8; margin:32px; }}
.card {{ background:#0B253D; border:1px solid #315D7C; border-radius:12px; padding:18px; margin:14px 0; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:6px; background:#0E3A5B; margin-right:8px; margin-top:4px; }}
pre {{ white-space:pre-wrap; background:#102A43; padding:12px; border-radius:8px; }}
img {{ max-width:100%; border-radius:10px; border:1px solid #315D7C; background:white; }}
</style></head><body>
<h1>VSLP Acoustic QC Dashboard</h1>
<div class='card'><span class='badge'>Files: {n}</span><span class='badge'>Review: {n_review}</span><span class='badge'>Pass: {n - n_review}</span></div>
<div class='card'><h2>QC thresholds</h2><pre>{cfg.to_dict()}</pre></div>
<div class='card'><h2>Warnings</h2><pre>{warnings}</pre></div>
{img('QC pass/review counts', qc_bar)}
{img('SNR distribution', snr_plot)}
{img('Speech fraction distribution', speech_plot)}
{img('QC flag counts', flag_plot)}
</body></html>"""
    path.write_text(html, encoding="utf-8")
