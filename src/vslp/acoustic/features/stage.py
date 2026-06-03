"""Acoustic feature extraction stage v0.1.

This first production stage intentionally separates:
1. implemented, auditable features;
2. registered-but-not-yet-implemented features from the uploaded 73-feature notebook.

That is safer than silently computing approximate versions of clinical features. The first
implemented feature family is timing/respiratory features derived from Silero speech and
nonspeech segments. Remaining registered features are emitted as NaN with explicit status
so downstream aggregation and GUI review can handle them transparently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vslp.acoustic.features.registry import build_acoustic_feature_registry
from vslp.core.project import ensure_stage_folders
from vslp.core.provenance import python_environment
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult

IMPLEMENTED_FEATURES = {
    "total_dur",
    "speech_dur",
    "percent_pause",
    "num_pause",
    "mean_pause_dur",
    "mean_phrase_dur",
    "cv_pause_dur",
    "cv_phrase_dur",
    "total_pause_dur",
    "speech_rate",
}


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
    minimum_pause_duration_sec
        Internal nonspeech runs shorter than this are ignored for pause summary features.
    """

    selected_subsystems: list[str] = field(default_factory=list)
    selected_features: list[str] = field(default_factory=list)
    task_word_counts: dict[str, float] = field(default_factory=dict)
    metadata_csv: str | None = None
    minimum_pause_duration_sec: float = 0.15

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cv(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    mean = float(np.mean(arr))
    if abs(mean) < 1e-12:
        return np.nan
    return float(np.std(arr, ddof=0) / mean)


def _compute_timing_features(segments_csv: Path, duration_sec: float, task: str | None, cfg: FeatureExtractionConfig) -> dict[str, float]:
    segs = pd.read_csv(segments_csv)
    out: dict[str, float] = {}

    if segs.empty:
        return {name: np.nan for name in IMPLEMENTED_FEATURES}

    speech = segs.loc[segs["segment_type"] == "speech"].copy()
    internal_pause = segs.loc[segs["segment_role"] == "internal_nonspeech"].copy()
    if "duration_sec" in internal_pause.columns:
        internal_pause = internal_pause.loc[internal_pause["duration_sec"] >= cfg.minimum_pause_duration_sec]

    leading = float(segs.loc[segs["segment_role"] == "leading_nonspeech", "duration_sec"].sum()) if "segment_role" in segs else 0.0
    trailing = float(segs.loc[segs["segment_role"] == "trailing_nonspeech", "duration_sec"].sum()) if "segment_role" in segs else 0.0
    effective_dur = float(duration_sec) - leading - trailing
    if not np.isfinite(effective_dur) or effective_dur <= 0:
        effective_dur = float(duration_sec) if np.isfinite(duration_sec) and duration_sec > 0 else np.nan

    speech_dur = float(speech["duration_sec"].sum()) if not speech.empty else 0.0
    total_pause_dur = float(internal_pause["duration_sec"].sum()) if not internal_pause.empty else 0.0

    out["total_dur"] = effective_dur
    out["speech_dur"] = speech_dur
    out["total_pause_dur"] = total_pause_dur
    out["percent_pause"] = float(total_pause_dur / effective_dur) if np.isfinite(effective_dur) and effective_dur > 0 else np.nan
    out["num_pause"] = float(len(internal_pause))
    out["mean_pause_dur"] = float(internal_pause["duration_sec"].mean()) if not internal_pause.empty else 0.0
    out["mean_phrase_dur"] = float(speech["duration_sec"].mean()) if not speech.empty else 0.0
    out["cv_pause_dur"] = _cv(internal_pause["duration_sec"]) if not internal_pause.empty else np.nan
    out["cv_phrase_dur"] = _cv(speech["duration_sec"]) if not speech.empty else np.nan

    # Speech rate is only scientifically meaningful when the task word count is known.
    normalized_task = str(task).strip().lower() if task is not None and pd.notna(task) else ""
    word_count = cfg.task_word_counts.get(normalized_task)
    if word_count is not None and np.isfinite(effective_dur) and effective_dur > 0:
        out["speech_rate"] = float(word_count / effective_dur * 60.0)
    else:
        out["speech_rate"] = np.nan

    return out


def _select_registry(cfg: FeatureExtractionConfig) -> pd.DataFrame:
    registry = build_acoustic_feature_registry()
    if cfg.selected_subsystems:
        registry = registry.loc[registry["subsystem"].isin(cfg.selected_subsystems)].copy()
    if cfg.selected_features:
        registry = registry.loc[registry["feature"].isin(cfg.selected_features)].copy()
    return registry.reset_index(drop=True)


def run_acoustic_feature_extraction(
    segmentation_summary_csv: str | Path,
    output_root: str | Path,
    config: FeatureExtractionConfig | None = None,
) -> StageResult:
    """Extract acoustic features from segmentation outputs.

    V0.1 computes the timing/respiratory subset from the Silero segment tables and
    emits all selected registered features with explicit implementation status.
    """
    cfg = config or FeatureExtractionConfig()
    segmentation_summary_csv = Path(segmentation_summary_csv)
    stage_dir = Path(output_root) / "acoustic" / "004_features"
    folders = ensure_stage_folders(stage_dir)

    registry = _select_registry(cfg)
    seg_summary = pd.read_csv(segmentation_summary_csv)
    if cfg.metadata_csv:
        metadata_path = Path(cfg.metadata_csv).expanduser()
        if metadata_path.exists():
            meta = pd.read_csv(metadata_path)
            if "file_name" in meta.columns:
                keep_cols = [c for c in ["file_name", "subject_id", "session_id", "iteration", "task", "recording_date", "diagnosis", "severity_score", "severity_bin", "record_key"] if c in meta.columns]
                seg_summary = seg_summary.merge(meta[keep_cols], on="file_name", how="left", suffixes=("", "_metadata"))
                for col in ["subject_id", "session_id", "iteration", "task", "recording_date", "diagnosis", "severity_score", "severity_bin", "record_key"]:
                    mcol = f"{col}_metadata"
                    if mcol in seg_summary.columns:
                        if col in seg_summary.columns:
                            seg_summary[col] = seg_summary[col].combine_first(seg_summary[mcol])
                        else:
                            seg_summary[col] = seg_summary[mcol]
                        seg_summary = seg_summary.drop(columns=[mcol])
        else:
            raise FileNotFoundError(f"metadata_csv was provided but does not exist: {metadata_path}")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    long_status_rows: list[dict[str, Any]] = []

    for _, row in seg_summary.iterrows():
        file_name = str(row.get("file_name", ""))
        out_row: dict[str, Any] = {
            "file_name": file_name,
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
        try:
            segments_path = Path(str(row.get("segments_csv_path", "")))
            if not segments_path.exists():
                raise FileNotFoundError(f"Missing segments CSV: {segments_path}")

            timing = _compute_timing_features(
                segments_csv=segments_path,
                duration_sec=float(row.get("duration_sec", np.nan)),
                task=row.get("task", None),
                cfg=cfg,
            )

            for _, feat in registry.iterrows():
                name = str(feat["feature"])
                subsystem = str(feat["subsystem"])
                if name in IMPLEMENTED_FEATURES:
                    out_row[name] = timing.get(name, np.nan)
                    status = "computed"
                    note = "computed_from_silero_segments"
                else:
                    out_row[name] = np.nan
                    status = "not_implemented_yet"
                    note = "registered_from_uploaded_feature_notebook_pending_validated_implementation"
                long_status_rows.append({
                    "file_name": file_name,
                    "feature": name,
                    "subsystem": subsystem,
                    "status": status,
                    "note": note,
                })
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
    _plot_feature_missingness(features_path, registry, missingness_plot)
    _plot_subsystem_status(status_path, subsystem_plot)

    report_path = folders["reports"] / "acoustic_feature_report.html"
    _write_feature_html_report(report_path, rows, long_status_rows, errors, cfg, missingness_plot, subsystem_plot)

    manifest = StageManifest(
        stage_name="acoustic_feature_extraction",
        stage_version="0.1.0",
        status="completed_with_warnings" if errors else "completed",
        input_artifacts=[ArtifactRef(path=str(segmentation_summary_csv), role="segmentation_summary", media_type="text/csv")],
        output_artifacts=[
            ArtifactRef(path=str(features_path), role="features_per_file", media_type="text/csv"),
            ArtifactRef(path=str(status_path), role="feature_status_long", media_type="text/csv"),
            ArtifactRef(path=str(registry_path), role="selected_feature_registry", media_type="text/csv"),
            ArtifactRef(path=str(report_path), role="feature_html_report", media_type="text/html"),
        ],
        config=cfg.to_dict(),
        environment={"python": python_environment()},
        warnings=[
            "V0.1 computes timing/respiratory features only; remaining registered features are explicit NaN placeholders."
        ],
        errors=errors,
        notes=["This is intentionally conservative to avoid silent approximate clinical-feature implementations."],
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


def _write_feature_html_report(
    path: Path,
    rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    cfg: FeatureExtractionConfig,
    missingness_plot: Path,
    subsystem_plot: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status_df = pd.DataFrame(status_rows)
    computed = int((status_df["status"] == "computed").sum()) if not status_df.empty else 0
    pending = int((status_df["status"] == "not_implemented_yet").sum()) if not status_df.empty else 0
    files_ok = int(sum(1 for r in rows if r.get("feature_extraction_status") == "ok"))
    failed = len(errors)
    missing_rel = Path("../plots") / missingness_plot.name
    subsystem_rel = Path("../plots") / subsystem_plot.name
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>VSLP Acoustic Feature Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; background:#071A2D; color:#EAF2F8; margin:32px; }}
.card {{ background:#0B253D; border:1px solid #315D7C; border-radius:12px; padding:18px; margin:14px 0; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:6px; background:#0E3A5B; margin-right:8px; }}
pre {{ white-space:pre-wrap; background:#102A43; padding:12px; border-radius:8px; }}
img {{ max-width:100%; border-radius:10px; border:1px solid #315D7C; background:white; }}
</style></head><body>
<h1>VSLP Acoustic Feature Extraction Report</h1>
<div class='card'><span class='badge'>Files OK: {files_ok}</span><span class='badge'>Files failed: {failed}</span><span class='badge'>Computed feature values: {computed}</span><span class='badge'>Pending placeholders: {pending}</span></div>
<div class='card'><h2>Current implementation scope</h2><p>V0.1 computes timing/respiratory features from validated Silero segment tables. Registered-but-not-yet-implemented features are written as NaN with explicit status, not silently approximated.</p></div>
<div class='card'><h2>Feature missingness</h2><img src='{missing_rel.as_posix()}'></div>
<div class='card'><h2>Subsystem implementation status</h2><img src='{subsystem_rel.as_posix()}'></div>
<div class='card'><h2>Configuration</h2><pre>{json.dumps(cfg.to_dict(), indent=2)}</pre></div>
</body></html>"""
    path.write_text(html, encoding="utf-8")
