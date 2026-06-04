"""Acoustic preprocessing stage.

Conservative policy:
- source files are never modified;
- QC is always measured;
- DC-offset removal is enabled by default;
- amplitude normalization and filters are optional and explicitly documented;
- feature WAVs preserve the original sampling rate unless resampling is requested;
- segmentation WAVs default to 16 kHz for Silero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

from vslp.acoustic.ingest.stage import DEFAULT_AUDIO_EXTENSIONS
from vslp.acoustic.preprocess.audio import (
    choose_best_mono_channel,
    count_clipping_runs,
    dc_offset_remove,
    decode_audio_ffmpeg,
    detect_powerline_interference,
    estimate_snr_db_low_energy,
    peak_normalize,
    summarize_audio_quality,
)
from vslp.core.io import discover_files
from vslp.core.project import ensure_stage_folders
from vslp.core.provenance import python_environment, sha256_file, tool_versions
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


@dataclass(frozen=True)
class FilterConfig:
    """Optional offline filter configuration."""

    enabled: bool = False
    kind: str = "none"  # none, lpf, hpf, bpf, notch
    low_hz: float | None = None
    high_hz: float | None = None
    notch_hz: float | None = None
    notch_q: float = 30.0
    order: int = 4


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for acoustic preprocessing."""

    segmentation_sample_rate_hz: int = 16000
    feature_sample_rate_hz: int | None = None
    make_segmentation_wav: bool = True
    make_feature_wav: bool = True
    remove_dc_offset: bool = True
    normalize_peak: bool = False
    normalize_peak_target_abs: float = 0.95
    mono_policy: str = "best_channel"
    filter: FilterConfig = FilterConfig()
    ffmpeg_bin: str = "ffmpeg"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_stem(path: str | Path) -> str:
    stem = Path(path).stem
    keep = []
    for ch in stem:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "audio"


def _apply_optional_filter(x: np.ndarray, sr: int, cfg: FilterConfig) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    x = np.asarray(x, dtype=np.float32)
    if not cfg.enabled or cfg.kind == "none":
        return x, warnings

    nyq = sr / 2.0
    kind = cfg.kind.lower()
    try:
        if kind == "lpf":
            if cfg.high_hz is None or not (0 < cfg.high_hz < nyq):
                raise ValueError(f"LPF requires 0 < high_hz < Nyquist ({nyq:.1f})")
            sos = signal.butter(cfg.order, cfg.high_hz, btype="lowpass", fs=sr, output="sos")
            return signal.sosfiltfilt(sos, x).astype(np.float32), warnings

        if kind == "hpf":
            if cfg.low_hz is None or not (0 < cfg.low_hz < nyq):
                raise ValueError(f"HPF requires 0 < low_hz < Nyquist ({nyq:.1f})")
            sos = signal.butter(cfg.order, cfg.low_hz, btype="highpass", fs=sr, output="sos")
            return signal.sosfiltfilt(sos, x).astype(np.float32), warnings

        if kind == "bpf":
            if cfg.low_hz is None or cfg.high_hz is None or not (0 < cfg.low_hz < cfg.high_hz < nyq):
                raise ValueError(f"BPF requires 0 < low_hz < high_hz < Nyquist ({nyq:.1f})")
            sos = signal.butter(cfg.order, [cfg.low_hz, cfg.high_hz], btype="bandpass", fs=sr, output="sos")
            return signal.sosfiltfilt(sos, x).astype(np.float32), warnings

        if kind == "notch":
            if cfg.notch_hz is None or not (0 < cfg.notch_hz < nyq):
                raise ValueError(f"notch requires 0 < notch_hz < Nyquist ({nyq:.1f})")
            b, a = signal.iirnotch(w0=cfg.notch_hz, Q=cfg.notch_q, fs=sr)
            return signal.filtfilt(b, a, x).astype(np.float32), warnings

        raise ValueError(f"Unknown filter kind: {cfg.kind}")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"optional filter failed; unfiltered audio used instead: {exc}")
        return x, warnings


def _write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(x, dtype=np.float32), sr, subtype="PCM_16")


def _resample_if_needed(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if int(sr_from) == int(sr_to):
        return np.asarray(x, dtype=np.float32)
    return signal.resample_poly(np.asarray(x, dtype=np.float32), up=int(sr_to), down=int(sr_from)).astype(np.float32)


def _preprocess_one(path: Path, folders: dict[str, Path], cfg: PreprocessConfig) -> dict[str, Any]:
    x_raw, sr_raw = decode_audio_ffmpeg(path, target_sr=None, mono=False, ffmpeg_bin=cfg.ffmpeg_bin)
    raw_qc = summarize_audio_quality(x_raw, sr_raw).to_dict()

    if cfg.mono_policy != "best_channel":
        raise NotImplementedError(f"Unsupported mono_policy for v1: {cfg.mono_policy}")
    x_mono, mono_info = choose_best_mono_channel(x_raw)

    x_processed = np.asarray(x_mono, dtype=np.float32)
    dc_removed = False
    if cfg.remove_dc_offset:
        x_processed = dc_offset_remove(x_processed)
        dc_removed = True

    filter_warnings: list[str]
    x_processed, filter_warnings = _apply_optional_filter(x_processed, sr_raw, cfg.filter)

    normalization_info = {
        "normalization_applied": False,
        "normalization_reason": "disabled",
        "original_peak_abs": float(np.max(np.abs(x_processed))) if len(x_processed) else np.nan,
        "target_peak_abs": cfg.normalize_peak_target_abs,
        "gain": 1.0,
    }
    if cfg.normalize_peak:
        x_processed, normalization_info = peak_normalize(x_processed, target_peak=cfg.normalize_peak_target_abs)

    x_processed = np.clip(np.nan_to_num(x_processed, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

    stem = safe_stem(path)
    source_hash = sha256_file(path)
    base_name = f"{stem}__{source_hash[:12]}"

    feature_wav_path: Path | None = None
    segmentation_wav_path: Path | None = None

    feature_sr = sr_raw if cfg.feature_sample_rate_hz is None else int(cfg.feature_sample_rate_hz)
    if cfg.make_feature_wav:
        x_feature = _resample_if_needed(x_processed, sr_raw, feature_sr)
        feature_wav_path = folders["artifacts"] / "feature_wav" / f"{base_name}__feature.wav"
        _write_wav(feature_wav_path, x_feature, feature_sr)

    segmentation_sr = int(cfg.segmentation_sample_rate_hz)
    if cfg.make_segmentation_wav:
        x_seg = _resample_if_needed(x_processed, sr_raw, segmentation_sr)
        segmentation_wav_path = folders["artifacts"] / "segmentation_wav" / f"{base_name}__seg16k.wav"
        _write_wav(segmentation_wav_path, x_seg, segmentation_sr)

    processed_qc = summarize_audio_quality(x_processed, sr_raw).to_dict()
    power_flags = detect_powerline_interference(x_processed, sr_raw)
    snr = estimate_snr_db_low_energy(x_processed, sr=sr_raw)
    clipping_fraction = float(np.mean(np.abs(x_processed) >= 0.98)) if len(x_processed) else np.nan
    clipping_runs = count_clipping_runs(x_processed)

    qc_json_path = folders["reports"] / "per_file_qc_json" / f"{base_name}__qc.json"
    qc_payload = {
        "source_file": str(path),
        "source_sha256": source_hash,
        "raw_qc": raw_qc,
        "processed_qc": processed_qc,
        "mono_selection": mono_info,
        "normalization": normalization_info,
        "filter_warnings": filter_warnings,
        "config": cfg.to_dict(),
    }
    qc_json_path.parent.mkdir(parents=True, exist_ok=True)
    qc_json_path.write_text(json.dumps(qc_payload, indent=2), encoding="utf-8")

    return {
        "file_name": path.name,
        "file_path": str(path),
        "source_sha256": source_hash,
        "status": "ok",
        "raw_sample_rate_hz": sr_raw,
        "raw_duration_sec": raw_qc["duration_sec"],
        "raw_n_channels": raw_qc["n_channels"],
        "raw_dc_offset": raw_qc["dc_offset"],
        "processed_dc_offset": processed_qc["dc_offset"],
        "processed_peak_abs": processed_qc["peak_abs"],
        "processed_rms": processed_qc["rms"],
        "clipping_fraction_near_full_scale": clipping_fraction,
        "clipping_run_count": clipping_runs,
        "snr_db_estimate": snr,
        "powerline_50hz_flag": power_flags["powerline_50hz_flag"],
        "powerline_60hz_flag": power_flags["powerline_60hz_flag"],
        "dc_offset_removed": dc_removed,
        "normalize_peak": cfg.normalize_peak,
        "normalization_applied": normalization_info.get("normalization_applied"),
        "normalization_gain": normalization_info.get("gain"),
        "normalization_target_peak_abs": cfg.normalize_peak_target_abs,
        "filter_enabled": cfg.filter.enabled,
        "filter_kind": cfg.filter.kind,
        "filter_low_hz": cfg.filter.low_hz,
        "filter_high_hz": cfg.filter.high_hz,
        "filter_notch_hz": cfg.filter.notch_hz,
        "filter_order": cfg.filter.order,
        "mono_policy": mono_info.get("stereo_policy"),
        "selected_channel": mono_info.get("selected_channel"),
        "feature_wav_path": str(feature_wav_path) if feature_wav_path else None,
        "feature_sample_rate_hz": feature_sr if cfg.make_feature_wav else None,
        "feature_resampled": bool(cfg.make_feature_wav and feature_sr != sr_raw),
        "segmentation_wav_path": str(segmentation_wav_path) if segmentation_wav_path else None,
        "segmentation_sample_rate_hz": segmentation_sr if cfg.make_segmentation_wav else None,
        "segmentation_resampled": bool(cfg.make_segmentation_wav and segmentation_sr != sr_raw),
        "qc_json_path": str(qc_json_path),
        "filter_warning": "; ".join(filter_warnings),
    }


def _build_main_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    cols = [
        "file_name", "status", "raw_duration_sec", "raw_sample_rate_hz", "raw_n_channels",
        "snr_db_estimate", "clipping_fraction_near_full_scale", "clipping_run_count",
        "raw_dc_offset", "processed_dc_offset", "powerline_50hz_flag", "powerline_60hz_flag",
        "dc_offset_removed", "normalize_peak", "filter_enabled", "filter_kind",
        "feature_sample_rate_hz", "feature_resampled", "segmentation_sample_rate_hz", "segmentation_resampled",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


def _build_qc_flags(rows: list[dict[str, Any]]) -> pd.DataFrame:
    out = []
    for r in rows:
        snr = r.get("snr_db_estimate")
        clip = float(r.get("clipping_fraction_near_full_scale") or 0.0)
        dc_raw = abs(float(r.get("raw_dc_offset") or 0.0))
        dur = float(r.get("raw_duration_sec") or 0.0)
        flags = []
        if pd.notna(snr) and snr is not None and float(snr) < 10.0:
            flags.append("low_snr")
        if clip > 0:
            flags.append("clipping")
        if dc_raw > 0.01:
            flags.append("dc_offset_present")
        if bool(r.get("powerline_50hz_flag")):
            flags.append("50hz_powerline")
        if bool(r.get("powerline_60hz_flag")):
            flags.append("60hz_powerline")
        if dur < 1.0:
            flags.append("very_short")
        out.append({
            "file_name": r.get("file_name"),
            "qc_status": "review" if flags else "ok",
            "flags": ";".join(flags),
            "n_flags": len(flags),
        })
    return pd.DataFrame(out)


def run_acoustic_preprocess(
    input_path: str | Path,
    output_root: str | Path,
    config: PreprocessConfig | None = None,
    extensions: list[str] | None = None,
) -> StageResult:
    cfg = config or PreprocessConfig()
    extensions = extensions or DEFAULT_AUDIO_EXTENSIONS
    stage_dir = Path(output_root) / "acoustic" / "002_preprocess"
    folders = ensure_stage_folders(stage_dir)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    files = discover_files(input_path, extensions)

    for path in files:
        try:
            rows.append(_preprocess_one(path, folders, cfg))
        except Exception as exc:  # noqa: BLE001
            errors.append({"file_name": path.name, "file_path": str(path), "status": "failed", "error": str(exc)})

    detailed_path = folders["tables"] / "acoustic_preprocess_summary.csv"
    main_path = folders["tables"] / "acoustic_preprocess_main_summary.csv"
    qc_flags_path = folders["tables"] / "acoustic_preprocess_qc_flags.csv"
    errors_path = folders["errors"] / "acoustic_preprocess_errors.csv"

    pd.DataFrame(rows).to_csv(detailed_path, index=False)
    _build_main_summary(rows).to_csv(main_path, index=False)
    _build_qc_flags(rows).to_csv(qc_flags_path, index=False)
    pd.DataFrame(errors).to_csv(errors_path, index=False)

    _write_preprocess_plots(folders["plots"], rows)
    report_path = folders["reports"] / "acoustic_preprocess_report.html"
    _write_preprocess_html_report(report_path, rows, errors, cfg)

    manifest = StageManifest(
        stage_name="acoustic_preprocess",
        stage_version="0.2.0",
        status="completed_with_warnings" if errors else "completed",
        output_artifacts=[
            ArtifactRef(path=str(detailed_path), role="preprocess_detailed_summary", media_type="text/csv"),
            ArtifactRef(path=str(main_path), role="preprocess_main_summary", media_type="text/csv"),
            ArtifactRef(path=str(qc_flags_path), role="preprocess_qc_flags", media_type="text/csv"),
            ArtifactRef(path=str(errors_path), role="preprocess_errors", media_type="text/csv"),
            ArtifactRef(path=str(report_path), role="preprocess_html_report", media_type="text/html"),
        ],
        config={"input_path": str(input_path), "extensions": extensions, **cfg.to_dict()},
        environment={"python": python_environment(), "tools": tool_versions(ffmpeg=cfg.ffmpeg_bin)},
        warnings=[f"{len(errors)} files failed preprocessing"] if errors else [],
        errors=errors,
        notes=[
            "SNR is estimated from low-energy frames and is QC-only, not reference SNR.",
            "Filters and normalization are optional because they can alter downstream acoustic features.",
        ],
    )
    manifest_path = folders["logs"] / "stage_manifest.json"
    manifest.write_json(manifest_path)

    return StageResult(
        status=manifest.status,
        manifest_path=manifest_path,
        summary_table=detailed_path,
        error_table=errors_path,
        report_path=report_path,
    )


def _write_preprocess_plots(plot_dir: Path, rows: list[dict[str, Any]]) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        return

    def save_hist(col: str, title: str, filename: str, xlabel: str) -> None:
        vals = pd.to_numeric(df.get(col), errors="coerce").dropna()
        if vals.empty:
            return
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        ax.hist(vals, bins=min(20, max(5, len(vals))))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Files")
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=160)
        plt.close(fig)

    save_hist("snr_db_estimate", "Estimated SNR distribution", "preprocess_snr_distribution.png", "Estimated SNR (dB)")
    save_hist("clipping_fraction_near_full_scale", "Clipping fraction distribution", "preprocess_clipping_distribution.png", "Fraction near full scale")

    if {"raw_dc_offset", "processed_dc_offset"}.issubset(df.columns):
        vals = pd.DataFrame({
            "Raw DC offset": pd.to_numeric(df["raw_dc_offset"], errors="coerce"),
            "Processed DC offset": pd.to_numeric(df["processed_dc_offset"], errors="coerce"),
        }).dropna(how="all")
        if not vals.empty:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111)
            ax.boxplot([vals["Raw DC offset"].dropna(), vals["Processed DC offset"].dropna()], tick_labels=["Raw", "Processed"])
            ax.set_title("DC offset before and after preprocessing")
            ax.set_ylabel("Mean amplitude offset")
            fig.tight_layout()
            fig.savefig(plot_dir / "preprocess_dc_offset_before_after.png", dpi=160)
            plt.close(fig)


def _write_preprocess_html_report(path: Path, rows: list[dict[str, Any]], errors: list[dict[str, Any]], cfg: PreprocessConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = len(rows)
    failed = len(errors)
    flags_50 = sum(bool(r.get("powerline_50hz_flag")) for r in rows)
    flags_60 = sum(bool(r.get("powerline_60hz_flag")) for r in rows)
    clipped = sum(float(r.get("clipping_fraction_near_full_scale") or 0) > 0 for r in rows)
    normed = sum(bool(r.get("normalization_applied")) for r in rows)
    filtered = sum(bool(r.get("filter_enabled")) for r in rows)
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>VSLP Acoustic Preprocess Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; background:#071A2D; color:#EAF2F8; margin:32px; }}
.card {{ background:#0B253D; border:1px solid #315D7C; border-radius:12px; padding:18px; margin:14px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }} th, td {{ border-bottom:1px solid #315D7C; padding:8px; text-align:left; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:6px; background:#0E3A5B; margin-right:8px; margin-bottom:6px; }}
pre {{ white-space: pre-wrap; }}
</style></head><body>
<h1>VSLP Acoustic Preprocess Report</h1>
<div class='card'>
<span class='badge'>Processed: {ok}</span><span class='badge'>Failed: {failed}</span><span class='badge'>Clipping flags: {clipped}</span><span class='badge'>50 Hz flags: {flags_50}</span><span class='badge'>60 Hz flags: {flags_60}</span><span class='badge'>Normalized: {normed}</span><span class='badge'>Filtered: {filtered}</span>
</div>
<div class='card'><h2>Configuration</h2><pre>{json.dumps(cfg.to_dict(), indent=2)}</pre></div>
</body></html>"""
    path.write_text(html, encoding="utf-8")
