"""Acoustic preprocessing stage.

This stage is intentionally conservative and non-destructive:
- source files are never modified;
- every successfully decoded input receives canonical PCM WAV outputs;
- QC metrics are written as CSV/JSON artifacts;
- failures are recorded and the batch continues.

The preprocessing stage creates two canonical audio files per input by default:
1. feature WAV: mono, DC-offset removed, original sample rate preserved unless configured otherwise.
2. segmentation WAV: mono, DC-offset removed, resampled to 16 kHz for Silero/segmentation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

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
    summarize_audio_quality,
)
from vslp.core.io import discover_files
from vslp.core.project import ensure_stage_folders
from vslp.core.provenance import python_environment, sha256_file, tool_versions
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


@dataclass(frozen=True)
class FilterConfig:
    """Optional offline filtering configuration.

    Filters are disabled by default. When enabled, they are applied with zero-phase
    second-order sections using scipy.signal.sosfiltfilt, which is suitable for
    offline research analysis but not for real-time causal inference.
    """

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
    mono_policy: str = "best_channel"  # v1 supports best_channel
    filter: FilterConfig = FilterConfig()
    ffmpeg_bin: str = "ffmpeg"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def safe_stem(path: str | Path) -> str:
    """Create a filesystem-safe stem while preserving enough human readability."""
    stem = Path(path).stem
    keep = []
    for ch in stem:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "audio"


def _apply_optional_filter(x: np.ndarray, sr: int, cfg: FilterConfig) -> tuple[np.ndarray, list[str]]:
    """Apply optional LPF/HPF/BPF/notch filtering and return warnings."""
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
    except Exception as exc:  # noqa: BLE001 - keep stage running but record the problem
        warnings.append(f"optional filter failed; unfiltered audio used instead: {exc}")
        return x, warnings


def _write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(x, dtype=np.float32), sr, subtype="PCM_16")


def _preprocess_one(path: Path, folders: dict[str, Path], cfg: PreprocessConfig) -> dict[str, Any]:
    """Preprocess one file and return one summary row."""
    # Decode without forcing mono first, so our selected-channel policy is documented.
    x_raw, sr_raw = decode_audio_ffmpeg(path, target_sr=None, mono=False, ffmpeg_bin=cfg.ffmpeg_bin)
    raw_qc = summarize_audio_quality(x_raw, sr_raw).to_dict()

    if cfg.mono_policy != "best_channel":
        raise NotImplementedError(f"Unsupported mono_policy for v1: {cfg.mono_policy}")
    x_mono, mono_info = choose_best_mono_channel(x_raw)

    x_processed = dc_offset_remove(x_mono) if cfg.remove_dc_offset else np.asarray(x_mono, dtype=np.float32)
    filter_warnings: list[str]
    x_processed, filter_warnings = _apply_optional_filter(x_processed, sr_raw, cfg.filter)
    x_processed = np.clip(np.nan_to_num(x_processed, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

    stem = safe_stem(path)
    source_hash_short = sha256_file(path)[:12]
    base_name = f"{stem}__{source_hash_short}"

    feature_wav_path: Path | None = None
    segmentation_wav_path: Path | None = None

    feature_sr = sr_raw if cfg.feature_sample_rate_hz is None else int(cfg.feature_sample_rate_hz)
    if cfg.make_feature_wav:
        if feature_sr != sr_raw:
            x_feature = signal.resample_poly(x_processed, up=feature_sr, down=sr_raw).astype(np.float32)
        else:
            x_feature = x_processed
        feature_wav_path = folders["artifacts"] / "feature_wav" / f"{base_name}__feature.wav"
        _write_wav(feature_wav_path, x_feature, feature_sr)
    else:
        x_feature = x_processed

    if cfg.make_segmentation_wav:
        segmentation_sr = int(cfg.segmentation_sample_rate_hz)
        if segmentation_sr != sr_raw:
            x_seg = signal.resample_poly(x_processed, up=segmentation_sr, down=sr_raw).astype(np.float32)
        else:
            x_seg = x_processed
        segmentation_wav_path = folders["artifacts"] / "segmentation_wav" / f"{base_name}__seg16k.wav"
        _write_wav(segmentation_wav_path, x_seg, segmentation_sr)
    else:
        x_seg = x_processed
        segmentation_sr = sr_raw

    processed_qc = summarize_audio_quality(x_processed, sr_raw).to_dict()
    power_flags = detect_powerline_interference(x_processed, sr_raw)
    snr = estimate_snr_db_low_energy(x_processed, sr=sr_raw)

    qc_json_path = folders["reports"] / "per_file_qc_json" / f"{base_name}__qc.json"
    qc_payload = {
        "source_file": str(path),
        "source_sha256": sha256_file(path),
        "raw_qc": raw_qc,
        "processed_qc": processed_qc,
        "mono_selection": mono_info,
        "filter_warnings": filter_warnings,
        "config": cfg.to_dict(),
    }
    qc_json_path.parent.mkdir(parents=True, exist_ok=True)
    qc_json_path.write_text(json.dumps(qc_payload, indent=2), encoding="utf-8")

    return {
        "file_name": path.name,
        "file_path": str(path),
        "source_sha256": qc_payload["source_sha256"],
        "status": "ok",
        "raw_sample_rate_hz": sr_raw,
        "raw_duration_sec": raw_qc["duration_sec"],
        "raw_n_channels": raw_qc["n_channels"],
        "raw_dc_offset": raw_qc["dc_offset"],
        "processed_dc_offset": processed_qc["dc_offset"],
        "processed_peak_abs": processed_qc["peak_abs"],
        "processed_rms": processed_qc["rms"],
        "clipping_fraction_near_full_scale": float(np.mean(np.abs(x_processed) >= 0.98)) if len(x_processed) else np.nan,
        "clipping_run_count": count_clipping_runs(x_processed),
        "snr_db_estimate": snr,
        "powerline_50hz_flag": power_flags["powerline_50hz_flag"],
        "powerline_60hz_flag": power_flags["powerline_60hz_flag"],
        "mono_policy": mono_info.get("stereo_policy"),
        "selected_channel": mono_info.get("selected_channel"),
        "feature_wav_path": str(feature_wav_path) if feature_wav_path else None,
        "feature_sample_rate_hz": feature_sr if cfg.make_feature_wav else None,
        "segmentation_wav_path": str(segmentation_wav_path) if segmentation_wav_path else None,
        "segmentation_sample_rate_hz": segmentation_sr if cfg.make_segmentation_wav else None,
        "qc_json_path": str(qc_json_path),
        "filter_warning": "; ".join(filter_warnings),
    }


def run_acoustic_preprocess(
    input_path: str | Path,
    output_root: str | Path,
    config: PreprocessConfig | None = None,
    extensions: list[str] | None = None,
) -> StageResult:
    """Run acoustic preprocessing over a file/folder.

    This accepts true WAV as well as mislabeled files such as WebM content saved with
    a .wav suffix, because decoding is delegated to ffmpeg rather than filename trust.
    """
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
        except Exception as exc:  # noqa: BLE001 - batch should continue
            errors.append({"file_name": path.name, "file_path": str(path), "status": "failed", "error": str(exc)})

    summary_path = folders["tables"] / "acoustic_preprocess_summary.csv"
    errors_path = folders["errors"] / "acoustic_preprocess_errors.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    pd.DataFrame(errors).to_csv(errors_path, index=False)

    report_path = folders["reports"] / "acoustic_preprocess_report.html"
    _write_preprocess_html_report(report_path, rows, errors, cfg)

    manifest = StageManifest(
        stage_name="acoustic_preprocess",
        stage_version="0.1.0",
        status="completed_with_warnings" if errors else "completed",
        output_artifacts=[
            ArtifactRef(path=str(summary_path), role="preprocess_summary", media_type="text/csv"),
            ArtifactRef(path=str(errors_path), role="preprocess_errors", media_type="text/csv"),
            ArtifactRef(path=str(report_path), role="preprocess_html_report", media_type="text/html"),
        ],
        config={"input_path": str(input_path), "extensions": extensions, **cfg.to_dict()},
        environment={"python": python_environment(), "tools": tool_versions(ffmpeg=cfg.ffmpeg_bin)},
        warnings=[f"{len(errors)} files failed preprocessing"] if errors else [],
        errors=errors,
        notes=[
            "SNR is estimated from low-energy frames and is QC-only, not reference SNR.",
            "Powerline flags are detection-only. Notch filtering is optional and disabled unless configured.",
        ],
    )
    manifest_path = folders["logs"] / "stage_manifest.json"
    manifest.write_json(manifest_path)

    return StageResult(
        status=manifest.status,
        manifest_path=manifest_path,
        summary_table=summary_path,
        error_table=errors_path,
        report_path=report_path,
    )


def _write_preprocess_html_report(path: Path, rows: list[dict[str, Any]], errors: list[dict[str, Any]], cfg: PreprocessConfig) -> None:
    """Write a small dependency-free HTML report for v1."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = len(rows)
    failed = len(errors)
    flags_50 = sum(bool(r.get("powerline_50hz_flag")) for r in rows)
    flags_60 = sum(bool(r.get("powerline_60hz_flag")) for r in rows)
    clipped = sum(float(r.get("clipping_fraction_near_full_scale") or 0) > 0 for r in rows)
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>VSLP Acoustic Preprocess Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; background:#071A2D; color:#EAF2F8; margin:32px; }}
.card {{ background:#0B253D; border:1px solid #315D7C; border-radius:12px; padding:18px; margin:14px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }} th, td {{ border-bottom:1px solid #315D7C; padding:8px; text-align:left; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:6px; background:#0E3A5B; margin-right:8px; }}
</style></head><body>
<h1>VSLP Acoustic Preprocess Report</h1>
<div class='card'>
<span class='badge'>Processed: {ok}</span><span class='badge'>Failed: {failed}</span><span class='badge'>50 Hz flags: {flags_50}</span><span class='badge'>60 Hz flags: {flags_60}</span><span class='badge'>Clipping flags: {clipped}</span>
</div>
<div class='card'><h2>Configuration</h2><pre>{json.dumps(cfg.to_dict(), indent=2)}</pre></div>
<div class='card'><h2>Notes</h2><p>SNR is estimated from low-energy frames and should be interpreted as a QC proxy only.</p><p>Source files are never modified. Canonical WAV files are written under <code>artifacts/</code>.</p></div>
</body></html>"""
    path.write_text(html, encoding="utf-8")
