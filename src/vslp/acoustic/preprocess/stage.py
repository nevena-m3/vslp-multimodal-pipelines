"""Canonical acoustic preprocessing stage.

Scientific contract
-------------------
Preprocessing produces one faithful analysis waveform per accepted Ingest record:
deterministic decode -> conservative mono-channel resolution -> optional constant DC-offset
removal and optional peak amplitude normalization -> native-rate mono FLOAT32 WAV ->
numerical integrity verification.

This stage does not perform acoustic quality classification, generic filtering,
or model-specific resampling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import json
import math
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.context import cleanup_stage, run_context
from vslp.acoustic.preprocess.audio import (
    ChannelResolution,
    apply_dc_offset_policy,
    decode_audio_ffmpeg,
    resolve_mono_channel,
)
from vslp.core.provenance import python_environment, sha256_file, tool_versions
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for canonical acoustic preprocessing."""

    remove_dc_offset: bool = True
    amplitude_normalization: bool = False
    ffmpeg_bin: str = "ffmpeg"
    audio_stream_selector: str = "0:a:0"
    duplicate_channel_correlation_min: float = 0.999
    duplicate_channel_gain_difference_db_max: float = 0.10
    silent_channel_rms_max: float = 1e-6
    plot_max_points: int = 12_000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MAIN_SUMMARY_COLUMNS = [
    "recording_id",
    "file_name",
    "source_file_path",
    "source_sha256",
    "project_name",
    "task_name",
    "run_id",
    "run_created_at_local",
    "run_created_at_utc",
    "status",
    "source_codec",
    "source_sample_rate_hz",
    "source_channels",
    "source_duration_sec",
    "n_audio_streams",
    "selected_audio_stream",
    "decoded_sample_rate_hz",
    "decoded_n_channels",
    "decoded_duration_sec",
    "selected_channel",
    "channel_resolution_status",
    "channel_resolution_reason",
    "remove_dc_offset",
    "amplitude_normalization",
    "normalization_scale",
    "dc_offset_before",
    "dc_offset_after",
    "dc_offset_removed",
    "analysis_sample_rate_hz",
    "analysis_n_channels",
    "analysis_duration_sec",
    "analysis_format",
    "analysis_subtype",
    "analysis_wav_path",
    "analysis_wav_sha256",
    "audit_plot_path",
    "waveform_max_abs_error_vs_expected_transform",
    "write_read_max_abs_error",
    "warning",
]


def safe_stem(path: str | Path) -> str:
    stem = Path(path).stem
    keep = [ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in stem]
    return "".join(keep).strip("_") or "audio"


def _as_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        df.to_csv(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _downsample_for_plot(x: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(x.shape[0])
    if n == 0:
        return np.array([], dtype=int), x
    step = max(1, int(math.ceil(n / max(1, int(max_points)))))
    idx = np.arange(0, n, step, dtype=int)
    return idx, x[idx]


def _shared_amplitude_limit(*arrays: np.ndarray) -> float:
    peaks = [float(np.max(np.abs(a))) for a in arrays if np.asarray(a).size]
    peak = max(peaks, default=1.0)
    return max(1e-6, 1.05 * peak)


def _choose_zoom_window(
    x: np.ndarray,
    sr: int,
    *,
    window_sec: float = 2.0,
    hop_sec: float = 0.25,
) -> tuple[int, int]:
    """Choose a high-energy local window for visual before/after inspection."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or x.size == 0 or sr <= 0:
        return 0, int(x.size)
    window = min(len(x), max(1, int(round(window_sec * sr))))
    if len(x) <= window:
        return 0, len(x)
    hop = max(1, int(round(hop_sec * sr)))
    best_start = 0
    best_rms = -1.0
    for start in range(0, len(x) - window + 1, hop):
        chunk = x[start : start + window].astype(np.float64)
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        if rms > best_rms:
            best_rms = rms
            best_start = start
    return best_start, min(len(x), best_start + window)


def _overview_envelope(
    x: np.ndarray,
    sr: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a peak-preserving min/max envelope for long-waveform display."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty
    max_points = max(100, int(max_points))
    if len(x) <= max_points:
        t = np.arange(len(x), dtype=float) / float(sr)
        values = x.astype(float)
        return t, values, values

    block = max(1, int(np.ceil(len(x) / max_points)))
    n_blocks = int(np.ceil(len(x) / block))
    padded = np.full(n_blocks * block, np.nan, dtype=np.float32)
    padded[: len(x)] = x
    blocks = padded.reshape(n_blocks, block)
    low = np.nanmin(blocks, axis=1).astype(float)
    high = np.nanmax(blocks, axis=1).astype(float)
    centers = (np.arange(n_blocks, dtype=float) * block + (block - 1) / 2.0) / float(sr)
    return centers, low, high


def _plot_overview(
    ax,
    x: np.ndarray,
    sr: int,
    *,
    max_points: int,
    title: str,
    y_limit: float,
) -> None:
    t, low, high = _overview_envelope(x, sr, max_points)
    if t.size:
        if np.array_equal(low, high):
            ax.plot(t, low, linewidth=0.65)
        else:
            ax.fill_between(t, low, high, alpha=0.85, linewidth=0.0)
    ax.axhline(0.0, linewidth=0.6, alpha=0.5)
    ax.set_xlim(0.0, len(x) / float(sr) if sr > 0 else 1.0)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_title(title)
    ax.set_ylabel("Amplitude")


def _write_audit_plot(
    path: Path,
    *,
    x_raw: np.ndarray,
    sr: int,
    resolution: ChannelResolution,
    file_name: str,
    recording_id: str,
    source_codec: str,
    x_selected: np.ndarray | None,
    x_analysis: np.ndarray | None,
    dc_before: float | None,
    dc_after: float | None,
    remove_dc_offset: bool,
    max_points: int,
) -> None:
    """Write a compact before/after audit with matched scales and a local difference view."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.78])
    ax_before = fig.add_subplot(grid[0, 0])
    ax_after = fig.add_subplot(grid[0, 1])
    ax_before_zoom = fig.add_subplot(grid[1, 0])
    ax_after_zoom = fig.add_subplot(grid[1, 1])
    ax_diff = fig.add_subplot(grid[2, :])

    if x_selected is not None and x_analysis is not None:
        full_limit = _shared_amplitude_limit(x_selected, x_analysis)
        _plot_overview(
            ax_before,
            x_selected,
            sr,
            max_points=max_points,
            title="A. Original waveform",
            y_limit=full_limit,
        )
        _plot_overview(
            ax_after,
            x_analysis,
            sr,
            max_points=max_points,
            title="B. Processed waveform",
            y_limit=full_limit,
        )

        start, stop = _choose_zoom_window(x_selected, sr)
        idx, before_zoom = _downsample_for_plot(x_selected[start:stop], min(max_points, 30_000))
        after_zoom = x_analysis[start:stop][idx]
        zoom_time = (start + idx) / float(sr)
        zoom_limit = _shared_amplitude_limit(before_zoom, after_zoom)

        ax_before_zoom.plot(zoom_time, before_zoom, linewidth=0.75)
        ax_after_zoom.plot(zoom_time, after_zoom, linewidth=0.75)
        for ax, title in (
            (ax_before_zoom, "C. Original waveform — zoom"),
            (ax_after_zoom, "D. Processed waveform — same zoom"),
        ):
            ax.axhline(0.0, linewidth=0.6, alpha=0.5)
            ax.set_ylim(-zoom_limit, zoom_limit)
            if zoom_time.size:
                ax.set_xlim(float(zoom_time[0]), float(zoom_time[-1]))
            ax.set_title(title)
            ax.set_ylabel("Amplitude")
            ax.set_xlabel("Time (s)")

        difference = (
            x_analysis[start:stop].astype(np.float64)
            - x_selected[start:stop].astype(np.float64)
        )[idx]
        ax_diff.plot(zoom_time, difference, linewidth=0.8)
        ax_diff.axhline(0.0, linewidth=0.7, alpha=0.6)
        if difference.size:
            lower = min(0.0, float(np.min(difference)))
            upper = max(0.0, float(np.max(difference)))
            span = upper - lower
            scale = max(abs(lower), abs(upper), 1e-12)
            margin = max(0.10 * span, 0.05 * scale, 1e-10)
            ax_diff.set_ylim(lower - margin, upper + margin)
            ax_diff.text(
                0.01,
                0.94,
                f"mean change = {float(np.mean(difference)):.3e}",
                transform=ax_diff.transAxes,
                va="top",
                fontsize=9,
            )
        ax_diff.set_title("E. Change introduced by preprocessing (processed − original)")
        ax_diff.set_ylabel("Δ amplitude")
        ax_diff.set_xlabel("Time (s)")
    else:
        # No canonical waveform exists for ambiguous multichannel audio. Show the
        # decoded source channels, but never imply that one was selected.
        source_limit = _shared_amplitude_limit(*[x_raw[:, ch] for ch in range(x_raw.shape[1])])
        idx, raw_plot = _downsample_for_plot(x_raw, max_points)
        t = idx / float(sr) if idx.size else np.array([])
        for ch in range(min(x_raw.shape[1], 8)):
            ax_before.plot(t, raw_plot[:, ch], linewidth=0.55, label=f"channel {ch}")
        ax_before.set_ylim(-source_limit, source_limit)
        ax_before.set_title("A. Original decoded channels")
        ax_before.set_ylabel("Amplitude")
        if x_raw.shape[1] <= 8:
            ax_before.legend(loc="upper right", fontsize=8)

        source_for_zoom = x_raw[:, int(np.argmax(resolution.per_channel_rms))]
        start, stop = _choose_zoom_window(source_for_zoom, sr)
        idx_zoom, _ = _downsample_for_plot(
            source_for_zoom[start:stop], min(max_points, 30_000)
        )
        zoom_t = (start + idx_zoom) / float(sr)
        for ch in range(min(x_raw.shape[1], 8)):
            ax_before_zoom.plot(
                zoom_t,
                x_raw[start:stop, ch][idx_zoom],
                linewidth=0.65,
                label=f"channel {ch}",
            )
        ax_before_zoom.set_title("C. Original decoded channels — zoom")
        ax_before_zoom.set_ylabel("Amplitude")
        ax_before_zoom.set_xlabel("Time (s)")

        for ax, title in (
            (ax_after, "B. Processed waveform"),
            (ax_after_zoom, "D. Processed waveform — zoom"),
            (ax_diff, "E. Change introduced by preprocessing"),
        ):
            ax.text(
                0.5,
                0.5,
                "No canonical waveform\nChannel review required",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

    duration = x_raw.shape[0] / float(sr)
    dc_state = "ON" if remove_dc_offset else "OFF"
    before_text = "n/a" if dc_before is None else f"{dc_before:.3e}"
    after_text = "n/a" if dc_after is None else f"{dc_after:.3e}"
    selected_text = "review" if resolution.selected_channel is None else str(resolution.selected_channel)
    fig.suptitle(
        f"{file_name}\n"
        f"{sr} Hz | {x_raw.shape[1]} channel(s) | selected channel {selected_text} | "
        f"duration {duration:.2f} s | DC removal {dc_state} | mean {before_text} → {after_text}",
        fontsize=11,
    )

    tmp = path.with_name(f".{path.stem}.{uuid4().hex}.tmp.png")
    try:
        fig.savefig(tmp, dpi=170)
        tmp.replace(path)
    finally:
        plt.close(fig)
        tmp.unlink(missing_ok=True)


def _write_float32_wav_verified(
    path: Path,
    x: np.ndarray,
    sr: int,
) -> tuple[str, float]:
    """Atomically write and independently re-read a mono FLOAT32 WAV."""
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Canonical WAV write requires non-empty mono audio")
    if not np.isfinite(x).all():
        raise ValueError("Canonical WAV write received NaN or infinite samples")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{uuid4().hex}.tmp.wav")
    try:
        sf.write(tmp, np.asarray(x, dtype=np.float32), int(sr), format="WAV", subtype="FLOAT")
        info = sf.info(tmp)
        if str(info.format).upper() != "WAV":
            raise RuntimeError(f"Canonical file format verification failed: {info.format}")
        if str(info.subtype).upper() != "FLOAT":
            raise RuntimeError(f"Canonical WAV subtype must be FLOAT, got {info.subtype}")
        if int(info.samplerate) != int(sr):
            raise RuntimeError(
                f"Canonical WAV sample rate changed during write: {sr} -> {info.samplerate}"
            )
        if int(info.channels) != 1:
            raise RuntimeError(f"Canonical WAV must be mono, got {info.channels} channels")
        if int(info.frames) != int(len(x)):
            raise RuntimeError(
                f"Canonical WAV frame count changed during write: {len(x)} -> {info.frames}"
            )

        reread, reread_sr = sf.read(tmp, dtype="float32", always_2d=False)
        reread = np.asarray(reread, dtype=np.float32)
        if int(reread_sr) != int(sr) or reread.ndim != 1 or len(reread) != len(x):
            raise RuntimeError("Canonical WAV failed independent read-back shape/rate verification")
        if not np.isfinite(reread).all():
            raise RuntimeError("Canonical WAV contains non-finite samples after write/read")
        max_abs_error = float(
            np.max(np.abs(reread.astype(np.float64) - x.astype(np.float64)))
        )
        tolerance = max(
            5e-7,
            4.0 * np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(x)))),
        )
        if max_abs_error > tolerance:
            raise RuntimeError(
                "Canonical WAV write/read changed waveform beyond float32 tolerance: "
                f"max_abs_error={max_abs_error:.3e}, tolerance={tolerance:.3e}"
            )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)

    return sha256_file(path), max_abs_error


def _source_path_from_ingest(row: pd.Series) -> Path:
    value = row.get("source_file_path", row.get("file_path", ""))
    if pd.isna(value) or not str(value).strip():
        raise ValueError("Ingest row is missing source_file_path")
    return Path(str(value)).expanduser()


def _base_row(row: pd.Series, context: dict[str, str]) -> dict[str, Any]:
    source_path = _source_path_from_ingest(row)
    source_sha = str(row.get("source_sha256", row.get("sha256", ""))).strip()
    recording_id = str(row.get("recording_id", source_sha)).strip()
    return {
        "recording_id": recording_id,
        "file_name": str(row.get("file_name", source_path.name)),
        "source_file_path": str(source_path),
        "source_sha256": source_sha,
        **context,
        "source_codec": row.get("audio_codec"),
        "source_sample_rate_hz": _as_int(row.get("sample_rate_hz")),
        "source_channels": _as_int(row.get("channels")),
        "source_duration_sec": _as_float(row.get("duration_sec")),
        "n_audio_streams": _as_int(row.get("n_audio_streams")),
        "selected_audio_stream": row.get("audio_stream_selector", "0:a:0"),
        "status": "pending",
        "warning": "",
    }


def _preprocess_one(
    row: pd.Series,
    *,
    stage_dir: Path,
    cfg: PreprocessConfig,
    context: dict[str, str],
) -> dict[str, Any]:
    out = _base_row(row, context)
    source_path = Path(out["source_file_path"])
    source_sha = str(out["source_sha256"])

    if len(source_sha) != 64:
        raise ValueError(f"Ingest row has invalid source_sha256 for {source_path.name}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Ingest-accepted source file is missing: {source_path}")

    current_sha = sha256_file(source_path)
    if current_sha != source_sha:
        raise RuntimeError(
            "Source file changed after Ingest; refusing to preprocess. "
            f"ingest_sha256={source_sha}, current_sha256={current_sha}, file={source_path}"
        )

    x_raw, sr = decode_audio_ffmpeg(
        source_path,
        target_sr=None,
        mono=False,
        ffmpeg_bin=cfg.ffmpeg_bin,
        audio_stream_selector=cfg.audio_stream_selector,
    )
    out["decoded_sample_rate_hz"] = int(sr)
    out["decoded_n_channels"] = int(x_raw.shape[1])
    out["decoded_n_samples"] = int(x_raw.shape[0])
    out["decoded_duration_sec"] = float(x_raw.shape[0] / sr)

    ingest_sr = out.get("source_sample_rate_hz")
    if ingest_sr is not None and int(ingest_sr) != int(sr):
        raise RuntimeError(
            f"Native sample rate changed during decode: ingest={ingest_sr}, decoded={sr}"
        )
    ingest_channels = out.get("source_channels")
    if ingest_channels is not None and int(ingest_channels) != int(x_raw.shape[1]):
        raise RuntimeError(
            "Channel count changed during decode: "
            f"ingest={ingest_channels}, decoded={x_raw.shape[1]}"
        )

    resolution = resolve_mono_channel(
        x_raw,
        duplicate_correlation_min=cfg.duplicate_channel_correlation_min,
        duplicate_gain_difference_db_max=cfg.duplicate_channel_gain_difference_db_max,
        silent_channel_rms_max=cfg.silent_channel_rms_max,
    )
    out["channel_resolution_status"] = resolution.status
    out["channel_resolution_reason"] = resolution.reason
    out["selected_channel"] = resolution.selected_channel
    out["channel_metrics_json"] = json.dumps(resolution.to_dict(), sort_keys=True)

    stem = safe_stem(source_path)
    base_name = f"{stem}__{source_sha[:12]}"
    audit_path = stage_dir / "plots" / f"{base_name}__preprocessing_audit.png"
    out["audit_plot_path"] = str(audit_path)

    if resolution.selected_channel is None:
        out["status"] = "needs_channel_review"
        out["warning"] = (
            "Multiple non-silent channels were not equivalent under conservative "
            "duplicate-channel criteria; canonical audio was intentionally withheld."
        )
        _write_audit_plot(
            audit_path,
            x_raw=x_raw,
            sr=sr,
            resolution=resolution,
            file_name=out["file_name"],
            recording_id=out["recording_id"],
            source_codec=str(out.get("source_codec") or ""),
            x_selected=None,
            x_analysis=None,
            dc_before=None,
            dc_after=None,
            remove_dc_offset=cfg.remove_dc_offset,
            max_points=cfg.plot_max_points,
        )
        return out

    selected = np.asarray(x_raw[:, int(resolution.selected_channel)], dtype=np.float32)
    analysis, dc_before, dc_after = apply_dc_offset_policy(
        selected,
        remove_dc_offset=cfg.remove_dc_offset,
    )

    if cfg.remove_dc_offset:
        expected = selected.astype(np.float64) - dc_before
    else:
        expected = selected.astype(np.float64)
    normalization_scale = float(np.max(np.abs(analysis))) if cfg.amplitude_normalization else 1.0
    if cfg.amplitude_normalization and normalization_scale > 0:
        analysis = (analysis.astype(np.float64) / normalization_scale).astype(np.float32)
        expected /= normalization_scale
    waveform_error = float(np.max(np.abs(analysis.astype(np.float64) - expected)))
    transform_tolerance = max(
        5e-7,
        4.0 * np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(selected)))),
    )
    if waveform_error > transform_tolerance:
        raise RuntimeError(
            "Preprocessing transform verification failed: "
            f"max_abs_error={waveform_error:.3e}, tolerance={transform_tolerance:.3e}"
        )

    if cfg.remove_dc_offset:
        dc_tolerance = max(
            1e-7,
            8.0 * np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(analysis)))),
        )
        if abs(dc_after) > dc_tolerance:
            raise RuntimeError(
                f"DC-offset removal verification failed: mean_after={dc_after:.3e}, "
                f"tolerance={dc_tolerance:.3e}"
            )

    analysis_path = stage_dir / "audio" / f"{base_name}__analysis.wav"
    analysis_sha, write_read_error = _write_float32_wav_verified(analysis_path, analysis, sr)

    info = sf.info(analysis_path)
    out.update(
        {
            "status": "ok",
            "remove_dc_offset": bool(cfg.remove_dc_offset),
            "amplitude_normalization": bool(cfg.amplitude_normalization),
            "normalization_scale": normalization_scale,
            "dc_offset_before": dc_before,
            "dc_offset_after": dc_after,
            "dc_offset_removed": float(dc_before - dc_after),
            "analysis_sample_rate_hz": int(info.samplerate),
            "analysis_n_channels": int(info.channels),
            "analysis_n_samples": int(info.frames),
            "analysis_duration_sec": float(info.frames / info.samplerate),
            "analysis_format": str(info.format),
            "analysis_subtype": str(info.subtype),
            "analysis_wav_path": str(analysis_path),
            "analysis_wav_sha256": analysis_sha,
            "waveform_max_abs_error_vs_expected_transform": waveform_error,
            "write_read_max_abs_error": write_read_error,
        }
    )

    if int(info.samplerate) != int(sr):
        raise RuntimeError("Canonical sample rate no longer matches decoded native sample rate")
    if int(info.frames) != int(len(selected)):
        raise RuntimeError("Canonical duration/frame count no longer matches decoded selected channel")

    _write_audit_plot(
        audit_path,
        x_raw=x_raw,
        sr=sr,
        resolution=resolution,
        file_name=out["file_name"],
        recording_id=out["recording_id"],
        source_codec=str(out.get("source_codec") or ""),
        x_selected=selected,
        x_analysis=analysis,
        dc_before=dc_before,
        dc_after=dc_after,
        remove_dc_offset=cfg.remove_dc_offset,
        max_points=cfg.plot_max_points,
    )
    return out


def _main_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in MAIN_SUMMARY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df.loc[:, MAIN_SUMMARY_COLUMNS]


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    cfg: PreprocessConfig,
) -> None:
    df = pd.DataFrame(rows)
    n_ok = int((df.get("status") == "ok").sum()) if not df.empty else 0
    n_review = int((df.get("status") == "needs_channel_review").sum()) if not df.empty else 0
    n_failed = int((df.get("status") == "failed").sum()) if not df.empty else 0

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('file_name', '')))}</td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            f"<td>{html.escape(str(row.get('decoded_sample_rate_hz', '')))}</td>"
            f"<td>{html.escape(str(row.get('decoded_n_channels', '')))}</td>"
            f"<td>{html.escape(str(row.get('selected_channel', '')))}</td>"
            f"<td>{html.escape(str(row.get('remove_dc_offset', '')))}</td>"
            f"<td><code>{html.escape(str(row.get('analysis_wav_path', '')))}</code></td>"
            f"<td><code>{html.escape(str(row.get('audit_plot_path', '')))}</code></td>"
            "</tr>"
        )

    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>VSLP Preprocessing Report</title>
<style>
body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif; background:#071A2D; color:#EAF2F8; margin:32px; }}
.card {{ background:#0B253D; border:1px solid #315D7C; border-radius:12px; padding:18px; margin:14px 0; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:6px; background:#0E3A5B; margin-right:8px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }}
th,td {{ border-bottom:1px solid #315D7C; padding:7px; text-align:left; vertical-align:top; }}
code {{ word-break:break-all; color:#8FD9C5; }}
</style></head><body>
<h1>VSLP Preprocessing Report</h1>
<div class="card">
<span class="badge">Input: {len(rows)}</span>
<span class="badge">Processed: {n_ok}</span>
<span class="badge">Channel review: {n_review}</span>
<span class="badge">Failed: {n_failed}</span>
</div>
<div class="card"><h2>Configuration</h2><pre>{html.escape(json.dumps(cfg.to_dict(), indent=2))}</pre></div>
<div class="card"><h2>Files</h2>
<table><tr><th>File</th><th>Status</th><th>Sample rate</th><th>Channels</th><th>Selected</th>
<th>DC removal</th><th>Canonical WAV</th><th>Audit plot</th></tr>
{''.join(table_rows)}
</table></div>
<div class="card">Issues: {len(issues)}</div>
</body></html>
"""
    _atomic_write_text(path, body)


@cleanup_stage
def run_acoustic_preprocess(
    ingest_summary_csv: str | Path,
    output_root: str | Path,
    config: PreprocessConfig | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> StageResult:
    """Create canonical analysis WAVs from the immutable accepted Ingest record set."""
    cfg = config or PreprocessConfig()
    output_root = Path(output_root)
    ingest_summary_csv = Path(ingest_summary_csv)
    stage_dir = output_root / "acoustic" / "001_preprocess"
    context = run_context(output_root)

    if not ingest_summary_csv.is_file():
        raise FileNotFoundError(f"Ingest summary not found: {ingest_summary_csv}")
    ingest_df = pd.read_csv(ingest_summary_csv)
    required = {"ingest_status", "source_sha256", "recording_id"}
    if not required.issubset(ingest_df.columns):
        missing = sorted(required.difference(ingest_df.columns))
        raise ValueError(f"Ingest summary is missing required columns: {missing}")

    accepted = ingest_df.loc[
        ingest_df["ingest_status"].astype(str).str.lower().eq("ok")
    ].copy()
    if accepted.empty:
        raise ValueError("Ingest summary contains no accepted records to preprocess")

    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    if progress_callback:
        progress_callback(0, len(accepted), "Preprocess")

    for index, (_, ingest_row) in enumerate(accepted.iterrows(), start=1):
        base = _base_row(ingest_row, context)
        try:
            result = _preprocess_one(
                ingest_row,
                stage_dir=stage_dir,
                cfg=cfg,
                context=context,
            )
            rows.append(result)
            if result.get("status") == "needs_channel_review":
                issues.append(
                    {
                        "recording_id": result.get("recording_id"),
                        "file_name": result.get("file_name"),
                        "source_file_path": result.get("source_file_path"),
                        "status": "needs_channel_review",
                        "error": result.get("warning", ""),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - one file must not terminate a batch
            failed = {
                **base,
                "status": "failed",
                "warning": str(exc),
            }
            rows.append(failed)
            issues.append(
                {
                    "recording_id": base.get("recording_id"),
                    "file_name": base.get("file_name"),
                    "source_file_path": base.get("source_file_path"),
                    "status": "failed",
                    "error": str(exc),
                }
            )
        finally:
            if progress_callback:
                progress_callback(index, len(accepted), f"Preprocess — {ingest_row.get('file_name', '')}")

    n_input = len(accepted)
    n_ok = sum(row.get("status") == "ok" for row in rows)
    n_review = sum(row.get("status") == "needs_channel_review" for row in rows)
    n_failed = sum(row.get("status") == "failed" for row in rows)
    if n_input != n_ok + n_review + n_failed:
        raise RuntimeError("Preprocess accounting invariant failed")

    detailed_path = stage_dir / "tables" / "acoustic_preprocess_summary.csv"
    main_path = stage_dir / "tables" / "acoustic_preprocess_main_summary.csv"
    report_path = stage_dir / "reports" / "acoustic_preprocess_report.html"
    config_path = output_root / "configs" / "preprocess_config.json"

    # Keep the detailed preprocessing table schema stable even when every
    # recording is withheld or fails before a canonical WAV is produced.
    # Downstream code and audit tests must be able to distinguish
    # "field is not applicable / unavailable" (NaN) from "column missing".
    detailed_df = pd.DataFrame(rows)
    for col in MAIN_SUMMARY_COLUMNS:
        if col not in detailed_df.columns:
            detailed_df[col] = np.nan
    extra_columns = [col for col in detailed_df.columns if col not in MAIN_SUMMARY_COLUMNS]
    detailed_df = detailed_df.loc[:, [*MAIN_SUMMARY_COLUMNS, *extra_columns]]

    _atomic_write_csv(detailed_path, detailed_df)
    _atomic_write_csv(main_path, _main_summary(rows))
    _atomic_write_text(
        config_path,
        json.dumps(
            {
                "preserve_native_sample_rate": True,
                "output_channels": 1,
                "output_format": "WAV",
                "output_subtype": "FLOAT",
                **cfg.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write_report(report_path, rows, issues, cfg)

    issues_path: Path | None = None
    if issues:
        issues_path = stage_dir / "errors" / "acoustic_preprocess_errors.csv"
        _atomic_write_csv(issues_path, pd.DataFrame(issues))

    if n_ok == 0:
        stage_status = "failed"
    elif issues:
        stage_status = "completed_with_warnings"
    else:
        stage_status = "completed"

    artifacts = [
        ArtifactRef(path=str(detailed_path), role="preprocess_summary", media_type="text/csv"),
        ArtifactRef(path=str(main_path), role="preprocess_main_summary", media_type="text/csv"),
        ArtifactRef(path=str(report_path), role="preprocess_report", media_type="text/html"),
        ArtifactRef(path=str(config_path), role="preprocess_config", media_type="application/json"),
    ]
    if issues_path is not None:
        artifacts.append(
            ArtifactRef(path=str(issues_path), role="preprocess_issues", media_type="text/csv")
        )

    manifest = StageManifest(
        stage_name="acoustic_preprocess",
        stage_version="1.0.0",
        status=stage_status,
        input_artifacts=[
            ArtifactRef(
                path=str(ingest_summary_csv),
                role="ingest_summary",
                media_type="text/csv",
                sha256=sha256_file(ingest_summary_csv),
            )
        ],
        output_artifacts=artifacts,
        config={
            "accepted_ingest_records_only": True,
            "preserve_native_sample_rate": True,
            "output_channels": 1,
            "output_subtype": "FLOAT",
            **cfg.to_dict(),
        },
        environment={"python": python_environment(), "tools": tool_versions(ffmpeg=cfg.ffmpeg_bin)},
        warnings=[
            item
            for item in (
                f"{n_review} recording(s) require explicit channel review" if n_review else "",
                f"{n_failed} recording(s) failed preprocessing" if n_failed else "",
            )
            if item
        ],
        errors=[item for item in issues if item.get("status") == "failed"],
        notes=[
            "Source files are never modified.",
            "Canonical audio is native-rate mono WAV with IEEE 32-bit float samples.",
            "DC-offset removal follows the run configuration; no other global waveform transform is applied.",
            "Ambiguous multichannel recordings are withheld rather than silently downmixed.",
            "Audit plots are visual diagnostics; numerical invariants are enforced in code.",
        ],
    )
    manifest_path = stage_dir / "logs" / "stage_manifest.json"
    manifest.write_json(manifest_path)

    return StageResult(
        status=stage_status,
        manifest_path=manifest_path,
        summary_table=detailed_path,
        error_table=issues_path,
        report_path=report_path,
    )
