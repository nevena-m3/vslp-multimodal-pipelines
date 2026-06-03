"""Shared audio utilities for acoustic feature plugins."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def read_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, always_2d=False, dtype="float32")
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = np.mean(x, axis=1).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x, int(sr)
    x = x - float(np.mean(x))
    peak = float(np.max(np.abs(x)))
    if peak > 1.0:
        x = x / peak
    return x.astype(np.float32), int(sr)


def frame_signal(x: np.ndarray, sr: int, frame_ms: float = 40.0, hop_ms: float = 10.0) -> tuple[np.ndarray, int, int]:
    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sr * hop_ms / 1000.0)))
    if x.size == 0:
        return np.empty((0, frame_len), dtype=np.float32), frame_len, hop_len
    n_frames = 1 + max(0, int(np.floor((len(x) - frame_len) / hop_len)))
    if n_frames <= 0:
        pad = frame_len - len(x)
        x_pad = np.pad(x, (0, pad), mode="constant")
        return x_pad.reshape(1, frame_len).astype(np.float32), frame_len, hop_len
    frames = np.lib.stride_tricks.sliding_window_view(x, frame_len)[::hop_len]
    return np.asarray(frames, dtype=np.float32), frame_len, hop_len


def rms_envelope(x: np.ndarray, sr: int, frame_ms: float = 30.0, hop_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    frames, _frame_len, hop_len = frame_signal(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if frames.size == 0:
        return np.array([]), np.array([])
    win = np.hanning(frames.shape[1]).astype(np.float32)
    rms = np.sqrt(np.mean((frames * win) ** 2, axis=1))
    t = (np.arange(len(rms)) * hop_len + frames.shape[1] / 2.0) / sr
    return t, rms.astype(float)


def robust_voiced_mask_from_rms(rms: np.ndarray) -> np.ndarray:
    if rms.size == 0:
        return np.array([], dtype=bool)
    finite = rms[np.isfinite(rms)]
    if finite.size == 0:
        return np.zeros_like(rms, dtype=bool)
    threshold = max(float(np.percentile(finite, 60)) * 0.35, float(np.percentile(finite, 20)))
    return rms > threshold


def simple_autocorr_f0(frame: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 500.0) -> float:
    frame = np.asarray(frame, dtype=float)
    if frame.size < 4 or np.max(np.abs(frame)) < 1e-5:
        return np.nan
    frame = frame - np.mean(frame)
    frame = frame * np.hanning(len(frame))
    corr = signal.correlate(frame, frame, mode="full", method="fft")
    corr = corr[len(corr) // 2:]
    if corr.size == 0 or corr[0] <= 0:
        return np.nan
    min_lag = max(1, int(sr / fmax))
    max_lag = min(len(corr) - 1, int(sr / fmin))
    if max_lag <= min_lag:
        return np.nan
    region = corr[min_lag:max_lag]
    if region.size == 0:
        return np.nan
    lag = min_lag + int(np.argmax(region))
    peak_ratio = corr[lag] / corr[0]
    if peak_ratio < 0.25:
        return np.nan
    return float(sr / lag)


def cepstral_peak_prominence_proxy(frame: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 500.0) -> float:
    frame = np.asarray(frame, dtype=float)
    if frame.size < 16 or np.max(np.abs(frame)) < 1e-5:
        return np.nan
    frame = (frame - np.mean(frame)) * np.hanning(len(frame))
    spec = np.abs(np.fft.rfft(frame)) + 1e-12
    cep = np.fft.irfft(np.log(spec))
    qmin = max(1, int(sr / fmax))
    qmax = min(len(cep) - 1, int(sr / fmin))
    if qmax <= qmin:
        return np.nan
    region = cep[qmin:qmax]
    peak = float(np.max(region))
    baseline = float(np.median(region))
    return float(20.0 * np.log10(max(abs(peak - baseline), 1e-12)))


def load_segments(path: Path | None):
    """Load a Silero segment table if available."""
    if path is None or not Path(path).exists():
        return None
    import pandas as pd

    df = pd.read_csv(path)
    required = {"segment_type", "start_sec", "end_sec", "duration_sec"}
    if not required.issubset(df.columns):
        return None
    return df


def mask_from_segments(
    n_samples: int,
    sr: int,
    segments,
    region: str = "speech_only",
    min_pause_duration_sec: float = 0.15,
) -> np.ndarray:
    """Build a sample mask from Silero speech/nonspeech segments.

    Parameters
    ----------
    region:
        - full_file: all samples.
        - speech_only: all Silero speech regions.
        - internal_pauses: nonspeech segments inside speech stream, excluding leading/trailing silence.
        - effective_task: speech + internal pauses, excluding leading/trailing silence.
    """
    mask = np.zeros(int(n_samples), dtype=bool)
    if n_samples <= 0:
        return mask
    region = (region or "speech_only").lower()
    if region == "full_file" or segments is None or len(segments) == 0:
        mask[:] = True
        return mask

    segs = segments.copy()
    if region == "speech_only":
        selected = segs.loc[segs["segment_type"].astype(str).str.lower().eq("speech")]
    elif region == "internal_pauses":
        if "segment_role" in segs.columns:
            selected = segs.loc[segs["segment_role"].astype(str).str.lower().eq("internal_nonspeech")]
        else:
            selected = segs.loc[segs["segment_type"].astype(str).str.lower().ne("speech")]
        if "duration_sec" in selected.columns:
            selected = selected.loc[selected["duration_sec"].astype(float) >= float(min_pause_duration_sec)]
    elif region == "effective_task":
        if "segment_role" in segs.columns:
            selected = segs.loc[~segs["segment_role"].astype(str).str.lower().isin(["leading_nonspeech", "trailing_nonspeech"])]
        else:
            selected = segs
    else:
        selected = segs.loc[segs["segment_type"].astype(str).str.lower().eq("speech")]

    for _, row in selected.iterrows():
        s = int(max(0, round(float(row["start_sec"]) * sr)))
        e = int(min(n_samples, round(float(row["end_sec"]) * sr)))
        if e > s:
            mask[s:e] = True
    if not mask.any() and region != "internal_pauses":
        # Fail open for signal features: if VAD produced no mask, use full audio but status notes will expose region policy.
        mask[:] = True
    return mask


def concatenate_masked_regions(x: np.ndarray, mask: np.ndarray, min_run_samples: int = 1) -> np.ndarray:
    """Concatenate contiguous True-mask regions into one analysis signal."""
    x = np.asarray(x, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if x.size == 0 or mask.size == 0 or not mask.any():
        return np.array([], dtype=np.float32)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    chunks = [x[s:e] for s, e in zip(starts, ends, strict=False) if e - s >= min_run_samples]
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def read_region_audio(
    wav_path: Path,
    segments_csv: Path | None,
    region: str = "speech_only",
    min_pause_duration_sec: float = 0.15,
) -> tuple[np.ndarray, int, str]:
    """Read canonical audio and return the selected analysis region.

    This is the core of region-aware feature extraction. It prevents speech features
    from being accidentally computed over leading/trailing silence or internal pauses.
    """
    x, sr = read_mono_audio(wav_path)
    if x.size == 0:
        return x, sr, "empty_audio"
    segs = load_segments(segments_csv)
    mask = mask_from_segments(
        n_samples=len(x),
        sr=sr,
        segments=segs,
        region=region,
        min_pause_duration_sec=min_pause_duration_sec,
    )
    min_run = max(1, int(round(0.02 * sr)))
    xr = concatenate_masked_regions(x, mask, min_run_samples=min_run)
    note = f"region={region}; selected_sec={len(xr)/sr:.3f}; source_sec={len(x)/sr:.3f}"
    if xr.size == 0:
        return xr, sr, note + "; no_selected_samples"
    return xr, sr, note
