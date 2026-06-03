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
