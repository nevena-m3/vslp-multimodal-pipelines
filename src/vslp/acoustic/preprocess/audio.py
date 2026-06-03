"""Audio decoding and preprocessing primitives.

Policy: never modify source files. All transforms return arrays and metadata; stage code writes outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from scipy import signal


@dataclass(frozen=True)
class AudioQualitySummary:
    duration_sec: float
    sample_rate_hz: int
    n_samples: int
    n_channels: int
    dc_offset: float
    peak_abs: float
    rms: float
    clipping_fraction_near_full_scale: float
    clipping_run_count: int
    snr_db_estimate: float | None
    powerline_50hz_flag: bool
    powerline_60hz_flag: bool

    def to_dict(self):
        return asdict(self)


def decode_audio_ffmpeg(file_path: str | Path, target_sr: int | None = None, mono: bool = True, ffmpeg_bin: str = "ffmpeg") -> tuple[np.ndarray, int]:
    """Decode arbitrary audio/video media to float waveform using ffmpeg."""
    file_path = Path(file_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        cmd = [str(ffmpeg_bin), "-y", "-i", str(file_path), "-vn"]
        if mono:
            cmd += ["-ac", "1"]
        if target_sr is not None:
            cmd += ["-ar", str(target_sr)]
        cmd += ["-f", "wav", tmp_wav]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {file_path}: {p.stderr}")
        x, sr = sf.read(tmp_wav, always_2d=False)
        x = np.asarray(x, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(x, -1.0, 1.0), int(sr)
    finally:
        if os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass


def dc_offset_remove(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x - np.nanmean(x, axis=0)


def choose_best_mono_channel(x: np.ndarray) -> tuple[np.ndarray, dict]:
    """Select mono channel.

    If stereo/multichannel, choose the channel with the best proxy SNR: highest RMS
    among non-clipped channels. This avoids destructive averaging when one channel is bad.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x, {"stereo_policy": "already_mono", "selected_channel": 0}
    metrics = []
    for ch in range(x.shape[1]):
        c = x[:, ch]
        rms = float(np.sqrt(np.mean(c**2))) if len(c) else 0.0
        clip_frac = float(np.mean(np.abs(c) >= 0.98)) if len(c) else 1.0
        score = rms - 10.0 * clip_frac
        metrics.append((score, ch, rms, clip_frac))
    score, ch, rms, clip_frac = max(metrics, key=lambda t: t[0])
    return x[:, ch], {"stereo_policy": "best_channel_by_rms_clip_penalty", "selected_channel": int(ch), "selected_channel_rms": rms, "selected_channel_clip_fraction": clip_frac}


def count_clipping_runs(x: np.ndarray, threshold: float = 0.999, min_run: int = 3) -> int:
    mask = np.abs(np.asarray(x)) >= threshold
    if not mask.any():
        return 0
    padded = np.concatenate([[False], mask, [False]])
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return int(np.sum((ends - starts) >= min_run))


def estimate_snr_db_low_energy(x: np.ndarray, frame_sec: float = 0.03, sr: int = 16000, noise_quantile: float = 0.1) -> float | None:
    """Estimate SNR using low-energy frames as noise floor.

    This is an estimated SNR, not reference SNR. Use for QC only.
    """
    x = np.asarray(x, dtype=np.float32)
    frame_len = max(1, int(frame_sec * sr))
    n_frames = len(x) // frame_len
    if n_frames < 3:
        return None
    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)
    power = np.mean(frames**2, axis=1) + 1e-12
    noise = float(np.quantile(power, noise_quantile))
    signal_power = float(np.mean(power))
    if noise <= 0 or signal_power <= noise:
        return None
    return float(10 * np.log10((signal_power - noise) / noise))


def detect_powerline_interference(x: np.ndarray, sr: int, base_freqs: tuple[int, int] = (50, 60), harmonic_max_hz: int = 300, prominence_db: float = 10.0) -> dict[str, bool]:
    """Flag likely 50/60 Hz powerline energy using Welch PSD local prominence."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < sr:
        return {"powerline_50hz_flag": False, "powerline_60hz_flag": False}
    freqs, psd = signal.welch(x, fs=sr, nperseg=min(len(x), 4096))
    psd_db = 10 * np.log10(psd + 1e-20)
    flags = {}
    for base in base_freqs:
        hit = False
        for f0 in range(base, harmonic_max_hz + 1, base):
            idx = np.argmin(np.abs(freqs - f0))
            local = (freqs >= f0 - 5) & (freqs <= f0 + 5)
            if local.sum() > 3 and psd_db[idx] - np.median(psd_db[local]) > prominence_db:
                hit = True
                break
        flags[f"powerline_{base}hz_flag"] = bool(hit)
    return flags


def summarize_audio_quality(x: np.ndarray, sr: int) -> AudioQualitySummary:
    x = np.asarray(x, dtype=np.float32)
    n_channels = 1 if x.ndim == 1 else x.shape[1]
    mono = x if x.ndim == 1 else np.mean(x, axis=1)
    power_flags = detect_powerline_interference(mono, sr)
    return AudioQualitySummary(
        duration_sec=float(len(mono) / sr) if sr > 0 else float("nan"),
        sample_rate_hz=int(sr),
        n_samples=int(len(mono)),
        n_channels=int(n_channels),
        dc_offset=float(np.mean(mono)) if len(mono) else float("nan"),
        peak_abs=float(np.max(np.abs(mono))) if len(mono) else float("nan"),
        rms=float(np.sqrt(np.mean(mono**2))) if len(mono) else float("nan"),
        clipping_fraction_near_full_scale=float(np.mean(np.abs(mono) >= 0.98)) if len(mono) else float("nan"),
        clipping_run_count=count_clipping_runs(mono),
        snr_db_estimate=estimate_snr_db_low_energy(mono, sr=sr),
        powerline_50hz_flag=power_flags["powerline_50hz_flag"],
        powerline_60hz_flag=power_flags["powerline_60hz_flag"],
    )
