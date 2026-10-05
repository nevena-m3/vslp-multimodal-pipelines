"""Explicit experimental F02 estimators; none claims source numerical parity."""

from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy import signal
from scipy.linalg import solve_toeplitz

from vslp.acoustic.features.family02_dsp import butterworth_sos


GNE_REPLICATION = {
    "band_centers_hz": [500, 1500, 2500],  # source lineage
    "bandwidth_hz": 1000,  # source lineage
    "filter_type": "Butterworth bandpass",  # VSLP replication choice
    "filter_order": 4,  # VSLP replication choice
    "lag_search_ms": 1.0,  # VSLP replication choice
    "internal_sample_rate_hz": 8000,
    "frame_length_ms": 30, "frame_step_ms": 10,
    "lpc_order": 10, "lpc_window": "Hamming",
    "envelope_preprocessing": "Hilbert magnitude, demean, no smoothing",
    "aggregation": "maximum normalized cross-band correlation",
}
PVI_REPLICATION = {
    "band_hz": [9, 14],  # source
    "welch_window_sec": 1.0,  # source
    "welch_overlap_fraction": .95,  # source
    "filter_order": 3,  # source lineage
    "interpolation_max_gap_sec": .02,  # VSLP replication choice
    "interpolation_max_fraction": .10,  # VSLP replication choice
    "spectrum_scaling": "sqrt(Welch scaling=spectrum, one-sided, Hann); sum bins",  # VSLP choice
}
HARMONIC_REPLICATION = {
    "n_harmonics": 8, "periods_per_window": 8, "samples_per_period": 512,
    "spectral_samples": 4096, "window": "Hamming",
    "scaling": "20log10(amplitude / global trial maximum)",
    "window_overlap_periods": 1, "window_hop_periods": 7,
}
CPP_PROFILE = {
    "pitch_floor_hz": 60, "pitch_ceiling_hz": 500, "time_step_sec": .002,
    "maximum_frequency_hz": 5000, "pre_emphasis_hz": 50,
    "interpolation": "Parabolic", "trend_quefrency_range_sec": [.001, .05],
    "trend_type": "Straight", "fit_method": "Robust slow",
    "smoothing": "none", "parselmouth_version": parselmouth.__version__,
    "embedded_praat_version": parselmouth.PRAAT_VERSION,
}


def gne_frames(audio: np.ndarray, sr: int) -> np.ndarray:
    """Framewise LPC-residual Hilbert-envelope correlations at internal 8 kHz."""
    x = np.asarray(audio, dtype=float)
    if x.size < sr // 10 or not np.isfinite(x).all() or sr < 16000:
        return np.array([])
    internal = GNE_REPLICATION["internal_sample_rate_hz"]
    from math import gcd
    divisor = gcd(sr, internal)
    x = signal.resample_poly(x, internal // divisor, sr // divisor)
    frame_length = round(.03 * internal)
    frame_step = round(.01 * internal)
    results = []
    half = GNE_REPLICATION["bandwidth_hz"] / 2
    filters = [butterworth_sos(4, half if center == half else
                              (center - half, center + half), internal,
                              "lowpass" if center == half else "bandpass")
               for center in GNE_REPLICATION["band_centers_hz"]]
    lag = round(internal * GNE_REPLICATION["lag_search_ms"] / 1000)
    for start in range(0, x.size - frame_length + 1, frame_step):
        frame = x[start:start + frame_length] * np.hamming(frame_length)
        correlation = signal.correlate(frame, frame, mode="full")
        acf = correlation[frame_length - 1:frame_length + 10]
        if acf[0] <= 1e-12:
            continue
        try:
            predictor = solve_toeplitz(acf[:10] + 1e-10 * acf[0] *
                                       np.r_[1, np.zeros(9)], acf[1:11])
        except (ValueError, np.linalg.LinAlgError):
            continue
        residual = signal.lfilter(np.r_[1, -predictor], [1], frame)
        envelopes = [np.abs(signal.hilbert(signal.sosfilt(sos, residual)))
                     for sos in filters]
        pairs = []
        for first in range(3):
            for second in range(first + 1, 3):
                for shift in range(-lag, lag + 1):
                    a = envelopes[first][max(shift, 0):frame_length + min(shift, 0)]
                    b = envelopes[second][max(-shift, 0):frame_length - max(shift, 0)]
                    if a.size > 3 and np.std(a) > 1e-12 and np.std(b) > 1e-12:
                        pairs.append(float(np.corrcoef(a, b)[0, 1]))
        if pairs:
            results.append(max(pairs))
    return np.asarray(results)


def gne_replication(audio: np.ndarray, sr: int) -> float:
    frames = gne_frames(audio, sr)
    return float(np.mean(frames)) if frames.size else np.nan


def pvi_replication(f0_hz: np.ndarray, time_sec: np.ndarray) -> tuple[float, float]:
    """9–14 Hz F0 modulation amplitude with a bounded missing-frame policy."""
    hz = np.asarray(f0_hz, dtype=float)
    time = np.asarray(time_sec, dtype=float)
    if hz.size < 3 or time.size != hz.size or not np.all(np.diff(time) > 0):
        return np.nan, np.nan
    valid = np.isfinite(hz) & (hz > 0)
    fraction = float(1 - np.mean(valid))
    if valid.sum() < 3 or fraction > PVI_REPLICATION["interpolation_max_fraction"]:
        return np.nan, fraction
    step = float(np.median(np.diff(time)))
    gaps = np.diff(np.flatnonzero(valid)) * step
    if (not valid[0] or not valid[-1] or
            np.any(gaps - step > PVI_REPLICATION["interpolation_max_gap_sec"])):
        return np.nan, fraction
    fs = 1 / step
    nperseg = round(fs * PVI_REPLICATION["welch_window_sec"])
    if hz.size < nperseg or fs <= 2 * PVI_REPLICATION["band_hz"][1]:
        return np.nan, fraction
    uniform = np.interp(time, time[valid], hz[valid])
    sos = butterworth_sos(PVI_REPLICATION["filter_order"],
                          tuple(PVI_REPLICATION["band_hz"]), fs, "bandpass")
    modulation = signal.sosfiltfilt(sos, uniform / np.mean(uniform) - 1)
    frequencies, power = signal.welch(
        modulation, fs=fs, window="hann", nperseg=nperseg,
        noverlap=round(nperseg * PVI_REPLICATION["welch_overlap_fraction"]),
        scaling="spectrum")
    band = (frequencies >= 9) & (frequencies <= 14)
    return float(np.sum(np.sqrt(power[band]))), fraction


def cpp_unsmoothed(audio: np.ndarray, sr: int, *, return_frames: bool = False):
    """Mean unsmoothed Praat PowerCepstrum peak prominence over 2-ms frames."""
    if len(audio) < sr // 10:
        return np.nan, 0
    sound = parselmouth.Sound(np.asarray(audio, dtype=float), sampling_frequency=sr)
    profile = CPP_PROFILE
    cepstrogram = call(sound, "To PowerCepstrogram", profile["pitch_floor_hz"],
                       profile["time_step_sec"], profile["maximum_frequency_hz"],
                       profile["pre_emphasis_hz"])
    first = 3 / profile["pitch_floor_hz"]
    last = len(audio) / sr - first
    values = []
    for time in np.arange(first, last, profile["time_step_sec"]):
        try:
            cepstrum = call(cepstrogram, "To PowerCepstrum (slice)", float(time))
            value = float(call(cepstrum, "Get peak prominence", 60, 500,
                               profile["interpolation"],
                               *profile["trend_quefrency_range_sec"],
                               profile["trend_type"], profile["fit_method"]))
            if np.isfinite(value):
                values.append(value)
        except (RuntimeError, ValueError):
            continue
    if return_frames:
        return np.asarray(values)
    return (float(np.mean(values)), len(values)) if values else (np.nan, 0)


def harmonic_windows(audio: np.ndarray, sr: int,
                     pulse_times_sec: np.ndarray) -> np.ndarray:
    """Raw H1–H8 amplitudes from eight-period, one-period-overlap windows."""
    pulses = np.asarray(pulse_times_sec, dtype=float)
    x = np.asarray(audio, dtype=float)
    rows = []
    for index in range(0, len(pulses) - 8, 7):
        left, right = pulses[index], pulses[index + 8]
        if right <= left or (right - left) / 8 > 1 / 60:
            continue
        window = x[round(left * sr):round(right * sr)]
        if window.size < 16:
            continue
        sampled = np.interp(np.linspace(0, 1, 4096, endpoint=False),
                            np.linspace(0, 1, window.size, endpoint=False), window)
        taper = signal.windows.hamming(sampled.size, sym=False)
        spectrum = np.fft.rfft(sampled * taper)
        amplitudes = np.abs(spectrum[np.arange(1, 9) * 8])
        rows.append(amplitudes)
    return np.asarray(rows).reshape(-1, 8)


def harmonic_statistics(windows: np.ndarray) -> dict[str, float]:
    """Normalize against one global maximum across harmonics and trial windows."""
    array = np.asarray(windows, dtype=float).reshape(-1, 8)
    if len(array) < 2 or np.max(array) <= 0 or not np.isfinite(array).all():
        return {}
    normalized = 20 * np.log10(np.maximum(array / np.max(array), 1e-12))
    mean = np.mean(normalized, axis=0)
    sd = np.std(normalized, axis=0, ddof=1)
    result = {}
    for index in range(8):
        p = index + 1
        result[f"h{p}_mean"] = float(mean[index])
        result[f"h{p}_sd"] = float(sd[index])
        denominator = abs(mean[index]) + sd[index]
        result[f"h{p}_relh"] = float(1 / denominator) if denominator > 0 else np.nan
    return result


def harmonic_replication(audio: np.ndarray, sr: int,
                         pulse_times_sec: np.ndarray) -> dict[str, float]:
    return harmonic_statistics(harmonic_windows(audio, sr, pulse_times_sec))
