"""Versioned F06 acoustic landmarks, separate from linguistic Alignment V1.

Automatic scores are signal prominence diagnostics, never probabilities of
correct alignment or clinical confidence. Approved manual events take priority.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DETECTOR_VERSION = "f06-native-subevents-v1"
FRAME_SEC = .004
HOP_SEC = .001
DETECTION_BAND_HZ = (1500., 5000.)
BURST_PROMINENCE_RATIO = 4.0
MIN_DIGITAL_POWER = 1e-8
NOISE_PEAK_FRACTION = .20
MIN_NOISE_DURATION_SEC = .020
RISE_SMOOTH_FRAMES = 5
RISE_PERSIST_FRAMES = 3


@dataclass(frozen=True)
class Subevent:
    kind: str
    start_sec: float
    end_sec: float
    source: str
    prominence_ratio: float
    qc_status: str = "ACCEPTED"
    reviewer: str = ""
    review_timestamp: str = ""


def _frames(audio: np.ndarray, rate: int, start: float, end: float,
            band: tuple[float, float] = DETECTION_BAND_HZ) -> tuple[np.ndarray, np.ndarray]:
    """Hann-corrected one-sided digital band power on original-time frames."""
    if rate / 2 <= band[1]:
        raise ValueError("sample_rate_insufficient")
    signal = np.asarray(audio, dtype=float)
    if signal.ndim != 1 or not np.isfinite(signal).all() or not 0 <= start < end <= len(signal) / rate:
        raise ValueError("target_outside_eligible_audio")
    length, hop = round(rate * FRAME_SEC), round(rate * HOP_SEC)
    first, last = int(np.ceil(start * rate)), int(np.floor(end * rate))
    if length < 4 or hop < 1 or last - first < length:
        raise ValueError("insufficient_spectral_support")
    window = np.hanning(length)
    frequency = np.fft.rfftfreq(length, 1 / rate)
    mask = (frequency >= band[0]) & (frequency <= band[1])
    if not mask.any():
        raise ValueError("insufficient_spectral_support")
    centers, powers = [], []
    for index in range(first, last - length + 1, hop):
        frame = signal[index:index + length] - np.mean(signal[index:index + length])
        spectrum = np.abs(np.fft.rfft(frame * window)) ** 2
        if len(spectrum) > 2:
            spectrum[1:-1] *= 2
        centers.append((index + length / 2) / rate)
        powers.append(float(spectrum[mask].sum() / (length * np.mean(window ** 2))))
    return np.asarray(centers), np.asarray(powers)


def detect_stop_burst(audio: np.ndarray, rate: int, phone_start: float,
                      phone_end: float) -> Subevent:
    """Strongest positive high-band power rise, with a five-frame baseline."""
    times, power = _frames(audio, rate, phone_start, phone_end)
    if len(power) < 7 or float(np.max(power)) < MIN_DIGITAL_POWER:
        raise ValueError("burst_not_detected")
    baseline = np.asarray([np.median(power[max(0, i - 5):i]) if i >= 2 else power[0]
                           for i in range(len(power))])
    gain = power / np.maximum(baseline, MIN_DIGITAL_POWER)
    candidates = np.flatnonzero((gain >= BURST_PROMINENCE_RATIO)
                                & (power >= MIN_DIGITAL_POWER))
    if not len(candidates):
        raise ValueError("burst_low_prominence")
    # Strongest ratio; first frame wins exact ties. No arbitrary maximum on silence.
    index = int(candidates[np.argmax(gain[candidates])])
    return Subevent("STOP_BURST", float(times[index]), float(times[index]),
                    "AUTOMATIC_VSLP_DETECTOR", float(gain[index]))


def detect_noise_interval(audio: np.ndarray, rate: int, phone_start: float,
                          phone_end: float) -> Subevent:
    """Longest contiguous high-band run above 20% of local peak power."""
    times, power = _frames(audio, rate, phone_start, phone_end)
    peak = float(np.max(power)) if len(power) else 0.
    if peak < MIN_DIGITAL_POWER:
        raise ValueError("noise_interval_not_detected")
    active = power >= max(NOISE_PEAK_FRACTION * peak, MIN_DIGITAL_POWER)
    edges = np.diff(np.r_[False, active, False].astype(int))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    if not runs:
        raise ValueError("noise_interval_not_detected")
    left, right = max(runs, key=lambda item: (item[1] - item[0], -item[0]))
    start = max(phone_start, float(times[left] - FRAME_SEC / 2))
    end = min(phone_end, float(times[right - 1] + FRAME_SEC / 2))
    if end - start < MIN_NOISE_DURATION_SEC:
        raise ValueError("noise_interval_too_short")
    return Subevent("APERIODIC_NOISE_INTERVAL", start, end,
                    "AUTOMATIC_VSLP_DETECTOR", peak / max(float(np.median(power)),
                                                            MIN_DIGITAL_POWER))


def detect_noise_rise(audio: np.ndarray, rate: int, domain_start: float,
                      domain_end: float) -> Subevent:
    """First persistent 10% then 90% crossing of a smoothed local envelope."""
    times, power = _frames(audio, rate, domain_start, domain_end)
    if len(power) < 2 * RISE_PERSIST_FRAMES + RISE_SMOOTH_FRAMES:
        raise ValueError("rise_not_detected")
    smoothed = np.convolve(power, np.ones(RISE_SMOOTH_FRAMES) / RISE_SMOOTH_FRAMES,
                          mode="same")
    baseline = float(np.median(smoothed[:max(3, len(smoothed) // 5)]))
    stable_peak = float(np.percentile(smoothed, 95))
    if stable_peak - baseline < MIN_DIGITAL_POWER:
        raise ValueError("rise_not_detected")
    normalized = (smoothed - baseline) / (stable_peak - baseline)

    def crossing(level: float, after: int = 0) -> int | None:
        for index in range(after, len(normalized) - RISE_PERSIST_FRAMES + 1):
            if np.all(normalized[index:index + RISE_PERSIST_FRAMES] >= level):
                return index
        return None

    start = crossing(.10)
    end = crossing(.90, start) if start is not None else None
    if start is None or end is None:
        raise ValueError("rise_not_detected")
    return Subevent("NOISE_RISE_INTERVAL", float(times[start]), float(times[end]),
                    "AUTOMATIC_VSLP_DETECTOR",
                    stable_peak / max(baseline, MIN_DIGITAL_POWER))
