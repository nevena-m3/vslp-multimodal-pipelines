"""Task-specific energy methods. All transforms act on private in-memory copies."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage, signal


@dataclass(frozen=True)
class DDKConfig:
    frame_ms: float = 20.0
    hop_ms: float = 5.0
    envelope_lowpass_hz: float = 200.0
    local_window_ms: float = 600.0
    threshold_fraction: float = 0.35
    min_event_ms: float = 35.0
    min_separation_ms: float = 55.0


@dataclass(frozen=True)
class PhonationConfig:
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    activity_fraction: float = 0.18
    max_internal_gap_ms: float = 500.0
    min_phonation_ms: float = 500.0
    onset_guard_ms: float = 1000.0
    offset_guard_ms: float = 300.0
    stable_duration_ms: float = 2000.0


def _frame_energy(x: np.ndarray, sr: int, frame_ms: float, hop_ms: float):
    frame = max(1, round(sr * frame_ms / 1000))
    hop = max(1, round(sr * hop_ms / 1000))
    starts = np.arange(0, len(x), hop, dtype=int)
    energy = np.array([
        float(np.mean(np.square(x[s:min(s + frame, len(x))], dtype=np.float64)))
        for s in starts
    ])
    centers = (starts + np.minimum(frame, len(x) - starts) / 2) / sr
    return starts, centers, energy


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def segment_ddk(x: np.ndarray, sr: int, config: DDKConfig = DDKConfig()) -> dict:
    """Energy envelope threshold crossings, adapted from Tanchip et al. (2022).

    The published 20-ms sum-of-squares envelope and 200-Hz low-pass working
    signal are retained. A local percentile floor and peak fraction replace a
    single global amplitude threshold so quiet syllables remain measurable.
    """
    if sr <= 0 or len(x) == 0:
        raise ValueError("DDK requires non-empty audio and a positive sample rate")
    work = np.asarray(x, dtype=np.float64).copy()
    work -= np.mean(work)
    peak = float(np.max(np.abs(work)))
    if peak > 0:
        work /= peak
    if sr > 500:
        sos = signal.butter(4, min(config.envelope_lowpass_hz, 0.45 * sr),
                            btype="lowpass", fs=sr, output="sos")
        work = signal.sosfiltfilt(sos, work) if len(work) > 3 * len(sos) else signal.sosfilt(sos, work)
    starts, times, energy = _frame_energy(work, sr, config.frame_ms, config.hop_ms)
    envelope = ndimage.uniform_filter1d(energy, size=3, mode="nearest")
    width = max(3, round(config.local_window_ms / config.hop_ms))
    if width % 2 == 0:
        width += 1
    floor = ndimage.percentile_filter(envelope, percentile=20, size=width, mode="nearest")
    ceiling = ndimage.percentile_filter(envelope, percentile=90, size=width, mode="nearest")
    threshold = floor + config.threshold_fraction * (ceiling - floor)
    global_noise = float(np.percentile(envelope, 15))
    threshold = np.maximum(threshold, global_noise * 1.5)
    active = envelope > threshold
    intervals = []
    nuclei = []
    contrasts = []
    for a, b in _runs(active):
        onset = max(0, int(starts[a]))
        offset = min(len(x), int(starts[b]) if b < len(starts) else len(x))
        if (offset - onset) * 1000 / sr < config.min_event_ms:
            continue
        k = a + int(np.argmax(envelope[a:b]))
        if envelope[k] < 0.015 * float(np.max(envelope)):
            continue
        nucleus = int(starts[k])
        support = float(envelope[k] / max(threshold[k], 1e-12))
        intervals.append((onset, offset))
        nuclei.append(nucleus)
        contrasts.append(support)
    flags = []
    if not intervals:
        flags.append("no_detectable_ddk_events")
    if intervals and min(contrasts) < 1.25:
        flags.append("poor_peak_valley_separability")
    if any((right[0] - left[1]) * 1000 / sr < config.min_separation_ms
           for left, right in zip(intervals[:-1], intervals[1:])):
        flags.append("closely_spaced_event_boundaries")
    if intervals and np.count_nonzero(np.abs(envelope - threshold) < 0.05 * np.maximum(threshold, 1e-12)) > 0.2 * len(envelope):
        flags.append("ambiguous_event_boundaries")
    perturb_counts = []
    for factor in (0.9, 1.1):
        perturb_counts.append(sum(
            (starts[b] if b < len(starts) else len(x)) - starts[a]
            >= round(config.min_event_ms * sr / 1000)
            and float(np.max(envelope[a:b])) >= 0.015 * float(np.max(envelope))
            for a, b in _runs(envelope > threshold * factor)
        ))
    if intervals and max(abs(count - len(intervals)) for count in perturb_counts) > max(2, 0.3 * len(intervals)):
        flags.append("algorithmic_instability")
    cycles = np.diff(nuclei) / sr
    return {
        "intervals_samples": intervals, "nuclei_samples": nuclei,
        "frame_times_sec": times, "energy_envelope": envelope,
        "adaptive_threshold": threshold, "flags": flags,
        "automatic_status": "REVIEW" if flags else "ACCEPTED",
        "n_events": len(intervals),
        "ddk_rate_hz": float(1 / np.mean(cycles)) if len(cycles) else np.nan,
        "cycle_mean_sec": float(np.mean(cycles)) if len(cycles) else np.nan,
        "cycle_sd_sec": float(np.std(cycles)) if len(cycles) else np.nan,
        "minimum_peak_threshold_ratio": min(contrasts) if contrasts else np.nan,
        "threshold_perturbation_event_counts": perturb_counts,
        "processing": {"dc_removed": True, "peak_scale": peak,
                       "lowpass_hz": config.envelope_lowpass_hz,
                       "lowpass_order": 4, "local_threshold": "p20 + fraction*(p90-p20)",
                       "config": asdict(config)},
    }


def segment_phonation(x: np.ndarray, sr: int, config: PhonationConfig = PhonationConfig()) -> dict:
    """Detect the whole phonatory episode and a separate guarded stable interior."""
    if sr <= 0 or len(x) == 0:
        raise ValueError("Phonation requires non-empty audio and a positive sample rate")
    work = np.asarray(x, dtype=np.float64)
    starts, times, energy = _frame_energy(work, sr, config.frame_ms, config.hop_ms)
    rms = np.sqrt(np.maximum(ndimage.uniform_filter1d(energy, size=3, mode="nearest"), 0))
    floor = float(np.percentile(rms, 10))
    high = float(np.percentile(rms, 95))
    threshold = floor + config.activity_fraction * (high - floor)
    active = rms > threshold if high > floor * 1.1 else np.zeros(len(rms), dtype=bool)
    raw_runs = _runs(active)
    max_gap_frames = round(config.max_internal_gap_ms / config.hop_ms)
    episodes = []
    for a, b in raw_runs:
        if episodes and a - episodes[-1][1] <= max_gap_frames:
            episodes[-1] = (episodes[-1][0], b)
        else:
            episodes.append((a, b))
    flags = []
    if not episodes:
        flags.append("no_detectable_phonation")
        full = None
        stable = None
        breaks = []
    else:
        a, b = max(episodes, key=lambda pair: pair[1] - pair[0])
        full = (int(starts[a]), min(len(x), int(starts[b]) if b < len(starts) else len(x)))
        breaks = [(int(starts[s]), int(starts[min(e, len(starts)-1)]))
                  for s, e in _runs(~active[a:b]) for s, e in [(s+a, e+a)]
                  if e - s > 1]
        duration_ms = (full[1] - full[0]) * 1000 / sr
        if duration_ms < config.min_phonation_ms:
            flags.append("short_phonation_episode")
        left = full[0] + round(config.onset_guard_ms * sr / 1000)
        right = full[1] - round(config.offset_guard_ms * sr / 1000)
        length = round(config.stable_duration_ms * sr / 1000)
        if right - left < length:
            stable = None
            flags.append("stable_region_unavailable")
        else:
            # Choose the most active interior window without deleting internal breaks.
            hop = max(1, round(config.hop_ms * sr / 1000))
            candidates = np.arange(left, right - length + 1, hop)
            scores = [float(np.mean(energy[max(0, np.searchsorted(starts, s)):
                                           max(1, np.searchsorted(starts, s + length))]))
                      for s in candidates]
            start = int(candidates[int(np.argmax(scores))])
            stable = (start, start + length)
    return {
        "full_samples": full, "stable_samples": stable,
        "internal_breaks_samples": breaks,
        "frame_times_sec": times, "activity_rms": rms,
        "activity_threshold": np.full_like(rms, threshold),
        "flags": flags, "automatic_status": "REVIEW" if flags else "ACCEPTED",
        "processing": {"canonical_transformed": False, "threshold": "p10 + fraction*(p95-p10)",
                       "config": asdict(config)},
    }
