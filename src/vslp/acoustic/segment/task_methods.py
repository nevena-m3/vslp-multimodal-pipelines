"""Task-specific energy methods. All transforms act on private in-memory copies."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage, signal


@dataclass(frozen=True)
class DDKConfig:
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    envelope_lowpass_hz: float = 200.0
    threshold_window_ms: float = 20.0
    fir_order: int = 100
    min_event_ms: float = 20.0
    min_separation_ms: float = 10.0


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
    """Tanchip 2022 Energy method; engineering choices are in processing provenance."""
    if sr <= 400 or len(x) == 0:
        raise ValueError("DDK requires nonempty audio sampled above 400 Hz")
    if config.frame_ms <= 0 or config.hop_ms <= 0 or config.threshold_window_ms <= 0:
        raise ValueError("DDK frame, hop and threshold window must be positive")
    work = np.asarray(x, dtype=np.float64).copy()
    work -= np.mean(work)
    scale = float(np.max(np.abs(work)))
    if scale:
        work /= scale
    taps = signal.firwin(config.fir_order + 1, config.envelope_lowpass_hz,
                         fs=sr, window="hamming")
    pad = max(config.fir_order, round(config.frame_ms * sr / 1000))
    padded = np.pad(work, (pad, pad))
    filtered = signal.fftconvolve(padded, taps, mode="same")[pad:pad + len(work)]
    frame = max(1, round(config.frame_ms * sr / 1000))
    hop = max(1, round(config.hop_ms * sr / 1000))
    starts = np.arange(0, len(x), hop, dtype=int)
    complete = np.pad(filtered, (0, frame))
    energy = np.array([np.sum(np.square(complete[s:s + frame], dtype=np.float64))
                       for s in starts])
    times = np.minimum(starts + frame / 2, len(x)) / sr
    width = max(1, round(config.threshold_window_ms / config.hop_ms))
    threshold = ndimage.uniform_filter1d(energy, size=width, mode="nearest")
    baseline = float(np.percentile(energy, 50))
    background_spread = max(0.0, baseline - float(np.percentile(energy, 10)))
    support_floor = max(baseline + 6 * background_spread, float(np.max(energy)) * 1e-5)
    active = (energy > threshold) & (energy > support_floor)
    raw_runs = _runs(active)
    raw_candidates = []
    for a, b in raw_runs:
        k = a + int(np.argmax(energy[a:b]))
        raw_candidates.append({"start_sample": int(starts[a]),
                               "end_sample": int(starts[b]) if b < len(starts) else len(x),
                               "energy_peak_sample": min(len(x) - 1, int(starts[k] + frame // 2)),
                               "energy_peak_value": float(energy[k])})
    # Candidate crossings can flicker several times inside one energy island.
    # Group crossings only while the envelope remains above its estimated
    # background support; the raw crossings remain separately auditable.
    supported = energy > support_floor
    merged_runs = []
    for a, b in _runs(supported):
        if not any(left < b and right > a for left, right in raw_runs):
            continue
        if merged_runs and (starts[a] - starts[merged_runs[-1][1]] if merged_runs[-1][1] < len(starts) else 0) * 1000 / sr <= config.min_separation_ms:
            merged_runs[-1] = (merged_runs[-1][0], b)
        else:
            merged_runs.append((a, b))
    raw = []
    accepted = []
    rejected = []
    for a, b in merged_runs:
        onset = int(starts[a])
        offset = int(starts[b]) if b < len(starts) else len(x)
        offset = min(offset, len(x))
        k = a + int(np.argmax(energy[a:b]))
        nucleus = min(len(x) - 1, int(starts[k] + frame // 2))
        item = {"start_sample": onset, "end_sample": offset,
                "energy_peak_sample": nucleus, "energy_peak_value": float(energy[k]),
                "threshold_at_peak": float(threshold[k]),
                "support_ratio": float(energy[k] / max(threshold[k], 1e-12))}
        raw.append(item)
        if (offset - onset) * 1000 / sr < config.min_event_ms:
            rejected.append({**item, "rejection_reason": "below_minimum_event_duration"})
        else:
            accepted.append(item)
    intervals = [(e["start_sample"], e["end_sample"]) for e in accepted]
    nuclei = [e["energy_peak_sample"] for e in accepted]
    cycles = np.diff(nuclei) / sr
    sequence_duration = (intervals[-1][1] - intervals[0][0]) / sr if intervals else np.nan
    flags = []
    if not accepted:
        flags.append("no_events_detected")
        flags.append("no_detectable_ddk_events")
    if accepted and min(e["support_ratio"] for e in accepted) < 1.05:
        flags.append("poor_energy_separation")
    if raw and len(rejected) / len(raw) > 0.25:
        flags.append("many_events_removed_by_postprocessing")
    if len(raw) > 4 * max(1, len(accepted)):
        flags.append("extreme_candidate_fragmentation")
    return {
        "intervals_samples": intervals, "nuclei_samples": nuclei,
        "raw_events": raw_candidates, "merged_candidates": raw,
        "final_events": accepted, "rejected_events": rejected,
        "raw_candidate_count": len(raw_runs),
        "raw_threshold_crossings_samples": [int(starts[k]) for k in np.flatnonzero(np.diff(active.astype(int))) + 1],
        "frame_times_sec": times, "energy_envelope": energy,
        "adaptive_threshold": threshold, "flags": flags,
        "automatic_status": "REVIEW" if flags else "ACCEPTED",
        "n_events": len(accepted), "n_syllables": len(accepted),
        "ddk_sequence_start_sec": intervals[0][0] / sr if intervals else np.nan,
        "ddk_sequence_end_sec": intervals[-1][1] / sr if intervals else np.nan,
        "ddk_sequence_duration_sec": sequence_duration,
        "ddk_rate_hz": len(accepted) / sequence_duration if np.isfinite(sequence_duration) and sequence_duration > 0 else np.nan,
        "ctv_sec": float(np.mean(np.abs(np.diff(cycles)))) if len(cycles) > 1 else np.nan,
        "cycle_mean_sec": float(np.mean(cycles)) if len(cycles) else np.nan,
        "cycle_sd_sec": float(np.std(cycles)) if len(cycles) else np.nan,
        "minimum_peak_threshold_ratio": min((e["support_ratio"] for e in accepted), default=np.nan),
        "threshold_perturbation_event_counts": [],
        "processing": {"dc_removed": True, "max_absolute_scale": scale,
                       "scaling_deviation": "max-absolute for numerical safety",
                       "lowpass_hz": config.envelope_lowpass_hz,
                       "fir_order": config.fir_order, "fir_window": "hamming",
                       "filter": "linear_phase_centered_convolution", "zero_padding_samples_each_side": pad,
                       "threshold": "20_ms_moving_average_of_sum_squares_energy",
                       "support_floor": support_floor,
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
