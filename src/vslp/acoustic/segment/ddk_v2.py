"""DDK syllable-nucleus segmentation (v2).

Each /pa/, /ta/ or /ka/ syllable is one vowel energy peak between two stop
closures. The method:

1. Band-limits the signal to 80-4000 Hz, the vowel and stop-release band. This
   rejects rumble and high fricative hiss, and works at any sample rate.
2. Computes a short-time RMS contour in dB (20 ms window, 5 ms hop) and smooths
   it lightly.
3. Keeps the peaks that rise at least ``min_prominence_db`` above the deeper of
   their neighbouring valleys and above a relative floor (P95 - 35 dB, and at
   least 6 dB over the noise floor). Peaks must be at least ``min_ioi_ms``
   apart, which allows up to about 14 syllables/s.
4. Bounds each syllable where the contour returns to within 3 dB of the
   closure valley on either side, or falls 20 dB below the peak when the
   closure is silent. Syllables therefore never overlap, and the closure
   silence is not counted as part of the syllable.
   Two peaks much closer than the typical spacing with a shallow dip (< 6 dB)
   between them are one slow syllable, not two.
5. Splits the syllables into sequences wherever an inter-onset gap exceeds
   max(1 s, 3 x the median gap). The longest sequence is the analysed DDK
   attempt; the other events stay listed but are marked outside the sequence.
   Isolated sounds at either end of it (pre-sounds, clicks) are trimmed.
   Empty or silent files, irregular rhythm (a wrong task?) and probably merged
   syllables are flagged for review.
   The same break, floor, dip and minimum-syllable values are used in the
   Paper 3 DDK method (ddk_v0.5).

For /pataka/, each syllable is labelled pa, ta or ka from its release-burst
spectrum. Release bursts are distinguished by their spectral balance: /t/ has
the most high-frequency energy, /p/ the least, and /k/ falls in between. The
labels are fitted to the cyclic pa-ta-ka order, and complete pa-ta-ka triplets
are formed from them.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy import ndimage, signal

ALGORITHM_VERSION = "vslp-ddk-nuclei-2.1"


def _db(values: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(values, 1e-12))


def _contour(x: np.ndarray, sr: int, frame_ms: float, hop_ms: float):
    nyquist = sr / 2
    high = min(4000.0, nyquist * 0.95)
    sos = signal.butter(4, [80.0, high], btype="bandpass", fs=sr, output="sos")
    band = signal.sosfiltfilt(sos, x)
    frame = max(1, round(sr * frame_ms / 1000))
    hop = max(1, round(sr * hop_ms / 1000))
    power = ndimage.uniform_filter1d(np.square(band), size=frame, mode="constant")
    centers = np.arange(0, len(x), hop, dtype=int)
    contour = _db(power[centers])
    smooth = max(1, round(15 / hop_ms))
    contour = ndimage.uniform_filter1d(contour, size=smooth, mode="nearest")
    return centers, contour, band


def _burst_balance(band_source: np.ndarray, sr: int, start: int, peak: int) -> float:
    """High-to-low spectral balance (dB) of the stop release before a vowel peak."""
    # Release burst: the 30 ms that end at the detected syllable onset.
    a = max(0, start - round(0.030 * sr))
    b = min(peak, start + round(0.002 * sr))
    if b - a < round(0.008 * sr):
        return np.nan
    segment = band_source[a:b] * np.hanning(b - a)
    spectrum = np.abs(np.fft.rfft(segment, n=max(512, 1 << int(np.ceil(np.log2(b - a))))))
    freqs = np.fft.rfftfreq(max(512, 1 << int(np.ceil(np.log2(b - a)))), 1 / sr)
    low = np.sum(spectrum[(freqs >= 300) & (freqs < 1500)] ** 2)
    high = np.sum(spectrum[(freqs >= 2500) & (freqs < min(6000, sr / 2))] ** 2)
    mid = np.sum(spectrum[(freqs >= 1500) & (freqs < 2500)] ** 2)
    # One axis: /p/ (low) < /k/ (mid-compact) < /t/ (high).
    return float(_db(high + 0.5 * mid) - _db(low + 0.5 * mid))


def label_pataka(balances: list[float]) -> tuple[list[str], float]:
    """Assign pa/ta/ka in cyclic order, choosing the phase the bursts support.

    For each of the three phases, the median burst balance of the syllables
    labelled pa, ka and ta should rise in that order (p < k < t). The fit is the
    smaller of the two steps, in units of the pooled spread (MAD). Speakers
    almost always start on "pa", so another phase is chosen only when it fits
    clearly better (by 0.5). A fit under ``min_pataka_label_fit`` is flagged.
    """
    order = ["pa", "ta", "ka"]
    values = np.asarray(balances, dtype=float)
    ok = np.isfinite(values)
    if ok.sum() < 6:
        return [order[index % 3] for index in range(len(values))], np.nan

    def fit(phase: int) -> float:
        labels = np.array([order[(index + phase) % 3] for index in range(len(values))])
        groups = {label: values[ok & (labels == label)] for label in order}
        if any(len(group) == 0 for group in groups.values()):
            return -np.inf
        medians = {label: float(np.median(group)) for label, group in groups.items()}
        spread = float(np.median(np.abs(values[ok] - np.median(values[ok])))) or 1.0
        return min(medians["ka"] - medians["pa"], medians["ta"] - medians["ka"]) / spread

    scores = [fit(phase) for phase in range(3)]
    best = int(np.argmax(scores))
    phase = best if scores[best] > scores[0] + 0.5 else 0
    return [order[(index + phase) % 3] for index in range(len(values))], float(scores[phase])


def _merge_split_syllables(contour, peaks, prominences, config):
    """Merge two energy humps of one slow, long vowel into one syllable.

    Between two real syllables the contour falls into a stop closure. Between
    two humps of one vowel it dips only a little. A pair is merged when its
    dip is shallow in absolute terms (< ``split_dip_db``) and also small next
    to this speaker's typical closure depth (the upper quartile of all dips),
    and the two humps are not further apart than a typical syllable. Speakers
    whose closures are all shallow (slurred DDK) are therefore never merged.
    (Real-data check: slow PLS/ALS syllables were split 2-3 ways.)
    """
    peaks, prominences = list(peaks), list(prominences)
    changed = True
    while changed and len(peaks) > 3:
        changed = False
        gaps = np.diff(peaks)
        dips = np.array([min(contour[a], contour[b]) - float(np.min(contour[a:b + 1]))
                         for a, b in zip(peaks[:-1], peaks[1:])])
        closure = float(np.percentile(dips, 75))
        typical = float(np.percentile(gaps, 75))
        candidates = [index for index in range(len(gaps))
                      if dips[index] < config.split_dip_db
                      and dips[index] < config.split_dip_fraction * closure
                      and gaps[index] < typical]
        if candidates:
            index = min(candidates, key=lambda k: dips[k])
            a, b = peaks[index], peaks[index + 1]
            drop = index if contour[a] < contour[b] else index + 1
            del peaks[drop], prominences[drop]
            changed = True
    return np.array(peaks, dtype=int), np.array(prominences, dtype=float)


def segment_ddk_v2(x: np.ndarray, sr: int, config) -> dict:
    if sr <= 400 or len(x) == 0:
        raise ValueError("DDK requires nonempty audio sampled above 400 Hz")
    work = np.asarray(x, dtype=np.float64).copy()
    work -= np.mean(work)
    centers, contour, band = _contour(work, sr, config.frame_ms, config.hop_ms)
    if len(x) / sr < 1.0 or float(np.max(np.abs(work))) < 1e-3:
        # Empty or digitally silent file (pilot: 0.6 s files, peak ~6e-5).
        peaks = np.array([], dtype=int)
        properties = {"prominences": np.array([])}
        silent = True
    else:
        silent = False
    times = np.minimum(centers, len(x) - 1) / sr
    noise_floor = float(np.percentile(contour, 5))
    top = float(np.percentile(contour, 95))
    floor = max(top - config.floor_below_p95_db, noise_floor + config.min_snr_db)
    distance = max(1, round(config.min_ioi_ms / config.hop_ms))
    if not silent:
        peaks, properties = signal.find_peaks(contour, height=floor, distance=distance,
                                              prominence=config.min_prominence_db)
        peaks, prominences = _merge_split_syllables(contour, peaks, properties["prominences"],
                                                    config)
        properties = {"prominences": prominences}
    flags: list[str] = ["silent_or_empty_recording"] if silent else []
    events: list[dict] = []
    for number, peak in enumerate(peaks):
        # Valleys (closure minima) toward the neighbouring peaks, or the file edges.
        lo = peaks[number - 1] if number > 0 else 0
        hi = peaks[number + 1] if number + 1 < len(peaks) else len(contour) - 1
        left_valley = lo + int(np.argmin(contour[lo:peak + 1]))
        right_valley = peak + int(np.argmin(contour[peak:hi + 1]))
        # Syllable edges: where the contour is back within 3 dB of the valley,
        # or 20 dB below the peak when the closure is silent.
        left_level = max(contour[left_valley] + 3.0, contour[peak] - 20.0)
        right_level = max(contour[right_valley] + 3.0, contour[peak] - 20.0)
        below = np.flatnonzero(contour[left_valley:peak] < left_level)
        left = left_valley + int(below[-1]) + 1 if below.size else left_valley
        below = np.flatnonzero(contour[peak:right_valley + 1] < right_level)
        right = peak + int(below[0]) if below.size else right_valley
        events.append({
            "start_sample": int(centers[left]), "end_sample": int(min(len(x), centers[right])),
            "energy_peak_sample": int(centers[peak]),
            "energy_peak_value": float(contour[peak]),
            "prominence_db": float(properties["prominences"][number]),
            "closure_before_sample": int(centers[left_valley]),
        })
    for event in events:
        if event["end_sample"] <= event["start_sample"]:
            event["end_sample"] = min(len(x), event["start_sample"] + round(0.01 * sr))
    # Sequences: split at long gaps between nuclei.
    nuclei = np.array([event["energy_peak_sample"] for event in events]) / sr
    sequences: list[list[int]] = []
    if len(events):
        gaps = np.diff(nuclei)
        median_gap = float(np.median(gaps)) if len(gaps) else np.inf
        limit = max(config.break_gap_s, config.break_gap_factor * median_gap)
        current = [0]
        for index, gap in enumerate(gaps, 1):
            if gap > limit:
                sequences.append(current)
                current = [index]
            else:
                current.append(index)
        sequences.append(current)
    main = max(sequences, key=len) if sequences else []
    # Drop isolated sounds at either end of the main sequence (pre-sounds,
    # clicks, "okay"): an edge event far from its neighbour is not a syllable.
    if len(main) >= 4:
        inner = np.diff(nuclei[main])
        typical = float(np.median(inner))
        trimmed = list(main)
        while len(trimmed) >= 4 and nuclei[trimmed[1]] - nuclei[trimmed[0]] > config.edge_gap_factor * typical:
            trimmed.pop(0)
        while len(trimmed) >= 4 and nuclei[trimmed[-1]] - nuclei[trimmed[-2]] > config.edge_gap_factor * typical:
            trimmed.pop()
        if trimmed != main:
            sequences = [seq for seq in sequences if seq is not main]
            outside = [index for index in main if index not in trimmed]
            sequences += [[index] for index in outside] + [trimmed]
            sequences.sort(key=lambda seq: seq[0])
            main = trimmed
    # Low-level noise next to the sequence (breathing, rustle, hum bumps in
    # the silence before and after) is often regular enough to survive the gap
    # rule. An edge event that is both much quieter and much less modulated
    # than the typical syllable, or very much quieter, is not a syllable.
    noise_edge: list[int] = []
    if len(main) >= 4:
        levels = np.array([events[index]["energy_peak_value"] for index in main])
        proms = np.array([events[index]["prominence_db"] for index in main])
        level_ref, prom_ref = float(np.median(levels)), float(np.median(proms))

        def weak(index: int) -> bool:
            event = events[index]
            drop = level_ref - event["energy_peak_value"]
            return (drop > config.noise_edge_level_db
                    or (drop > config.weak_edge_level_db
                        and event["prominence_db"] < config.weak_edge_prominence_fraction * prom_ref))

        trimmed = list(main)
        while len(trimmed) >= 4 and weak(trimmed[0]):
            noise_edge.append(trimmed.pop(0))
        while len(trimmed) >= 4 and weak(trimmed[-1]):
            noise_edge.append(trimmed.pop())
        if noise_edge:
            sequences = [seq for seq in sequences if seq is not main]
            sequences += [[index] for index in noise_edge] + [trimmed]
            sequences.sort(key=lambda seq: seq[0])
            main = trimmed
    accepted, rejected = [], []
    for sequence_number, members in enumerate(sequences, 1):
        for index in members:
            event = {**events[index], "sequence_index": sequence_number}
            if members is main:
                accepted.append(event)
            else:
                reason = ("low_level_noise_at_sequence_edge" if index in noise_edge
                          else "outside_main_ddk_sequence")
                rejected.append({**event, "rejection_reason": reason})
    labels: list[str] = []
    label_fit = np.nan
    if getattr(config, "target", "") == "pataka" and accepted:
        balances = [_burst_balance(band, sr, event["start_sample"], event["energy_peak_sample"])
                    for event in accepted]
        labels, label_fit = label_pataka(balances)
        for event, label, balance in zip(accepted, labels, balances):
            event.update(syllable_label=label, burst_balance_db=balance)
        if not np.isfinite(label_fit) or label_fit < config.min_pataka_label_fit:
            flags.append("pataka_labels_uncertain")
    intervals = [(event["start_sample"], event["end_sample"]) for event in accepted]
    nucleus_samples = [event["energy_peak_sample"] for event in accepted]
    cycles = np.diff(nucleus_samples) / sr
    if not accepted:
        flags += ["no_events_detected", "no_detectable_ddk_events"]
    else:
        if len(accepted) < config.min_syllables:
            flags.append("fewer_than_min_syllables")
        if np.median([event["prominence_db"] for event in accepted]) < config.low_modulation_db:
            flags.append("low_modulation")
        if noise_edge:
            flags.append("noise_removed_at_sequence_edges")
        if len(cycles) > 2 and np.max(cycles) > config.missed_syllable_factor * np.median(cycles):
            flags.append("suspected_missed_syllable")
        if len(sequences) > 1 and sum(len(s) >= config.min_syllables for s in sequences) > 1:
            flags.append("multiple_ddk_sequences")
        if len(cycles) > 3 and np.std(cycles) / np.mean(cycles) > config.irregular_cv:
            flags.append("irregular_rhythm_check_task")
        lengths = np.array([(b - a) / sr for a, b in intervals])
        if len(lengths) > 3 and np.max(lengths) > config.long_event_factor * np.median(lengths):
            flags.append("possible_merged_syllables")
    duration = (intervals[-1][1] - intervals[0][0]) / sr if intervals else np.nan
    threshold = np.full_like(contour, floor)
    return {
        "intervals_samples": intervals, "nuclei_samples": nucleus_samples,
        "raw_events": events, "merged_candidates": events,
        "final_events": accepted, "rejected_events": rejected,
        "raw_candidate_count": len(events),
        "raw_threshold_crossings_samples": [],
        "frame_times_sec": times, "energy_envelope": contour,
        "adaptive_threshold": threshold, "flags": flags,
        "automatic_status": "REVIEW" if flags else "ACCEPTED",
        "n_events": len(accepted), "n_syllables": len(accepted),
        "n_sequences": len(sequences), "syllable_labels": labels,
        "pataka_label_fit": label_fit,
        "ddk_sequence_start_sec": intervals[0][0] / sr if intervals else np.nan,
        "ddk_sequence_end_sec": intervals[-1][1] / sr if intervals else np.nan,
        "ddk_sequence_duration_sec": duration,
        "ddk_rate_hz": len(accepted) / duration if np.isfinite(duration) and duration > 0 else np.nan,
        "ctv_sec": float(np.mean(np.abs(np.diff(cycles)))) if len(cycles) > 1 else np.nan,
        "cycle_mean_sec": float(np.mean(cycles)) if len(cycles) else np.nan,
        "cycle_sd_sec": float(np.std(cycles)) if len(cycles) else np.nan,
        "minimum_peak_threshold_ratio": min((event["prominence_db"] for event in accepted),
                                            default=np.nan),
        "threshold_perturbation_event_counts": [],
        "processing": {"algorithm": ALGORITHM_VERSION, "band_hz": [80, 4000],
                       "contour": "short-time RMS dB", "noise_floor_db": noise_floor,
                       "floor_db": floor, "config": asdict(config)},
    }
