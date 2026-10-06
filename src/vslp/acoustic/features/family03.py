"""Open F03 glottal-source and amplitude-tremor measurements.

These project-defined outputs do not reproduce proprietary VCS Pr scores.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal

from vslp.acoustic.features.family01_stage import _f02_voiced_islands
from vslp.acoustic.features.family03_gif import (
    GIF_ALGORITHM_VERSION, GIF_PARAMETER_SET_ID, cycles_from_estimate,
    qcp_glottal_estimate,
)


ALGORITHM_VERSION = "family03-open-review-1.0.0"
PARAMETER_SET_ID = "family03_reviewed_a_v1"
GIF_IDS = frozenset({
    "glottal_mfdr", "glottal_naq", "glottal_open_quotient",
    "glottal_closed_quotient", "glottal_closing_quotient",
    "glottal_speed_quotient", "cycle_waveform_instability",
    "cycle_amplitude_instability_pct",
})
TREMOR_IDS = frozenset({"amplitude_tremor_mdat_pct",
                         "amplitude_tremor_depth_short_v1_pct"})
BLOCKAGE_IDS = frozenset({"vibration_blockage_pct"})
FAMILY03_IDS = GIF_IDS | TREMOR_IDS | BLOCKAGE_IDS
ENVELOPE_RATE_HZ = 50
TREMOR_WINDOW_SEC = 2.0
TREMOR_HOP_SEC = .1
TREMOR_MIN_HZ = 1.
TREMOR_MAX_HZ = 20.


def mfdr_naq(flow: np.ndarray, derivative: np.ndarray,
             period_sec: float) -> tuple[float, float]:
    """Uncalibrated flow-rate proxy and dimensionless Alku-style NAQ."""
    if (len(flow) != len(derivative) or len(flow) < 3 or period_sec <= 0 or
            not np.isfinite(flow).all() or not np.isfinite(derivative).all()):
        return np.nan, np.nan
    mfdr = -float(np.min(derivative))
    amplitude = float(np.ptp(flow))
    if mfdr <= 0 or amplitude <= 0:
        return np.nan, np.nan
    return mfdr, amplitude / (period_sec * mfdr)


def phase_quotients(flow: np.ndarray) -> dict[str, float]:
    """VSLP 5%-of-cycle-AC event convention; cycle edges are GCI candidates.

    Opening is the first upward 5% crossing after the first GCI. Maximum flow
    ends opening and begins closing. The next GCI ends closing. The initial
    GCI-to-opening span is closed phase. These are source-flow landmarks and
    never raw microphone-amplitude thresholds.
    """
    invalid = {key: np.nan for key in ("glottal_open_quotient",
                                           "glottal_closed_quotient",
                                           "glottal_closing_quotient",
                                           "glottal_speed_quotient")}
    x = np.asarray(flow, dtype=float)
    if x.size < 8 or not np.isfinite(x).all() or np.ptp(x) <= 1e-12:
        return invalid
    baseline = float(np.min(x))
    threshold = baseline + .05 * float(np.ptp(x))
    peak = int(np.argmax(x))
    upward = np.flatnonzero((x[:-1] <= threshold) & (x[1:] > threshold)) + 1
    upward = upward[upward < peak]
    if not len(upward) or peak >= len(x) - 2:
        return invalid
    opening = int(upward[0])
    n = len(x)
    close_duration = n - peak
    if opening <= 0 or close_duration <= 0:
        return invalid
    return {"glottal_open_quotient": (n - opening) / n,
            "glottal_closed_quotient": opening / n,
            "glottal_closing_quotient": close_duration / n,
            "glottal_speed_quotient": (peak - opening) / close_duration}


def cycle_waveform_instability(cycles: list[dict], normalized_size: int = 128) -> float:
    """Median 1−Pearson r for adjacent valid cycles within the same span."""
    differences = _cycle_waveform_differences(cycles, normalized_size)
    return float(np.median(differences)) if differences else np.nan


def _cycle_waveform_differences(cycles: list[dict], normalized_size: int = 128) -> list[float]:
    """Return each valid adjacent-pair difference for trial-level aggregation."""
    differences = []
    for first, second in zip(cycles[:-1], cycles[1:], strict=True):
        if second["cycle_id"] != first["cycle_id"] + 1:
            continue
        shapes = []
        for cycle in (first, second):
            wave = np.asarray(cycle["flow"], dtype=float)
            shape = np.interp(np.linspace(0, 1, normalized_size),
                              np.linspace(0, 1, len(wave)), wave)
            shape -= shape.mean()
            scale = np.linalg.norm(shape)
            if scale <= 1e-12:
                break
            shapes.append(shape / scale)
        if len(shapes) == 2:
            differences.append(1. - float(np.dot(*shapes)))
    return differences


def cycle_amplitude_instability_pct(cycles: list[dict]) -> float:
    amplitudes = np.asarray([np.ptp(cycle["flow"]) for cycle in cycles], dtype=float)
    amplitudes = amplitudes[np.isfinite(amplitudes) & (amplitudes > 0)]
    if len(amplitudes) < 3:
        return np.nan
    middle = float(np.median(amplitudes))
    return float(100 * 1.4826 * np.median(np.abs(amplitudes - middle)) / middle)


def _envelope_50hz(audio: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    x = np.asarray(audio, dtype=float)
    if x.size < sample_rate_hz * TREMOR_WINDOW_SEC or not np.isfinite(x).all():
        raise ValueError("insufficient_tremor_duration")
    envelope = np.abs(signal.hilbert(x))
    ratio = Fraction(ENVELOPE_RATE_HZ, sample_rate_hz)
    contour = signal.resample_poly(envelope, ratio.numerator, ratio.denominator)
    if not np.isfinite(contour).all() or np.median(contour) <= 1e-10:
        raise ValueError("unstable_amplitude_envelope")
    return contour


def _modulation_window(contour: np.ndarray) -> tuple[float, float]:
    """Return strongest normalized 1–20-Hz modulation and its frequency."""
    candidates = _modulation_candidates(contour)
    return candidates[0] if candidates else (np.nan, np.nan)


def _modulation_candidates(contour: np.ndarray) -> list[tuple[float, float]]:
    """Top five Gaussian-windowed spectral peaks in the 1–20-Hz band."""
    baseline = float(np.mean(contour))
    if baseline <= 1e-10:
        return []
    detrended = signal.detrend(contour / baseline - 1., type="linear")
    weighting = signal.windows.gaussian(len(contour), std=len(contour) / 6,
                                         sym=False)
    spectrum = np.abs(np.fft.rfft(detrended * weighting, n=4096))
    frequencies = np.fft.rfftfreq(4096, 1 / ENVELOPE_RATE_HZ)
    band = np.flatnonzero((frequencies >= TREMOR_MIN_HZ) &
                          (frequencies <= TREMOR_MAX_HZ))
    if not band.size:
        return []
    local, _ = signal.find_peaks(spectrum[band])
    peaks = band[local] if local.size else band[np.array([np.argmax(spectrum[band])])]
    peaks = sorted(peaks, key=lambda index: spectrum[index], reverse=True)[:5]
    return [(float(100 * 2 * spectrum[index] / np.sum(weighting)),
             float(frequencies[index])) for index in peaks]


def _dominant_tremor_track(windows: list[list[tuple[float, float]]],
                           duration_sec: float) -> tuple[float, float, float] | None:
    """Link successive 1-Hz-close candidates; return source-lineage track median."""
    tracks: list[list[tuple[int, float, float]]] = []
    for window_index, candidates in enumerate(windows):
        occupied = set()
        for depth, frequency in candidates:
            matches = [(abs(track[-1][2] - frequency), track_index)
                       for track_index, track in enumerate(tracks)
                       if track_index not in occupied and
                       track[-1][0] == window_index - 1 and
                       abs(track[-1][2] - frequency) <= 1.]
            if matches:
                _, chosen = min(matches)
                tracks[chosen].append((window_index, depth, frequency))
                occupied.add(chosen)
            else:
                tracks.append([(window_index, depth, frequency)])
                occupied.add(len(tracks) - 1)
    eligible = [track for track in tracks if len(track) >= 2 and
                TREMOR_WINDOW_SEC + (len(track) - 1) * TREMOR_HOP_SEC >=
                min(3., .5 * duration_sec)]
    if not eligible:
        return None
    best = max(eligible, key=lambda track: sum(item[1] for item in track))
    return (float(np.median([item[1] for item in best])),
            float(np.median([item[2] for item in best])),
            TREMOR_WINDOW_SEC + (len(best) - 1) * TREMOR_HOP_SEC)


def amplitude_tremor_trial(audio: np.ndarray, sample_rate_hz: int) -> tuple[dict, dict]:
    """Separate VSLP 2-s analogue from source-track MDAT (never impute it)."""
    failures = {feature: (np.nan, "insufficient_tremor_duration") for feature in TREMOR_IDS}
    audit = {"modulation_sample_rate_hz": ENVELOPE_RATE_HZ,
             "n_windows": 0, "dominant_tremor_frequency_hz": np.nan,
             "dominant_track_duration_sec": 0., "valid_track_fraction": 0.,
             "mode": "vslp_short_duration", "source_equivalent": False}
    try:
        contour = _envelope_50hz(audio, sample_rate_hz)
    except ValueError as exc:
        return {key: (np.nan, str(exc)) for key in TREMOR_IDS}, audit
    window = round(TREMOR_WINDOW_SEC * ENVELOPE_RATE_HZ)
    hop = round(TREMOR_HOP_SEC * ENVELOPE_RATE_HZ)
    candidates = [_modulation_candidates(contour[left:left + window])
                  for left in range(0, len(contour) - window + 1, hop)]
    valid = [entry for window_candidates in candidates for entry in window_candidates]
    audit["n_windows"] = len(candidates)
    if not valid:
        return {key: (np.nan, "unstable_amplitude_envelope") for key in TREMOR_IDS}, audit
    # The short analogue remains a distinct output even when a longer source
    # track is available. The normal VSLP <=2-s stage has only one window.
    depth, frequency = max(valid, key=lambda pair: pair[0])
    audit["dominant_tremor_frequency_hz"] = frequency
    audit["amplitude_modulation_depth_pct"] = depth
    failures["amplitude_tremor_depth_short_v1_pct"] = (depth, "")
    track = _dominant_tremor_track(candidates, len(audio) / sample_rate_hz)
    if track and len(audio) / sample_rate_hz > TREMOR_WINDOW_SEC:
        track_depth, track_frequency, track_duration = track
        failures["amplitude_tremor_mdat_pct"] = (track_depth, "")
        audit.update(dominant_tremor_frequency_hz=track_frequency,
                     dominant_track_duration_sec=track_duration,
                     valid_track_fraction=track_duration / (len(audio) / sample_rate_hz),
                     mode="source_track_lineage", source_equivalent=False)
    else:
        failures["amplitude_tremor_mdat_pct"] = (
            np.nan, "source_track_duration_unavailable")
    return failures, audit


def vibration_blockage(track, audio: np.ndarray, sample_rate_hz: int,
                        pulse_times_sec: np.ndarray) -> tuple[float, str, dict]:
    """Operational interior 3-cycle loss rule, excluding acquisition dropout."""
    valid = np.asarray(track.voiced_valid, dtype=bool)
    times = np.asarray(track.time_sec, dtype=float)
    if len(times) < 10 or len(valid) != len(times):
        return np.nan, "insufficient_f0_frames", {}
    good_hz = np.asarray(track.f0_hz, dtype=float)[valid]
    if good_hz.size < 20:
        return np.nan, "insufficient_valid_f0_frames", {}
    cycle_sec = 1 / float(np.median(good_hz))
    min_gap = 3 * cycle_sec
    inner = np.flatnonzero(~valid)
    if not len(inner):
        return 0., "", {"blockage_count": 0, "acquisition_dropout_count": 0}
    runs = np.split(inner, np.flatnonzero(np.diff(inner) != 1) + 1)
    blocked = 0.
    count = dropouts = 0
    frame_step = float(np.median(np.diff(times)))
    pulses = np.asarray(pulse_times_sec, dtype=float)
    reference_rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=float)**2)))
    for run in runs:
        if run[0] == 0 or run[-1] == len(valid) - 1:
            continue  # onset and offset are not blockage.
        left, right = times[run[0]] - frame_step / 2, times[run[-1]] + frame_step / 2
        duration = right - left
        if duration < min_gap:
            continue
        # A valid pulse inside the interval defeats the missing-cycle premise.
        if np.any((pulses >= left) & (pulses < right)):
            continue
        chunk = audio[max(0, round((left - track.analysis_start_sec) * sample_rate_hz)):
                      min(len(audio), round((right - track.analysis_start_sec) * sample_rate_hz))]
        if (not len(chunk) or np.sqrt(np.mean(chunk**2)) < .05 * reference_rms or
                np.max(np.abs(chunk)) >= .999):
            dropouts += 1
            continue
        blocked += duration
        count += 1
    if dropouts:
        return np.nan, "acquisition_dropout_requires_review", {
            "blockage_count": count, "acquisition_dropout_count": dropouts}
    eligible = track.analysis_end_sec - track.analysis_start_sec
    return 100 * blocked / eligible, "", {"blockage_count": count,
                                            "acquisition_dropout_count": 0}


def glottal_trial(parts: list, audio: np.ndarray, sample_rate_hz: int,
                  selected: set[str]) -> tuple[dict, list[dict], dict]:
    """Process each eligible contiguous span independently; never join exclusions."""
    values: dict[str, list[float]] = {key: [] for key in selected}
    cycle_rows = []
    n_candidate = n_valid = 0
    flags = set()
    trial_cycles: list[dict] = []
    pulse_times: list[float] = []
    valid_cycle_intervals: list[tuple[float, float]] = []
    for span_index, track in enumerate(parts, 1):
        for run_index, (run_start, run_end) in enumerate(
                _f02_voiced_islands(track, sample_rate_hz), 1):
            left = round(run_start * sample_rate_hz)
            right = round(run_end * sample_rate_hz)
            segment = audio[left:right]
            try:
                estimate = qcp_glottal_estimate(segment, sample_rate_hz, run_start)
                cycles, rejected = cycles_from_estimate(estimate)
            except (ValueError, RuntimeError) as exc:
                flags.add(str(exc))
                continue
            n_candidate += len(cycles) + len(rejected)
            n_valid += len(cycles)
            pulse_times.extend(run_start + estimate.pulse_samples / estimate.sample_rate_hz)
            valid_cycle_intervals.extend((cycle["start_sec"], cycle["end_sec"])
                                         for cycle in cycles)
            for rejection in rejected:
                cycle_rows.append({"span_index": span_index, "run_index": run_index,
                                   **rejection, "qc_status": "rejected"})
            for cycle in cycles:
                mfdr, naq = mfdr_naq(cycle["flow"], cycle["derivative"],
                                      cycle["period_sec"])
                phase = phase_quotients(cycle["flow"])
                row = {"span_index": span_index, "run_index": run_index,
                       **{key: val for key, val in cycle.items()
                          if key not in {"flow", "derivative"}},
                       "glottal_mfdr": mfdr, "glottal_naq": naq,
                       **phase, "flow_amplitude": float(np.ptp(cycle["flow"])),
                       "qc_status": "valid"}
                cycle_rows.append(row)
                for key in selected & {"glottal_mfdr", "glottal_naq",
                                       "glottal_open_quotient", "glottal_closed_quotient",
                                       "glottal_closing_quotient", "glottal_speed_quotient"}:
                    if np.isfinite(row[key]):
                        values[key].append(float(row[key]))
            trial_cycles.extend(cycles)
            if "cycle_waveform_instability" in selected:
                values["cycle_waveform_instability"].extend(
                    _cycle_waveform_differences(cycles))
    if "cycle_amplitude_instability_pct" in selected:
        value = cycle_amplitude_instability_pct(trial_cycles)
        if np.isfinite(value):
            values["cycle_amplitude_instability_pct"].append(value)
    result = {key: ((float(np.median(measures)), "") if measures else
                    (np.nan, "no_valid_glottal_cycles")) for key, measures in values.items()}
    audit = {"gif_method": "qcp", "gif_algorithm_version": GIF_ALGORITHM_VERSION,
             "gif_parameter_set_id": GIF_PARAMETER_SET_ID,
             "internal_sample_rate_hz": 8000, "n_candidate_cycles": n_candidate,
             "n_valid_cycles": n_valid,
             "rejected_cycle_count": n_candidate - n_valid,
             "valid_cycle_fraction": n_valid / n_candidate if n_candidate else 0.,
             "pulse_times_sec": pulse_times,
             "valid_cycle_intervals": valid_cycle_intervals,
             "qc_flags": sorted(flags), "cycle_event_definition":
             "GCI-bounded; opening upward 5%-AC crossing; peak begins closing"}
    return result, cycle_rows, audit
