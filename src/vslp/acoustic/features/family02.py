"""Family 02 Praat perturbation and source-replication voice measures."""

from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call

from vslp.acoustic.features.family02_research import (
    cpp_unsmoothed, gne_frames, harmonic_windows, harmonic_statistics, pvi_replication,
)
from vslp.acoustic.features.family02_harmonic_difference import FEATURE_IDS as H1H2_IDS


ALGORITHM_VERSION = "family02-task-aware-2.0.0"
PARAMETER_SET_ID = "family02_task_aware_v2"
PRAAT_IDS = frozenset({
    "jitter_local_pct", "jitter_absolute_s", "jitter_rap_pct", "jitter_ppq5_pct",
    "jitter_ddp_pct", "shimmer_local_pct", "shimmer_local_db", "shimmer_apq3_pct",
    "shimmer_apq5_pct", "shimmer_apq11_pct", "shimmer_dda_pct", "hnr_mean_db", "dfp_pct",
})
HARMONIC_IDS = frozenset(f"h{p}_{kind}" for p in range(1, 9)
                         for kind in ("mean", "sd", "relh"))
RESEARCH_IDS = frozenset({"gne_source_replication_only", "gne_mean", "gne_sd",
                          "pvi_9_14hz_source",
                          "cpp_db"}) | HARMONIC_IDS
FAMILY02_IDS = PRAAT_IDS | RESEARCH_IDS | H1H2_IDS
PITCH_FLOOR_HZ = 60.0
PITCH_CEILING_HZ = 500.0
PERIOD_FLOOR_SEC = 0.8 / PITCH_CEILING_HZ
PERIOD_CEILING_SEC = 1.25 / PITCH_FLOOR_HZ
MAX_PERIOD_FACTOR = 1.3
MAX_AMPLITUDE_FACTOR = 1.6
MIN_USABLE_CYCLES = 100


def directional_perturbation_factor(periods: np.ndarray) -> float:
    """Zeros are non-reversals, including either side of a zero difference."""
    periods = np.asarray(periods, dtype=float)
    if periods.size < 4 or not np.isfinite(periods).all():
        return np.nan
    delta_sign = np.sign(np.diff(periods))
    return float(100.0 * np.sum((delta_sign[1:] * delta_sign[:-1]) < 0) /
                 (periods.size - 1))


def _dfp_contiguous_valid_periods(periods: np.ndarray) -> float:
    valid = (periods >= PERIOD_FLOOR_SEC) & (periods <= PERIOD_CEILING_SEC)
    edges = np.diff(np.r_[False, valid, False].astype(int))
    numerator = denominator = 0
    for first, last in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1),
                           strict=True):
        run = periods[first:last]
        if run.size >= 4:
            delta = np.sign(np.diff(run))
            numerator += int(np.sum(delta[1:] * delta[:-1] < 0))
            denominator += run.size - 1
    return float(100 * numerator / denominator) if denominator else np.nan


def trial_perturbation_from_runs(period_runs: list[np.ndarray],
                                 amplitude_runs: list[np.ndarray]) -> dict[str, float]:
    """Aggregate primitive differences without constructing terms across run gaps.

    The pulse and amplitude observations come from Praat's PointProcess and
    period AmplitudeTier. Numerical parity with Praat's own cross-run policy
    remains a separate reference-validation task.
    """
    valid_periods = []
    for source in period_runs:
        run = np.asarray(source, dtype=float)
        valid = np.isfinite(run) & (run >= PERIOD_FLOOR_SEC) & (
            run <= PERIOD_CEILING_SEC)
        edges = np.diff(np.r_[False, valid, False].astype(int))
        valid_periods.extend(run[first:last] for first, last in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))
    periods = np.concatenate(valid_periods) if valid_periods else np.array([])
    amplitudes = [np.asarray(run, dtype=float) for run in amplitude_runs]
    amplitudes = [run[np.isfinite(run) & (run > 0)] for run in amplitudes]
    all_amplitudes = np.concatenate(amplitudes) if amplitudes else np.array([])
    output: dict[str, float] = {}

    def terms(runs: list[np.ndarray], width: int, amplitude: bool = False) -> np.ndarray:
        found = []
        radius = width // 2
        for run in runs:
            for index in range(radius, len(run) - radius):
                neighborhood = run[index - radius:index + radius + 1]
                if np.any(np.maximum(neighborhood[1:], neighborhood[:-1]) /
                          np.minimum(neighborhood[1:], neighborhood[:-1]) >
                          (MAX_AMPLITUDE_FACTOR if amplitude else MAX_PERIOD_FACTOR)):
                    continue
                found.append(abs(run[index] - np.mean(neighborhood)))
        return np.asarray(found)

    differences = [np.abs(np.diff(run))[np.maximum(run[1:], run[:-1]) /
                   np.minimum(run[1:], run[:-1]) <= MAX_PERIOD_FACTOR]
                   for run in valid_periods if len(run) >= 2]
    local = np.concatenate(differences) if differences else np.array([])
    period_mean = float(np.mean(periods)) if periods.size else np.nan
    output["jitter_absolute_s"] = float(np.mean(local)) if local.size else np.nan
    output["jitter_local_pct"] = (100 * output["jitter_absolute_s"] / period_mean
                                  if local.size else np.nan)
    for width, name in ((3, "jitter_rap_pct"), (5, "jitter_ppq5_pct")):
        values = terms(valid_periods, width)
        output[name] = (float(100 * np.mean(values) / period_mean)
                        if values.size else np.nan)
    output["jitter_ddp_pct"] = 3 * output["jitter_rap_pct"]
    amplitude_mean = float(np.mean(all_amplitudes)) if all_amplitudes.size else np.nan
    amp_pairs = []
    db_pairs = []
    for run in amplitudes:
        if len(run) < 2:
            continue
        ratio = np.maximum(run[1:], run[:-1]) / np.minimum(run[1:], run[:-1])
        allowed = ratio <= MAX_AMPLITUDE_FACTOR
        amp_pairs.extend(np.abs(np.diff(run))[allowed])
        db_pairs.extend(np.abs(20 * np.log10(run[1:] / run[:-1]))[allowed])
    output["shimmer_local_pct"] = (float(100 * np.mean(amp_pairs) / amplitude_mean)
                                    if amp_pairs else np.nan)
    output["shimmer_local_db"] = float(np.mean(db_pairs)) if db_pairs else np.nan
    for width, name in ((3, "shimmer_apq3_pct"), (5, "shimmer_apq5_pct"),
                        (11, "shimmer_apq11_pct")):
        values = terms(amplitudes, width, amplitude=True)
        output[name] = (float(100 * np.mean(values) / amplitude_mean)
                        if values.size else np.nan)
    output["shimmer_dda_pct"] = 3 * output["shimmer_apq3_pct"]
    reversal_count = sum(int(np.sum(np.sign(np.diff(run))[1:] *
                                    np.sign(np.diff(run))[:-1] < 0))
                         for run in valid_periods if len(run) >= 4)
    valid_count = sum(len(run) - 1 for run in valid_periods if len(run) >= 4)
    output["dfp_pct"] = (100 * reversal_count / valid_count
                         if valid_count else np.nan)
    return output


def calculate_voice_quality(audio: np.ndarray, sr: int,
                            selected_features: set[str] | None = None,
                            minimum_usable_cycles: int = MIN_USABLE_CYCLES):
    """Return exact values, reasons, and pulse/HNR diagnostics on a private region."""
    selected = PRAAT_IDS if selected_features is None else set(selected_features)
    x = np.asarray(audio, dtype=np.float64)
    if sr < 16000:
        return ({feature_id: (np.nan, "native_sample_rate_below_16000_hz")
                 for feature_id in FAMILY02_IDS},
                {"n_cycles": 0, "n_pulses": 0, "hnr_valid_frames": 0})
    if x.ndim != 1 or x.size < sr // 10 or not np.isfinite(x).all():
        return ({feature_id: (np.nan, "invalid_analysis_audio") for feature_id in FAMILY02_IDS},
                {"n_cycles": 0, "n_pulses": 0, "hnr_valid_frames": 0})
    sound = parselmouth.Sound(x, sampling_frequency=float(sr))
    results = {feature_id: (np.nan, "not_requested" if feature_id not in selected else
                           "analysis_unavailable") for feature_id in FAMILY02_IDS}
    audit = {"n_cycles": 0, "n_pulses": 0, "hnr_valid_frames": 0}
    if "hnr_mean_db" in selected:
        try:
            harmonicity = sound.to_harmonicity_cc(time_step=.01, minimum_pitch=60.0,
                                                  silence_threshold=.1, periods_per_window=4.5)
            hnr = np.asarray(harmonicity.values, dtype=float).ravel()
            valid_hnr = hnr[np.isfinite(hnr) & (hnr > -200)]
            audit["hnr_frame_values"] = valid_hnr
            audit["hnr_valid_frames"] = int(valid_hnr.size)
            results["hnr_mean_db"] = ((float(np.mean(valid_hnr)), "") if valid_hnr.size >= 20
                                      else (np.nan, "insufficient_valid_hnr_frames"))
        except (RuntimeError, ValueError):
            results["hnr_mean_db"] = (np.nan, "harmonicity_failed")
    if selected & {"gne_source_replication_only", "gne_mean", "gne_sd"}:
        try:
            frames = gne_frames(x, sr)
            audit["gne_frame_values"] = frames
            value = float(np.mean(frames)) if frames.size else np.nan
            spread = float(np.std(frames, ddof=1)) if frames.size >= 2 else np.nan
            for key, candidate in (("gne_source_replication_only", value),
                                   ("gne_mean", value), ("gne_sd", spread)):
                if key in selected:
                    results[key] = ((candidate, "") if np.isfinite(candidate) else
                                    (np.nan, "gne_filter_or_envelope_failed"))
        except (ValueError, RuntimeError):
            for key in selected & {"gne_source_replication_only", "gne_mean", "gne_sd"}:
                results[key] = (np.nan, "gne_filter_or_envelope_failed")
    if "pvi_9_14hz_source" in selected:
        try:
            from vslp.acoustic.features.family01 import praat_f0_track
            track = praat_f0_track(x, sr)
            value, fraction = pvi_replication(
                np.where(track.voiced_valid, track.f0_hz, np.nan), track.time_sec)
            audit["pvi_interpolated_fraction"] = fraction
            results["pvi_9_14hz_source"] = (
                (value, "") if np.isfinite(value) else
                (np.nan, "pvi_missing_f0_policy_or_short_track"))
        except (RuntimeError, ValueError):
            results["pvi_9_14hz_source"] = (np.nan, "pvi_f0_or_filter_failed")
    if "cpp_db" in selected:
        try:
            frame_values = cpp_unsmoothed(x, sr, return_frames=True)
            frames = len(frame_values)
            value = float(np.mean(frame_values)) if frames else np.nan
            audit["cpp_valid_frames"] = frames
            audit["cpp_frame_values"] = frame_values
            results["cpp_db"] = ((value, "") if np.isfinite(value) else
                                 (np.nan, "cpp_peak_search_failed"))
        except (ValueError, RuntimeError):
            results["cpp_db"] = (np.nan, "cpp_peak_search_failed")
    if not selected & ((PRAAT_IDS - {"hnr_mean_db"}) | HARMONIC_IDS):
        return results, audit
    try:
        point = call(sound, "To PointProcess (periodic, cc)", PITCH_FLOOR_HZ, PITCH_CEILING_HZ)
        n_points = int(call(point, "Get number of points"))
        times = np.array([float(call(point, "Get time from index", i))
                          for i in range(1, n_points + 1)])
        periods = np.diff(times)
        audit["n_pulses"] = n_points
        audit["n_cycles"] = int(np.sum((periods >= PERIOD_FLOOR_SEC)
                                         & (periods <= PERIOD_CEILING_SEC)))
        audit["pulse_time_sec"] = times
        audit["period_sec"] = periods
        if selected & {key for key in PRAAT_IDS if key.startswith("shimmer_")}:
            try:
                tier = call([sound, point], "To AmplitudeTier (period)",
                            0, 0, PERIOD_FLOOR_SEC, PERIOD_CEILING_SEC,
                            MAX_PERIOD_FACTOR)
                count = int(call(tier, "Get number of points"))
                audit["amplitude_values"] = np.array([
                    float(call(tier, "Get value at index", i))
                    for i in range(1, count + 1)])
            except (RuntimeError, ValueError):
                audit["amplitude_values"] = np.array([])
        if audit["n_cycles"] < minimum_usable_cycles:
            for feature_id in selected & (PRAAT_IDS - {"hnr_mean_db"}):
                results[feature_id] = (np.nan, "insufficient_usable_cycles")
        if selected & HARMONIC_IDS:
            try:
                windows = harmonic_windows(x, sr, times)
                harmonics = harmonic_statistics(windows)
                audit["harmonic_windows"] = windows
            except (ValueError, RuntimeError):
                harmonics = {}
            for feature_id in selected & HARMONIC_IDS:
                value = harmonics.get(feature_id, np.nan)
                results[feature_id] = ((value, "") if np.isfinite(value) else
                                       (np.nan, "insufficient_harmonic_windows"))
            audit["harmonic_window_count"] = len(audit.get("harmonic_windows", []))
        if audit["n_cycles"] < minimum_usable_cycles:
            return results, audit
        args = (0.0, 0.0, PERIOD_FLOOR_SEC, PERIOD_CEILING_SEC, MAX_PERIOD_FACTOR)
        shimmer_args = (*args, MAX_AMPLITUDE_FACTOR)
        commands = {
            "jitter_local_pct": (point, "Get jitter (local)", args, 100.0),
            "jitter_absolute_s": (point, "Get jitter (local, absolute)", args, 1.0),
            "jitter_rap_pct": (point, "Get jitter (rap)", args, 100.0),
            "jitter_ppq5_pct": (point, "Get jitter (ppq5)", args, 100.0),
            "jitter_ddp_pct": (point, "Get jitter (ddp)", args, 100.0),
            "shimmer_local_pct": ([sound, point], "Get shimmer (local)", shimmer_args, 100.0),
            "shimmer_local_db": ([sound, point], "Get shimmer (local_dB)", shimmer_args, 1.0),
            "shimmer_apq3_pct": ([sound, point], "Get shimmer (apq3)", shimmer_args, 100.0),
            "shimmer_apq5_pct": ([sound, point], "Get shimmer (apq5)", shimmer_args, 100.0),
            "shimmer_apq11_pct": ([sound, point], "Get shimmer (apq11)", shimmer_args, 100.0),
        }
        for feature_id, (object_, command, parameters, scale) in commands.items():
            try:
                value = float(call(object_, command, *parameters)) * scale
                results[feature_id] = ((value, "") if np.isfinite(value)
                                       else (np.nan, "praat_undefined_measure"))
            except (RuntimeError, ValueError):
                results[feature_id] = (np.nan, "praat_measure_failed")
        apq3, apq3_reason = results["shimmer_apq3_pct"]
        results["shimmer_dda_pct"] = ((3.0 * apq3, "") if not apq3_reason else
                                      (np.nan, apq3_reason))
        dfp = _dfp_contiguous_valid_periods(periods)
        results["dfp_pct"] = ((dfp, "") if np.isfinite(dfp)
                              else (np.nan, "insufficient_usable_cycles"))
    except (RuntimeError, ValueError):
        for feature_id in selected & (PRAAT_IDS - {"hnr_mean_db"} | HARMONIC_IDS):
            results[feature_id] = (np.nan, "pointprocess_failed")
    return results, audit
