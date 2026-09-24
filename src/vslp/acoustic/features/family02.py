"""Family 02 Praat cycle perturbation and cross-correlation harmonicity."""

from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call


ALGORITHM_VERSION = "family02-praat-1.0.0"
PARAMETER_SET_ID = "family02_stable_vowel_praat_v1"
FAMILY02_IDS = frozenset({
    "jitter_local_pct", "jitter_absolute_s", "jitter_rap_pct", "jitter_ppq5_pct",
    "jitter_ddp_pct", "shimmer_local_pct", "shimmer_local_db", "shimmer_apq3_pct",
    "shimmer_apq5_pct", "shimmer_apq11_pct", "shimmer_dda_pct", "hnr_mean_db", "dfp_pct",
})
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
                 (periods.size - 2))


def calculate_voice_quality(audio: np.ndarray, sr: int):
    """Return exact values, reasons, and pulse/HNR diagnostics on a private region."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or x.size < sr // 10 or not np.isfinite(x).all():
        return ({feature_id: (np.nan, "invalid_analysis_audio") for feature_id in FAMILY02_IDS},
                {"n_cycles": 0, "n_pulses": 0, "hnr_valid_frames": 0})
    sound = parselmouth.Sound(x, sampling_frequency=float(sr))
    results = {feature_id: (np.nan, "pointprocess_unavailable") for feature_id in FAMILY02_IDS}
    audit = {"n_cycles": 0, "n_pulses": 0, "hnr_valid_frames": 0}
    harmonicity = sound.to_harmonicity_cc(time_step=.01, minimum_pitch=60.0,
                                          silence_threshold=.1, periods_per_window=4.5)
    hnr = np.asarray(harmonicity.values, dtype=float).ravel()
    valid_hnr = hnr[np.isfinite(hnr) & (hnr > -200)]
    audit["hnr_valid_frames"] = int(valid_hnr.size)
    results["hnr_mean_db"] = ((float(np.mean(valid_hnr)), "") if valid_hnr.size >= 20
                              else (np.nan, "insufficient_valid_hnr_frames"))
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
        if audit["n_cycles"] < MIN_USABLE_CYCLES:
            for feature_id in FAMILY02_IDS - {"hnr_mean_db"}:
                results[feature_id] = (np.nan, "insufficient_usable_cycles")
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
        valid_periods = periods[(periods >= PERIOD_FLOOR_SEC) & (periods <= PERIOD_CEILING_SEC)]
        dfp = directional_perturbation_factor(valid_periods)
        results["dfp_pct"] = ((dfp, "") if np.isfinite(dfp)
                              else (np.nan, "insufficient_usable_cycles"))
    except (RuntimeError, ValueError):
        for feature_id in FAMILY02_IDS - {"hnr_mean_db"}:
            results[feature_id] = (np.nan, "pointprocess_failed")
    return results, audit
