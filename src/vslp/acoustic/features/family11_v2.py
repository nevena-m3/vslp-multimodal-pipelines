"""Trial-level digital amplitude measurements from final reviewed native audio.

These are digital waveform quantities, not calibrated acoustic intensity.
"""

from __future__ import annotations

from math import gcd

import numpy as np
import scipy
from scipy.signal import resample_poly
from scipy.stats import kurtosis, skew

from vslp.acoustic.features.family11 import natural_visibility_density


VERSION = "family11-reviewed-amplitude-2.0.0"
PARAMETER_SET_ID = "family11_reviewed_regions_v2"
GRAPH_VERSION = "natural_visibility_strict_slope_v1"
ANALYSIS_RATE_HZ = 48_000
WINDOW_SEC = 0.050
LEGACY_IDS = frozenset({"absolute_energy_fs2", "sound_power_digital", "wave_amp_skew",
                        "wave_amp_excess_kurtosis", "visibility_graph_density_amp",
                        "amp_mean_fs", "amp_sd_fs", "amp_min_fs", "amp_max_fs"})
TASKS = {"vowel": "sustained_a", "s5": "sentence_s5", "bamboo": "bamboo_passage"}
PUBLIC_IDS = frozenset({
    "vowel_absolute_energy_fs2", "s5_absolute_energy_fs2", "bamboo_absolute_energy_fs2",
    "vowel_sound_power_digital", "s5_sound_power_digital", "bamboo_sound_power_digital",
    "vowel_wave_amp_skew", "s5_wave_amp_skew",
    "vowel_wave_amp_excess_kurtosis", "s5_wave_amp_excess_kurtosis",
    "vowel_visibility_graph_density_amp", "bamboo_visibility_graph_density_amp",
    *(f"{task}_amp_{stat}_fs" for task in ("vowel", "s5")
      for stat in ("mean", "sd", "min", "max")),
})


def _private_48k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Use the same deterministic polyphase primitive as the F12 private copy."""
    if sample_rate == ANALYSIS_RATE_HZ:
        return audio.copy()
    factor = gcd(sample_rate, ANALYSIS_RATE_HZ)
    return resample_poly(audio, ANALYSIS_RATE_HZ // factor, sample_rate // factor)


def _result(value: float = np.nan, reason: str = "") -> tuple[float, str]:
    number = float(value)
    return (number, reason if reason or np.isfinite(number) else "nonfinite_feature")


def measure_trial(waveform: np.ndarray, sample_rate: int, task_id: str, *,
                  energy_intervals: list[np.ndarray] | None = None,
                  connected_waveform: np.ndarray | None = None,
                  connected_issue: str = "") -> tuple[dict[str, tuple[float, str]], dict]:
    """Measure one trial; speech-only audio is never treated as continuous time."""
    prefix = next((name for name, task in TASKS.items() if task == task_id), "")
    ids = {feature for feature in PUBLIC_IDS if feature.startswith(prefix + "_")} if prefix else set()
    values = {feature: _result(reason="invalid_analysis_audio") for feature in ids}
    x = np.asarray(waveform, dtype=np.float64)
    audit = {"algorithm_version": VERSION, "parameter_set_id": PARAMETER_SET_ID,
             "native_sample_rate_hz": sample_rate, "analysis_sample_rate_hz": ANALYSIS_RATE_HZ,
             "moments_graph_sample_rate_hz": sample_rate,
             "resampler": "scipy.signal.resample_poly", "resampler_version": scipy.__version__,
             "n_native_samples": int(x.size), "n_analysis_samples": 0,
             "peak_abs_amplitude": np.nan, "clipping_fraction": np.nan,
             "near_clipping_fraction": np.nan, "dc_offset": np.nan,
             "qc_flags": "",
             "mean_square_power_support": np.nan,
             "visibility_graph_version": GRAPH_VERSION, "visibility_window_sec": WINDOW_SEC,
             "visibility_window_samples": 0, "visibility_complete_windows": 0,
             "visibility_discarded_samples": 0,
             "visibility_peak_abs_amplitude": np.nan,
             "visibility_clipping_fraction": np.nan,
             "visibility_near_clipping_fraction": np.nan,
             "visibility_dc_offset": np.nan}
    if sample_rate <= 0 or x.ndim != 1 or x.size == 0:
        return values, audit
    if not np.isfinite(x).all():
        return {key: _result(reason="nonfinite_waveform") for key in ids}, audit
    audit.update(peak_abs_amplitude=float(np.max(np.abs(x))),
                 clipping_fraction=float(np.mean(np.abs(x) >= .999)),
                 near_clipping_fraction=float(np.mean(np.abs(x) >= .98)),
                 dc_offset=float(np.mean(x)))
    audit["qc_flags"] = ";".join(flag for flag, present in (
        ("digital_clipping", audit["clipping_fraction"] > 0),
        ("near_clipping", audit["near_clipping_fraction"] > 0),
        ("near_zero_variance", np.std(x) <= np.finfo(float).eps)) if present)
    if prefix in {"vowel", "s5", "bamboo"}:
        # Resample each reviewed interval independently. A polyphase filter must
        # never borrow samples across an artificial speech-only concatenation.
        pieces = ([x] if energy_intervals is None else
                  [np.asarray(piece, dtype=np.float64) for piece in energy_intervals])
        if not pieces or any(piece.ndim != 1 or not np.isfinite(piece).all()
                             for piece in pieces):
            return {key: _result(reason="invalid_analysis_audio") for key in ids}, audit
        copied = [_private_48k(piece, sample_rate) for piece in pieces if piece.size]
        n_samples = sum(piece.size for piece in copied)
        duration = n_samples / ANALYSIS_RATE_HZ
        audit["n_analysis_samples"] = n_samples
        audit["n_energy_intervals"] = len(copied)
        if duration <= 0:
            reason = "zero_duration"
        else:
            energy = float(sum(np.dot(piece, piece) for piece in copied))
            reason = "" if np.isfinite(energy) else "nonfinite_feature"
            values[f"{prefix}_absolute_energy_fs2"] = _result(energy if not reason else np.nan, reason)
            values[f"{prefix}_sound_power_digital"] = _result(energy / duration if not reason else np.nan, reason)
            audit["mean_square_power_support"] = energy / n_samples
    if prefix in {"vowel", "s5"}:
        if x.size < 4 or np.std(x) <= np.finfo(float).eps:
            reason = "near_zero_variance"
            for feature in ids:
                if "wave_amp_" in feature:
                    values[feature] = _result(reason=reason)
        else:
            values[f"{prefix}_wave_amp_skew"] = _result(skew(x, bias=False))
            values[f"{prefix}_wave_amp_excess_kurtosis"] = _result(
                kurtosis(x, fisher=True, bias=False))
        values[f"{prefix}_amp_mean_fs"] = _result(np.mean(x))
        values[f"{prefix}_amp_min_fs"] = _result(np.min(x))
        values[f"{prefix}_amp_max_fs"] = _result(np.max(x))
        values[f"{prefix}_amp_sd_fs"] = (_result(np.std(x, ddof=1)) if x.size >= 2
                                             else _result(reason="insufficient_samples"))
    if prefix in {"vowel", "bamboo"}:
        key = f"{prefix}_visibility_graph_density_amp"
        series = x if connected_waveform is None else np.asarray(connected_waveform, dtype=np.float64)
        if connected_issue:
            values[key] = _result(reason=connected_issue)
        elif series.ndim != 1 or not np.isfinite(series).all():
            values[key] = _result(reason="nonfinite_waveform")
        else:
            if series.size:
                audit.update(visibility_peak_abs_amplitude=float(np.max(np.abs(series))),
                             visibility_clipping_fraction=float(np.mean(np.abs(series) >= .999)),
                             visibility_near_clipping_fraction=float(np.mean(np.abs(series) >= .98)),
                             visibility_dc_offset=float(np.mean(series)))
            width = round(WINDOW_SEC * sample_rate)
            count = series.size // width if width else 0
            audit.update(visibility_window_samples=width, visibility_complete_windows=count,
                         visibility_discarded_samples=series.size - count * width)
            if count < 2:
                values[key] = _result(reason="insufficient_local_sd_windows")
            else:
                local_sd = np.std(series[:count * width].reshape(count, width), axis=1, ddof=1)
                density = natural_visibility_density(local_sd)
                values[key] = (_result(density) if np.isfinite(density)
                               else _result(reason="nonfinite_feature"))
    return values, audit
