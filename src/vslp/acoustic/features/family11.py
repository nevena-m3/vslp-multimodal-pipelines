"""Family 11 digital-amplitude features on an immutable native-rate region."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

ALGORITHM_VERSION = "family11-digital-amplitude-1.0.0"
PARAMETER_SET_ID = "family11_amplitude_native_v1"
FAMILY11_IDS = frozenset({
    "absolute_energy_fs2", "sound_power_digital", "wave_amp_skew",
    "wave_amp_excess_kurtosis", "amp_mean_fs", "amp_sd_fs", "amp_min_fs",
    "amp_max_fs", "visibility_graph_density_amp",
})


def natural_visibility_density(series: np.ndarray) -> float:
    """Natural visibility graph density using strict line-of-sight slopes."""
    values = np.asarray(series, dtype=np.float64)
    count = values.size
    if count < 2 or not np.isfinite(values).all():
        return np.nan
    edges = 0
    for left in range(count - 1):
        max_slope = -np.inf
        for right in range(left + 1, count):
            slope = (values[right] - values[left]) / (right - left)
            if slope > max_slope:
                edges += 1
                max_slope = slope
    return float(2 * edges / (count * (count - 1)))
DIRECT_AMPLITUDE_IDS = frozenset({
    "absolute_energy_fs2", "sound_power_digital", "amp_mean_fs", "amp_sd_fs",
    "amp_min_fs", "amp_max_fs",
})


def calculate_amplitude_features(audio: np.ndarray, sample_rate: int, *,
                                 amplitude_normalized: bool = False):
    """Return exact values/reasons plus compact amplitude audit metadata."""
    x = np.asarray(audio, dtype=np.float64)
    result = {feature_id: (np.nan, "invalid_analysis_audio") for feature_id in FAMILY11_IDS}
    audit = {
        "native_sample_rate_hz": int(sample_rate), "n_samples": int(x.size),
        "duration_sec": float(x.size / sample_rate) if sample_rate > 0 else np.nan,
        "amplitude_normalization_applied": bool(amplitude_normalized),
        "clipping_fraction": np.nan,
    }
    if sample_rate <= 0 or x.ndim != 1 or x.size < 4 or not np.isfinite(x).all():
        return result, audit
    audit["clipping_fraction"] = float(np.mean(np.abs(x) >= 0.999))
    if not amplitude_normalized:
        energy = float(np.sum(np.square(x)))
        duration = x.size / sample_rate
        result.update({
            "absolute_energy_fs2": (energy, ""),
            "sound_power_digital": (energy / duration, ""),
            "amp_mean_fs": (float(np.mean(x)), ""),
            "amp_sd_fs": (float(np.std(x, ddof=1)), ""),
            "amp_min_fs": (float(np.min(x)), ""),
            "amp_max_fs": (float(np.max(x)), ""),
        })
    else:
        for feature_id in DIRECT_AMPLITUDE_IDS:
            result[feature_id] = (np.nan, "incompatible_amplitude_normalization")
    if np.std(x) > np.finfo(float).eps:
        result["wave_amp_skew"] = (float(skew(x, bias=False)), "")
        result["wave_amp_excess_kurtosis"] = (
            float(kurtosis(x, fisher=True, bias=False)), "")
    else:
        result["wave_amp_skew"] = (np.nan, "zero_amplitude_variance")
        result["wave_amp_excess_kurtosis"] = (np.nan, "zero_amplitude_variance")
    frame_length = max(2, round(0.050 * sample_rate))
    count = x.size // frame_length
    if count >= 2:
        local_sd = np.std(x[:count * frame_length].reshape(count, frame_length),
                          axis=1, ddof=1)
        result["visibility_graph_density_amp"] = (natural_visibility_density(local_sd), "")
    else:
        result["visibility_graph_density_amp"] = (np.nan, "insufficient_local_sd_windows")
    return result, audit
