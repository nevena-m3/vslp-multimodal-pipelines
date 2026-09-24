"""Family 06 source-defined phonetic contrasts over approved acoustic sub-events."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ALGORITHM_VERSION = "family06-aligned-subevents-1.0.0"
PARAMETER_SET_ID = "family06_native_subevents_v1"
IMPLEMENTED_IDS = frozenset({"m1_t_minus_k_hz", "wideband_noise_energy_0_10khz",
                             "stop_burst_spectral_tilt_db_khz"})
UNAVAILABLE_IDS = frozenset({"blocked_aural_analytics_ap", "band_noise_contrast_variant1",
                              "band_noise_contrast_variant2"})
TEMPLATES = ("noise_duration_<target>_s", "noise_rise_time_<definition>_s",
             "normalized_duration_contrast_<A>_<B>")
M1_WINDOW_SEC = 0.020
TILT_WINDOW_SEC = 0.010
FFT_SIZE = 2048
TILT_BAND_HZ = (1500.0, 5000.0)
WIDEBAND_HZ = (0.0, 10000.0)
AUDIT_COLUMNS = ("recording_id", "file_name", "task_id", "alignment_run_id", "target_id",
                 "pair_id", "role", "word_index", "phone_index", "phone", "phone_start_sec",
                 "phone_end_sec", "measurement_start_sec", "measurement_end_sec",
                 "measurement_boundary_source", "contrast_partner", "feature_id",
                 "token_value", "unit", "validity", "failure_reason",
                 "parameter_set_id", "algorithm_version", "annotation_source", "reviewer")


def load_target_manifest(path: str | Path | None, task_id: str) -> tuple[list[dict], str]:
    """Read explicit task/word/phone/index targets; do not discover targets from audio."""
    if not path or not Path(path).is_file():
        return [], "missing_target_definition"
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("manifest_version") != "1" or data.get("task_id") != task_id:
            return [], "target_manifest_task_mismatch"
        targets = data.get("targets", [])
        if not isinstance(targets, list) or not targets:
            return [], "missing_target_definition"
        seen = set()
        for target in targets:
            required = ("target_id", "feature_id", "word", "phone", "word_index",
                        "phone_index", "acoustic_region_type", "pairing_rule", "aggregation")
            if any(target.get(field) in (None, "") for field in required):
                return [], "invalid_target_definition"
            if target["feature_id"] not in IMPLEMENTED_IDS or target["target_id"] in seen:
                return [], "invalid_target_definition"
            if int(target["word_index"]) < 1 or int(target["phone_index"]) < 1:
                return [], "invalid_target_definition"
            if target["feature_id"] == "m1_t_minus_k_hz":
                if (target.get("role") not in {"t", "k"} or not target.get("pair_id")
                        or str(target["phone"]).upper() != target["role"].upper()
                        or target["acoustic_region_type"] != "post_burst_20ms"
                        or target["pairing_rule"] != "matched_pair_id"
                        or target["aggregation"] != "mean_target_moments_then_difference"):
                    return [], "invalid_target_definition"
            elif target["feature_id"] == "wideband_noise_energy_0_10khz":
                if (target["acoustic_region_type"] != "validated_noise_interval"
                        or target["aggregation"] != "mean_valid_tokens"):
                    return [], "invalid_target_definition"
            elif (target["acoustic_region_type"] != "post_burst_10ms"
                  or target["aggregation"] != "mean_valid_tokens"):
                return [], "invalid_target_definition"
            seen.add(target["target_id"])
        for pair_id in {item["pair_id"] for item in targets
                        if item["feature_id"] == "m1_t_minus_k_hz"}:
            pair = [item for item in targets if item.get("pair_id") == pair_id]
            if len(pair) != 2 or {item["role"] for item in pair} != {"t", "k"}:
                return [], "invalid_target_definition"
        return targets, ""
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return [], "invalid_target_definition"


def load_validated_subevents(path: str | Path | None) -> tuple[pd.DataFrame, str]:
    """Accept approved original-clock annotations, never phone boundaries as bursts."""
    if not path or not Path(path).is_file():
        return pd.DataFrame(), "missing_acoustic_sub_event"
    try:
        table = pd.read_csv(path, keep_default_na=False)
        required = {"recording_id", "target_id", "word_index", "phone_index",
                    "annotation_source", "reviewer", "approved", "time_axis"}
        if not required <= set(table):
            return pd.DataFrame(), "invalid_acoustic_sub_event_schema"
        if not table.approved.astype(str).str.lower().isin({"true", "1"}).all():
            return pd.DataFrame(), "unapproved_acoustic_sub_event"
        if table.annotation_source.astype(str).str.strip().eq("").any():
            return pd.DataFrame(), "missing_acoustic_sub_event_provenance"
        if table.reviewer.astype(str).str.strip().eq("").any():
            return pd.DataFrame(), "missing_acoustic_sub_event_provenance"
        if not table.time_axis.eq("original_recording_seconds").all():
            return pd.DataFrame(), "invalid_acoustic_sub_event_time_axis"
        if table.duplicated(["recording_id", "target_id"]).any():
            return pd.DataFrame(), "duplicate_acoustic_sub_event"
        return table, ""
    except (OSError, ValueError):
        return pd.DataFrame(), "invalid_acoustic_sub_event_schema"


def _window(audio: np.ndarray, sample_rate: int, start_sec: float,
            duration_sec: float) -> np.ndarray:
    start = round(start_sec * sample_rate)
    count = round(duration_sec * sample_rate)
    if start < 0 or count < 2 or start + count > len(audio):
        raise ValueError("sub_event_outside_canonical_audio")
    return np.asarray(audio[start:start + count], dtype=float)


def first_spectral_moment(audio: np.ndarray, sample_rate: int,
                          burst_start_sec: float) -> float:
    """20-ms post-burst Hann power-spectrum centroid on original-clock audio."""
    frame = _window(audio, sample_rate, burst_start_sec, M1_WINDOW_SEC)
    n_fft = max(FFT_SIZE, 1 << (len(frame) - 1).bit_length())
    power = np.abs(np.fft.rfft(frame * np.hanning(len(frame)), n=n_fft)) ** 2
    total = float(power.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("zero_burst_spectral_power")
    frequency = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    return float(np.dot(frequency, power) / total)


def matched_m1_contrast(token_rows: pd.DataFrame) -> tuple[float, str]:
    """Use approved complete pairs, then difference of target means."""
    if token_rows.empty or not {"pair_id", "role", "token_value"} <= set(token_rows):
        return np.nan, "missing_matched_t_k_targets"
    complete = []
    for _pair_id, pair in token_rows.groupby("pair_id"):
        if set(pair.role) == {"t", "k"} and len(pair) == 2:
            complete.append(pair)
    if not complete:
        return np.nan, "missing_matched_t_k_targets"
    valid = pd.concat(complete)
    if not np.isfinite(pd.to_numeric(valid.token_value, errors="coerce")).all():
        return np.nan, "invalid_burst_spectral_moment"
    means = valid.groupby("role").token_value.mean()
    return float(means["t"] - means["k"]), ""


def wideband_noise_energy(audio: np.ndarray, sample_rate: int,
                          start_sec: float, end_sec: float) -> float:
    """Digital 0–10 kHz energy; source Nyquist must exceed 10 kHz."""
    if sample_rate / 2 <= WIDEBAND_HZ[1]:
        raise ValueError("insufficient_bandwidth")
    if not start_sec < end_sec:
        raise ValueError("invalid_noise_interval")
    frame = _window(audio, sample_rate, start_sec, end_sec - start_sec)
    window = np.hanning(len(frame))
    mean_square = float(np.mean(window ** 2))
    if mean_square <= 0:
        raise ValueError("noise_interval_too_short")
    n_fft = max(FFT_SIZE, 1 << (len(frame) - 1).bit_length())
    power = np.abs(np.fft.rfft(frame * window, n=n_fft)) ** 2
    if len(power) > 2:
        power[1:-1] *= 2
    frequency = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    return float(power[(frequency >= 0) & (frequency <= 10000)].sum() /
                 (n_fft * mean_square))


def stop_burst_tilt(audio: np.ndarray, sample_rate: int,
                    burst_start_sec: float) -> float:
    """10-ms Hann log-amplitude OLS slope over 1.5–5 kHz, dB/kHz."""
    if sample_rate / 2 <= TILT_BAND_HZ[1]:
        raise ValueError("insufficient_bandwidth")
    frame = _window(audio, sample_rate, burst_start_sec, TILT_WINDOW_SEC)
    n_fft = max(FFT_SIZE, 1 << (len(frame) - 1).bit_length())
    magnitude = np.abs(np.fft.rfft(frame * np.hanning(len(frame)), n=n_fft))
    if magnitude.max() <= 0:
        raise ValueError("zero_burst_spectral_power")
    frequency = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    band = (frequency >= 1500) & (frequency <= 5000)
    floor = float(magnitude.max()) * 1e-12
    level = 20 * np.log10(np.maximum(magnitude[band], floor))
    return float(np.polyfit(frequency[band] / 1000, level, 1)[0])


def noise_duration(start_sec: float, end_sec: float) -> float:
    if not np.isfinite([start_sec, end_sec]).all() or not 0 <= start_sec < end_sec:
        raise ValueError("invalid_noise_interval")
    return end_sec - start_sec


def normalized_duration_contrast(duration_a: float, duration_b: float) -> float:
    if not np.isfinite([duration_a, duration_b]).all() or min(duration_a, duration_b) <= 0:
        raise ValueError("invalid_duration_pair")
    return (duration_a - duration_b) / ((duration_a + duration_b) / 2)
