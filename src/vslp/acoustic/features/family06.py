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
UNAVAILABLE_IDS = frozenset({"blocked_aural_analytics_ap",
                              "normalized_temporal_contrast_historical",
                              "wideband_noise_energy_0_10khz"})
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


# The legacy generic target-manifest functions above remain for historical
# audit compatibility. Production V1 targets below are explicit and task bound.
F06_VERSION = "family06-native-subevents-2.0.0"
F06_PARAMETER_SET_ID = "family06_native_subevents_v2"
M1_BAND_HZ = (500., 5000.)
TARGETS_V2 = {
    "see_s": ("wstg", "wstg_we_see_three_geese", "see", "S"),
    "geese_s": ("wstg", "wstg_we_see_three_geese", "geese", "S"),
    "three_th": ("wstg", "wstg_we_see_three_geese", "three", "TH"),
    "spot_t": ("sentence_s3", "sentence_s3_blue_spot_key", "spot", "T"),
    "key_k": ("sentence_s3", "sentence_s3_blue_spot_key", "key", "K"),
    "judge_jh": ("sentence_s7", "sentence_s7_municipal_judge", "judge", "JH"),
    "sentenced_t": ("sentence_s7", "sentence_s7_municipal_judge", "sentenced", "T"),
    "keep_k": ("sentence_s9", "sentence_s9_peter_will_keep", "keep", "K"),
    "at_t": ("sentence_s9", "sentence_s9_peter_will_keep", "at", "T"),
}
F06_PUBLIC_IDS = frozenset({
    *(f"band_noise_{variant}_{target}" for variant in ("a", "b")
      for target in ("see_s", "geese_s", "three_th")),
    "band_noise_a_s_minus_th", "band_noise_b_s_minus_th",
    "spectral_m1_t_minus_k_hz",
    "fricative_noise_duration_s_ms", "fricative_noise_duration_th_ms",
    "fricative_noise_duration_s_minus_th_ms",
    "noise_rise_time_judge_jh_ms", "noise_rise_time_sentenced_t_ms",
    "noise_rise_time_t_minus_jh_ms",
    "spectral_tilt_k_minus_t_db_per_khz",
})


def resolve_v1_target(store, recording_id: str, trial_id: str,
                      target_id: str) -> tuple[pd.DataFrame, str]:
    """Use only feature-facing canonical word/phone identity from frozen V1."""
    task_id, prompt_id, word_text, phone_label = TARGETS_V2[target_id]
    trial = store.trials.loc[store.trials.recording_id.astype(str).eq(recording_id)
                             & store.trials.trial_id.astype(str).eq(trial_id)]
    if len(trial) != 1 or str(trial.iloc[0].task_id) != task_id:
        return pd.DataFrame(), "wrong_task"
    if str(trial.iloc[0].prompt_id) != prompt_id:
        return pd.DataFrame(), "wrong_prompt"
    words = store.words_for_trial(recording_id, trial_id)
    parents = words.loc[words.canonical_word_text.astype(str).str.casefold().eq(word_text)]
    if parents.empty:
        return pd.DataFrame(), "target_word_missing"
    phones = store.phones_for_trial(recording_id, trial_id)
    matching_label = phones.loc[phones.phone_normalized.astype(str).str.upper().eq(phone_label)]
    if matching_label.empty:
        return pd.DataFrame(), "target_phone_missing"
    matched = matching_label.loc[matching_label.parent_word_id.astype(str).isin(
        parents.word_id.astype(str))]
    if matched.empty or not matched.parent_word_text.astype(str).str.casefold().eq(word_text).all():
        return pd.DataFrame(), "wrong_parent_word"
    if target_id == "sentenced_t":
        # english_us_arpa: S EH1 N T AH0 N S T. V1 phone_index_in_word is
        # 1-based provider order. Require the prefix as well as position 4 so
        # a later T cannot shift into position 4 after missing earlier phones.
        if "phone_index_in_word" not in phones.columns:
            return pd.DataFrame(), "target_phone_position_unavailable"
        prefixes = []
        for parent_id in parents.word_id.astype(str):
            prefix = phones.loc[phones.parent_word_id.astype(str).eq(parent_id)].copy()
            prefix["_position"] = pd.to_numeric(prefix.phone_index_in_word,
                                                   errors="coerce")
            prefix = prefix.sort_values("_position")
            if (prefix.head(4)._position.tolist() == [1, 2, 3, 4]
                    and prefix.head(4).phone_normalized.astype(str).str.upper().tolist()
                    == ["S", "EH", "N", "T"]):
                prefixes.append(parent_id)
        selected = matched.loc[pd.to_numeric(matched.phone_index_in_word,
                                              errors="coerce").eq(4)
                               & matched.parent_word_id.astype(str).isin(prefixes)]
        if selected.empty:
            return pd.DataFrame(), "target_phone_missing"
        if selected.groupby(selected.parent_word_id.astype(str)).size().gt(1).any():
            return pd.DataFrame(), "ambiguous_target_phone"
        return selected.copy(), ""
    if matched.groupby(matched.parent_word_id.astype(str)).size().gt(1).any():
        return pd.DataFrame(), "ambiguous_target_phone"
    return matched.copy(), ""


def _one_sided_power(audio: np.ndarray, rate: int, start: float,
                     duration: float, required_hz: float) -> tuple[np.ndarray, np.ndarray, float]:
    if rate / 2 <= required_hz:
        raise ValueError("sample_rate_insufficient")
    frame = _window(audio, rate, start, duration)
    frame = frame - np.mean(frame)  # DC only; never alter native source audio.
    window = np.hanning(len(frame))
    n_fft = max(FFT_SIZE, 1 << (len(frame) - 1).bit_length())
    frequency = np.fft.rfftfreq(n_fft, 1 / rate)
    power = np.abs(np.fft.rfft(frame * window, n=n_fft)) ** 2
    if len(power) > 2:
        power[1:-1] *= 2
    denominator = rate * float(np.sum(window ** 2))
    if denominator <= 0:
        raise ValueError("insufficient_spectral_support")
    return frequency, power / denominator, rate / n_fft


def band_noise_power(audio: np.ndarray, rate: int, start: float,
                     end: float, variant: str) -> float:
    """One-sided integrated digital PSD; A/B retain their source band formulae."""
    ceiling = {"a": 10000., "b": 6500.}.get(variant)
    if ceiling is None:
        raise ValueError("unknown_noise_band_variant")
    frequency, density, step = _one_sided_power(audio, rate, start,
                                                 end - start, ceiling)

    def energy(low: float, high: float) -> float:
        return float(density[(frequency >= low) & (frequency <= high)].sum() * step)

    if variant == "a":
        value = energy(720, 10000) - energy(4200, 10000)
    elif variant == "b":
        value = energy(720, 6500) - energy(720, 2150)
    if value < -1e-12:
        raise ValueError("invalid_band_contrast_power")
    return max(0., value)  # remove only floating-point subtraction roundoff


def first_spectral_moment_band(audio: np.ndarray, rate: int,
                               burst_start: float) -> float:
    """Hann one-sided power centroid over the common 500–5000-Hz band."""
    frequency, density, _ = _one_sided_power(audio, rate, burst_start,
                                              M1_WINDOW_SEC, M1_BAND_HZ[1])
    mask = (frequency >= M1_BAND_HZ[0]) & (frequency <= M1_BAND_HZ[1])
    values = density[mask]
    total = float(values.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("insufficient_spectral_support")
    return float(np.dot(frequency[mask], values) / total)
