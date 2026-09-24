"""Family 01: shared native-rate Praat F0 track and named statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import parselmouth


ALGORITHM_VERSION = "family01-praat-ac-1.0.0"
PARAMETER_SET_ID = "family01_f0_praat_ac_v1"
FAMILY01_IDS = frozenset({"f0_mean_hz", "f0_median_hz", "f0_sd_st", "f0_iqr_st",
                          "f0_range_st", "pfr_maxmin_st", "intonation_f0_sd_st",
                          "intonation_f0_iqr_st"})
PITCH_FLOOR_HZ = 60.0
PITCH_CEILING_HZ = 500.0
TIME_STEP_SEC = 0.005
MIN_VALID_FRAMES = 20
MIN_SUSTAINED_YIELD = 0.80


@dataclass(frozen=True)
class F0Track:
    time_sec: np.ndarray
    f0_hz: np.ndarray
    voiced_valid: np.ndarray
    sample_rate_hz: int
    analysis_start_sec: float
    analysis_end_sec: float
    analysis_region: str
    algorithm_version: str = ALGORITHM_VERSION
    parameter_set_id: str = PARAMETER_SET_ID

    @property
    def tracking_yield(self) -> float:
        return float(np.mean(self.voiced_valid)) if self.voiced_valid.size else 0.0


def praat_f0_track(audio: np.ndarray, sample_rate_hz: int, *,
                   analysis_start_sec: float = 0.0,
                   analysis_region: str = "reviewed_voiced_region") -> F0Track:
    """Measure a private region; the caller's canonical waveform is never changed."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or x.size < int(sample_rate_hz * 0.1) or not np.isfinite(x).all():
        raise ValueError("invalid_analysis_audio")
    sound = parselmouth.Sound(x, sampling_frequency=float(sample_rate_hz))
    pitch = sound.to_pitch_ac(time_step=TIME_STEP_SEC, pitch_floor=PITCH_FLOOR_HZ,
                              pitch_ceiling=PITCH_CEILING_HZ)
    hz = np.asarray(pitch.selected_array["frequency"], dtype=float)
    times = np.asarray(pitch.xs(), dtype=float) + analysis_start_sec
    valid = np.isfinite(hz) & (hz >= PITCH_FLOOR_HZ) & (hz <= PITCH_CEILING_HZ)
    return F0Track(times, hz, valid, sample_rate_hz, analysis_start_sec,
                   analysis_start_sec + x.size / sample_rate_hz, analysis_region)


def calculate_f0_features(track: F0Track, *, sustained: bool) -> dict[str, tuple[float, str]]:
    """Use the same valid track for all Family 01 scalar outputs."""
    valid = np.asarray(track.f0_hz[track.voiced_valid], dtype=float)
    reason = ("insufficient_valid_f0_frames" if valid.size < MIN_VALID_FRAMES else
              "poor_sustained_tracking_yield" if sustained and track.tracking_yield < MIN_SUSTAINED_YIELD
              else "")
    if reason:
        return {feature_id: (np.nan, reason) for feature_id in FAMILY01_IDS}
    median = float(np.median(valid))
    semitones = 12.0 * np.log2(valid / median)
    values = {
        "f0_mean_hz": float(np.mean(valid)),
        "f0_median_hz": median,
        "f0_sd_st": float(np.std(semitones, ddof=1)),
        "f0_iqr_st": float(np.percentile(semitones, 75) - np.percentile(semitones, 25)),
        "f0_range_st": float(np.max(semitones) - np.min(semitones)),
        "pfr_maxmin_st": float(12.0 * np.log2(np.max(valid) / np.min(valid))),
        "intonation_f0_sd_st": float(np.std(semitones, ddof=1)),
        "intonation_f0_iqr_st": float(
            np.percentile(semitones, 75) - np.percentile(semitones, 25)),
    }
    return {feature_id: (value, "") for feature_id, value in values.items()}
