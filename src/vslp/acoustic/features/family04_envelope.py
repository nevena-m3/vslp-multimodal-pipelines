"""Versioned VSLP all-pole envelope analogue for the S3 /a/-/i/ pair.

The source's complete pitch-synchronous LPC/grid/normalization settings have not
been recovered. This implementation must never be labelled source-parity tested.
"""

from __future__ import annotations

import numpy as np
import parselmouth
from scipy.linalg import solve_toeplitz


ENVELOPE_PARAMETER_SET_ID = "family04_s3_allpole_period_window_v1"
ENVELOPE_SOURCE_PARITY = "pending"
ENVELOPE_GRID_HZ = np.linspace(0., 5000., 256)


def allpole_token_envelope(audio: np.ndarray, sample_rate: int,
                           start_sec: float, end_sec: float) -> np.ndarray:
    """Mean-centered dB LPC envelope from a pitch-period-sized midpoint frame."""
    if sample_rate <= 11000 or end_sec - start_sec < .05:
        raise ValueError("invalid_envelope_token")
    midpoint = (start_sec + end_sec) / 2
    token_left = max(0, round(start_sec * sample_rate))
    token_right = min(len(audio), round(end_sec * sample_rate))
    token_audio = np.asarray(audio[token_left:token_right], dtype=float)
    if len(token_audio) < round(.05 * sample_rate):
        raise ValueError("invalid_envelope_token")
    sound = parselmouth.Sound(token_audio, sampling_frequency=sample_rate,
                             start_time=token_left / sample_rate)
    pitch = sound.to_pitch_ac(time_step=.005, pitch_floor=60, pitch_ceiling=500)
    f0 = float(pitch.get_value_at_time(midpoint))
    if not np.isfinite(f0) or not 60 <= f0 <= 500:
        raise ValueError("invalid_envelope_f0")
    pulses = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 60, 500)
    pulse_index = int(parselmouth.praat.call(pulses, "Get nearest index", midpoint))
    if pulse_index < 1:
        raise ValueError("missing_envelope_pulse")
    pulse_time = float(parselmouth.praat.call(pulses, "Get time from index", pulse_index))
    if not start_sec <= pulse_time <= end_sec:
        raise ValueError("envelope_pulse_outside_token")
    duration = min(end_sec - start_sec, max(.025, 4. / f0))
    half = round(duration * sample_rate / 2)
    center = round(pulse_time * sample_rate)
    signal = np.asarray(audio[max(token_left, center - half):min(token_right, center + half)], dtype=float)
    if len(signal) < 32 or not np.isfinite(signal).all():
        raise ValueError("invalid_envelope_frame")
    # VSLP source-lineage settings: Hamming, first-order pre-emphasis at 50 Hz,
    # LPC(14), 256 common bins from 0 to 5 kHz, mean-centered log magnitude.
    pre = np.r_[signal[0], signal[1:] - np.exp(-2 * np.pi * 50 / sample_rate) * signal[:-1]]
    windowed = pre * np.hamming(len(pre))
    order = 14
    corr = np.correlate(windowed, windowed, mode="full")[len(windowed) - 1:len(windowed) + order]
    if corr[0] <= 1e-12:
        raise ValueError("invalid_envelope_energy")
    coeff = solve_toeplitz((corr[:order] + corr[0] * 1e-7 * (np.arange(order) == 0)),
                           -corr[1:order + 1])
    polynomial = np.r_[1., coeff]
    omega = 2 * np.pi * ENVELOPE_GRID_HZ / sample_rate
    denominator = np.abs(np.exp(-1j * np.outer(omega, np.arange(len(polynomial)))) @ polynomial)
    if np.any(denominator <= 1e-12) or not np.isfinite(denominator).all():
        raise ValueError("invalid_envelope_lpc")
    envelope_db = -20 * np.log10(denominator)
    return envelope_db - float(np.mean(envelope_db))


def envelope_l1_distance(envelope_i: np.ndarray, envelope_a: np.ndarray) -> float:
    """Source-lineage mean L1 distance on the shared envelope grid."""
    left, right = np.asarray(envelope_i, dtype=float), np.asarray(envelope_a, dtype=float)
    if left.shape != right.shape or left.shape != ENVELOPE_GRID_HZ.shape:
        raise ValueError("envelope_grid_mismatch")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("nonfinite_envelope")
    return float(np.mean(np.abs(left - right)))
