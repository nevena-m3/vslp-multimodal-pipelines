"""Versioned, checked IIR designs used by F02 replication estimators."""

from __future__ import annotations

import numpy as np
from scipy import signal


FILTER_PARAMETER_SET_ID = "family02_sos_filters_v1"


def butterworth_sos(order: int, cutoff_hz: tuple[float, float] | float,
                    sample_rate_hz: float, kind: str) -> np.ndarray:
    """Design an SOS filter and reject nonfinite or unstable coefficients."""
    sos = signal.butter(order, cutoff_hz, btype=kind, fs=sample_rate_hz, output="sos")
    if not np.isfinite(sos).all():
        raise ValueError("nonfinite_filter_coefficients")
    poles = np.concatenate([np.roots(section[3:]) for section in sos])
    if not np.isfinite(poles).all() or np.any(np.abs(poles) >= 1):
        raise ValueError("unstable_filter")
    return sos


def filter_response(sos: np.ndarray, sample_rate_hz: float,
                    points: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    frequency, response = signal.sosfreqz(sos, worN=points, fs=sample_rate_hz)
    return frequency, 20 * np.log10(np.maximum(np.abs(response), 1e-12))
