"""QCP-lineage glottal inverse filtering on one reviewed contiguous vowel span.

The acoustic waveform and its native sample rate remain unchanged. The internal
8-kHz representation is used only for this estimator. Estimated Praat pulse
times are GCI candidates, not electroglottographic ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy import signal


GIF_ALGORITHM_VERSION = "vslp-qcp-lineage-1.0.0"
GIF_PARAMETER_SET_ID = "family03_qcp_8khz_ame_v1"
INTERNAL_RATE_HZ = 8000
FRAME_SEC = .030
HOP_SEC = .010
LP_ORDER = 10
PREEMPHASIS_ZERO = .99
AME_FLOOR = 1e-5
AME_POSITION_QUOTIENT = .05
AME_RAMP_SAMPLES = 7
MIN_PERIOD_SEC = .8 / 500
MAX_PERIOD_SEC = 1.25 / 60


@dataclass(frozen=True)
class GlottalEstimate:
    flow: np.ndarray
    derivative: np.ndarray
    sample_rate_hz: int
    native_sample_rate_hz: int
    pulse_samples: np.ndarray
    start_sec: float
    qc_flags: tuple[str, ...]


def _dq_for_f0(f0_hz: float) -> float:
    if f0_hz < 190:
        return .95
    if f0_hz < 280:
        return .55
    return .70


def _ame_weights(n_samples: int, pulses: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    """Pitch-adaptive AME: unit weight in quasi-closed phase, attenuated elsewhere."""
    weights = np.full(n_samples, AME_FLOOR, dtype=float)
    for left, right in zip(pulses[:-1], pulses[1:], strict=True):
        period = int(right - left)
        if period <= 2 * AME_RAMP_SAMPLES:
            continue
        f0 = sample_rate_hz / period
        first = int(left + round(AME_POSITION_QUOTIENT * period))
        last = min(int(right), first + round(_dq_for_f0(f0) * period))
        first = max(0, min(n_samples, first))
        last = max(first, min(n_samples, last))
        weights[first:last] = 1.0
        ramp = min(AME_RAMP_SAMPLES, first, n_samples - last)
        if ramp:
            weights[first - ramp:first] = np.linspace(AME_FLOOR, 1., ramp + 1)[:-1]
            weights[last:last + ramp] = np.linspace(1., AME_FLOOR, ramp + 1)[1:]
    return weights


def _weighted_lp(frame: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Solve the weighted normal equation of Airaksinen et al. (2013), Eq. 4."""
    if len(frame) <= LP_ORDER * 3:
        raise ValueError("qcp_frame_too_short")
    target = frame[LP_ORDER:]
    history = np.column_stack([frame[LP_ORDER - lag:-lag]
                               for lag in range(1, LP_ORDER + 1)])
    root_w = np.sqrt(weights[LP_ORDER:])
    predictor, *_ = np.linalg.lstsq(history * root_w[:, None],
                                     target * root_w, rcond=None)
    if not np.isfinite(predictor).all():
        raise ValueError("qcp_nonfinite_filter")
    polynomial = np.r_[1., -predictor]
    roots = np.roots(polynomial)
    if np.any(np.abs(roots) >= 1):
        roots = np.where(np.abs(roots) >= 1, 1 / np.conj(roots), roots)
        polynomial = np.real_if_close(np.poly(roots)).real
    return polynomial


def _pulse_candidates(audio: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate_hz)
    points = call(sound, "To PointProcess (periodic, cc)", 60., 500.)
    count = int(call(points, "Get number of points"))
    return np.array([float(call(points, "Get time from index", index))
                     for index in range(1, count + 1)], dtype=float)


def qcp_glottal_estimate(audio: np.ndarray, native_sample_rate_hz: int,
                         start_sec: float, pulse_times_sec: np.ndarray | None = None
                         ) -> GlottalEstimate:
    """Estimate derivative by AME-WLP inverse filtering and integrate lip radiation."""
    x = np.asarray(audio, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or x.size < .1 * native_sample_rate_hz:
        raise ValueError("invalid_gif_audio")
    if pulse_times_sec is None:
        pulse_times_sec = _pulse_candidates(x, native_sample_rate_hz)
    pulses_sec = np.asarray(pulse_times_sec, dtype=float)
    if pulses_sec.size < 4 or not np.isfinite(pulses_sec).all():
        raise ValueError("insufficient_gci_candidates")
    ratio = Fraction(INTERNAL_RATE_HZ, native_sample_rate_hz)
    work = signal.resample_poly(x, ratio.numerator, ratio.denominator)
    pulses = np.unique(np.rint(pulses_sec * INTERNAL_RATE_HZ).astype(int))
    pulses = pulses[(pulses >= 0) & (pulses < work.size)]
    if pulses.size < 4:
        raise ValueError("insufficient_gci_candidates")
    ame = _ame_weights(work.size, pulses, INTERNAL_RATE_HZ)
    # Pre-emphasis is for vocal-tract model estimation only. Inverse filtering
    # applies the estimated tract inverse to original pressure; integration then
    # compensates first-order lip radiation.
    emphasized = signal.lfilter([1., -PREEMPHASIS_ZERO], [1.], work)
    frame_size = round(FRAME_SEC * INTERNAL_RATE_HZ)
    hop = round(HOP_SEC * INTERNAL_RATE_HZ)
    accumulated = np.zeros_like(work)
    support = np.zeros_like(work)
    fitted = 0
    for start in range(0, max(1, work.size - frame_size + 1), hop):
        end = min(start + frame_size, work.size)
        if end - start < frame_size:
            continue
        try:
            polynomial = _weighted_lp(emphasized[start:end], ame[start:end])
        except (ValueError, np.linalg.LinAlgError):
            continue
        residual = signal.lfilter(polynomial, [1.], work[start:end])
        taper = signal.windows.hann(frame_size, sym=False)
        accumulated[start:end] += residual * taper
        support[start:end] += taper
        fitted += 1
    if fitted < 2:
        raise ValueError("qcp_vocal_tract_estimation_failed")
    derivative = np.divide(accumulated, support, out=np.full_like(work, np.nan),
                           where=support > 1e-8)
    finite = np.isfinite(derivative)
    if not finite.any():
        raise ValueError("qcp_nonfinite_inverse_filter")
    derivative[~finite] = 0.
    flow = np.cumsum(derivative) / INTERNAL_RATE_HZ
    flow = signal.detrend(flow, type="linear")
    if not np.isfinite(flow).all():
        raise ValueError("qcp_nonfinite_flow")
    flags = ("gci_candidates_from_acoustic_praat_pulses",
             "not_reference_qcp_parity")
    return GlottalEstimate(flow, derivative, INTERNAL_RATE_HZ,
                           native_sample_rate_hz, pulses, start_sec, flags)


def cycles_from_estimate(estimate: GlottalEstimate) -> tuple[list[dict], list[dict]]:
    """Keep only whole, finite same-run cycles; retain original coordinates."""
    accepted, rejected = [], []
    for index, (left, right) in enumerate(zip(estimate.pulse_samples[:-1],
                                              estimate.pulse_samples[1:], strict=True), 1):
        period = (right - left) / estimate.sample_rate_hz
        reason = ""
        if not MIN_PERIOD_SEC <= period <= MAX_PERIOD_SEC:
            reason = "cycle_period_out_of_bounds"
        elif left <= 0 or right >= len(estimate.flow) - 1:
            reason = "truncated_cycle"
        flow = estimate.flow[left:right]
        derivative = estimate.derivative[left:right]
        if not reason and (len(flow) < 8 or not np.isfinite(flow).all() or
                           not np.isfinite(derivative).all()):
            reason = "nonfinite_or_short_cycle"
        if not reason and np.ptp(flow) <= 1e-12:
            reason = "unstable_glottal_flow"
        record = {"cycle_id": index, "start_sec": estimate.start_sec + left /
                  estimate.sample_rate_hz, "end_sec": estimate.start_sec + right /
                  estimate.sample_rate_hz, "period_sec": period,
                  "flow": flow, "derivative": derivative}
        if reason:
            rejected.append({**{key: value for key, value in record.items()
                                if key not in {"flow", "derivative"}}, "reason": reason})
        else:
            accepted.append(record)
    return accepted, rejected
