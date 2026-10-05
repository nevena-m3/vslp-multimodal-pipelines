"""Frame-matched raw and Iseli-corrected H1–H2 on reviewed F02 audio."""

from __future__ import annotations

import numpy as np
import parselmouth
from scipy import signal

from vslp.acoustic.features.family04 import FORMANT_BOUNDS, profile_by_id


PARAMETER_SET_ID = "family02_h1h2_iseli_2007_v1"
ALGORITHM_VERSION = "family02-h1h2-iseli-1.0.0"
FRAME_LENGTH_SEC = .025
FRAME_STEP_SEC = .005
FFT_SIZE = 8192
SPECTRAL_WINDOW = "Hann"
MIN_VALID_FRAMES = 20
LOW_YIELD_FRACTION = .5  # VSLP QC flag, not a clinical threshold.
EXTREME_CORRECTION_DB = 30.0  # VSLP QC flag; values are never clipped.
FEATURE_IDS = frozenset({"h1_h2_db_raw", "h1star_h2star_db"})


def _formants_for_span(segment: np.ndarray, sample_rate_hz: int):
    profile = profile_by_id()
    sound = parselmouth.Sound(segment, sampling_frequency=sample_rate_hz)
    return sound.to_formant_burg(
        time_step=profile.time_step_sec,
        max_number_of_formants=profile.max_formants,
        maximum_formant=profile.ceiling_hz,
        window_length=profile.window_length_sec,
        pre_emphasis_from=profile.pre_emphasis_from_hz)


def raw_h1_h2_db(h1_db: float, h2_db: float) -> float:
    return float(h1_db - h2_db)


def mannell_bandwidth_hz(formant_hz: float) -> float:
    """Iseli et al. (2007), Eq. 2, citing Mannell (1998)."""
    return float(80.0 + 120.0 * formant_hz / 5000.0)


def iseli_formant_boost_db(harmonic_hz: float, sample_rate_hz: int,
                           formants_hz: tuple[float, float],
                           bandwidths_hz: tuple[float, float]) -> float:
    """Equation 1/A5 of Iseli et al. 2007, N=2, in the sampled domain."""
    values = (*formants_hz, *bandwidths_hz, harmonic_hz, sample_rate_hz)
    if (not np.isfinite(values).all() or harmonic_hz <= 0 or
            harmonic_hz >= sample_rate_hz / 2 or
            not 0 < formants_hz[0] < formants_hz[1] < sample_rate_hz / 2 or
            any(b <= 0 for b in bandwidths_hz)):
        raise ValueError("invalid_formant_correction_inputs")
    omega = 2 * np.pi * harmonic_hz / sample_rate_hz
    boost = 0.0
    for formant, bandwidth in zip(formants_hz, bandwidths_hz, strict=True):
        radius = np.exp(-np.pi * bandwidth / sample_rate_hz)
        theta = 2 * np.pi * formant / sample_rate_hz
        numerator = (1 - 2 * radius * np.cos(theta) + radius**2)**2
        denominator = ((1 - 2 * radius * np.cos(omega + theta) + radius**2) *
                       (1 - 2 * radius * np.cos(omega - theta) + radius**2))
        if numerator <= 0 or denominator <= 0:
            raise ValueError("nonfinite_formant_correction")
        boost += 10 * np.log10(numerator / denominator)
    if not np.isfinite(boost):
        raise ValueError("nonfinite_formant_correction")
    return float(boost)


def corrected_h1_h2_db(h1_db: float, h2_db: float, f0_hz: float,
                        sample_rate_hz: int, formants_hz: tuple[float, float],
                        bandwidths_hz: tuple[float, float]) -> float:
    first = iseli_formant_boost_db(f0_hz, sample_rate_hz, formants_hz, bandwidths_hz)
    second = iseli_formant_boost_db(2 * f0_hz, sample_rate_hz, formants_hz,
                                    bandwidths_hz)
    return raw_h1_h2_db(h1_db, h2_db) - first + second


def _harmonic_amplitudes_db(frame: np.ndarray, sample_rate_hz: int,
                            f0_hz: float) -> tuple[float, float]:
    if frame.size > FFT_SIZE or not np.isfinite(frame).all() or f0_hz <= 0:
        raise ValueError("harmonic_amplitude_failure")
    windowed = frame * signal.windows.hann(frame.size, sym=False)
    spectrum = np.abs(np.fft.rfft(windowed, n=FFT_SIZE))
    bins = np.fft.rfftfreq(FFT_SIZE, 1 / sample_rate_hz)
    amplitudes = np.interp([f0_hz, 2 * f0_hz], bins, spectrum)
    if not np.isfinite(amplitudes).all() or np.any(amplitudes <= 1e-12):
        raise ValueError("harmonic_amplitude_failure")
    return tuple(20 * np.log10(amplitudes))


def harmonic_difference_trial(parts: list, audio: np.ndarray, sample_rate_hz: int
                              ) -> tuple[dict[str, tuple[float, str]], list[dict], dict]:
    """Measure each eligible F0 frame, keeping original time and span breaks."""
    rows: list[dict] = []
    half = round(FRAME_LENGTH_SEC * sample_rate_hz / 2)
    for track in parts:
        start = int(round(track.analysis_start_sec * sample_rate_hz))
        end = int(round(track.analysis_end_sec * sample_rate_hz))
        try:
            formants = _formants_for_span(audio[start:end], sample_rate_hz)
        except (ValueError, RuntimeError):
            formants = None
        for time, f0, voiced in zip(track.time_sec, track.f0_hz,
                                    track.voiced_valid, strict=True):
            row = {"time_sec": float(time), "f0_hz": float(f0),
                   "f1_hz": np.nan, "f2_hz": np.nan,
                   "b1_hz": np.nan, "b2_hz": np.nan,
                   "bandwidth_source": "", "h1_db": np.nan, "h2_db": np.nan,
                   "h1_h2_db_raw": np.nan, "h1star_h2star_db": np.nan,
                   "qc_flags": ""}
            flags = []
            center = int(round(time * sample_rate_hz))
            left, right = center - half, center + half
            if not voiced or not np.isfinite(f0) or f0 <= 0 or 2 * f0 >= sample_rate_hz / 2:
                flags.append("invalid_f0")
            elif left < start or right > end:
                flags.append("frame_outside_eligible_span")
            else:
                try:
                    h1_db, h2_db = _harmonic_amplitudes_db(audio[left:right],
                                                            sample_rate_hz, float(f0))
                    row.update(h1_db=h1_db, h2_db=h2_db,
                               h1_h2_db_raw=raw_h1_h2_db(h1_db, h2_db))
                except ValueError:
                    flags.append("harmonic_amplitude_failure")
                if not flags and formants is None:
                    flags.extend(("invalid_f1", "invalid_f2"))
                if not flags:
                    local_time = float(time - track.analysis_start_sec)
                    try:
                        frequencies = [float(formants.get_value_at_time(i, local_time))
                                       for i in (1, 2)]
                    except (ValueError, RuntimeError):
                        frequencies = [np.nan, np.nan]
                    for index, frequency in enumerate(frequencies, 1):
                        low, high = FORMANT_BOUNDS[f"f{index}"]
                        if not np.isfinite(frequency) or not low <= frequency <= high:
                            flags.append(f"invalid_f{index}")
                    if np.isfinite(frequencies).all() and frequencies[1] <= frequencies[0]:
                        flags.append("invalid_formant_order")
                    if not flags:
                        try:
                            measured = [float(formants.get_bandwidth_at_time(i, local_time))
                                        for i in (1, 2)]
                        except (ValueError, RuntimeError):
                            measured = [np.nan, np.nan]
                        credible = [np.isfinite(b) and 20 <= b <= 1000 and b < f
                                    for b, f in zip(measured, frequencies, strict=True)]
                        bandwidths = (measured if all(credible) else
                                      [mannell_bandwidth_hz(f) for f in frequencies])
                        row.update(f1_hz=frequencies[0], f2_hz=frequencies[1],
                                   b1_hz=bandwidths[0], b2_hz=bandwidths[1],
                                   bandwidth_source="measured" if all(credible) else
                                   "formula_fallback")
                        if not all(credible):
                            flags.extend(f"invalid_b{i}" for i, ok in enumerate(credible, 1)
                                         if not ok)
                        try:
                            corrected = corrected_h1_h2_db(
                                h1_db, h2_db, float(f0), sample_rate_hz,
                                tuple(frequencies), tuple(bandwidths))
                            if np.isfinite(corrected):
                                row["h1star_h2star_db"] = corrected
                                if abs(corrected - row["h1_h2_db_raw"]) > EXTREME_CORRECTION_DB:
                                    flags.append("correction_extreme")
                            else:
                                flags.append("nonfinite_formant_correction")
                        except ValueError:
                            flags.append("nonfinite_formant_correction")
            row["qc_flags"] = ";".join(flags)
            rows.append(row)
    candidate = len(rows)
    values = {}
    fractions = {}
    for feature_id in FEATURE_IDS:
        valid = np.array([row[feature_id] for row in rows
                          if np.isfinite(row[feature_id])], dtype=float)
        fractions[feature_id] = len(valid) / candidate if candidate else 0.0
        if len(valid) >= MIN_VALID_FRAMES:
            values[feature_id] = (float(np.median(valid)), "")
        else:
            reasons = {flag for row in rows for flag in row["qc_flags"].split(";") if flag}
            priority = (["invalid_f1", "invalid_f2", "invalid_formant_order",
                         "invalid_b1", "invalid_b2", "nonfinite_formant_correction"]
                        if feature_id == "h1star_h2star_db" else
                        ["harmonic_amplitude_failure", "invalid_f0"])
            reason = next((flag for flag in priority if flag in reasons),
                          "insufficient_valid_frames")
            values[feature_id] = (np.nan, reason)
    flags = sorted({flag for row in rows for flag in row["qc_flags"].split(";") if flag})
    if any(fraction < LOW_YIELD_FRACTION for fraction in fractions.values()):
        flags.append("low_valid_frame_yield")
    if any(value[1] for value in values.values()):
        flags.append("insufficient_valid_frames")
    return values, rows, {"n_candidate_frames": candidate,
                          "n_valid_raw_frames": sum(np.isfinite(row["h1_h2_db_raw"])
                                                    for row in rows),
                          "n_valid_corrected_frames": sum(np.isfinite(row["h1star_h2star_db"])
                                                          for row in rows),
                          "bandwidth_sources": sorted({row["bandwidth_source"] for row in rows
                                                       if row["bandwidth_source"]}),
                          "valid_frame_fraction": fractions,
                          "qc_flags": sorted(set(flags))}
