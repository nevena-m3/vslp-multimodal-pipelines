"""Family 12 explicit 48 kHz private spectral and MFCC representations."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
import librosa
from scipy.fft import dct
from scipy.signal import resample_poly

ALGORITHM_VERSION = "family12-spectral-1.0.0"
MFCC_PARAMETER_SET_ID = "family12_mfcc_48k_v1"
SPECTRAL_PARAMETER_SET_ID = "family12_spectral_48k_v1"
WORKING_SAMPLE_RATE = 48_000
N_FFT = 2048
WINDOW_LENGTH = 1200
HOP_LENGTH = 480
MINIMUM_FRAMES = 10
MFCC_IDS = frozenset({*(f"mfcc{i:02d}_mean" for i in range(1, 14)),
                      *(f"mfcc{i:02d}_sd" for i in range(1, 14))})
SPECTRAL_IDS = frozenset({
    "spectral_bandwidth_p2_mean_hz", "spectral_bandwidth_p2_median_hz",
    "zcr_mean_fraction", "zcr_median_fraction", "spec_m1_hz", "spec_m2_hz",
    "spec_m3_skew", "spec_m4_kurtosis", "spectral_centroid_hz",
    "spectral_spread_hz", "spectral_skew", "spectral_kurtosis",
    "spectral_slope_db_khz", "spectral_flux", "spectral_crest", "spectral_flatness",
})
CONTRAST_IDS = frozenset(f"spectral_contrast_band{band}_db" for band in range(7))
FAMILY12_IDS = MFCC_IDS | SPECTRAL_IDS | CONTRAST_IDS


def spectral_contrast_band_edges(sample_rate: int = WORKING_SAMPLE_RATE) -> np.ndarray:
    """Effective librosa band edges; the last band extends through Nyquist."""
    edges = np.zeros(8, dtype=float)
    edges[1:] = 200.0 * 2.0 ** np.arange(7)
    edges[-1] = sample_rate / 2.0
    return edges


@dataclass(frozen=True)
class SpectralAudit:
    native_sample_rate_hz: int
    working_sample_rate_hz: int
    resampled: bool
    n_frames: int
    mfcc_matrix: np.ndarray


def _private_48k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64)
    if sample_rate == WORKING_SAMPLE_RATE:
        return x.copy()
    factor = gcd(sample_rate, WORKING_SAMPLE_RATE)
    return resample_poly(x, WORKING_SAMPLE_RATE // factor, sample_rate // factor)


def frame_signal(audio: np.ndarray, frame_length: int = WINDOW_LENGTH,
                 hop_length: int = HOP_LENGTH) -> np.ndarray:
    """Deterministic center=False framing without padding."""
    x = np.asarray(audio, dtype=np.float64)
    if x.size < frame_length:
        return np.empty((0, frame_length), dtype=np.float64)
    count = 1 + (x.size - frame_length) // hop_length
    return np.lib.stride_tricks.sliding_window_view(x, frame_length)[::hop_length][:count].copy()


def _hz_to_mel(hz):
    hz = np.asarray(hz, dtype=float)
    f_sp = 200.0 / 3
    mel = hz / f_sp
    log_start = 1000.0
    log_step = np.log(6.4) / 27.0
    mask = hz >= log_start
    mel[mask] = 15.0 + np.log(hz[mask] / log_start) / log_step
    return mel


def _mel_to_hz(mel):
    mel = np.asarray(mel, dtype=float)
    hz = mel * (200.0 / 3)
    mask = mel >= 15.0
    hz[mask] = 1000.0 * np.exp(np.log(6.4) / 27.0 * (mel[mask] - 15.0))
    return hz


def _mel_filters() -> np.ndarray:
    frequencies = np.fft.rfftfreq(N_FFT, 1 / WORKING_SAMPLE_RATE)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(np.array([50.0]))[0],
                                   _hz_to_mel(np.array([8000.0]))[0], 42))
    filters = np.zeros((40, frequencies.size))
    for index, (left, middle, right) in enumerate(zip(edges[:-2], edges[1:-1], edges[2:], strict=True)):
        filters[index] = np.maximum(0, np.minimum((frequencies-left)/(middle-left),
                                                  (right-frequencies)/(right-middle)))
        filters[index] *= 2.0 / (right - left)  # Slaney area normalization.
    return filters


def calculate_spectral_features(audio: np.ndarray, sample_rate: int):
    x = np.asarray(audio, dtype=np.float64)
    empty = {feature_id: (np.nan, "invalid_analysis_audio") for feature_id in FAMILY12_IDS}
    if sample_rate <= 0 or x.ndim != 1 or not np.isfinite(x).all():
        return empty, SpectralAudit(sample_rate, WORKING_SAMPLE_RATE, False, 0, np.empty((0, 13)))
    work = _private_48k(x, sample_rate)
    frames = frame_signal(work)
    audit = SpectralAudit(sample_rate, WORKING_SAMPLE_RATE, sample_rate != WORKING_SAMPLE_RATE,
                          len(frames), np.empty((0, 13)))
    if len(frames) < MINIMUM_FRAMES:
        return ({feature_id: (np.nan, "insufficient_spectral_frames") for feature_id in FAMILY12_IDS}, audit)
    windowed = frames * np.hanning(WINDOW_LENGTH)
    spectrum = np.abs(np.fft.rfft(windowed, n=N_FFT, axis=1))
    power = np.square(spectrum)
    energy = power.sum(axis=1)
    valid = energy > np.finfo(float).tiny
    if valid.sum() < MINIMUM_FRAMES:
        return ({feature_id: (np.nan, "insufficient_nonzero_spectral_frames") for feature_id in FAMILY12_IDS}, audit)
    spectrum, power, frames = spectrum[valid], power[valid], frames[valid]
    frequencies = np.fft.rfftfreq(N_FFT, 1 / WORKING_SAMPLE_RATE)
    weights = power / power.sum(axis=1, keepdims=True)
    magnitude_weights = spectrum / spectrum.sum(axis=1, keepdims=True)
    m1 = weights @ frequencies
    centered = frequencies[None, :] - m1[:, None]
    m2 = np.sqrt(np.sum(weights * centered**2, axis=1))
    bandwidth_centroid = magnitude_weights @ frequencies
    bandwidth = np.sqrt(np.sum(magnitude_weights *
                               (frequencies[None, :] - bandwidth_centroid[:, None])**2,
                               axis=1))
    usable = m2 > np.finfo(float).eps
    m3 = np.full(len(m1), np.nan)
    m4 = np.full(len(m1), np.nan)
    m3[usable] = np.sum(weights[usable] * centered[usable]**3, axis=1) / m2[usable]**3
    m4[usable] = np.sum(weights[usable] * centered[usable]**4, axis=1) / m2[usable]**4
    mel_energy = power @ _mel_filters().T
    mfcc = dct(np.log(np.maximum(mel_energy, np.finfo(float).tiny)), type=2,
               axis=1, norm="ortho")[:, :13]
    result = {}
    for i in range(13):
        result[f"mfcc{i+1:02d}_mean"] = (float(np.mean(mfcc[:, i])), "")
        result[f"mfcc{i+1:02d}_sd"] = (float(np.std(mfcc[:, i], ddof=1)), "")
    result.update({
        "spectral_bandwidth_p2_mean_hz": (float(np.mean(bandwidth)), ""),
        "spectral_bandwidth_p2_median_hz": (float(np.median(bandwidth)), ""),
        "spec_m1_hz": (float(np.mean(m1)), ""), "spec_m2_hz": (float(np.mean(m2)), ""),
        "spec_m3_skew": (float(np.nanmean(m3)), ""),
        "spec_m4_kurtosis": (float(np.nanmean(m4)), ""),
        "spectral_centroid_hz": (float(np.mean(m1)), ""),
        "spectral_spread_hz": (float(np.mean(m2)), ""),
        "spectral_skew": (float(np.nanmean(m3)), ""),
        "spectral_kurtosis": (float(np.nanmean(m4)), ""),
    })
    contrast = librosa.feature.spectral_contrast(
        y=work, sr=WORKING_SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WINDOW_LENGTH, window="hann", center=False, fmin=200.0,
        n_bands=6, quantile=0.02, linear=False)
    for band in range(7):
        result[f"spectral_contrast_band{band}_db"] = (
            float(np.mean(contrast[band])), "")
    # ZCR has mandatory feature-local DC removal and no denoising.
    dc_frames = frames - np.mean(frames, axis=1, keepdims=True)
    signs = dc_frames >= 0
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
    result["zcr_mean_fraction"] = (float(np.mean(zcr)), "")
    result["zcr_median_fraction"] = (float(np.median(zcr)), "")
    normalized = spectrum / np.maximum(np.linalg.norm(spectrum, axis=1, keepdims=True),
                                       np.finfo(float).tiny)
    flux = np.linalg.norm(np.diff(normalized, axis=0), axis=1)
    db = 20 * np.log10(np.maximum(spectrum, np.finfo(float).tiny))
    freq_khz = frequencies / 1000
    slopes = np.array([np.polyfit(freq_khz, row, 1)[0] for row in db])
    crest = np.max(spectrum, axis=1) / np.maximum(np.mean(spectrum, axis=1), np.finfo(float).tiny)
    flatness = np.exp(np.mean(np.log(np.maximum(power, np.finfo(float).tiny)), axis=1)) / np.mean(power, axis=1)
    result.update({
        "spectral_slope_db_khz": (float(np.mean(slopes)), ""),
        "spectral_flux": ((float(np.mean(flux)), "") if flux.size else (np.nan, "insufficient_spectral_flux_frames")),
        "spectral_crest": (float(np.mean(crest)), ""),
        "spectral_flatness": (float(np.mean(flatness)), ""),
    })
    return result, SpectralAudit(sample_rate, WORKING_SAMPLE_RATE,
                                 sample_rate != WORKING_SAMPLE_RATE, len(frames), mfcc)
