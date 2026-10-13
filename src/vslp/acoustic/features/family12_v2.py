"""F12 task-specific descriptors. Each reviewed continuous interval is framed alone."""
from __future__ import annotations

from math import gcd
import numpy as np
from scipy import __version__ as SCIPY_VERSION
from scipy.fft import dct
from scipy.signal import resample_poly

VERSION = "family12-spectral-2.0.1"
FS, WIN, HOP, FFT, MIN_FRAMES = 48000, 1200, 480, 2048, 10
EPS = np.finfo(float).tiny
PREFIX = {"sustained_a": "vowel", "wstg": "s1", "sentence_s2": "s2",
          "sentence_s5": "s5", "sentence_s6": "s6", "sentence_s7": "s7",
          "bamboo_passage": "bamboo"}
MFCC = {"sustained_a", "sentence_s2", "sentence_s6"}
CONTRAST = {"sustained_a", "sentence_s5"}
MOMENTS = {"wstg", "sentence_s2", "sentence_s7"}
ZCR = {"wstg", "sentence_s7"}
SHAPE = MOMENTS | {"bamboo_passage"}


def specs():
    rows = []
    def add(task, construct, suffix, unit, selectable=True):
        rows.append({"task_id": task, "construct_id": construct,
                     "feature_id": PREFIX[task] + "_" + suffix,
                     "human_name": suffix.replace("_", " ") + " — " + PREFIX[task].upper(),
                     "unit": unit, "selectable": selectable,
                     "blocked_reason": "" if selectable else "research_pending_source_config"})
    for task in MFCC:
        for i in range(1, 14):
            for stat in ("mean", "sd"):
                add(task, "C065", f"mfcc{i:02d}_{stat}", "cepstral coefficient")
    for stat in ("mean", "median"):
        add("bamboo_passage", "C066", f"spectral_bandwidth_p2_{stat}_hz", "Hz")
    for task in CONTRAST:
        for band in range(7):
            add(task, "C067", f"spectral_contrast_band{band}_db", "dB")
    for task in ZCR:
        add(task, "C068", "spectral_band_1_4khz_percentile1", "source-specific", False)
        for stat in ("mean", "median"):
            add(task, "C069", f"zcr_{stat}_fraction", "fraction")
    for task in MOMENTS:
        for suffix, unit in (("m1_hz", "Hz"), ("m2_hz", "Hz"),
                             ("m3_skew", "dimensionless"), ("m4_kurtosis", "dimensionless")):
            add(task, "C070", "spec_" + suffix, unit)
    shape = (("spectral_centroid_hz", "Hz"), ("spectral_spread_hz", "Hz"),
             ("spectral_skew", "dimensionless"), ("spectral_kurtosis", "dimensionless"),
             ("spectral_slope_db_khz", "dB/kHz"), ("spectral_flux", "dimensionless"),
             ("spectral_crest", "dimensionless"), ("spectral_flatness", "dimensionless"))
    for task in SHAPE:
        for suffix, unit in shape if task == "bamboo_passage" else shape[4:]:
            add(task, "C071", suffix, unit)
    return sorted(rows, key=lambda row: row["feature_id"])


PUBLIC_IDS = frozenset(row["feature_id"] for row in specs() if row["selectable"])


def _frames(audio, sr):
    x = np.asarray(audio, dtype=float)
    if sr != FS:
        divisor = gcd(sr, FS)
        x = resample_poly(x, FS // divisor, sr // divisor)
    else:
        x = x.copy()
    if len(x) < WIN:
        return np.empty((0, WIN))
    return np.lib.stride_tricks.sliding_window_view(x, WIN)[::HOP].copy()


def _mel_filters(freq):
    def to_mel(hz):
        return 3 * hz / 200 if hz < 1000 else 15 + np.log(hz / 1000) / (np.log(6.4) / 27)
    def to_hz(mel):
        return 200 * mel / 3 if mel < 15 else 1000 * np.exp((mel - 15) * np.log(6.4) / 27)
    edges = np.array([to_hz(v) for v in np.linspace(to_mel(50), to_mel(8000), 42)])
    filters = np.zeros((40, len(freq)))
    for i, (left, mid, right) in enumerate(zip(edges[:-2], edges[1:-1], edges[2:], strict=True)):
        filters[i] = np.maximum(0, np.minimum((freq-left)/(mid-left), (right-freq)/(right-mid)))
        filters[i] *= 2/(right-left)
    return filters



def spectral_moments(power, frequencies):
    """Power-weighted centroid, spread, skewness and raw Pearson kurtosis."""
    p = np.asarray(power, dtype=float)
    f = np.asarray(frequencies, dtype=float)
    weights = p/p.sum(axis=1, keepdims=True)
    m1 = weights @ f
    delta = f[None, :] - m1[:, None]
    m2 = np.sqrt(np.sum(weights*delta**2, axis=1))
    m3 = np.divide(np.sum(weights*delta**3, axis=1), m2**3,
                   out=np.full_like(m2, np.nan), where=m2 > EPS)
    m4 = np.divide(np.sum(weights*delta**4, axis=1), m2**4,
                   out=np.full_like(m2, np.nan), where=m2 > EPS)
    return m1, m2, m3, m4

def _spectral_slope_frames(magnitude, frequencies):
    """Frame-relative magnitude slope over positive frequencies through Nyquist."""
    slope_bins = np.asarray(frequencies) > 0
    amplitudes = np.asarray(magnitude, dtype=float)[:, slope_bins]
    reference = np.max(amplitudes, axis=1)
    slopes = np.full(len(amplitudes), np.nan)
    valid = np.isfinite(reference) & (reference > 0)
    if not np.any(valid):
        return slopes
    levels_db = 20 * np.log10(np.maximum(amplitudes[valid] / reference[valid, None], 1e-6))
    khz = np.asarray(frequencies)[slope_bins] / 1000
    centered_khz = khz - np.mean(khz)
    slopes[valid] = (levels_db @ centered_khz) / np.sum(centered_khz**2)
    return slopes


def measure_trial(intervals, sr, task):
    """Return (feature -> (value, reason), audit); no frame crosses an interval join."""
    selected = [row for row in specs() if row["task_id"] == task]
    out = {row["feature_id"]: (np.nan, "insufficient_spectral_frames" if row["selectable"]
           else "research_pending_source_config") for row in selected}
    audit = {"native_sample_rate_hz": sr, "analysis_sample_rate_hz": FS,
             "resampler": "scipy.signal.resample_poly", "resampler_version": SCIPY_VERSION,
             "n_intervals": len(intervals), "n_frames": 0, "n_flux_pairs": 0}
    if sr <= 0 or task not in PREFIX:
        return {key: (np.nan, "invalid_analysis_audio") for key in out}, audit
    blocks = []
    sizes = []
    for interval in intervals:
        x = np.asarray(interval, dtype=float)
        if x.ndim != 1 or not np.isfinite(x).all():
            return {key: (np.nan, "invalid_analysis_audio") for key in out}, audit
        block = _frames(x, sr)
        if len(block):
            blocks.append(block)
            sizes.append(len(block))
    if sum(sizes) < MIN_FRAMES:
        return out, audit
    frames = np.concatenate(blocks)
    mag = np.abs(np.fft.rfft(frames * np.hanning(WIN), n=FFT, axis=1))
    power = mag**2
    valid = np.isfinite(power).all(axis=1) & (power.sum(axis=1) > EPS)
    if valid.sum() < MIN_FRAMES:
        return {key: (np.nan, "insufficient_nonzero_spectral_frames") for key in out}, audit
    audit["n_frames"] = int(valid.sum())
    key_prefix = PREFIX[task] + "_"
    def put(suffix, value, reason=""):
        key = key_prefix + suffix
        if key in out:
            out[key] = (float(value), reason)
    freq = np.fft.rfftfreq(FFT, 1/FS)
    good_mag, good_power = mag[valid], power[valid]
    m1, m2, m3, m4 = spectral_moments(good_power, freq)
    def median(vector):
        finite = np.asarray(vector)[np.isfinite(vector)]
        return (float(np.median(finite)), "") if len(finite) >= MIN_FRAMES else (
            np.nan, "insufficient_spectral_frames")
    for suffix, vector in (("spec_m1_hz", m1), ("spec_m2_hz", m2),
                           ("spec_m3_skew", m3), ("spec_m4_kurtosis", m4),
                           ("spectral_centroid_hz", m1), ("spectral_spread_hz", m2),
                           ("spectral_skew", m3), ("spectral_kurtosis", m4)):
        put(suffix, *median(vector))
    mag_weights = good_mag / good_mag.sum(axis=1, keepdims=True)
    centroid = mag_weights @ freq
    bandwidth = np.sqrt(np.sum(mag_weights * (freq[None, :] - centroid[:, None])**2, axis=1))
    put("spectral_bandwidth_p2_mean_hz", np.mean(bandwidth))
    put("spectral_bandwidth_p2_median_hz", np.median(bandwidth))
    mel = good_power @ _mel_filters(freq).T
    mfcc = dct(np.log(np.maximum(mel, EPS)), type=2, norm="ortho", axis=1)[:, :13]
    for i in range(13):
        put(f"mfcc{i+1:02d}_mean", np.mean(mfcc[:, i]))
        put(f"mfcc{i+1:02d}_sd", np.std(mfcc[:, i], ddof=1))
    if task in CONTRAST:
        try:
            import librosa
        except ImportError:
            for band in range(7):
                put(f"spectral_contrast_band{band}_db", np.nan, "librosa_unavailable")
        else:
            contrast = librosa.feature.spectral_contrast(
                S=good_mag.T, sr=FS, n_fft=FFT, fmin=200, n_bands=6,
                quantile=0.02, linear=False)
            for band in range(7):
                put(f"spectral_contrast_band{band}_db", np.mean(contrast[band]))
    dc = frames[valid] - frames[valid].mean(axis=1, keepdims=True)
    signs = dc >= 0
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
    put("zcr_mean_fraction", np.mean(zcr))
    put("zcr_median_fraction", np.median(zcr))
    slope = _spectral_slope_frames(good_mag, freq)
    put("spectral_slope_db_khz", *median(slope))
    put("spectral_crest", np.median(good_mag.max(axis=1)/good_mag.mean(axis=1)))
    flatness = np.exp(np.mean(np.log(np.maximum(good_power, EPS)), axis=1))/good_power.mean(axis=1)
    put("spectral_flatness", np.median(flatness))
    flux = []
    cursor = 0
    for size in sizes:
        piece = mag[cursor:cursor+size]
        good = valid[cursor:cursor+size]
        if len(piece) >= 2:
            unit = piece/np.maximum(np.linalg.norm(piece, axis=1, keepdims=True), EPS)
            pairs = good[1:] & good[:-1]
            flux.extend(np.linalg.norm(np.diff(unit, axis=0)[pairs], axis=1))
        cursor += size
    audit["n_flux_pairs"] = len(flux)
    put("spectral_flux", np.median(flux) if flux else np.nan,
        "" if flux else "insufficient_spectral_flux_frames")
    return out, audit
