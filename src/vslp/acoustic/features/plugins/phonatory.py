"""Validated-local phonatory acoustic feature plugin.

VSLP v0.28 implements the complete registered phonatory feature group with
transparent local signal processing:
- F0 track from normalized autocorrelation;
- HNR from autocorrelation peak ratio;
- CPP from line-normalized cepstral peak prominence;
- jitter/shimmer perturbation families from voiced-frame period/amplitude tracks;
- voice-break counts from internal unvoiced runs;
- H1/H2 harmonic frequency/amplitude estimates from voiced-frame spectra.

Important validation note
-------------------------
These formulas are implemented explicitly and tested on synthetic signals, but perturbation
features are not bit-for-bit Praat/MDVP measures because VSLP does not yet perform glottal
cycle boundary marking. They should be treated as local, auditable implementations pending
external reference validation before clinical interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import signal

from vslp.acoustic.features.plugins.audio_utils import frame_signal, read_region_audio
from vslp.acoustic.features.plugins.base import AcousticFeaturePlugin, FeatureContext, FeatureValue

PHONATORY_FEATURES = (
    "f0_mean",
    "f0_std",
    "CPP_mean",
    "HNR",
    "localJitter",
    "localabsoluteJitter",
    "rapJitter",
    "ppq5Jitter",
    "ddpJitter",
    "localShimmer",
    "localdbShimmer",
    "apq3Shimmer",
    "apq5Shimmer",
    "apq11Shimmer",
    "num_voicebreaks",
    "H1freq",
    "H1amp",
    "H2freq",
    "H2amp",
)


def _nan_feature(name: str, status: str, note: str) -> FeatureValue:
    return FeatureValue(name, np.nan, status, note)


def _safe_mean(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def _safe_std(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=0)) if arr.size else np.nan


def _normalized_autocorr(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=float)
    if frame.size < 8:
        return np.array([], dtype=float)
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-8:
        return np.array([], dtype=float)
    frame = frame * np.hanning(len(frame))
    corr = signal.correlate(frame, frame, mode="full", method="fft")
    corr = corr[len(corr) // 2 :]
    if corr.size == 0 or corr[0] <= 1e-12:
        return np.array([], dtype=float)
    return corr / corr[0]


def _frame_f0_hnr(frame: np.ndarray, sr: int, fmin: float, fmax: float, min_peak_ratio: float) -> tuple[float, float, float]:
    corr = _normalized_autocorr(frame)
    if corr.size == 0:
        return np.nan, np.nan, np.nan
    min_lag = max(1, int(np.floor(sr / fmax)))
    max_lag = min(len(corr) - 1, int(np.ceil(sr / fmin)))
    if max_lag <= min_lag:
        return np.nan, np.nan, np.nan
    search = corr[min_lag : max_lag + 1]
    if search.size == 0:
        return np.nan, np.nan, np.nan
    lag = min_lag + int(np.argmax(search))
    r = float(corr[lag])
    if not np.isfinite(r) or r < min_peak_ratio:
        return np.nan, np.nan, r
    f0 = float(sr / lag)
    # Boersma-style harmonicity approximation from normalized autocorrelation strength.
    r_clip = min(max(r, 1e-6), 0.999999)
    hnr = float(10.0 * np.log10(r_clip / (1.0 - r_clip)))
    return f0, hnr, r


def _line_normalized_cpp(frame: np.ndarray, sr: int, fmin: float, fmax: float) -> float:
    frame = np.asarray(frame, dtype=float)
    if frame.size < 64 or np.max(np.abs(frame)) < 1e-8:
        return np.nan
    frame = (frame - np.mean(frame)) * np.hanning(len(frame))
    nfft = int(2 ** np.ceil(np.log2(max(len(frame), 1024))))
    spec = np.abs(np.fft.rfft(frame, n=nfft)) + 1e-12
    log_power = np.log(spec**2)
    cep = np.fft.irfft(log_power, n=nfft)
    qmin = max(1, int(np.floor(sr / fmax)))
    qmax = min(len(cep) - 1, int(np.ceil(sr / fmin)))
    if qmax <= qmin + 3:
        return np.nan
    q = np.arange(qmin, qmax + 1, dtype=float) / sr
    y = cep[qmin : qmax + 1]
    if y.size < 4 or not np.isfinite(y).all():
        return np.nan
    # Linear baseline in the quefrency search region; CPP is peak height above the line.
    coef = np.polyfit(q, y, 1)
    baseline = np.polyval(coef, q)
    prominence = float(np.max(y - baseline))
    # Use dB-like scaling relative to cepstral baseline magnitude; stable for synthetic + real speech.
    return float(20.0 * np.log10(max(prominence, 1e-12)))


def _harmonic_estimates(frame: np.ndarray, sr: int, f0: float) -> tuple[float, float, float, float]:
    if not np.isfinite(f0) or f0 <= 0 or frame.size < 64:
        return np.nan, np.nan, np.nan, np.nan
    frame = np.asarray(frame, dtype=float)
    frame = (frame - np.mean(frame)) * np.hanning(len(frame))
    nfft = int(2 ** np.ceil(np.log2(max(len(frame), 2048))))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    spec_db = 20.0 * np.log10(np.abs(np.fft.rfft(frame, n=nfft)) + 1e-12)

    def _peak_near(target: float) -> tuple[float, float]:
        if target <= 0 or target >= sr / 2:
            return np.nan, np.nan
        half_width = max(15.0, target * 0.08)
        mask = (freqs >= target - half_width) & (freqs <= target + half_width)
        if not mask.any():
            return np.nan, np.nan
        idxs = np.flatnonzero(mask)
        idx = idxs[int(np.argmax(spec_db[mask]))]
        return float(freqs[idx]), float(spec_db[idx])

    h1f, h1a = _peak_near(f0)
    h2f, h2a = _peak_near(2.0 * f0)
    return h1f, h1a, h2f, h2a


def _local_jitter(periods: np.ndarray) -> tuple[float, float]:
    periods = periods[np.isfinite(periods) & (periods > 0)]
    if periods.size < 2:
        return np.nan, np.nan
    diffs = np.abs(np.diff(periods))
    mean_period = float(np.mean(periods))
    if mean_period <= 0:
        return np.nan, np.nan
    return float(np.mean(diffs) / mean_period * 100.0), float(np.mean(diffs))


def _period_perturbation(periods: np.ndarray, width: int) -> float:
    periods = periods[np.isfinite(periods) & (periods > 0)]
    if periods.size < width:
        return np.nan
    half = width // 2
    vals = []
    for i in range(half, periods.size - half):
        local = periods[i - half : i + half + 1]
        vals.append(abs(periods[i] - np.mean(local)))
    if not vals:
        return np.nan
    mean_period = float(np.mean(periods))
    return float(np.mean(vals) / mean_period * 100.0) if mean_period > 0 else np.nan


def _local_shimmer(amps: np.ndarray) -> tuple[float, float]:
    amps = amps[np.isfinite(amps) & (amps > 1e-12)]
    if amps.size < 2:
        return np.nan, np.nan
    diffs = np.abs(np.diff(amps))
    mean_amp = float(np.mean(amps))
    local_pct = float(np.mean(diffs) / mean_amp * 100.0) if mean_amp > 0 else np.nan
    local_db = float(np.mean(np.abs(20.0 * np.log10(amps[1:] / amps[:-1]))))
    return local_pct, local_db


def _amplitude_perturbation(amps: np.ndarray, width: int) -> float:
    amps = amps[np.isfinite(amps) & (amps > 1e-12)]
    if amps.size < width:
        return np.nan
    half = width // 2
    vals = []
    for i in range(half, amps.size - half):
        local = amps[i - half : i + half + 1]
        vals.append(abs(amps[i] - np.mean(local)))
    if not vals:
        return np.nan
    mean_amp = float(np.mean(amps))
    return float(np.mean(vals) / mean_amp * 100.0) if mean_amp > 0 else np.nan


def _count_voice_breaks(voiced: np.ndarray, hop_sec: float, min_break_sec: float) -> int:
    voiced = np.asarray(voiced, dtype=bool)
    if voiced.size == 0 or voiced.sum() == 0:
        return 0
    # Only count unvoiced runs between first and last voiced frame.
    first = int(np.argmax(voiced))
    last = len(voiced) - 1 - int(np.argmax(voiced[::-1]))
    if last <= first:
        return 0
    internal = ~voiced[first : last + 1]
    if not internal.any():
        return 0
    edges = np.diff(np.r_[False, internal, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int(sum((e - s) * hop_sec >= min_break_sec for s, e in zip(starts, ends, strict=False)))


@dataclass(frozen=True)
class PhonatoryPlugin(AcousticFeaturePlugin):
    subsystem: str = "phonatory"
    feature_names: tuple[str, ...] = PHONATORY_FEATURES

    def compute(self, context: FeatureContext) -> dict[str, FeatureValue]:
        if context.segmentation_wav_path is None or not context.segmentation_wav_path.exists():
            return {name: _nan_feature(name, "failed", "segmentation_wav_missing") for name in self.feature_names}

        min_pause = float(getattr(context.config, "minimum_pause_duration_sec", 0.15))
        fmin = float(getattr(context.config, "phonatory_f0_min_hz", 60.0))
        fmax = float(getattr(context.config, "phonatory_f0_max_hz", 400.0))
        min_peak_ratio = float(getattr(context.config, "phonatory_min_autocorr_peak", 0.30))
        frame_ms = float(getattr(context.config, "phonatory_frame_ms", 40.0))
        hop_ms = float(getattr(context.config, "phonatory_hop_ms", 10.0))
        min_voice_break = float(getattr(context.config, "phonatory_min_voice_break_sec", 0.10))

        x, sr, region_note = read_region_audio(
            context.segmentation_wav_path,
            context.segments_csv,
            region=context.analysis_region,
            min_pause_duration_sec=min_pause,
        )
        if x.size == 0:
            return {name: _nan_feature(name, "failed", "empty_audio") for name in self.feature_names}

        frames, _frame_len, hop_len = frame_signal(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
        if frames.size == 0:
            return {name: _nan_feature(name, "failed", "empty_frames") for name in self.feature_names}
        hop_sec = hop_len / float(sr)

        f0_vals: list[float] = []
        hnr_vals: list[float] = []
        cpp_vals: list[float] = []
        amp_vals: list[float] = []
        h1f_vals: list[float] = []
        h1a_vals: list[float] = []
        h2f_vals: list[float] = []
        h2a_vals: list[float] = []
        voiced_flags: list[bool] = []

        for frame in frames:
            f0, hnr, r = _frame_f0_hnr(frame, sr, fmin=fmin, fmax=fmax, min_peak_ratio=min_peak_ratio)
            voiced = bool(np.isfinite(f0))
            voiced_flags.append(voiced)
            if not voiced:
                continue
            amp = float(np.sqrt(np.mean(np.asarray(frame, dtype=float) ** 2)))
            cpp = _line_normalized_cpp(frame, sr, fmin=fmin, fmax=fmax)
            h1f, h1a, h2f, h2a = _harmonic_estimates(frame, sr, f0)
            f0_vals.append(float(f0))
            if np.isfinite(hnr):
                hnr_vals.append(float(hnr))
            if np.isfinite(cpp):
                cpp_vals.append(float(cpp))
            if np.isfinite(amp) and amp > 0:
                amp_vals.append(amp)
            if np.isfinite(h1f):
                h1f_vals.append(h1f)
            if np.isfinite(h1a):
                h1a_vals.append(h1a)
            if np.isfinite(h2f):
                h2f_vals.append(h2f)
            if np.isfinite(h2a):
                h2a_vals.append(h2a)

        f0_arr = np.asarray(f0_vals, dtype=float)
        periods = 1.0 / f0_arr[np.isfinite(f0_arr) & (f0_arr > 0)]
        amps = np.asarray(amp_vals, dtype=float)
        voiced_arr = np.asarray(voiced_flags, dtype=bool)

        local_jitter, abs_jitter = _local_jitter(periods)
        rap = _period_perturbation(periods, 3)
        ppq5 = _period_perturbation(periods, 5)
        ddp = float(3.0 * rap) if np.isfinite(rap) else np.nan
        local_shim, local_db_shim = _local_shimmer(amps)
        apq3 = _amplitude_perturbation(amps, 3)
        apq5 = _amplitude_perturbation(amps, 5)
        apq11 = _amplitude_perturbation(amps, 11)
        voicebreaks = _count_voice_breaks(voiced_arr, hop_sec=hop_sec, min_break_sec=min_voice_break)

        voice_fraction = float(np.mean(voiced_arr)) if voiced_arr.size else 0.0
        note = (
            "validated_local_phonatory_v0.28; frame_autocorrelation_f0; line_normalized_cpp; "
            "perturbation_from_voiced_frame_period_and_rms_tracks; external_praat_mdvp_validation_recommended; "
            f"voiced_frames={int(np.sum(voiced_arr))}/{len(voiced_arr)}; voice_fraction={voice_fraction:.3f}; "
            f"f0_range={fmin:.1f}-{fmax:.1f}Hz; {region_note}"
        )
        low_voicing_note = note + "; warning_low_voiced_frame_count" if f0_arr.size < 5 else note
        status = "computed" if f0_arr.size >= 5 else "computed_with_warning"

        values = {
            "f0_mean": _safe_mean(f0_arr),
            "f0_std": _safe_std(f0_arr),
            "CPP_mean": _safe_mean(cpp_vals),
            "HNR": _safe_mean(hnr_vals),
            "localJitter": local_jitter,
            "localabsoluteJitter": abs_jitter,
            "rapJitter": rap,
            "ppq5Jitter": ppq5,
            "ddpJitter": ddp,
            "localShimmer": local_shim,
            "localdbShimmer": local_db_shim,
            "apq3Shimmer": apq3,
            "apq5Shimmer": apq5,
            "apq11Shimmer": apq11,
            "num_voicebreaks": float(voicebreaks),
            "H1freq": _safe_mean(h1f_vals),
            "H1amp": _safe_mean(h1a_vals),
            "H2freq": _safe_mean(h2f_vals),
            "H2amp": _safe_mean(h2a_vals),
        }
        return {name: FeatureValue(name, value, status, low_voicing_note if status != "computed" else note) for name, value in values.items()}
