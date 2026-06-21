"""Validated-local resonatory / nasality acoustic feature plugin.

VSLP v0.30 implements the registered resonatory/nasality feature family using
an auditable single-microphone spectral method. These features are scientifically
fragile: A1-P0/A1-P1/A3-P0 depend on vowel identity, harmonic placement, F0,
recording bandwidth, spectral smoothing, and whether the analyzed region really
contains the intended oral vowel. Therefore, this plugin returns values with
explicit status notes and uses broad validity filters rather than silently
pretending every spectrum is interpretable.

Implemented features
--------------------
Raw nasality ratios:
- A1P0: F1/amplitude A1 minus low-frequency nasal pole P0 amplitude.
- A1P1: F1/amplitude A1 minus ~1 kHz nasal pole P1 amplitude.
- A3P0: F3/amplitude A3 minus P0 amplitude.

Support features:
- P0freq, P0amp, P0prom, P1amp.
- F1freq/F1amp/F1width, F2freq/F2amp/F2width, F3freq/F3amp/F3width.
- RMSamp.

Compensated variants:
- A1P0comp and A1P1comp are reported as conservative local proxies unless the
  exact Chen/Praat compensation path is later configured. The raw A1-P0/A1-P1
  values are the primary validated-local outputs. The compensated values are
  emitted with status ``computed_proxy`` and a note explaining that external
  reference validation remains required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from vslp.acoustic.features.plugins.audio_utils import frame_signal, read_region_audio
from vslp.acoustic.features.plugins.base import AcousticFeaturePlugin, FeatureContext, FeatureValue

RESONATORY_FEATURES: tuple[str, ...] = (
    "A1P0",
    "A1P0comp",
    "A1P1",
    "A1P1comp",
    "A3P0",
    "P0freq",
    "P0amp",
    "P0prom",
    "P1amp",
    "F1freq",
    "F1amp",
    "F1width",
    "F2freq",
    "F2amp",
    "F2width",
    "F3freq",
    "F3amp",
    "F3width",
    "RMSamp",
)

# Broad frequency windows for single-microphone spectral nasality screening.
# They are intentionally conservative and auditable. Exact vowel-targeted research
# analyses should later use task/vowel-specific time points or forced alignment.
DEFAULT_BANDS = {
    "P0": (180.0, 500.0),
    "P1": (790.0, 1100.0),
    "F1": (250.0, 1100.0),
    "F2": (800.0, 2600.0),
    "F3": (1800.0, 3600.0),
}


def _nan_feature(name: str, status: str, note: str) -> FeatureValue:
    return FeatureValue(name, np.nan, status, note)


def _all_nan(status: str, note: str) -> dict[str, FeatureValue]:
    return {name: _nan_feature(name, status, note) for name in RESONATORY_FEATURES}


def _next_power_two(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(16, n))))


def _smooth_db_spectrum(db: np.ndarray, hz_per_bin: float, smooth_hz: float) -> np.ndarray:
    if db.size == 0 or smooth_hz <= 0 or hz_per_bin <= 0:
        return db
    win = int(round(float(smooth_hz) / float(hz_per_bin)))
    win = max(3, win)
    if win % 2 == 0:
        win += 1
    if win >= db.size:
        win = db.size - 1 if db.size % 2 == 0 else db.size
    if win < 3:
        return db
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(db, kernel, mode="same")


def _frame_spectrum_db(frame: np.ndarray, sr: int, nfft: int, smooth_hz: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(frame, dtype=float)
    if x.size < 8 or np.max(np.abs(x)) < 1e-8:
        return np.array([]), np.array([])
    x = x - np.mean(x)
    x = x * np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    if len(freqs) > 1:
        db = _smooth_db_spectrum(db, hz_per_bin=float(freqs[1] - freqs[0]), smooth_hz=smooth_hz)
    return freqs, db


def _peak_in_band(freqs: np.ndarray, db: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    mask = (freqs >= lo) & (freqs <= hi) & np.isfinite(db)
    if not np.any(mask):
        return np.nan, np.nan
    idxs = np.flatnonzero(mask)
    sub = db[idxs]
    if sub.size == 0:
        return np.nan, np.nan
    i = idxs[int(np.nanargmax(sub))]
    return float(freqs[i]), float(db[i])


def _amplitude_near(freqs: np.ndarray, db: np.ndarray, center: float, half_width_hz: float) -> float:
    if not np.isfinite(center):
        return np.nan
    lo = max(0.0, center - half_width_hz)
    hi = center + half_width_hz
    _f, amp = _peak_in_band(freqs, db, lo, hi)
    return amp


def _prominence_in_band(freqs: np.ndarray, db: np.ndarray, peak_freq: float, peak_amp: float, lo: float, hi: float, exclude_hz: float) -> float:
    if not np.isfinite(peak_freq) or not np.isfinite(peak_amp):
        return np.nan
    mask = (freqs >= lo) & (freqs <= hi) & (np.abs(freqs - peak_freq) > exclude_hz) & np.isfinite(db)
    vals = db[mask]
    if vals.size < 3:
        return np.nan
    return float(peak_amp - np.median(vals))


def _half_prominence_width(freqs: np.ndarray, db: np.ndarray, peak_freq: float, peak_amp: float, lo: float, hi: float) -> float:
    """Approximate spectral peak width in Hz using a -3 dB crossing inside a band."""
    if not np.isfinite(peak_freq) or not np.isfinite(peak_amp):
        return np.nan
    mask = (freqs >= lo) & (freqs <= hi) & np.isfinite(db)
    if not np.any(mask):
        return np.nan
    f = freqs[mask]
    y = db[mask]
    if f.size < 5:
        return np.nan
    peak_idx = int(np.argmin(np.abs(f - peak_freq)))
    level = peak_amp - 3.0
    left_candidates = np.flatnonzero(y[: peak_idx + 1] <= level)
    right_candidates = np.flatnonzero(y[peak_idx:] <= level)
    if left_candidates.size == 0 or right_candidates.size == 0:
        return np.nan
    left = f[left_candidates[-1]]
    right = f[peak_idx + right_candidates[0]]
    if right <= left:
        return np.nan
    return float(right - left)


def _valid_energy_mask(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.array([], dtype=bool)
    rms = np.sqrt(np.mean(frames.astype(float) ** 2, axis=1))
    finite = rms[np.isfinite(rms)]
    if finite.size == 0:
        return np.zeros(frames.shape[0], dtype=bool)
    threshold = max(float(np.percentile(finite, 40)), float(np.median(finite)) * 0.20, 1e-5)
    return rms >= threshold


def _median(x: list[float] | np.ndarray) -> float:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else np.nan


def _robust_status(valid_fraction: float, min_valid_fraction: float, overlap_fraction: float) -> tuple[str, str]:
    if valid_fraction < min_valid_fraction:
        return "low_validity", f"valid_spectral_frame_fraction={valid_fraction:.3f} below minimum {min_valid_fraction:.3f}"
    if overlap_fraction > 0.35:
        return "computed_with_warnings", f"valid_spectral_frame_fraction={valid_fraction:.3f}; frequent_P0_F1_overlap_fraction={overlap_fraction:.3f}"
    return "computed", f"valid_spectral_frame_fraction={valid_fraction:.3f}; P0_F1_overlap_fraction={overlap_fraction:.3f}"


@dataclass(frozen=True)
class ResonatoryPlugin(AcousticFeaturePlugin):
    subsystem: str = "resonatory"
    feature_names: tuple[str, ...] = RESONATORY_FEATURES

    def compute(self, context: FeatureContext) -> dict[str, FeatureValue]:
        if context.segmentation_wav_path is None or not context.segmentation_wav_path.exists():
            return _all_nan("failed", "segmentation_wav_missing")

        cfg = context.config
        min_pause = float(getattr(cfg, "minimum_pause_duration_sec", 0.30))
        region = str(getattr(cfg, "resonatory_region_policy", "speech_only"))
        if region not in {"speech_only", "effective_task", "full_file"}:
            region = "speech_only"
        frame_ms = float(getattr(cfg, "resonatory_frame_ms", 40.0))
        hop_ms = float(getattr(cfg, "resonatory_hop_ms", 10.0))
        smooth_hz = float(getattr(cfg, "resonatory_spectral_smoothing_hz", 50.0))
        peak_half_width = float(getattr(cfg, "resonatory_peak_half_width_hz", 60.0))
        min_valid_fraction = float(getattr(cfg, "resonatory_min_valid_frame_fraction", 0.10))

        x, sr, region_note = read_region_audio(
            context.segmentation_wav_path,
            context.segments_csv,
            region=region,
            min_pause_duration_sec=min_pause,
        )
        if x.size == 0:
            return _all_nan("failed", "empty_audio")
        if x.size < int(0.10 * sr):
            return _all_nan("failed", f"audio_too_short_for_nasality; {region_note}")

        frames, frame_len, _hop_len = frame_signal(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
        if frames.size == 0:
            return _all_nan("failed", f"no_frames; {region_note}")
        energy_ok = _valid_energy_mask(frames)
        nfft = max(_next_power_two(frame_len * 4), 4096)

        per: dict[str, list[float]] = {k: [] for k in RESONATORY_FEATURES if k != "RMSamp"}
        valid_count = 0
        overlap_count = 0

        for idx, frame in enumerate(frames):
            if not energy_ok[idx]:
                continue
            freqs, db = _frame_spectrum_db(frame, sr=sr, nfft=nfft, smooth_hz=smooth_hz)
            if freqs.size == 0:
                continue

            p0freq, p0amp = _peak_in_band(freqs, db, *DEFAULT_BANDS["P0"])
            _p1freq, p1amp = _peak_in_band(freqs, db, *DEFAULT_BANDS["P1"])
            # Estimate F1 after P0 so the low nasal peak is not accidentally reused
            # as A1. This is a conservative default for sentence-level screening;
            # high-vowel /i/ remains intrinsically difficult and is flagged by the
            # P0/F1 overlap diagnostic when relevant.
            f1_lo, f1_hi = DEFAULT_BANDS["F1"]
            if np.isfinite(p0freq):
                f1_lo = max(f1_lo, float(p0freq) + 100.0)
            f1freq, _f1rough = _peak_in_band(freqs, db, f1_lo, f1_hi)
            if not np.isfinite(f1freq):
                f1freq, _f1rough = _peak_in_band(freqs, db, *DEFAULT_BANDS["F1"])
            f2freq, _f2rough = _peak_in_band(freqs, db, *DEFAULT_BANDS["F2"])
            f3freq, _f3rough = _peak_in_band(freqs, db, *DEFAULT_BANDS["F3"])
            if not (np.isfinite(p0amp) and np.isfinite(p1amp) and np.isfinite(f1freq) and np.isfinite(f3freq)):
                continue

            f1amp = _amplitude_near(freqs, db, f1freq, peak_half_width)
            f2amp = _amplitude_near(freqs, db, f2freq, peak_half_width)
            f3amp = _amplitude_near(freqs, db, f3freq, peak_half_width)
            if not (np.isfinite(f1amp) and np.isfinite(f3amp)):
                continue

            f1width = _half_prominence_width(freqs, db, f1freq, f1amp, f1_lo, DEFAULT_BANDS["F1"][1])
            f2width = _half_prominence_width(freqs, db, f2freq, f2amp, *DEFAULT_BANDS["F2"])
            f3width = _half_prominence_width(freqs, db, f3freq, f3amp, *DEFAULT_BANDS["F3"])
            p0prom = _prominence_in_band(freqs, db, p0freq, p0amp, *DEFAULT_BANDS["P0"], exclude_hz=45.0)

            a1p0 = f1amp - p0amp
            a1p1 = f1amp - p1amp
            a3p0 = f3amp - p0amp

            # Conservative local compensation proxies. Exact Chen/Praat compensation requires
            # vowel-targeted formant bandwidth/frequency path and should be externally validated.
            # We remove only a small within-frame F1-bandwidth penalty to avoid representing raw
            # values as fully compensated reference values.
            bw_penalty = 0.0
            if np.isfinite(f1width):
                bw_penalty = float(np.clip((f1width - 120.0) / 120.0, -1.0, 2.0))
            a1p0comp = a1p0 - bw_penalty
            a1p1comp = a1p1 - bw_penalty

            per["A1P0"].append(float(a1p0))
            per["A1P0comp"].append(float(a1p0comp))
            per["A1P1"].append(float(a1p1))
            per["A1P1comp"].append(float(a1p1comp))
            per["A3P0"].append(float(a3p0))
            per["P0freq"].append(float(p0freq))
            per["P0amp"].append(float(p0amp))
            per["P0prom"].append(float(p0prom))
            per["P1amp"].append(float(p1amp))
            per["F1freq"].append(float(f1freq))
            per["F1amp"].append(float(f1amp))
            per["F1width"].append(float(f1width))
            per["F2freq"].append(float(f2freq))
            per["F2amp"].append(float(f2amp))
            per["F2width"].append(float(f2width))
            per["F3freq"].append(float(f3freq))
            per["F3amp"].append(float(f3amp))
            per["F3width"].append(float(f3width))
            valid_count += 1
            if np.isfinite(p0freq) and np.isfinite(f1freq) and abs(p0freq - f1freq) < 100.0:
                overlap_count += 1

        total_frames = max(1, int(np.sum(energy_ok)))
        valid_fraction = float(valid_count / total_frames)
        overlap_fraction = float(overlap_count / max(1, valid_count))
        status, validity_note = _robust_status(valid_fraction, min_valid_fraction, overlap_fraction)
        note = (
            "validated_local_resonatory_v0.30; single_microphone_spectral_screening; "
            f"{region_note}; resonatory_region={region}; frame_ms={frame_ms}; hop_ms={hop_ms}; "
            f"spectral_smoothing_hz={smooth_hz}; peak_half_width_hz={peak_half_width}; "
            f"P0_band={DEFAULT_BANDS['P0']}; P1_band={DEFAULT_BANDS['P1']}; "
            f"{validity_note}; raw_A1P0_A1P1_A3P0_are_primary; compensated_variants_are_local_proxies_pending_reference_validation"
        )
        proxy_note = note + "; computed_proxy_compensation_not_praat_or_chen_identical"

        rmsamp = float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2))) if x.size else np.nan
        out: dict[str, FeatureValue] = {}
        for name in RESONATORY_FEATURES:
            if name == "RMSamp":
                out[name] = FeatureValue(name, rmsamp, status, note)
            elif name in {"A1P0comp", "A1P1comp"}:
                out[name] = FeatureValue(name, _median(per[name]), "computed_proxy" if status == "computed" else status, proxy_note)
            else:
                out[name] = FeatureValue(name, _median(per[name]), status, note)
        return out
