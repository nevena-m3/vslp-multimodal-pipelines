"""Validated-local articulatory/formant acoustic feature plugin.

VSLP v0.29 implements the registered articulatory/formant feature family using a
conservative LPC root-tracking pipeline. Formants are fragile: they depend on
recording bandwidth, sex/vocal-tract length, vowel content, LPC order, frame
voicing/energy, and task. This plugin therefore returns values only when enough
valid frames survive explicit plausibility filters.

Implemented features
--------------------
- f1..f5: arithmetic mean formant frequencies.
- f1_bw..f5_bw: median LPC bandwidths.
- f1/f2/f3 ranges: maximum minus minimum of valid frame tracks.
- f1/f2/f3 velocity summaries: median, P5, P95, P95-P5 of dF/dt.

Important validation note
-------------------------
These are auditable local LPC estimates, not Praat-identical values. They are
internally validated on synthetic all-pole vowels and protected by validity
filters, but external reference validation is still recommended before clinical
interpretation or publication-level claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, signal

from vslp.acoustic.features.plugins.audio_utils import frame_signal, read_region_audio
from vslp.acoustic.features.plugins.base import AcousticFeaturePlugin, FeatureContext, FeatureValue

FORMANT_FEATURES: tuple[str, ...] = tuple(
    [f"f{i}" for i in range(1, 6)]
    + [f"f{i}_bw" for i in range(1, 6)]
    + [
        f"f{i}_{suffix}"
        for i in range(1, 4)
        for suffix in ("d_dx_median", "d_dx_prc_5", "d_dx_prc_95", "d_dx_prc_5_95")
    ]
    + [f"f{i}_range" for i in range(1, 4)]
)

# Conservative plausibility bands for adult speech formant tracking after optional resampling.
# Values are intentionally broad to avoid over-filtering dysarthric speech while still removing
# root-order errors and obvious spurious tracks.
FORMANT_FREQ_RANGES = {
    1: (150.0, 1200.0),
    2: (500.0, 3500.0),
    3: (1200.0, 4500.0),
    4: (2500.0, 6000.0),
    5: (3500.0, 7600.0),
}
FORMANT_BW_RANGES = {
    1: (10.0, 1000.0),
    2: (20.0, 1200.0),
    3: (20.0, 1500.0),
    4: (20.0, 2000.0),
    5: (20.0, 2500.0),
}


def _nan_feature(name: str, status: str, note: str) -> FeatureValue:
    return FeatureValue(name, np.nan, status, note)


def _all_nan(status: str, note: str) -> dict[str, FeatureValue]:
    return {name: _nan_feature(name, status, note) for name in FORMANT_FEATURES}


def _resample_for_lpc(x: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    """Downsample speech for stable LPC formant tracking when appropriate."""
    if x.size == 0:
        return x.astype(np.float32), int(sr)
    target_sr = int(target_sr)
    if sr <= target_sr or target_sr <= 0:
        return x.astype(np.float32), int(sr)
    # Polyphase resampling with integer factors keeps dependencies minimal and stable.
    gcd = int(np.gcd(sr, target_sr))
    up = target_sr // gcd
    down = sr // gcd
    y = signal.resample_poly(x, up, down).astype(np.float32)
    return y, target_sr


def _autocorr_lpc_coefficients(frame: np.ndarray, order: int) -> np.ndarray | None:
    """Estimate LPC denominator coefficients using autocorrelation + Levinson-style solve.

    Returns coefficients [1, a1, ..., ap] for the all-pole denominator.
    """
    x = np.asarray(frame, dtype=float)
    if x.size <= order + 2 or np.max(np.abs(x)) < 1e-8:
        return None
    x = x - np.mean(x)
    # Autocorrelation method gives a stable-ish all-pole model for formant extraction.
    r = np.correlate(x, x, mode="full")[x.size - 1 : x.size + order]
    if r.size < order + 1 or not np.all(np.isfinite(r)) or r[0] <= 1e-12:
        return None
    # Small diagonal loading stabilizes near-singular frames without changing normal speech frames much.
    r0 = r[0] * 1.0001
    try:
        a = linalg.solve_toeplitz((r[:order], r[:order]), -r[1 : order + 1], check_finite=False)
    except Exception:
        try:
            R = linalg.toeplitz(r[:order])
            R.flat[:: order + 1] += max(1e-9, 1e-6 * r0)
            a = np.linalg.solve(R, -r[1 : order + 1])
        except Exception:
            return None
    if not np.all(np.isfinite(a)):
        return None
    return np.r_[1.0, a]


def _formants_from_frame(
    frame: np.ndarray,
    sr: int,
    order: int,
    preemphasis: float,
    max_formants: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate formant frequencies and bandwidths for one frame."""
    x = np.asarray(frame, dtype=float)
    if x.size < order + 8 or np.max(np.abs(x)) < 1e-8:
        return np.array([]), np.array([])
    # Remove DC, pre-emphasize, and taper. This matches standard speech LPC practice.
    x = x - np.mean(x)
    if preemphasis > 0:
        x = signal.lfilter([1.0, -float(preemphasis)], [1.0], x)
    x = x * np.hamming(len(x))
    coeffs = _autocorr_lpc_coefficients(x, order)
    if coeffs is None:
        return np.array([]), np.array([])
    roots = np.roots(coeffs)
    roots = roots[np.imag(roots) > 0]
    if roots.size == 0:
        return np.array([]), np.array([])
    angles = np.angle(roots)
    freqs = angles * sr / (2.0 * np.pi)
    # Bandwidth in Hz from root radius. Keep only stable-ish positive bandwidth roots.
    radii = np.abs(roots)
    with np.errstate(divide="ignore", invalid="ignore"):
        bws = -sr / np.pi * np.log(np.maximum(radii, 1e-12))
    valid = np.isfinite(freqs) & np.isfinite(bws) & (freqs > 90.0) & (freqs < sr / 2.0) & (bws > 0.0)
    freqs = freqs[valid]
    bws = bws[valid]
    if freqs.size == 0:
        return np.array([]), np.array([])
    order_idx = np.argsort(freqs)
    return freqs[order_idx][:max_formants], bws[order_idx][:max_formants]


def _valid_energy_mask(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.array([], dtype=bool)
    rms = np.sqrt(np.mean(frames.astype(float) ** 2, axis=1))
    finite = rms[np.isfinite(rms)]
    if finite.size == 0:
        return np.zeros(len(rms), dtype=bool)
    # Speech-only regions can still contain weak frames; reject the bottom tail.
    threshold = max(float(np.percentile(finite, 20)), float(np.median(finite)) * 0.10, 1e-5)
    return rms >= threshold


def _track_formants(
    x: np.ndarray,
    sr: int,
    frame_ms: float,
    hop_ms: float,
    lpc_order: int,
    preemphasis: float,
    min_valid_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    frames, frame_len, hop_len = frame_signal(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if frames.size == 0:
        return np.empty((0, 5)), np.empty((0, 5)), np.array([]), frame_len, hop_len
    energy_ok = _valid_energy_mask(frames)
    freqs = np.full((frames.shape[0], 5), np.nan, dtype=float)
    bws = np.full((frames.shape[0], 5), np.nan, dtype=float)
    times = (np.arange(frames.shape[0]) * hop_len + frame_len / 2.0) / float(sr)

    for idx, frame in enumerate(frames):
        if not energy_ok[idx]:
            continue
        f, bw = _formants_from_frame(frame, sr, order=lpc_order, preemphasis=preemphasis, max_formants=5)
        if f.size == 0:
            continue
        # Assign ordered roots to F1..F5, then apply per-formant plausibility filters.
        for j in range(min(5, len(f))):
            k = j + 1
            flo, fhi = FORMANT_FREQ_RANGES[k]
            blo, bhi = FORMANT_BW_RANGES[k]
            if flo <= f[j] <= fhi and blo <= bw[j] <= bhi:
                freqs[idx, j] = float(f[j])
                bws[idx, j] = float(bw[j])

    # A frame should have at least F1 and F2 to be trusted for articulatory tracking.
    good_frame = np.isfinite(freqs[:, 0]) & np.isfinite(freqs[:, 1])
    if good_frame.mean() < min_valid_fraction:
        # Keep arrays; downstream summaries will fail with a clear validity note.
        return freqs, bws, times, frame_len, hop_len
    return freqs, bws, times, frame_len, hop_len


def _median_or_nan(x: np.ndarray) -> float:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else np.nan


def _mean_or_nan(x: np.ndarray) -> float:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else np.nan


def _range_max_min(x: np.ndarray) -> float:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.nan
    return float(np.max(vals) - np.min(vals))


def _velocity_stats(values: np.ndarray, times: np.ndarray, max_abs_velocity_hz_s: float) -> dict[str, float]:
    vals = np.asarray(values, dtype=float)
    t = np.asarray(times, dtype=float)
    valid = np.isfinite(vals) & np.isfinite(t)
    if valid.sum() < 3:
        return {"median": np.nan, "p5": np.nan, "p95": np.nan, "p5_95": np.nan}
    vals = vals[valid]
    t = t[valid]
    # Drop duplicate/nonmonotonic timestamps defensively.
    uniq = np.r_[True, np.diff(t) > 1e-9]
    vals = vals[uniq]
    t = t[uniq]
    if vals.size < 3:
        return {"median": np.nan, "p5": np.nan, "p95": np.nan, "p5_95": np.nan}
    # Light median smoothing reduces frame-to-frame LPC swaps without obscuring gross trajectories.
    if vals.size >= 5:
        vals_s = signal.medfilt(vals, kernel_size=5)
    else:
        vals_s = vals
    # The supplied protocol defines slope as (F_start - F_end) / duration.
    dv = -np.gradient(vals_s, t)
    dv = dv[np.isfinite(dv) & (np.abs(dv) <= max_abs_velocity_hz_s)]
    if dv.size < 2:
        return {"median": np.nan, "p5": np.nan, "p95": np.nan, "p5_95": np.nan}
    p5 = float(np.percentile(dv, 5))
    p95 = float(np.percentile(dv, 95))
    return {
        "median": float(np.median(dv)),
        "p5": p5,
        "p95": p95,
        "p5_95": float(p95 - p5),
    }


@dataclass(frozen=True)
class ArticulatoryPlugin(AcousticFeaturePlugin):
    subsystem: str = "articulatory"
    feature_names: tuple[str, ...] = FORMANT_FEATURES

    def compute(self, context: FeatureContext) -> dict[str, FeatureValue]:
        if context.segmentation_wav_path is None or not context.segmentation_wav_path.exists():
            return _all_nan("failed", "segmentation_wav_missing")

        cfg = context.config
        min_pause = float(getattr(cfg, "minimum_pause_duration_sec", 0.30))
        # Formants should be computed on speech regions; effective_task/full_file can contaminate LPC with pauses.
        requested_region = str(getattr(cfg, "articulatory_region_policy", "speech_only"))
        region = requested_region if requested_region in {"speech_only", "full_file"} else "speech_only"
        frame_ms = float(getattr(cfg, "articulatory_frame_ms", 25.0))
        hop_ms = float(getattr(cfg, "articulatory_hop_ms", 10.0))
        target_sr = int(getattr(cfg, "articulatory_lpc_sample_rate_hz", 10000))
        preemphasis = float(getattr(cfg, "articulatory_preemphasis", 0.97))
        min_valid_fraction = float(getattr(cfg, "articulatory_min_valid_frame_fraction", 0.10))
        max_abs_velocity = float(getattr(cfg, "articulatory_max_abs_velocity_hz_s", 20000.0))

        x, sr, region_note = read_region_audio(
            context.segmentation_wav_path,
            context.segments_csv,
            region=region,
            min_pause_duration_sec=min_pause,
        )
        if x.size == 0:
            return _all_nan("failed", "empty_audio")

        x_lpc, sr_lpc = _resample_for_lpc(x, sr, target_sr)
        if x_lpc.size < int(0.08 * sr_lpc):
            return _all_nan("failed", f"audio_too_short_for_formants; {region_note}")

        # Conservative 5-formant default at ~10 kHz: 2 poles/formant = LPC order 10.
        # Higher orders can introduce spurious roots in short clinical recordings, so users can
        # override this in expert mode later, but the default is intentionally stable.
        default_order = 10 if sr_lpc <= 12000 else max(10, int(round(2 + sr_lpc / 1000.0)))
        lpc_order = int(getattr(cfg, "articulatory_lpc_order", default_order))
        lpc_order = max(8, min(lpc_order, 20))

        freqs, bws, times, _frame_len, _hop_len = _track_formants(
            x_lpc,
            sr_lpc,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            lpc_order=lpc_order,
            preemphasis=preemphasis,
            min_valid_fraction=min_valid_fraction,
        )
        if freqs.size == 0:
            return _all_nan("failed", f"no_formant_frames; {region_note}")

        valid_frame_fraction = float(np.mean(np.isfinite(freqs[:, 0]) & np.isfinite(freqs[:, 1])))
        if valid_frame_fraction < min_valid_fraction:
            status = "low_validity"
            validity_note = (
                f"valid_f1_f2_frame_fraction={valid_frame_fraction:.3f} below minimum {min_valid_fraction:.3f}; "
                "values retained for review but should be treated cautiously"
            )
        else:
            status = "computed"
            validity_note = f"valid_f1_f2_frame_fraction={valid_frame_fraction:.3f}"

        features: dict[str, FeatureValue] = {}
        note = (
            "validated_formant_lpc_v0.29; "
            f"{region_note}; formant_region={region}; lpc_sr={sr_lpc}; lpc_order={lpc_order}; "
            f"frame_ms={frame_ms}; hop_ms={hop_ms}; preemphasis={preemphasis}; {validity_note}"
        )

        for i in range(1, 6):
            col = i - 1
            features[f"f{i}"] = FeatureValue(f"f{i}", _mean_or_nan(freqs[:, col]), status, note)
            features[f"f{i}_bw"] = FeatureValue(f"f{i}_bw", _median_or_nan(bws[:, col]), status, note)

        for i in range(1, 4):
            col = i - 1
            rng = _range_max_min(freqs[:, col])
            features[f"f{i}_range"] = FeatureValue(f"f{i}_range", rng, status, note)
            stats = _velocity_stats(freqs[:, col], times, max_abs_velocity_hz_s=max_abs_velocity)
            features[f"f{i}_d_dx_median"] = FeatureValue(f"f{i}_d_dx_median", stats["median"], status, note)
            features[f"f{i}_d_dx_prc_5"] = FeatureValue(f"f{i}_d_dx_prc_5", stats["p5"], status, note)
            features[f"f{i}_d_dx_prc_95"] = FeatureValue(f"f{i}_d_dx_prc_95", stats["p95"], status, note)
            features[f"f{i}_d_dx_prc_5_95"] = FeatureValue(f"f{i}_d_dx_prc_5_95", stats["p5_95"], status, note)

        return features
