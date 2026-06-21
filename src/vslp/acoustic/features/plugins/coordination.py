"""Validated-local coordination acoustic feature plugin.

VSLP v0.31 implements the registered coordination features using an explicit
trajectory-coupling summary:

- build time-aligned frame trajectories for CPP, F1, and F2;
- robustly standardize each trajectory;
- compute a time-delay correlation matrix across small lags;
- summarize the eigenspectrum of that matrix as a participation-ratio complexity.

The resulting values are dimensionless coupling-complexity indices. They are not
intended to be diagnostic cutoffs. Low values indicate that the coupling matrix is
more dominated by a small number of eigenmodes; higher values indicate a broader
set of effective modes. Interpretation should remain comparative and task-specific.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from vslp.acoustic.features.plugins.audio_utils import frame_signal, read_region_audio
from vslp.acoustic.features.plugins.articulatory import _formants_from_frame, _resample_for_lpc, FORMANT_FREQ_RANGES
from vslp.acoustic.features.plugins.phonatory import _line_normalized_cpp
from vslp.acoustic.features.plugins.base import AcousticFeaturePlugin, FeatureContext, FeatureValue

COORDINATION_FEATURES = (
    "CPP_F1_comp",
    "CPP_F2_comp",
    "F1_F2_comp",
)


def _nan_feature(name: str, status: str, note: str) -> FeatureValue:
    return FeatureValue(name, np.nan, status, note)


def _all_nan(status: str, note: str) -> dict[str, FeatureValue]:
    return {name: _nan_feature(name, status, note) for name in COORDINATION_FEATURES}


def _robust_z(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    med = np.nanmedian(arr)
    q25 = np.nanpercentile(arr, 25)
    q75 = np.nanpercentile(arr, 75)
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr <= 1e-9:
        sd = np.nanstd(arr)
        if not np.isfinite(sd) or sd <= 1e-9:
            return np.full_like(arr, np.nan, dtype=float)
        return (arr - med) / sd
    return (arr - med) / (iqr / 1.349)


def _interpolate_missing(x: np.ndarray, max_gap_frames: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float).copy()
    n = arr.size
    valid = np.isfinite(arr)
    if valid.sum() < 3:
        return arr
    idx = np.arange(n)
    interp = np.interp(idx, idx[valid], arr[valid])
    # Do not fill long gaps: detect consecutive NaN runs and restore long gaps.
    missing = ~valid
    if missing.any():
        edges = np.diff(np.r_[False, missing, False].astype(int))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for s, e in zip(starts, ends, strict=False):
            if e - s > max_gap_frames:
                interp[s:e] = np.nan
    return interp


def _frame_energy_ok(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.array([], dtype=bool)
    rms = np.sqrt(np.mean(np.asarray(frames, dtype=float) ** 2, axis=1))
    finite = rms[np.isfinite(rms)]
    if finite.size == 0:
        return np.zeros(len(rms), dtype=bool)
    threshold = max(float(np.percentile(finite, 20)), float(np.median(finite)) * 0.10, 1e-5)
    return rms >= threshold


def _extract_cpp_f1_f2_tracks(
    x: np.ndarray,
    sr: int,
    frame_ms: float,
    hop_ms: float,
    lpc_target_sr: int,
    lpc_order: int | None,
    preemphasis: float,
    f0_min_hz: float,
    f0_max_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return time-aligned CPP/F1/F2 trajectories at LPC analysis sample rate."""
    y, y_sr = _resample_for_lpc(np.asarray(x, dtype=np.float32), int(sr), int(lpc_target_sr))
    if y.size < int(y_sr * 0.25):
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, "too_short_for_coordination_tracking"

    order = int(lpc_order) if lpc_order is not None else int(round(2 + y_sr / 1000.0))
    order = max(8, min(order, 18))
    frames, frame_len, hop_len = frame_signal(y, y_sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if frames.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, "no_frames"
    times = (np.arange(frames.shape[0]) * hop_len + frame_len / 2.0) / float(y_sr)
    energy_ok = _frame_energy_ok(frames)
    cpp = np.full(frames.shape[0], np.nan, dtype=float)
    f1 = np.full(frames.shape[0], np.nan, dtype=float)
    f2 = np.full(frames.shape[0], np.nan, dtype=float)

    for i, frame in enumerate(frames):
        if not energy_ok[i]:
            continue
        cpp[i] = _line_normalized_cpp(frame, y_sr, fmin=f0_min_hz, fmax=f0_max_hz)
        freqs, _bws = _formants_from_frame(frame, y_sr, order=order, preemphasis=preemphasis, max_formants=5)
        if freqs.size >= 1:
            lo, hi = FORMANT_FREQ_RANGES[1]
            if lo <= freqs[0] <= hi:
                f1[i] = float(freqs[0])
        if freqs.size >= 2:
            lo, hi = FORMANT_FREQ_RANGES[2]
            if lo <= freqs[1] <= hi:
                f2[i] = float(freqs[1])
    note = f"frames={len(times)}; lpc_sr={y_sr}; lpc_order={order}; frame_ms={frame_ms}; hop_ms={hop_ms}"
    return times, cpp, f1, f2, note


def _delay_embedding_pair(x: np.ndarray, y: np.ndarray, max_lag_frames: int) -> tuple[np.ndarray, str]:
    """Build a lagged correlation matrix for one trajectory pair.

    The matrix rows are x(t+lag) and y(t+lag) for lags -L..L. Correlations among
    these rows summarize how the two trajectories couple across short delays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size == 0:
        return np.empty((0, 0)), "empty_or_unequal_tracks"
    L = int(max(0, max_lag_frames))
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < max(8, 2 * L + 4):
        return np.empty((0, 0)), "insufficient_pairwise_valid_frames"
    # Build common central time support for all lags so matrix columns align.
    cols = []
    labels = []
    start = L
    end = len(x) - L
    if end <= start + 3:
        return np.empty((0, 0)), "insufficient_time_support_for_lags"
    for lag in range(-L, L + 1):
        idx = np.arange(start, end) + lag
        cols.append(x[idx])
        labels.append(f"x_lag{lag}")
    for lag in range(-L, L + 1):
        idx = np.arange(start, end) + lag
        cols.append(y[idx])
        labels.append(f"y_lag{lag}")
    M = np.vstack(cols)
    # Require complete columns across all lagged rows for correlation matrix.
    col_valid = np.isfinite(M).all(axis=0)
    if col_valid.sum() < max(8, M.shape[0] + 2):
        return np.empty((0, 0)), "insufficient_complete_lagged_columns"
    M = M[:, col_valid]
    # Remove rows with negligible variance.
    row_sd = np.std(M, axis=1)
    keep_rows = row_sd > 1e-9
    M = M[keep_rows]
    if M.shape[0] < 2 or M.shape[1] < M.shape[0] + 1:
        return np.empty((0, 0)), "insufficient_rank_after_variance_filter"
    C = np.corrcoef(M)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    return C, f"lagged_rows={M.shape[0]}; complete_frames={M.shape[1]}; max_lag_frames={L}"


def _eigenspectrum_complexity(C: np.ndarray) -> float:
    """Participation-ratio eigenspectrum complexity.

    PR = (sum(lambda))^2 / sum(lambda^2). It behaves like an effective number of
    eigenmodes. For a correlation matrix it ranges from about 1 to matrix rank.
    We normalize by matrix size so values are roughly 0..1 for comparability.
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] < 2:
        return np.nan
    vals = np.linalg.eigvalsh(C)
    vals = np.clip(vals, 0.0, None)
    denom = float(np.sum(vals ** 2))
    if denom <= 1e-12:
        return np.nan
    pr = float((np.sum(vals) ** 2) / denom)
    return float(pr / C.shape[0])


def _coordination_value(a: np.ndarray, b: np.ndarray, max_lag_frames: int) -> tuple[float, str]:
    a_filled = _interpolate_missing(a, max_gap_frames=max(1, int(max_lag_frames)))
    b_filled = _interpolate_missing(b, max_gap_frames=max(1, int(max_lag_frames)))
    az = _robust_z(a_filled)
    bz = _robust_z(b_filled)
    C, note = _delay_embedding_pair(az, bz, max_lag_frames=max_lag_frames)
    if C.size == 0:
        return np.nan, note
    return _eigenspectrum_complexity(C), note


@dataclass(frozen=True)
class CoordinationPlugin(AcousticFeaturePlugin):
    subsystem: str = "coordination"
    feature_names: tuple[str, ...] = COORDINATION_FEATURES

    def compute(self, context: FeatureContext) -> dict[str, FeatureValue]:
        if context.segmentation_wav_path is None or not context.segmentation_wav_path.exists():
            return _all_nan("failed", "segmentation_wav_missing")

        cfg = context.config
        region = getattr(cfg, "coordination_region_policy", "effective_task") if cfg is not None else "effective_task"
        min_pause = float(getattr(cfg, "minimum_pause_duration_sec", 0.30)) if cfg is not None else 0.30
        x, sr, region_note = read_region_audio(context.segmentation_wav_path, context.segments_csv, region=region, min_pause_duration_sec=min_pause)
        if x.size < int(max(sr, 1) * 0.5):
            return _all_nan("low_validity", f"selected_region_too_short; {region_note}")

        frame_ms = float(getattr(cfg, "coordination_frame_ms", 40.0)) if cfg is not None else 40.0
        hop_ms = float(getattr(cfg, "coordination_hop_ms", 10.0)) if cfg is not None else 10.0
        max_lag_ms = float(getattr(cfg, "coordination_max_lag_ms", 250.0)) if cfg is not None else 250.0
        max_lag_frames = max(1, int(round(max_lag_ms / hop_ms)))
        target_sr = int(getattr(cfg, "formant_lpc_target_sr_hz", 10000)) if cfg is not None else 10000
        preemphasis = float(getattr(cfg, "formant_preemphasis", 0.97)) if cfg is not None else 0.97
        lpc_order_raw = getattr(cfg, "formant_lpc_order", None) if cfg is not None else None
        lpc_order = int(lpc_order_raw) if lpc_order_raw not in (None, "", 0) else None
        f0_min = float(getattr(cfg, "phonatory_f0_min_hz", 60.0)) if cfg is not None else 60.0
        f0_max = float(getattr(cfg, "phonatory_f0_max_hz", 400.0)) if cfg is not None else 400.0

        times, cpp, f1, f2, track_note = _extract_cpp_f1_f2_tracks(
            x=x,
            sr=sr,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            lpc_target_sr=target_sr,
            lpc_order=lpc_order,
            preemphasis=preemphasis,
            f0_min_hz=f0_min,
            f0_max_hz=f0_max,
        )
        if times.size == 0:
            return _all_nan("low_validity", f"track_extraction_failed: {track_note}; {region_note}")

        valid_cpp = float(np.isfinite(cpp).mean()) if cpp.size else 0.0
        valid_f1 = float(np.isfinite(f1).mean()) if f1.size else 0.0
        valid_f2 = float(np.isfinite(f2).mean()) if f2.size else 0.0
        min_valid = float(getattr(cfg, "coordination_min_valid_fraction", 0.35)) if cfg is not None else 0.35
        base_note = f"{region_note}; {track_note}; valid_cpp={valid_cpp:.3f}; valid_f1={valid_f1:.3f}; valid_f2={valid_f2:.3f}; complexity=normalized_participation_ratio"

        outputs: dict[str, FeatureValue] = {}
        pairs = {
            "CPP_F1_comp": (cpp, f1),
            "CPP_F2_comp": (cpp, f2),
            "F1_F2_comp": (f1, f2),
        }
        for name, (a, b) in pairs.items():
            pair_valid = float((np.isfinite(a) & np.isfinite(b)).mean()) if a.size and b.size else 0.0
            value, note = _coordination_value(a, b, max_lag_frames=max_lag_frames)
            status = "computed" if np.isfinite(value) else "low_validity"
            if pair_valid < min_valid:
                status = "low_validity" if np.isfinite(value) else "failed"
            outputs[name] = FeatureValue(
                feature=name,
                value=value,
                status=status,
                note=f"{base_note}; pair_valid={pair_valid:.3f}; {note}",
            )
        return outputs
