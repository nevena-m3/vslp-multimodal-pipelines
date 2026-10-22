"""F13 VSLP operational nonlinear components; no source factor scores."""
from __future__ import annotations

from math import gcd
from importlib import metadata

import numpy as np
from scipy import __version__ as SCIPY_VERSION
from scipy.fft import dct
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

from vslp.acoustic.features.family01 import (
    MIN_SUSTAINED_YIELD, MIN_VALID_FRAMES, praat_f0_track,
)
from vslp.acoustic.features.family11 import natural_visibility_density
from vslp.acoustic.features.family12_v2 import FFT, FS, HOP, WIN, _frames, _mel_filters

VERSION = "family13-operational-1.0.0"
MIN_SENTENCES = 4
RQA_PROFILE = {"segment_sec": 1.0, "m": 3, "tau": 5, "eps": 0.2,
               "distance": "euclidean", "l_min": 2, "theiler": 1,
               "delta_half_window_frames": 2}
WPD_PROFILE = {"wavelet": "db4", "level": 3, "mode": "symmetric",
               "node_order": "freq"}
CRITICAL_BANDS = {"100_300": (100, 300), "300_800": (300, 800),
                  "1000_3000": (1000, 3000), "3000_8000": (3000, 8000)}
RHYTHMS = {"delta": (0.9, 2.5), "theta": (2.5, 12.0),
           "beta_gamma": (12.0, 40.0)}


def specs() -> list[dict]:
    rows = []
    def add(construct, task, feature_id, unit, name):
        rows.append({"construct_id": construct, "task_id": task, "feature_id": feature_id,
                     "human_name": name, "unit": unit, "selectable": True,
                     "blocked_reason": ""})
    add("C072", "sustained_a", "vowel_ppe_bits", "bits", "PPE — sustained /a/")
    for task, prefix in (("sustained_a", "vowel"), ("wstg", "s1")):
        add("C073", task, f"{prefix}_tsfel_shannon_entropy_norm", "ratio",
            f"TSFEL Shannon entropy — {prefix}")
    for node in range(8):
        add("C074", "bamboo_passage", f"bamboo_wpd_shannon_entropy_node{node}_norm",
            "ratio", f"WPD node {node} normalized Shannon entropy — Bamboo")
        for task, prefix in (("sustained_a", "vowel"), ("bamboo_passage", "bamboo")):
            add("C075", task, f"{prefix}_wpd_energy_share_node{node}", "ratio",
                f"WPD node {node} energy share — {prefix}")
    for i in range(1, 14):
        for task, prefix in (("sustained_a", "vowel"), ("bamboo_passage", "bamboo")):
            add("C076", task, f"{prefix}_mfcc{i:02d}_rqa_det", "ratio",
                f"MFCC C{i-1} RQA determinism — {prefix}")
        add("C077", "bamboo_passage", f"bamboo_dmfcc{i:02d}_rqa_det", "ratio",
            f"Delta MFCC C{i-1} RQA determinism — Bamboo")
    for rhythm in RHYTHMS:
        for band in CRITICAL_BANDS:
            add("C078", "bamboo_passage", f"bamboo_moddepth_{rhythm}_{band}", "ratio",
                f"Modulation depth {rhythm} {band} Hz — Bamboo")
    for coupling in ("delta_theta", "theta_beta_gamma"):
        for band in CRITICAL_BANDS:
            add("C079", "bamboo_passage", f"bamboo_psi_{coupling}_{band}", "ratio",
                f"PSI {coupling} {band} Hz — Bamboo")
    add("C079", "bamboo_passage", "bamboo_regularity_visibility_density_10ms",
        "ratio", "10 ms visibility density — Bamboo")
    assert len(rows) == 87
    return rows


PUBLIC_IDS = frozenset(row["feature_id"] for row in specs())


def analysis_copy(x: np.ndarray, native_sr: int) -> np.ndarray:
    """Resample one continuous interval, never a concatenation of intervals."""
    x = np.asarray(x, dtype=float)
    if native_sr <= 0 or x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("invalid_analysis_audio")
    if native_sr == FS:
        return x.copy()
    divisor = gcd(native_sr, FS)
    return resample_poly(x, FS // divisor, native_sr // divisor)


def ppe_from_f0(f0_hz: np.ndarray, *, tracking_yield: float = 1.0) -> tuple[float, str, dict]:
    """Order-2 covariance LPC residual, fixed 31-bin Shannon entropy."""
    values = np.asarray(f0_hz, dtype=float)
    voiced = values[np.isfinite(values) & (values > 0)]
    audit = {"n_voiced_f0": int(voiced.size), "n_residuals": 0, "n_in_range": 0,
             "fraction_out_of_range": np.nan, "lpc_order": 2}
    if tracking_yield < MIN_SUSTAINED_YIELD:
        return np.nan, "unreliable_f0", audit
    if voiced.size < max(50, MIN_VALID_FRAMES):
        return np.nan, "insufficient_voiced_f0", audit
    pitch = 12 * np.log2(voiced / (np.mean(voiced) / np.sqrt(2)))
    predictors = np.column_stack((pitch[1:-1], pitch[:-2]))
    try:
        coefficients = np.linalg.lstsq(predictors, pitch[2:], rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan, "lpc_failure", audit
    residual = pitch[2:] - predictors @ coefficients
    if not np.isfinite(residual).all():
        return np.nan, "lpc_failure", audit
    in_range = residual[(residual >= -1.5) & (residual <= 1.5)]
    audit.update(n_residuals=int(residual.size), n_in_range=int(in_range.size),
                 fraction_out_of_range=float(1 - in_range.size / residual.size),
                 lpc_phi1=float(coefficients[0]), lpc_phi2=float(coefficients[1]))
    if in_range.size < 20:
        return np.nan, "insufficient_ppe_residuals", audit
    counts, _ = np.histogram(in_range, bins=31, range=(-1.5, 1.5))
    probabilities = counts[counts > 0] / counts.sum()
    result = -float(np.sum(probabilities * np.log2(probabilities)))
    if not np.isfinite(result):
        return np.nan, "nonfinite_ppe", audit
    return result, "", audit


def ppe_from_audio(audio: np.ndarray, native_sr: int) -> tuple[float, str, dict]:
    try:
        track = praat_f0_track(audio, native_sr)
    except ValueError:
        return np.nan, "unreliable_f0", {}
    return ppe_from_f0(track.f0_hz[track.voiced_valid],
                       tracking_yield=track.tracking_yield)


def tsfel_entropy(audio: np.ndarray) -> tuple[float, str, dict]:
    try:
        import tsfel
    except ImportError:
        return np.nan, "tsfel_unavailable", {"tsfel_version": ""}
    try:
        result = float(tsfel.entropy(np.asarray(audio, dtype=float), prob="standard"))
    except (ValueError, TypeError, FloatingPointError):
        return np.nan, "tsfel_entropy_failed", {"tsfel_version": metadata.version("tsfel")}
    if not np.isfinite(result):
        return np.nan, "nonfinite_signal_entropy", {"tsfel_version": metadata.version("tsfel")}
    return result, "", {"tsfel_version": metadata.version("tsfel")}


def wpd_components(chunks: list[np.ndarray]) -> tuple[dict, str, dict]:
    """Combine per-interval terminal-node sufficient statistics without waveform joins."""
    try:
        import pywt
    except ImportError:
        return {}, "pywt_unavailable", dict(WPD_PROFILE)
    energies = np.zeros(8)
    energy_log_energy = np.zeros(8)
    counts = np.zeros(8, dtype=int)
    for chunk in chunks:
        if len(chunk) < 16 or not np.isfinite(chunk).all():
            continue
        packet = pywt.WaveletPacket(data=chunk, wavelet="db4", mode="symmetric", maxlevel=3)
        nodes = packet.get_level(3, order="freq")
        if len(nodes) != 8:
            return {}, "wpd_decomposition_failed", dict(WPD_PROFILE)
        for i, node in enumerate(nodes):
            squared = np.square(np.asarray(node.data, dtype=float))
            if not np.isfinite(squared).all():
                return {}, "wpd_decomposition_failed", dict(WPD_PROFILE)
            energies[i] += squared.sum()
            positive = squared[squared > 0]
            energy_log_energy[i] += np.sum(positive * np.log2(positive))
            counts[i] += len(squared)
    total = energies.sum()
    if not np.isfinite(total) or total <= 0:
        return {}, "zero_wavelet_energy", dict(WPD_PROFILE)
    shares = energies / total
    normalized = np.full(8, np.nan)
    raw_bits = np.full(8, np.nan)
    for i, energy in enumerate(energies):
        if energy > 0 and counts[i] > 1:
            raw_bits[i] = np.log2(energy) - energy_log_energy[i] / energy
            normalized[i] = np.clip(raw_bits[i] / np.log2(counts[i]), 0, 1)
    result = {"shares": shares, "entropy_norm": normalized, "entropy_bits": raw_bits,
              "raw_energy": energies, "coefficient_count": counts}
    return result, "", {**WPD_PROFILE, "pywt_version": pywt.__version__,
                        "pywt_distribution_version": metadata.version("PyWavelets")}


def mfcc_trajectory(audio_48k: np.ndarray) -> np.ndarray:
    """Exact F12 log-mel, orthonormal DCT-II C0..C12 trajectory."""
    frames = _frames(audio_48k, FS)
    if len(frames) == 0:
        return np.empty((0, 13))
    power = np.abs(np.fft.rfft(frames * np.hanning(WIN), n=FFT, axis=1)) ** 2
    freq = np.fft.rfftfreq(FFT, 1 / FS)
    mel = power @ _mel_filters(freq).T
    return dct(np.log(np.maximum(mel, np.finfo(float).tiny)),
               type=2, norm="ortho", axis=1)[:, :13]


def delta_mfcc(coefs: np.ndarray) -> np.ndarray:
    """Five-frame regression delta, edge-replicated, at the F12 10 ms frame rate."""
    padded = np.pad(coefs, ((2, 2), (0, 0)), mode="edge")
    return (padded[3:-1] - padded[1:-3] +
            2 * (padded[4:] - padded[:-4])) / 10


def rqa_determinism(values: np.ndarray) -> tuple[float, str]:
    """Euclidean RQA with z-scored scalar trajectory and excluded identity diagonal."""
    series = np.asarray(values, dtype=float)
    if len(series) < 3 + 2 * RQA_PROFILE["tau"] or not np.isfinite(series).all():
        return np.nan, "insufficient_rqa_frames"
    sd = np.std(series, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.nan, "zero_mfcc_variation"
    z = (series - np.mean(series)) / sd
    embedded = np.column_stack([z[i * 5:len(z) - (2 - i) * 5] for i in range(3)])
    differences = embedded[:, None, :] - embedded[None, :, :]
    recurrence = np.linalg.norm(differences, axis=2) <= 0.2
    np.fill_diagonal(recurrence, False)
    total = int(recurrence.sum())
    if total == 0:
        return np.nan, "no_recurrence_points"
    line_points = 0
    for offset in range(-len(embedded) + 1, len(embedded)):
        if offset == 0:
            continue
        diagonal = np.diagonal(recurrence, offset=offset)
        padded = np.r_[False, diagonal, False]
        starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
        ends = np.flatnonzero(np.diff(padded.astype(int)) == -1)
        line_points += int(np.sum((ends - starts)[ends - starts >= 2]))
    return line_points / total, ""


def rqa_components(chunks_48k: list[np.ndarray], *, delta: bool = False,
                   trajectories: list[np.ndarray] | None = None) -> tuple[np.ndarray, list[str], dict]:
    by_coefficient: list[list[float]] = [[] for _ in range(13)]
    if trajectories is None:
        trajectories = [mfcc_trajectory(chunk[start:start + FS])
                        for chunk in chunks_48k
                        for start in range(0, len(chunk) - FS + 1, FS)]
    for ordinary in trajectories:
        trajectory = delta_mfcc(ordinary) if delta else ordinary
        for i in range(13):
            value, reason = rqa_determinism(trajectory[:, i])
            if not reason:
                by_coefficient[i].append(value)
    values = np.array([np.median(items) if items else np.nan for items in by_coefficient])
    reasons = ["" if items else "no_valid_rqa_segment" for items in by_coefficient]
    return values, reasons, {"n_valid_segments_by_coefficient": [len(x) for x in by_coefficient],
                             **RQA_PROFILE, "mfcc_mapping": "mfcc01=C0,...,mfcc13=C12",
                             "frame_step_sec": HOP / FS}


def greenwood_edges() -> np.ndarray:
    """29 boundaries equally spaced on Greenwood cochlear position."""
    low = np.log10(100 / 165.4 + 1) / 2.1
    high = np.log10(10000 / 165.4 + 1) / 2.1
    positions = np.linspace(low, high, 29)
    edges = 165.4 * (10 ** (2.1 * positions) - 1)
    edges[0], edges[-1] = 100, 10000
    return edges


def _filter_envelope_chunks(chunks_48k: list[np.ndarray]) -> tuple[dict, str, dict]:
    edges = greenwood_edges()
    groups = {name: [] for name in CRITICAL_BANDS}
    centers = np.sqrt(edges[:-1] * edges[1:])
    smooth = butter(2, 40, btype="lowpass", fs=100, output="sos")
    for chunk in chunks_48k:
        if len(chunk) < FS // 2:
            continue
        narrow = []
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            sos = butter(4, (float(low), float(high)), btype="bandpass", fs=FS, output="sos")
            filtered = sosfiltfilt(sos, chunk)
            envelope = np.abs(hilbert(filtered))
            down = resample_poly(envelope, 1, 480)
            smoothed = sosfiltfilt(smooth, down)
            narrow.append(smoothed)
        narrow = np.asarray(narrow)
        # Remove 100 ms at each end after both zero-phase filters.
        if narrow.shape[1] <= 40:
            continue
        narrow = narrow[:, 10:-10]
        for name, (low, high) in CRITICAL_BANDS.items():
            indices = (centers >= low) & (centers < high)
            if not indices.any():
                return {}, "empty_critical_band", {"greenwood_edges_hz": edges.tolist()}
            groups[name].append(np.mean(narrow[indices], axis=0))
    if not all(groups.values()):
        return {}, "insufficient_rhythm_audio", {"greenwood_edges_hz": edges.tolist()}
    audit = {"greenwood_edges_hz": edges.tolist(), "narrow_bands": 28,
             "filter": "fourth-order zero-phase Butterworth SOS",
             "envelope_resampler": "scipy.signal.resample_poly 1/480",
             "smoothing": "second-order zero-phase Butterworth 40 Hz",
             "edge_trim_sec": 0.1, "critical_aggregation": "mean narrowband envelope"}
    return groups, "", audit


def modulation_depth(envelope: np.ndarray) -> tuple[dict, str]:
    series = np.asarray(envelope, dtype=float)
    if len(series) < 50 or not np.isfinite(series).all():
        return {}, "insufficient_modulation_audio"
    if len(series) > 2048:
        return {}, "modulation_sequence_exceeds_fft_support"
    spectrum = np.abs(np.fft.rfft(series * np.hamming(len(series)), n=2048)) ** 2
    freq = np.fft.rfftfreq(2048, 1 / 100)
    valid = freq > 0  # Exclude DC; retain one-sided Nyquist.
    denominator = float(spectrum[valid].sum())
    if denominator <= 0 or not np.isfinite(denominator):
        return {}, "zero_modulation_power"
    return {name: float(spectrum[(freq >= low) & (freq < high)].sum() / denominator)
            for name, (low, high) in RHYTHMS.items()}, ""


def phase_synchrony(envelope_chunks: list[np.ndarray]) -> tuple[dict, str]:
    accum = {"delta_theta": [], "theta_beta_gamma": []}
    for chunk in envelope_chunks:
        if len(chunk) < 50:
            continue
        phases = {}
        for name, (low, high) in RHYTHMS.items():
            sos = butter(4, (low, high), btype="bandpass", fs=100, output="sos")
            phase = np.angle(hilbert(sosfiltfilt(sos, chunk)))
            phases[name] = phase[10:-10]
        accum["delta_theta"].extend(np.exp(1j * (2 * phases["delta"] -
                                                   phases["theta"])))
        accum["theta_beta_gamma"].extend(np.exp(1j * (3 * phases["theta"] -
                                                      phases["beta_gamma"])))
    if not all(accum.values()):
        return {}, "insufficient_psi_audio"
    return {key: float(np.abs(np.mean(values))) for key, values in accum.items()}, ""


def rhythm_components(chunks_48k: list[np.ndarray]) -> tuple[dict, str, dict]:
    groups, reason, audit = _filter_envelope_chunks(chunks_48k)
    if reason:
        return {}, reason, audit
    result = {}
    for band, pieces in groups.items():
        mod, mod_reason = modulation_depth(np.concatenate(pieces))
        psi, psi_reason = phase_synchrony(pieces)
        if mod_reason or psi_reason:
            return {}, mod_reason or psi_reason, audit
        for rhythm, value in mod.items():
            result[f"bamboo_moddepth_{rhythm}_{band}"] = value
        for coupling, value in psi.items():
            result[f"bamboo_psi_{coupling}_{band}"] = value
    return result, "", audit


def visibility_10ms(chunks_48k: list[np.ndarray]) -> tuple[float, str, dict]:
    local = []
    for chunk in chunks_48k:
        n = len(chunk) // 480
        if n:
            local.extend(np.std(chunk[:n * 480].reshape(n, 480), axis=1, ddof=1))
    value = natural_visibility_density(np.asarray(local))
    if not np.isfinite(value):
        return np.nan, "insufficient_visibility_windows", {"n_windows": len(local)}
    return value, "", {"n_windows": len(local), "window_sec": 0.01,
                       "sd_ddof": 1, "graph": "natural_visibility"}
