"""Validated rhythm / envelope-modulation acoustic feature plugin.

The rhythm block implements the envelope modulation spectrum (EMS) features used
for connected-speech rhythm analysis. The implementation is intentionally
conservative and auditable:

- rhythm analysis defaults to the effective task interval, not concatenated
  speech-only audio, because pauses and speech/pause alternation are part of
  connected-speech rhythm;
- the amplitude envelope is extracted after a speech-band prefilter
  (default 300--1000 Hz), following the common EMS convention used in dysarthria
  work;
- the modulation spectrum is computed over 0--10 Hz and split at the 4-Hz
  boundary used to separate slow phrase/stress-like modulation from faster
  syllabic modulation;
- energies are normalized by total 0--10 Hz modulation energy, so band features
  are proportions and comparable across amplitude scales.

These features are appropriate primarily for passages / connected speech. They
should not be interpreted as DDK rate or syllable-count features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from vslp.acoustic.features.plugins.audio_utils import read_region_audio, rms_envelope
from vslp.acoustic.features.plugins.base import AcousticFeaturePlugin, FeatureContext, FeatureValue

RHYTHM_FEATURES = (
    "intensity_CV",
    "fft_peaks1",
    "fft_peaks2",
    "fft_ampli1",
    "fft_ampli2",
    "nrj_below_boundary",
    "nrj_above_boundary",
    "nrj_3_6",
    "ratio_below_above",
)


def _nan_map(note: str, status: str = "failed") -> dict[str, FeatureValue]:
    return {name: FeatureValue(name, np.nan, status, note) for name in RHYTHM_FEATURES}


def _safe_float(obj, default: float) -> float:
    try:
        val = float(obj)
        return val if np.isfinite(val) else default
    except Exception:
        return default


def _bandpass_for_envelope(x: np.ndarray, sr: int, low_hz: float, high_hz: float) -> tuple[np.ndarray, str]:
    """Return speech-band filtered audio for envelope extraction.

    If the sampling rate cannot support the requested high cutoff, the cutoff is
    reduced safely. If filtering fails, the original signal is returned with an
    audit note so the stage does not silently crash.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 8 or sr <= 0:
        return x, "envelope_prefilter=skipped_short_or_invalid_audio"
    nyq = sr / 2.0
    lo = max(20.0, float(low_hz))
    hi = min(float(high_hz), 0.45 * sr)
    if hi <= lo or hi >= nyq:
        hi = min(max(lo + 50.0, 0.45 * sr), 0.95 * nyq)
    if hi <= lo or hi <= 0 or lo >= nyq:
        return x, "envelope_prefilter=skipped_cutoffs_not_supported_by_sr"
    try:
        sos = signal.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
        y = signal.sosfiltfilt(sos, x)
        return y.astype(float), f"envelope_prefilter=butter_bandpass_{lo:g}_{hi:g}_hz"
    except Exception as exc:  # noqa: BLE001
        return x, f"envelope_prefilter=failed_using_unfiltered_audio:{exc}"


def _hilbert_envelope_modulation(x: np.ndarray, sr: int, cfg) -> tuple[np.ndarray, float, str]:
    """Build a uniformly sampled amplitude envelope for EMS analysis."""
    low = _safe_float(getattr(cfg, "rhythm_envelope_bandpass_low_hz", 300.0), 300.0)
    high = _safe_float(getattr(cfg, "rhythm_envelope_bandpass_high_hz", 1000.0), 1000.0)
    target_sr = _safe_float(getattr(cfg, "rhythm_envelope_sample_rate_hz", 100.0), 100.0)
    target_sr = min(max(target_sr, 40.0), 200.0)

    y, filt_note = _bandpass_for_envelope(x, sr, low, high)
    if y.size < max(16, int(0.5 * sr)):
        return np.array([], dtype=float), target_sr, filt_note + "; too_short_for_ems"

    try:
        env = np.abs(signal.hilbert(y))
    except Exception:
        # Robust fallback; RMS envelope is less faithful to EMS convention but keeps
        # the stage usable and the note exposes the substitution.
        _t, env = rms_envelope(y, sr, frame_ms=30.0, hop_ms=10.0)
        env_sr = 100.0 if len(env) > 1 else target_sr
        return np.asarray(env, dtype=float), env_sr, filt_note + "; envelope=rms_fallback"

    env = np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
    if env.size == 0:
        return env, target_sr, filt_note + "; empty_envelope"
    env = env - np.nanmedian(env)
    if np.max(np.abs(env)) > 0:
        env = env / np.max(np.abs(env))

    n_target = max(8, int(round(len(env) * target_sr / sr)))
    try:
        env_ds = signal.resample_poly(env, up=int(target_sr), down=int(sr))
        if env_ds.size < n_target * 0.5 or env_ds.size > n_target * 2:
            env_ds = signal.resample(env, n_target)
    except Exception:
        env_ds = signal.resample(env, n_target)
    env_ds = np.nan_to_num(env_ds, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
    return env_ds, float(target_sr), filt_note + f"; envelope=hilbert; envelope_sr={target_sr:g}"


def _top_two_peaks(freqs: np.ndarray, spectrum: np.ndarray) -> tuple[list[float], list[float]]:
    if freqs.size == 0 or spectrum.size == 0:
        return [np.nan, np.nan], [np.nan, np.nan]
    peaks, _props = signal.find_peaks(spectrum)
    if len(peaks) > 0:
        order = peaks[np.argsort(spectrum[peaks])[::-1]][:2]
    else:
        order = np.argsort(spectrum)[::-1][:2]
    peak_freqs = [float(freqs[i]) for i in order]
    peak_amps = [float(spectrum[i]) for i in order]
    while len(peak_freqs) < 2:
        peak_freqs.append(np.nan)
        peak_amps.append(np.nan)
    return peak_freqs, peak_amps


def _band_energy(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float, denom: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask) or not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.sum(power[mask]) / denom)


def _speech_intensity_cv(context: FeatureContext, min_pause: float) -> tuple[float, str]:
    """Return sample CV of frame intensity over speech-only audio.

    WAV samples are interpreted using the conventional 20-uPa reference used for
    acoustic intensity. Consumer recordings are not pressure calibrated, so the
    result is suitable for within-protocol variability analysis, not absolute SPL.
    """
    x, sr, note = read_region_audio(
        context.segmentation_wav_path,
        context.segments_csv,
        region="speech_only",
        min_pause_duration_sec=min_pause,
    )
    if x.size == 0 or sr <= 0:
        return np.nan, note + "; intensity_cv_unavailable"
    _times, rms = rms_envelope(x, sr, frame_ms=25.0, hop_ms=10.0)
    rms = rms[np.isfinite(rms) & (rms > 0)]
    if rms.size < 2:
        return np.nan, note + "; intensity_cv_too_few_frames"
    reference_pressure_pa = 20e-6
    intensity_db = 20.0 * np.log10(np.maximum(rms, 1e-12) / reference_pressure_pa)
    mean_db = float(np.mean(intensity_db))
    if not np.isfinite(mean_db) or abs(mean_db) < 1e-12:
        return np.nan, note + "; intensity_cv_zero_mean_db"
    cv = float(np.std(intensity_db, ddof=1) / abs(mean_db))
    return cv, note + "; frames=25ms; hop=10ms; sample_sd; nominal_20uPa_reference; recording_not_spl_calibrated"


@dataclass(frozen=True)
class RhythmPlugin(AcousticFeaturePlugin):
    subsystem: str = "rhythm"
    feature_names: tuple[str, ...] = RHYTHM_FEATURES

    def compute(self, context: FeatureContext) -> dict[str, FeatureValue]:
        if context.segmentation_wav_path is None or not context.segmentation_wav_path.exists():
            return _nan_map("segmentation_wav_missing")

        cfg = context.config
        min_pause = _safe_float(getattr(cfg, "minimum_pause_duration_sec", 0.30), 0.30)
        rhythm_region = str(getattr(cfg, "rhythm_region_policy", "effective_task") or "effective_task")

        # Rhythm should preserve pauses inside the task; speech_only concatenation
        # destroys rhythmic timing. The user can override via rhythm_region_policy.
        x, sr, region_note = read_region_audio(
            context.segmentation_wav_path,
            context.segments_csv,
            region=rhythm_region,
            min_pause_duration_sec=min_pause,
        )
        if x.size == 0:
            return _nan_map("empty_audio; " + region_note)
        duration_sec = x.size / float(sr) if sr else np.nan
        if not np.isfinite(duration_sec) or duration_sec < 1.0:
            return _nan_map(f"too_short_for_rhythm_features; duration_sec={duration_sec}; {region_note}")

        intensity_cv, intensity_note = _speech_intensity_cv(context, min_pause)

        env, env_sr, env_note = _hilbert_envelope_modulation(x, sr, cfg)
        if env.size < 16:
            return _nan_map("too_few_envelope_samples; " + region_note + "; " + env_note)
        env = signal.detrend(env, type="constant")
        try:
            win = signal.windows.tukey(len(env), alpha=0.25)
        except Exception:
            win = np.hanning(len(env))
        env_win = env * win

        spec = np.abs(np.fft.rfft(env_win))
        freqs = np.fft.rfftfreq(len(env_win), d=1.0 / env_sr)
        valid = (freqs > 0.0) & (freqs <= 10.0)
        if not np.any(valid):
            return _nan_map("no_valid_0_to_10hz_modulation_frequencies; " + env_note)
        vf = freqs[valid]
        vs = spec[valid]
        power = vs ** 2
        denom = float(np.sum(power)) if np.sum(power) > 0 else np.nan

        # The specification defines the peak search over 0.5--10 Hz.
        peak_mask = vf >= 0.5
        peak_freqs, peak_amps_raw = _top_two_peaks(vf[peak_mask], vs[peak_mask])
        peak_amps = [float(a**2) if np.isfinite(a) else np.nan for a in peak_amps_raw]

        below = _band_energy(vf, power, 0.0, 4.0, denom)
        above = _band_energy(vf, power, 4.0, 10.0, denom)
        band36 = _band_energy(vf, power, 3.0, 6.0, denom)
        ratio = float(below / above) if np.isfinite(below) and np.isfinite(above) and above > 0 else np.nan

        note = (
            "validated_rhythm_v0.27; envelope_modulation_spectrum; "
            "modulation_band=0.5_to_10_hz_for_peaks; boundary=4_hz; "
            "energy_units=proportion_of_0_to_10hz_power_excluding_dc; peak_amplitude_units=relative_power; "
            f"{region_note}; {env_note}"
        )
        return {
            "intensity_CV": FeatureValue("intensity_CV", intensity_cv, "computed_with_warning", note + "; " + intensity_note),
            "fft_peaks1": FeatureValue("fft_peaks1", peak_freqs[0], "computed", note),
            "fft_peaks2": FeatureValue("fft_peaks2", peak_freqs[1], "computed", note),
            "fft_ampli1": FeatureValue("fft_ampli1", peak_amps[0], "computed", note),
            "fft_ampli2": FeatureValue("fft_ampli2", peak_amps[1], "computed", note),
            "nrj_below_boundary": FeatureValue("nrj_below_boundary", below, "computed", note),
            "nrj_above_boundary": FeatureValue("nrj_above_boundary", above, "computed", note),
            "nrj_3_6": FeatureValue("nrj_3_6", band36, "computed", note),
            "ratio_below_above": FeatureValue("ratio_below_above", ratio, "computed", note),
        }
