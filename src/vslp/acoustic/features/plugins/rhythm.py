"""Rhythm and intensity acoustic feature plugin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from vslp.acoustic.features.plugins.audio_utils import read_mono_audio, rms_envelope
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


@dataclass(frozen=True)
class RhythmPlugin(AcousticFeaturePlugin):
    subsystem: str = "rhythm"
    feature_names: tuple[str, ...] = RHYTHM_FEATURES

    def compute(self, context: FeatureContext) -> dict[str, FeatureValue]:
        if context.segmentation_wav_path is None or not context.segmentation_wav_path.exists():
            return _nan_map("segmentation_wav_missing")
        x, sr = read_mono_audio(context.segmentation_wav_path)
        if x.size == 0:
            return _nan_map("empty_audio")

        t, rms = rms_envelope(x, sr, frame_ms=30.0, hop_ms=10.0)
        if rms.size < 16 or t.size < 16:
            return _nan_map("too_few_envelope_frames")
        rms = np.nan_to_num(rms, nan=0.0)
        voiced = rms[rms > max(np.percentile(rms, 20), 1e-8)]
        intensity_cv = float(np.std(voiced, ddof=0) / np.mean(voiced)) if voiced.size and np.mean(voiced) > 0 else np.nan

        env = rms - np.mean(rms)
        env_sr = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 100.0
        if len(env) >= 5:
            # Low-pass smooth envelope to reduce frame-level jitter before modulation spectrum.
            try:
                cutoff = min(20.0, 0.45 * env_sr)
                b, a = signal.butter(2, cutoff, btype="lowpass", fs=env_sr)
                env = signal.filtfilt(b, a, env)
            except Exception:
                pass
        spec = np.abs(np.fft.rfft(env))
        freqs = np.fft.rfftfreq(len(env), d=1.0 / env_sr)
        valid = (freqs >= 0.25) & (freqs <= 12.0)
        if not np.any(valid):
            return _nan_map("no_valid_modulation_frequencies")
        vf = freqs[valid]
        vs = spec[valid]
        peaks, props = signal.find_peaks(vs)
        if len(peaks) == 0:
            order = np.argsort(vs)[::-1][:2]
        else:
            order = peaks[np.argsort(vs[peaks])[::-1]][:2]
        peak_freqs = [float(vf[i]) for i in order]
        peak_amps = [float(vs[i]) for i in order]
        while len(peak_freqs) < 2:
            peak_freqs.append(np.nan)
            peak_amps.append(np.nan)

        power = vs ** 2
        total_power = float(np.sum(power)) if np.sum(power) > 0 else np.nan

        def band_energy(lo: float, hi: float) -> float:
            mask = (vf >= lo) & (vf < hi)
            if not np.any(mask) or not np.isfinite(total_power) or total_power <= 0:
                return np.nan
            return float(np.sum(power[mask]) / total_power)

        below = band_energy(0.25, 4.0)
        above = band_energy(4.0, 10.0)
        band36 = band_energy(3.0, 6.0)
        ratio = float(below / above) if np.isfinite(below) and np.isfinite(above) and above > 0 else np.nan
        note = "computed_from_rms_envelope_modulation_spectrum"
        return {
            "intensity_CV": FeatureValue("intensity_CV", intensity_cv, "computed", note),
            "fft_peaks1": FeatureValue("fft_peaks1", peak_freqs[0], "computed", note),
            "fft_peaks2": FeatureValue("fft_peaks2", peak_freqs[1], "computed", note),
            "fft_ampli1": FeatureValue("fft_ampli1", peak_amps[0], "computed", note),
            "fft_ampli2": FeatureValue("fft_ampli2", peak_amps[1], "computed", note),
            "nrj_below_boundary": FeatureValue("nrj_below_boundary", below, "computed", note),
            "nrj_above_boundary": FeatureValue("nrj_above_boundary", above, "computed", note),
            "nrj_3_6": FeatureValue("nrj_3_6", band36, "computed", note),
            "ratio_below_above": FeatureValue("ratio_below_above", ratio, "computed", note),
        }
