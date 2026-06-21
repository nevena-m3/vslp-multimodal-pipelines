"""Task-scoped diadochokinetic rate and regularity features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from vslp.acoustic.features.plugins.audio_utils import read_region_audio, rms_envelope
from vslp.acoustic.features.plugins.base import AcousticFeaturePlugin, FeatureContext, FeatureValue

DDK_FEATURES = ("DDKrate", "DDKregularity")
DDK_TASK_TOKENS = ("ddk", "amr", "smr", "pataka", "pa_ta_ka", "papapa", "tatata", "kakaka")


def _is_ddk_task(task: object) -> bool:
    normalized = str(task or "").strip().lower().replace("-", "_").replace(" ", "_")
    return any(token in normalized for token in DDK_TASK_TOKENS)


def _detect_syllable_nuclei(x: np.ndarray, sr: int) -> tuple[np.ndarray, str]:
    """Detect envelope peaks using fixed, physiologically plausible constraints."""
    if x.size < int(0.5 * sr):
        return np.array([], dtype=float), "ddk_audio_too_short"
    try:
        sos = signal.butter(4, [80.0, min(3000.0, 0.45 * sr)], btype="bandpass", fs=sr, output="sos")
        y = signal.sosfiltfilt(sos, np.asarray(x, dtype=float))
    except Exception:
        y = np.asarray(x, dtype=float)
    times, env = rms_envelope(y, sr, frame_ms=20.0, hop_ms=5.0)
    if env.size < 5:
        return np.array([], dtype=float), "ddk_envelope_too_short"
    smooth_frames = max(3, int(round(0.030 / 0.005)))
    if smooth_frames % 2 == 0:
        smooth_frames += 1
    env = signal.savgol_filter(env, smooth_frames, 2, mode="interp")
    median = float(np.median(env))
    mad = float(np.median(np.abs(env - median)))
    prominence = max(1e-8, 1.5 * mad)
    minimum_distance = max(1, int(round(0.080 / 0.005)))
    peaks, _ = signal.find_peaks(
        env,
        distance=minimum_distance,
        prominence=prominence,
        height=median + mad,
    )
    note = (
        "detector=rms_syllable_nuclei; bandpass=80-3000Hz_or_nyquist_limited; "
        "frame=20ms; hop=5ms; smooth=30ms; min_interval=80ms; prominence=1.5MAD"
    )
    return np.asarray(times[peaks], dtype=float), note


@dataclass(frozen=True)
class DDKPlugin(AcousticFeaturePlugin):
    subsystem: str = "articulatory"
    feature_names: tuple[str, ...] = DDK_FEATURES

    def compute(self, context: FeatureContext) -> dict[str, FeatureValue]:
        if not _is_ddk_task(context.task):
            return {
                name: FeatureValue(name, np.nan, "not_applicable", "feature_requires_ddk_amr_or_smr_task")
                for name in self.feature_names
            }
        if context.segmentation_wav_path is None or not context.segmentation_wav_path.exists():
            return {name: FeatureValue(name, np.nan, "failed", "segmentation_wav_missing") for name in self.feature_names}

        min_pause = float(getattr(context.config, "minimum_pause_duration_sec", 0.30))
        x, sr, region_note = read_region_audio(
            context.segmentation_wav_path,
            context.segments_csv,
            region="effective_task",
            min_pause_duration_sec=min_pause,
        )
        duration = x.size / float(sr) if sr > 0 else np.nan
        nuclei, detector_note = _detect_syllable_nuclei(x, sr)
        note = f"ddk_spec_v1; {detector_note}; {region_note}; detected_nuclei={nuclei.size}"
        if not np.isfinite(duration) or duration <= 0 or nuclei.size < 2:
            return {name: FeatureValue(name, np.nan, "failed", note + "; insufficient_events") for name in self.feature_names}

        intervals = np.diff(nuclei)
        rate = float(nuclei.size / duration)
        regularity = (
            float(np.std(intervals, ddof=1) / np.mean(intervals))
            if intervals.size >= 2 and float(np.mean(intervals)) > 0
            else np.nan
        )
        status = "computed_with_warning"
        note += "; automatic_nucleus_detection_requires_task_specific_manual_validation_before_clinical_use"
        return {
            "DDKrate": FeatureValue("DDKrate", rate, status, note),
            "DDKregularity": FeatureValue("DDKregularity", regularity, status, note),
        }
