"""Phonatory acoustic feature plugin.

Current implementation is a transparent, local signal-processing baseline. It computes
F0 with a frame autocorrelation estimator and CPP as a cepstral prominence proxy. These
are useful engineering features, but should be validated against the uploaded reference
notebook/Praat-style definitions before any clinical claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vslp.acoustic.features.plugins.audio_utils import (
    cepstral_peak_prominence_proxy,
    frame_signal,
    read_region_audio,
    rms_envelope,
    robust_voiced_mask_from_rms,
    simple_autocorr_f0,
)
from vslp.acoustic.features.plugins.base import AcousticFeaturePlugin, FeatureContext, FeatureValue

PHONATORY_FEATURES = ("f0_mean", "f0_std", "CPP_mean")


@dataclass(frozen=True)
class PhonatoryPlugin(AcousticFeaturePlugin):
    subsystem: str = "phonatory"
    feature_names: tuple[str, ...] = PHONATORY_FEATURES

    def compute(self, context: FeatureContext) -> dict[str, FeatureValue]:
        if context.segmentation_wav_path is None or not context.segmentation_wav_path.exists():
            return {name: FeatureValue(name, np.nan, "failed", "segmentation_wav_missing") for name in self.feature_names}
        min_pause = float(getattr(context.config, "minimum_pause_duration_sec", 0.15))
        x, sr, region_note = read_region_audio(context.segmentation_wav_path, context.segments_csv, region=context.analysis_region, min_pause_duration_sec=min_pause)
        if x.size == 0:
            return {name: FeatureValue(name, np.nan, "failed", "empty_audio") for name in self.feature_names}

        frames, _frame_len, _hop_len = frame_signal(x, sr, frame_ms=40.0, hop_ms=10.0)
        _t, rms = rms_envelope(x, sr, frame_ms=40.0, hop_ms=10.0)
        voiced = robust_voiced_mask_from_rms(rms)
        if len(voiced) != len(frames):
            voiced = np.ones(len(frames), dtype=bool)

        f0_vals: list[float] = []
        cpp_vals: list[float] = []
        for frame, keep in zip(frames, voiced, strict=False):
            if not keep:
                continue
            f0 = simple_autocorr_f0(frame, sr)
            cpp = cepstral_peak_prominence_proxy(frame, sr)
            if np.isfinite(f0):
                f0_vals.append(float(f0))
            if np.isfinite(cpp):
                cpp_vals.append(float(cpp))

        f0_arr = np.asarray(f0_vals, dtype=float)
        cpp_arr = np.asarray(cpp_vals, dtype=float)
        note = "computed_local_autocorr_f0_and_cpp_proxy_requires_reference_validation; " + region_note
        return {
            "f0_mean": FeatureValue("f0_mean", float(np.mean(f0_arr)) if f0_arr.size else np.nan, "computed_proxy", note),
            "f0_std": FeatureValue("f0_std", float(np.std(f0_arr, ddof=0)) if f0_arr.size else np.nan, "computed_proxy", note),
            "CPP_mean": FeatureValue("CPP_mean", float(np.mean(cpp_arr)) if cpp_arr.size else np.nan, "computed_proxy", note),
        }
