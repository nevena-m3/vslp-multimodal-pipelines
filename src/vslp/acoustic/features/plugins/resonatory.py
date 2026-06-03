"""Small validated-safe resonatory/intensity plugin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vslp.acoustic.features.plugins.audio_utils import read_mono_audio
from vslp.acoustic.features.plugins.base import AcousticFeaturePlugin, FeatureContext, FeatureValue

RESONATORY_BASELINE_FEATURES = ("RMSamp",)


@dataclass(frozen=True)
class ResonatoryBaselinePlugin(AcousticFeaturePlugin):
    subsystem: str = "resonatory"
    feature_names: tuple[str, ...] = RESONATORY_BASELINE_FEATURES

    def compute(self, context: FeatureContext) -> dict[str, FeatureValue]:
        if context.segmentation_wav_path is None or not context.segmentation_wav_path.exists():
            return {"RMSamp": FeatureValue("RMSamp", np.nan, "failed", "segmentation_wav_missing")}
        x, _sr = read_mono_audio(context.segmentation_wav_path)
        if x.size == 0:
            return {"RMSamp": FeatureValue("RMSamp", np.nan, "failed", "empty_audio")}
        return {"RMSamp": FeatureValue("RMSamp", float(np.sqrt(np.mean(x ** 2))), "computed", "global_rms_amplitude_from_canonical_segmentation_wav")}
