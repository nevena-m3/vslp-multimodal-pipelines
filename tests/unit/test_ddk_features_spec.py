from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.features.plugins.base import FeatureContext
from vslp.acoustic.features.plugins.ddk import DDKPlugin


class Config:
    minimum_pause_duration_sec = 0.30


def test_ddk_rate_and_regularity_from_regular_synthetic_nuclei(tmp_path: Path):
    sr = 16000
    duration = 4.0
    t = np.arange(int(sr * duration)) / sr
    x = np.zeros_like(t)
    event_times = np.arange(0.25, 3.76, 0.25)
    for event_time in event_times:
        envelope = np.exp(-0.5 * ((t - event_time) / 0.025) ** 2)
        x += 0.4 * envelope * np.sin(2 * np.pi * 500 * t)
    wav = tmp_path / "subject_ddk_pa.wav"
    sf.write(wav, x.astype("float32"), sr)
    segments = tmp_path / "segments.csv"
    pd.DataFrame([
        {"segment_type": "speech", "segment_role": "speech", "start_sec": 0.0, "end_sec": duration, "duration_sec": duration}
    ]).to_csv(segments, index=False)

    ctx = FeatureContext(
        file_name=wav.name,
        row=pd.Series({"task": "DDK-pa"}),
        segmentation_wav_path=wav,
        segments_csv=segments,
        duration_sec=duration,
        task="DDK-pa",
        config=Config(),
    )
    out = DDKPlugin().compute(ctx)
    assert out["DDKrate"].status == "computed_with_warning"
    assert abs(float(out["DDKrate"].value) - len(event_times) / duration) < 0.3
    assert float(out["DDKregularity"].value) < 0.05


def test_ddk_features_are_not_applied_to_passage(tmp_path: Path):
    ctx = FeatureContext(file_name="passage.wav", row=pd.Series({"task": "Bamboo"}), task="Bamboo", config=Config())
    out = DDKPlugin().compute(ctx)
    assert out["DDKrate"].status == "not_applicable"
    assert np.isnan(out["DDKrate"].value)
