from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.features.plugins import implemented_feature_names
from vslp.acoustic.features.plugins.base import FeatureContext
from vslp.acoustic.features.plugins.phonatory import PhonatoryPlugin
from vslp.acoustic.features.plugins.rhythm import RhythmPlugin
from vslp.acoustic.features.plugins.timing import TimingPlugin


class DummyConfig:
    minimum_pause_duration_sec = 0.15
    task_word_counts = {"bamboo": 60}


def test_implemented_feature_names_include_v08_features():
    names = implemented_feature_names()
    assert "percent_pause" in names
    assert "f0_mean" in names
    assert "CPP_mean" in names
    assert "fft_peaks1" in names
    assert "RMSamp" in names


def test_timing_plugin_from_segments(tmp_path: Path):
    segs = pd.DataFrame(
        [
            {"segment_type": "nonspeech", "segment_role": "leading_nonspeech", "duration_sec": 0.2},
            {"segment_type": "speech", "segment_role": "speech", "duration_sec": 1.0},
            {"segment_type": "nonspeech", "segment_role": "internal_nonspeech", "duration_sec": 0.3},
            {"segment_type": "speech", "segment_role": "speech", "duration_sec": 1.5},
            {"segment_type": "nonspeech", "segment_role": "trailing_nonspeech", "duration_sec": 0.2},
        ]
    )
    path = tmp_path / "segments.csv"
    segs.to_csv(path, index=False)
    ctx = FeatureContext(file_name="x.wav", row=pd.Series({"task": "bamboo"}), segments_csv=path, task="bamboo", duration_sec=3.2, config=DummyConfig())
    out = TimingPlugin().compute(ctx)
    assert out["num_pause"].value == 1.0
    assert out["speech_dur"].value == 2.5
    assert np.isfinite(out["speech_rate"].value)


def test_phonatory_and_rhythm_plugins_on_synthetic_audio(tmp_path: Path):
    sr = 16000
    t = np.arange(0, 2.0, 1 / sr)
    x = 0.25 * np.sin(2 * np.pi * 120 * t) * (0.7 + 0.3 * np.sin(2 * np.pi * 3 * t))
    wav = tmp_path / "tone.wav"
    sf.write(wav, x.astype("float32"), sr)
    ctx = FeatureContext(file_name="tone.wav", row=pd.Series({}), segmentation_wav_path=wav, duration_sec=2.0, config=DummyConfig())
    ph = PhonatoryPlugin().compute(ctx)
    rh = RhythmPlugin().compute(ctx)
    assert ph["f0_mean"].status == "computed_proxy"
    assert np.isfinite(ph["f0_mean"].value)
    assert rh["intensity_CV"].status == "computed"
    assert np.isfinite(rh["fft_peaks1"].value)
