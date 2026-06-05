from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.features.plugins.coordination import CoordinationPlugin, _eigenspectrum_complexity
from vslp.acoustic.features.plugins.base import FeatureContext


def test_eigenspectrum_complexity_identity_bounds():
    c = np.eye(6)
    val = _eigenspectrum_complexity(c)
    assert 0.99 <= val <= 1.0


def test_coordination_plugin_outputs_three_features(tmp_path: Path):
    sr = 16000
    dur = 2.0
    t = np.arange(int(sr * dur)) / sr
    # Simple vowel-like harmonic signal with amplitude modulation; enough structure for CPP and LPC roots.
    x = 0.25 * np.sin(2 * np.pi * 120 * t) + 0.12 * np.sin(2 * np.pi * 240 * t) + 0.05 * np.sin(2 * np.pi * 700 * t)
    x *= 1.0 + 0.2 * np.sin(2 * np.pi * 3.0 * t)
    wav = tmp_path / "synthetic.wav"
    sf.write(wav, x.astype(np.float32), sr)
    seg = tmp_path / "segments.csv"
    pd.DataFrame([
        {"segment_type": "speech", "segment_role": "speech", "start_sec": 0.1, "end_sec": 1.9, "duration_sec": 1.8}
    ]).to_csv(seg, index=False)

    ctx = FeatureContext(file_name="synthetic.wav", row=pd.Series({}), segments_csv=seg, segmentation_wav_path=wav, config=None)
    out = CoordinationPlugin().compute(ctx)
    assert set(out) == {"CPP_F1_comp", "CPP_F2_comp", "F1_F2_comp"}
    # At least one pair should compute on this synthetic stable vowel-like signal.
    assert any(np.isfinite(v.value) for v in out.values())
    for v in out.values():
        if np.isfinite(v.value):
            assert 0.0 <= float(v.value) <= 1.0
            assert v.status == "computed"
