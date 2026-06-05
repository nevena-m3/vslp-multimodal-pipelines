from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.features.plugins.base import FeatureContext
from vslp.acoustic.features.plugins.resonatory import ResonatoryPlugin
from vslp.acoustic.features.plugins import implemented_feature_names


def _write_signal(tmp_path: Path, freqs_amps: list[tuple[float, float]], sr: int = 16000, dur: float = 1.2, name: str = "synthetic_resonatory.wav") -> Path:
    t = np.arange(int(sr * dur)) / sr
    x = np.zeros_like(t)
    for f, a in freqs_amps:
        x += a * np.sin(2 * np.pi * f * t)
    x = x / (np.max(np.abs(x)) + 1e-12) * 0.7
    path = tmp_path / name
    sf.write(path, x.astype(np.float32), sr)
    return path


def _segments(tmp_path: Path, dur: float = 1.2) -> Path:
    path = tmp_path / "segments.csv"
    pd.DataFrame(
        [
            {
                "segment_type": "speech",
                "segment_role": "speech",
                "start_sec": 0.0,
                "end_sec": dur,
                "duration_sec": dur,
            }
        ]
    ).to_csv(path, index=False)
    return path


def test_resonatory_plugin_computes_raw_nasality_ratios(tmp_path: Path):
    # P0 weaker than A1, P1 moderate, F3 strong enough to produce positive A3-P0.
    wav = _write_signal(tmp_path, [(250, 0.15), (700, 0.9), (950, 0.25), (2400, 0.55)])
    seg = _segments(tmp_path)
    cfg = SimpleNamespace(minimum_pause_duration_sec=0.15, resonatory_region_policy="speech_only")
    ctx = FeatureContext(file_name="x.wav", row=pd.Series({}), segments_csv=seg, segmentation_wav_path=wav, config=cfg)
    out = ResonatoryPlugin().compute(ctx)

    for name in ["A1P0", "A1P1", "A3P0", "P0freq", "P0amp", "P1amp", "F1freq", "F1amp", "F3amp", "RMSamp"]:
        assert name in out
        assert np.isfinite(float(out[name].value)), f"{name} should be finite"
    assert 180 <= float(out["P0freq"].value) <= 500
    assert float(out["A1P0"].value) > 0
    assert float(out["A1P1"].value) > 0
    assert out["A1P0"].status in {"computed", "computed_with_warnings"}
    assert out["A1P0comp"].status in {"computed_proxy", "computed_with_warnings", "low_validity"}


def test_resonatory_plugin_detects_lower_a1p0_when_p0_is_stronger(tmp_path: Path):
    seg = _segments(tmp_path)
    cfg = SimpleNamespace(minimum_pause_duration_sec=0.15, resonatory_region_policy="speech_only")

    oral_wav = _write_signal(tmp_path, [(250, 0.10), (700, 0.9), (950, 0.20), (2400, 0.55)], name="oral.wav")
    nasal_wav = _write_signal(tmp_path, [(250, 0.75), (700, 0.5), (950, 0.45), (2400, 0.40)], name="nasal.wav")

    oral = ResonatoryPlugin().compute(FeatureContext("oral.wav", pd.Series({}), seg, oral_wav, config=cfg))
    nasal = ResonatoryPlugin().compute(FeatureContext("nasal.wav", pd.Series({}), seg, nasal_wav, config=cfg))

    assert np.isfinite(float(oral["A1P0"].value))
    assert np.isfinite(float(nasal["A1P0"].value))
    assert float(nasal["A1P0"].value) < float(oral["A1P0"].value)


def test_resonatory_features_registered_as_implemented():
    names = implemented_feature_names()
    for name in ["A1P0", "A1P1", "A3P0", "P0freq", "P0amp", "P0prom", "P1amp", "F1freq", "F1amp", "F1width", "F2freq", "F2amp", "F2width", "F3freq", "F3amp", "F3width", "RMSamp"]:
        assert name in names
