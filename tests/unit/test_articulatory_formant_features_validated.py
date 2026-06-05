from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

from vslp.acoustic.features.plugins.articulatory import ArticulatoryPlugin
from vslp.acoustic.features.plugins.base import FeatureContext


class ArticulatoryConfig:
    minimum_pause_duration_sec = 0.15
    articulatory_region_policy = "speech_only"
    articulatory_frame_ms = 25.0
    articulatory_hop_ms = 10.0
    articulatory_lpc_sample_rate_hz = 10000
    articulatory_preemphasis = 0.97
    articulatory_min_valid_frame_fraction = 0.05
    articulatory_max_abs_velocity_hz_s = 20000.0


def _write_segments(path: Path, dur: float) -> None:
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


def _synthetic_vowel(sr: int, dur: float, formants: list[tuple[float, float]], seed: int = 0) -> np.ndarray:
    """Generate an all-pole vowel-like signal from specified formant freq/bw pairs."""
    rng = np.random.default_rng(seed)
    n = int(round(sr * dur))
    # Excitation: impulse train plus small noise, enough to excite resonances.
    f0 = 120.0
    excitation = np.zeros(n, dtype=float)
    excitation[:: int(round(sr / f0))] = 1.0
    excitation += 0.005 * rng.standard_normal(n)

    a = np.array([1.0])
    for freq, bw in formants:
        r = np.exp(-np.pi * bw / sr)
        theta = 2.0 * np.pi * freq / sr
        section = np.array([1.0, -2.0 * r * np.cos(theta), r * r])
        a = np.convolve(a, section)
    y = signal.lfilter([1.0], a, excitation)
    y = y - np.mean(y)
    y = y / max(np.max(np.abs(y)), 1e-9) * 0.35
    return y.astype("float32")


def test_articulatory_plugin_recovers_synthetic_formants(tmp_path: Path):
    sr = 16000
    dur = 2.0
    target = [(500.0, 80.0), (1500.0, 120.0), (2500.0, 160.0), (3500.0, 220.0), (4500.0, 300.0)]
    x = _synthetic_vowel(sr, dur, target)
    wav = tmp_path / "vowel.wav"
    sf.write(wav, x, sr)
    seg = tmp_path / "segments.csv"
    _write_segments(seg, dur)

    ctx = FeatureContext(
        file_name="vowel.wav",
        row=pd.Series({"task": "vowel"}),
        segmentation_wav_path=wav,
        segments_csv=seg,
        duration_sec=dur,
        task="vowel",
        config=ArticulatoryConfig(),
        analysis_region="speech_only",
    )
    out = ArticulatoryPlugin().compute(ctx)
    assert out["f1"].status in {"computed", "low_validity"}
    assert abs(float(out["f1"].value) - 500.0) < 180.0
    assert abs(float(out["f2"].value) - 1500.0) < 250.0
    assert abs(float(out["f3"].value) - 2500.0) < 350.0
    assert np.isfinite(float(out["f1_bw"].value))
    assert np.isfinite(float(out["f2_bw"].value))
    assert "validated_formant_lpc_v0.29" in out["f1"].note


def test_articulatory_plugin_fails_cleanly_on_empty_audio(tmp_path: Path):
    wav = tmp_path / "empty.wav"
    sf.write(wav, np.array([], dtype="float32"), 16000)
    ctx = FeatureContext(
        file_name="empty.wav",
        row=pd.Series({}),
        segmentation_wav_path=wav,
        segments_csv=None,
        config=ArticulatoryConfig(),
    )
    out = ArticulatoryPlugin().compute(ctx)
    assert out["f1"].status == "failed"
    assert np.isnan(out["f1"].value)


def test_articulatory_velocity_features_are_present(tmp_path: Path):
    sr = 16000
    dur = 1.5
    target = [(600.0, 90.0), (1700.0, 140.0), (2600.0, 180.0)]
    x = _synthetic_vowel(sr, dur, target, seed=1)
    wav = tmp_path / "steady.wav"
    sf.write(wav, x, sr)
    seg = tmp_path / "segments.csv"
    _write_segments(seg, dur)
    ctx = FeatureContext(file_name="steady.wav", row=pd.Series({}), segmentation_wav_path=wav, segments_csv=seg, config=ArticulatoryConfig())
    out = ArticulatoryPlugin().compute(ctx)
    for name in ["f1_range", "f2_range", "f1_d_dx_median", "f2_d_dx_prc_5_95"]:
        assert name in out
        assert out[name].status in {"computed", "low_validity"}
