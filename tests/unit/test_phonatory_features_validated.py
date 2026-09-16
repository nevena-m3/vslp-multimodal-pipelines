from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.features.plugins.base import FeatureContext
from vslp.acoustic.features.plugins.phonatory import PHONATORY_FEATURES, PhonatoryPlugin
from vslp.acoustic.features.plugins import implemented_feature_names


class PhonatoryConfig:
    minimum_pause_duration_sec = 0.15
    phonatory_f0_min_hz = 60.0
    phonatory_f0_max_hz = 400.0
    phonatory_min_autocorr_peak = 0.30
    phonatory_frame_ms = 40.0
    phonatory_hop_ms = 10.0
    phonatory_min_voice_break_sec = 0.08


def _speech_segments(path: Path, dur: float) -> None:
    pd.DataFrame([
        {"segment_type": "speech", "segment_role": "speech", "start_sec": 0.0, "end_sec": dur, "duration_sec": dur},
    ]).to_csv(path, index=False)


def test_phonatory_plugin_computes_full_registered_group_on_stable_tone(tmp_path: Path):
    sr = 16000
    dur = 2.5
    f0 = 140.0
    t = np.arange(0, dur, 1 / sr)
    # Add a weak second harmonic so H1/H2 estimates are both meaningful.
    x = 0.25 * np.sin(2 * np.pi * f0 * t) + 0.06 * np.sin(2 * np.pi * 2 * f0 * t)
    wav = tmp_path / "phonatory.wav"
    seg = tmp_path / "segments.csv"
    sf.write(wav, x.astype("float32"), sr)
    _speech_segments(seg, dur)

    ctx = FeatureContext(
        file_name="phonatory.wav",
        row=pd.Series({"task": "vowel"}),
        segmentation_wav_path=wav,
        segments_csv=seg,
        task="vowel",
        duration_sec=dur,
        config=PhonatoryConfig(),
        analysis_region="speech_only",
    )
    out = PhonatoryPlugin().compute(ctx)
    assert set(PHONATORY_FEATURES).issubset(out.keys())
    assert all(out[name].status == "computed" for name in ["f0_mean", "f0_std", "HNR", "localJitter", "localShimmer"])
    assert "praat_parselmouth_pointprocess" in out["localJitter"].note
    assert abs(float(out["f0_mean"].value) - f0) < 3.0
    assert float(out["f0_std"].value) < 3.0
    assert float(out["localJitter"].value) < 2.0
    assert float(out["localShimmer"].value) < 5.0
    assert abs(float(out["H1freq"].value) - f0) < 8.0
    assert abs(float(out["H2freq"].value) - 2 * f0) < 12.0
    assert "validated_local_phonatory_v0.28" in out["CPP_mean"].note


def test_phonatory_plugin_counts_internal_voice_break(tmp_path: Path):
    sr = 16000
    f0 = 120.0
    tone1 = 0.25 * np.sin(2 * np.pi * f0 * np.arange(0, 0.8, 1 / sr))
    gap = np.zeros(int(0.2 * sr))
    tone2 = 0.25 * np.sin(2 * np.pi * f0 * np.arange(0, 0.8, 1 / sr))
    x = np.r_[tone1, gap, tone2]
    dur = len(x) / sr
    wav = tmp_path / "break.wav"
    seg = tmp_path / "segments.csv"
    sf.write(wav, x.astype("float32"), sr)
    _speech_segments(seg, dur)
    ctx = FeatureContext(file_name="break.wav", row=pd.Series({}), segmentation_wav_path=wav, segments_csv=seg, config=PhonatoryConfig())
    out = PhonatoryPlugin().compute(ctx)
    assert float(out["num_voicebreaks"].value) >= 1.0


def test_implemented_features_include_complete_phonatory_group():
    names = implemented_feature_names()
    for feat in PHONATORY_FEATURES:
        assert feat in names


def test_sustained_phonation_uses_stable_native_region_and_full_episode_for_breaks(tmp_path: Path):
    sr = 16000
    t = np.arange(sr * 4) / sr
    x = 0.2 * np.sin(2 * np.pi * 130 * t)
    x[(t >= 2.0) & (t < 2.2)] = 0
    wav = tmp_path / "sustained.wav"
    sf.write(wav, x.astype("float32"), sr, subtype="FLOAT")
    ctx = FeatureContext(
        file_name="sustained.wav", segmentation_wav_path=wav,
        row=pd.Series({"segmentation_method": "sustained_phonation",
                       "full_phonation_start": 0.0, "full_phonation_end": 4.0,
                       "stable_region_start": 0.5, "stable_region_end": 1.7}),
        config=PhonatoryConfig(),
    )
    out = PhonatoryPlugin().compute(ctx)
    assert "region=stable_phonation_native_rate" in out["f0_mean"].note
    assert "voice_breaks_from_full_episode" in out["num_voicebreaks"].note
    assert float(out["num_voicebreaks"].value) >= 1.0
