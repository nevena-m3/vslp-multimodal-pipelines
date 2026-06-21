from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.features.plugins.base import FeatureContext
from vslp.acoustic.features.plugins.rhythm import RhythmPlugin


class RhythmConfig:
    minimum_pause_duration_sec = 0.30
    rhythm_region_policy = "effective_task"
    rhythm_envelope_bandpass_low_hz = 300.0
    rhythm_envelope_bandpass_high_hz = 1000.0
    rhythm_envelope_sample_rate_hz = 100.0


def _write_segments(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_rhythm_features_detect_known_modulation_frequency(tmp_path: Path):
    sr = 16000
    dur = 6.0
    t = np.arange(0, dur, 1 / sr)
    modulation_hz = 3.0
    # Carrier lies inside the default 300--1000 Hz EMS prefilter. The amplitude
    # envelope is modulated at 3 Hz, so the dominant EMS peak should be near 3 Hz.
    env = 0.55 + 0.45 * (1 + np.sin(2 * np.pi * modulation_hz * t)) / 2
    x = (0.35 * env * np.sin(2 * np.pi * 500 * t)).astype("float32")
    wav = tmp_path / "rhythm.wav"
    sf.write(wav, x, sr)
    segments = tmp_path / "segments.csv"
    _write_segments(
        segments,
        [
            {"segment_type": "speech", "segment_role": "speech", "start_sec": 0.0, "end_sec": dur, "duration_sec": dur},
        ],
    )

    ctx = FeatureContext(
        file_name="rhythm.wav",
        row=pd.Series({"task": "bamboo"}),
        segmentation_wav_path=wav,
        segments_csv=segments,
        duration_sec=dur,
        task="bamboo",
        config=RhythmConfig(),
        analysis_region="speech_only",
    )
    out = RhythmPlugin().compute(ctx)
    assert out["fft_peaks1"].status == "computed"
    assert abs(float(out["fft_peaks1"].value) - modulation_hz) < 0.35
    assert 0.0 <= float(out["nrj_3_6"].value) <= 1.0
    assert 0.0 <= float(out["nrj_below_boundary"].value) <= 1.0
    assert "validated_rhythm_v0.27" in out["ratio_below_above"].note
    assert "region=effective_task" in out["ratio_below_above"].note


def test_rhythm_features_preserve_internal_pause_region_by_default(tmp_path: Path):
    sr = 16000
    dur = 4.0
    t = np.arange(0, dur, 1 / sr)
    x = (0.2 * np.sin(2 * np.pi * 500 * t)).astype("float32")
    wav = tmp_path / "pause_sensitive.wav"
    sf.write(wav, x, sr)
    segments = tmp_path / "segments.csv"
    _write_segments(
        segments,
        [
            {"segment_type": "nonspeech", "segment_role": "leading_nonspeech", "start_sec": 0.0, "end_sec": 0.5, "duration_sec": 0.5},
            {"segment_type": "speech", "segment_role": "speech", "start_sec": 0.5, "end_sec": 1.5, "duration_sec": 1.0},
            {"segment_type": "nonspeech", "segment_role": "internal_nonspeech", "start_sec": 1.5, "end_sec": 2.0, "duration_sec": 0.5},
            {"segment_type": "speech", "segment_role": "speech", "start_sec": 2.0, "end_sec": 3.5, "duration_sec": 1.5},
            {"segment_type": "nonspeech", "segment_role": "trailing_nonspeech", "start_sec": 3.5, "end_sec": 4.0, "duration_sec": 0.5},
        ],
    )
    ctx = FeatureContext(file_name="x.wav", row=pd.Series({}), segmentation_wav_path=wav, segments_csv=segments, config=RhythmConfig())
    out = RhythmPlugin().compute(ctx)
    assert out["intensity_CV"].status == "computed_with_warning"
    assert "recording_not_spl_calibrated" in out["intensity_CV"].note
    assert "region=effective_task" in out["intensity_CV"].note
    # Effective task = 0.5 to 3.5 sec = 3.0 seconds; internal pause is preserved.
    assert "selected_sec=3." in out["intensity_CV"].note
