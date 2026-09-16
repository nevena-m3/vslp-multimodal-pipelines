from __future__ import annotations

import numpy as np

from vslp.acoustic.segment.stage import _prepare_silero_audio


def _dominant_frequency_hz(x: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(x.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    return float(freqs[int(np.argmax(spectrum))])


def test_silero_resampling_preserves_duration_and_passband_tone_44100_to_16000():
    sr = 44100
    duration = 1.25
    t = np.arange(int(round(sr * duration)), dtype=np.float64) / sr
    x = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    y, info = _prepare_silero_audio(x, sr, 16000)

    assert y.dtype == np.float32
    assert info["silero_input_resampled"] is True
    assert info["source_analysis_sample_rate_hz"] == 44100
    assert info["silero_input_sample_rate_hz"] == 16000
    assert abs((len(y) / 16000.0) - (len(x) / sr)) <= (1 / sr + 1 / 16000 + 1e-9)
    assert abs(_dominant_frequency_hz(y, 16000) - 440.0) < 2.0


def test_silero_resampling_preserves_duration_48000_to_16000():
    sr = 48000
    t = np.arange(sr, dtype=np.float64) / sr
    x = (0.2 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)

    y, info = _prepare_silero_audio(x, sr, 16000)

    assert len(y) == 16000
    assert info["silero_input_resampled"] is True
    assert info["silero_input_clipped_sample_count"] == 0


def test_silero_no_resample_at_16000():
    x = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    y, info = _prepare_silero_audio(x, 16000, 16000)
    assert np.array_equal(y, x)
    assert info["silero_input_resampled"] is False
