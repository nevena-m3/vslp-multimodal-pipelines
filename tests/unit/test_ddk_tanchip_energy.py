"""Synthetic regression cases for the Tanchip Energy implementation."""

import numpy as np
import pytest

from vslp.acoustic.segment.task_methods import DDKConfig, segment_ddk


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100])
@pytest.mark.parametrize("gain,irregular,leading_noise", [
    (1.0, False, False), (.01, False, False),
    (1.0, True, False), (1.0, False, True),
])
def test_ta_pattern_decreasing_amplitude_and_native_timing(sample_rate, gain, irregular, leading_noise):
    rng = np.random.default_rng(19)
    x = np.zeros(sample_rate * 9, dtype=np.float32)
    starts = np.arange(3.0, 8.8, .23)
    if irregular:
        starts = starts + .025 * np.sin(np.arange(len(starts)))
    for i, start in enumerate(starts):
        a, b = round(start * sample_rate), round((start + .09) * sample_rate)
        t = np.arange(b - a) / sample_rate
        x[a:b] += (gain * .4 * (1 - .8 * i / len(starts)) * np.sin(2 * np.pi * 140 * t))
    if leading_noise:
        x[round(.5 * sample_rate):round(.6 * sample_rate)] += (
            gain * .015 * rng.normal(size=round(.1 * sample_rate)))
    before = x.copy()
    result = segment_ddk(x, sample_rate)
    assert np.array_equal(x, before)
    assert result["automatic_status"] in {"ACCEPTED", "REVIEW"}
    assert result["n_syllables"] >= 20
    assert result["ddk_sequence_start_sec"] < 3.2 or leading_noise
    assert result["ddk_sequence_end_sec"] <= 9.0
    assert result["ddk_rate_hz"] > 0
    assert result["raw_candidate_count"] >= result["n_events"]
    assert result["processing"]["lowpass_hz"] == 200
    assert result["processing"]["config"]["frame_ms"] == 20
    assert result["processing"]["config"]["threshold_window_ms"] == 20


def test_ddk_rate_and_ctv_follow_authoritative_event_times():
    sr = 16000
    x = np.zeros(sr * 4, dtype=np.float32)
    for start in (.4, .8, 1.3, 1.9, 2.6):
        a, b = round(start * sr), round((start + .12) * sr)
        x[a:b] = .3 * np.sin(2 * np.pi * 150 * np.arange(b - a) / sr)
    result = segment_ddk(x, sr, DDKConfig())
    assert result["n_syllables"] == 5
    assert result["ddk_rate_hz"] == pytest.approx(
        5 / (result["ddk_sequence_end_sec"] - result["ddk_sequence_start_sec"]))
    cycles = np.diff(result["nuclei_samples"]) / sr
    assert result["ctv_sec"] == pytest.approx(np.mean(np.abs(np.diff(cycles))))
