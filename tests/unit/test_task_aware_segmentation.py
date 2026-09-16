"""Deterministic synthetic tests for task-specific segmentation."""

import numpy as np
import pytest

from vslp.acoustic.segment.pipeline import SegmentationConfig, _segments
from vslp.acoustic.segment.selection import DDK, PHONATION, SILERO, recommend_method
from vslp.acoustic.segment.silero_reference import (
    Interval, boundary_alignment_diagnostics, classify_reading_segmentation,
)
from vslp.acoustic.segment.task_methods import DDKConfig, PhonationConfig, segment_ddk, segment_phonation


SR = 16000


def _ddk(starts, amplitude=0.3, noise=0.0, seed=9):
    x = np.random.default_rng(seed).normal(0, noise, SR * 6)
    for start in starts:
        a = round(start * SR)
        b = a + round(0.16 * SR)
        x[a:b] += amplitude * np.sin(2 * np.pi * 145 * np.arange(b - a) / SR)
    return x.astype(np.float32)


@pytest.mark.parametrize("starts", [
    [0.5 + i * 0.45 for i in range(9)],
    [0.5, 0.9, 1.6, 2.0, 2.8, 3.7, 4.5],
    [0.5, 0.72, 1.15, 1.8, 2.7, 3.9],
])
def test_ddk_regular_variable_and_irregular_rate_are_retained(starts):
    result = segment_ddk(_ddk(starts), SR)
    assert result["n_events"] == len(starts)
    assert result["automatic_status"] in {"ACCEPTED", "REVIEW"}
    assert all(abs(a / SR - expected) < 0.04 for (a, _), expected in zip(result["intervals_samples"], starts))
    assert result["ddk_rate_hz"] > 0


@pytest.mark.parametrize("amplitude,noise", [(0.015, 0), (0.15, 0.003)])
def test_ddk_low_amplitude_and_background_noise(amplitude, noise):
    starts = [0.6 + i * 0.5 for i in range(7)]
    result = segment_ddk(_ddk(starts, amplitude, noise), SR)
    assert result["n_events"] >= 6
    assert result["n_events"] <= 9


def test_ddk_missing_weak_event_and_silence_are_not_clinical_exclusions():
    starts = [0.6, 1.1, 2.1, 2.6]
    x = _ddk(starts)
    weak = _ddk([1.6], amplitude=0.015)
    result = segment_ddk(x + weak, SR)
    assert result["automatic_status"] != "EXCLUDED"
    assert result["n_events"] >= 4
    silent = segment_ddk(np.zeros(SR * 2, dtype=np.float32), SR)
    assert silent["automatic_status"] == "REVIEW"
    assert "no_detectable_ddk_events" in silent["flags"]


def _phonation(onset=0.5, offset=4.6, amplitude=0.2, ramp=False, gap=None, noise=0.0):
    t = np.arange(SR * 6) / SR
    env = ((t >= onset) & (t < offset)).astype(float) * amplitude
    if ramp:
        env *= np.clip((t - onset) / 0.35, 0, 1) * np.clip((offset - t) / 0.4, 0, 1)
    if gap:
        env[(t >= gap[0]) & (t < gap[1])] = 0
    x = env * (np.sin(2 * np.pi * 170 * t) + 0.08 * np.sin(2 * np.pi * 245 * t))
    return (x + np.random.default_rng(2).normal(0, noise, len(t))).astype(np.float32)


@pytest.mark.parametrize("kwargs", [
    {}, {"ramp": True}, {"amplitude": 0.015},
    {"gap": (2.0, 2.3)}, {"noise": 0.0005},
])
def test_phonation_full_episode_and_stable_region(kwargs):
    result = segment_phonation(_phonation(**kwargs), SR)
    full = result["full_samples"]
    stable = result["stable_samples"]
    assert full is not None
    assert abs(full[0] / SR - 0.5) < (0.25 if kwargs.get("ramp") else 0.06)
    assert abs(full[1] / SR - 4.6) < (0.2 if kwargs.get("ramp") else 0.06)
    assert stable is not None
    assert full[0] < stable[0] < stable[1] < full[1]
    if kwargs.get("gap"):
        assert result["internal_breaks_samples"]


def test_short_phonation_is_flagged_but_full_episode_retained():
    result = segment_phonation(_phonation(offset=1.2), SR)
    assert result["full_samples"] is not None
    assert result["stable_samples"] is None
    assert result["automatic_status"] == "REVIEW"


def test_silero_boundary_diagnostic_does_not_move_primary_interval():
    x = np.zeros(SR * 3, dtype=np.float32)
    x[SR:2 * SR] = 0.15
    exact = Interval(1.003, 1.997)
    audit = boundary_alignment_diagnostics(x, SR, [exact])
    assert audit.loc[0, "start_sec"] == exact.start_sec
    assert audit.loc[0, "end_sec"] == exact.end_sec
    segments = _segments([exact], 3.0)
    assert segments.loc[1, "start_sec"] == exact.start_sec
    assert SegmentationConfig().speech_pad_ms == 0


@pytest.mark.parametrize("task,method", [
    ("Bamboo Passage", SILERO), ("Buy Bobby a Puppy", SILERO),
    ("WSTG", SILERO), ("/pa/", DDK), ("/pataka/", DDK),
    ("AMR ka", DDK), ("sustained /a/", PHONATION), ("vowel /i/", PHONATION),
])
def test_task_recommendation(task, method):
    assert recommend_method(task) == method


def test_method_defaults_are_task_specific():
    assert DDKConfig().frame_ms == 20
    assert PhonationConfig().stable_duration_ms == 2000
    assert SegmentationConfig().threshold == 0.50


def test_reading_triage_flags_segmentation_quality_only():
    base = {"duration_sec": 7.0, "speech_fraction": 0.5, "n_speech_segments": 4,
            "longest_internal_nonspeech_sec": 0.5, "rms_db_median": -30.0,
            "rms_db_std": 10.0}
    assert classify_reading_segmentation(base)["qc_status"] == "accepted"
    assert classify_reading_segmentation({**base, "n_speech_segments": 0})["qc_status"] == "excluded"
    assert classify_reading_segmentation({**base, "longest_internal_nonspeech_sec": 5.5})["qc_status"] == "flagged"


@pytest.mark.parametrize("sample_rate", [8000, 22050, 44100])
def test_native_sample_rates_preserve_ddk_and_phonation_boundaries(sample_rate):
    t = np.arange(sample_rate * 4) / sample_rate
    ddk = np.zeros(len(t), dtype=np.float32)
    for start in (0.5, 1.1, 1.7, 2.3):
        a, b = round(start * sample_rate), round((start + 0.18) * sample_rate)
        ddk[a:b] = 0.3 * np.sin(2 * np.pi * 140 * np.arange(b - a) / sample_rate)
    measured = segment_ddk(ddk, sample_rate)
    assert measured["n_events"] == 4
    assert abs(measured["intervals_samples"][0][0] / sample_rate - 0.5) < 0.04
    vowel = np.where((t >= 0.4) & (t < 3.6),
                     0.2 * np.sin(2 * np.pi * 170 * t), 0).astype(np.float32)
    episode = segment_phonation(vowel, sample_rate)
    assert episode["full_samples"] is not None
    assert abs(episode["full_samples"][0] / sample_rate - 0.4) < 0.04
