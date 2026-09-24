"""Numerical and native-rate checks for the shared Family 01 F0 track."""

from __future__ import annotations

import numpy as np

from vslp.acoustic.features.family01 import (
    FAMILY01_IDS, F0Track, PARAMETER_SET_ID, calculate_f0_features, praat_f0_track,
)


def test_exact_ids_and_semitone_statistics():
    hz = np.array([100.0, 100.0, 200.0, 200.0] * 10)
    track = F0Track(np.arange(hz.size) * .005, hz, np.ones(hz.size, dtype=bool),
                    48000, 0, .2, "stable_phonation")
    values = calculate_f0_features(track, sustained=True)
    assert set(values) == FAMILY01_IDS
    assert values["f0_mean_hz"] == (150.0, "")
    assert values["f0_median_hz"] == (150.0, "")
    assert np.isclose(values["f0_iqr_st"][0], 12.0)
    assert np.isclose(values["f0_range_st"][0], 12.0)
    assert np.isclose(values["pfr_maxmin_st"][0], 12.0)
    assert np.isclose(values["f0_sd_st"][0], np.std(12 * np.log2(hz / 150), ddof=1))


def test_tracking_failure_is_nan_with_reason():
    hz = np.full(40, 180.0)
    track = F0Track(np.arange(40) * .005, hz, np.arange(40) < 19,
                    44100, 0, .2, "stable_phonation")
    assert all(np.isnan(value) and reason == "insufficient_valid_f0_frames"
               for value, reason in calculate_f0_features(track, sustained=True).values())
    track = F0Track(track.time_sec, hz, np.arange(40) < 30,
                    44100, 0, .2, "stable_phonation")
    assert all(np.isnan(value) and reason == "poor_sustained_tracking_yield"
               for value, reason in calculate_f0_features(track, sustained=True).values())


def test_native_rate_praat_track_and_canonical_unchanged():
    sr = 44100
    t = np.arange(sr * 2) / sr
    audio = (.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    original = audio.copy()
    track = praat_f0_track(audio, sr, analysis_start_sec=1.0,
                           analysis_region="stable_phonation")
    assert track.sample_rate_hz == sr
    assert track.parameter_set_id == PARAMETER_SET_ID
    assert track.time_sec.min() >= 1.0
    assert np.array_equal(audio, original)
    assert abs(calculate_f0_features(track, sustained=True)["f0_mean_hz"][0] - 180) < 2
