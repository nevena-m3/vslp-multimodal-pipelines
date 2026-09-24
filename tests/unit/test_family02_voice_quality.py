"""Formula and synthetic behavior tests for Family 02."""

from __future__ import annotations

import numpy as np

from vslp.acoustic.features.family02 import (
    FAMILY02_IDS, PARAMETER_SET_ID, calculate_voice_quality,
    directional_perturbation_factor,
)


def _tone(*, noisy=False, amplitude_modulation=False, frequency_modulation=False):
    sr = 44100
    t = np.arange(sr * 2) / sr
    amplitude = 0.2 * (1 + .25 * np.sin(2 * np.pi * 4 * t)) if amplitude_modulation else .2
    phase = 2 * np.pi * 180 * t
    if frequency_modulation:
        phase += .18 * np.sin(2 * np.pi * 5 * t)
    signal = amplitude * np.sin(phase)
    if noisy:
        signal += np.random.default_rng(7).normal(0, .08, t.size)
    return signal.astype(np.float32), sr


def test_exact_ids_profile_and_constant_period_behavior():
    x, sr = _tone()
    values, audit = calculate_voice_quality(x, sr)
    assert set(values) == FAMILY02_IDS
    assert PARAMETER_SET_ID == "family02_stable_vowel_praat_v1"
    assert audit["n_cycles"] >= 100
    assert values["jitter_local_pct"][0] < .01
    assert values["shimmer_local_pct"][0] < .01
    assert values["jitter_absolute_s"][1] == ""
    assert values["shimmer_local_db"][1] == ""


def test_perturbation_and_noise_directions():
    clean, sr = _tone()
    perturbed, _ = _tone(amplitude_modulation=True, frequency_modulation=True)
    noisy, _ = _tone(noisy=True)
    clean_values, _ = calculate_voice_quality(clean, sr)
    perturbed_values, _ = calculate_voice_quality(perturbed, sr)
    noisy_values, _ = calculate_voice_quality(noisy, sr)
    assert perturbed_values["jitter_local_pct"][0] > clean_values["jitter_local_pct"][0]
    assert perturbed_values["shimmer_local_pct"][0] > clean_values["shimmer_local_pct"][0]
    assert clean_values["hnr_mean_db"][0] > noisy_values["hnr_mean_db"][0]


def test_dfp_formula_and_insufficient_cycles():
    periods = np.array([.01, .011, .010, .012, .011])
    assert directional_perturbation_factor(periods) == 100.0
    assert np.isnan(directional_perturbation_factor(periods[:3]))
    x = np.zeros(4410, dtype=np.float32)
    values, _ = calculate_voice_quality(x, 44100)
    assert np.isnan(values["jitter_local_pct"][0])
    assert values["jitter_local_pct"][1] in {"insufficient_usable_cycles", "pointprocess_failed"}
