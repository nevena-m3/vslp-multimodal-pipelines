import numpy as np

from vslp.acoustic.features.family11 import calculate_amplitude_features, natural_visibility_density


def test_family11_exact_scaling_duration_and_moments():
    x = np.array([-1.0, 0.0, 1.0, 0.0])
    values, audit = calculate_amplitude_features(x, 4)
    assert values["absolute_energy_fs2"] == (2.0, "")
    assert values["sound_power_digital"] == (2.0, "")
    doubled, _ = calculate_amplitude_features(np.tile(x, 2), 4)
    assert doubled["absolute_energy_fs2"][0] == 4.0
    assert doubled["sound_power_digital"][0] == 2.0
    scaled, _ = calculate_amplitude_features(x * 2, 4)
    assert scaled["absolute_energy_fs2"][0] == 8.0
    assert np.isclose(values["wave_amp_skew"][0], 0.0)
    assert audit["clipping_fraction"] == 0.5


def test_family11_normalization_policy_and_zero_signal():
    x = np.array([-0.5, 0.0, 0.5, 0.0])
    values, audit = calculate_amplitude_features(x, 4, amplitude_normalized=True)
    assert values["absolute_energy_fs2"][1] == "incompatible_amplitude_normalization"
    assert values["amp_sd_fs"][1] == "incompatible_amplitude_normalization"
    assert values["wave_amp_skew"][1] == ""
    assert audit["amplitude_normalization_applied"] is True
    zero, _ = calculate_amplitude_features(np.zeros(100), 100)
    assert zero["absolute_energy_fs2"] == (0.0, "")
    assert zero["wave_amp_skew"][1] == "zero_amplitude_variance"


def test_family11_visibility_density_is_bounded_and_scale_invariant():
    series = np.array([1.0, 3.0, 2.0, 5.0, 1.0])
    density = natural_visibility_density(series)
    assert 0 < density <= 1
    assert natural_visibility_density(series * 10) == density
