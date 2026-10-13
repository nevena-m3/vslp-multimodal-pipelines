"""Synthetic engineering checks for the uncalibrated F11 digital measurements."""

import numpy as np
import pytest

from vslp.acoustic.features.catalog import load_feature_catalog, task_fit
from vslp.acoustic.features.family11_v2 import PUBLIC_IDS, _private_48k, measure_trial


def test_exact_twenty_task_specific_leaves_and_no_legacy_selection():
    leaves = [item for item in load_feature_catalog()["outputs"] if item["family_id"] == "F11"]
    assert len(leaves) == 20
    assert {item["feature_id"] for item in leaves} == PUBLIC_IDS
    assert {item["construct_id"] for item in leaves} == {"C060", "C061", "C062", "C063", "C064"}
    assert all(item["gui_selectable"] for item in leaves)
    assert all(not item["prerequisites"]["requires_alignment"] for item in leaves)
    labels = {"sustained_a": "Sustained /a/", "sentence_s5": "S5",
              "bamboo_passage": "Bamboo Passage"}
    assert all(task_fit(item, labels[item["task_id"]]) == "Recommended" for item in leaves)
    assert all(task_fit(item, "wstg") == "Not specified" for item in leaves)


@pytest.mark.parametrize("sr", [44100, 48000])
def test_50_ms_native_windows_and_remainder(sr):
    x = np.tile(np.linspace(-.2, .2, sr // 20), 5)
    _, audit = measure_trial(x, sr, "sustained_a")
    assert audit["visibility_window_samples"] == sr // 20
    assert audit["visibility_complete_windows"] == 5
    assert audit["visibility_discarded_samples"] == 0
    _, partial = measure_trial(np.r_[x, np.ones(17)], sr, "sustained_a")
    assert partial["visibility_complete_windows"] == 5
    assert partial["visibility_discarded_samples"] == 17


def test_48k_copy_removes_sample_rate_dependence_without_normalizing_gain():
    signals = [np.sin(2 * np.pi * 120 * np.arange(sr) / sr) * .2
               for sr in (44100, 48000)]
    measured = [measure_trial(x, sr, "sustained_a")[0] for x, sr in
                zip(signals, (44100, 48000), strict=True)]
    for key in ("vowel_absolute_energy_fs2", "vowel_sound_power_digital"):
        assert measured[0][key][0] == pytest.approx(measured[1][key][0], rel=1e-3)
    assert all(_private_48k(x, sr).size == 48000 for x, sr in
               zip(signals, (44100, 48000), strict=True))


def test_energy_resamples_reviewed_speech_intervals_without_filtering_across_join():
    sr = 44100
    first = np.r_[np.zeros(500), np.ones(500) * .4]
    second = np.r_[np.ones(500) * -.3, np.zeros(500)]
    values, audit = measure_trial(np.r_[first, second], sr, "sentence_s5",
                                  energy_intervals=[first, second])
    copies = [_private_48k(piece, sr) for piece in (first, second)]
    expected = sum(np.dot(piece, piece) for piece in copies)
    assert values["s5_absolute_energy_fs2"][0] == pytest.approx(expected)
    assert values["s5_sound_power_digital"][0] == pytest.approx(
        expected / (sum(piece.size for piece in copies) / 48000))
    assert audit["n_energy_intervals"] == 2
    assert expected != pytest.approx(np.dot(_private_48k(np.r_[first, second], sr),
                                          _private_48k(np.r_[first, second], sr)))


def test_gain_scaling_moments_graph_and_no_peak_normalization():
    rng = np.random.default_rng(31)
    x = rng.normal(0, .07, 48000)
    a, audit = measure_trial(x, 48000, "sustained_a")
    b, _ = measure_trial(2 * x, 48000, "sustained_a")
    for suffix in ("absolute_energy_fs2", "sound_power_digital"):
        assert b[f"vowel_{suffix}"][0] == pytest.approx(4 * a[f"vowel_{suffix}"][0])
    for suffix in ("amp_sd_fs", "amp_min_fs", "amp_max_fs"):
        assert b[f"vowel_{suffix}"][0] == pytest.approx(2 * a[f"vowel_{suffix}"][0])
    for suffix in ("wave_amp_skew", "wave_amp_excess_kurtosis",
                   "visibility_graph_density_amp"):
        assert b[f"vowel_{suffix}"][0] == pytest.approx(a[f"vowel_{suffix}"][0])
    assert audit["mean_square_power_support"] == pytest.approx(np.mean(x * x))
    for factor in (.1, 5.):
        scaled, _ = measure_trial(factor * x, 48000, "sustained_a")
        assert scaled["vowel_visibility_graph_density_amp"][0] == pytest.approx(
            a["vowel_visibility_graph_density_amp"][0])


def test_clipping_dc_and_missingness():
    x = np.array([1., 1., .99, .5, -.5, -1.])
    values, audit = measure_trial(x, 48000, "sustained_a")
    assert audit["clipping_fraction"] == pytest.approx(3 / 6)
    assert audit["near_clipping_fraction"] == pytest.approx(4 / 6)
    assert "digital_clipping" in audit["qc_flags"]
    assert audit["dc_offset"] == pytest.approx(np.mean(x))
    assert values["vowel_absolute_energy_fs2"][0] > 0
    invalid, _ = measure_trial(np.array([np.nan]), 48000, "sustained_a")
    assert invalid["vowel_absolute_energy_fs2"][1] == "nonfinite_waveform"
    flat, _ = measure_trial(np.zeros(4800), 48000, "sustained_a")
    assert flat["vowel_wave_amp_skew"][1] == "near_zero_variance"


def test_bamboo_graph_is_separate_from_speech_only_energy():
    rng = np.random.default_rng(3)
    speech = rng.normal(0, .1, 48000)
    connected = np.r_[speech[:24000], np.zeros(24000), speech[24000:]]
    clean, _ = measure_trial(speech, 48000, "bamboo_passage", connected_waveform=connected)
    changed, _ = measure_trial(speech, 48000, "bamboo_passage",
                               connected_waveform=np.r_[speech[:24000],
                                                        np.zeros(48000), speech[24000:]])
    assert clean["bamboo_absolute_energy_fs2"] == changed["bamboo_absolute_energy_fs2"]
    assert clean["bamboo_sound_power_digital"] == changed["bamboo_sound_power_digital"]
    rejected, _ = measure_trial(speech, 48000, "bamboo_passage",
                                connected_issue="hard_exclusion_ambiguity")
    assert rejected["bamboo_visibility_graph_density_amp"][1] == "hard_exclusion_ambiguity"


def test_insufficient_visibility_windows_is_explicit():
    values, _ = measure_trial(np.ones(1000), 48000, "sustained_a")
    assert values["vowel_visibility_graph_density_amp"][1] == "insufficient_local_sd_windows"
