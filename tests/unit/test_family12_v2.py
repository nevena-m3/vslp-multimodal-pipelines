"""Deterministic nonclinical F12 contract checks."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, resample_poly, sosfilt

from vslp.acoustic.features.catalog import load_feature_catalog
from vslp.acoustic.features.family12_v2 import PUBLIC_IDS, measure_trial, specs, spectral_moments


def sine(freq, sr=48000, duration=1):
    t = np.arange(round(sr*duration))/sr
    return np.sin(2*np.pi*freq*t)


def test_leaf_count_and_task_map():
    rows = specs()
    assert len(PUBLIC_IDS) == 130
    assert len(rows) == 132
    assert sum(row["construct_id"] == "C068" and not row["selectable"] for row in rows) == 2
    assert "s1_spec_m1_hz" in PUBLIC_IDS
    assert "bamboo_spectral_centroid_hz" in PUBLIC_IDS
    assert "s1_spectral_centroid_hz" not in PUBLIC_IDS
    assert "vowel_mfcc01_mean" in PUBLIC_IDS
    assert "s2_mfcc13_sd" in PUBLIC_IDS
    assert "s6_mfcc13_mean" in PUBLIC_IDS
    assert "s5_spectral_contrast_band6_db" in PUBLIC_IDS
    catalog = [row for row in load_feature_catalog()["outputs"] if row["family_id"] == "F12"]
    assert sum(row["selectable"] for row in catalog) == 130
    assert all(not row["prerequisites"]["requires_alignment"] for row in catalog)


def test_native_44100_and_48000_and_mfcc_c0():
    rng = np.random.default_rng(3)
    reference = sosfilt(butter(6, 8000, fs=48000, output="sos"),
                        rng.normal(size=48000))
    native = resample_poly(reference, 147, 160)
    before = native.copy()
    a, audit = measure_trial([native], 44100, "sentence_s2")
    b, _ = measure_trial([reference], 48000, "sentence_s2")
    assert np.array_equal(native, before)
    assert audit["native_sample_rate_hz"] == 44100
    assert audit["analysis_sample_rate_hz"] == 48000
    assert a["s2_spec_m1_hz"][0] == pytest.approx(b["s2_spec_m1_hz"][0], rel=0.05)
    assert a["s2_mfcc01_mean"][0] == pytest.approx(b["s2_mfcc01_mean"][0], rel=0.02)
    assert sum(key.startswith("s2_mfcc") for key in a) == 26


def test_bandwidth_zcr_and_raw_moment_kurtosis():
    tone, _ = measure_trial([sine(500)], 48000, "bamboo_passage")
    rng = np.random.default_rng(2)
    broad, _ = measure_trial([rng.normal(size=48000)], 48000, "bamboo_passage")
    assert broad["bamboo_spectral_bandwidth_p2_mean_hz"][0] > tone[
        "bamboo_spectral_bandwidth_p2_mean_hz"][0]
    low, _ = measure_trial([sine(300)], 48000, "wstg")
    high, _ = measure_trial([sine(2000)], 48000, "wstg")
    assert high["s1_zcr_mean_fraction"][0] > low["s1_zcr_mean_fraction"][0]
    assert np.isfinite(low["s1_spec_m4_kurtosis"][0])
    assert low["s1_spec_m4_kurtosis"][0] > 0


def test_flux_never_bridges_intervals_or_trials():
    a, audit = measure_trial([sine(300), sine(3000)], 48000, "bamboo_passage")
    one, one_audit = measure_trial([sine(300)], 48000, "bamboo_passage")
    assert audit["n_flux_pairs"] == 2*one_audit["n_flux_pairs"]
    assert a["bamboo_spectral_flux"][0] == pytest.approx(one["bamboo_spectral_flux"][0], abs=1e-7)
    assert audit["n_intervals"] == 2


def test_c068_never_falls_back_to_generic_band_energy():
    results, _ = measure_trial([sine(1000)], 48000, "wstg")
    assert results["s1_spectral_band_1_4khz_percentile1"][1] == "research_pending_source_config"
    assert np.isnan(results["s1_spectral_band_1_4khz_percentile1"][0])



def test_known_power_moments_use_raw_kurtosis():
    # Equally weighted bins at -1, 0, +1: M1=0, M2=sqrt(2/3), M3=0,
    # M4=(2/3)/(2/3)^2=1.5, with no Fisher subtraction.
    m1, m2, m3, m4 = spectral_moments(np.array([[1., 1., 1.]]),
                                      np.array([-1., 0., 1.]))
    assert m1[0] == pytest.approx(0)
    assert m2[0] == pytest.approx(np.sqrt(2/3))
    assert m3[0] == pytest.approx(0)
    assert m4[0] == pytest.approx(1.5)


def test_missing_librosa_is_explicit_not_a_generic_contrast(monkeypatch):
    import builtins
    original = builtins.__import__

    def no_librosa(name, *args, **kwargs):
        if name == "librosa":
            raise ImportError("synthetic missing dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_librosa)
    values, _ = measure_trial([sine(400)], 48000, "sentence_s5")
    assert all(values[f"s5_spectral_contrast_band{i}_db"][1] == "librosa_unavailable"
               for i in range(7))


def test_versioned_family12_registry_matches_algorithm_ids():
    path = Path(__file__).parents[2]/"src/vslp/acoustic/features/data/family12.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    assert {row["feature_id"] for row in source["outputs"]} == {
        row["feature_id"] for row in specs()}
    assert source["algorithm_version"] == "family12-spectral-2.0.1"
    assert source["spectral_slope"]["exclude_dc"] is True
    assert source["spectral_slope"]["include_nyquist"] is True
    assert source["spectral_slope"]["floor_relative_db"] == -120


def test_mfcc_public_indices_are_dct_c0_through_c12():
    from scipy.fft import dct
    from vslp.acoustic.features.family12_v2 import EPS, FFT, FS, WIN, _frames, _mel_filters

    signal = sine(470) + 0.31 * sine(1500)
    frames = _frames(signal, FS)
    power = np.abs(np.fft.rfft(frames * np.hanning(WIN), n=FFT, axis=1)) ** 2
    freq = np.fft.rfftfreq(FFT, 1 / FS)
    coefficients = dct(
        np.log(np.maximum(power @ _mel_filters(freq).T, EPS)),
        type=2, norm="ortho", axis=1,
    )
    values, _ = measure_trial([signal], FS, "sentence_s2")
    for index in range(13):
        assert values[f"s2_mfcc{index + 1:02d}_mean"][0] == pytest.approx(
            np.mean(coefficients[:, index])
        )
        assert values[f"s2_mfcc{index + 1:02d}_sd"][0] == pytest.approx(
            np.std(coefficients[:, index], ddof=1)
        )
    quieter, _ = measure_trial([signal * 0.5], FS, "sentence_s2")
    assert quieter["s2_mfcc01_mean"][0] < values["s2_mfcc01_mean"][0]


def test_slope_zero_bins_are_floored_at_minus_120_db():
    from vslp.acoustic.features.family12_v2 import _spectral_slope_frames

    freq = np.array([0., 1000., 2000., 3000.])
    magnitude = np.array([[8., 0., 2., 0.]])
    levels = 20 * np.log10(np.maximum(magnitude[0, 1:] / 2, 1e-6))
    assert np.min(levels) == pytest.approx(-120)
    assert np.max(levels) == pytest.approx(0)
    slope = _spectral_slope_frames(magnitude, freq)
    assert np.isfinite(slope[0])
    assert slope[0] == pytest.approx(np.polyfit(freq[1:] / 1000, levels, 1)[0])


def test_slope_is_gain_invariant_and_matches_trial_measurement():
    from vslp.acoustic.features.family12_v2 import FFT, FS, WIN, _frames, _spectral_slope_frames

    signal = sine(500) + 0.2 * sine(2800)
    values = []
    for gain in (0.1, 1.0, 10.0):
        result, _ = measure_trial([gain * signal], FS, "bamboo_passage")
        values.append(result["bamboo_spectral_slope_db_khz"][0])
    assert values[0] == pytest.approx(values[1], abs=1e-8)
    assert values[2] == pytest.approx(values[1], abs=1e-8)
    frames = _frames(signal, FS)
    magnitude = np.abs(np.fft.rfft(frames * np.hanning(WIN), n=FFT, axis=1))
    freq = np.fft.rfftfreq(FFT, 1 / FS)
    assert len(freq) == 1025 and freq[0] == 0 and freq[-1] == 24000
    assert values[1] == pytest.approx(np.median(_spectral_slope_frames(magnitude, freq)))


def test_slope_recovers_known_linear_db_per_khz():
    from vslp.acoustic.features.family12_v2 import _spectral_slope_frames

    freq = np.array([0., 1000., 2000., 3000., 4000.])
    known_slope = -7.5
    magnitude = np.concatenate(([1.], 10 ** (known_slope * (freq[1:] / 1000) / 20)))
    actual = _spectral_slope_frames(magnitude[None, :], freq)
    assert actual[0] == pytest.approx(known_slope, abs=1e-12)


def test_slope_excludes_dc_and_includes_nyquist():
    from vslp.acoustic.features.family12_v2 import _spectral_slope_frames

    freq = np.array([0., 1000., 2000., 3000.])
    magnitude = np.array([[1., 1., 0.4, 0.01]])
    baseline = _spectral_slope_frames(magnitude, freq)[0]
    dc_changed = magnitude.copy()
    dc_changed[0, 0] = 1e12
    assert _spectral_slope_frames(dc_changed, freq)[0] == pytest.approx(baseline)
    nyquist_changed = magnitude.copy()
    nyquist_changed[0, -1] = 0.8
    assert _spectral_slope_frames(nyquist_changed, freq)[0] != pytest.approx(baseline)


def test_c070_c071_public_ownership_has_no_duplicate_moments():
    rows = specs()
    for task in ("wstg", "sentence_s2", "sentence_s7"):
        c070 = [row for row in rows if row["task_id"] == task and row["construct_id"] == "C070"]
        c071 = [row for row in rows if row["task_id"] == task and row["construct_id"] == "C071"]
        assert len(c070) == 4
        assert len(c071) == 4
        assert all(
            not any(name in row["feature_id"] for name in ("centroid", "spread", "skew", "kurtosis"))
            for row in c071
        )
    assert len([row for row in rows if row["task_id"] == "bamboo_passage"
                and row["construct_id"] == "C071"]) == 8
