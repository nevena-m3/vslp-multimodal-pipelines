import numpy as np

from vslp.acoustic.features.family12 import (
    FAMILY12_IDS, calculate_spectral_features, frame_signal, spectral_contrast_band_edges,
)


def test_family12_framing_and_exact_mfcc_ids():
    assert frame_signal(np.zeros(2160)).shape == (3, 1200)
    assert {f"mfcc{i:02d}_mean" for i in range(1, 14)} <= FAMILY12_IDS
    assert {f"mfcc{i:02d}_sd" for i in range(1, 14)} <= FAMILY12_IDS


def test_family12_tone_bandwidth_and_native_audio_immutability():
    sr = 16_000
    time = np.arange(sr) / sr
    audio = np.sin(2 * np.pi * 1000 * time)
    original = audio.copy()
    result, audit = calculate_spectral_features(audio, sr)
    assert np.array_equal(audio, original)
    assert audit.working_sample_rate_hz == 48_000
    assert audit.resampled is True
    assert audit.mfcc_matrix.shape[1] == 13
    assert result["spectral_centroid_hz"][0] == result["spec_m1_hz"][0]
    assert result["spectral_bandwidth_p2_mean_hz"][0] < 1000


def test_family12_noise_has_greater_flatness_and_short_audio_fails():
    rng = np.random.default_rng(9)
    sr = 48_000
    time = np.arange(sr) / sr
    tone, _ = calculate_spectral_features(np.sin(2*np.pi*500*time), sr)
    noise, _ = calculate_spectral_features(rng.normal(size=sr), sr)
    assert noise["spectral_flatness"][0] > tone["spectral_flatness"][0]
    short, audit = calculate_spectral_features(np.ones(2000), sr)
    assert audit.n_frames < 10
    assert short["mfcc01_mean"][1] == "insufficient_spectral_frames"
    assert short["spectral_contrast_band6_db"][1] == "insufficient_spectral_frames"


def test_spectral_contrast_has_seven_valid_48khz_bands_and_tonal_behavior():
    edges = spectral_contrast_band_edges(48_000)
    assert np.array_equal(edges, [0, 200, 400, 800, 1600, 3200, 6400, 24000])
    sr = 48_000
    time = np.arange(sr) / sr
    rng = np.random.default_rng(17)
    tonal, _ = calculate_spectral_features(np.sin(2*np.pi*1000*time), sr)
    noisy, _ = calculate_spectral_features(rng.normal(size=sr), sr)
    identifiers = [f"spectral_contrast_band{band}_db" for band in range(7)]
    assert set(identifiers) <= FAMILY12_IDS
    assert all(tonal[feature_id][1] == "" for feature_id in identifiers)
    assert np.mean([tonal[feature_id][0] for feature_id in identifiers]) > np.mean(
        [noisy[feature_id][0] for feature_id in identifiers])
