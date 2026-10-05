"""Nonclinical F02 replication and task-contract checks."""

import numpy as np

from vslp.acoustic.features.catalog import load_feature_catalog, task_fit
from vslp.acoustic.features.family01_task import task_context
from vslp.acoustic.features.family02 import (
    FAMILY02_IDS, HARMONIC_IDS, _dfp_contiguous_valid_periods, calculate_voice_quality,
    directional_perturbation_factor,
)
from vslp.acoustic.features.family02_research import (
    CPP_PROFILE, GNE_REPLICATION, HARMONIC_REPLICATION, PVI_REPLICATION,
    cpp_unsmoothed, gne_replication, harmonic_replication, pvi_replication,
)


def _harmonic_tone(sr=48000, seconds=2):
    time = np.arange(sr * seconds) / sr
    audio = sum(.2 / harmonic * np.sin(2 * np.pi * 180 * harmonic * time)
                for harmonic in range(1, 9))
    return audio, sr


def test_nine_constructs_task_recommendations_and_identity():
    outputs = [item for item in load_feature_catalog()["outputs"]
               if item["family_id"] == "F02"]
    by_id = {item["feature_id"]: item for item in outputs}
    assert {item["construct_id"] for item in outputs} == {
        *(f"C{number:03d}" for number in range(5, 13)), "C080"}
    assert set(FAMILY02_IDS) <= {key for key, item in by_id.items() if item["selectable"]}
    for feature_id in ("jitter_local_pct", "shimmer_local_pct", "hnr_mean_db",
                       "dfp_pct", "h1_mean"):
        assert task_fit(by_id[feature_id], "Sustained /a/") == "Recommended"
        assert task_fit(by_id[feature_id], "S5") == "Recommended"
        assert task_fit(by_id[feature_id], "S6") != "Recommended"
    assert task_fit(by_id["cpp_db"], "S6") == "Recommended"
    assert task_fit(by_id["cpp_db"], "Bamboo Passage") == "Recommended"
    assert task_fit(by_id["gne_source_replication_only"], "S5") != "Recommended"
    assert by_id["jitter_local_pct"]["executable_for_task"]["sentence:S6"]
    assert by_id["gne_source_replication_only"]["executable_for_task"]["sentence:S5"]
    assert task_context({"task_id": "sentence_s6", "task_name": "S6",
                         "task_type": "Sentence / controlled speech"})["stimulus_text"] == (
                             "My mama makes lemon jam")
    assert by_id["cpp_db"]["selectable"]
    assert by_id["cpp_db"]["recommended_tasks"] == [
        "sustained_a", "sentence_s5", "sentence_s6", "bamboo_passage"]
    assert not by_id["cpps_db"]["selectable"]


def test_gne_experimental_filter_profile_and_failure():
    audio, sr = _harmonic_tone()
    coherent = gne_replication(audio, sr)
    noise = np.random.default_rng(17).normal(0, .2, audio.size)
    noisy = gne_replication(noise, sr)
    assert np.isfinite(coherent) and coherent > noisy
    assert GNE_REPLICATION["band_centers_hz"] == [500, 1500, 2500]
    assert GNE_REPLICATION["filter_order"] == 4
    assert np.isnan(gne_replication(np.zeros(sr // 20), sr))


def test_pvi_source_band_and_explicit_missing_f0_policy():
    time = np.arange(400) / 200
    flat = np.full(time.size, 180.)
    inside = flat + 5 * np.sin(2 * np.pi * 11 * time)
    outside = flat + 5 * np.sin(2 * np.pi * 4 * time)
    strong, fraction = pvi_replication(inside, time)
    weak, _ = pvi_replication(outside, time)
    baseline, _ = pvi_replication(flat, time)
    assert strong > weak > baseline
    assert fraction == 0
    missing = inside.copy()
    missing[100:110] = np.nan
    value, missing_fraction = pvi_replication(missing, time)
    assert np.isnan(value) and missing_fraction > 0
    assert PVI_REPLICATION["welch_window_sec"] == 1
    assert PVI_REPLICATION["welch_overlap_fraction"] == .95


def test_unsmoothed_cpp_is_separate_from_cpps():
    audio, sr = _harmonic_tone()
    value, frames = cpp_unsmoothed(audio, sr)
    assert np.isfinite(value) and frames > 100
    assert CPP_PROFILE["smoothing"] == "none"
    assert CPP_PROFILE["pitch_floor_hz"] == 60
    assert CPP_PROFILE["pitch_ceiling_hz"] == 500
    assert CPP_PROFILE["time_step_sec"] == .002
    assert CPP_PROFILE["maximum_frequency_hz"] == 5000
    assert np.isnan(cpp_unsmoothed(audio[:100], sr)[0])


def test_pitch_synchronous_harmonics_preserve_24_components():
    audio, sr = _harmonic_tone()
    pulses = np.arange(1 / 180, 2, 1 / 180)
    values = harmonic_replication(audio, sr, pulses)
    assert set(values) == HARMONIC_IDS
    assert values["h1_mean"] > values["h8_mean"]
    assert values["h8_mean"] < 0
    assert np.isclose(values["h1_relh"], 1 / (
        abs(values["h1_mean"]) + values["h1_sd"]))
    assert HARMONIC_REPLICATION["periods_per_window"] == 8
    assert HARMONIC_REPLICATION["spectral_samples"] == 4096
    assert HARMONIC_REPLICATION["window_hop_periods"] == 7


def test_dfp_zero_differences_do_not_reverse_and_failure_is_nan():
    assert directional_perturbation_factor(np.array([.01, .011, .011, .01])) == 0
    assert _dfp_contiguous_valid_periods(np.array([
        .01, .011, .010, .012, .1, .01, .011, .010, .012])) == 100 * 4 / 6
    audio = np.zeros(4800)
    results, _ = calculate_voice_quality(audio, 48000, {"jitter_local_pct"})
    assert np.isnan(results["jitter_local_pct"][0])
    assert results["jitter_local_pct"][1]
