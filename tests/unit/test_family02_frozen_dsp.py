"""Synthetic checks of F02 source-lineage DSP and run boundaries."""

import json

import numpy as np
from scipy import signal

from vslp.acoustic.features.family02 import (
    calculate_voice_quality, trial_perturbation_from_runs,
)
from vslp.acoustic.features.family02_dsp import butterworth_sos, filter_response
from vslp.acoustic.features.family02_filter_qa import generate_filter_qa
from vslp.acoustic.features.family02_research import (
    GNE_REPLICATION, HARMONIC_REPLICATION, PVI_REPLICATION,
    gne_frames, harmonic_statistics, harmonic_windows, pvi_replication,
)
from vslp.acoustic.features.family01 import F0Track
from vslp.acoustic.features.family01_stage import _f02_trial_values


def test_filters_are_stable_and_attenuate_outside_passband(tmp_path):
    pvi = butterworth_sos(3, (9, 14), 200, "bandpass")
    frequency, db = filter_response(pvi, 200)
    def at(hz):
        return db[np.argmin(abs(frequency - hz))]
    assert at(11) > at(4) + 15
    assert at(11) > at(25) + 15
    for cutoff, kind in ((1000., "lowpass"), ((1000., 2000.), "bandpass"),
                         ((2000., 3000.), "bandpass")):
        sos = butterworth_sos(4, cutoff, 8000, kind)
        assert np.isfinite(signal.sosfilt(sos, np.ones(1000))).all()
    image, metadata = generate_filter_qa(tmp_path)
    assert image.stat().st_size > 1000
    rows = json.loads(metadata.read_text())["filters"]
    assert len(rows) == 4
    assert all(row["maximum_pole_radius"] < 1 for row in rows)


def test_perturbation_terms_do_not_cross_run_gaps():
    first = np.array([.005, .0052, .0049, .0051])
    second = np.array([.006, .0062, .0059, .0061])
    split = trial_perturbation_from_runs([first, second], [])
    joined = trial_perturbation_from_runs([np.r_[first, second]], [])
    assert split["jitter_absolute_s"] < joined["jitter_absolute_s"]
    assert np.isclose(split["dfp_pct"], 100 * 4 / 6)
    assert np.isnan(split["jitter_ppq5_pct"])
    amp = trial_perturbation_from_runs([], [np.array([1., 1.1, .9]),
                                           np.array([2., 2.1, 1.9])])
    assert np.isnan(amp["shimmer_apq5_pct"])


def test_gne_pvi_and_harmonic_source_geometry():
    assert GNE_REPLICATION["internal_sample_rate_hz"] == 8000
    assert GNE_REPLICATION["frame_length_ms"] == 30
    assert GNE_REPLICATION["frame_step_ms"] == 10
    assert GNE_REPLICATION["lpc_order"] == 10
    assert GNE_REPLICATION["bandwidth_hz"] == 1000
    assert PVI_REPLICATION["filter_order"] == 3
    time = np.arange(400) / 200
    flat = np.full(400, 180.)
    inside = flat * (1 + .03 * np.sin(2 * np.pi * 11 * time))
    outside = flat * (1 + .03 * np.sin(2 * np.pi * 4 * time))
    assert pvi_replication(inside, time)[0] > 5 * pvi_replication(outside, time)[0]
    audio_time = np.arange(44100) / 44100
    audio = .2 * np.sin(2 * np.pi * 180 * audio_time)
    assert len(gne_frames(audio, 44100)) > 20
    assert HARMONIC_REPLICATION["window"] == "Hamming"
    assert HARMONIC_REPLICATION["spectral_samples"] == 4096
    pulses = np.arange(1 / 180, 1, 1 / 180)
    windows = harmonic_windows(audio, 44100, pulses)
    assert len(windows) == len(range(0, len(pulses) - 8, 7))
    assert harmonic_statistics(windows)["h1_mean"] <= 0


def test_native_rates_and_explicit_unsupported_rate():
    for sr in (44100, 48000):
        time = np.arange(sr) / sr
        sound = .2 * np.sin(2 * np.pi * 180 * time)
        values, audit = calculate_voice_quality(sound, sr, {"hnr_mean_db", "gne_mean"})
        assert np.isfinite(values["hnr_mean_db"][0])
        assert np.isfinite(values["gne_mean"][0])
        assert len(audit["gne_frame_values"]) > 20
    low = np.zeros(8000)
    values, _ = calculate_voice_quality(low, 8000, {"hnr_mean_db"})
    assert values["hnr_mean_db"][1] == "native_sample_rate_below_16000_hz"


def test_single_run_primitive_aggregation_agrees_with_praat_on_synthetic_voice():
    sr = 44100
    time = np.arange(sr * 2) / sr
    sound = (.2 * (1 + .2 * np.sin(2 * np.pi * 4 * time)) *
             np.sin(2 * np.pi * 180 * time + .1 * np.sin(2 * np.pi * 5 * time)))
    selected = {"jitter_local_pct", "jitter_rap_pct", "jitter_ppq5_pct",
                "shimmer_local_pct", "shimmer_apq3_pct", "shimmer_local_db"}
    praat, audit = calculate_voice_quality(sound, sr, selected)
    primitive = trial_perturbation_from_runs([audit["period_sec"]],
                                             [audit["amplitude_values"]])
    for key in selected:
        assert np.isclose(primitive[key], praat[key][0], rtol=.001)


def test_harmonic_global_normalization_uses_all_windows():
    first = np.ones((2, 8))
    second = np.ones((2, 8)) * 10
    combined = harmonic_statistics(np.concatenate((first, second)))
    assert np.isclose(combined["h1_mean"], -10)
    assert np.isclose(combined["h1_sd"], np.std([-20, -20, 0, 0], ddof=1))


def test_trial_hnr_uses_all_frames_not_mean_of_island_means(monkeypatch):
    import vslp.acoustic.features.family01_stage as stage

    sr = 16000
    audio = np.r_[np.ones(sr // 5), np.zeros(sr // 5), np.ones(sr // 5) * 2]
    tracks = [F0Track(np.array([.01]), np.array([180.]), np.array([True]),
                      sr, 0., .2, "synthetic"),
              F0Track(np.array([.41]), np.array([180.]), np.array([True]),
                      sr, .4, .6, "synthetic")]

    def measured(segment, _sr, selected, minimum_usable_cycles=100):
        frame_values = (np.array([10.]) if segment[0] == 1 else
                        np.array([30., 30., 30.]))
        return ({"hnr_mean_db": (float(np.mean(frame_values)), "")},
                {"n_cycles": 100, "n_pulses": 101,
                 "hnr_valid_frames": len(frame_values),
                 "hnr_frame_values": frame_values})

    monkeypatch.setattr(stage, "calculate_voice_quality", measured)
    values, _, _, _ = _f02_trial_values(tracks, audio, sr, {"hnr_mean_db"}, True)
    assert values["hnr_mean_db"] == (25., "")
