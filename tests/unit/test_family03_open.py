"""Deterministic, nonclinical checks of the F03 open review build."""

from types import SimpleNamespace

import numpy as np
import pytest

from vslp.acoustic.features.catalog import feature_availability, load_feature_catalog, task_fit
from vslp.acoustic.features.family03 import (
    amplitude_tremor_trial, cycle_amplitude_instability_pct,
    cycle_waveform_instability, mfdr_naq, phase_quotients, vibration_blockage,
)
from vslp.acoustic.features.family03_gif import (
    _ame_weights, cycles_from_estimate, qcp_glottal_estimate,
)


def test_active_constructs_and_proprietary_non_equivalence():
    catalog = load_feature_catalog()
    parents = {c["construct_id"]: c for c in catalog["constructs"]
               if c["family_id"] == "F03" and c.get("active_in_gui")}
    assert set(parents) == {"C013", "C014", "C016", "C020"}
    assert all(parents[cid]["outputs"] for cid in parents)
    for cid in ("C015", "C017", "C018", "C019"):
        assert not next(c for c in catalog["constructs"]
                        if c["construct_id"] == cid)["active_in_gui"]
    outputs = [o for o in catalog["outputs"] if o["family_id"] == "F03"]
    assert len(outputs) == 11
    assert all(not o["source_equivalent"] for o in outputs)
    assert sum(feature_availability(o)[0] == "Implemented" for o in outputs) == 10
    assert all(task_fit(o, "Sustained /a/") == "Recommended" for o in outputs
               if o["selectable"])
    assert all(task_fit(o, task) != "Recommended" for o in outputs
               for task in ("S5", "S6", "Bamboo Passage", "DDK"))
    mdat = next(o for o in outputs if o["feature_id"] == "amplitude_tremor_mdat_pct")
    assert not mdat["selectable"] and "2-s" in mdat["blocked_reason"]


def test_mfdr_naq_exact_formula_and_scaling():
    flow = np.array([0., 1., 2., 1., 0.])
    derivative = np.array([0., 1., 1., -2., 0.])
    mfdr, naq = mfdr_naq(flow, derivative, .01)
    assert mfdr == 2
    assert naq == pytest.approx(100.)
    twice, naq_scaled = mfdr_naq(2 * flow, 2 * derivative, .01)
    assert twice == 4 and naq_scaled == pytest.approx(naq)
    assert np.isnan(mfdr_naq(flow, np.zeros_like(flow), .01)[1])


def test_phase_landmarks_and_invalid_order():
    result = phase_quotients(np.array([0., 0., 0., .1, .4, 1., .8, .5, .2, 0.]))
    assert result["glottal_open_quotient"] == pytest.approx(.7)
    assert result["glottal_closed_quotient"] == pytest.approx(.3)
    assert result["glottal_closing_quotient"] == pytest.approx(.5)
    assert result["glottal_speed_quotient"] == pytest.approx(.4)
    assert np.isnan(phase_quotients(np.ones(20))["glottal_open_quotient"])


def test_instability_is_distinct_from_acoustic_jitter_shimmer():
    x = np.sin(np.linspace(0, 2 * np.pi, 128))
    same = [{"cycle_id": i, "flow": x.copy()} for i in range(1, 8)]
    distorted = [{"cycle_id": i, "flow": x + .1 * i * np.sin(3 *
                  np.linspace(0, 2 * np.pi, 128))} for i in range(1, 8)]
    assert cycle_waveform_instability(same) == pytest.approx(0, abs=1e-12)
    assert cycle_waveform_instability(distorted) > 0
    separated = [same[0], {"cycle_id": 3, "flow": -x}]
    assert np.isnan(cycle_waveform_instability(separated))
    assert cycle_amplitude_instability_pct(same) == pytest.approx(0)
    variable = [{"cycle_id": i, "flow": x * scale}
                for i, scale in enumerate([1, 1.1, .9, 1.2, 1, 8], 1)]
    assert 0 < cycle_amplitude_instability_pct(variable) < 30


def test_qcp_lineage_finite_synthetic_flow_and_ame():
    sr = 48000
    time = np.arange(sr) / sr
    audio = .2 * np.sin(2 * np.pi * 180 * time) + .08 * np.sin(2 * np.pi * 360 * time)
    estimate = qcp_glottal_estimate(audio, sr, 0)
    valid, rejected = cycles_from_estimate(estimate)
    assert estimate.native_sample_rate_hz == sr
    assert estimate.sample_rate_hz == 8000
    assert np.isfinite(estimate.flow).all()
    assert 100 < len(valid) + len(rejected) < 250
    assert estimate.qc_flags and "not_reference_qcp_parity" in estimate.qc_flags
    weights = _ame_weights(800, np.arange(0, 800, 80), 8000)
    assert np.min(weights) == pytest.approx(1e-5)
    assert np.max(weights) == pytest.approx(1.)
    assert np.array_equal(estimate.flow, qcp_glottal_estimate(audio, sr, 0).flow)


def test_short_amplitude_tremor_is_not_mdat_or_pvi():
    sr = 48000
    time = np.arange(2 * sr) / sr
    carrier = np.sin(2 * np.pi * 180 * time)
    flat, _ = amplitude_tremor_trial(.2 * carrier, sr)
    mild, audit = amplitude_tremor_trial(.2 * (1 + .1 * np.sin(2 * np.pi * 4 * time)) *
                                         carrier, sr)
    strong, _ = amplitude_tremor_trial(.2 * (1 + .3 * np.sin(2 * np.pi * 4 * time)) *
                                       carrier, sr)
    short_id = "amplitude_tremor_depth_short_v1_pct"
    assert flat[short_id][0] < mild[short_id][0] < strong[short_id][0]
    assert audit["dominant_tremor_frequency_hz"] == pytest.approx(4., abs=.2)
    assert audit["modulation_sample_rate_hz"] == 50
    assert audit["n_windows"] == 1
    assert np.isnan(mild["amplitude_tremor_mdat_pct"][0])
    assert mild["amplitude_tremor_mdat_pct"][1] == "source_track_duration_unavailable"
    long_time = np.arange(4 * sr) / sr
    long_audio = .2 * (1 + .2 * np.sin(2 * np.pi * 4 * long_time)) * (
        np.sin(2 * np.pi * 180 * long_time))
    long_values, long_audit = amplitude_tremor_trial(long_audio, sr)
    assert np.isfinite(long_values["amplitude_tremor_mdat_pct"][0])
    assert long_audit["n_windows"] > 1
    assert long_audit["dominant_track_duration_sec"] >= 2
    assert long_audit["mode"] == "source_track_lineage"


def test_blockage_rejects_dropout_and_onset_offset():
    sr = 48000
    time = np.arange(sr) / sr
    audio = .2 * np.sin(2 * np.pi * 180 * time)
    frames = np.arange(.0025, .9975, .005)
    valid = np.ones(len(frames), dtype=bool)
    valid[(frames >= .4) & (frames < .425)] = False
    track = SimpleNamespace(voiced_valid=valid, time_sec=frames,
                            f0_hz=np.full(len(frames), 180.),
                            analysis_start_sec=0., analysis_end_sec=1.)
    pulses = np.arange(0, 1, 1 / 180)
    pulses = pulses[(pulses < .4) | (pulses >= .425)]
    value, reason, audit = vibration_blockage(track, audio, sr, pulses)
    assert reason == "" and value > 0 and audit["blockage_count"] == 1
    dropout = audio.copy()
    dropout[round(.4 * sr):round(.425 * sr)] = 0
    value, reason, audit = vibration_blockage(track, dropout, sr, pulses)
    assert np.isnan(value) and reason == "acquisition_dropout_requires_review"
    assert audit["acquisition_dropout_count"] == 1
