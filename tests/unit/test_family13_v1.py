"""Synthetic, nonclinical checks for the F13 operational components."""

import numpy as np
import pandas as pd
import pytest

from vslp.acoustic.features.catalog import load_feature_catalog
from vslp.acoustic.features.family13_v1 import (
    PUBLIC_IDS, RQA_PROFILE, WPD_PROFILE, delta_mfcc, greenwood_edges,
    mfcc_trajectory, modulation_depth, ppe_from_f0, phase_synchrony,
    rhythm_components, rqa_components, rqa_determinism, specs, tsfel_entropy,
    visibility_10ms, wpd_components,
)
from vslp.acoustic.features.family13_v1_stage import (
    _bamboo_sentence_indices, _reviewed_chunks, _sentence_spans,
)
from vslp.acoustic.alignment.final_contract import canonical_units
import json
from pathlib import Path
from vslp.acoustic.features.mixed_dispatch import _executors


def test_registry_exact_ownership_and_task_map():
    catalog = load_feature_catalog()
    rows = [row for row in catalog["outputs"] if row["family_id"] == "F13"]
    active = [row for row in rows if row["selectable"]]
    assert len(PUBLIC_IDS) == len(active) == 87
    assert len({row["construct_id"] for row in active}) == 8
    assert {c: sum(row["construct_id"] == c for row in active)
            for c in (f"C{i:03d}" for i in range(72, 80))} == {
                "C072": 1, "C073": 2, "C074": 8, "C075": 16,
                "C076": 26, "C077": 13, "C078": 12, "C079": 9}
    assert all(row["recommended_tasks"] == [row["task_id"]] for row in active)
    assert all(row["prerequisites"]["requires_alignment"] ==
               (row["task_id"] == "bamboo_passage") for row in active)
    assert "vowel_sample_entropy_pending_definition" in {
        row["feature_id"] for row in rows if not row["selectable"]}
    assert "signal_entropy" not in PUBLIC_IDS
    assert "F13" in next(families for families, _ in _executors().values()
                         if "F13" in families)
    assert len(specs()) == 87


def test_ppe_fixed_bin_order_two_and_failure():
    rng = np.random.default_rng(4)
    stable = np.full(120, 160.0)
    irregular = 160 * np.power(2, rng.normal(0, 0.02, 120))
    low, reason, audit = ppe_from_f0(stable)
    high, high_reason, _ = ppe_from_f0(irregular)
    assert not reason and not high_reason
    assert 0 <= low < high <= np.log2(31)
    assert audit["lpc_order"] == 2
    assert audit["n_residuals"] == 118
    assert ppe_from_f0(stable, tracking_yield=0.1)[1] == "unreliable_f0"
    assert ppe_from_f0(stable[:10])[1] == "insufficient_voiced_f0"


def test_wpd_profile_and_missing_library_status():
    assert WPD_PROFILE == {"wavelet": "db4", "level": 3,
                           "mode": "symmetric", "node_order": "freq"}
    x = np.sin(2 * np.pi * 320 * np.arange(4800) / 48000)
    value, reason, _ = wpd_components([x])
    if reason == "pywt_unavailable":
        assert value == {}
    else:
        scaled, second_reason, _ = wpd_components([10 * x])
        assert not reason and not second_reason
        assert len(value["shares"]) == 8
        assert sum(value["shares"]) == pytest.approx(1)
        assert value["shares"] == pytest.approx(scaled["shares"])
        assert value["entropy_norm"] == pytest.approx(scaled["entropy_norm"])


def test_tsfel_missing_is_explicit_and_no_fallback():
    value, reason, _ = tsfel_entropy(np.ones(100))
    if reason == "tsfel_unavailable":
        assert np.isnan(value)


def test_rqa_delta_and_sentence_contract():
    assert RQA_PROFILE["m"] == 3 and RQA_PROFILE["tau"] == 5
    assert RQA_PROFILE["eps"] == 0.2 and RQA_PROFILE["l_min"] == 2
    assert _bamboo_sentence_indices() == [
        (1, 8), (9, 16), (17, 30), (31, 45), (46, 60),
        (61, 75), (76, 91), (92, 97)]
    assert delta_mfcc(np.zeros((100, 13))).shape == (100, 13)
    assert rqa_determinism(np.ones(100))[1] == "zero_mfcc_variation"
    periodic = np.sin(np.arange(100) * 2 * np.pi / 10)
    value, reason = rqa_determinism(periodic)
    assert not reason and 0 <= value <= 1
    random_value, random_reason = rqa_determinism(
        np.random.default_rng(5).normal(size=100))
    assert not random_reason and value > random_value


def test_modulation_and_visibility_components():
    t = np.arange(500) / 100
    theta, reason = modulation_depth(1 + 0.5 * np.sin(2 * np.pi * 5 * t))
    assert not reason
    assert theta["theta"] > theta["delta"]
    delta, reason = modulation_depth(1 + 0.5 * np.sin(2 * np.pi * 1.5 * t))
    assert not reason
    assert delta["delta"] > delta["theta"]
    fast, reason = modulation_depth(1 + 0.5 * np.sin(2 * np.pi * 20 * t))
    assert not reason and fast["beta_gamma"] > fast["theta"]
    psi, reason = phase_synchrony([np.sin(2 * np.pi * 2 * t) +
                                   np.sin(2 * np.pi * 4 * t) +
                                   np.sin(2 * np.pi * 12 * t)])
    assert not reason and all(0 <= item <= 1 for item in psi.values())
    long_t = np.arange(1000) / 100
    locked, reason = phase_synchrony([np.sin(2 * np.pi * 5 * long_t) +
                                      np.sin(2 * np.pi * 15 * long_t)])
    assert not reason and locked["theta_beta_gamma"] > 0.9
    assert len(greenwood_edges()) == 29
    assert greenwood_edges()[[0, -1]] == pytest.approx([100, 10000])
    noise = np.random.default_rng(8).normal(size=48000)
    density, reason, audit = visibility_10ms([noise[:24000], noise[24000:]])
    assert not reason and 0 <= density <= 1
    assert audit["n_windows"] == 100


def test_bamboo_spans_use_canonical_words_and_original_seconds():
    path = Path(__file__).parents[2] / "src/vslp/acoustic/alignment/task_prompts.json"
    tasks = json.loads(path.read_text(encoding="utf-8"))["tasks"]
    prompt = next(t for t in tasks if t["task_id"] == "bamboo_passage")["prompts"][0]
    tokens = canonical_units(prompt["exact_expected_text"])
    class Alignment:
        def words_for_trial(self, recording_id, trial_id):
            assert (recording_id, trial_id) == ("r1", "t1")
            return pd.DataFrame({"canonical_word_index": range(1, 98),
                                 "canonical_word_text": tokens,
                                 "start_sec": np.arange(97) * 0.2 + 4,
                                 "end_sec": np.arange(97) * 0.2 + 4.15})
    spans = _sentence_spans(Alignment(), "r1", "t1")
    assert len(spans) == 8
    assert spans[0]["start_sec"] == 4
    assert spans[-1]["end_sec"] == pytest.approx(23.35)


def test_reviewed_chunks_do_not_join_across_hard_exclusion():
    audio = np.arange(48000, dtype=float)
    frame = pd.DataFrame([
        {"segment_role": "speech", "start_sec": 0.0, "end_sec": 1.0},
        {"segment_role": "manual_exclusion", "start_sec": 0.4, "end_sec": 0.6},
    ])
    spans, chunks = _reviewed_chunks(audio, 48000, frame, 0, 1)
    assert spans == [(0.0, 0.4), (0.6, 1.0)]
    assert len(chunks) == 2
    assert chunks[0][-1] < chunks[1][0]


def test_installed_tsfel_pywavelets_and_librosa_numerical_paths():
    from importlib import metadata

    import librosa
    import pywt

    rng = np.random.default_rng(17)
    signal = rng.normal(size=48000)
    entropy, reason, audit = tsfel_entropy(signal)
    assert not reason and entropy == pytest.approx(1)
    assert audit["tsfel_version"] == metadata.version("tsfel")
    assert tsfel_entropy(np.ones(1000))[0] == pytest.approx(0)
    wpd, reason, profile = wpd_components([signal])
    assert not reason and len(wpd["shares"]) == 8
    assert sum(wpd["shares"]) == pytest.approx(1)
    assert np.isfinite(wpd["entropy_norm"]).all()
    assert np.all((0 <= wpd["entropy_norm"]) & (wpd["entropy_norm"] <= 1))
    assert profile["wavelet"] == "db4" and profile["level"] == 3
    assert profile["mode"] == "symmetric" and profile["node_order"] == "freq"
    assert profile["pywt_version"] == pywt.__version__
    track = mfcc_trajectory(signal)
    assert track.shape == (98, 13)  # F12-compatible C0..C12, 25/10 ms framing
    expected = librosa.feature.delta(track.T, width=5, order=1, mode="nearest").T
    assert delta_mfcc(track) == pytest.approx(expected, abs=1e-12)
    raw, raw_reasons, _ = rqa_components([signal])
    delta, delta_reasons, _ = rqa_components([signal], delta=True)
    assert sum(not item for item in raw_reasons) >= 10
    assert sum(not item for item in delta_reasons) >= 10
    assert np.isfinite(raw).sum() >= 10 and np.isfinite(delta).sum() >= 10


def test_all_rhythm_and_phase_outputs_execute_with_actual_filterbank():
    t = np.arange(4 * 48000) / 48000
    carriers = sum(np.sin(2 * np.pi * f * t) for f in (200, 500, 2000, 5000))
    envelope = (1 + .4 * np.sin(2 * np.pi * 5 * t) +
                .3 * np.sin(2 * np.pi * 1.5 * t) +
                .2 * np.sin(2 * np.pi * 20 * t))
    values, reason, audit = rhythm_components([envelope * carriers])
    assert not reason and len(values) == 20
    assert sum(key.startswith("bamboo_moddepth_") for key in values) == 12
    assert sum(key.startswith("bamboo_psi_") for key in values) == 8
    assert all(np.isfinite(value) and 0 <= value <= 1 for value in values.values())
    assert len(audit["greenwood_edges_hz"]) == 29


def test_fft_zero_padding_denominator_and_no_silent_truncation():
    series = 1 + .5 * np.sin(2 * np.pi * 5 * np.arange(100) / 100)
    values, reason = modulation_depth(series)
    assert not reason and values["theta"] > values["delta"]
    spectrum = abs(np.fft.rfft(series * np.hamming(len(series)), n=2048)) ** 2
    frequencies = np.fft.rfftfreq(2048, 1 / 100)
    expected = spectrum[(frequencies >= 2.5) & (frequencies < 12)].sum() / (
        spectrum[frequencies > 0].sum())
    assert values["theta"] == pytest.approx(expected)
    assert modulation_depth(np.ones(2049))[1] == "modulation_sequence_exceeds_fft_support"


def test_psi_orientation_correct_harmonics_beat_wrong_harmonics():
    t = np.arange(1000) / 100
    right_2 = phase_synchrony([np.sin(2 * np.pi * 2 * t) +
                               np.sin(2 * np.pi * 4 * t)])[0]["delta_theta"]
    wrong_2 = phase_synchrony([np.sin(2 * np.pi * 2 * t) +
                               np.sin(2 * np.pi * 5 * t)])[0]["delta_theta"]
    right_3 = phase_synchrony([np.sin(2 * np.pi * 5 * t) +
                               np.sin(2 * np.pi * 15 * t)])[0]["theta_beta_gamma"]
    wrong_3 = phase_synchrony([np.sin(2 * np.pi * 5 * t) +
                               np.sin(2 * np.pi * 17 * t)])[0]["theta_beta_gamma"]
    assert right_2 > .9 > wrong_2
    assert right_3 > .9 > wrong_3
