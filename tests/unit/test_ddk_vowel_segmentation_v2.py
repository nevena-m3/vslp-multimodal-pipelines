"""DDK (nuclei v2) and sustained-vowel (voicing v2) segmentation on synthetic speech
with known syllables and trials, plus the reviewed DDK event contract for F10."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _speech_synth import ddk, vowel  # noqa: E402

from vslp.acoustic.features.family10_v2 import measure_trial
from vslp.acoustic.segment.ddk_events import ddk_event_rows, ddk_events_frame, ddk_target
from vslp.acoustic.segment.task_methods import (DDKConfig, PhonationConfig, segment_ddk,
                                                segment_phonation)


@pytest.mark.parametrize("syllables", ["pa" * 15, "ta" * 15, "pataka" * 6])
@pytest.mark.parametrize("rate", [4, 8])
@pytest.mark.parametrize("dip_db", [60, 6])  # 6 dB = slurred, incomplete closures
@pytest.mark.parametrize("sr", [16000, 44100])
def test_ddk_counts_every_syllable_once(syllables, rate, dip_db, sr):
    x, truth = ddk(sr=sr, syllables=syllables, rate=rate, dip_db=dip_db, snr_db=15,
                   decay=0.6, seed=rate + dip_db)
    result = segment_ddk(x, sr, DDKConfig())
    assert result["n_events"] == len(truth)
    nuclei = np.array(result["nuclei_samples"]) / sr
    for syllable, nucleus in zip(truth, nuclei):
        assert syllable["vowel_on"] - .03 <= nucleus <= syllable["vowel_off"] + .03


def test_pataka_labels_follow_the_bursts():
    x, truth = ddk(sr=44100, syllables="pataka" * 6, rate=6, snr_db=35)
    result = segment_ddk(x, 44100, DDKConfig(target="pataka"))
    assert result["syllable_labels"] == [s["cons"] + "a" for s in truth]


def test_ddk_ignores_presounds_and_a_second_attempt():
    sr = 16000
    x, truth = ddk(sr=sr, syllables="pa" * 20, rate=6, lead=2.0, seed=3)
    blip = np.zeros_like(x)
    blip[int(.4 * sr):int(.5 * sr)] = .3 * np.sin(2 * np.pi * 200 * np.arange(int(.1 * sr)) / sr)
    second, _ = ddk(sr=sr, syllables="pa" * 4, rate=6, lead=0.2, seed=4)
    signal = np.concatenate([x + blip, second])
    result = segment_ddk(signal, sr, DDKConfig())
    assert result["n_events"] == 20
    assert abs(result["ddk_sequence_start_sec"] - truth[0]["vowel_on"]) < .05
    assert {e["rejection_reason"] for e in result["rejected_events"]} == {"outside_main_ddk_sequence"}


@pytest.mark.parametrize("length", [0.6, 6.0])
def test_empty_or_silent_files_are_flagged_not_measured(length):
    x = np.random.default_rng(0).normal(0, 3e-5, int(16000 * length)).astype(np.float32)
    ddk_result = segment_ddk(x, 16000, DDKConfig())
    assert ddk_result["n_events"] == 0 and "silent_or_empty_recording" in ddk_result["flags"]
    vowel_result = segment_phonation(x, 16000, PhonationConfig())
    assert vowel_result["full_samples"] is None
    assert "silent_or_empty_recording" in vowel_result["flags"]


@pytest.mark.parametrize("case,kwargs", [
    ("three_trials", dict(trials=((1.0, 5.0), (7.0, 11.5), (13.5, 18.0)))),
    ("short_trials", dict(trials=((1.0, 4.3), (6.0, 9.3)))),
    ("voice_break", dict(trials=((1.0, 6.0),), breaks=((3.0, 3.2),))),
    ("tremor_after_4_5s", dict(trials=((1.0, 7.0),), wobble_start=4.5)),
    ("noisy", dict(trials=((1.0, 6.0),), snr_db=12)),
])
def test_vowel_stable_window_lies_in_steady_voicing(case, kwargs):
    sr = 44100
    result = segment_phonation(vowel(sr=sr, **kwargs), sr, PhonationConfig())
    assert result["n_trials"] == len(kwargs["trials"])
    a, b = (value / sr for value in result["stable_samples"])
    assert any(t0 <= a and b <= t1 for t0, t1 in kwargs["trials"])
    assert not any(a < q1 and b > q0 for q0, q1 in kwargs.get("breaks", ()))
    if kwargs.get("wobble_start"):
        assert b <= kwargs["wobble_start"] + .05


def test_truncated_phonation_is_flagged():
    result = segment_phonation(vowel(trials=((1.0, 6.0),), total=6.0), 44100, PhonationConfig())
    assert "phonation_truncated_at_file_end" in result["flags"]


@pytest.mark.parametrize("name,expected", [
    ("P01_270_1_20230101_242_DDK_PA.wav", "pa"), ("P01_270_1_20230101_260_DDK_TA.wav", "ta"),
    ("P01_270_1_20230101_261_DDK_KA.wav", "ka"), ("P01_270_1_20230101_262_DDK_PATAKA.wav", "pataka"),
    ("bamboo.wav", ""),
])
def test_ddk_target_from_file_name(name, expected):
    assert ddk_target(name) == expected


def _events(x, sr, name):
    result = segment_ddk(x, sr, DDKConfig())
    intervals = [(a / sr, b / sr) for a, b in result["intervals_samples"]]
    return ddk_events_frame(ddk_event_rows(x, sr, intervals, recording_id="r1",
                                           file_name=name))


def test_reviewed_pa_events_give_the_true_ddk_rate():
    sr = 16000
    x, truth = ddk(sr=sr, syllables="pa" * 15, rate=6, rate_jitter=0, snr_db=35)
    events = _events(x, sr, "S_270_1_20230101_242_DDK_PA.wav")
    assert events.valid.all() and len(events) == 15
    result, _audit, summary = measure_trial(events, "pa")
    assert summary["n_events_used"] == 10
    # 10 syllables from first onset to tenth offset at 6 syll/s.
    assert result["rate"][0] == pytest.approx(10 / (truth[9]["vowel_off"] - truth[0]["vowel_on"]),
                                              rel=.08)


def test_reviewed_pataka_events_form_triplets_for_f10():
    sr = 44100
    x, _truth = ddk(sr=sr, syllables="pataka" * 12, rate=6, snr_db=35)
    events = _events(x, sr, "S_270_1_20230101_262_DDK_PATAKA.wav")
    complete = events.loc[events.sequence_complete.astype(bool)]
    assert len(complete) == 12 and complete.sequence_label.eq("pa-ta-ka").all()
    result, _audit, summary = measure_trial(events, "pataka")
    assert summary["syllable_equivalent_count"] == 30
    assert 4 < result["rate"][0] < 8


# Failure modes found when v2 was checked on 72 real recordings.

@pytest.mark.parametrize("sr", [16000, 44100])
def test_ddk_rejects_regular_noise_bumps_in_the_silences(sr):
    rng = np.random.default_rng(1)
    x, truth = ddk(sr=sr, syllables="ka" * 15, rate=4, lead=2.0, snr_db=35, seed=1)
    x = x.astype(float)
    noise = np.std(x[: int(0.5 * sr)])
    end = truth[-1]["vowel_off"]
    for start in list(np.arange(0.3, 1.8, 0.15)) + list(np.arange(end + 0.3, end + 0.9, 0.15)):
        n, a = int(0.04 * sr), int(start * sr)
        x[a:a + n] += rng.standard_normal(n) * noise * 10 ** (rng.uniform(3, 10) / 20) * np.hanning(n)
    result = segment_ddk(x.astype(np.float32), sr, DDKConfig())
    assert result["n_events"] == len(truth)


@pytest.mark.parametrize("sr", [16000, 44100])
def test_ddk_does_not_split_slow_syllables_with_two_energy_humps(sr):
    x, truth = ddk(sr=sr, syllables="pa" * 15, rate=2.5, lead=2.0, snr_db=35, seed=1)
    x = x.astype(float)
    t = np.arange(len(x)) / sr
    for syllable in truth:
        mid = (syllable["vowel_on"] + syllable["vowel_off"]) / 2
        width = (syllable["vowel_off"] - syllable["vowel_on"]) * 0.18
        a, b = int((mid - 3 * width) * sr), int((mid + 3 * width) * sr)
        x[a:b] *= 1 - 0.45 * np.exp(-0.5 * ((t[a:b] - mid) / width) ** 2)
    result = segment_ddk(x.astype(np.float32), sr, DDKConfig())
    assert result["n_events"] == len(truth)


@pytest.mark.parametrize("sr", [16000, 44100])
def test_vowel_stable_window_avoids_low_level_hum_at_another_pitch(sr):
    # The vowel (210 Hz) is followed and preceded by a steady 100 Hz hum 18 dB
    # lower; the old score picked the hum because its F0 is flatter.
    a = vowel(sr=sr, trials=((1.7, 4.7),), total=9.0, f0=210, jitter=0.015, shimmer=0.08, seed=3)
    hum = vowel(sr=sr, trials=((4.6, 8.3),), total=9.0, f0=100, jitter=0.002, shimmer=0.01,
                onset_ramp=0.05, seed=4)
    pre = vowel(sr=sr, trials=((0.7, 1.7),), total=9.0, f0=100, jitter=0.002, shimmer=0.01,
                onset_ramp=0.05, seed=5)
    x = (a.astype(float) + 10 ** (-18 / 20) * (hum + pre)).astype(np.float32)
    result = segment_phonation(x, sr, PhonationConfig())
    full = np.array(result["full_samples"]) / sr
    stable = np.array(result["stable_samples"]) / sr
    assert 1.5 < full[0] < 1.95 and 4.4 < full[1] < 4.9
    assert full[0] <= stable[0] and stable[1] <= full[1]


def test_vowel_octave_tracking_errors_are_not_trimmed_as_off_pitch():
    # Real-data check, TCV005: Praat read a steady 260 Hz vowel as 130 Hz for
    # its middle 5 s. The whole vowel must stay, up to the file end.
    from vslp.acoustic.segment.phonation_v2 import _stable_window, _trim_trial

    config = PhonationConfig()
    f0 = np.full(700, 260.0)
    f0[110:580] = 130.0
    intensity = np.full(700, -20.0)
    assert _trim_trial(f0, intensity, 0, 700, config) == (0, 700)
    window = _stable_window(f0, intensity, 0, 700, config)
    assert window["qualified"] and window["f0_sd_semitones"] < 0.5


def test_vowel_weak_hum_at_another_pitch_is_trimmed():
    from vslp.acoustic.segment.phonation_v2 import _trim_trial

    f0 = np.r_[np.full(300, 210.0), np.full(300, 100.0)]
    intensity = np.r_[np.full(300, -20.0), np.full(300, -38.0)]
    a, b = _trim_trial(f0, intensity, 0, 600, PhonationConfig())
    assert a == 0 and 295 <= b <= 300


@pytest.mark.parametrize("sr", [16000, 44100])
def test_ddk_rejects_a_sustained_hum_before_the_sequence(sr):
    x, truth = ddk(sr=sr, syllables="pa" * 15, rate=4, lead=2.5, snr_db=35, seed=2)
    x = x.astype(float)
    hum = vowel(sr=sr, trials=((0.8, 1.9),), total=len(x) / sr, f0=110, seed=6).astype(float)
    x = (x + 0.5 * hum[:len(x)]).astype(np.float32)
    result = segment_ddk(x, sr, DDKConfig())
    assert result["n_events"] == len(truth)


def test_slow_irregular_ddk_stays_one_sequence():
    # EPLS02-like: slow, irregular PLS syllables with split humps must not be
    # broken into several sequences of which only one is kept.
    sr = 16000
    x, truth = ddk(sr=sr, syllables="pa" * 14, rate=1.6, rate_jitter=0.25, lead=2.0, seed=5)
    x = x.astype(float)
    t = np.arange(len(x)) / sr
    for syllable in truth[::2]:
        mid = (syllable["vowel_on"] + syllable["vowel_off"]) / 2
        width = (syllable["vowel_off"] - syllable["vowel_on"]) * 0.18
        a, b = int((mid - 3 * width) * sr), int((mid + 3 * width) * sr)
        x[a:b] *= 1 - 0.45 * np.exp(-0.5 * ((t[a:b] - mid) / width) ** 2)
    result = segment_ddk(x.astype(np.float32), sr, DDKConfig())
    assert abs(result["ddk_sequence_start_sec"] - truth[0]["vowel_on"]) < .1
    assert abs(result["ddk_sequence_end_sec"] - truth[-1]["vowel_off"]) < .1


def test_quiet_lead_in_ramp_before_the_onset_is_rejected():
    sr = 16000
    rng = np.random.default_rng(3)
    x, truth = ddk(sr=sr, syllables="ka" * 15, rate=5, lead=2.0, snr_db=45, seed=7)
    x = x.astype(float)
    loud = np.max(np.abs(x))
    for k, start in enumerate(np.arange(0.9, 1.9, 0.2)):  # rising bumps 22 -> 14 dB down
        n, a = int(0.05 * sr), int(start * sr)
        x[a:a + n] += rng.standard_normal(n) * loud * 10 ** (-(22 - 2 * k) / 20) * np.hanning(n)
    result = segment_ddk(x.astype(np.float32), sr, DDKConfig())
    assert result["n_events"] == len(truth)


def test_rhythm_check_never_crashes_on_irregular_audio():
    # Real-data check: an irregular PLS file and a wrong-task file crashed the
    # rhythm estimate (no autocorrelation peak above zero).
    from vslp.acoustic.segment.ddk_v2 import rhythm_period

    t = np.arange(2000) * 0.005
    contour = np.sin(2 * np.pi * t / 1.0) + 0.3 * np.sin(2 * np.pi * t / 0.1)
    # every autocorrelation peak between 0.4 and 0.6 s is negative
    assert np.isnan(rhythm_period(contour, 0.005, min_s=0.4, max_s=0.6))
    rng = np.random.default_rng(0)
    sr = 16000
    x = np.zeros(12 * sr, dtype=np.float32)
    for start in rng.uniform(0.5, 11, 25):  # irregular bursts, no rhythm
        a = int(start * sr)
        x[a:a + int(rng.uniform(.05, .4) * sr)] += .3
    x += rng.standard_normal(len(x)).astype(np.float32) * 1e-3
    segment_ddk(x, sr, DDKConfig())


def test_implausibly_slow_ddk_is_flagged_as_possible_wrong_task():
    sr = 16000
    x, _truth = ddk(sr=sr, syllables="ta" * 12, rate=1.1, lead=1.0, seed=8)
    result = segment_ddk(x, sr, DDKConfig())
    assert result["flags"].count("irregular_rhythm_check_task") == 1


def test_segmentation_keeps_every_vowel_repetition_with_its_own_stable_window():
    """Nev, 2026-11-12: the GUI kept only the steadiest repetition."""
    from vslp.acoustic.segment.pipeline import SegmentationConfig, _method_result
    from vslp.acoustic.segment.selection import PHONATION

    sr = 16000
    x = vowel(sr=sr, trials=((1.0, 4.5), (6.0, 9.5), (11.0, 14.5)), total=15.5)
    result = _method_result(x, sr, SegmentationConfig(method=PHONATION), None)
    primary, stable = result[0], result[9]
    assert len(primary) == 3
    assert len(stable) == 3
    for interval, window in zip(primary, stable):
        assert interval.start_sec <= window.start_sec < window.end_sec <= interval.end_sec
