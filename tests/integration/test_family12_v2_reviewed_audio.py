"""F12 runs from frozen reviewed timing and native audio without Alignment."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.features.family12_v2 import measure_trial
from vslp.acoustic.features.family12_v2_stage import _trial_regions, run_family12_stage
from vslp.acoustic.features.stage import FeatureExtractionConfig, _select_registry


def sine(freq, sr=48000, duration=1):
    t = np.arange(round(sr*duration))/sr
    return np.sin(2*np.pi*freq*t)


def test_trial_first_stage_uses_reviewed_native_audio(tmp_path, monkeypatch):
    sr = 48000
    audio = np.r_[sine(300), np.zeros(sr), sine(1200)]
    source = tmp_path/"native.wav"
    sf.write(source, audio, sr, subtype="FLOAT")
    final = tmp_path/"acoustic"/"003_segmentation_review"/"final"
    final.mkdir(parents=True)
    decisions = final/"final_segmentation_decisions.csv"
    intervals = final/"final_segmentation_intervals.csv"
    decisions.write_text("recording_id,final_decision\nr1,KEEP_MANUAL\n", encoding="utf-8")
    pd.DataFrame([{"recording_id": "r1", "segment_role": "speech", "start_sec": 0, "end_sec": 1},
                  {"recording_id": "r1", "segment_role": "speech", "start_sec": 2, "end_sec": 3}]
                 ).to_csv(intervals, index=False)
    (tmp_path/"project_manifest.json").write_text(json.dumps({
        "task_id": "wstg", "task_name": "WSTG", "participant_id": "synthetic",
        "session_id": "test"}), encoding="utf-8")
    kept = pd.DataFrame([{"recording_id": "r1", "file_name": "native.wav",
                          "source_file_path": str(source), "source_sha256": "synthetic"}])
    monkeypatch.setattr("vslp.acoustic.features.family12_v2_stage.load_final_segmentation",
                        lambda *_: kept)
    monkeypatch.setattr("vslp.acoustic.features.family12_v2_stage.load_confirmed_trials",
                        lambda *_: ([{"trial_id": "t1", "sequence_id": "q1",
                                     "start_sec": 0., "end_sec": 1.},
                                    {"trial_id": "t2", "sequence_id": "q2",
                                     "start_sec": 2., "end_sec": 3.}], ""))
    selected = ["s1_spec_m1_hz", "s1_zcr_mean_fraction"]
    config = FeatureExtractionConfig(selected_features=selected)
    result = run_family12_stage(decisions, tmp_path, config, intervals, _select_registry(config))
    trials = pd.read_csv(result.summary_table.parent/"native_measurements"/"family12_trial_values.csv")
    values = pd.read_csv(result.summary_table)
    assert trials.trial_id.tolist() == ["t1", "t2"]
    assert trials.sequence_id.tolist() == ["q1", "q2"]
    assert trials.native_sample_rate_hz.tolist() == [sr, sr]
    assert values.s1_spec_m1_hz.iloc[0] == pytest.approx(np.median(trials.s1_spec_m1_hz))




def test_vowel_uses_centered_reviewed_stable_two_second_span():
    sr = 48000
    audio = np.arange(5*sr, dtype=float)
    frame = pd.DataFrame([{"segment_role": "speech", "start_sec": .5, "end_sec": 4.5}])
    decision = SimpleNamespace(stable_region_start=1., stable_region_end=4.)
    trial = {"trial_id": "t1", "start_sec": .5, "end_sec": 4.5}
    spans, chunks = _trial_regions(audio, sr, frame, trial, "sustained_a", decision)
    assert spans == [(1.5, 3.5)]
    assert len(chunks) == 1
    assert len(chunks[0]) == 2*sr


def test_hard_exclusion_splits_stft_source_intervals():
    sr = 48000
    audio = sine(500, duration=2)
    frame = pd.DataFrame([
        {"segment_role": "speech", "start_sec": 0., "end_sec": 2.},
        {"segment_role": "manual_exclusion", "start_sec": .8, "end_sec": 1.2},
    ])
    trial = {"trial_id": "t1", "start_sec": 0., "end_sec": 2.}
    spans, chunks = _trial_regions(audio, sr, frame, trial, "bamboo_passage", None)
    assert spans == [(0., .8), (1.2, 2.)]
    _, audit = measure_trial(chunks, sr, "bamboo_passage")
    assert audit["n_intervals"] == 2
