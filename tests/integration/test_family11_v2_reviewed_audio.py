"""F11 uses native source audio and reviewed timing without Alignment or F12."""

import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.features.family11_v2_stage import _trial_audio, run_family11_stage
from vslp.acoustic.features.stage import FeatureExtractionConfig, _select_registry
from vslp.core.provenance import sha256_file


def test_speech_only_omits_gaps_but_bamboo_graph_retains_internal_pause():
    sr = 48000
    audio = np.r_[np.zeros(sr), np.ones(sr) * .1, np.zeros(sr), np.ones(sr) * .2,
                  np.zeros(sr)]
    frame = pd.DataFrame([{"segment_role": "speech", "start_sec": 1., "end_sec": 2.},
                          {"segment_role": "speech", "start_sec": 3., "end_sec": 4.}])
    trial = {"start_sec": 0., "end_sec": 5.}
    speech, connected, issue, eligible, chunks = _trial_audio(
        audio, sr, frame, trial, "bamboo_passage")
    assert speech.size == 2 * sr
    assert connected.size == 3 * sr
    assert issue == ""
    assert eligible == [(1., 2.), (3., 4.)]
    assert len(chunks) == 2
    excluded = pd.concat([frame, pd.DataFrame([{"segment_role": "manual_exclusion",
                                             "start_sec": 2.1, "end_sec": 2.4}])])
    speech, connected, issue, _, _ = _trial_audio(audio, sr, excluded, trial, "bamboo_passage")
    assert speech.size == 2 * sr
    assert connected is None
    assert issue == "hard_exclusion_ambiguity"
    crossing = pd.concat([frame, pd.DataFrame([{"segment_role": "manual_exclusion",
                                             "start_sec": 1.2, "end_sec": 1.4}])])
    speech, _, issue, eligible, _ = _trial_audio(audio, sr, crossing, trial, "bamboo_passage")
    assert speech.size == round(1.8 * sr)
    assert eligible == [(1., 1.2), (1.4, 2.), (3., 4.)]
    assert issue == "hard_exclusion_ambiguity"


def test_sustained_a_uses_full_accepted_interval_not_center_window():
    sr = 48000
    audio = np.arange(5 * sr, dtype=float) / sr
    frame = pd.DataFrame([{"segment_role": "speech", "start_sec": .5, "end_sec": 4.5}])
    speech, connected, issue, eligible, chunks = _trial_audio(
        audio, sr, frame, {"start_sec": .5, "end_sec": 4.5}, "sustained_a")
    assert speech.size == 4 * sr
    assert connected is None and not issue
    assert eligible == [(.5, 4.5)]
    assert len(chunks) == 1


def test_stage_keeps_trials_separate_and_uses_native_source(tmp_path, monkeypatch):
    sr = 48000
    audio = np.r_[np.full(sr, .1), np.zeros(sr), np.full(sr, .2)]
    source = tmp_path / "native.wav"
    sf.write(source, audio, sr, subtype="FLOAT")
    source_hash = sha256_file(source)
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    decisions.write_text("recording_id,final_decision\nr1,KEEP_MANUAL\n", encoding="utf-8")
    pd.DataFrame([{"recording_id": "r1", "segment_role": "speech",
                   "start_sec": 0., "end_sec": 1.},
                  {"recording_id": "r1", "segment_role": "speech",
                   "start_sec": 2., "end_sec": 3.}]).to_csv(intervals, index=False)
    (tmp_path / "project_manifest.json").write_text(json.dumps({
        "task_id": "sentence_s5", "task_name": "S5", "run_id": "run1",
        "participant_id": "synthetic", "session_id": "visit1"}), encoding="utf-8")
    kept = pd.DataFrame([{"recording_id": "r1", "file_name": "native.wav",
                          "source_file_path": str(source), "source_sha256": source_hash}])
    monkeypatch.setattr("vslp.acoustic.features.family11_v2_stage.load_final_segmentation",
                        lambda *_: kept)
    monkeypatch.setattr("vslp.acoustic.features.family11_v2_stage._trials",
                        lambda *_: ([{"trial_id": "t1", "sequence_id": "s1",
                                     "start_sec": 0., "end_sec": 1.},
                                    {"trial_id": "t2", "sequence_id": "s2",
                                     "start_sec": 2., "end_sec": 3.}], ""))
    selected = ["s5_absolute_energy_fs2", "s5_sound_power_digital", "s5_amp_mean_fs"]
    config = FeatureExtractionConfig(selected_features=selected)
    output = run_family11_stage(decisions, tmp_path, config, intervals, _select_registry(config))
    values = pd.read_csv(output.summary_table)
    trials = pd.read_csv(output.summary_table.parent / "native_measurements" /
                         "family11_trial_values.csv")
    assert len(trials) == 2
    assert trials.trial_id.tolist() == ["t1", "t2"]
    expected = [float(trials.loc[i, "s5_absolute_energy_fs2"]) for i in (0, 1)]
    assert values.s5_absolute_energy_fs2.iloc[0] == pytest.approx(np.median(expected))
    assert trials.s5_amp_mean_fs.tolist() == pytest.approx([.1, .2])
    assert trials.native_sample_rate_hz.tolist() == [sr, sr]
    assert sha256_file(source) == source_hash
