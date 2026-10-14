"""Nonclinical F13 paths: reviewed vowel without Alignment and canonical Bamboo."""

import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.alignment.final_contract import canonical_units
from vslp.acoustic.features.family13_v1_stage import run_family13_stage
from vslp.acoustic.features.stage import FeatureExtractionConfig, _select_registry


def _project(tmp_path, task_id, audio, sr=48000):
    source = tmp_path / "native.wav"
    sf.write(source, audio, sr, subtype="FLOAT")
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    decisions.write_text("recording_id,final_decision\nr1,KEEP_MANUAL\n", encoding="utf-8")
    pd.DataFrame([{"recording_id": "r1", "segment_role": "speech",
                   "start_sec": 0, "end_sec": len(audio) / sr}]).to_csv(intervals, index=False)
    (tmp_path / "project_manifest.json").write_text(json.dumps({
        "task_id": task_id, "task_name": task_id}), encoding="utf-8")
    kept = pd.DataFrame([{"recording_id": "r1", "file_name": "native.wav",
                          "source_file_path": str(source), "source_sha256": "synthetic",
                          "stable_region_start": 0.1,
                          "stable_region_end": len(audio) / sr - 0.1}])
    return decisions, intervals, kept


def test_vowel_ppe_does_not_load_alignment(tmp_path, monkeypatch):
    sr = 48000
    t = np.arange(2 * sr) / sr
    decisions, intervals, kept = _project(tmp_path, "sustained_a",
                                           np.sin(2 * np.pi * 160 * t))
    monkeypatch.setattr("vslp.acoustic.features.family13_v1_stage.load_final_segmentation",
                        lambda *_: kept)
    monkeypatch.setattr("vslp.acoustic.features.family13_v1_stage.load_final_alignment",
                        lambda *_: (_ for _ in ()).throw(AssertionError("Alignment called")))
    config = FeatureExtractionConfig(selected_features=["vowel_ppe_bits"])
    result = run_family13_stage(decisions, tmp_path, config, intervals,
                                _select_registry(config))
    trial = pd.read_csv(result.summary_table.parent / "native_measurements" /
                        "family13_trial_values.csv")
    assert len(trial) == 1
    assert trial.native_sample_rate_hz.iloc[0] == sr
    assert trial.input_contract.iloc[0] == "FINAL_REVIEWED_SEGMENTATION"
    assert trial.trial_id.iloc[0] == "r1:reviewed_001"


def test_bamboo_uses_final_canonical_words_and_sentence_audit(tmp_path, monkeypatch):
    sr = 48000
    t = np.arange(24 * sr) / sr
    decisions, intervals, kept = _project(tmp_path, "bamboo_passage",
                                           np.sin(2 * np.pi * 200 * t))
    monkeypatch.setattr("vslp.acoustic.features.family13_v1_stage.load_final_segmentation",
                        lambda *_: kept)
    path = Path(__file__).parents[2] / "src/vslp/acoustic/alignment/task_prompts.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    prompt = next(t for t in registry["tasks"] if t["task_id"] == "bamboo_passage")["prompts"][0]
    units = canonical_units(prompt["exact_expected_text"])
    class FinalAlignment:
        manifest = {"alignment_run_id": "synthetic-v1"}
        trials = pd.DataFrame([{"recording_id": "r1", "trial_id": "t1",
                                "sequence_id": "q1", "start_sec": 0., "end_sec": 24.}])
        def words_for_trial(self, recording_id, trial_id):
            assert (recording_id, trial_id) == ("r1", "t1")
            return pd.DataFrame({"canonical_word_index": range(1, 98),
                                 "canonical_word_text": units,
                                 "start_sec": np.arange(97) * .24,
                                 "end_sec": np.arange(97) * .24 + .20})
    monkeypatch.setattr("vslp.acoustic.features.family13_v1_stage.load_final_alignment",
                        lambda *_: (FinalAlignment(), ""))
    config = FeatureExtractionConfig(selected_features=[
        "bamboo_regularity_visibility_density_10ms"])
    result = run_family13_stage(decisions, tmp_path, config, intervals,
                                _select_registry(config))
    base = result.summary_table.parent / "native_measurements"
    sentences = pd.read_csv(base / "family13_sentence_values.csv")
    trials = pd.read_csv(base / "family13_trial_values.csv")
    assert len(sentences) == 8
    assert sentences.sentence_index.tolist() == list(range(1, 9))
    assert trials.alignment_run_id.iloc[0] == "synthetic-v1"
    assert trials.input_contract.iloc[0] == "FINAL_ALIGNMENT_CONTRACT_V1"


def test_missing_alignment_is_local_explicit_failure(tmp_path, monkeypatch):
    sr = 48000
    t = np.arange(sr) / sr
    decisions, intervals, kept = _project(tmp_path, "bamboo_passage",
                                           np.sin(2 * np.pi * 200 * t))
    monkeypatch.setattr("vslp.acoustic.features.family13_v1_stage.load_final_segmentation",
                        lambda *_: kept)
    monkeypatch.setattr("vslp.acoustic.features.family13_v1_stage.load_final_alignment",
                        lambda *_: (None, "missing_alignment"))
    config = FeatureExtractionConfig(selected_features=[
        "bamboo_regularity_visibility_density_10ms"])
    result = run_family13_stage(decisions, tmp_path, config, intervals,
                                _select_registry(config))
    statuses = pd.read_csv(result.summary_table.parent /
                           "acoustic_feature_status_long.csv")
    assert statuses.failure_reason.tolist() == ["missing_final_alignment"]
    assert np.isnan(statuses.value.iloc[0])


def test_both_tsfel_task_outputs_execute_without_alignment(tmp_path, monkeypatch):
    rng = np.random.default_rng(23)
    for task, feature in (("sustained_a", "vowel_tsfel_shannon_entropy_norm"),
                          ("wstg", "s1_tsfel_shannon_entropy_norm")):
        root = tmp_path / task
        root.mkdir()
        decisions, intervals, kept = _project(root, task, rng.normal(size=96000))
        monkeypatch.setattr("vslp.acoustic.features.family13_v1_stage.load_final_segmentation",
                            lambda *_, kept=kept: kept)
        monkeypatch.setattr("vslp.acoustic.features.family13_v1_stage.load_final_alignment",
                            lambda *_: (_ for _ in ()).throw(AssertionError("Alignment called")))
        if task == "wstg":
            monkeypatch.setattr(
                "vslp.acoustic.features.family13_v1_stage.load_confirmed_trials",
                lambda *_: ([{"trial_id": "w1", "sequence_id": "w1",
                              "start_sec": 0., "end_sec": 2.}], ""))
        config = FeatureExtractionConfig(selected_features=[feature])
        result = run_family13_stage(decisions, root, config, intervals,
                                    _select_registry(config))
        status = pd.read_csv(result.summary_table.parent /
                             "acoustic_feature_status_long.csv")
        assert status.status.tolist() == ["computed"]
        assert 0 <= status.value.iloc[0] <= 1
