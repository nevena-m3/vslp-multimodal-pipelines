"""Nonclinical F06 execution from canonical Final Alignment V1 tokens."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.features.catalog import load_feature_catalog
from vslp.acoustic.features.family06_v2_stage import run_family06_stage


class _Alignment:
    manifest = {"alignment_run_id": "synthetic-final-v1"}

    def __init__(self, task: str, prompt: str, targets: list[tuple[str, str]]):
        self.trials = pd.DataFrame([
            {"recording_id": "r", "trial_id": f"trial_{i}", "sequence_id": f"seq_{i}",
             "task_id": task, "prompt_id": prompt, "start_sec": .1, "end_sec": 1.8}
            for i in (1, 2)])
        self.words = pd.DataFrame([
            {"recording_id": "r", "trial_id": f"trial_{i}",
             "word_id": f"w{i}_{j}", "canonical_word_text": word}
            for i in (1, 2) for j, (word, _) in enumerate(targets)])
        self.phones = pd.DataFrame([
            {"recording_id": "r", "trial_id": f"trial_{i}",
             "sequence_id": f"seq_{i}", "task_item": prompt,
             "phone_id": f"p{i}_{j}", "parent_word_id": f"w{i}_{j}",
             "parent_word_text": word, "phone_normalized": phone,
             "phone_index_in_word": 4 if word == "sentenced" else 1,
             "start_sec": .25 + j * .3, "end_sec": .40 + j * .3,
             "boundary_source": "REVIEWED_MANUAL_CORRECTION" if j == 0 else "MFA",
             "alignment_review_status": "HUMAN_ACCEPTED"}
            for i in (1, 2) for j, (word, phone) in enumerate(targets)])
        if any(word == "sentenced" for word, _phone in targets):
            prefixes = []
            for i in (1, 2):
                parent = self.phones.loc[self.phones.trial_id.eq(f"trial_{i}")
                                         & self.phones.parent_word_text.eq("sentenced")].iloc[0]
                for position, label in enumerate(("S", "EH", "N"), 1):
                    row = parent.to_dict()
                    row.update(phone_id=f"prefix_{i}_{position}",
                               phone_index_in_word=position,
                               phone_normalized=label, start_sec=.15, end_sec=.2)
                    prefixes.append(row)
            self.phones = pd.concat([self.phones, pd.DataFrame(prefixes)],
                                    ignore_index=True)

    def words_for_trial(self, recording_id, trial_id):
        return self.words.loc[self.words.trial_id.eq(trial_id)].copy()

    def phones_for_trial(self, recording_id, trial_id):
        return self.phones.loc[self.phones.trial_id.eq(trial_id)].copy()


@pytest.mark.parametrize(("task", "prompt", "targets", "feature", "kind", "manual"), [
    ("wstg", "wstg_we_see_three_geese", [("see", "S"), ("geese", "S"), ("three", "TH")],
     "fricative_noise_duration_s_minus_th_ms", "APERIODIC_NOISE_INTERVAL", True),
    ("wstg", "wstg_we_see_three_geese", [("see", "S"), ("geese", "S"), ("three", "TH")],
     "fricative_noise_duration_s_minus_th_ms", "APERIODIC_NOISE_INTERVAL", False),
    ("sentence_s3", "sentence_s3_blue_spot_key", [("spot", "T"), ("key", "K")],
     "spectral_m1_t_minus_k_hz", "STOP_BURST", True),
    ("sentence_s7", "sentence_s7_municipal_judge", [("judge", "JH"), ("sentenced", "T")],
     "noise_rise_time_t_minus_jh_ms", "NOISE_RISE_INTERVAL", True),
    ("sentence_s9", "sentence_s9_peter_will_keep", [("keep", "K"), ("at", "T")],
     "spectral_tilt_k_minus_t_db_per_khz", "STOP_BURST", True),
])
def test_final_alignment_v1_native_manual_landmarks_and_trials(
        tmp_path, monkeypatch, task, prompt, targets, feature, kind, manual):
    root = tmp_path / "project"
    review = root / "acoustic" / "003_segmentation_review" / "final"
    review.mkdir(parents=True)
    decisions = review / "final_segmentation_decisions.csv"
    intervals = review / "final_segmentation_intervals.csv"
    decisions.write_text("recording_id\nr\n", encoding="utf-8")
    pd.DataFrame([{"recording_id": "r", "segment_role": "speech",
                   "start_sec": .1, "end_sec": 1.8}]).to_csv(intervals, index=False)
    (root / "project_manifest.json").write_text(json.dumps({"task_id": task}), encoding="utf-8")
    rate = 48000
    time = np.arange(rate * 2) / rate
    audio = .15 * np.sin(2 * np.pi * 3000 * time)
    path = root / "native.wav"
    sf.write(path, audio, rate)
    kept = pd.DataFrame([{"recording_id": "r", "file_name": "nonclinical.wav",
                          "analysis_wav_path": str(path)}])
    monkeypatch.setattr("vslp.acoustic.features.family06_v2_stage.load_final_segmentation",
                        lambda *_: kept)
    alignment = _Alignment(task, prompt, targets)
    monkeypatch.setattr("vslp.acoustic.features.family06_v2_stage.load_final_alignment",
                        lambda *_: (alignment, ""))
    annotations = pd.DataFrame([
        {"recording_id": "r", "trial_id": f"trial_{i}", "phone_id": f"p{i}_{j}",
         "event_kind": kind, "start_sec": .28 + j * .3,
         "end_sec": .34 + .01 * (i - 1) + j * .3
         if kind != "STOP_BURST" else .28 + j * .3,
         "approved": True, "time_axis": "original_recording_seconds"}
        for i in (1, 2) for j in range(len(targets))])
    annotations_path = root / "nonclinical_subevents.csv"
    if manual:
        annotations.to_csv(annotations_path, index=False)
    spec = next(item for item in load_feature_catalog()["outputs"]
                if item["feature_id"] == feature)
    registry = pd.DataFrame([{"feature": feature, "unit": spec["unit"]}])
    config = SimpleNamespace(selected_features=[feature],
                             acoustic_subevent_annotations_csv=str(annotations_path)
                             if manual else None)
    result = run_family06_stage(decisions, root, config, intervals, registry,
                                execution_stage_dir=root / "f06_run")
    tables = result.summary_table.parent
    status = pd.read_csv(tables / "acoustic_feature_status_long.csv")
    assert status.iloc[0].status == "computed"
    tokens = pd.read_csv(tables / "native_measurements" / "family06_tokens.csv")
    trials = pd.read_csv(tables / "native_measurements" / "family06_trials.csv")
    events = pd.read_csv(tables / "native_measurements" / "family06_subevents.csv")
    assert set(tokens.trial_id) == {"trial_1", "trial_2"}
    assert set(tokens.sequence_id) == {"seq_1", "seq_2"}
    assert set(tokens.event_source) == ({"VALIDATED_MANUAL_OVERRIDE"} if manual else
                                        {"AUTOMATIC_VSLP_DETECTOR"})
    assert set(tokens.native_sample_rate_hz) == {48000}
    assert "REVIEWED_MANUAL_CORRECTION" in set(tokens.boundary_source)
    assert len(trials) == 2 and trials.failure_reason.isna().all()
    assert len(events) == 2 * len(targets)
    values = pd.read_csv(tables / "acoustic_features_per_file.csv")
    assert values[feature].iloc[0] == pytest.approx(float(np.median(trials.value)))


@pytest.mark.parametrize(("issue", "expected"), [
    ("missing_alignment", "missing_final_alignment"),
    ("alignment_source_audio_changed", "stale_final_alignment"),
])
def test_missing_or_stale_final_alignment_is_local_unavailable(
        tmp_path, monkeypatch, issue, expected):
    root = tmp_path / "project"
    review = root / "acoustic" / "003_segmentation_review" / "final"
    review.mkdir(parents=True)
    decisions = review / "final_segmentation_decisions.csv"
    intervals = review / "final_segmentation_intervals.csv"
    decisions.write_text("recording_id\nr\n", encoding="utf-8")
    pd.DataFrame([{"recording_id": "r", "segment_role": "speech",
                   "start_sec": .1, "end_sec": .8}]).to_csv(intervals, index=False)
    (root / "project_manifest.json").write_text(json.dumps({"task_id": "wstg"}),
                                                encoding="utf-8")
    path = root / "native.wav"
    sf.write(path, np.zeros(48000), 48000)
    monkeypatch.setattr("vslp.acoustic.features.family06_v2_stage.load_final_segmentation",
                        lambda *_: pd.DataFrame([{"recording_id": "r",
                                                  "file_name": "nonclinical.wav",
                                                  "analysis_wav_path": str(path)}]))
    monkeypatch.setattr("vslp.acoustic.features.family06_v2_stage.load_final_alignment",
                        lambda *_: (None, issue))
    feature = "fricative_noise_duration_th_ms"
    registry = pd.DataFrame([{"feature": feature, "unit": "ms"}])
    config = SimpleNamespace(selected_features=[feature], acoustic_subevent_annotations_csv=None)
    result = run_family06_stage(decisions, root, config, intervals, registry,
                                execution_stage_dir=root / "f06_run")
    status = pd.read_csv(result.summary_table.parent / "acoustic_feature_status_long.csv")
    assert status.iloc[0].status == "unavailable"
    assert status.iloc[0].failure_reason == expected
