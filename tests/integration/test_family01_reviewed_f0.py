"""Family 01 consumes the frozen reviewed region without altering native audio."""

import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.features.family01_stage import run_family01_stage
from vslp.acoustic.features.stage import FeatureExtractionConfig, _select_registry
from vslp.core.provenance import sha256_file


def test_reviewed_stable_region_native_rate_and_audit(tmp_path, monkeypatch):
    sr = 44100
    t = np.arange(sr * 2) / sr
    audio = (.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    wav = tmp_path / "canonical.wav"
    sf.write(wav, audio, sr, subtype="FLOAT")
    original_hash = sha256_file(wav)
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    decisions.write_text("recording_id,final_decision\nr1,KEEP_AUTO\n", encoding="utf-8")
    pd.DataFrame([{"recording_id": "r1", "segment_role": "speech", "start_sec": .2,
                   "end_sec": 1.8}]).to_csv(intervals, index=False)
    (tmp_path / "project_manifest.json").write_text(json.dumps({
        "task_id": "sustained_a", "task_type": "Sustained phonation",
        "task_name": "Sustained /a/", "run_id": "run01",
    }), encoding="utf-8")
    from vslp.acoustic.alignment.task_workflow import save_project_choices
    save_project_choices(tmp_path, {"schema_version": "1", "recordings": {},
        "trials": {"r1": [{"trial_id": "trial_001", "start_sec": .2, "end_sec": .9},
                           {"trial_id": "trial_002", "start_sec": 1.0,
                            "end_sec": 1.8}]}})
    kept = pd.DataFrame([{
        "recording_id": "r1", "file_name": "vowel.wav",
        "segmentation_method": "sustained_phonation",
        "stable_region_start": .5, "stable_region_end": 1.5,
        "analysis_wav_path": str(wav), "source_sha256": "a" * 64,
        "source_file_path": str(wav), "segmentation_run_id": "seg01",
        "review_run_id": "rev01",
    }])
    monkeypatch.setattr("vslp.acoustic.features.family01_stage.load_final_segmentation",
                        lambda *_: kept)
    cfg = FeatureExtractionConfig(selected_features=["f0_mean_hz", "pfr_maxmin_st"])
    result = run_family01_stage(decisions, tmp_path, cfg, intervals, _select_registry(cfg))
    values = pd.read_csv(result.summary_table)
    assert abs(values.f0_mean_hz.iloc[0] - 180) < 2
    assert abs(values.pfr_maxmin_st.iloc[0]) < 1
    assert values.task_type.iloc[0] == "Sustained phonation"
    tracks = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                         "native_measurements" / "f0_tracks.csv")
    assert tracks.sample_rate_hz.eq(sr).all()
    assert tracks.time_sec.min() >= .5
    assert tracks.time_sec.max() <= 1.5
    assert set(tracks.trial_id.dropna()) == {"trial_001", "trial_002"}
    assert tracks.task_family.eq("sustained_vowel").all()
    assert tracks.task_item.eq("a").all()
    assert not tracks.time_sec.between(.9, 1.0, inclusive="left").any()
    trials = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                         "native_measurements" / "f0_trial_summaries.csv")
    assert set(trials.trial_id) == {"trial_001", "trial_002"}
    assert trials.task_item.eq("a").all()
    assert trials.n_candidate_frames.gt(0).all()
    assert trials.n_valid_f0_frames.gt(0).all()
    assert trials.analysis_duration_sec.gt(0).all()
    statuses = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                           "acoustic_feature_status_long.csv")
    assert statuses.n_candidate_frames.gt(0).all()
    assert statuses.f0_qc_flag_policy_id.eq("family01_technical_f0_flags_v1").all()
    assert sha256_file(wav) == original_hash


def test_family02_reviewed_stable_region_and_pulse_audit(tmp_path, monkeypatch):
    sr = 48000
    t = np.arange(sr * 2) / sr
    wav = tmp_path / "voice.wav"
    sf.write(wav, (.2 * np.sin(2 * np.pi * 160 * t)).astype(np.float32), sr, subtype="FLOAT")
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    decisions.write_text("recording_id,final_decision\nr2,KEEP_AUTO\n", encoding="utf-8")
    pd.DataFrame([{"recording_id": "r2", "segment_role": "speech", "start_sec": .2,
                   "end_sec": 1.8}]).to_csv(intervals, index=False)
    (tmp_path / "project_manifest.json").write_text(json.dumps({
        "task_id": "sustained_a", "task_type": "Sustained phonation",
        "task_name": "Sustained /a/", "run_id": "run02",
    }), encoding="utf-8")
    kept = pd.DataFrame([{
        "recording_id": "r2", "file_name": "voice.wav",
        "segmentation_method": "sustained_phonation",
        "stable_region_start": .4, "stable_region_end": 1.6,
        "analysis_wav_path": str(wav), "source_sha256": "b" * 64,
        "source_file_path": str(wav), "segmentation_run_id": "seg02",
        "review_run_id": "rev02",
    }])
    monkeypatch.setattr("vslp.acoustic.features.family01_stage.load_final_segmentation",
                        lambda *_: kept)
    selected = ["jitter_local_pct", "shimmer_local_pct", "hnr_mean_db", "dfp_pct"]
    cfg = FeatureExtractionConfig(selected_features=selected)
    result = run_family01_stage(decisions, tmp_path, cfg, intervals, _select_registry(cfg))
    values = pd.read_csv(result.summary_table)
    assert values.jitter_local_pct.iloc[0] < .01
    assert values.shimmer_local_pct.iloc[0] < .01
    assert np.isfinite(values.hnr_mean_db.iloc[0])
    pulses = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                         "native_measurements" / "voice_pulses.csv")
    assert not pulses.empty
    assert pulses.time_sec.between(.4, 1.6).all()


@pytest.mark.parametrize("task_id,task_type,task_name,expected_family,expected_item,region", [
    ("sentence_s5", "Sentence / controlled speech", "S5", "sentence", "S5",
     "reviewed_sentence_voiced_frames"),
    ("pataka", "DDK", "pataka", "ddk", "pataka", "reviewed_ddk_voiced_frames"),
    ("bamboo_passage", "Passage / connected speech", "Bamboo Passage",
     "bamboo_passage", "bamboo", "reviewed_passage_voiced_frames"),
])
def test_task_identity_reaches_f01_without_alignment(
        tmp_path, monkeypatch, task_id, task_type, task_name,
        expected_family, expected_item, region):
    sr = 16000
    time = np.arange(sr * 2) / sr
    wav = tmp_path / "sentence.wav"
    sf.write(wav, (.2 * np.sin(2 * np.pi * 180 * time)).astype(np.float32),
             sr, subtype="FLOAT")
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    decisions.write_text("recording_id,final_decision\nr3,KEEP_AUTO\n", encoding="utf-8")
    pd.DataFrame([{"recording_id": "r3", "segment_role": "speech", "start_sec": .2,
                   "end_sec": 1.8}]).to_csv(intervals, index=False)
    (tmp_path / "project_manifest.json").write_text(json.dumps({
        "task_id": task_id, "task_type": task_type,
        "task_name": task_name, "run_id": "run03",
    }), encoding="utf-8")
    kept = pd.DataFrame([{
        "recording_id": "r3", "file_name": "sentence.wav", "segmentation_method": "silero_vad",
        "stable_region_start": np.nan, "stable_region_end": np.nan,
        "analysis_wav_path": str(wav), "source_sha256": "c" * 64,
        "source_file_path": str(wav), "segmentation_run_id": "seg03",
        "review_run_id": "rev03",
    }])
    monkeypatch.setattr("vslp.acoustic.features.family01_stage.load_final_segmentation",
                        lambda *_: kept)
    cfg = FeatureExtractionConfig(selected_features=["f0_mean_hz", "pfr_maxmin_st",
                                                    "f0_sd_st", "intonation_f0_sd_st"])
    result = run_family01_stage(decisions, tmp_path, cfg, intervals, _select_registry(cfg))
    values = pd.read_csv(result.summary_table)
    assert values.task_family.iloc[0] == expected_family
    assert values.task_item.iloc[0] == expected_item
    if expected_item == "S5":
        assert values.stimulus_text.iloc[0] == "We were away a year ago"
    if expected_family == "bamboo_passage":
        assert values.intonation_analysis_unit.iloc[0] == "passage_fallback"
        audit = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                            "native_measurements" / "f0_sentence_summaries.csv")
        assert audit.intonation_analysis_unit.eq("passage_fallback").all()
        assert audit.construct_role.eq("intonation_component").all()
    if expected_family == "sentence":
        assert values.intonation_analysis_unit.iloc[0] == "sentence"
        assert values.intonation_f0_sd_st.iloc[0] == values.f0_sd_st.iloc[0]
    assert abs(values.f0_mean_hz.iloc[0] - 180) < 2
    tracks = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                         "native_measurements" / "f0_tracks.csv")
    assert tracks.analysis_region.eq(region).all()
    assert tracks.time_sec.between(.2, 1.8).all()
    if expected_family == "bamboo_passage":
        pd.DataFrame([
            {"recording_id": "r3", "segment_role": "speech", "start_sec": .2,
             "end_sec": 1.0, "sentence_id": "one", "sentence_review_status": "APPROVED"},
            {"recording_id": "r3", "segment_role": "speech", "start_sec": 1.0,
             "end_sec": 1.8, "sentence_id": "two", "sentence_review_status": "APPROVED"},
        ]).to_csv(intervals, index=False)
        result = run_family01_stage(decisions, tmp_path, cfg, intervals,
                                    _select_registry(cfg))
        sentence_values = pd.read_csv(result.summary_table)
        sentence_audit = pd.read_csv(tmp_path / "acoustic" / "005_features" /
                                     "tables" / "native_measurements" /
                                     "f0_sentence_summaries.csv")
        assert sentence_values.intonation_analysis_unit.iloc[0] == "sentence"
        assert set(sentence_audit.sentence_id) == {"one", "two"}
        assert np.isclose(sentence_values.intonation_f0_sd_st.iloc[0],
                          np.median(sentence_audit.f0_sd_st))


def test_sustained_trial_first_median_preserves_legacy_pooled_audit(tmp_path, monkeypatch):
    sr = 16000
    time = np.arange(sr * 6) / sr
    hz = np.where(time < 2., 100., 200.)
    wav = tmp_path / "synthetic.wav"
    sf.write(wav, (.2 * np.sin(2 * np.pi * hz * time)).astype(np.float32),
             sr, subtype="FLOAT")
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    decisions.write_text("recording_id,final_decision\nr1,KEEP_AUTO\n", encoding="utf-8")
    pd.DataFrame([{"recording_id": "r1", "segment_role": "speech",
                   "start_sec": 0., "end_sec": 6.}]).to_csv(intervals, index=False)
    (tmp_path / "project_manifest.json").write_text(json.dumps({
        "task_id": "sustained_a", "task_type": "Sustained phonation",
        "task_name": "Sustained /a/", "run_id": "run01",
    }), encoding="utf-8")
    from vslp.acoustic.alignment.task_workflow import save_project_choices

    save_project_choices(tmp_path, {"schema_version": "1", "recordings": {},
        "trials": {"r1": [
            {"trial_id": "one", "start_sec": .2, "end_sec": 1.8,
             "stable_region_start": .4, "stable_region_end": 1.6},
            {"trial_id": "two", "start_sec": 2.1, "end_sec": 5.6,
             "stable_region_start": 2.4, "stable_region_end": 5.4}]}})
    kept = pd.DataFrame([{
        "recording_id": "r1", "file_name": "synthetic.wav",
        "segmentation_method": "sustained_phonation",
        "stable_region_start": .4, "stable_region_end": 1.6,
        "analysis_wav_path": str(wav), "source_sha256": "a" * 64,
        "source_file_path": str(wav), "segmentation_run_id": "seg01",
        "review_run_id": "rev01",
    }])
    monkeypatch.setattr("vslp.acoustic.features.family01_stage.load_final_segmentation",
                        lambda *_: kept)
    cfg = FeatureExtractionConfig(selected_features=["f0_mean_hz", "f0_sd_st",
                                                    "observed_f0_range_p05_p95_st"])
    result = run_family01_stage(decisions, tmp_path, cfg, intervals, _select_registry(cfg))
    values = pd.read_csv(result.summary_table).iloc[0]
    native = tmp_path / "acoustic" / "005_features" / "tables" / "native_measurements"
    trials = pd.read_csv(native / "f0_trial_summaries.csv")
    legacy = pd.read_csv(native / "f0_legacy_pooled_recording.csv").iloc[0]
    assert set(trials.trial_id) == {"one", "two"}
    assert trials.analysis_duration_sec.max() <= 2.0 + 1e-6
    for feature_id in cfg.selected_features:
        assert np.isclose(values[feature_id], np.median(trials[feature_id]))
    assert not np.isclose(values.f0_mean_hz, legacy.f0_mean_hz)
    assert values.task_item == "a"
