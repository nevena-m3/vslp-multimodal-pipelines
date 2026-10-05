"""Nonclinical F02 reviewed-domain, trial, and connected-speech integration."""

import json

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.features.family01_stage import run_family01_stage
from vslp.acoustic.features.stage import FeatureExtractionConfig, _select_registry


def test_h1h2_raw_and_corrected_use_separate_trials_and_frame_audit(tmp_path, monkeypatch):
    sr = 48000
    time = np.arange(sr * 5) / sr
    audio = (.3 * np.sin(2 * np.pi * 180 * time) +
             .12 * np.sin(2 * np.pi * 360 * time))
    audio[(time >= 2.4) & (time < 2.6)] = 0
    decisions, intervals = _project(
        tmp_path, monkeypatch, audio, sr, "sustained_a", "Sustained phonation",
        "Sustained /a/", [{"segment_role": "speech", "start_sec": 0,
                           "end_sec": 5},
                          {"segment_role": "manual_exclusion", "start_sec": 2.4,
                           "end_sec": 2.6}],
        [{"trial_id": "attempt_1", "start_sec": .1, "end_sec": 2.3},
         {"trial_id": "attempt_2", "start_sec": 2.7, "end_sec": 4.9}])

    class Formants:
        def get_value_at_time(self, index, _time):
            return 500. if index == 1 else 1500.

        def get_bandwidth_at_time(self, index, _time):
            return 92. if index == 1 else 116.

    monkeypatch.setattr("vslp.acoustic.features.family02_harmonic_difference._formants_for_span",
                        lambda *_: Formants())
    selected = ["h1_h2_db_raw", "h1star_h2star_db"]
    cfg = FeatureExtractionConfig(selected_features=selected)
    run = run_family01_stage(decisions, tmp_path, cfg, intervals, _select_registry(cfg))
    result = pd.read_csv(run.summary_table)
    trial = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                        "native_measurements" / "voice_quality_trial_summaries.csv")
    frames = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                         "native_measurements" / "h1h2_frames.csv")
    assert set(trial.trial_id) == {"attempt_1", "attempt_2"}
    assert all(np.isfinite(result[key].iloc[0]) for key in selected)
    assert all(result[key].iloc[0] == np.median(trial[key]) for key in selected)
    assert all(trial[key].iloc[0] == np.median(frames.loc[
        frames.trial_id.eq("attempt_1"), key].dropna()) for key in selected)
    assert not frames.time_sec.between(2.4, 2.6).any()
    assert frames.bandwidth_source.dropna().eq("measured").all()
    assert frames.correction_model.eq("Iseli_2007").all()


def test_h1h2_invalid_formants_do_not_substitute_raw_for_corrected(tmp_path, monkeypatch):
    sr = 48000
    time = np.arange(sr * 2.4) / sr
    audio = (.3 * np.sin(2 * np.pi * 180 * time) +
             .1 * np.sin(2 * np.pi * 360 * time))
    decisions, intervals = _project(
        tmp_path, monkeypatch, audio, sr, "sustained_a", "Sustained phonation",
        "Sustained /a/", [{"segment_role": "speech", "start_sec": 0,
                           "end_sec": 2.4}],
        [{"trial_id": "attempt_1", "start_sec": 0, "end_sec": 2.4}])

    class InvalidFormants:
        def get_value_at_time(self, _index, _time):
            return np.nan

    monkeypatch.setattr("vslp.acoustic.features.family02_harmonic_difference._formants_for_span",
                        lambda *_: InvalidFormants())
    cfg = FeatureExtractionConfig(selected_features=["h1_h2_db_raw",
                                                     "h1star_h2star_db"])
    run = run_family01_stage(decisions, tmp_path, cfg, intervals, _select_registry(cfg))
    statuses = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                           "acoustic_feature_status_long.csv")
    raw = statuses.loc[statuses.feature_id.eq("h1_h2_db_raw")].iloc[0]
    corrected = statuses.loc[statuses.feature_id.eq("h1star_h2star_db")].iloc[0]
    assert run.status == "completed_with_warnings"
    assert raw.status == "computed" and np.isfinite(raw.value)
    assert corrected.status == "unavailable" and np.isnan(corrected.value)
    assert corrected.failure_reason == "invalid_f1"
    frames = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                         "native_measurements" / "h1h2_frames.csv")
    assert frames.qc_flags.str.contains("invalid_f1").any()


def _project(tmp_path, monkeypatch, audio, sr, task_id, task_type, task_name,
             intervals_data, trials):
    wav = tmp_path / "synthetic.wav"
    sf.write(wav, audio.astype(np.float32), sr, subtype="FLOAT")
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    decisions.write_text("recording_id,final_decision\nr1,KEEP_AUTO\n", encoding="utf-8")
    pd.DataFrame([{"recording_id": "r1", **item} for item in intervals_data]).to_csv(
        intervals, index=False)
    (tmp_path / "project_manifest.json").write_text(json.dumps({
        "task_id": task_id, "task_type": task_type, "task_name": task_name,
        "run_id": "synthetic_nonclinical",
    }), encoding="utf-8")
    from vslp.acoustic.alignment.task_workflow import save_project_choices
    save_project_choices(tmp_path, {"schema_version": "1", "recordings": {},
                                    "trials": {"r1": trials}})
    kept = pd.DataFrame([{
        "recording_id": "r1", "file_name": "synthetic.wav",
        "segmentation_method": "sustained_phonation" if task_id == "sustained_a" else
        "silero_vad", "stable_region_start": 0 if task_id == "sustained_a" else np.nan,
        "stable_region_end": len(audio) / sr if task_id == "sustained_a" else np.nan,
        "analysis_wav_path": str(wav), "source_sha256": "a" * 64,
        "source_file_path": str(wav), "segmentation_run_id": "seg",
        "review_run_id": "review",
    }])
    monkeypatch.setattr("vslp.acoustic.features.family01_stage.load_final_segmentation",
                        lambda *_: kept)
    return decisions, intervals


def test_sustained_a_f02_uses_centered_windows_and_trial_median(tmp_path, monkeypatch):
    sr = 48000
    time = np.arange(sr * 5) / sr
    amplitude = np.where(time < 2.5, .1, .3)
    audio = amplitude * np.sin(2 * np.pi * 180 * time)
    trials = [{"trial_id": "attempt_1", "start_sec": .1, "end_sec": 2.3},
              {"trial_id": "attempt_2", "start_sec": 2.7, "end_sec": 4.9}]
    decisions, intervals = _project(tmp_path, monkeypatch, audio, sr,
                                    "sustained_a", "Sustained phonation", "Sustained /a/",
                                    [{"segment_role": "speech", "start_sec": 0,
                                      "end_sec": 5}], trials)

    def measured(segment, _sr, selected, minimum_usable_cycles=100):
        value = float(np.mean(np.abs(segment)) * 100)
        return ({key: (value, "") for key in selected},
                {"n_cycles": 200, "n_pulses": 201, "hnr_valid_frames": 100,
                 "hnr_frame_values": np.full(100, value),
                 "pulse_time_sec": np.array([]), "period_sec": np.array([])})

    monkeypatch.setattr("vslp.acoustic.features.family01_stage.calculate_voice_quality",
                        measured)
    cfg = FeatureExtractionConfig(selected_features=["hnr_mean_db"])
    run = run_family01_stage(decisions, tmp_path, cfg, intervals, _select_registry(cfg))
    result = pd.read_csv(run.summary_table)
    trials_out = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                             "native_measurements" / "voice_quality_trial_summaries.csv")
    assert set(trials_out.trial_id) == {"attempt_1", "attempt_2"}
    assert trials_out.analysis_duration_sec.le(2 + 1e-9).all()
    assert np.isclose(result.hnr_mean_db.iloc[0], np.median(trials_out.hnr_mean_db))
    assert trials_out.measurement_context.eq("sustained_vowel").all()
    assert trials_out.task_item.eq("a").all()


def test_s5_f02_keeps_voiced_islands_apart_from_manual_exclusion(tmp_path, monkeypatch):
    sr = 48000
    time = np.arange(sr * 3) / sr
    audio = .2 * np.sin(2 * np.pi * 180 * time)
    audio[(time >= 1.2) & (time < 1.8)] = 0
    decisions, intervals = _project(tmp_path, monkeypatch, audio, sr,
                                    "sentence_s5", "Sentence / controlled speech", "S5",
                                    [{"segment_role": "speech", "start_sec": .1,
                                      "end_sec": 1.2},
                                     {"segment_role": "manual_exclusion", "start_sec": 1.2,
                                      "end_sec": 1.8},
                                     {"segment_role": "speech", "start_sec": 1.8,
                                      "end_sec": 2.9}],
                                    [{"trial_id": "sentence_1", "start_sec": .1,
                                      "end_sec": 2.9}])
    cfg = FeatureExtractionConfig(selected_features=["jitter_local_pct", "hnr_mean_db",
                                                     "h1_mean"])
    run = run_family01_stage(decisions, tmp_path, cfg, intervals, _select_registry(cfg))
    result = pd.read_csv(run.summary_table)
    trial = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                        "native_measurements" / "voice_quality_trial_summaries.csv")
    pulses = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                         "native_measurements" / "voice_pulses.csv")
    assert result.task_item.iloc[0] == "S5"
    assert result.stimulus_text.iloc[0] == "We were away a year ago"
    assert np.isfinite(result.hnr_mean_db.iloc[0])
    assert np.isfinite(result.h1_mean.iloc[0])
    assert trial.measurement_context.iloc[0] == "sentence"
    assert len(json.loads(trial.voiced_islands.iloc[0])) >= 2
    assert not pulses.time_sec.between(1.2, 1.8).any()


def test_research_f02_and_unsmoothed_cpp_execute_through_reviewed_stage(
        tmp_path, monkeypatch):
    sr = 48000
    time = np.arange(sr * 2.4) / sr
    audio = sum(.2 / harmonic * np.sin(2 * np.pi * 180 * harmonic * time)
                for harmonic in range(1, 9))
    decisions, intervals = _project(tmp_path, monkeypatch, audio, sr,
                                    "sustained_a", "Sustained phonation", "Sustained /a/",
                                    [{"segment_role": "speech", "start_sec": 0,
                                      "end_sec": 2.4}],
                                    [{"trial_id": "attempt_1", "start_sec": 0,
                                      "end_sec": 2.4}])
    selected = ["gne_source_replication_only", "pvi_9_14hz_source", "cpp_db",
                "h1_mean", "h1_sd", "h1_relh"]
    cfg = FeatureExtractionConfig(selected_features=selected)
    run = run_family01_stage(decisions, tmp_path, cfg, intervals, _select_registry(cfg))
    result = pd.read_csv(run.summary_table)
    assert all(np.isfinite(result[key].iloc[0]) for key in selected)
    statuses = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                           "acoustic_feature_status_long.csv")
    assert statuses.loc[statuses.feature_id.eq("cpp_db"), "source_parity_status"].iloc[0] == (
        "reference_validation_pending")
    assert statuses.loc[statuses.feature_id.eq("gne_source_replication_only"),
                        "source_parity_status"].iloc[0] == "pending"
    assert statuses.sample_rate_hz.eq(sr).all()
    assert statuses.native_sample_rate_hz.eq(sr).all()
    assert statuses.loc[statuses.feature_id.eq("gne_source_replication_only"),
                        "internal_sample_rate_hz"].iloc[0] == 8000
