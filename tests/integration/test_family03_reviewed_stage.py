"""Nonclinical final-review and mixed-family integration for open F03."""

import json

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.features.family03_stage import run_family03_stage
from vslp.acoustic.features.mixed_dispatch import run_mixed_feature_stage
from vslp.acoustic.features.stage import FeatureExtractionConfig, _select_registry


def _project(tmp_path, monkeypatch):
    sr = 48000
    time = np.arange(sr * 5) / sr
    audio = (.2 * np.sin(2 * np.pi * 180 * time) +
             .08 * np.sin(2 * np.pi * 360 * time))
    audio[(time >= 2.35) & (time < 2.65)] = 0
    wav = tmp_path / "synthetic_nonclinical.wav"
    sf.write(wav, audio.astype(np.float32), sr, subtype="FLOAT")
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    decisions.write_text("recording_id,final_decision\nr1,KEEP_AUTO\n", encoding="utf-8")
    pd.DataFrame([
        {"recording_id": "r1", "segment_role": "speech", "start_sec": 0,
         "end_sec": 2.35},
        {"recording_id": "r1", "segment_role": "manual_exclusion",
         "start_sec": 2.35, "end_sec": 2.65},
        {"recording_id": "r1", "segment_role": "speech", "start_sec": 2.65,
         "end_sec": 5},
    ]).to_csv(intervals, index=False)
    (tmp_path / "project_manifest.json").write_text(json.dumps({
        "task_id": "sustained_a", "task_type": "Sustained phonation",
        "task_name": "Sustained /a/", "run_id": "synthetic_nonclinical",
    }), encoding="utf-8")
    from vslp.acoustic.alignment.task_workflow import save_project_choices
    save_project_choices(tmp_path, {"schema_version": "1", "recordings": {},
                                    "trials": {"r1": [
                                        {"trial_id": "attempt_1", "start_sec": .05,
                                         "end_sec": 2.3},
                                        {"trial_id": "attempt_2", "start_sec": 2.7,
                                         "end_sec": 4.95}]}})
    kept = pd.DataFrame([{
        "recording_id": "r1", "file_name": "synthetic_nonclinical.wav",
        "segmentation_method": "sustained_phonation",
        "stable_region_start": 0., "stable_region_end": 5.,
        "analysis_wav_path": str(wav), "source_sha256": "a" * 64,
        "source_file_path": str(wav), "segmentation_run_id": "seg",
        "review_run_id": "review",
    }])
    monkeypatch.setattr("vslp.acoustic.features.family03_stage.load_final_segmentation",
                        lambda *_: kept)
    monkeypatch.setattr("vslp.acoustic.features.family01_stage.load_final_segmentation",
                        lambda *_: kept)
    monkeypatch.setattr("vslp.acoustic.features.mixed_dispatch.load_final_segmentation",
                        lambda *_: kept)
    return decisions, intervals


def test_f03_trial_first_with_hard_exclusion_and_short_tremor(tmp_path, monkeypatch):
    decisions, intervals = _project(tmp_path, monkeypatch)
    selected = ["glottal_mfdr", "glottal_naq",
                "amplitude_tremor_depth_short_v1_pct"]
    config = FeatureExtractionConfig(selected_features=selected)
    result = run_family03_stage(decisions, tmp_path, config, intervals,
                                _select_registry(config))
    table = pd.read_csv(result.summary_table)
    trials = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                         "native_measurements" / "family03_trial_summaries.csv")
    cycles = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                         "native_measurements" / "family03_cycle_measurements.csv")
    assert set(trials.trial_id) == {"attempt_1", "attempt_2"}
    assert trials.analysis_duration_sec.le(2 + 1e-9).all()
    assert all(np.isfinite(table[feature].iloc[0]) for feature in selected)
    assert all(np.isclose(table[feature].iloc[0], np.median(trials[feature]))
               for feature in selected)
    assert not cycles.start_sec.between(2.35, 2.65).any()
    assert not cycles.end_sec.between(2.35, 2.65).any()
    assert cycles.source_equivalent.eq(False).all()


def test_f03_and_f01_selected_only_in_same_mixed_run(tmp_path, monkeypatch):
    decisions, intervals = _project(tmp_path, monkeypatch)
    config = FeatureExtractionConfig(selected_features=[
        "f0_mean_hz", "amplitude_tremor_depth_short_v1_pct"])
    result = run_mixed_feature_stage(decisions, tmp_path, config, intervals,
                                     _select_registry(config))
    table = pd.read_csv(result.summary_table)
    assert {"f0_mean_hz", "amplitude_tremor_depth_short_v1_pct"} <= set(table)
    assert np.isfinite(table.f0_mean_hz.iloc[0])
    assert np.isfinite(table.amplitude_tremor_depth_short_v1_pct.iloc[0])
    assert "glottal_mfdr" not in table
