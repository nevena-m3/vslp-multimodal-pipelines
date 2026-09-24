"""Family 01 consumes the frozen reviewed region without altering native audio."""

import json

import numpy as np
import pandas as pd
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
