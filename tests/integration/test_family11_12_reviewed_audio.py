"""Families 11/12 consume reviewed native audio without mutating it."""

import json

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.features.family11_12_stage import run_family11_12_stage
from vslp.acoustic.features.stage import FeatureExtractionConfig, _select_registry
from vslp.core.provenance import sha256_file


def test_family11_12_stage_provenance_normalization_and_mfcc_audit(tmp_path, monkeypatch):
    sr = 16_000
    time = np.arange(sr * 2) / sr
    audio = (.25 * np.sin(2*np.pi*600*time)).astype(np.float32)
    wav = tmp_path / "canonical.wav"
    sf.write(wav, audio, sr, subtype="FLOAT")
    original_hash = sha256_file(wav)
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    decisions.write_text("recording_id,final_decision\nr1,KEEP_AUTO\n", encoding="utf-8")
    pd.DataFrame([{"recording_id":"r1", "segment_role":"speech", "start_sec":.2,
                   "end_sec":1.8}]).to_csv(intervals, index=False)
    (tmp_path / "project_manifest.json").write_text(json.dumps({
        "task_id":"wstg", "task_type":"Sentence / controlled speech",
        "task_name":"WSTG", "run_id":"run1"}), encoding="utf-8")
    preprocess = tmp_path / "acoustic" / "001_preprocess" / "tables"
    preprocess.mkdir(parents=True)
    pd.DataFrame([{"recording_id":"r1", "remove_dc_offset":True,
                   "amplitude_normalization":True}]).to_csv(
                       preprocess / "acoustic_preprocess_summary.csv", index=False)
    kept = pd.DataFrame([{
        "recording_id":"r1", "file_name":"WSTG.wav", "segmentation_method":"silero_vad",
        "analysis_wav_path":str(wav), "source_sha256":"a"*64,
        "source_file_path":str(wav), "segmentation_run_id":"seg1", "review_run_id":"rev1",
    }])
    monkeypatch.setattr("vslp.acoustic.features.family11_12_stage.load_final_segmentation",
                        lambda *_: kept)
    selected = ["absolute_energy_fs2", "wave_amp_skew", "mfcc01_mean",
                "spectral_centroid_hz", "zcr_mean_fraction", "spectral_contrast_band6_db"]
    cfg = FeatureExtractionConfig(selected_features=selected)
    result = run_family11_12_stage(decisions, tmp_path, cfg, intervals, _select_registry(cfg))
    status = pd.read_csv(result.summary_table.parent / "acoustic_feature_status_long.csv")
    assert status.loc[status.feature_id.eq("absolute_energy_fs2"), "failure_reason"].iloc[0] == "incompatible_amplitude_normalization"
    assert status.loc[status.feature_id.eq("mfcc01_mean"), "status"].iloc[0] == "computed"
    contrast_status = status.loc[status.feature_id.eq("spectral_contrast_band6_db")].iloc[0]
    assert contrast_status.status == "computed"
    assert contrast_status.parameter_set_id == "family12_spectral_48k_v1"
    audit = pd.read_csv(result.summary_table.parent / "native_measurements" /
                        "family11_12_waveform_spectral_audit.csv")
    assert audit.native_sample_rate_hz.iloc[0] == sr
    assert audit.working_sample_rate_hz.iloc[0] == 48_000
    assert bool(audit.amplitude_normalization_applied.iloc[0]) is True
    assert list((result.summary_table.parent / "native_measurements" /
                 "mfcc_matrices").glob("WSTG__*__mfcc.npz"))
    assert sha256_file(wav) == original_hash
