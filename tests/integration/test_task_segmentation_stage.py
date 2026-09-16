from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.segment.pipeline import SegmentationConfig, run_acoustic_segmentation
from vslp.acoustic.segment.selection import DDK, PHONATION, SILERO
from vslp.acoustic.segment.task_methods import PhonationConfig
from vslp.core.provenance import sha256_file


@pytest.mark.parametrize("method", [DDK, PHONATION])
def test_task_stage_schema_status_plots_and_canonical_audio(tmp_path: Path, method: str):
    sr = 22050  # Native rate must remain unchanged.
    t = np.arange(sr * 5) / sr
    if method == DDK:
        envelope = sum(((t > 0.5 + i * 0.5) & (t < 0.65 + i * 0.5)).astype(float)
                       for i in range(8))
    else:
        envelope = ((t > 0.5) & (t < 4.5)).astype(float)
    x = (0.25 * envelope * np.sin(2 * np.pi * 170 * t)).astype(np.float32)
    wav = tmp_path / "canonical.wav"
    sf.write(wav, x, sr, subtype="FLOAT")
    before = sha256_file(wav)
    summary = tmp_path / "preprocess.csv"
    pd.DataFrame([
        {"recording_id": "rec1", "file_name": "voice.wav", "status": "ok",
         "analysis_wav_path": str(wav), "source_sha256": "abc", "task_name": "pa",
         "project_name": "P", "run_id": "R"},
        {"recording_id": "rec2", "file_name": "withheld.wav", "status": "failed",
         "analysis_wav_path": "", "source_sha256": "def", "task_name": "pa",
         "project_name": "P", "run_id": "R"},
    ]).to_csv(summary, index=False)
    cfg = SegmentationConfig(method=method, phonation=PhonationConfig(onset_guard_ms=500,
        offset_guard_ms=200, stable_duration_ms=1500))
    result = run_acoustic_segmentation(summary, tmp_path, cfg)
    assert result.summary_table.is_file()
    table = pd.read_csv(result.summary_table)
    assert len(table) == 2
    assert set(table.automatic_status) <= {"ACCEPTED", "REVIEW", "EXCLUDED", "FAILED"}
    assert table.loc[1, "automatic_status"] == "EXCLUDED"
    assert table.loc[0, "task_name"] == "pa"
    assert table.loc[0, "segmentation_method"] == method
    assert Path(table.loc[0, "plot_png_path"]).is_file()
    assert Path(table.loc[1, "plot_png_path"]).is_file()
    assert Path(table.loc[0, "frame_csv_path"]).is_file()
    assert Path(table.loc[0, "segments_csv_path"]).is_file()
    assert Path(table.loc[0, "boundaries_csv_path"]).is_file()
    assert pd.read_csv(table.loc[0, "frame_csv_path"])["speech_mask_strict"].dtype == bool
    assert pd.isna(table.loc[1, "frame_csv_path"])
    assert sha256_file(wav) == before
    assert sf.info(wav).samplerate == sr
    stage = tmp_path / "acoustic" / "002_segmentation"
    assert (stage / "tables" / "segmentation_review_queue.csv").is_file()
    assert (stage / "logs" / "stage_manifest.json").is_file()
    assert not (stage / "errors").exists()
    if method == PHONATION:
        assert table.loc[0, "full_phonation_start"] < table.loc[0, "stable_region_start"]
        assert table.loc[0, "stable_region_end"] < table.loc[0, "full_phonation_end"]


def test_pinned_onnx_silero_stage_preserves_native_audio(tmp_path: Path):
    sr = 44100
    wav = tmp_path / "quiet.wav"
    sf.write(wav, np.zeros(sr * 2, dtype=np.float32), sr, subtype="FLOAT")
    before = sha256_file(wav)
    source = tmp_path / "preprocess.csv"
    pd.DataFrame([{"recording_id": "quiet", "file_name": "quiet.wav", "status": "ok",
                   "analysis_wav_path": str(wav), "source_sha256": "abc", "task_name": "Bamboo Passage"}]).to_csv(source, index=False)
    result = run_acoustic_segmentation(source, tmp_path, SegmentationConfig(method=SILERO))
    row = pd.read_csv(result.summary_table).iloc[0]
    assert row.automatic_status == "EXCLUDED"
    assert "no_speech_detected" in row["flags"]
    assert row.silero_input_sample_rate_hz == 16000
    assert row.silero_input_resampled
    assert Path(row.boundary_audit_csv_path).is_file()
    assert sha256_file(wav) == before


def test_unreadable_canonical_wav_is_reported_without_silent_drop(tmp_path: Path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav")
    source = tmp_path / "preprocess.csv"
    pd.DataFrame([{"recording_id": "bad", "file_name": "bad.wav", "status": "ok",
                   "analysis_wav_path": str(bad), "source_sha256": "abc", "task_name": "pa"}]).to_csv(source, index=False)
    result = run_acoustic_segmentation(source, tmp_path, SegmentationConfig(method=DDK))
    row = pd.read_csv(result.summary_table).iloc[0]
    assert row.automatic_status == "FAILED"
    assert row.review_required
    assert Path(row.plot_png_path).exists()
    assert result.error_table is not None and result.error_table.exists()
