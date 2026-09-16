from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.quality.stage import QualityControlConfig, run_acoustic_quality_control


def test_quality_control_stage_writes_family_tables(tmp_path: Path):
    out = tmp_path / "project"
    wav_dir = out / "acoustic" / "001_preprocess" / "artifacts" / "segmentation_wav"
    seg_tables = out / "acoustic" / "002_segmentation" / "tables"
    wav_dir.mkdir(parents=True)
    (seg_tables / "frames").mkdir(parents=True)
    (seg_tables / "segments").mkdir(parents=True)
    (seg_tables / "boundaries").mkdir(parents=True)

    sr = 16000
    t = np.arange(sr * 2) / sr
    x = (0.05 * np.random.default_rng(0).normal(size=t.size) + 0.2 * np.sin(2 * np.pi * 180 * t)).astype("float32")
    wav = wav_dir / "file1__seg16k.wav"
    sf.write(wav, x, sr)

    frames = pd.DataFrame({
        "frame_idx": np.arange(60),
        "mid_sec": np.arange(60) * 0.03 + 0.015,
        "rms_db": np.r_[np.linspace(-25, -20, 25), np.linspace(-55, -50, 10), np.linspace(-24, -21, 25)],
        "speech_vad_raw": [True]*25 + [False]*10 + [True]*25,
        "speech_vad_smooth": [True]*25 + [False]*10 + [True]*25,
        "speech_mask_strict": [True]*25 + [False]*10 + [True]*25,
        "nonspeech_mask_strict": [False]*25 + [True]*10 + [False]*25,
        "threshold": 0.5,
        "frame_ms": 30,
    })
    frame_csv = seg_tables / "frames" / "file1__frames.csv"
    frames.to_csv(frame_csv, index=False)

    segments = pd.DataFrame({
        "segment_type": ["speech", "nonspeech", "speech"],
        "segment_role": ["speech", "internal_nonspeech", "speech"],
        "start_sec": [0.0, 0.75, 1.05],
        "end_sec": [0.75, 1.05, 1.8],
        "duration_sec": [0.75, 0.30, 0.75],
    })
    segments_csv = seg_tables / "segments" / "file1__segments.csv"
    segments.to_csv(segments_csv, index=False)

    seg_summary = seg_tables / "acoustic_segmentation_summary.csv"
    pd.DataFrame([{
        "file_name": "file1.wav",
        "segmentation_wav_path": str(wav),
        "frame_csv_path": str(frame_csv),
        "segments_csv_path": str(segments_csv),
        "speech_fraction": 0.83,
        "n_speech_segments": 2,
    }]).to_csv(seg_summary, index=False)

    result = run_acoustic_quality_control(seg_summary, out, QualityControlConfig())
    assert result.summary_table.exists()
    df = pd.read_csv(result.summary_table)
    assert "qadd_pause_rms_db_median" in df.columns
    assert "qgain_speech_rms_db_std" in df.columns
    assert "qtemp_waveform_continuity_break_score" in df.columns
    assert (out / "acoustic" / "003_quality_control" / "tables" / "acoustic_quality_family_status.csv").exists()
    assert result.report_path.exists()
