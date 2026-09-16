from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.ingest.stage import run_acoustic_ingest
from vslp.acoustic.preprocess.stage import PreprocessConfig, run_acoustic_preprocess
from vslp.acoustic.run_setup import initialize_acoustic_run


def _require_media_tools() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe are required for this integration test")


def _new_run(tmp_path: Path, input_dir: Path, task: str = "Bamboo Passage") -> Path:
    return initialize_acoustic_run(
        project_name="Study",
        task_name=task,
        input_folder=input_dir,
        output_parent=tmp_path / "output",
        setup_values={},
    ).root


def test_acoustic_preprocess_stage_writes_verified_native_float32_canonical_audio(tmp_path):
    _require_media_tools()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    sr = 44100
    t = np.arange(0, 1.0, 1 / sr)
    x = (0.15 * np.sin(2 * np.pi * 220 * t) + 0.05).astype(np.float32)
    wav = input_dir / "SUBJ001_SESSION01_ITERATION01_BAMBOO.wav"
    sf.write(wav, x, sr, subtype="PCM_16")

    out = _new_run(tmp_path, input_dir)
    ingest = run_acoustic_ingest(input_path=input_dir, output_root=out)
    assert ingest.summary_table is not None

    result = run_acoustic_preprocess(
        ingest_summary_csv=ingest.summary_table,
        output_root=out,
    )

    assert result.status == "completed"
    assert result.summary_table is not None and result.summary_table.exists()
    row = pd.read_csv(result.summary_table).iloc[0]

    assert row["status"] == "ok"
    assert row["recording_id"] == row["source_sha256"]
    assert row["task_name"] == "Bamboo Passage"
    assert int(row["source_sample_rate_hz"]) == sr
    assert int(row["analysis_sample_rate_hz"]) == sr
    assert int(row["analysis_n_channels"]) == 1
    assert row["analysis_subtype"] == "FLOAT"
    assert abs(float(row["dc_offset_after"])) < 1e-7

    analysis_files = list(
        (out / "acoustic" / "001_preprocess" / "audio").glob("*__analysis.wav")
    )
    assert len(analysis_files) == 1
    info = sf.info(analysis_files[0])
    assert info.samplerate == sr
    assert info.channels == 1
    assert info.subtype == "FLOAT"

    assert Path(str(row["audit_plot_path"])).exists()

    # Canonical preprocessing must not create downstream/model-specific WAVs.
    assert not (out / "acoustic" / "001_preprocess" / "artifacts" / "segmentation_wav").exists()
    assert not (out / "acoustic" / "001_preprocess" / "artifacts" / "feature_wav").exists()


def test_preprocess_refuses_source_changed_after_ingest(tmp_path):
    _require_media_tools()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    wav = input_dir / "recording.wav"
    sf.write(wav, (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)

    out = _new_run(tmp_path, input_dir)
    ingest = run_acoustic_ingest(input_path=input_dir, output_root=out)
    assert ingest.summary_table is not None

    # Mutate the source only after Ingest; preprocessing must detect this.
    sf.write(wav, (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), sr)

    result = run_acoustic_preprocess(ingest.summary_table, out)
    row = pd.read_csv(result.summary_table).iloc[0]

    assert result.status == "failed"
    assert row["status"] == "failed"
    assert "changed after Ingest" in str(row["warning"])
    assert not (out / "acoustic" / "001_preprocess" / "audio").exists()


def test_ambiguous_stereo_is_withheld_and_audited(tmp_path):
    _require_media_tools()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    stereo = np.column_stack(
        [
            0.2 * np.sin(2 * np.pi * 180 * t),
            0.2 * np.sin(2 * np.pi * 310 * t),
        ]
    ).astype(np.float32)
    wav = input_dir / "stereo.wav"
    sf.write(wav, stereo, sr, subtype="PCM_16")

    out = _new_run(tmp_path, input_dir)
    ingest = run_acoustic_ingest(input_path=input_dir, output_root=out)
    result = run_acoustic_preprocess(ingest.summary_table, out)
    row = pd.read_csv(result.summary_table).iloc[0]

    assert result.status == "failed"  # no safe canonical waveform was produced
    assert row["status"] == "needs_channel_review"
    assert pd.isna(row["analysis_wav_path"])
    assert Path(str(row["audit_plot_path"])).exists()


def test_preprocess_can_preserve_dc_when_option_is_disabled(tmp_path):
    _require_media_tools()
    input_dir = tmp_path / "input_dc_off"
    input_dir.mkdir()
    sr = 48000
    t = np.arange(sr, dtype=np.float64) / sr
    wav = input_dir / "dc_preserved.wav"
    x = (0.12 * np.sin(2 * np.pi * 200.0 * t) + 0.06).astype(np.float32)
    sf.write(wav, x, sr, subtype="PCM_16")

    out = _new_run(tmp_path, input_dir)
    ingest = run_acoustic_ingest(input_path=input_dir, output_root=out)
    result = run_acoustic_preprocess(
        ingest.summary_table,
        out,
        config=PreprocessConfig(remove_dc_offset=False),
    )
    row = pd.read_csv(result.summary_table).iloc[0]

    assert result.status == "completed"
    assert bool(row["remove_dc_offset"]) is False
    assert abs(float(row["dc_offset_after"]) - float(row["dc_offset_before"])) < 1e-9
    assert abs(float(row["dc_offset_removed"])) < 1e-9
    assert int(row["analysis_sample_rate_hz"]) == sr
    assert row["analysis_subtype"] == "FLOAT"

    y, y_sr = sf.read(Path(str(row["analysis_wav_path"])), dtype="float32")
    assert y_sr == sr
    assert abs(float(np.mean(y, dtype=np.float64)) - float(row["dc_offset_before"])) < 1e-7
