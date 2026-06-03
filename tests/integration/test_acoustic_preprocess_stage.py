from __future__ import annotations

import shutil

import numpy as np
import soundfile as sf

from vslp.acoustic.preprocess.stage import run_acoustic_preprocess


def test_acoustic_preprocess_stage_writes_expected_outputs(tmp_path):
    if shutil.which("ffmpeg") is None:
        return
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    sr = 16000
    t = np.arange(0, 1.0, 1 / sr)
    x = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    wav = input_dir / "SUBJ001_SESSION01_ITERATION01_BAMBOO.wav"
    sf.write(wav, x, sr, subtype="PCM_16")

    out = tmp_path / "project"
    result = run_acoustic_preprocess(input_dir, out)

    assert result.status == "completed"
    assert result.summary_table is not None and result.summary_table.exists()
    assert (out / "acoustic" / "002_preprocess" / "artifacts" / "feature_wav").exists()
    assert (out / "acoustic" / "002_preprocess" / "artifacts" / "segmentation_wav").exists()
    assert result.report_path is not None and result.report_path.exists()
