from __future__ import annotations

import shutil

import numpy as np
import soundfile as sf

from vslp.acoustic.preprocess.stage import run_acoustic_preprocess
from vslp.acoustic.run_setup import initialize_acoustic_run


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

    out = initialize_acoustic_run(project_name="Study", task_name="Passage", input_folder=input_dir, output_parent=tmp_path / "output", setup_values={}).root
    result = run_acoustic_preprocess(input_dir, out)

    assert result.status == "completed"
    assert result.summary_table is not None and result.summary_table.exists()
    assert (out / "acoustic" / "001_preprocess" / "artifacts" / "feature_wav").exists()
    assert (out / "acoustic" / "001_preprocess" / "artifacts" / "segmentation_wav").exists()
    assert result.report_path is not None and result.report_path.exists()
    import pandas as pd
    row = pd.read_csv(result.summary_table).iloc[0]
    assert row["recording_id"] == row["source_sha256"]
    assert row["task_name"] == "Passage"
