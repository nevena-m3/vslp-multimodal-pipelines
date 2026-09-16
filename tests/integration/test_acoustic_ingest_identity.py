from __future__ import annotations

import hashlib
import shutil
import wave

import pandas as pd

from vslp.acoustic.ingest.stage import run_acoustic_ingest
from vslp.acoustic.run_setup import initialize_acoustic_run


def test_ingest_uses_source_sha_for_stable_recording_id(tmp_path):
    if shutil.which("ffprobe") is None:
        return
    source = tmp_path / "input"
    source.mkdir()
    wav = source / "sample.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(bytes(32000))
    run = initialize_acoustic_run(
        project_name="Study", task_name="Passage", input_folder=source,
        output_parent=tmp_path / "output", setup_values={},
    )
    result = run_acoustic_ingest(source, run.root)
    row = pd.read_csv(result.summary_table).iloc[0]
    digest = hashlib.sha256(wav.read_bytes()).hexdigest()
    assert row["recording_id"] == row["source_sha256"] == digest
    assert row["file_name"] == "sample.wav"
    assert row["task_name"] == "Passage"
    assert row["run_id"] == run.run_id
    assert result.summary_table.parent.parent.name == "000_ingest"
    assert not (run.root / "acoustic" / "000_ingest" / "plots").exists()
