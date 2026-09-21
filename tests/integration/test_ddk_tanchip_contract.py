"""End-to-end contracts for optional preprocessing and immutable DDK runs."""

from pathlib import Path
import shutil
import os

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.ingest.stage import run_acoustic_ingest
from vslp.acoustic.preprocess.stage import PreprocessConfig, run_acoustic_preprocess
from vslp.acoustic.run_setup import initialize_acoustic_run
from vslp.acoustic.segment.pipeline import SegmentationConfig, run_acoustic_segmentation, segmentation_runs
from vslp.acoustic.segment.review import select_segmentation_run, initialize_segmentation_review
from vslp.acoustic.segment.selection import DDK, PHONATION, SILERO
from vslp.core.provenance import sha256_file
from PySide6.QtWidgets import QApplication
from vslp.gui.acoustic_app.review_widget import SegmentationReviewWidget


def _ddk_source(path: Path, sr: int = 16000) -> None:
    t = np.arange(sr * 9) / sr
    x = np.zeros(len(t), dtype=np.float32)
    for i, start in enumerate(np.arange(3.0, 8.8, .23)):
        a, b = round(start * sr), round((start + .09) * sr)
        x[a:b] = (0.35 * (1 - .65 * i / 26) * np.sin(2 * np.pi * 140 * t[a:b])).astype(np.float32)
    sf.write(path, x, sr, subtype="PCM_16")


@pytest.mark.parametrize("dc,normalize", [(False, False), (True, False),
                                         (False, True), (True, True)])
def test_ddk_segment_after_each_preprocess_option(tmp_path: Path, dc: bool, normalize: bool) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    original = incoming / "clean_DDK_TA.wav"
    _ddk_source(original)
    source_hash = sha256_file(original)
    root = initialize_acoustic_run(project_name="P", task_name="TA", input_folder=incoming,
                                   output_parent=tmp_path / "out", setup_values={}).root
    ingest = run_acoustic_ingest(incoming, root)
    pre = run_acoustic_preprocess(ingest.summary_table, root,
                                  PreprocessConfig(remove_dc_offset=dc,
                                                   amplitude_normalization=normalize))
    pre_row = pd.read_csv(pre.summary_table).iloc[0]
    assert bool(pre_row.amplitude_normalization) == normalize
    segmented = run_acoustic_segmentation(pre.summary_table, root, SegmentationConfig(method=DDK))
    row = pd.read_csv(segmented.summary_table).iloc[0]
    assert row.input_source_stage == "preprocess"
    assert row.automatic_status in {"ACCEPTED", "REVIEW"}
    assert row.n_events >= 18
    assert Path(row.raw_events_csv_path).is_file()
    assert Path(row.final_events_csv_path).is_file()
    assert original.is_file() and sha256_file(original) == source_hash


def test_ingest_only_ddk_and_multiple_run_selection(tmp_path: Path) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    original = incoming / "clean_DDK_TA.wav"
    _ddk_source(original)
    source_hash = sha256_file(original)
    root = initialize_acoustic_run(project_name="P", task_name="TA", input_folder=incoming,
                                   output_parent=tmp_path / "out", setup_values={}).root
    ingest = run_acoustic_ingest(incoming, root)
    first = run_acoustic_segmentation(ingest.summary_table, root, SegmentationConfig(method=DDK))
    first_hash = sha256_file(first.summary_table)
    second = run_acoustic_segmentation(ingest.summary_table, root, SegmentationConfig(method=DDK))
    assert first.summary_table != second.summary_table
    assert sha256_file(first.summary_table) == first_hash
    assert len(segmentation_runs(root)) == 2
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    widget = SegmentationReviewWidget(lambda: root)
    widget.refresh()
    assert widget.run_combo.currentData() == ""
    assert widget.recording_list.count() == 0
    widget.run_combo.setCurrentIndex(2)
    assert widget._current_run_id == segmentation_runs(root)[1]["segmentation_run_id"]
    widget.close()
    assert app is not None
    for entry in segmentation_runs(root):
        source = select_segmentation_run(root, entry["segmentation_run_id"])
        review = initialize_segmentation_review(source, root)
        queue = pd.read_csv(review.summary_table)
        assert queue.segmentation_run_id.iloc[0] == entry["segmentation_run_id"]
    row = pd.read_csv(second.summary_table).iloc[0]
    assert row.input_source_stage == "ingest"
    assert row.automatic_status in {"ACCEPTED", "REVIEW"}
    assert sha256_file(original) == source_hash


@pytest.mark.parametrize("method,task", [(PHONATION, "sustained a"),
                                         (SILERO, "Bamboo Passage")])
def test_other_methods_accept_ingest_only_native_audio(tmp_path: Path, method: str, task: str) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    sr = 22050
    time = np.arange(sr * 4) / sr
    envelope = ((time > .5) & (time < 3.5)).astype(np.float32)
    audio = .25 * envelope * np.sin(2 * np.pi * 170 * time)
    original = incoming / "sample.wav"
    sf.write(original, audio, sr, subtype="PCM_16")
    root = initialize_acoustic_run(project_name="P", task_name=task, input_folder=incoming,
                                   output_parent=tmp_path / "out", setup_values={}).root
    ingest = run_acoustic_ingest(incoming, root)
    result = run_acoustic_segmentation(ingest.summary_table, root, SegmentationConfig(method=method))
    row = pd.read_csv(result.summary_table).iloc[0]
    assert row.input_source_stage == "ingest"
    assert row.automatic_status in {"ACCEPTED", "REVIEW", "EXCLUDED"}
    assert Path(row.analysis_wav_path).is_file()
    assert sf.info(row.analysis_wav_path).samplerate == sr
