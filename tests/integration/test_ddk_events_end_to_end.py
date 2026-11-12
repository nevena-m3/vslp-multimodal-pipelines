"""Segment -> review -> freeze writes final_ddk_events.csv, and F10 measures it."""

from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parents[1] / "unit"))
from _speech_synth import ddk  # noqa: E402

from vslp.acoustic.features.family10_v2 import load_reviewed_ddk_events, measure_trial
from vslp.acoustic.ingest.stage import run_acoustic_ingest
from vslp.acoustic.run_setup import initialize_acoustic_run
from vslp.acoustic.segment.pipeline import SegmentationConfig, run_acoustic_segmentation
from vslp.acoustic.segment.review import (finalize_segmentation_review,
                                          initialize_segmentation_review,
                                          save_segmentation_review_entry)
from vslp.acoustic.segment.selection import DDK


def test_frozen_ddk_review_feeds_ddk_features(tmp_path: Path) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    x, truth = ddk(sr=16000, syllables="pa" * 15, rate=6, rate_jitter=0, snr_db=35)
    sf.write(incoming / "S01_270_1_20230101_242_DDK_PA.wav", x, 16000, subtype="PCM_16")
    root = initialize_acoustic_run(project_name="P", task_name="DDK", input_folder=incoming,
                                   output_parent=tmp_path / "out", setup_values={}).root
    ingest = run_acoustic_ingest(incoming, root)
    segmented = run_acoustic_segmentation(ingest.summary_table, root,
                                          SegmentationConfig(method=DDK))
    assert pd.read_csv(segmented.summary_table).n_events.iloc[0] == 15
    review = initialize_segmentation_review(segmented.summary_table, root)
    for recording_id in pd.read_csv(review.summary_table).recording_id:
        save_segmentation_review_entry(root, recording_id, "KEEP_AUTO", "Reviewer")
    finalize_segmentation_review(root)
    events, issue = load_reviewed_ddk_events(
        root / "acoustic" / "003_segmentation_review" / "final" / "final_ddk_events.csv")
    assert issue == "" and len(events) == 15 and events.ddk_target.eq("pa").all()
    result, _audit, summary = measure_trial(events, "pa")
    assert summary["n_events_used"] == 10
    expected = 10 / (truth[9]["vowel_off"] - truth[0]["vowel_on"])
    assert result["rate"][0] == pytest.approx(expected, rel=.08)
