from pathlib import Path

import numpy as np
import pandas as pd

from vslp.acoustic.features.plugins.base import FeatureContext
from vslp.acoustic.features.plugins.timing import TimingPlugin


class TimingConfig:
    minimum_pause_duration_sec = 0.30
    task_word_counts = {"bamboo": 10, "buy_bobby_a_puppy": 4}


def _write_segments(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_timing_features_use_first_to_last_speech_interval_and_percent_units(tmp_path: Path):
    segments = tmp_path / "segments.csv"
    _write_segments(
        segments,
        [
            {"segment_type": "nonspeech", "segment_role": "leading_nonspeech", "start_sec": 0.0, "end_sec": 1.0, "duration_sec": 1.0},
            {"segment_type": "speech", "segment_role": "speech", "start_sec": 1.0, "end_sec": 3.0, "duration_sec": 2.0},
            {"segment_type": "nonspeech", "segment_role": "internal_nonspeech", "start_sec": 3.0, "end_sec": 3.5, "duration_sec": 0.5},
            {"segment_type": "speech", "segment_role": "speech", "start_sec": 3.5, "end_sec": 5.0, "duration_sec": 1.5},
            {"segment_type": "nonspeech", "segment_role": "trailing_nonspeech", "start_sec": 5.0, "end_sec": 6.0, "duration_sec": 1.0},
        ],
    )
    ctx = FeatureContext(
        file_name="bamboo.wav",
        row=pd.Series({"task": "bamboo"}),
        segments_csv=segments,
        task="bamboo",
        duration_sec=6.0,
        config=TimingConfig(),
    )
    out = TimingPlugin().compute(ctx)
    assert out["total_dur"].value == 4.0  # first speech start to last speech end
    assert out["speech_dur"].value == 3.5
    assert out["total_pause_dur"].value == 0.5
    assert out["percent_pause"].value == 12.5
    assert out["num_pause"].value == 1.0
    assert out["mean_pause_dur"].value == 0.5
    assert out["mean_phrase_dur"].value == 1.75
    assert np.isclose(out["speech_rate"].value, 150.0)
    assert "percent_pause_units=percent_0_to_100" in out["percent_pause"].note


def test_timing_features_ignore_short_internal_pauses(tmp_path: Path):
    segments = tmp_path / "segments.csv"
    _write_segments(
        segments,
        [
            {"segment_type": "speech", "segment_role": "speech", "start_sec": 0.0, "end_sec": 1.0, "duration_sec": 1.0},
            {"segment_type": "nonspeech", "segment_role": "internal_nonspeech", "start_sec": 1.0, "end_sec": 1.05, "duration_sec": 0.05},
            {"segment_type": "speech", "segment_role": "speech", "start_sec": 1.05, "end_sec": 2.0, "duration_sec": 0.95},
        ],
    )
    ctx = FeatureContext(file_name="x.wav", row=pd.Series({"task": "unknown"}), segments_csv=segments, task="unknown", duration_sec=2.0, config=TimingConfig())
    out = TimingPlugin().compute(ctx)
    assert out["num_pause"].value == 0.0
    assert out["total_pause_dur"].value == 0.0
    assert np.isnan(out["mean_pause_dur"].value)
    assert np.isnan(out["speech_rate"].value)


def test_timing_features_fail_cleanly_without_speech(tmp_path: Path):
    segments = tmp_path / "segments.csv"
    _write_segments(
        segments,
        [{"segment_type": "nonspeech", "segment_role": "leading_nonspeech", "start_sec": 0.0, "end_sec": 2.0, "duration_sec": 2.0}],
    )
    ctx = FeatureContext(file_name="silent.wav", row=pd.Series({}), segments_csv=segments, duration_sec=2.0, config=TimingConfig())
    out = TimingPlugin().compute(ctx)
    assert out["speech_dur"].status == "failed"
    assert np.isnan(out["speech_dur"].value)


def test_timing_uses_sample_cv_and_excludes_sub_300ms_phrases(tmp_path: Path):
    segments = tmp_path / "segments.csv"
    _write_segments(
        segments,
        [
            {"segment_type": "speech", "start_sec": 0.0, "end_sec": 1.0, "duration_sec": 1.0},
            {"segment_type": "nonspeech", "start_sec": 1.0, "end_sec": 1.3, "duration_sec": 0.3},
            {"segment_type": "speech", "start_sec": 1.3, "end_sec": 1.5, "duration_sec": 0.2},
            {"segment_type": "nonspeech", "start_sec": 1.5, "end_sec": 2.1, "duration_sec": 0.6},
            {"segment_type": "speech", "start_sec": 2.1, "end_sec": 4.1, "duration_sec": 2.0},
        ],
    )
    ctx = FeatureContext(file_name="x.wav", row=pd.Series({}), segments_csv=segments, duration_sec=4.1, config=TimingConfig())
    out = TimingPlugin().compute(ctx)
    assert np.isclose(out["mean_phrase_dur"].value, 1.5)
    assert np.isclose(out["cv_phrase_dur"].value, np.std([1.0, 2.0], ddof=1) / 1.5)
    assert np.isclose(out["cv_pause_dur"].value, np.std([0.3, 0.6], ddof=1) / 0.45)
