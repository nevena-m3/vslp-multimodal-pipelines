import pandas as pd

from vslp.analysis.kinematics.ingest import (
    build_format_summary,
    build_warning_summary,
    summarize_ingest_manifest,
)


def test_summarize_ingest_manifest_passes_clean_dataset():
    df = pd.DataFrame([
        {"source_path": "a.mp4", "status": "pass", "warning": "", "fps": 30.0, "duration_sec": 10.0, "n_frames_estimated": 300, "width": 1920, "height": 1080, "extension": ".mp4", "codec_name": "h264", "container_name": "mov,mp4"},
        {"source_path": "b.mp4", "status": "pass", "warning": "", "fps": 30.0, "duration_sec": 12.0, "n_frames_estimated": 360, "width": 1920, "height": 1080, "extension": ".mp4", "codec_name": "h264", "container_name": "mov,mp4"},
    ])
    summary = summarize_ingest_manifest(df)
    assert summary["readiness"] == "PASS"
    assert summary["n_videos"] == 2
    assert summary["readable_videos"] == 2
    assert summary["warning_videos"] == 0
    assert summary["total_estimated_frames"] == 660
    assert summary["median_fps"] == 30.0


def test_summarize_ingest_manifest_reviews_warning_dataset():
    df = pd.DataFrame([
        {"source_path": "a.mp4", "relative_path": "a.mp4", "video_id": "a", "status": "pass", "warning": "", "fps": 30.0, "duration_sec": 10.0, "n_frames_estimated": 300, "width": 1920, "height": 1080, "extension": ".mp4", "codec_name": "h264", "container_name": "mov,mp4"},
        {"source_path": "b.webm", "relative_path": "b.webm", "video_id": "b", "status": "warning", "warning": "Probe failed", "fps": None, "duration_sec": None, "n_frames_estimated": None, "width": None, "height": None, "extension": ".webm", "codec_name": None, "container_name": None},
    ])
    summary = summarize_ingest_manifest(df)
    warnings = build_warning_summary(df)
    assert summary["readiness"] == "REVIEW"
    assert summary["readable_videos"] == 1
    assert summary["warning_videos"] == 1
    assert list(warnings["video_id"]) == ["b"]


def test_summarize_ingest_manifest_fails_empty_dataset():
    summary = summarize_ingest_manifest(pd.DataFrame())
    assert summary["readiness"] == "FAIL"
    assert summary["n_videos"] == 0


def test_build_format_summary_groups_resolution_and_warnings():
    df = pd.DataFrame([
        {"source_path": "a.mp4", "status": "pass", "warning": "", "fps": 30.0, "duration_sec": 10.0, "n_frames_estimated": 300, "width": 1920, "height": 1080, "extension": ".mp4", "codec_name": "h264", "container_name": "mov,mp4"},
        {"source_path": "b.mp4", "status": "warning", "warning": "Missing FPS", "fps": None, "duration_sec": 8.0, "n_frames_estimated": None, "width": 1920, "height": 1080, "extension": ".mp4", "codec_name": "h264", "container_name": "mov,mp4"},
    ])
    grouped = build_format_summary(df)
    assert len(grouped) == 1
    assert int(grouped.loc[0, "files"]) == 2
    assert int(grouped.loc[0, "warnings"]) == 1
    assert grouped.loc[0, "resolution"] == "1920 x 1080"
