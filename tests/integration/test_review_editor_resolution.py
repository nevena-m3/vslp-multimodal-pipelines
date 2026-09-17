import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.features.stage import FeatureExtractionConfig, run_acoustic_feature_extraction
from vslp.acoustic.quality.stage import QualityControlConfig, run_acoustic_quality_control
from vslp.acoustic.segment.review import (finalize_segmentation_review, initialize_segmentation_review,
    load_review_exclusions, load_review_state, resolve_reviewed_intervals,
    save_segmentation_review_entry, validate_analysis_window, validate_review_exclusions)
from vslp.acoustic.segment.silero_reference import Interval
from vslp.core.provenance import sha256_file


def _run(tmp_path: Path):
    sr = 16000
    t = np.arange(4 * sr) / sr
    x = (0.1 * np.sin(2 * np.pi * 145 * t)).astype("float32")
    wav = tmp_path / "canonical.wav"
    sf.write(wav, x, sr, subtype="FLOAT")
    segments = tmp_path / "automatic_segments.csv"
    pd.DataFrame([
        {"segment_type": "speech", "segment_role": "speech", "start_sec": 0.2, "end_sec": 1.2},
        {"segment_type": "nonspeech", "segment_role": "internal_nonspeech", "start_sec": 1.2, "end_sec": 1.4},
        {"segment_type": "speech", "segment_role": "speech", "start_sec": 1.4, "end_sec": 3.5},
    ]).to_csv(segments, index=False)
    summary = tmp_path / "acoustic" / "002_segmentation" / "tables" / "acoustic_segmentation_summary.csv"
    summary.parent.mkdir(parents=True)
    pd.DataFrame([{"recording_id": "r", "file_name": "r.wav", "task_name": "Bamboo Passage",
        "segmentation_method": "silero_vad", "automatic_status": "EXCLUDED", "review_required": True,
        "duration_sec": 4, "analysis_wav_path": str(wav), "segments_csv_path": str(segments),
        "source_sha256": "a" * 64, "project_name": "P", "run_id": "R"}]).to_csv(summary, index=False)
    (tmp_path / "project_manifest.json").write_text(json.dumps({
        "project_name": "P", "task_name": "Bamboo Passage", "run_id": "R",
        "created_at_local": "2026-01-01T00:00:00", "created_at_utc": "2026-01-01T05:00:00Z",
    }), encoding="utf-8")
    initialize_segmentation_review(summary, tmp_path)
    return wav, segments, summary


@pytest.mark.parametrize("start,end", [(-1, 2), (1, 1), (3, 2), (0, 5), (float("nan"), 2)])
def test_invalid_analysis_window(start, end):
    with pytest.raises(ValueError):
        validate_analysis_window(start, end, 4)


def test_role_reconstruction_excludes_trim_and_contamination():
    speech, timeline = resolve_reviewed_intervals([Interval(.2, 1.2), Interval(1.4, 3.5)],
        duration_sec=4, analysis_start_sec=.7, analysis_end_sec=3.7,
        exclusions=[{"start_sec": 2, "end_sec": 2.3, "exclusion_reason": "Cough / throat clear"}],
        boundary_source="AUTO_MODIFIED")
    assert [(r["start_sec"], r["end_sec"]) for r in speech] == [(.7, 1.2), (1.4, 2), (2.3, 3.5)]
    assert timeline.iloc[0].segment_role == "outside_analysis_window"
    assert timeline.iloc[0].end_sec == .7
    assert timeline.iloc[-1].segment_role == "outside_analysis_window"
    assert set(timeline.segment_role) >= {"speech", "internal_nonspeech", "trailing_nonspeech", "manual_exclusion"}
    assert timeline.loc[timeline.segment_role.eq("manual_exclusion"), "manual_exclusion_reason"].tolist() == ["Cough / throat clear"]
    assert not timeline.loc[timeline.segment_role.eq("leading_nonspeech"), "start_sec"].lt(.7).any()


def test_other_speaker_exclusion_and_leading_role():
    speech, timeline = resolve_reviewed_intervals([Interval(.2, 1.0), Interval(1.5, 2.5)],
        duration_sec=3, analysis_start_sec=1, analysis_end_sec=2.7,
        exclusions=[{"start_sec": 1.8, "end_sec": 2.0, "exclusion_reason": "Other speaker"}])
    assert speech[0]["start_sec"] == 1.5
    leading = timeline.loc[timeline.segment_role.eq("leading_nonspeech")].iloc[0]
    assert (leading.start_sec, leading.end_sec) == (1.0, 1.5)
    assert timeline.loc[timeline.segment_role.eq("manual_exclusion"), "manual_exclusion_reason"].iloc[0] == "Other speaker"


def test_exclusion_validation():
    with pytest.raises(ValueError):
        validate_review_exclusions([{"start_sec": 1, "end_sec": 2, "exclusion_reason": "guess"}], duration_sec=4)
    with pytest.raises(ValueError):
        validate_review_exclusions([{"start_sec": 1, "end_sec": 2, "exclusion_reason": "Other speaker"},
            {"start_sec": 1.5, "end_sec": 2.2, "exclusion_reason": "Cough / throat clear"}], duration_sec=4)


def test_exclusion_must_be_inside_reviewed_window(tmp_path: Path):
    _run(tmp_path)
    with pytest.raises(ValueError, match="within the analysis window"):
        save_segmentation_review_entry(tmp_path, "r", "KEEP_AUTO", "Reviewer", "Trimmed setup",
            analysis_start_sec=1.5, exclusion_intervals=[{
                "start_sec": 1, "end_sec": 1.2, "exclusion_reason": "Other speaker"}])


def test_review_corrections_persist_and_rescued_recording_reaches_downstream(tmp_path: Path):
    wav, automatic_segments, summary = _run(tmp_path)
    hashes = sha256_file(wav), sha256_file(automatic_segments), sha256_file(summary)
    save_segmentation_review_entry(tmp_path, "r", "KEEP_MANUAL", "Reviewer", "Other speaker and cough excluded",
        "0.8,1.2\n1.4,3.5", analysis_start_sec=.7, analysis_end_sec=3.7,
        exclusion_intervals=[{"start_sec": 2, "end_sec": 2.3,
            "exclusion_reason": "Cough / throat clear", "notes": "Cough"}])
    initialize_segmentation_review(summary, tmp_path)
    decisions, overrides = load_review_state(tmp_path)
    exclusions = load_review_exclusions(tmp_path)
    assert decisions.iloc[0].analysis_start_sec == "0.7"
    assert len(overrides) == 2 and len(exclusions) == 1
    assert exclusions.iloc[0].exclusion_reason == "Cough / throat clear"
    finalize_segmentation_review(tmp_path)
    tables = tmp_path / "acoustic" / "003_segmentation_review" / "tables"
    final = pd.read_csv(tables / "final_segmentation_intervals.csv", keep_default_na=False)
    primary = final.loc[final.view.eq("primary_speech")]
    assert primary.boundary_source.eq("MANUAL").all()
    assert primary.start_sec.astype(float).tolist() == [0.8, 1.4, 2.3]
    timeline = final.loc[final.view.eq("timeline")]
    assert timeline.segment_role.eq("manual_exclusion").sum() == 1
    assert timeline.segment_role.eq("outside_analysis_window").sum() == 2
    qc = run_acoustic_quality_control(tables / "final_segmentation_decisions.csv", tmp_path,
        config=QualityControlConfig(selected_families=["gain_dynamics"]),
        final_segmentation_intervals_csv=tables / "final_segmentation_intervals.csv")
    features = run_acoustic_feature_extraction(tables / "final_segmentation_decisions.csv", tmp_path,
        config=FeatureExtractionConfig(selected_features=["f0_mean"]),
        final_segmentation_intervals_csv=tables / "final_segmentation_intervals.csv")
    assert pd.read_csv(qc.summary_table).recording_id.tolist() == ["r"]
    assert pd.read_csv(features.summary_table).file_name.tolist() == ["r.wav"]
    assert hashes == (sha256_file(wav), sha256_file(automatic_segments), sha256_file(summary))


def test_auto_modified_provenance_and_trim_without_leading_pause(tmp_path: Path):
    wav, automatic_segments, _ = _run(tmp_path)
    original = sha256_file(automatic_segments)
    save_segmentation_review_entry(tmp_path, "r", "KEEP_AUTO", "Reviewer", "Removed setup speech",
        analysis_start_sec=1.3, analysis_end_sec=3.7,
        exclusion_intervals=[{"start_sec": 2, "end_sec": 2.2,
            "exclusion_reason": "Other speaker", "notes": "Background voice"}])
    finalize_segmentation_review(tmp_path)
    tables = tmp_path / "acoustic" / "003_segmentation_review" / "tables"
    decisions = pd.read_csv(tables / "final_segmentation_decisions.csv", keep_default_na=False)
    timeline = pd.read_csv(tables / "final_segmentation_intervals.csv", keep_default_na=False)
    assert decisions.iloc[0].final_decision == "KEEP_AUTO"
    assert decisions.iloc[0].boundary_source == "AUTO_MODIFIED"
    assert timeline.loc[timeline.view.eq("primary_speech"), "boundary_source"].eq("AUTO_MODIFIED").all()
    leading = timeline.loc[timeline.view.eq("timeline") & timeline.segment_role.eq("leading_nonspeech")].iloc[0]
    assert (float(leading.start_sec), float(leading.end_sec)) == (1.3, 1.4)
    outside = timeline.loc[timeline.view.eq("timeline") & timeline.segment_role.eq("outside_analysis_window")].iloc[0]
    assert (float(outside.start_sec), float(outside.end_sec)) == (0, 1.3)
    assert sha256_file(automatic_segments) == original and wav.is_file()


def test_atomic_review_entry_recovers_stale_csv_views(tmp_path: Path):
    _run(tmp_path)
    save_segmentation_review_entry(tmp_path, "r", "KEEP_MANUAL", "Reviewer", "Edited event",
        "1.4,3.4", analysis_start_sec=1.2)
    tables = tmp_path / "acoustic" / "003_segmentation_review" / "tables"
    stale = pd.read_csv(tables / "segmentation_review_decisions.csv", keep_default_na=False)
    stale.loc[0, "final_decision"] = ""
    stale.to_csv(tables / "segmentation_review_decisions.csv", index=False)
    pd.DataFrame(columns=["recording_id", "file_name", "segment_index", "start_sec", "end_sec",
                          "reviewer", "review_date", "review_notes"]).to_csv(
        tables / "manual_segmentation_overrides.csv", index=False)
    decisions, overrides = load_review_state(tmp_path)
    assert decisions.iloc[0].final_decision == "KEEP_MANUAL"
    assert decisions.iloc[0].analysis_start_sec == "1.2"
    assert len(overrides) == 1
