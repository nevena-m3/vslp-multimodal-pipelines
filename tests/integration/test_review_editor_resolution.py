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


def _run(tmp_path: Path, *, task_name="Bamboo Passage", method="silero_vad"):
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
    pd.DataFrame([{"recording_id": "r", "file_name": "r.wav", "task_name": task_name,
        "segmentation_run_id": "seg_test",
        "segmentation_method": method, "automatic_status": "EXCLUDED", "review_required": True,
        "duration_sec": 4, "analysis_wav_path": str(wav), "segments_csv_path": str(segments),
        "source_sha256": "a" * 64, "project_name": "P", "run_id": "R"}]).to_csv(summary, index=False)
    (tmp_path / "project_manifest.json").write_text(json.dumps({
        "project_name": "P", "task_name": task_name, "run_id": "R",
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
    tables = tmp_path / "acoustic" / "003_segmentation_review" / "runs" / "review_default" / "tables"
    final = pd.read_csv(tmp_path / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_intervals.csv", keep_default_na=False)
    primary = final.loc[final.segment_role.eq("speech")]
    assert primary.boundary_source.eq("MANUAL").all()
    assert primary.start_sec.astype(float).tolist() == [0.8, 1.4, 2.3]
    timeline = final.loc[final.view.eq("authoritative")]
    assert timeline.segment_role.eq("manual_exclusion").sum() == 1
    assert timeline.segment_role.eq("outside_analysis_window").sum() == 2
    qc = run_acoustic_quality_control(tmp_path / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_decisions.csv", tmp_path,
        config=QualityControlConfig(selected_families=["gain_dynamics"]),
        final_segmentation_intervals_csv=tmp_path / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_intervals.csv")
    with pytest.raises(ValueError, match="legacy feature IDs cannot execute"):
        run_acoustic_feature_extraction(tmp_path / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_decisions.csv", tmp_path,
            config=FeatureExtractionConfig(selected_features=["f0_mean"]),
            final_segmentation_intervals_csv=tmp_path / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_intervals.csv")
    assert pd.read_csv(qc.summary_table).recording_id.tolist() == ["r"]
    assert not (tmp_path / "acoustic" / "005_features" / "tables").exists()
    assert hashes == (sha256_file(wav), sha256_file(automatic_segments), sha256_file(summary))


def test_family08_uses_frozen_reviewed_intervals_and_exact_ids(tmp_path: Path):
    wav, automatic_segments, summary = _run(tmp_path)
    hashes = sha256_file(wav), sha256_file(automatic_segments)
    save_segmentation_review_entry(
        tmp_path, "r", "KEEP_MANUAL", "Reviewer", "Cough excluded",
        "0.8,1.2\n1.4,3.5", analysis_start_sec=.7, analysis_end_sec=3.7,
        exclusion_intervals=[{"start_sec": 2, "end_sec": 2.3,
                              "exclusion_reason": "Cough / throat clear"}])
    finalize_segmentation_review(tmp_path)
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    prompt = tmp_path / "prompt_counts.json"
    prompt.write_text(json.dumps({
        "schema_version": "1", "task_id": "bamboo_passage",
        "prompt_version": "Bamboo-v1", "count_source": "frozen_prompt_count",
        "syllable_count": 12, "word_count": 6,
        "applies_to_all_recordings": True,
    }), encoding="utf-8")
    progress = []
    result = run_acoustic_feature_extraction(
        final / "final_segmentation_decisions.csv", tmp_path,
        config=FeatureExtractionConfig(
            selected_features=["speaking_rate_syll_s", "speaking_rate_words_min",
                               "articulation_rate_syll_s"],
            prompt_manifest_path=str(prompt)),
        final_segmentation_intervals_csv=final / "final_segmentation_intervals.csv",
        progress_callback=lambda done, total, message: progress.append((done, total, message)))
    values = pd.read_csv(result.summary_table)
    assert values.file_name.tolist() == ["r.wav"]
    assert np.isclose(values.speaking_rate_syll_s.iloc[0], 12 / 2.4)
    assert np.isclose(values.speaking_rate_words_min.iloc[0], 60 * 6 / 2.4)
    assert np.isclose(values.articulation_rate_syll_s.iloc[0], 12 / 2.4)
    assert "speech_rate" not in values
    status = pd.read_csv(result.summary_table.parent / "acoustic_feature_status_long.csv")
    assert set(status.feature) == {"speaking_rate_syll_s", "speaking_rate_words_min",
                                   "articulation_rate_syll_s"}
    assert status.status.eq("computed").all()
    assert status.algorithm_version.eq("family08-rate-1.0.0").all()
    assert status.parameter_set_id.eq("family08_bamboo_reviewed_v1").all()
    assert status.unit.notna().all()
    assert sha256_file(tmp_path / "acoustic" / "005_features" / "family_runs" / "timing" / "configs" /
                       "prompt_count_manifest.json") == sha256_file(prompt)
    assert progress[0][:2] == (0, 1) and progress[-1][:2] == (1, 1)
    assert hashes == (sha256_file(wav), sha256_file(automatic_segments))


def test_family08_missing_prompt_count_is_nan_with_status_reason(tmp_path: Path):
    _wav, _automatic_segments, _summary = _run(tmp_path)
    save_segmentation_review_entry(
        tmp_path, "r", "KEEP_MANUAL", "Reviewer", "Recovered segmentation",
        "0.8,1.2\n1.4,3.5")
    finalize_segmentation_review(tmp_path)
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    result = run_acoustic_feature_extraction(
        final / "final_segmentation_decisions.csv", tmp_path,
        config=FeatureExtractionConfig(selected_features=["speaking_rate_syll_s"]),
        final_segmentation_intervals_csv=final / "final_segmentation_intervals.csv")
    values = pd.read_csv(result.summary_table)
    status = pd.read_csv(result.summary_table.parent / "acoustic_feature_status_long.csv")
    assert np.isnan(values.speaking_rate_syll_s.iloc[0])
    assert status.status.tolist() == ["unavailable"]
    assert status.reason.tolist() == ["prompt_manifest_missing"]
    assert "speech_rate" not in values


def test_family09_uses_only_frozen_review_and_writes_shared_events(tmp_path: Path):
    wav, automatic_segments, _summary = _run(tmp_path)
    original_hashes = sha256_file(wav), sha256_file(automatic_segments)
    save_segmentation_review_entry(
        tmp_path, "r", "KEEP_MANUAL", "Reviewer", "Cough excluded",
        "0.8,1.2\n1.6,2.0\n2.3,3.5", analysis_start_sec=.7,
        analysis_end_sec=3.7,
        exclusion_intervals=[{"start_sec": 2.0, "end_sec": 2.3,
                              "exclusion_reason": "Cough / throat clear"}])
    finalize_segmentation_review(tmp_path)
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    result = run_acoustic_feature_extraction(
        final / "final_segmentation_decisions.csv", tmp_path,
        config=FeatureExtractionConfig(selected_features=[
            "total_pause_duration_s", "pause_count", "percent_pause_ge300ms",
            "mean_pause_duration_s", "cv_pause_duration", "mean_phrase_duration_s"]),
        final_segmentation_intervals_csv=final / "final_segmentation_intervals.csv")
    values = pd.read_csv(result.summary_table)
    status = pd.read_csv(result.summary_table.parent / "acoustic_feature_status_long.csv")
    events_path = result.summary_table.parent.parent / "family_runs" / "timing" / "tables" / "native_measurements" / "pause_phrase_events.csv"
    events = pd.read_csv(events_path)
    assert events_path.is_file()
    assert np.isclose(values.total_pause_duration_s.iloc[0], .4)
    assert values.pause_count.iloc[0] == 1
    assert np.isclose(values.percent_pause_ge300ms.iloc[0], 100 * .4 / 2.4)
    assert np.isclose(values.mean_pause_duration_s.iloc[0], .4)
    assert np.isnan(values.cv_pause_duration.iloc[0])
    assert status.loc[status.feature.eq("cv_pause_duration"), "reason"].iloc[0] == "insufficient_pause_events"
    assert status.segmentation_run_id.notna().all()
    assert status.review_run_id.notna().all()
    assert status.parameter_set_id.eq("family09_bamboo_reviewed_v1").all()
    assert events.loc[events.event_type.eq("pause"), "duration_sec"].sum() == pytest.approx(.4)
    assert events.event_type.eq("excluded_contamination").sum() == 1
    assert events.loc[events.event_type.eq("excluded_contamination"), "qualifies_as_pause"].eq(False).all()
    assert original_hashes == (sha256_file(wav), sha256_file(automatic_segments))


def test_family09_requires_frozen_authoritative_paths(tmp_path: Path):
    _wav, _automatic_segments, summary = _run(tmp_path)
    with pytest.raises(ValueError, match="frozen authoritative"):
        run_acoustic_feature_extraction(
            summary, tmp_path,
            config=FeatureExtractionConfig(selected_features=["pause_count"]))


def test_nonrecommended_wstg_can_extract_implemented_families_with_real_inputs(tmp_path: Path):
    _wav, _automatic_segments, _summary = _run(tmp_path, task_name="WSTG")
    save_segmentation_review_entry(
        tmp_path, "r", "KEEP_MANUAL", "Reviewer", "Exploratory WSTG timing",
        "0.8,1.2\n1.6,3.5")
    finalize_segmentation_review(tmp_path)
    prompt = tmp_path / "wstg_prompt.json"
    prompt.write_text(json.dumps({
        "schema_version": "1", "task_id": "wstg", "prompt_version": "WSTG-v1",
        "count_source": "explicit_reviewed_prompt", "syllable_count": 12,
        "word_count": 6, "applies_to_all_recordings": True,
    }), encoding="utf-8")
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    result = run_acoustic_feature_extraction(
        final / "final_segmentation_decisions.csv", tmp_path,
        config=FeatureExtractionConfig(selected_features=[
            "speaking_rate_syll_s", "articulation_rate_syll_s", "pause_count"],
            prompt_manifest_path=str(prompt)),
        final_segmentation_intervals_csv=final / "final_segmentation_intervals.csv")
    values = pd.read_csv(result.summary_table)
    status = pd.read_csv(result.summary_table.parent / "acoustic_feature_status_long.csv")
    assert values.task_id.iloc[0] == "wstg"
    assert np.isclose(values.speaking_rate_syll_s.iloc[0], 12 / 2.7)
    assert np.isclose(values.articulation_rate_syll_s.iloc[0], 12 / 2.3)
    assert values.pause_count.iloc[0] == 1
    assert status.status.eq("computed").all()
    assert not status.failure_reason.astype(str).str.contains("task_not_supported").any()


def test_family10_consumes_frozen_reviewed_ddk_events_and_exclusion(tmp_path: Path):
    wav, automatic_segments, _summary = _run(tmp_path, task_name="DDK", method="ddk_energy")
    original_hashes = sha256_file(wav), sha256_file(automatic_segments)
    manual_events = [(center - .05, center + .05) for center in
                     (.4, .7, 1.0, 1.3, 1.6, 2.6, 2.9)]
    save_segmentation_review_entry(
        tmp_path, "r", "KEEP_MANUAL", "Reviewer", "Cough interrupted DDK",
        "\n".join(f"{start},{end}" for start, end in manual_events),
        analysis_start_sec=.2, analysis_end_sec=3.2,
        exclusion_intervals=[{"start_sec": 2.0, "end_sec": 2.2,
                              "exclusion_reason": "Cough / throat clear"}])
    finalize_segmentation_review(tmp_path)
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    result = run_acoustic_feature_extraction(
        final / "final_segmentation_decisions.csv", tmp_path,
        config=FeatureExtractionConfig(selected_features=["ddk_rate_syll_s", "ddk_cycle_mad_s"]),
        final_segmentation_intervals_csv=final / "final_segmentation_intervals.csv")
    values = pd.read_csv(result.summary_table)
    status = pd.read_csv(result.summary_table.parent / "acoustic_feature_status_long.csv")
    events = pd.read_csv(result.summary_table.parent.parent / "family_runs" / "ddk" / "tables" / "native_measurements" /
                         "ddk_feature_events.csv")
    assert values.n_ddk_events.iloc[0] == 7
    assert values.n_valid_sequences.iloc[0] == 2
    assert np.isclose(values.raw_analysis_duration_sec.iloc[0], 3)
    assert np.isclose(values.excluded_contamination_duration_sec.iloc[0], .2)
    assert np.isclose(values.ddk_rate_syll_s.iloc[0], 7 / 2.8)
    assert np.isclose(values.ddk_cycle_mad_s.iloc[0], 0)
    assert status.status.eq("computed").all()
    assert status.family_id.eq("F10").all()
    assert status.segmentation_run_id.eq("seg_test").all()
    assert events.event_kind.eq("manual_exclusion").sum() == 1
    assert events.loc[events.event_time_sec.eq(1.6), "next_event_interval_sec"].isna().all()
    assert original_hashes == (sha256_file(wav), sha256_file(automatic_segments))


def test_auto_modified_provenance_and_trim_without_leading_pause(tmp_path: Path):
    wav, automatic_segments, _ = _run(tmp_path)
    original = sha256_file(automatic_segments)
    save_segmentation_review_entry(tmp_path, "r", "KEEP_AUTO", "Reviewer", "Removed setup speech",
        analysis_start_sec=1.3, analysis_end_sec=3.7,
        exclusion_intervals=[{"start_sec": 2, "end_sec": 2.2,
            "exclusion_reason": "Other speaker", "notes": "Background voice"}])
    finalize_segmentation_review(tmp_path)
    tables = tmp_path / "acoustic" / "003_segmentation_review" / "runs" / "review_default" / "tables"
    decisions = pd.read_csv(tmp_path / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_decisions.csv", keep_default_na=False)
    timeline = pd.read_csv(tmp_path / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_intervals.csv", keep_default_na=False)
    assert decisions.iloc[0].final_decision == "KEEP_AUTO"
    assert decisions.iloc[0].boundary_source == "AUTO_MODIFIED"
    assert timeline.loc[timeline.segment_role.eq("speech"), "boundary_source"].eq("AUTO_MODIFIED").all()
    leading = timeline.loc[timeline.view.eq("authoritative") & timeline.segment_role.eq("leading_nonspeech")].iloc[0]
    assert (float(leading.start_sec), float(leading.end_sec)) == (1.3, 1.4)
    outside = timeline.loc[timeline.view.eq("authoritative") & timeline.segment_role.eq("outside_analysis_window")].iloc[0]
    assert (float(outside.start_sec), float(outside.end_sec)) == (0, 1.3)
    assert sha256_file(automatic_segments) == original and wav.is_file()


def test_atomic_review_entry_recovers_stale_csv_views(tmp_path: Path):
    _run(tmp_path)
    save_segmentation_review_entry(tmp_path, "r", "KEEP_MANUAL", "Reviewer", "Edited event",
        "1.4,3.4", analysis_start_sec=1.2)
    tables = tmp_path / "acoustic" / "003_segmentation_review" / "runs" / "review_default" / "tables"
    stale = pd.read_csv(tables / "review_decisions.csv", keep_default_na=False)
    stale.loc[0, "final_decision"] = ""
    stale.to_csv(tables / "review_decisions.csv", index=False)
    pd.DataFrame(columns=["recording_id", "file_name", "segment_index", "start_sec", "end_sec",
                          "reviewer", "review_date", "review_notes"]).to_csv(
        tables / "manual_overrides.csv", index=False)
    decisions, overrides = load_review_state(tmp_path)
    assert decisions.iloc[0].final_decision == "KEEP_MANUAL"
    assert decisions.iloc[0].analysis_start_sec == "1.2"
    assert len(overrides) == 1
