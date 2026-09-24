"""Family 07 integration with frozen review and explicit alignment inputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.alignment import AlignmentConfig, freeze_alignment, list_alignment_runs, run_acoustic_alignment
from vslp.acoustic.alignment.stage import PHONE_SET
from vslp.acoustic.features.family07 import FAMILY07_IDS
from vslp.acoustic.features.family08 import run_reviewed_timing_stage
from vslp.acoustic.features.stage import FeatureExtractionConfig, _select_registry


def test_family07_stage_uses_reviewed_timing_alignment_and_audit(tmp_path, monkeypatch):
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    wave = tmp_path / "reviewed.wav"
    sf.write(wave, np.zeros(10 * 16000, dtype="float32"), 16000, subtype="FLOAT")
    pd.DataFrame([{"recording_id": "r1", "file_name": "reading.wav",
                   "final_decision": "KEEP_MANUAL", "analysis_wav_path": str(wave),
                   "analysis_start_sec": 0, "analysis_end_sec": 10,
                   "segmentation_run_id": "seg1", "review_run_id": "rev1"}]).to_csv(decisions, index=False)
    timeline = []
    for index, (role, start, end) in enumerate([
        ("leading_nonspeech", 0, 1), ("speech", 1, 2),
        ("internal_nonspeech", 2, 2.2), ("speech", 2.2, 4),
        ("manual_exclusion", 4, 4.5), ("speech", 4.5, 6),
        ("internal_nonspeech", 6, 6.4), ("speech", 6.4, 9),
        ("trailing_nonspeech", 9, 10),
    ]):
        timeline.append({"recording_id": "r1", "view": "authoritative",
                         "interval_index": index + 1, "segment_role": role,
                         "start_sec": start, "end_sec": end,
                         "analysis_start_sec": 0, "analysis_end_sec": 10,
                         "boundary_source": "MANUAL"})
    pd.DataFrame(timeline).to_csv(intervals, index=False)
    alignment = pd.DataFrame([
        {"recording_id": "r1", "token_type": "word", "label": f"w{i}",
             "start_sec": start, "end_sec": start + duration, "expected_count": 4,
             "alignment_source": "frozen_test"}
        for i, (start, duration) in enumerate([(1.1, .25), (2.5, .25),
                                                (5.0, .35), (7.0, .45)])
    ] + [
        {"recording_id": "r1", "token_type": "phone", "label": "AA1",
         "start_sec": start, "end_sec": start + duration,
         "alignment_source": "frozen_test"}
        for start, duration in [(1.2, .1), (2.52, .2), (5.02, .3), (7.02, .4)]
    ])
    alignment_path = tmp_path / "alignment.csv"
    alignment.to_csv(alignment_path, index=False)
    prompt_path = tmp_path / "prompt.json"
    prompt_path.write_text(json.dumps({"manifest_version": "1", "task_id": "bamboo_passage",
                                       "prompt_version": "v1", "language": "en",
                                       "transcript": "w0 w1 w2 w3",
                                       "expected_words": ["w0", "w1", "w2", "w3"],
                                       "phone_set": PHONE_SET}), encoding="utf-8")
    (tmp_path / "project_manifest.json").write_text(json.dumps({
        "project_name": "P", "task_name": "Bamboo Passage",
        "task_id": "bamboo_passage", "task_type": "Passage / connected speech",
        "run_id": "run1",
    }), encoding="utf-8")
    kept = pd.DataFrame([{
        "recording_id": "r1", "file_name": "reading.wav", "duration_sec": 10,
        "source_sha256": "a" * 64, "source_file_path": "source.wav",
        "segmentation_run_id": "seg1", "review_run_id": "rev1",
        "boundary_source": "MANUAL",
    }])
    monkeypatch.setattr("vslp.acoustic.features.family08.load_final_segmentation",
                        lambda *_: kept)
    run_acoustic_alignment(tmp_path, AlignmentConfig(
        prompt_manifest_path=str(prompt_path), words_csv=str(alignment_path),
        phones_csv=str(alignment_path)))
    freeze_alignment(tmp_path, list_alignment_runs(tmp_path)[0]["alignment_run_id"])
    cfg = FeatureExtractionConfig(selected_features=sorted(FAMILY07_IDS))
    result = run_reviewed_timing_stage(decisions, tmp_path, cfg, intervals,
                                       _select_registry(cfg))
    values = pd.read_csv(result.summary_table)
    assert values.speech_time_s.iloc[0] == 7.1
    assert np.isclose(values.task_elapsed_duration_s.iloc[0], 5.85)
    assert values.task_boundary_method.iloc[0] == "first_to_last_valid_aligned_word"
    assert np.isclose(values.mean_word_duration_s.iloc[0], .325)
    vowel_durations = np.array([.1, .2, .3, .4])
    assert np.isclose(values.delta_v_s.iloc[0], np.std(vowel_durations, ddof=1))
    assert np.isclose(values.varco_v_pct.iloc[0],
                      100 * np.std(vowel_durations, ddof=1) / vowel_durations.mean())
    assert values.task_type.iloc[0] == "Passage / connected speech"
    audit = pd.read_csv(tmp_path / "acoustic" / "005_features" / "tables" /
                        "native_measurements" / "family07_duration_events.csv")
    assert set(audit.event_type) >= {"word", "vowel", "pause", "excluded_contamination"}
    assert not audit.loc[audit.event_type.eq("manual_exclusion")].any().any()


def test_family07_alignment_outputs_fail_explicitly_when_alignment_missing(tmp_path, monkeypatch):
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    decisions.write_text("recording_id,final_decision\nr1,KEEP_AUTO\n", encoding="utf-8")
    pd.DataFrame([{"recording_id": "r1", "view": "authoritative",
                   "segment_role": "speech", "start_sec": 1, "end_sec": 3,
                   "analysis_start_sec": 0, "analysis_end_sec": 4,
                   "boundary_source": "AUTO"}]).to_csv(intervals, index=False)
    (tmp_path / "project_manifest.json").write_text(json.dumps({
        "project_name": "P", "task_name": "Bamboo Passage", "run_id": "run1",
    }), encoding="utf-8")
    kept = pd.DataFrame([{"recording_id": "r1", "file_name": "x.wav", "duration_sec": 4,
                         "source_sha256": "b" * 64, "source_file_path": "source.wav",
                         "segmentation_run_id": "seg", "review_run_id": "rev",
                         "boundary_source": "AUTO"}])
    monkeypatch.setattr("vslp.acoustic.features.family08.load_final_segmentation",
                        lambda *_: kept)
    cfg = FeatureExtractionConfig(selected_features=["mean_word_duration_s"])
    result = run_reviewed_timing_stage(decisions, tmp_path, cfg, intervals,
                                       _select_registry(cfg))
    status = pd.read_csv(result.summary_table.parent / "acoustic_feature_status_long.csv")
    assert status.failure_reason.iloc[0] == "missing_alignment"
    assert np.isnan(status.value.iloc[0])
