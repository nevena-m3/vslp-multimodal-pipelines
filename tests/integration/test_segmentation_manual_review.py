import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.segment.review import (
    finalize_segmentation_review, initialize_segmentation_review,
    load_final_segmentation, load_review_state, parse_manual_intervals_text,
    preview_manual_segmentation, save_segmentation_review_entry,
    validate_manual_interval_list,
)
from vslp.core.provenance import sha256_file
from vslp.acoustic.quality.stage import QualityControlConfig, run_acoustic_quality_control
from vslp.acoustic.features.stage import FeatureExtractionConfig, run_acoustic_feature_extraction


@pytest.fixture
def review_run(tmp_path: Path):
    sr = 16000
    t = np.arange(sr * 3) / sr
    wav = tmp_path / "canonical.wav"
    sf.write(wav, (0.2 * np.sin(2 * np.pi * 140 * t)).astype("float32"), sr, subtype="FLOAT")
    records = []
    for index, status in enumerate(("ACCEPTED", "REVIEW", "EXCLUDED", "FAILED")):
        recording_id = status.lower()
        segments = tmp_path / f"{recording_id}_auto.csv"
        if status != "FAILED":
            rows = [{"segment_type": "nonspeech", "segment_role": "leading_nonspeech", "start_sec": 0, "end_sec": .4, "duration_sec": .4}]
            if status != "EXCLUDED":
                rows.append({"segment_type": "speech", "segment_role": "speech", "start_sec": .4, "end_sec": 2.5, "duration_sec": 2.1})
            pd.DataFrame(rows).to_csv(segments, index=False)
        records.append({"recording_id": recording_id, "file_name": f"{recording_id}.wav",
            "task_name": "Bamboo Passage", "segmentation_method": "silero_vad",
            "automatic_status": status, "flags": "weak_boundary" if status == "REVIEW" else "",
            "review_required": status in {"REVIEW", "EXCLUDED"}, "duration_sec": 3.0,
            "analysis_wav_path": str(wav), "segments_csv_path": str(segments) if status != "FAILED" else "",
            "frame_csv_path": "", "boundaries_csv_path": "", "plot_png_path": "",
            "source_sha256": chr(ord("a") + index) * 64, "project_name": "P", "run_id": "R"})
    source = tmp_path / "automatic.csv"
    pd.DataFrame(records).to_csv(source, index=False)
    return tmp_path, source, wav


def test_keep_auto_manual_exclude_rescue_and_freeze(review_run):
    root, source, wav = review_run
    source_hash, wav_hash = sha256_file(source), sha256_file(wav)
    initialize_segmentation_review(source, root)
    decisions, _ = load_review_state(root)
    assert decisions.set_index("recording_id").loc["accepted", "final_decision"] == "KEEP_AUTO"
    assert decisions.set_index("recording_id").loc["review", "final_decision"] == ""
    assert decisions.set_index("recording_id").loc["failed", "final_decision"] == "FAILED"
    with pytest.raises(ValueError, match="required segmentation reviews"):
        finalize_segmentation_review(root)
    save_segmentation_review_entry(root, "review", "KEEP_AUTO", "Nevena")
    save_segmentation_review_entry(root, "excluded", "KEEP_MANUAL", "Nevena",
                                   "Weak automatic detection", "0.5,1.1\n1.3,2.4")
    preview = preview_manual_segmentation(root, "excluded", "0.5,1.1\n1.3,2.4")
    assert preview.is_file()
    initialize_segmentation_review(source, root)  # Restart must preserve state.
    decisions, overrides = load_review_state(root)
    assert decisions.set_index("recording_id").loc["excluded", "final_decision"] == "KEEP_MANUAL"
    assert len(overrides) == 2
    result = finalize_segmentation_review(root)
    assert result.summary_table.is_file()
    paths = root / "acoustic" / "003_segmentation_review" / "tables"
    final = pd.read_csv(paths / "final_segmentation_decisions.csv", keep_default_na=False)
    intervals = pd.read_csv(paths / "final_segmentation_intervals.csv")
    assert set(final.final_decision) == {"KEEP_AUTO", "KEEP_MANUAL", "FAILED"}
    manual = intervals.loc[intervals.recording_id.eq("excluded") & intervals.view.eq("primary_speech")]
    assert len(manual) == 2 and manual.boundary_source.eq("MANUAL").all()
    assert manual.reviewer.eq("Nevena").all()
    assert len(load_final_segmentation(paths / "final_segmentation_intervals.csv",
                                       paths / "final_segmentation_decisions.csv")) == 3
    assert sha256_file(source) == source_hash and sha256_file(wav) == wav_hash
    with pytest.raises(FileExistsError):
        finalize_segmentation_review(root)


def test_exclusion_and_failed_are_distinct(review_run):
    root, source, _ = review_run
    initialize_segmentation_review(source, root)
    with pytest.raises(ValueError, match="FAILED"):
        save_segmentation_review_entry(root, "failed", "KEEP_MANUAL", "Nevena", "rescue", "0,1")
    save_segmentation_review_entry(root, "review", "EXCLUDE", "Nevena", "Ambiguous recording")
    save_segmentation_review_entry(root, "excluded", "EXCLUDE", "Nevena", "No usable boundaries")
    finalize_segmentation_review(root)
    tables = root / "acoustic" / "003_segmentation_review" / "tables"
    final = pd.read_csv(tables / "final_segmentation_decisions.csv", keep_default_na=False)
    assert final.set_index("recording_id").loc["review", "boundary_source"] == "NONE"
    assert final.set_index("recording_id").loc["failed", "final_decision"] == "FAILED"
    assert not pd.read_csv(tables / "final_segmentation_intervals.csv").recording_id.isin(["review", "failed"]).any()


@pytest.mark.parametrize("text", ["", "x,1", "nan,1", "0,0", "-1,1", "0,4", "0,1\n0.5,1.5"])
def test_invalid_manual_intervals_are_rejected(text):
    with pytest.raises(ValueError):
        validate_manual_interval_list(parse_manual_intervals_text(text), duration_sec=3)


def test_qc_and_features_consume_frozen_intervals(review_run):
    root, source, wav = review_run
    (root / "project_manifest.json").write_text(json.dumps({
        "project_name": "P", "task_name": "Bamboo Passage", "run_id": "R",
        "created_at_local": "2026-01-01T00:00:00", "created_at_utc": "2026-01-01T05:00:00Z",
    }), encoding="utf-8")
    initialize_segmentation_review(source, root)
    save_segmentation_review_entry(root, "review", "KEEP_MANUAL", "Reviewer", "Adjusted onset", "0.6,2.6")
    save_segmentation_review_entry(root, "excluded", "EXCLUDE", "Reviewer", "No usable event")
    finalize_segmentation_review(root)
    tables = root / "acoustic" / "003_segmentation_review" / "tables"
    decisions = tables / "final_segmentation_decisions.csv"
    intervals = tables / "final_segmentation_intervals.csv"
    original_hash = sha256_file(wav)
    qc = run_acoustic_quality_control(decisions, root,
        config=QualityControlConfig(selected_families=["gain_dynamics"]),
        final_segmentation_intervals_csv=intervals)
    assert qc.summary_table.is_file()
    features = run_acoustic_feature_extraction(decisions, root,
        config=FeatureExtractionConfig(selected_features=["f0_mean"]),
        final_segmentation_intervals_csv=intervals)
    assert features.summary_table.is_file()
    measured = pd.read_csv(features.summary_table)
    assert set(measured.file_name) == {"accepted.wav", "review.wav"}
    assert sha256_file(wav) == original_hash
