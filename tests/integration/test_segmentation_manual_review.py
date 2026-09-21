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
from vslp.acoustic.segment import review as review_module
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
    assert decisions.set_index("recording_id").loc["review", "final_decision"] == "PENDING"
    assert decisions.set_index("recording_id").loc["failed", "final_decision"] == "FAILED"
    with pytest.raises(ValueError, match="required segmentation reviews"):
        finalize_segmentation_review(root)
    save_segmentation_review_entry(root, "review", "KEEP_AUTO", "Nevena")
    save_segmentation_review_entry(root, "excluded", "KEEP_MANUAL", "Nevena",
                                   "Weak automatic detection", "0.5,1.1\n1.3,2.4")
    save_segmentation_review_entry(root, "failed", "EXCLUDE", "Nevena", "Computational failure")
    preview = preview_manual_segmentation(root, "excluded", "0.5,1.1\n1.3,2.4")
    assert preview.is_file()
    initialize_segmentation_review(source, root)  # Restart must preserve state.
    decisions, overrides = load_review_state(root)
    assert decisions.set_index("recording_id").loc["excluded", "final_decision"] == "KEEP_MANUAL"
    assert len(overrides) == 2
    result = finalize_segmentation_review(root)
    assert result.summary_table.is_file()
    paths = root / "acoustic" / "003_segmentation_review" / "runs" / "review_default" / "tables"
    final_dir = root / "acoustic" / "003_segmentation_review" / "final"
    final = pd.read_csv(final_dir / "final_segmentation_decisions.csv", keep_default_na=False)
    intervals = pd.read_csv(final_dir / "final_segmentation_intervals.csv")
    assert not (root / "acoustic" / "003_segmentation_review" / "tables").exists()
    assert set(final.recording_id) == {"accepted", "review", "excluded", "failed"}
    assert final.set_index("recording_id").loc["accepted", "review_status"] == "UNREVIEWED"
    assert final.set_index("recording_id").loc["accepted", "boundary_source"] == "AUTO"
    assert intervals.loc[intervals.recording_id.eq("accepted") & intervals.segment_role.eq("speech")].shape[0] == 1
    assert intervals.loc[intervals.recording_id.eq("failed")].empty
    summary = pd.read_csv(final_dir / "final_segmentation_summary.csv").iloc[0]
    assert (summary.n_total, summary.n_keep_auto, summary.n_keep_manual, summary.n_excluded, summary.n_failed) == (4, 2, 1, 0, 1)
    manifest = json.loads((final_dir / "final_manifest.json").read_text(encoding="utf-8"))
    assert manifest["decisions_sha256"] == sha256_file(final_dir / "final_segmentation_decisions.csv")
    assert manifest["intervals_sha256"] == sha256_file(final_dir / "final_segmentation_intervals.csv")
    assert (root / "acoustic" / "003_segmentation_review" / "logs" / "review_registry.json").is_file()
    assert set(final.final_decision) == {"KEEP_AUTO", "KEEP_MANUAL", "FAILED"}
    assert final.set_index("recording_id").loc["failed", "automatic_status"] == "FAILED"
    manual = intervals.loc[intervals.recording_id.eq("excluded") & intervals.segment_role.eq("speech")]
    assert len(manual) == 2 and manual.boundary_source.eq("MANUAL").all()
    assert manual.reviewer.eq("Nevena").all()
    assert len(load_final_segmentation(final_dir / "final_segmentation_intervals.csv",
                                       final_dir / "final_segmentation_decisions.csv")) == 3
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
    save_segmentation_review_entry(root, "failed", "EXCLUDE", "Nevena", "Computational failure")
    finalize_segmentation_review(root)
    tables = root / "acoustic" / "003_segmentation_review" / "runs" / "review_default" / "tables"
    final = pd.read_csv(root / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_decisions.csv", keep_default_na=False)
    assert final.set_index("recording_id").loc["review", "boundary_source"] == "NONE"
    assert final.set_index("recording_id").loc["failed", "final_decision"] == "FAILED"
    assert final.set_index("recording_id").loc["failed", "automatic_status"] == "FAILED"
    assert not pd.read_csv(root / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_intervals.csv").recording_id.isin(["review", "failed"]).any()


def test_unreviewed_computational_failure_remains_failed(review_run):
    root, source, _ = review_run
    initialize_segmentation_review(source, root)
    save_segmentation_review_entry(root, "review", "KEEP_AUTO", "Reviewer")
    save_segmentation_review_entry(root, "excluded", "EXCLUDE", "Reviewer", "No usable interval")
    finalize_segmentation_review(root)
    final_dir = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = pd.read_csv(final_dir / "final_segmentation_decisions.csv", keep_default_na=False)
    assert decisions.set_index("recording_id").loc["failed", "final_decision"] == "FAILED"
    intervals = pd.read_csv(final_dir / "final_segmentation_intervals.csv")
    assert not intervals.recording_id.eq("failed").any()


def test_working_tables_audit_and_reviewed_plot_are_immediate(review_run):
    root, source, wav = review_run
    automatic_hash = sha256_file(source)
    wav_hash = sha256_file(wav)
    initialize_segmentation_review(source, root)
    save_segmentation_review_entry(root, "review", "PENDING", "Reviewer", "Trim setup",
        analysis_start_sec=.5, analysis_end_sec=2.8, action="SET_ANALYSIS_START")
    save_segmentation_review_entry(root, "review", "KEEP_MANUAL", "Reviewer", "Corrected task",
        "0.6,1.2\n1.4,2.5", analysis_start_sec=.5, analysis_end_sec=2.8,
        exclusion_intervals=[{"start_sec": 1.8, "end_sec": 1.9,
                              "exclusion_reason": "Cough / throat clear", "notes": "cough"}])
    tables = root / "acoustic" / "003_segmentation_review" / "runs" / "review_default" / "tables"
    decisions = pd.read_csv(tables / "review_decisions.csv", keep_default_na=False)
    selected = decisions.set_index("recording_id").loc["review"]
    assert selected.final_decision == "KEEP_MANUAL"
    assert selected.analysis_start_sec == .5
    assert len(pd.read_csv(tables / "manual_overrides.csv")) == 2
    assert len(pd.read_csv(tables / "exclusion_intervals.csv")) == 1
    assert len(pd.read_csv(tables / "analysis_windows.csv")) == 1
    audit = pd.read_csv(tables / "audit_log.csv")
    assert list(audit.action) == ["SET_ANALYSIS_START", "SAVE_MANUAL"]
    assert Path(selected.reviewed_plot_path).is_file()
    assert "review__" in Path(selected.reviewed_plot_path).name
    assert sha256_file(source) == automatic_hash and sha256_file(wav) == wav_hash
    initialize_segmentation_review(source, root)
    restarted, overrides = load_review_state(root)
    assert restarted.set_index("recording_id").loc["review", "final_decision"] == "KEEP_MANUAL"
    assert len(overrides) == 2


def test_atomic_projection_failure_keeps_previous_csv(review_run, monkeypatch):
    root, source, _ = review_run
    initialize_segmentation_review(source, root)
    save_segmentation_review_entry(root, "review", "KEEP_AUTO", "Reviewer")
    destination = root / "acoustic" / "003_segmentation_review" / "runs" / "review_default" / "tables" / "review_decisions.csv"
    previous = destination.read_bytes()
    original_replace = review_module.os.replace

    def interrupt(src, dst):
        if Path(dst) == destination:
            raise OSError("interrupted projection")
        return original_replace(src, dst)

    monkeypatch.setattr(review_module.os, "replace", interrupt)
    with pytest.raises(OSError, match="interrupted projection"):
        save_segmentation_review_entry(root, "review", "EXCLUDE", "Reviewer", "Exclude")
    assert destination.read_bytes() == previous
    monkeypatch.setattr(review_module.os, "replace", original_replace)
    recovered, _ = load_review_state(root)
    assert recovered.set_index("recording_id").loc["review", "final_decision"] == "EXCLUDE"


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
    save_segmentation_review_entry(root, "failed", "EXCLUDE", "Reviewer", "Computational failure")
    finalize_segmentation_review(root)
    tables = root / "acoustic" / "003_segmentation_review" / "runs" / "review_default" / "tables"
    decisions = root / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_decisions.csv"
    intervals = root / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_intervals.csv"
    original_hash = sha256_file(wav)
    qc = run_acoustic_quality_control(decisions, root,
        config=QualityControlConfig(selected_families=["gain_dynamics"]),
        final_segmentation_intervals_csv=intervals)
    assert qc.summary_table.is_file()
    with pytest.raises(ValueError, match="legacy feature IDs cannot execute"):
        run_acoustic_feature_extraction(decisions, root,
            config=FeatureExtractionConfig(selected_features=["f0_mean"]),
            final_segmentation_intervals_csv=intervals)
    assert not (root / "acoustic" / "005_features" / "tables").exists()
    with pytest.raises(ValueError, match="authoritative final"):
        run_acoustic_quality_control(source, root,
            config=QualityControlConfig(selected_families=["gain_dynamics"]))
    with pytest.raises(ValueError, match="authoritative final"):
        run_acoustic_feature_extraction(source, root,
            config=FeatureExtractionConfig(selected_features=["f0_mean"]))
    final_decisions = pd.read_csv(decisions)
    assert set(final_decisions.file_name) == {
        "accepted.wav", "review.wav", "excluded.wav", "failed.wav"}
    assert sha256_file(wav) == original_hash
