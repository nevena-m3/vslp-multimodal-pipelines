"""Optional nonclinical task-first WSTG GUI contract against real external MFA."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.alignment import freeze_alignment, list_alignment_runs, load_final_alignment
from vslp.acoustic.alignment.self_test import generate_windows_speech
from vslp.acoustic.alignment.task_workflow import check_task_preflight, run_task_alignment


def test_real_wstg_task_first_alignment(tmp_path):
    if os.environ.get("VSLP_RUN_REAL_MFA") != "1":
        pytest.skip("Set VSLP_RUN_REAL_MFA=1 for real nonclinical WSTG alignment")
    root = tmp_path / "wstg_engineering_project"
    root.mkdir()
    wav = root / "opaque_recording.wav"
    generate_windows_speech(wav, "We see three geese.")
    duration = sf.info(wav).duration
    (root / "project_manifest.json").write_text(json.dumps({
        "task_id": "wstg", "task_name": "WSTG",
        "task_type": "Sentence / controlled speech"}), encoding="utf-8")
    review = root / "acoustic" / "003_segmentation_review" / "final"
    review.mkdir(parents=True)
    pd.DataFrame([{"recording_id": "r1", "file_name": wav.name,
        "final_decision": "KEEP_MANUAL", "analysis_wav_path": str(wav),
        "analysis_start_sec": 0.0, "analysis_end_sec": duration,
        "segmentation_run_id": "demo_seg", "review_run_id": "demo_review"}]).to_csv(
        review / "final_segmentation_decisions.csv", index=False)
    pd.DataFrame([{"recording_id": "r1", "view": "authoritative",
        "segment_role": "speech", "start_sec": 0.0, "end_sec": duration}]).to_csv(
        review / "final_segmentation_intervals.csv", index=False)
    metadata = root / "acoustic" / "000_metadata" / "tables"
    metadata.mkdir(parents=True)
    pd.DataFrame([{"recording_id": "r1", "subject_id": "synthetic_speaker_1",
        "metadata_source": "metadata_csv", "subject_id_source": "metadata_csv"}]).to_csv(
        metadata / "project_file_index.csv", index=False)
    assert check_task_preflight(root)["status"] == "READY"
    result = run_task_alignment(root)
    assert result.status == "completed"
    run_id = list_alignment_runs(root)[-1]["alignment_run_id"]
    freeze_alignment(root, run_id)
    store, issue = load_final_alignment(root)
    assert issue == "" and store is not None
    assert set(store.words.word.str.casefold()) >= {"we", "see", "three", "geese"}
    assert len(store.phones) > 0
    assert store.manifest["prompt_id_by_recording"] == {"r1": "wstg_we_see_three_geese"}
