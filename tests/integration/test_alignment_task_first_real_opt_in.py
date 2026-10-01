"""Optional nonclinical task-first WSTG GUI contract against real external MFA."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.alignment import freeze_alignment, list_alignment_runs, load_final_alignment
from vslp.acoustic.alignment.self_test import generate_windows_speech
from vslp.acoustic.alignment.task_workflow import check_task_preflight, run_task_alignment
from vslp.acoustic.alignment.task_workflow import save_project_choices
from vslp.acoustic.alignment.trial_review import set_trial_review


def test_real_wstg_task_first_alignment(tmp_path):
    if os.environ.get("VSLP_RUN_REAL_MFA") != "1":
        pytest.skip("Set VSLP_RUN_REAL_MFA=1 for real nonclinical WSTG alignment")
    root = tmp_path / "wstg_engineering_project"
    root.mkdir()
    wav = root / "opaque_recording.wav"
    sentence = root / "nonclinical_sentence.wav"
    generate_windows_speech(sentence, "We see three geese.")
    samples, rate = sf.read(sentence, dtype="float32")
    silence = np.zeros(rate, dtype="float32")
    sf.write(wav, np.concatenate([samples, silence, samples, silence, samples]), rate)
    duration = sf.info(wav).duration
    sentence_duration = len(samples) / rate
    trial_rows = [{"trial_id": f"trial_{number + 1:03d}",
                   "start_sec": number * (sentence_duration + 1),
                   "end_sec": min(duration, number * (sentence_duration + 1) + sentence_duration)}
                  for number in range(3)]
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
        "segment_role": "speech", "start_sec": trial["start_sec"],
        "end_sec": trial["end_sec"]} for trial in trial_rows]).to_csv(
        review / "final_segmentation_intervals.csv", index=False)
    save_project_choices(root, {"schema_version": "1", "recordings": {},
        "trials": {"r1": trial_rows}})
    metadata = root / "acoustic" / "000_metadata" / "tables"
    metadata.mkdir(parents=True)
    pd.DataFrame([{"recording_id": "r1", "subject_id": "synthetic_speaker_1",
        "metadata_source": "metadata_csv", "subject_id_source": "metadata_csv"}]).to_csv(
        metadata / "project_file_index.csv", index=False)
    preflight = check_task_preflight(root)
    assert preflight["status"] == "READY", preflight.get("issue")
    result = run_task_alignment(root)
    assert result.status == "completed"
    run_id = list_alignment_runs(root)[-1]["alignment_run_id"]
    for trial in trial_rows:
        set_trial_review(root, run_id, "r1", trial["trial_id"], "ACCEPTED")
    freeze_alignment(root, run_id)
    store, issue = load_final_alignment(root)
    assert issue == "" and store is not None
    assert len(store.words) == 12
    assert set(store.words.word.str.casefold()) >= {"we", "see", "three", "geese"}
    assert len(store.phones) > 0
    assert store.manifest["prompt_id_by_recording"] == {"r1": "wstg_we_see_three_geese"}
    assert store.manifest["protocol_repetition_shortfall_by_recording"] == {"r1": 0}
