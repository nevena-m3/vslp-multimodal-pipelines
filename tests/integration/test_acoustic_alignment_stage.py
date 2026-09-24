"""External and provider alignment resolve to one frozen output contract."""

from __future__ import annotations

import json
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.alignment import (
    AlignmentConfig, freeze_alignment, list_alignment_runs, load_final_alignment,
    run_acoustic_alignment,
)
from vslp.acoustic.alignment.stage import PHONE_SET, WORD_COLUMNS, PHONE_COLUMNS


def _reviewed_root(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "project_manifest.json").write_text(json.dumps({
        "task_id": "bamboo", "task_name": "Bamboo Passage"}), encoding="utf-8")
    wave = root / "canonical.wav"
    sf.write(wave, np.zeros(10 * 16000, dtype="float32"), 16000, subtype="FLOAT")
    final = root / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    pd.DataFrame([{"recording_id": "r1", "file_name": "reading.wav",
                   "final_decision": "KEEP_MANUAL", "analysis_wav_path": str(wave),
                   "analysis_start_sec": 1, "analysis_end_sec": 9,
                   "segmentation_run_id": "seg1", "review_run_id": "rev1"}]).to_csv(
                       final / "final_segmentation_decisions.csv", index=False)
    pd.DataFrame([{"recording_id": "r1", "view": "authoritative",
                   "segment_role": "speech", "start_sec": 1, "end_sec": 4},
                  {"recording_id": "r1", "view": "authoritative",
                   "segment_role": "manual_exclusion", "start_sec": 4, "end_sec": 4.5},
                  {"recording_id": "r1", "view": "authoritative",
                   "segment_role": "speech", "start_sec": 4.5, "end_sec": 9}]).to_csv(
                       final / "final_segmentation_intervals.csv", index=False)
    prompt = root / "prompt.json"
    prompt.write_text(json.dumps({"manifest_version": "1", "task_id": "bamboo",
                                  "prompt_version": "p1", "language": "en",
                                  "transcript": "one two three", "expected_words": ["one", "two", "three"],
                                  "phone_set": PHONE_SET}), encoding="utf-8")
    return root, prompt


def test_external_import_freeze_and_reload_respect_exclusion(tmp_path):
    root, prompt = _reviewed_root(tmp_path)
    words = root / "words.csv"
    phones = root / "phones.csv"
    pd.DataFrame([{"recording_id": "r1", "word": "one", "start_sec": 1.2, "end_sec": 1.5},
                  {"recording_id": "r1", "word": "two", "start_sec": 4.6, "end_sec": 4.9},
                  {"recording_id": "r1", "word": "three", "start_sec": 5, "end_sec": 5.4},
                  {"recording_id": "r1", "word": "cough", "start_sec": 4.1, "end_sec": 4.3}]).to_csv(words, index=False)
    pd.DataFrame([{"recording_id": "r1", "phone": "AH1", "word_index": 1,
                   "start_sec": 1.25, "end_sec": 1.45},
                  {"recording_id": "r1", "phone": "T", "word_index": 2,
                   "start_sec": 4.6, "end_sec": 4.7}]).to_csv(phones, index=False)
    progress = []
    result = run_acoustic_alignment(root, AlignmentConfig(
        prompt_manifest_path=str(prompt), words_csv=str(words), phones_csv=str(phones)),
        progress_callback=lambda done, total, message: progress.append((done, total, message)))
    assert result.status == "completed_with_warnings"
    assert [item[:2] for item in progress] == [(0, 1), (1, 1)]
    diagnostics = pd.read_csv(result.summary_table)
    assert diagnostics.status.iloc[0] == "INVALID_BOUNDARIES"
    # Corrected external annotation makes a separate immutable run.
    pd.read_csv(words).iloc[:3].to_csv(words, index=False)
    second = run_acoustic_alignment(root, AlignmentConfig(
        prompt_manifest_path=str(prompt), words_csv=str(words), phones_csv=str(phones)))
    assert second.status == "completed"
    runs = list_alignment_runs(root)
    assert len(runs) == 2
    run_state = root / "acoustic" / "004_alignment" / "runs" / runs[-1]["alignment_run_id"] / "logs" / "run_state.json"
    assert json.loads(run_state.read_text())["status"] == "COMPLETED"
    freeze_alignment(root, runs[-1]["alignment_run_id"])
    store, issue = load_final_alignment(root)
    assert issue == ""
    assert store is not None
    assert len(store.get_word_tokens("r1")) == 3
    assert len(store.get_vowel_tokens("r1")) == 1
    assert len(store.get_tokens_by_label("r1", "AH")) == 1
    assert set(WORD_COLUMNS) <= set(store.words)
    assert set(PHONE_COLUMNS) <= set(store.phones)
    assert store.words.start_sec.min() == pytest.approx(1.2)
    assert store.diagnostics.word_coverage.iloc[0] == 1
    assert store.manifest["alignment_status"] == "FROZEN"
    with pytest.raises(FileExistsError, match="immutable"):
        freeze_alignment(root, runs[0]["alignment_run_id"])


def test_provider_working_time_map_and_missing_prompt(tmp_path):
    root, prompt = _reviewed_root(tmp_path)
    wave = root / "canonical.wav"
    sf.write(wave, np.zeros(10 * 48000, dtype="float32"), 48000, subtype="FLOAT")
    original_hash = sha256(wave.read_bytes()).hexdigest()

    class FakeProvider:
        name = "fake"
        version = "1"

        def align_corpus(self, corpus, output, profile, logs):
            assert sf.info(corpus / "speaker_1" / "r1.wav").samplerate == 16000
            assert (corpus / "speaker_1" / "r1.lab").read_text() == "ONE TWO THREE"
            output.mkdir(parents=True)
            grid = output / "r1.TextGrid"
            grid.write_text('''name = "words"
intervals [1]:
  xmin = 0.2
  xmax = 0.5
  text = "one"
intervals [2]:
  xmin = 3.1
  xmax = 3.4
  text = "two"
intervals [3]:
  xmin = 3.6
  xmax = 4.0
  text = "three"
name = "phones"
intervals [1]:
  xmin = 0.25
  xmax = 0.4
  text = "AH1"
''', encoding="utf-8")
            return {"r1": grid}

    with pytest.raises(ValueError, match="MISSING_PROMPT"):
        run_acoustic_alignment(root, AlignmentConfig(source="mfa"))
    speakers = root / "speakers.json"
    speakers.write_text(json.dumps({"recordings": [{"recording_id": "r1", "speaker_id": "speaker_1"}]}))
    result = run_acoustic_alignment(root, AlignmentConfig(
        source="mfa", prompt_manifest_path=str(prompt), provider=FakeProvider(),
        speaker_manifest_path=str(speakers)))
    assert result.status == "completed"
    run_id = list_alignment_runs(root)[0]["alignment_run_id"]
    mapped = pd.read_csv(root / "acoustic" / "004_alignment" / "runs" / run_id /
                         "tables" / "alignment_words.csv")
    assert mapped.start_sec.tolist() == pytest.approx([1.2, 4.6, 5.1])
    assert mapped.sequence_id.tolist() == [1, 2, 2]
    assert sha256(wave.read_bytes()).hexdigest() == original_hash
    freeze_alignment(root, run_id)
    store, issue = load_final_alignment(root)
    assert issue == ""
    assert len(store.get_word_tokens("r1")) == 3


def test_batch_progress_includes_failed_recording(tmp_path):
    root, prompt = _reviewed_root(tmp_path)
    final = root / "acoustic" / "003_segmentation_review" / "final"
    decisions_path = final / "final_segmentation_decisions.csv"
    decisions = pd.read_csv(decisions_path)
    failed = decisions.iloc[0].copy()
    failed["recording_id"] = "r2"
    failed["file_name"] = "missing.wav"
    failed["analysis_wav_path"] = str(root / "missing.wav")
    pd.concat([decisions, pd.DataFrame([failed])], ignore_index=True).to_csv(decisions_path, index=False)
    words = root / "words.csv"
    pd.DataFrame([{"recording_id": "r1", "word": label, "start_sec": start,
                   "end_sec": start + .25}
                  for label, start in (("one", 1.2), ("two", 4.6), ("three", 5.0))]).to_csv(
                      words, index=False)
    progress = []
    result = run_acoustic_alignment(root, AlignmentConfig(
        prompt_manifest_path=str(prompt), words_csv=str(words)),
        progress_callback=lambda done, total, message: progress.append((done, total, message)))
    assert [(done, total) for done, total, _ in progress] == [(0, 2), (1, 2), (2, 2)]
    assert "missing.wav" in progress[-1][2]
    diagnostics = pd.read_csv(result.summary_table)
    assert diagnostics.set_index("recording_id").loc["r2", "status"] == "ALIGNMENT_FAILED"
    assert diagnostics.set_index("recording_id").loc["r1", "status"] == "PARTIAL_ALIGNMENT"
    freeze_alignment(root, list_alignment_runs(root)[0]["alignment_run_id"])
    store, issue = load_final_alignment(root)
    assert issue == "" and store is not None
    assert len(store.get_word_tokens("r1")) == 3
    assert store.get_phone_tokens("r1").empty


def test_freeze_rechecks_structural_domain_even_with_matching_run_hash(tmp_path):
    root, prompt = _reviewed_root(tmp_path)
    words = root / "words.csv"
    pd.DataFrame([{"recording_id": "r1", "word": "one", "start_sec": 1.2, "end_sec": 1.5},
                  {"recording_id": "r1", "word": "two", "start_sec": 4.6, "end_sec": 4.9},
                  {"recording_id": "r1", "word": "three", "start_sec": 5, "end_sec": 5.4}]).to_csv(words, index=False)
    run_acoustic_alignment(root, AlignmentConfig(prompt_manifest_path=str(prompt),
                                                  words_csv=str(words)))
    run_id = list_alignment_runs(root)[0]["alignment_run_id"]
    base = root / "acoustic" / "004_alignment" / "runs" / run_id
    manifest_path = base / "logs" / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table_path = base / "tables" / "alignment_words.csv"
    table = pd.read_csv(table_path)
    table.loc[2, "end_sec"] = 9.5
    table.to_csv(table_path, index=False)
    manifest["words_sha256"] = sha256(table_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="alignment_boundary_outside_reviewed_domain"):
        freeze_alignment(root, run_id)
