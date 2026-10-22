"""Focused tests for standalone task-first Alignment jobs."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from threading import Thread
from time import sleep

import numpy as np
import pytest
import soundfile as sf

from vslp.acoustic.alignment.workbench import (
    canonical_task, create_job, discover_audio, inspect_audio_files,
    load_job, local_writable_workspace, prepare_corpus, qc_job, set_alignment_scope,
)
from vslp.acoustic.alignment.mfa_provider import parse_textgrid
from vslp.acoustic.alignment.workbench_review import (
    load_corrections, reviewed_tokens, save_corrections,
)
from vslp.acoustic.alignment.cancellable_runner import (
    AlignmentCancelled, CancellableMfaRunner,
)


def _audio(path: Path) -> None:
    sf.write(path, np.zeros(1600, dtype=np.float32), 16000)


def test_recursive_header_only_ingest_and_duplicates(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "a.wav"
    second = nested / "b.wav"
    _audio(first)
    _audio(second)
    assert discover_audio(tmp_path) == [first, second]
    rows = inspect_audio_files([first, second, first, tmp_path / "wrong.mp3"])
    assert [row["status"] for row in rows] == [
        "READY", "READY", "DUPLICATE", "UNSUPPORTED FORMAT"]
    assert rows[0]["sample_rate_hz"] == 16000
    assert rows[0]["duration_sec"] == pytest.approx(.1)


def test_job_persists_canonical_repetitions_and_one_corpus(tmp_path):
    audio = tmp_path / "s2.wav"
    _audio(audio)
    workspace = tmp_path / "workspace"
    job = create_job(workspace, name="S2 batch", audio_rows=inspect_audio_files([audio]),
                     task_id="sentence_s2", repetitions=3)
    manifest = load_job(job)
    spec = manifest["task_specification"]
    assert spec["expected_repetitions"] == 3
    assert spec["alignment_transcript"].split() == spec["normalized_transcript"].split() * 3
    assert manifest["recordings"][0]["source_path"] == str(audio)
    assert prepare_corpus(job)["status"] == "CORPUS_READY"
    corpus_audio = next((job / "corpus").rglob("*.wav"))
    lab = next((job / "corpus").rglob("*.lab"))
    assert corpus_audio.read_bytes() == audio.read_bytes()
    assert lab.read_text(encoding="utf-8") == spec["alignment_transcript"]
    assert json.loads((job / "job_manifest.json").read_text())["state"] == "CORPUS_READY"


def test_task_and_workspace_guards(tmp_path):
    with pytest.raises(ValueError, match="bamboo_repetition_contract_mismatch"):
        canonical_task("bamboo_passage", 2)
    desktop = tmp_path / "Desktop" / "VSLP"
    assert local_writable_workspace(desktop) == desktop
    with pytest.raises(ValueError, match="choose_an_absolute_workspace_path"):
        local_writable_workspace(Path("relative-workspace"))


def test_implausibly_long_mfa_word_is_flagged_for_review(tmp_path):
    audio = tmp_path / "s2.wav"
    _audio(audio)
    root = create_job(tmp_path / "workspace", name="S2",
                      audio_rows=inspect_audio_files([audio]),
                      task_id="sentence_s2", repetitions=1)
    job = load_job(root)
    grid = root / "mfa_output" / "record_000001.TextGrid"
    grid.write_text('''File type = "ooTextFile"
Object class = "TextGrid"
item [1]:
name = "words"
intervals [1]:
xmin = 0.0
xmax = 2.5
text = "BUY"
item [2]:
name = "phones"
intervals [1]:
xmin = 0.0
xmax = 2.5
text = "B"''', encoding="utf-8")
    job["recordings"][0]["textgrid_path"] = str(grid)
    job["recordings"][0]["duration_sec"] = 3.0
    (root / "job_manifest.json").write_text(json.dumps(job), encoding="utf-8")
    result = qc_job(root)
    assert result["needs_review"] == 1
    assert "long_aligned_word_span_requires_review" in result["records"][0]["flags"]


def test_structural_qc_flags_transcript_mismatch_without_finalizing(tmp_path):
    audio = tmp_path / "s2.wav"
    _audio(audio)
    job_root = create_job(tmp_path / "workspace", name="S2", audio_rows=inspect_audio_files([audio]),
                          task_id="sentence_s2", repetitions=1)
    grid = job_root / "mfa_output" / "record_000001.TextGrid"
    grid.write_text('''File type = "ooTextFile"
Object class = "TextGrid"
item [1]:
    name = "words"
    intervals [1]:
        xmin = 0.01
        xmax = 0.04
        text = "WRONG"
item [2]:
    name = "phones"
    intervals [1]:
        xmin = 0.01
        xmax = 0.04
        text = "W"
''', encoding="utf-8")
    manifest = load_job(job_root)
    manifest["recordings"][0]["textgrid_path"] = str(grid)
    (job_root / "job_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = qc_job(job_root)
    assert result["needs_review"] == 1
    assert "transcript_token_mismatch" in result["records"][0]["flags"]
    assert load_job(job_root)["state"] == "NEEDS_REVIEW"


def test_external_runner_cancels_without_killing_caller():
    runner = CancellableMfaRunner()
    outcome = []

    def run() -> None:
        try:
            runner([sys.executable, "-c", "import time; time.sleep(10)"], timeout=20)
        except (AlignmentCancelled, subprocess.TimeoutExpired) as exc:
            outcome.append(type(exc))

    thread = Thread(target=run)
    thread.start()
    sleep(.3)
    runner.cancel()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert outcome == [AlignmentCancelled]


def test_scoped_mfa_working_audio_maps_local_words_to_original_time(tmp_path):
    rate = 16000
    audio = np.zeros(20 * rate, dtype=np.float32)
    audio[5 * rate:15 * rate] = .2 * np.sin(
        2 * np.pi * 220 * np.arange(10 * rate) / rate)
    source = tmp_path / "speech.wav"
    sf.write(source, audio, rate)
    root = create_job(tmp_path / "workspace", name="Scoped", audio_rows=inspect_audio_files([source]),
                      task_id="sentence_s2", repetitions=1)
    scope = set_alignment_scope(root, "record_000001", 5, 15, "reviewer")
    assert scope["source"] == "manual"
    assert prepare_corpus(root)["status"] == "CORPUS_READY"
    row = load_job(root)["recordings"][0]
    assert sf.info(row["corpus_audio"]).duration == pytest.approx(10)
    assert sf.info(source).duration == pytest.approx(20)
    assert row["alignment_working_transform"] == "sample_exact_utterance_crop_v1"
    grid = root / "mfa_output" / "record_000001.TextGrid"
    grid.write_text('''item [1]:
name = "words"
intervals [1]:
xmin = 0.20
xmax = 0.60
text = "BUY"
item [2]:
name = "phones"
intervals [1]:
xmin = 0.20
xmax = 0.40
text = "B"''', encoding="utf-8")
    local_words, _ = parse_textgrid(grid)
    assert local_words.start_sec.iloc[0] == pytest.approx(.20)
    job = load_job(root)
    job["recordings"][0]["textgrid_path"] = str(grid)
    (root / "job_manifest.json").write_text(json.dumps(job), encoding="utf-8")
    words, phones = reviewed_tokens(root, "record_000001")
    assert words.start_sec.iloc[0] == pytest.approx(5.20)
    assert phones.start_sec.iloc[0] == pytest.approx(5.20)
    save_corrections(root, "record_000001", {"word_bounds": {"1": [5.3, 5.5]},
                      "decision": "ACCEPT"}, "reviewer", "old alignment")
    set_alignment_scope(root, "record_000001", 5.1, 15, "reviewer")
    assert load_job(root)["recordings"][0]["textgrid_path"] == ""
    assert load_corrections(root, "record_000001")["word_bounds"] == {}
    assert load_corrections(root, "record_000001")["decision"] == ""


def test_alignment_scope_proposal_ignores_long_leading_silence(tmp_path):
    from vslp.acoustic.alignment.utterance_scope import propose_scope

    rate = 16000
    audio = np.zeros(20 * rate, dtype=np.float32)
    audio[6 * rate:18 * rate] = .2 * np.sin(
        2 * np.pi * 220 * np.arange(12 * rate) / rate)
    source = tmp_path / "long_lead.wav"
    sf.write(source, audio, rate)
    scope = propose_scope(source)
    assert 5.7 < scope["start_sec"] < 6.1
    assert 17.9 < scope["end_sec"] < 18.4
    assert scope["source"] == "alignment_energy_proposal_v1"
