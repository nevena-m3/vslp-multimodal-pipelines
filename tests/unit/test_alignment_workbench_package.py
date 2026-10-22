"""Standalone Package V1 additions leave the embedded package path intact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.alignment.package_v1 import (
    import_alignment_package, inspect_alignment_package, load_imported_alignment,
    validate_alignment_package,
)
from vslp.acoustic.alignment.workbench import (
    create_job, inspect_audio_files, load_job, prepare_corpus, qc_job,
    realign_recording, set_alignment_scope,
)
from vslp.acoustic.alignment.workbench_package import export_standalone_package
from vslp.acoustic.alignment.workbench_review import save_corrections
from hashlib import sha256
from vslp.acoustic.alignment.mfa_provider import parse_textgrid
from vslp.acoustic.alignment.final_contract import canonical_units
from vslp.acoustic.alignment.workbench import canonical_task


def _textgrid(path: Path, labels: list[str]) -> None:
    rows = ['File type = "ooTextFile"', 'Object class = "TextGrid"',
            'item [1]:', '    name = "words"']
    for index, label in enumerate(labels, 1):
        start = index / 10
        rows.extend([f'    intervals [{index}]:', f'        xmin = {start}',
                     f'        xmax = {start + .09}', f'        text = "{label}"'])
    rows.extend(['item [2]:', '    name = "phones"'])
    for index in range(1, len(labels) + 1):
        start = index / 10
        rows.extend([f'    intervals [{index}]:', f'        xmin = {start}',
                     f'        xmax = {start + .09}', '        text = "B"'])
    path.write_text("\n".join(rows), encoding="utf-8")


def test_textgrid_silence_does_not_extend_lexical_word(tmp_path):
    grid = tmp_path / "silence.TextGrid"
    grid.write_text('\n'.join([
        'File type = "ooTextFile"', 'Object class = "TextGrid"',
        'item [1]:', 'name = "words"',
        'intervals [1]:', 'xmin = 0.0', 'xmax = 6.1', 'text = ""',
        'intervals [2]:', 'xmin = 6.1', 'xmax = 6.5', 'text = "bamboo"',
        'intervals [3]:', 'xmin = 6.5', 'xmax = 6.8', 'text = ""',
        'intervals [4]:', 'xmin = 6.8', 'xmax = 7.2', 'text = "walls"',
        'item [2]:', 'name = "phones"',
        'intervals [1]:', 'xmin = 6.1', 'xmax = 6.5', 'text = "B"',
        'intervals [2]:', 'xmin = 6.8', 'xmax = 7.2', 'text = "W"',
    ]), encoding="utf-8")
    words, _phones = parse_textgrid(grid)
    assert words.label.tolist() == ["bamboo", "walls"]
    assert words.start_sec.tolist() == [6.1, 6.8]
    assert words.end_sec.tolist() == [6.5, 7.2]


def test_bamboo_protocol_words_and_mfa_tokens_are_distinct():
    spec = canonical_task("bamboo_passage", 1)
    assert len(canonical_units(spec["canonical_transcript"])) == 97
    assert len(spec["alignment_transcript"].split()) == 98
    assert "GOOD LOOKING" in spec["alignment_transcript"]


def test_standalone_package_matches_audio_and_remaps_record_identity(tmp_path):
    audio = tmp_path / "s2.wav"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    job_root = create_job(tmp_path / "workspace", name="S2", audio_rows=inspect_audio_files([audio]),
                          task_id="sentence_s2", repetitions=1)
    manifest = load_job(job_root)
    record = manifest["recordings"][0]
    grid = job_root / "mfa_output" / "record_000001.TextGrid"
    _textgrid(grid, manifest["task_specification"]["normalized_transcript"].split())
    record["textgrid_path"] = str(grid)
    (job_root / "job_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert qc_job(job_root)["auto_pass"] == 1
    package = export_standalone_package(job_root, tmp_path / "package")

    project = tmp_path / "project"
    review = project / "acoustic" / "003_segmentation_review" / "final"
    review.mkdir(parents=True)
    pd.DataFrame([{"recording_id": "project_record_key", "analysis_wav_path": str(audio),
                   "source_file_path": str(audio), "final_decision": "KEEP_MANUAL"}]).to_csv(
                       review / "final_segmentation_decisions.csv", index=False)
    (project / "project_manifest.json").write_text(json.dumps({"task_id": "sentence_s2"}),
                                                    encoding="utf-8")
    result = validate_alignment_package(package, project)
    assert result["matched"] == 1
    assert result["record_key_map"] == {"record_000001": "project_record_key"}
    assert result["recording_status"][0]["identity_basis"] == "raw_source"
    import_alignment_package(package, project)
    store, issue = load_imported_alignment(project)
    assert not issue
    assert set(store.trials.recording_id) == {"project_record_key"}
    assert set(store.canonical_phones.recording_id) == {"project_record_key"}
    assert store.canonical_phones.parent_word_id.str.startswith(
        "project_record_key:").all()

    other = tmp_path / "different.wav"
    sf.write(other, np.ones(16000, dtype=np.float32), 16000)
    pd.DataFrame([{"recording_id": "project_record_key", "analysis_wav_path": str(other),
                   "source_file_path": str(other), "final_decision": "KEEP_MANUAL"}]).to_csv(
                       review / "final_segmentation_decisions.csv", index=False)
    with pytest.raises(ValueError, match="standalone_audio_identity_mismatch"):
        validate_alignment_package(package, project)
    rejected = inspect_alignment_package(package, project)
    assert rejected["error"].startswith("standalone_audio_identity_mismatch")
    assert (rejected["matched"], rejected["missing"], rejected["invalid"]) == (0, 0, 1)


def test_standalone_raw_source_matches_time_preserving_analysis_copy(tmp_path):
    source = tmp_path / "raw.wav"
    sf.write(source, np.linspace(-.4, .6, 16000, dtype=np.float32), 16000)
    root = create_job(tmp_path / "workspace", name="S2",
                      audio_rows=inspect_audio_files([source]),
                      task_id="sentence_s2", repetitions=1)
    job = load_job(root)
    grid = root / "mfa_output" / "record_000001.TextGrid"
    _textgrid(grid, job["task_specification"]["normalized_transcript"].split())
    job["recordings"][0]["textgrid_path"] = str(grid)
    (root / "job_manifest.json").write_text(json.dumps(job), encoding="utf-8")
    qc_job(root)
    package = export_standalone_package(root, tmp_path / "package")
    audio, rate = sf.read(source, dtype="float32")
    analysis = tmp_path / "analysis.wav"
    sf.write(analysis, audio - np.mean(audio), rate, subtype="FLOAT")
    project = tmp_path / "project"
    review = project / "acoustic" / "003_segmentation_review" / "final"
    review.mkdir(parents=True)
    source_hash = sha256(source.read_bytes()).hexdigest()
    pd.DataFrame([{"recording_id": "project_record_key", "source_file_path": str(source),
                   "source_sha256": source_hash, "analysis_wav_path": str(analysis),
                   "final_decision": "KEEP_MANUAL"}]).to_csv(
                       review / "final_segmentation_decisions.csv", index=False)
    preprocess = project / "acoustic" / "001_preprocess" / "tables"
    preprocess.mkdir(parents=True)
    pd.DataFrame([{"recording_id": "project_record_key", "status": "ok",
                   "source_sha256": source_hash, "decoded_sample_rate_hz": rate,
                   "analysis_sample_rate_hz": rate, "decoded_n_samples": len(audio),
                   "analysis_n_samples": len(audio)}]).to_csv(
                       preprocess / "acoustic_preprocess_summary.csv", index=False)
    (project / "project_manifest.json").write_text(json.dumps({"task_id": "sentence_s2"}),
                                                    encoding="utf-8")
    result = validate_alignment_package(package, project)
    assert result["record_key_map"] == {"record_000001": "project_record_key"}
    assert result["recording_status"][0]["identity_basis"] == "raw_source"
    sf.write(analysis, audio[:-1], rate, subtype="FLOAT")
    with pytest.raises(ValueError, match="standalone_time_axis_mismatch"):
        validate_alignment_package(package, project)


def test_scoped_working_audio_package_retains_raw_identity_and_original_time(tmp_path):
    source = tmp_path / "raw.wav"
    audio = np.zeros(5 * 16000, dtype=np.float32)
    audio[2 * 16000:3 * 16000] = .2
    sf.write(source, audio, 16000)
    root = create_job(tmp_path / "workspace", name="S2",
                      audio_rows=inspect_audio_files([source]),
                      task_id="sentence_s2", repetitions=1)
    set_alignment_scope(root, "record_000001", 2, 3, "reviewer")
    prepare_corpus(root)
    job = load_job(root)
    grid = root / "mfa_output" / "record_000001.TextGrid"
    _textgrid(grid, job["task_specification"]["normalized_transcript"].split())
    job["recordings"][0]["textgrid_path"] = str(grid)
    (root / "job_manifest.json").write_text(json.dumps(job), encoding="utf-8")
    assert qc_job(root)["auto_pass"] == 1
    package = export_standalone_package(root, tmp_path / "package")
    trials = pd.read_csv(package / "final_alignment_contract_trials.csv")
    assert trials.start_sec.iloc[0] == pytest.approx(2.1)
    status = pd.read_csv(package / "recording_status.csv")
    assert status.alignment_working_transform.iloc[0] == "sample_exact_utterance_crop_v1"
    assert status.alignment_scope_start_sec.iloc[0] == pytest.approx(2)
    project = tmp_path / "project"
    review = project / "acoustic" / "003_segmentation_review" / "final"
    review.mkdir(parents=True)
    pd.DataFrame([{"recording_id": "project_key", "source_file_path": str(source),
                   "analysis_wav_path": str(source), "final_decision": "KEEP_MANUAL"}]).to_csv(
                       review / "final_segmentation_decisions.csv", index=False)
    (project / "project_manifest.json").write_text(
        json.dumps({"task_id": "sentence_s2"}), encoding="utf-8")
    assert validate_alignment_package(package, project)["matched"] == 1


def test_partial_standalone_package_keeps_missing_record_separate(tmp_path):
    source = tmp_path / "first.wav"
    other = tmp_path / "second.wav"
    sf.write(source, np.zeros(16000, dtype=np.float32), 16000)
    sf.write(other, np.ones(16000, dtype=np.float32), 16000)
    root = create_job(tmp_path / "workspace", name="S2",
                      audio_rows=inspect_audio_files([source]),
                      task_id="sentence_s2", repetitions=1)
    job = load_job(root)
    grid = root / "mfa_output" / "record_000001.TextGrid"
    _textgrid(grid, job["task_specification"]["normalized_transcript"].split())
    job["recordings"][0]["textgrid_path"] = str(grid)
    (root / "job_manifest.json").write_text(json.dumps(job), encoding="utf-8")
    qc_job(root)
    package = export_standalone_package(root, tmp_path / "package")
    project = tmp_path / "project"
    review = project / "acoustic" / "003_segmentation_review" / "final"
    review.mkdir(parents=True)
    pd.DataFrame([
        {"recording_id": "first_project_key", "source_file_path": str(source),
         "analysis_wav_path": str(source), "final_decision": "KEEP_MANUAL"},
        {"recording_id": "second_project_key", "source_file_path": str(other),
         "analysis_wav_path": str(other), "final_decision": "KEEP_MANUAL"},
    ]).to_csv(review / "final_segmentation_decisions.csv", index=False)
    (project / "project_manifest.json").write_text(json.dumps({"task_id": "sentence_s2"}),
                                                    encoding="utf-8")
    result = validate_alignment_package(package, project)
    assert (result["matched"], result["missing"], result["stale"]) == (1, 1, 0)
    assert {row["record_key"]: row["status"] for row in result["recording_status"]} == {
        "first_project_key": "MATCHED + FINALIZED", "second_project_key": "MISSING"}


def test_old_word_removal_shortcut_cannot_fake_lexical_realignment(tmp_path):
    audio = tmp_path / "s2.wav"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    job_root = create_job(tmp_path / "workspace", name="S2", audio_rows=inspect_audio_files([audio]),
                          task_id="sentence_s2", repetitions=1)
    manifest = load_job(job_root)
    labels = manifest["task_specification"]["normalized_transcript"].split() + ["EXTRA"]
    grid = job_root / "mfa_output" / "record_000001.TextGrid"
    _textgrid(grid, labels)
    manifest["recordings"][0]["textgrid_path"] = str(grid)
    (job_root / "job_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert qc_job(job_root)["needs_review"] == 1
    save_corrections(job_root, "record_000001", {"removed_word_indices": [len(labels)],
                                                       "decision": "ACCEPT"},
                     reviewer="test_reviewer", reason="Synthetic extra token")
    assert qc_job(job_root)["needs_review"] == 1
    with pytest.raises(ValueError, match="no_structurally_accepted_recordings_to_export"):
        export_standalone_package(job_root, tmp_path / "package")


def test_unresolved_review_does_not_leave_partial_package(tmp_path):
    audio = tmp_path / "s2.wav"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    root = create_job(tmp_path / "workspace", name="S2",
                      audio_rows=inspect_audio_files([audio]),
                      task_id="sentence_s2", repetitions=1)
    job = load_job(root)
    record = job["recordings"][0]
    grid = root / "mfa_output" / "record_000001.TextGrid"
    labels = job["task_specification"]["normalized_transcript"].split()
    _textgrid(grid, labels)
    record["textgrid_path"] = str(grid)
    record["state"] = "REVIEWED"
    (root / "job_manifest.json").write_text(json.dumps(job), encoding="utf-8")
    destination = tmp_path / "package"
    with pytest.raises(ValueError, match="reviewed_alignment_not_available"):
        export_standalone_package(root, destination)
    assert not destination.exists()


def test_lexical_edit_runs_new_alignment_and_preserves_old_grid(tmp_path):
    source = tmp_path / "s2.wav"
    sf.write(source, np.zeros(16000, dtype=np.float32), 16000)
    root = create_job(tmp_path / "workspace", name="S2",
                      audio_rows=inspect_audio_files([source]),
                      task_id="sentence_s2", repetitions=1)
    job = load_job(root)
    old_grid = root / "mfa_output" / "record_000001.TextGrid"
    _textgrid(old_grid, job["task_specification"]["normalized_transcript"].split())
    job["recordings"][0]["textgrid_path"] = str(old_grid)
    (root / "job_manifest.json").write_text(json.dumps(job), encoding="utf-8")

    class Provider:
        def preflight(self, corpus, profile, transcripts, logs):
            assert transcripts == ["BUY BOBBY A PUPPY EXTRA"]
            assert next(corpus.rglob("*.lab")).read_text() == transcripts[0]
            return {"status": "AVAILABLE"}

        def configure_from_preflight(self, profile, report):
            assert report["status"] == "AVAILABLE"

        def align_corpus(self, corpus, output, profile, logs):
            output.mkdir(parents=True)
            grid = output / "record_000001.TextGrid"
            _textgrid(grid, ["BUY", "BOBBY", "A", "PUPPY", "EXTRA"])
            return {"record_000001": grid}

    result = realign_recording(root, "record_000001", "Buy Bobby a puppy extra",
                              "reviewer", provider=Provider())
    assert old_grid.is_file()
    updated = load_job(root)["recordings"][0]
    assert updated["textgrid_path"] != str(old_grid)
    assert result["needs_review"] == 1
    assert "spoken_transcript_differs_from_canonical" in updated["qc_flags"]
    assert updated["extra_aligned_tokens"] == ["EXTRA"]
