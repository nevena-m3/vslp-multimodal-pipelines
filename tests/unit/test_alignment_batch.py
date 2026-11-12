"""One-time batch aligner: inventory, zip extraction, per-task packages."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.alignment import batch
from vslp.acoustic.alignment.package_v1 import (import_alignment_package,
                                                load_imported_alignment)


class FakeProvider:
    """Writes one word/phone interval per expected token; 'bad' files get none."""

    commands: list = []

    def preflight(self, corpus, profile, transcripts, logs):
        return {"status": "AVAILABLE", "version": "fake", "oov_words": []}

    def configure_from_preflight(self, profile, environment):
        return None

    def align_corpus(self, corpus, output, profile, logs):
        grids = {}
        for lab in Path(corpus).rglob("*.lab"):
            labels = lab.read_text(encoding="utf-8").split()
            if "bad" in lab.parent.name:
                labels = labels[:-1]  # token mismatch -> NEEDS_REVIEW
            rows = ['File type = "ooTextFile"', 'Object class = "TextGrid"',
                    'item [1]:', 'name = "words"']
            for index, label in enumerate(labels, 1):
                rows += [f'intervals [{index}]:', f'xmin = {index / 10}',
                         f'xmax = {index / 10 + .09}', f'text = "{label}"']
            rows += ['item [2]:', 'name = "phones"']
            for index, _ in enumerate(labels, 1):
                rows += [f'intervals [{index}]:', f'xmin = {index / 10}',
                         f'xmax = {index / 10 + .09}', 'text = "B"']
            grid = Path(output) / f"{lab.stem}.TextGrid"
            grid.parent.mkdir(parents=True, exist_ok=True)
            grid.write_text("\n".join(rows), encoding="utf-8")
            grids[lab.stem] = grid
        return grids


def _wav(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    audio = np.zeros(4 * 16000, dtype="float32")
    audio[8000:32000] = rng.standard_normal(24000) * .2  # "speech" 0.5-2.0 s
    sf.write(path, audio, 16000)


def test_file_names_map_to_batch_roles():
    names = ["P01_1_1_20240101_240_PSG_BAMBOO.wav", "P01_1_1_20240101_242_DDK_PA.wav",
             "P01_1_1_20240101_258_SENT_CAPEV.wav", "P01_1_1_20240101_999_OTHER.wav",
             "notes.txt"]
    table = batch.build_inventory(None, names)
    assert table.batch_role.tolist() == ["fixed_prompt", "not_aligned", "identify_first",
                                         "not_in_batch"]
    assert table.task_id.iloc[0] == "bamboo_passage"
    assert table.subject_id.iloc[0] == "P01"


def test_batch_from_zip_writes_importable_package_and_skips_on_rerun(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    names = ["GOOD1_1_1_20240101_256_NSM_NAS_WSTG.wav", "GOOD2_1_1_20240102_256_NSM_NAS_WSTG.wav",
             "bad_1_1_20240101_256_NSM_NAS_WSTG.wav", "GOOD1_1_1_20240101_242_DDK_PA.wav"]
    for seed, name in enumerate(names):
        _wav(source / name, seed)
    archive = tmp_path / "wavs.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for name in names:
            handle.write(source / name, f"WAVs/{name}")
    pd.DataFrame({"Raw Media File name": names,
                  "SubjectID": ["GOOD1", "GOOD2", "bad", "GOOD1"]}).to_csv(
        tmp_path / "WAVS.csv", index=False)
    readings = tmp_path / "readings.csv"
    pd.DataFrame({"file_name": names[:3], "readings": [3, 3, 3]}).to_csv(readings, index=False)

    workspace = tmp_path / "ws"
    result = batch.run_batch(inventory_csv=str(tmp_path / "WAVS.csv"), workspace=workspace,
                             zip_path=str(archive), provider=FakeProvider(), plot="flagged",
                             readings_csv=str(readings), log=lambda *_: None)
    job = result["jobs"]["256_NSM_NAS_WSTG__wstg__x3"]
    assert job["status"] == "DONE"
    assert (job["recordings"], job["auto_pass"], job["needs_review"], job["exported"]) == (3, 2, 1, 2)
    assert sorted(path.name for path in (workspace / "audio").iterdir()) == sorted(names[:3])
    # An Acoustic project holding one of the batch recordings (any copy of it)
    # imports the whole-task package; only its own recording comes in.
    project = tmp_path / "project"
    review = project / "acoustic" / "003_segmentation_review" / "final"
    review.mkdir(parents=True)
    pd.DataFrame([{"recording_id": "p_good2", "analysis_wav_path": str(source / names[1]),
                   "source_file_path": str(source / names[1]),
                   "final_decision": "KEEP_MANUAL"}]).to_csv(
        review / "final_segmentation_decisions.csv", index=False)
    (project / "project_manifest.json").write_text(json.dumps({"task_id": "wstg"}),
                                                    encoding="utf-8")
    import_alignment_package(job["package"], project)
    store, issue = load_imported_alignment(project)
    assert not issue
    assert set(store.words.recording_id.astype(str)) == {"p_good2"}
    assert len(store.trials) == 3  # three repetitions of "We see three geese."
    table = pd.read_csv(workspace / "batch_recordings.csv")
    assert table.set_index("speaker_id").loc["bad", "state"] == "NEEDS_REVIEW"
    assert list((workspace / "plots" / "256_NSM_NAS_WSTG__wstg__x3").glob("*.png"))

    limited = batch.run_batch(inventory_csv=str(tmp_path / "WAVS.csv"), workspace=tmp_path / "ws_pilot",
                              audio_dir=str(source), provider=FakeProvider(), plot="none",
                              limit_per_task=1, readings_csv=str(readings),
                              log=lambda *_: None)
    assert limited["jobs"]["256_NSM_NAS_WSTG__wstg__x3"]["recordings"] == 1

    again = batch.run_batch(inventory_csv=str(tmp_path / "WAVS.csv"), workspace=workspace,
                            zip_path=str(archive), provider=FakeProvider(), plot="none",
                            readings_csv=str(readings), log=lambda *_: None)
    assert again["jobs"]["256_NSM_NAS_WSTG__wstg__x3"]["finished_utc"] == job["finished_utc"]
    state = json.loads((workspace / "batch_state.json").read_text(encoding="utf-8"))
    assert list(state["jobs"]) == ["256_NSM_NAS_WSTG__wstg__x3"]


def test_batch_profile_widens_the_mfa_beam_only():
    from vslp.acoustic.alignment.mfa_provider import MfaProfile, MfaProvider

    wide = MfaProfile.load(batch.BATCH_PROFILE)
    base = MfaProfile.load(batch.BATCH_PROFILE.with_name(
        "mfa_vslp_en_us_arpa_334_engineering_v1.json"))
    assert (wide.acoustic_model, wide.dictionary) == (base.acoustic_model, base.dictionary)
    provider = MfaProvider.__new__(MfaProvider)
    provider._mfa = lambda *args: ["mfa", *args]
    command = provider.alignment_command(Path("c"), Path("o"), wide)
    assert command[-4:] == ["--beam", "100", "--retry_beam", "400"]
    assert "--beam" not in provider.alignment_command(Path("c"), Path("o"), base)


def test_reading_counts_follow_the_speech_not_the_protocol():
    from vslp.acoustic.alignment.readings import count_readings

    assert len(count_readings("How hard did he hit me? How hard did he hit him? "
                              "How hard did he hit him?", "How hard did he hit him?")) == 3
    assert len(count_readings("buy bobby a puppy " * 8, "Buy Bobby a puppy.")) == 8
    assert len(count_readings("Okay. We see three geese. We see three geese, "
                              "we see three geese. Done?", "We see three geese.")) == 3
    assert count_readings("Nothing appeared on the screen. Should I read this?",
                          "We see three geese.") == []
    # Pilot 2 cases: dysarthric ASR slips count, a false start and "thank you" don't.
    assert len(count_readings("Bye, Bubby and Puppy", "Buy Bobby a puppy.")) == 1
    assert count_readings("Thank you very much.", "Buy Bobby a puppy.") == []
    assert len(count_readings("did he hit him? He's not a drip boy | How hard did he hit him? | "
                              "How hard did he hit him? | how hard did he hit him",
                              "How hard did he hit him?")) == 3


def test_unknown_or_zero_reading_counts_are_held_back_not_guessed(tmp_path):
    names = ["A_1_1_20240101_256_NSM_NAS_WSTG.wav", "B_1_1_20240101_256_NSM_NAS_WSTG.wav",
             "C_1_1_20240101_256_NSM_NAS_WSTG.wav", "A_1_1_20240101_240_PSG_BAMBOO.wav"]
    fixed = batch.build_inventory(None, names)
    readings = tmp_path / "r.csv"
    pd.DataFrame({"file_name": names[:2], "readings": [5, 0]}).to_csv(readings, index=False)
    kept, held = batch.apply_reading_counts(fixed, readings)
    assert dict(zip(kept.file_name, kept.repetitions)) == {names[0]: 5, names[3]: 1}
    assert dict(zip(held.file_name, held.reason)) == {
        names[1]: "no_reading_of_prompt_detected", names[2]: "reading_count_missing"}


def test_pilot_sample_spreads_over_protocols():
    names = [f"S{i:02d}_{p}_1_20240101_240_PSG_BAMBOO.wav"
             for p in (1075, 272) for i in range(10 if p == 1075 else 3)]
    sample = batch.pilot_sample(batch.build_inventory(None, names), 4)
    assert len(sample) == 4
    assert set(sample.protocol) == {"1075", "272"}


def test_one_reading_spread_over_several_is_flagged(tmp_path):
    """Pilot case: 'how' in reading 1, 'hard did' in reading 2, 'he hit him' in 3."""
    from vslp.acoustic.alignment.workbench import (create_job, inspect_audio_files, load_job,
                                                   qc_job)

    audio = tmp_path / "hit.wav"
    sf.write(audio, np.full(8 * 16000, .1, dtype="float32"), 16000)
    job_root = create_job(tmp_path / "ws", name="hit", audio_rows=inspect_audio_files([audio]),
                          task_id="sentence_capev_hit", repetitions=1)
    job = load_job(job_root)
    starts = [1.2, 3.2, 3.5, 5.6, 5.8, 6.0]
    labels = job["task_specification"]["alignment_transcript"].split()
    rows = ['File type = "ooTextFile"', 'Object class = "TextGrid"', 'item [1]:',
            'name = "words"']
    for index, (start, label) in enumerate(zip(starts, labels), 1):
        rows += [f'intervals [{index}]:', f'xmin = {start}', f'xmax = {start + .15}',
                 f'text = "{label}"']
    rows += ['item [2]:', 'name = "phones"']
    for index, start in enumerate(starts, 1):
        rows += [f'intervals [{index}]:', f'xmin = {start}', f'xmax = {start + .15}',
                 'text = "B"']
    grid = job_root / "mfa_output" / "record_000001.TextGrid"
    grid.write_text("\n".join(rows), encoding="utf-8")
    job["recordings"][0]["textgrid_path"] = str(grid)
    (job_root / "job_manifest.json").write_text(json.dumps(job), encoding="utf-8")
    flags = qc_job(job_root)["records"][0]["flags"]
    assert "long_pause_within_reading_requires_review" in flags


def test_readings_from_chunks_never_overcount():
    from vslp.acoustic.alignment.readings import estimate_readings

    prompt = "Buy Bobby a puppy."
    one = lambda start, length, text: {"start": start, "end": start + length, "text": text}  # noqa: E731
    # Pilot 3, CNEC044: four readings with no pause between them became one long
    # chunk that Whisper wrote once.
    merged = [one(0.5, 1.2, "Buy Bobby a puppy"), one(2.2, 1.1, "Buy Bobby a puppy"),
              one(4.0, 4.6, "Buy Bobby a puppy")]
    # Duration never adds readings (pilot 4: it overcounted clean files);
    # the undercount goes to review through structural QC instead.
    assert estimate_readings(merged, prompt)[0] == 3
    clean = [one(0.5, 1.2, "How hard did he hit him?"), one(2.4, 2.1, "How hard did he hit him?"),
             one(5.0, 1.3, "How hard did he hit him?")]
    assert estimate_readings(clean, "How hard did he hit him?")[0] == 3
    # A reading split across two chunks still counts once.
    split = [one(0.5, .6, "My mama"), one(1.3, .9, "makes lemon jam"),
             one(3.0, 1.5, "My mama makes lemon jam"), one(5.0, 1.5, "My mama makes lemon jam")]
    assert estimate_readings(split, "My mama makes lemon jam")[0] == 3


def test_unrecognised_speech_falls_back_to_protocol_count_not_dropped(tmp_path):
    names = ["A_1_1_20240101_256_NSM_NAS_WSTG.wav", "B_1_1_20240101_256_NSM_NAS_WSTG.wav"]
    fixed = batch.build_inventory(None, names)
    readings = tmp_path / "r.csv"
    pd.DataFrame({"file_name": names, "readings": [0, 0],
                  "speech_sec": [6.0, 0.3]}).to_csv(readings, index=False)
    kept, held = batch.apply_reading_counts(fixed, readings)
    assert kept.file_name.tolist() == [names[0]] and kept.repetitions.tolist() == [3]
    assert held.file_name.tolist() == [names[1]]


def test_extra_reading_squeezed_into_a_pause_is_flagged(tmp_path):
    """Pilot 4, CNEC023: 3 readings aligned as 4; the 4th got ~0.08 s words in a pause."""
    from vslp.acoustic.alignment.workbench import (create_job, inspect_audio_files, load_job,
                                                   qc_job)

    audio = tmp_path / "hit.wav"
    sf.write(audio, np.full(9 * 16000, .1, dtype="float32"), 16000)
    job_root = create_job(tmp_path / "ws", name="hit", audio_rows=inspect_audio_files([audio]),
                          task_id="sentence_capev_hit", repetitions=4)
    job = load_job(job_root)
    labels = job["task_specification"]["alignment_transcript"].split()
    times = []
    for start, step in [(0.5, .25), (2.0, .25), (3.2, .08), (4.2, .25)]:
        times += [(start + k * step, start + k * step + step * .9) for k in range(6)]
    rows = ['File type = "ooTextFile"', 'Object class = "TextGrid"', 'item [1]:',
            'name = "words"']
    for index, ((a, b), label) in enumerate(zip(times, labels), 1):
        rows += [f'intervals [{index}]:', f'xmin = {a}', f'xmax = {b}', f'text = "{label}"']
    rows += ['item [2]:', 'name = "phones"']
    for index, (a, b) in enumerate(times, 1):
        rows += [f'intervals [{index}]:', f'xmin = {a}', f'xmax = {b}', 'text = "B"']
    grid = job_root / "mfa_output" / "record_000001.TextGrid"
    grid.write_text("\n".join(rows), encoding="utf-8")
    job["recordings"][0]["textgrid_path"] = str(grid)
    (job_root / "job_manifest.json").write_text(json.dumps(job), encoding="utf-8")
    record = qc_job(job_root)["records"][0]
    assert "reading_compressed_requires_review" in record["flags"]


def test_failed_mfa_validation_lists_recordings_for_review(tmp_path):
    class FailingPreflight(FakeProvider):
        def preflight(self, corpus, profile, transcripts, logs):
            return {"status": "FAILED", "version": "fake", "oov_words": []}

    audio = tmp_path / "audio"
    audio.mkdir()
    name = "CNES01_1078_1_20240101_1048_BBP_SHORT.wav"
    _wav(audio / name, 1)
    readings = tmp_path / "r.csv"
    pd.DataFrame({"file_name": [name], "readings": [3]}).to_csv(readings, index=False)
    batch.run_batch(inventory_csv=None, workspace=tmp_path / "ws", audio_dir=str(audio),
                    readings_csv=str(readings), provider=FailingPreflight(), plot="none",
                    log=lambda *_: None)
    table = pd.read_csv(tmp_path / "ws" / "batch_recordings.csv")
    assert table.state.tolist() == ["NEEDS_REVIEW"]
    assert table.qc_flags.str.startswith("alignment_not_completed").all()


def test_reading_counts_are_recomputed_from_the_cache_without_speech_to_text(tmp_path, monkeypatch):
    from vslp.acoustic.alignment import readings

    audio = tmp_path / "audio"
    audio.mkdir()
    name = "CNES01_1075_1_20240101_1064_CAPEV_2.wav"
    _wav(audio / name, 1)
    out = tmp_path / "readings.csv"
    chunk = {"start": 0.5, "end": 1.9, "text": "How hard did he hit him?"}
    Path(str(out) + ".chunks.jsonl").write_text(json.dumps(
        {"file_name": name, "chunks": [chunk, {**chunk, "start": 2.5, "end": 3.9}],
         "whole_text": ""}) + "\n", encoding="utf-8")

    def no_whisper(*args, **kwargs):
        raise AssertionError("speech-to-text must not run for cached files")

    monkeypatch.setattr(readings, "transcribe", no_whisper)
    readings.main(["--audio-dir", str(audio), "--out", str(out)])
    table = pd.read_csv(out)
    assert table.readings.tolist() == [2]
