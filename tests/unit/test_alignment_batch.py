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

    workspace = tmp_path / "ws"
    result = batch.run_batch(inventory_csv=str(tmp_path / "WAVS.csv"), workspace=workspace,
                             zip_path=str(archive), provider=FakeProvider(), plot="flagged",
                             log=lambda *_: None)
    job = result["jobs"]["256_NSM_NAS_WSTG__wstg"]
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
    assert list((workspace / "plots" / "256_NSM_NAS_WSTG__wstg").glob("*.png"))

    again = batch.run_batch(inventory_csv=str(tmp_path / "WAVS.csv"), workspace=workspace,
                            zip_path=str(archive), provider=FakeProvider(), plot="none",
                            log=lambda *_: None)
    assert again["jobs"]["256_NSM_NAS_WSTG__wstg"]["finished_utc"] == job["finished_utc"]
    state = json.loads((workspace / "batch_state.json").read_text(encoding="utf-8"))
    assert list(state["jobs"]) == ["256_NSM_NAS_WSTG__wstg"]


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
