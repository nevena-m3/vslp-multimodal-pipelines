"""SIT alignment with per-recording transcripts, and the review sheet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import pandas as pd
import soundfile as sf

from vslp.acoustic.alignment import review_sheet, sit_align


class FakeProvider:
    """One 0.3 s interval per .lab token; a 'drift' speaker loses its last word."""

    def __init__(self, dictionary: Path):
        self.resolved_dictionary = str(dictionary)
        self.commands = []

    def inspect_environment(self, profile):
        return {"status": "AVAILABLE"}

    def preflight(self, corpus, profile, transcripts, logs):
        return {"status": "AVAILABLE"}

    def configure_from_preflight(self, profile, environment):
        return None

    def align_corpus(self, corpus, output, profile, logs):
        grids = {}
        for lab in Path(corpus).rglob("*.lab"):
            labels = lab.read_text(encoding="utf-8").split()
            if "DRIFT" in lab.parent.name:
                labels = labels[:-1]
            rows = ['item [1]:', 'name = "words"']
            for index, label in enumerate(labels, 1):
                rows += [f"intervals [{index}]:", f"xmin = {index * .3}",
                         f"xmax = {index * .3 + .25}", f'text = "{label}"']
            rows += ['item [2]:', 'name = "phones"']
            for index, label in enumerate(labels, 1):
                rows += [f"intervals [{index}]:", f"xmin = {index * .3}",
                         f"xmax = {index * .3 + .25}",
                         f'text = "{"spn" if label == "<unk>" else "AH0"}"']
            grid = Path(output) / lab.parent.name / f"{lab.stem}.TextGrid"
            grid.parent.mkdir(parents=True, exist_ok=True)
            grid.write_text("\n".join(rows), encoding="utf-8")
            grids[lab.stem] = grid
        return grids


def test_examiner_numbers_become_garbage_words_and_articles_stay():
    lab, problems = sit_align.mfa_transcript("5A. The boy ran home. 6A. A dog sat.")
    assert problems == []
    assert lab == "THE BOY RAN HOME <unk> A DOG SAT"
    lab, _ = sit_align.mfa_transcript("Number 7 the cat. Number 8 a hat.")
    assert lab == "THE CAT <unk> A HAT"
    dropped, _ = sit_align.mfa_transcript("5A. The boy. 6A. A dog.", examiner="drop")
    assert dropped == "THE BOY A DOG"


def test_spoken_preamble_and_spelled_labels_are_examiner_speech():
    lab, problems = sit_align.mfa_transcript(
        "It's March 5th, 2025, and this is CAPT0000138, 12 months. 5A. The boy ran. "
        "Six A. A dog sat. Elevenee. The cat ran.")
    assert problems == []
    assert lab == "<unk> THE BOY RAN <unk> A DOG SAT <unk> THE CAT RAN"
    assert sit_align.mfa_transcript("Cap 00060 5A. The boy.", examiner="drop")[0] == "THE BOY"



def test_preamble_without_item_label_and_ids_anywhere_are_examiner_speech():
    cases = {
        "The participant number is C-A-P-T-0-0-0-0-0-0-5 underscore 0-4-M. Capture ID "
        "is CAPT0000106. We bought a cake.": "<unk> WE BOUGHT A CAKE",
        "It's December 20th, 2024, and I'm here. The little dog ran.":
            "<unk> THE LITTLE DOG RAN",
        "Cap 000, 130, 12 months. She sold shells in the 1990s.":
            "<unk> SHE SOLD SHELLS IN THE NINETEEN NINETIES",
        "Today's the 23rd of March. CAPTCHA0000106 underscore 08M. The little dog ran.":
            "<unk> THE LITTLE DOG RAN",
    }
    for text, expected in cases.items():
        lab, problems = sit_align.mfa_transcript(text)
        assert problems == [] and lab == expected, (text, lab, problems)


def test_numbers_inside_sentences_are_spelled_out():
    assert sit_align.mfa_transcript("5A. The workout lasted 2 hours.")[0] == (
        "THE WORKOUT LASTED TWO HOURS")
    assert sit_align.spell_number("1990s") == "nineteen nineties"
    assert sit_align.spell_number("2025") == "twenty twenty five"
    assert sit_align.spell_number("21st") == "twenty first"
    assert sit_align.spell_number("3rd") == "third"
    assert sit_align.spell_number("40") == "forty"
    assert sit_align.spell_number("1500") == "fifteen hundred"
    assert sit_align.spell_number("007") is None
    # A decimal stays a number problem, never guessed.
    assert sit_align.mfa_transcript("5A. It cost 2.50 today.")[1] == [
        "transcript_contains_number_requires_review"]


def test_unusable_transcripts_are_held_not_guessed():
    assert sit_align.mfa_transcript("5A.")[1] == ["empty_transcript"]
    assert sit_align.mfa_transcript("He paid 2.5 percent")[1] == [
        "transcript_contains_number_requires_review"]


def test_transcript_text_reads_common_whisper_layouts():
    assert sit_align.transcript_text({"whole_text": "a b"}) == "a b"
    assert sit_align.transcript_text({"chunks": [{"text": " a"}, {"text": "b "}]}) == "a b"
    assert sit_align.transcript_text({"segments": [{"text": "a"}]}) == "a"
    assert sit_align.transcript_text({"words": [{"word": "a"}, {"word": "b"}]}) == "a b"


def _setup(tmp_path: Path) -> dict:
    audio, json_dir = tmp_path / "audio", tmp_path / "json"
    audio.mkdir()
    json_dir.mkdir()
    texts = {"P01_272_1_20240101_1059_SIT": "1A. The boy ran home. 2A. A dog sat.",
             "DRIFT_272_1_20240101_1059_SIT": "1A. The boy ran home.",
             "P03_272_1_20240101_1059_SIT": "1A. The zorblax ran.",
             "P04_272_1_20240101_1059_SIT": "examiner talking only"}
    for stem, text in texts.items():
        sf.write(audio / f"{stem}.wav", np.zeros(16000 * 5, dtype="float32"), 16000)
        (json_dir / f"{stem}.json").write_text(json.dumps({"whole_text": text}), "utf-8")
    pd.DataFrame({"file": [f"{stem}.wav" for stem in texts],
                  "looks_like_reading": ["True", "True", "True", "False"],
                  "avg_logprob": ["-0.2", "-0.2", "-0.9", "-0.2"]}).to_csv(
        tmp_path / "sit_screen.csv", index=False)
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("\n".join(f"{w} AH0" for w in
                                    ["the", "boy", "ran", "home", "a", "dog", "sat"]), "utf-8")
    return {"screen_csv": tmp_path / "sit_screen.csv", "json_dir": json_dir,
            "audio_dir": audio, "provider": FakeProvider(dictionary)}


def test_sit_run_aligns_own_transcripts_and_flags_problems(tmp_path):
    inputs = _setup(tmp_path)
    out = tmp_path / "out"
    table = sit_align.run(**inputs, out=out, plots="none", log=lambda *_: None)
    by_name = table.set_index("file_name")
    assert len(table) == 3  # the non-reading recording is not aligned
    good = by_name.loc["P01_272_1_20240101_1059_SIT.wav"]
    assert good.state == "AUTO_PASS" and good.aligned_words == 7
    assert "transcript_token_mismatch" in by_name.loc[
        "DRIFT_272_1_20240101_1059_SIT.wav"].qc_flags
    held = by_name.loc["P03_272_1_20240101_1059_SIT.wav"]
    assert held.state == "NEEDS_REVIEW" and "words_not_in_dictionary:zorblax" in held.qc_flags
    words = pd.read_csv(out / "sit_words.csv")
    assert "<unk>" not in set(words.label)  # examiner numbers never reach the tables
    assert (out / "textgrids" / "P01_272_1_20240101_1059_SIT.TextGrid").is_file()


def test_review_sheet_groups_flags_by_reason(tmp_path):
    inputs = _setup(tmp_path)
    out = tmp_path / "out"
    sit_align.run(**inputs, out=out, plots="flagged", log=lambda *_: None)
    batch = pd.DataFrame({"job": ["240_PSG_BAMBOO__bamboo_passage__x1"] * 2,
                          "file_name": ["P01_1_1_20240101_240_PSG_BAMBOO.wav",
                                        "P02_1_1_20240101_240_PSG_BAMBOO.wav"],
                          "speaker_id": ["P01", "P02"], "state": ["AUTO_PASS", "NEEDS_REVIEW"],
                          "qc_flags": ["", "leading_silence_too_long"]})
    batch.to_csv(tmp_path / "batch_recordings.csv", index=False)
    review = review_sheet.collect([tmp_path / "batch_recordings.csv",
                                   out / "sit_recordings.csv"], [out / "plots"])
    assert len(review) == 3
    assert set(review.reason) == {"leading_silence_too_long", "transcript_token_mismatch",
                                  "words_not_in_dictionary"}
    assert set(review.task) == {"240_PSG_BAMBOO", "1059_SIT"}
    assert review.loc[review.task.eq("1059_SIT"), "plot"].str.endswith(".png").all()
    pytest.importorskip("openpyxl")
    target = review_sheet.write(review, tmp_path / "review_sheet.xlsx")
    sheets = pd.read_excel(target, sheet_name=None)
    assert set(sheets) == {"Summary", "Review"}
    assert int(sheets["Summary"].total.sum()) == 3
