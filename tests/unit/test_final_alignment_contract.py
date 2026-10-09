"""Nonclinical final-token fixtures; no MFA provider is instantiated."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.alignment.final_contract import (
    FINAL_ALIGNMENT_CONTRACT_VERSION, build_feature_tables, canonical_units,
)
from vslp.acoustic.alignment.task_workflow import task_registry
from vslp.acoustic.alignment.stage import normalize_transcript


def _prompt(task_id: str) -> dict:
    return next(item for item in task_registry()["tasks"] if item["task_id"] == task_id)


def _fixture(task_id: str, count: int = 1, corrected: bool = False):
    item = _prompt(task_id)
    prompt = item["prompts"][0]
    units = canonical_units(prompt["exact_expected_text"])
    trial_rows, link_rows, word_rows, phone_rows, reviews = [], [], [], [], []
    token_index = 0
    for number in range(1, count + 1):
        trial_id = f"trial_{number:03d}"
        trial_rows.append({"recording_id": "synthetic", "trial_id": trial_id,
                           "prompt_id": prompt["prompt_id"], "start_sec": number * 10.,
                           "end_sec": number * 10. + 8.})
        reviews.append({"recording_id": "synthetic", "trial_id": trial_id,
                        "alignment_review_status": "HUMAN_ACCEPTED"})
        for unit in units:
            for part in normalize_transcript(unit).split():
                token_index += 1
                start = number * 10. + token_index * .001
                source = ("reviewed_manual_correction" if corrected and token_index == 1
                          else "mfa")
                word_rows.append({"recording_id": "synthetic", "word_index": token_index,
                                  "word": part, "start_sec": start, "end_sec": start + .0008,
                                  "alignment_source": source})
                phone_rows.append({"recording_id": "synthetic", "word_index": token_index,
                                   "phone_index": token_index, "phone_raw": "UW1",
                                   "phone_normalized": "UW", "stress": "1", "is_vowel": True,
                                   "start_sec": start, "end_sec": start + .0008,
                                   "alignment_source": source})
                link_rows.extend({"recording_id": "synthetic", "trial_id": trial_id,
                                  "token_type": kind, "token_index": token_index}
                                 for kind in ("word", "phone"))
    return (pd.DataFrame(trial_rows), pd.DataFrame(link_rows),
            pd.DataFrame(word_rows), pd.DataFrame(phone_rows),
            pd.DataFrame(reviews), {"synthetic": {**prompt, "task_id": task_id}})


@pytest.mark.parametrize("task_id,count", [("wstg", 3), ("sentence_s3", 1),
                                           ("sentence_s5", 1), ("bamboo_passage", 1)])
def test_nonclinical_feature_tables_preserve_trial_and_canonical_words(task_id, count):
    tables = build_feature_tables(*_fixture(task_id, count), "synthetic_run")
    trials, words, phones = tables
    assert len(trials) == count
    assert words.trial_id.nunique() == count
    assert phones.trial_id.nunique() == count
    assert words.alignment_contract_version.eq(FINAL_ALIGNMENT_CONTRACT_VERSION).all()
    assert phones.sequence_id.nunique() == count
    assert set(phones.parent_word_id) <= set(words.word_id)
    assert phones.phone_normalized.eq("UW").all()
    assert phones.stress.astype(str).eq("1").all()
    assert phones.is_vowel.all()
    if task_id == "bamboo_passage":
        assert len(words) == 97
        assert len(json.loads(words.loc[
            words.canonical_word_text.eq("good-looking"), "provider_token_ids"].iloc[0])) == 2
        assert words.provider_token_count.sum() == 98
    if task_id == "sentence_s3":
        assert {"blue", "spot", "key"} <= set(words.canonical_word_text)


def test_manual_boundary_is_feature_facing_and_raw_remains_separate():
    fixture = _fixture("wstg", corrected=True)
    _, words, phones = build_feature_tables(*fixture, "synthetic_run")
    assert words.iloc[0].boundary_source == "REVIEWED_MANUAL_CORRECTION"
    assert phones.iloc[0].boundary_source == "REVIEWED_MANUAL_CORRECTION"
    assert fixture[2].iloc[0].alignment_source == "reviewed_manual_correction"


def test_unmappable_provider_tokens_require_review():
    fixture = list(_fixture("wstg"))
    fixture[2].loc[0, "word"] = "UNEXPECTED"
    with pytest.raises(ValueError, match="mapping_requires_review"):
        build_feature_tables(*fixture, "synthetic_run")


def test_bamboo_evidence_uses_final_canonical_labels_and_creates_detail(tmp_path):
    from vslp.acoustic.alignment.evidence import create_final_evidence

    trials, words, phones = build_feature_tables(
        *_fixture("bamboo_passage"), "synthetic_run")
    root = tmp_path / "nonclinical"
    review = root / "acoustic" / "003_segmentation_review" / "final"
    review.mkdir(parents=True)
    wav = root / "synthetic.wav"
    samples = np.arange(20 * 16000) / 16000
    sf.write(wav, .02 * np.sin(2 * np.pi * 190 * samples), 16000)
    pd.DataFrame([{"recording_id": "synthetic", "analysis_wav_path": str(wav)}]).to_csv(
        review / "final_segmentation_decisions.csv", index=False)
    pd.DataFrame([{"recording_id": "synthetic", "segment_role": "speech",
                   "start_sec": 10., "end_sec": 18.}]).to_csv(
        review / "final_segmentation_intervals.csv", index=False)
    destination = root / "acoustic" / "004_alignment" / "final"
    index = create_final_evidence(root, destination, trials, words, phones,
                                  {"provider": "synthetic_fixture"})
    assert len(index) == 1
    assert (destination / index.iloc[0].overview_plot_path).is_file()
    assert (destination / index.iloc[0].detail_plot_path).is_file()


@pytest.mark.parametrize("task_id,repetitions", [("wstg", 3), ("sentence_s3", 1),
                                                ("sentence_s5", 1),
                                                ("bamboo_passage", 1)])
def test_feature_development_package_loads_without_mfa(tmp_path, task_id, repetitions):
    from tests.fixtures.final_alignment_contract_v1 import write_nonclinical_final_alignment
    from vslp.acoustic.alignment import load_final_alignment

    root = write_nonclinical_final_alignment(tmp_path / task_id, task_id,
                                             repetitions, corrected_phone=True)
    store, issue = load_final_alignment(root)
    assert issue == "" and store is not None
    assert store.trials.trial_id.nunique() == repetitions
    assert store.canonical_phones.iloc[0].boundary_source == "REVIEWED_MANUAL_CORRECTION"
    if task_id == "bamboo_passage":
        assert len(store.canonical_words) == 97


def test_f06_target_localization_uses_final_canonical_phone(tmp_path):
    from tests.fixtures.final_alignment_contract_v1 import write_nonclinical_final_alignment
    from vslp.acoustic.alignment import load_final_alignment
    from vslp.acoustic.features.family06_stage import _aligned_target

    root = write_nonclinical_final_alignment(tmp_path / "s3", "sentence_s3")
    store, issue = load_final_alignment(root)
    assert issue == "" and store is not None
    phone = _aligned_target(store, "nonclinical", {
        "word_index": 2, "word": "blue", "phone_index": 1, "phone": "UW"})
    assert phone.parent_word_text == "blue"
    assert phone.trial_id == "trial_001"
    assert phone.sequence_id == "nonclinical:trial_001"
