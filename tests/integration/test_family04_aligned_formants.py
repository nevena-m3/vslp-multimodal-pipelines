"""Family 04 reads frozen Alignment and emits native token/vowel dimensions."""

from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.alignment import AlignmentConfig, freeze_alignment, list_alignment_runs, run_acoustic_alignment
from vslp.acoustic.features.family04 import CORNER_IDS, FRAME_COLUMNS, GENERIC_IDS, TOKEN_COLUMNS
from vslp.acoustic.features.stage import FeatureExtractionConfig, run_acoustic_feature_extraction


def _case(tmp_path, *, freeze: bool):
    root = tmp_path / "run"
    root.mkdir()
    (root / "project_manifest.json").write_text(json.dumps({
        "project_name": "P", "task_name": "Bamboo Passage", "task_id": "bamboo_passage",
        "task_type": "Passage / connected speech", "run_id": "run1"}), encoding="utf-8")
    wav = root / "canonical.wav"
    sf.write(wav, np.zeros(2 * 16000, dtype="float32"), 16000, subtype="FLOAT")
    final = root / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    pd.DataFrame([{"recording_id": "r1", "file_name": "reading.wav",
                   "final_decision": "KEEP_AUTO", "analysis_wav_path": str(wav),
                   "analysis_start_sec": 0, "analysis_end_sec": 2,
                   "segmentation_run_id": "seg1", "review_run_id": "rev1"}]).to_csv(decisions, index=False)
    pd.DataFrame([{"recording_id": "r1", "view": "authoritative",
                   "segment_role": "speech", "start_sec": 0, "end_sec": 2}]).to_csv(intervals, index=False)
    prompt = root / "prompt.json"
    prompt.write_text(json.dumps({"manifest_version": "1", "task_id": "bamboo_passage",
                                  "prompt_version": "v1", "language": "en",
                                  "transcript": "i a u", "expected_words": ["i", "a", "u"],
                                  "phone_set": "ARPABET_CMU_39"}), encoding="utf-8")
    mapping = root / "vowels.json"
    mapping.write_text(json.dumps({"manifest_version": "1", "task_id": "bamboo_passage",
                                   "phone_set": "ARPABET_CMU_39",
                                   "categories": [
                                       {"canonical_vowel": "i", "accepted_phone_labels": ["IY"]},
                                       {"canonical_vowel": "a", "accepted_phone_labels": ["AA"]},
                                       {"canonical_vowel": "u", "accepted_phone_labels": ["UW"]}]}),
                       encoding="utf-8")
    if freeze:
        words = root / "words.csv"
        phones = root / "phones.csv"
        pd.DataFrame([{"recording_id": "r1", "word": word,
                       "start_sec": start, "end_sec": start + .3}
                      for word, start in (("i", .2), ("a", .7), ("u", 1.2))]).to_csv(words, index=False)
        pd.DataFrame([{"recording_id": "r1", "phone": phone, "word_index": index,
                       "start_sec": start, "end_sec": start + .2}
                      for index, (phone, start) in enumerate(
                          (("IY1", .25), ("AA1", .75), ("UW1", 1.25)), 1)]).to_csv(phones, index=False)
        run_acoustic_alignment(root, AlignmentConfig(
            prompt_manifest_path=str(prompt), words_csv=str(words), phones_csv=str(phones)))
        freeze_alignment(root, list_alignment_runs(root)[0]["alignment_run_id"])
        # The old external-only fixture has no trial manifest. Supply a
        # deterministic, nonclinical V1 canonical view for feature testing.
        from vslp.acoustic.alignment.final_contract import (
            FINAL_ALIGNMENT_CONTRACT_VERSION, build_feature_tables)

        frozen = root / "acoustic" / "004_alignment" / "final"
        manifest_path = frozen / "final_alignment_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provider_words = pd.read_csv(frozen / "final_alignment_words.csv")
        provider_phones = pd.read_csv(frozen / "final_alignment_phones.csv")
        links = pd.DataFrame([
            {"recording_id": "r1", "trial_id": "trial_001", "token_type": kind,
             "token_index": index}
            for kind, indices in (("word", provider_words.word_index),
                                  ("phone", provider_phones.phone_index))
            for index in indices])
        contract = build_feature_tables(
            pd.DataFrame([{"recording_id": "r1", "trial_id": "trial_001",
                           "prompt_id": "synthetic_iau", "start_sec": 0., "end_sec": 2.}]),
            links, provider_words, provider_phones,
            pd.DataFrame([{"recording_id": "r1", "trial_id": "trial_001",
                           "alignment_review_status": "HUMAN_ACCEPTED"}]),
            {"r1": {"task_id": "bamboo_passage", "prompt_version": "v1",
                    "exact_expected_text": "i a u"}}, manifest["alignment_run_id"])
        names = ("final_alignment_contract_trials.csv",
                 "final_alignment_canonical_words.csv",
                 "final_alignment_canonical_phones.csv")
        for name, table in zip(names, contract):
            table.to_csv(frozen / name, index=False)
        manifest["final_alignment_contract_version"] = FINAL_ALIGNMENT_CONTRACT_VERSION
        manifest["final_alignment_contract_artifact_sha256"] = {
            name: sha256((frozen / name).read_bytes()).hexdigest() for name in names}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    kept = pd.DataFrame([{"recording_id": "r1", "file_name": "reading.wav",
                          "analysis_wav_path": str(wav), "source_sha256": "a" * 64,
                          "source_file_path": str(wav), "segmentation_run_id": "seg1",
                          "review_run_id": "rev1"}])
    return root, decisions, intervals, mapping, kept, wav


def test_family04_end_to_end_native_outputs_and_provenance(tmp_path, monkeypatch):
    root, decisions, intervals, mapping, kept, wav = _case(tmp_path, freeze=True)
    original_hash = sha256(wav.read_bytes()).hexdigest()
    monkeypatch.setattr("vslp.acoustic.features.family04_stage.load_final_segmentation",
                        lambda *_: kept)
    monkeypatch.setattr("vslp.acoustic.features.mixed_dispatch.load_final_segmentation",
                        lambda *_: kept)

    def frozen_tracker(audio, rate, phones, profile, **kwargs):
        assert rate == 16000
        assert kwargs["category_mapping"] == {"IY": "i", "AA": "a", "UW": "u"}
        rows = []
        frames = []
        for index, (phone, category, f1, f2) in enumerate(
                (("IY", "i", 300, 2300), ("AA", "a", 800, 1200), ("UW", "u", 350, 900)), 1):
            token = phones.iloc[index - 1]
            rows.append({"recording_id": "r1", "file_name": "reading.wav",
                         "task_id": "bamboo_passage", "alignment_run_id": kwargs["alignment_run_id"],
                         "vowel_category": category, "phone_normalized": phone,
                         "phone_raw": token.phone_raw, "word_index": index, "phone_index": index,
                         "token_start_sec": token.start_sec, "token_end_sec": token.end_sec,
                         "token_duration_sec": .2, "measurement_start_sec": token.start_sec + .05,
                         "measurement_end_sec": token.end_sec - .05,
                         "f1_token_hz": f1, "f2_token_hz": f2, "f3_token_hz": 2900,
                         "f1_iqr_hz": 0, "f2_iqr_hz": 0, "f3_iqr_hz": 0,
                         "n_valid_frames": 5, "tracking_status": "OK",
                         "formant_profile_id": profile.profile_id})
            frames.append({"token_index": index, "time_sec": token.start_sec + .1,
                           "f1_hz": f1, "f2_hz": f2, "f3_hz": 2900,
                           "valid_ordered": True, "in_middle_50_percent": True})
        return pd.DataFrame(rows).reindex(columns=TOKEN_COLUMNS), pd.DataFrame(frames).reindex(columns=FRAME_COLUMNS)

    monkeypatch.setattr("vslp.acoustic.features.family04_stage.track_aligned_vowels", frozen_tracker)
    selected = sorted(CORNER_IDS | GENERIC_IDS)
    cfg = FeatureExtractionConfig(selected_features=selected,
                                  vowel_category_manifest_path=str(mapping))
    result = run_acoustic_feature_extraction(decisions, root, cfg, intervals)
    assert result.status == "completed"
    assert sha256(wav.read_bytes()).hexdigest() == original_hash
    values = pd.read_csv(result.summary_table)
    assert values.vsa3_iau_hz2.iloc[0] == 322500
    assert values.f1_median_hz.iloc[0] == 350
    assert "f1_token_hz" not in values
    assert "f1_vowel_median_hz" not in values
    native = root / "acoustic" / "005_features" / "family_runs" / "formants" / "tables" / "native_measurements"
    tokens = pd.read_csv(native / "formant_token_measurements.csv")
    vowels = pd.read_csv(native / "formant_vowel_measurements.csv")
    assert len(tokens) == 3 and set(vowels.target_vowel) == {"i", "a", "u"}
    assert tokens.alignment_run_id.iloc[0]
    assert len(list((native / "formant_frame_tracks").glob("*.npz"))) == 1
    status = pd.read_csv(result.summary_table.parent / "acoustic_feature_status_long.csv")
    assert status.set_index("feature_id").loc["f1_median_hz", "status"] == "computed"


def test_family04_missing_frozen_alignment_fails_per_feature(tmp_path, monkeypatch):
    root, decisions, intervals, mapping, kept, _ = _case(tmp_path, freeze=False)
    monkeypatch.setattr("vslp.acoustic.features.family04_stage.load_final_segmentation",
                        lambda *_: kept)
    monkeypatch.setattr("vslp.acoustic.features.mixed_dispatch.load_final_segmentation",
                        lambda *_: kept)
    cfg = FeatureExtractionConfig(selected_features=["f1_median_hz", "vsa3_iau_hz2"],
                                  vowel_category_manifest_path=str(mapping))
    result = run_acoustic_feature_extraction(decisions, root, cfg, intervals)
    status = pd.read_csv(result.summary_table.parent / "acoustic_feature_status_long.csv")
    assert set(status.failure_reason) == {"missing_alignment"}
    assert np.isnan(pd.read_csv(result.summary_table).vsa3_iau_hz2.iloc[0])


def test_family04_trial_first_uses_median_of_trial_values(tmp_path, monkeypatch):
    root, decisions, intervals, _mapping, kept, _wav = _case(tmp_path, freeze=True)
    monkeypatch.setattr("vslp.acoustic.features.family04_stage.load_final_segmentation",
                        lambda *_: kept)
    monkeypatch.setattr("vslp.acoustic.features.mixed_dispatch.load_final_segmentation",
                        lambda *_: kept)

    def tracker(_audio, _rate, _phones, profile, **_kwargs):
        rows = []
        for trial, shift in (("trial_001", 0), ("trial_002", 100)):
            for vowel, f1, f2 in (("a", 800, 1200), ("i", 300, 2300),
                                  ("u", 350, 900)):
                rows.append({"recording_id": "r1", "file_name": "reading.wav",
                             "task_id": "bamboo_passage", "trial_id": trial,
                             "alignment_run_id": "frozen", "vowel_category": vowel,
                             "phone_normalized": {"a": "AA", "i": "IY", "u": "UW"}[vowel],
                             "word": vowel, "tracking_status": "OK", "f1_token_hz": f1,
                             "f2_token_hz": f2 + shift, "f3_token_hz": 2900,
                             "n_valid_frames": 5, "formant_profile_id": profile.profile_id})
        return pd.DataFrame(rows).reindex(columns=TOKEN_COLUMNS), pd.DataFrame(columns=FRAME_COLUMNS)

    monkeypatch.setattr("vslp.acoustic.features.family04_stage.track_aligned_vowels", tracker)
    config = FeatureExtractionConfig(selected_features=["f2_median_hz", "vsa3_iau_hz2"])
    result = run_acoustic_feature_extraction(decisions, root, config, intervals)
    assert result.status == "completed"
    values = pd.read_csv(result.summary_table)
    assert values.f2_median_hz.iloc[0] == 1250  # median of trial medians 1200 and 1300
    trial_path = (root / "acoustic" / "005_features" / "family_runs" / "formants" /
                  "tables" / "native_measurements" / "formant_trial_summaries.csv")
    trials = pd.read_csv(trial_path)
    assert set(trials.trial_id) == {"trial_001", "trial_002"}
    assert len(trials) == 4


def test_sustained_a_f2_midpoint_runs_without_linguistic_alignment(tmp_path, monkeypatch):
    root, decisions, intervals, _mapping, kept, _wav = _case(tmp_path, freeze=False)
    monkeypatch.setattr("vslp.acoustic.features.family04_stage.load_final_alignment",
                        lambda *_: (_ for _ in ()).throw(AssertionError(
                            "sustained_vowel_loaded_linguistic_alignment")))
    project = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    project.update(task_id="sustained_a", task_name="Sustained /a/",
                   task_type="Sustained phonation", task_item="a")
    (root / "project_manifest.json").write_text(json.dumps(project), encoding="utf-8")
    monkeypatch.setattr("vslp.acoustic.features.family04_stage.load_final_segmentation",
                        lambda *_: kept)
    monkeypatch.setattr("vslp.acoustic.features.mixed_dispatch.load_final_segmentation",
                        lambda *_: kept)
    monkeypatch.setattr("vslp.acoustic.features.family04_stage.centered_stable_spans",
                        lambda *_args: [(.2, 1.2, "t1", .2, 1.2),
                                        (1.3, 1.9, "t2", 1.3, 1.9)])

    def midpoint(_audio, _rate, regions, _profile, **_kwargs):
        assert regions == [("t1", .2, 1.2), ("t2", 1.3, 1.9)]
        return pd.DataFrame([{"trial_id": trial, "tracking_status": "OK",
                              "vowel_category": "a", "f1_token_hz": 500.,
                              "f2_token_hz": f2, "f3_token_hz": 3000.,
                              "n_valid_frames": 1} for trial, f2 in (("t1", 1000.), ("t2", 1200.))])

    monkeypatch.setattr("vslp.acoustic.features.family04_stage.track_sustained_midpoints", midpoint)
    cfg = FeatureExtractionConfig(selected_features=["f2_median_hz"])
    result = run_acoustic_feature_extraction(decisions, root, cfg, intervals)
    assert result.status == "completed"
    assert pd.read_csv(result.summary_table).f2_median_hz.iloc[0] == 1100.


def test_s3_lexical_pairs_and_envelope_dispatch(tmp_path, monkeypatch):
    root, decisions, intervals, _mapping, kept, _wav = _case(tmp_path, freeze=False)
    project = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    project.update(task_id="sentence_s3", task_name="S3",
                   task_type="Sentence / controlled speech", task_item="S3")
    (root / "project_manifest.json").write_text(json.dumps(project), encoding="utf-8")
    monkeypatch.setattr("vslp.acoustic.features.family04_stage.load_final_segmentation",
                        lambda *_: kept)
    monkeypatch.setattr("vslp.acoustic.features.mixed_dispatch.load_final_segmentation",
                        lambda *_: kept)
    words = root / "s3_words.csv"
    phones = root / "s3_phones.csv"
    words.write_text("word\nblue\nspot\nkey\n", encoding="utf-8")
    phones.write_text("phone\nUW\nAA\nIY\n", encoding="utf-8")

    class Alignment:
        manifest = {"alignment_run_id": "s3_final", "provider": "reviewed",
                    "final_words_sha256": "a" * 64, "final_phones_sha256": "b" * 64,
                    "final_words_path": str(words), "final_phones_path": str(phones)}
        trials = pd.DataFrame([{"recording_id": "r1", "trial_id": "trial_001",
                                "start_sec": 0., "end_sec": 2.}])

        @staticmethod
        def get_vowel_tokens(_recording_id):
            return pd.DataFrame([{"word_index": i, "phone_index": i,
                                  "trial_id": "trial_001", "parent_word_text": word,
                                  "phone_raw": phone + "1", "phone_normalized": phone,
                                  "start_sec": start, "end_sec": start + .2}
                                 for i, (word, phone, start) in enumerate(
                                     (("blue", "UW", .1), ("spot", "AA", .5),
                                      ("key", "IY", .9)), 1)])

        @staticmethod
        def get_word_tokens(_recording_id):
            return pd.DataFrame([{"word_index": i, "word": word}
                                 for i, word in enumerate(("blue", "spot", "key"), 1)])

    monkeypatch.setattr("vslp.acoustic.features.family04_stage.load_final_alignment",
                        lambda *_: (Alignment(), ""))

    def tracker(_audio, _rate, tokens, profile, **kwargs):
        assert tokens.word.tolist() == ["blue", "spot", "key"]
        assert kwargs["task_item"] == "S3"
        rows = [{"recording_id": "r1", "trial_id": "trial_001", "task_id": "sentence_s3",
                 "word": word, "phone_normalized": phone, "tracking_status": "OK",
                 "f1_token_hz": f1, "f2_token_hz": f2, "f3_token_hz": 3000.,
                 "token_start_sec": start, "token_end_sec": start + .2,
                 "n_valid_frames": 5, "formant_profile_id": profile.profile_id}
                for word, phone, f1, f2, start in (("blue", "UW", 350, 900, .1),
                                                   ("spot", "AA", 800, 1200, .5),
                                                   ("key", "IY", 300, 2300, .9))]
        return pd.DataFrame(rows).reindex(columns=TOKEN_COLUMNS), pd.DataFrame(columns=FRAME_COLUMNS)

    monkeypatch.setattr("vslp.acoustic.features.family04_stage.track_aligned_vowels", tracker)
    monkeypatch.setattr("vslp.acoustic.features.family04_envelope.allpole_token_envelope",
                        lambda _audio, _rate, start, _end: np.full(256, start))
    selected = ["vowel_distance_ai_hz", "vowel_distance_au_hz",
                "vowel_distance_iu_hz", "f2_distance_i_a_hz",
                "vowel_envelope_distance_ai_source"]
    result = run_acoustic_feature_extraction(
        decisions, root, FeatureExtractionConfig(selected_features=selected), intervals)
    assert result.status == "completed"
    values = pd.read_csv(result.summary_table)
    assert values.f2_distance_i_a_hz.iloc[0] == 1100
    assert values.vowel_envelope_distance_ai_source.iloc[0] == pytest.approx(.4)
