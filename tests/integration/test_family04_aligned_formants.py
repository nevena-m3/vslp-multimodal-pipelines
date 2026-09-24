"""Family 04 reads frozen Alignment and emits native token/vowel dimensions."""

from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.alignment import AlignmentConfig, freeze_alignment, list_alignment_runs, run_acoustic_alignment
from vslp.acoustic.features.family04 import FAMILY04_IDS, FRAME_COLUMNS, TOKEN_COLUMNS
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
    selected = sorted(FAMILY04_IDS)
    cfg = FeatureExtractionConfig(selected_features=selected,
                                  vowel_category_manifest_path=str(mapping))
    result = run_acoustic_feature_extraction(decisions, root, cfg, intervals)
    assert result.status == "completed"
    assert sha256(wav.read_bytes()).hexdigest() == original_hash
    values = pd.read_csv(result.summary_table)
    assert values.vsa3_iau_hz2.iloc[0] == 322500
    assert "f1_token_hz" not in values
    assert "f1_vowel_median_hz" not in values
    native = root / "acoustic" / "005_features" / "family_runs" / "formants" / "tables" / "native_measurements"
    tokens = pd.read_csv(native / "formant_token_measurements.csv")
    vowels = pd.read_csv(native / "formant_vowel_measurements.csv")
    assert len(tokens) == 3 and set(vowels.target_vowel) == {"i", "a", "u"}
    assert tokens.alignment_run_id.iloc[0]
    assert len(list((native / "formant_frame_tracks").glob("*.npz"))) == 1
    status = pd.read_csv(result.summary_table.parent / "acoustic_feature_status_long.csv")
    assert status.set_index("feature_id").loc["f1_token_hz", "status"] == "computed_native"


def test_family04_missing_frozen_alignment_fails_per_feature(tmp_path, monkeypatch):
    root, decisions, intervals, mapping, kept, _ = _case(tmp_path, freeze=False)
    monkeypatch.setattr("vslp.acoustic.features.family04_stage.load_final_segmentation",
                        lambda *_: kept)
    monkeypatch.setattr("vslp.acoustic.features.mixed_dispatch.load_final_segmentation",
                        lambda *_: kept)
    cfg = FeatureExtractionConfig(selected_features=["f1_token_hz", "vsa3_iau_hz2"],
                                  vowel_category_manifest_path=str(mapping))
    result = run_acoustic_feature_extraction(decisions, root, cfg, intervals)
    status = pd.read_csv(result.summary_table.parent / "acoustic_feature_status_long.csv")
    assert set(status.failure_reason) == {"missing_alignment"}
    assert np.isnan(pd.read_csv(result.summary_table).vsa3_iau_hz2.iloc[0])
