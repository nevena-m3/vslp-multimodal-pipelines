"""A single selected-output run preserves independent family results and audits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.features.family04 import FRAME_COLUMNS, TOKEN_COLUMNS
from vslp.acoustic.features.stage import FeatureExtractionConfig, run_acoustic_feature_extraction


def _case(tmp_path, monkeypatch, *, aligned=False, normalized=False):
    root = tmp_path / "run"
    root.mkdir()
    (root / "project_manifest.json").write_text(json.dumps({
        "project_name": "P", "task_name": "Bamboo Passage", "task_id": "bamboo_passage",
        "task_type": "Passage / connected speech", "run_id": "run1"}), encoding="utf-8")
    sr = 16000
    t = np.arange(2 * sr) / sr
    wav = root / "canonical.wav"
    sf.write(wav, (.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, subtype="FLOAT")
    final = root / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    decisions.write_text("recording_id,final_decision\nr1,KEEP_AUTO\n", encoding="utf-8")
    pd.DataFrame([
        {"recording_id": "r1", "view": "authoritative", "segment_role": "speech",
         "start_sec": .2, "end_sec": .8, "boundary_source": "AUTO",
         "analysis_start_sec": 0, "analysis_end_sec": 2},
        {"recording_id": "r1", "view": "authoritative", "segment_role": "internal_nonspeech",
         "start_sec": .8, "end_sec": 1.2, "boundary_source": "AUTO",
         "analysis_start_sec": 0, "analysis_end_sec": 2},
        {"recording_id": "r1", "view": "authoritative", "segment_role": "speech",
         "start_sec": 1.2, "end_sec": 1.8, "boundary_source": "AUTO",
         "analysis_start_sec": 0, "analysis_end_sec": 2},
    ]).to_csv(intervals, index=False)
    kept = pd.DataFrame([{
        "recording_id": "r1", "file_name": "Bamboo.wav", "duration_sec": 2.,
        "analysis_wav_path": str(wav), "source_sha256": "a" * 64,
        "source_file_path": str(wav), "segmentation_run_id": "seg1",
        "review_run_id": "rev1", "boundary_source": "AUTO",
        "segmentation_method": "silero_vad", "analysis_start_sec": 0.,
        "analysis_end_sec": 2., "stable_region_start": .2, "stable_region_end": 1.8,
    }])
    for module in ("mixed_dispatch", "family01_stage", "family04_stage", "family06_stage", "family08",
                   "family10", "family11_12_stage"):
        monkeypatch.setattr(f"vslp.acoustic.features.{module}.load_final_segmentation",
                            lambda *_: kept)
    if normalized:
        prep = root / "acoustic" / "001_preprocess" / "tables"
        prep.mkdir(parents=True)
        pd.DataFrame([{"recording_id": "r1", "amplitude_normalization": True,
                       "remove_dc_offset": False}]).to_csv(
                           prep / "acoustic_preprocess_summary.csv", index=False)
    mapping = root / "vowels.json"
    mapping.write_text(json.dumps({"manifest_version": "1", "task_id": "bamboo_passage",
        "phone_set": "ARPABET_CMU_39", "categories": [
            {"canonical_vowel": "i", "accepted_phone_labels": ["IY"]}]}), encoding="utf-8")
    if aligned:
        words = root / "words.csv"
        phones = root / "phones.csv"
        words.write_text("word\ntest\n", encoding="utf-8")
        phones.write_text("phone\nIY\n", encoding="utf-8")

        class Alignment:
            manifest = {"alignment_run_id": "align1", "provider": "validated_external",
                        "final_words_sha256": "b" * 64, "final_phones_sha256": "c" * 64,
                        "final_words_path": str(words), "final_phones_path": str(phones)}

            @staticmethod
            def get_vowel_tokens(_recording_id):
                return pd.DataFrame([{"start_sec": .3, "end_sec": .5,
                                      "phone_raw": "IY1", "phone_normalized": "IY",
                                      "word_index": 1, "phone_index": 1}])

        monkeypatch.setattr("vslp.acoustic.features.family04_stage.load_final_alignment",
                            lambda *_: (Alignment(), ""))

        def tracker(_audio, _sr, _tokens, profile, **kwargs):
            token = pd.DataFrame([{"recording_id": "r1", "file_name": "Bamboo.wav",
                "task_id": "bamboo_passage", "alignment_run_id": "align1",
                "vowel_category": "i", "tracking_status": "OK", "n_valid_frames": 5,
                "f1_token_hz": 300., "f2_token_hz": 2200., "f3_token_hz": 3000.,
                "formant_profile_id": profile.profile_id}]).reindex(columns=TOKEN_COLUMNS)
            frame = pd.DataFrame([{"token_index": 1, "time_sec": .4, "f1_hz": 300.,
                "f2_hz": 2200., "f3_hz": 3000., "valid_ordered": True,
                "in_middle_50_percent": True}]).reindex(columns=FRAME_COLUMNS)
            return token, frame

        monkeypatch.setattr("vslp.acoustic.features.family04_stage.track_aligned_vowels", tracker)
    return root, decisions, intervals, mapping


def _run(root, decisions, intervals, ids, mapping=None):
    cfg = FeatureExtractionConfig(selected_features=ids,
                                  vowel_category_manifest_path=str(mapping) if mapping else None)
    result = run_acoustic_feature_extraction(decisions, root, cfg, intervals)
    statuses = pd.read_csv(result.summary_table.parent / "acoustic_feature_status_long.csv",
                           keep_default_na=False)
    values = pd.read_csv(result.summary_table)
    return result, statuses, values


def test_voice_and_pause_succeed_in_one_run(tmp_path, monkeypatch):
    root, decisions, intervals, _ = _case(tmp_path, monkeypatch)
    _, status, values = _run(root, decisions, intervals, ["f0_mean_hz", "pause_count"])
    assert set(status.feature_id) == {"f0_mean_hz", "pause_count"}
    assert status.failure_reason.eq("").all()
    assert values.f0_mean_hz.iloc[0] > 0
    assert values.pause_count.iloc[0] == 1


@pytest.mark.parametrize("aligned", [True, False])
def test_formants_and_pause_isolate_alignment(tmp_path, monkeypatch, aligned):
    root, decisions, intervals, mapping = _case(tmp_path, monkeypatch, aligned=aligned)
    _, status, values = _run(root, decisions, intervals,
                             ["f1_vowel_median_hz", "pause_count"], mapping)
    assert values.pause_count.iloc[0] == 1
    formant = status.loc[status.feature_id.eq("f1_vowel_median_hz")].iloc[0]
    assert formant.status == ("computed_native" if aligned else "unavailable")
    assert formant.failure_reason == ("" if aligned else "missing_alignment")
    if aligned:
        audit = (root / "acoustic" / "005_features" / "family_runs" / "formants" /
                 "tables" / "native_measurements" / "formant_token_measurements.csv")
        assert audit.is_file()


def test_prompt_failure_only_affects_rate_and_spectrum_succeeds(tmp_path, monkeypatch):
    root, decisions, intervals, _ = _case(tmp_path, monkeypatch)
    _, status, values = _run(root, decisions, intervals,
                             ["speaking_rate_syll_s", "mfcc01_mean"])
    assert status.loc[status.feature_id.eq("speaking_rate_syll_s"),
                      "failure_reason"].iloc[0]
    assert status.loc[status.feature_id.eq("mfcc01_mean"), "failure_reason"].iloc[0] == ""
    assert np.isfinite(values.mfcc01_mean.iloc[0])


def test_normalization_failure_only_affects_absolute_energy(tmp_path, monkeypatch):
    root, decisions, intervals, _ = _case(tmp_path, monkeypatch, normalized=True)
    _, status, values = _run(root, decisions, intervals,
                             ["absolute_energy_fs2", "mfcc01_mean"])
    assert status.loc[status.feature_id.eq("absolute_energy_fs2"),
                      "failure_reason"].iloc[0] == "incompatible_amplitude_normalization"
    assert np.isfinite(values.mfcc01_mean.iloc[0])


def test_four_families_selected_only_unique_and_audited(tmp_path, monkeypatch):
    root, decisions, intervals, mapping = _case(tmp_path, monkeypatch, aligned=True)
    ids = ["f0_mean_hz", "f1_token_hz", "pause_count", "mfcc01_mean"]
    result, status, values = _run(root, decisions, intervals, ids, mapping)
    assert set(status.feature_id) == set(ids)
    assert not status.duplicated(["recording_id", "feature_id"]).any()
    assert set(values.columns) >= {"f0_mean_hz", "pause_count", "mfcc01_mean"}
    assert "f1_token_hz" not in values
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert set(manifest["config"]["families_invoked"]) == {"F01", "F04", "F09", "F12"}
    assert manifest["config"]["failed_outputs"] == 0
    assert (root / "acoustic" / "005_features" / "family_runs" / "voice" /
            "tables" / "native_measurements" / "f0_tracks.csv").is_file()
    assert (root / "acoustic" / "005_features" / "family_runs" / "spectrum" /
            "tables" / "native_measurements" / "mfcc_matrices").is_dir()


def test_executor_failure_is_isolated(tmp_path, monkeypatch):
    root, decisions, intervals, _ = _case(tmp_path, monkeypatch)

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic_executor_failure")

    import vslp.acoustic.features.family01_stage as voice_stage

    monkeypatch.setattr(voice_stage, "run_family01_stage", fail)
    result, status, values = _run(root, decisions, intervals,
                                  ["f0_mean_hz", "pause_count"])
    assert result.status == "completed_with_warnings"
    assert status.loc[status.feature_id.eq("f0_mean_hz"),
                      "failure_reason"].iloc[0].startswith("family_executor_failed")
    assert values.pause_count.iloc[0] == 1


def test_ddk_prerequisite_does_not_block_pause(tmp_path, monkeypatch):
    root, decisions, intervals, _ = _case(tmp_path, monkeypatch)
    result, status, values = _run(root, decisions, intervals,
                                  ["ddk_rate_syll_s", "pause_count"])
    assert result.status == "completed_with_warnings"
    assert values.pause_count.iloc[0] == 1
    assert status.loc[status.feature_id.eq("ddk_rate_syll_s"),
                      "failure_reason"].iloc[0] == "missing_final_ddk_segmentation"


def test_duplicate_selection_rejected(tmp_path, monkeypatch):
    root, decisions, intervals, _ = _case(tmp_path, monkeypatch)
    cfg = FeatureExtractionConfig(selected_features=["pause_count", "pause_count"])
    with pytest.raises(ValueError, match="Duplicate selected feature ID"):
        run_acoustic_feature_extraction(decisions, root, cfg, intervals)


def test_segmental_missing_target_does_not_block_three_other_families(tmp_path, monkeypatch):
    root, decisions, intervals, _ = _case(tmp_path, monkeypatch)
    result, status, values = _run(
        root, decisions, intervals,
        ["m1_t_minus_k_hz", "pause_count", "mfcc01_mean", "f0_mean_hz"])
    segmental = status.loc[status.feature_id.eq("m1_t_minus_k_hz")].iloc[0]
    assert segmental.failure_reason in {"missing_alignment", "missing_target_definition"}
    assert np.isnan(values.m1_t_minus_k_hz.iloc[0])
    for feature_id in ("pause_count", "mfcc01_mean", "f0_mean_hz"):
        assert status.loc[status.feature_id.eq(feature_id), "failure_reason"].iloc[0] == ""
    assert result.status == "completed_with_warnings"
    audit = (root / "acoustic" / "005_features" / "family_runs" / "segmental" /
             "tables" / "native_measurements" / "family06_segmental_targets.csv")
    assert audit.is_file()


def test_segmental_approved_pair_produces_auditable_contrast(tmp_path, monkeypatch):
    root, decisions, intervals, _ = _case(tmp_path, monkeypatch)

    class Alignment:
        manifest = {"alignment_run_id": "align1", "provider": "validated_external"}

        @staticmethod
        def get_word_tokens(_recording_id):
            return pd.DataFrame([
                {"word_index": 1, "word": "testt"},
                {"word_index": 2, "word": "testk"}])

        @staticmethod
        def get_phone_tokens(_recording_id):
            return pd.DataFrame([
                {"word_index": 1, "phone_index": 1, "phone_normalized": "T",
                 "start_sec": .25, "end_sec": .45},
                {"word_index": 2, "phone_index": 1, "phone_normalized": "K",
                 "start_sec": 1.25, "end_sec": 1.45}])

    monkeypatch.setattr("vslp.acoustic.features.family06_stage.load_final_alignment",
                        lambda *_: (Alignment(), ""))
    target_path = root / "targets.json"
    targets = []
    annotations = []
    for role, index, onset in (("t", 1, .3), ("k", 2, 1.3)):
        target_id = f"p1_{role}"
        targets.append({"target_id": target_id, "feature_id": "m1_t_minus_k_hz",
                        "word": f"test{role}", "phone": role.upper(),
                        "word_index": index, "phone_index": 1, "role": role,
                        "pair_id": "p1", "acoustic_region_type": "post_burst_20ms",
                        "pairing_rule": "matched_pair_id",
                        "aggregation": "mean_target_moments_then_difference"})
        annotations.append({"recording_id": "r1", "target_id": target_id,
                            "word_index": index, "phone_index": 1,
                            "annotation_source": "test_validated", "reviewer": "tester",
                            "approved": True, "time_axis": "original_recording_seconds",
                            "burst_start_sec": onset})
    target_path.write_text(json.dumps({"manifest_version": "1", "task_id": "bamboo_passage",
                                       "targets": targets}), encoding="utf-8")
    annotations_path = root / "subevents.csv"
    pd.DataFrame(annotations).to_csv(annotations_path, index=False)
    cfg = FeatureExtractionConfig(
        selected_features=["m1_t_minus_k_hz", "pause_count"],
        segmental_target_manifest_path=str(target_path),
        acoustic_subevent_annotations_csv=str(annotations_path))
    result = run_acoustic_feature_extraction(decisions, root, cfg, intervals)
    status = pd.read_csv(result.summary_table.parent / "acoustic_feature_status_long.csv",
                         keep_default_na=False)
    assert status.failure_reason.eq("").all()
    audit = pd.read_csv(root / "acoustic" / "005_features" / "family_runs" /
                        "segmental" / "tables" / "native_measurements" /
                        "family06_segmental_targets.csv")
    assert set(audit.role) == {"t", "k"}
    assert set(audit.measurement_boundary_source) == {"validated_external_annotation"}
    assert set(audit.alignment_run_id) == {"align1"}
