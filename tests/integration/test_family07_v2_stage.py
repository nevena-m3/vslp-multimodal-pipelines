"""Nonclinical mixed-prerequisite F07 stage tests; no MFA is invoked."""

from __future__ import annotations

import json
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vslp.acoustic.features.catalog import load_feature_catalog
from vslp.acoustic.alignment.reviewed_trials import confirmed_trial_rows
from vslp.acoustic.features.family07_v2 import S3_EXPECTED_WORD_VOWELS
from vslp.acoustic.features.family07_v2_stage import run_family07_stage


def _case(tmp_path, monkeypatch, task):
    root = tmp_path / "nonclinical"
    final = root / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    pd.DataFrame([{"recording_id": "r", "file_name": "synthetic.wav",
                   "final_decision": "KEEP_MANUAL"}]).to_csv(decisions, index=False)
    speech = ([('speech', .1, .6), ('internal_nonspeech', .6, .8),
               ('speech', .8, 1.2)] if task == "bamboo_passage" else
              [('speech', .1, 2.1)])
    pd.DataFrame([{"recording_id": "r", "view": "authoritative",
                   "segment_role": role, "start_sec": start, "end_sec": end}
                  for role, start, end in speech]).to_csv(intervals, index=False)
    (root / "project_manifest.json").write_text(json.dumps({"task_id": task,
        "run_id": "nonclinical"}), encoding="utf-8")
    kept = pd.DataFrame([{"recording_id": "r", "file_name": "synthetic.wav"}])
    monkeypatch.setattr("vslp.acoustic.features.family07_v2_stage.load_final_segmentation",
                        lambda *_: kept)
    monkeypatch.setattr("vslp.acoustic.features.mixed_dispatch.load_final_segmentation",
                        lambda *_: kept.assign(segmentation_run_id="seg", review_run_id="review"))
    if task == "sentence_s3":
        config = root / "configs"
        config.mkdir()
        rows = confirmed_trial_rows(root, recording_id="r", task_id=task,
                                    prompt_id="sentence_s3_blue_spot_key", bounds=[(.1, 2.1)])
        (config / "alignment_choices.json").write_text(
            json.dumps({"trials": {"r": rows}}), encoding="utf-8")
    return root, decisions, intervals


def _run(root, decisions, intervals, selected):
    catalog = {item["feature_id"]: item for item in load_feature_catalog()["outputs"]}
    registry = pd.DataFrame([{"feature": feature, "unit": catalog[feature]["unit"]}
                             for feature in selected])
    result = run_family07_stage(decisions, root, SimpleNamespace(selected_features=selected),
                                intervals, registry, execution_stage_dir=root / "f07")
    tables = result.summary_table.parent
    return (pd.read_csv(tables / "acoustic_features_per_file.csv"),
            pd.read_csv(tables / "acoustic_feature_status_long.csv"), tables)


def test_bamboo_timing_without_alignment_and_local_alignment_failure(tmp_path, monkeypatch):
    root, decisions, intervals = _case(tmp_path, monkeypatch, "bamboo_passage")
    import vslp.acoustic.alignment as alignment

    monkeypatch.setattr(alignment, "load_final_alignment", lambda *_: (None, "missing_alignment"))
    values, status, tables = _run(root, decisions, intervals,
        ["bamboo_speech_duration_s", "bamboo_total_duration_s", "delta_v_s"])
    assert values.bamboo_speech_duration_s.iloc[0] == pytest.approx(.9)
    assert values.bamboo_total_duration_s.iloc[0] == pytest.approx(1.1)
    by_id = status.set_index("feature_id")
    assert by_id.loc["bamboo_speech_duration_s", "status"] == "computed"
    assert by_id.loc["bamboo_total_duration_s", "status"] == "computed"
    assert by_id.loc["delta_v_s", "failure_reason"] == "missing_final_alignment"
    assert (tables / "native_measurements" / "family07_timing_events.csv").is_file()


def test_s3_sentence_duration_never_loads_alignment(tmp_path, monkeypatch):
    root, decisions, intervals = _case(tmp_path, monkeypatch, "sentence_s3")
    import vslp.acoustic.alignment as alignment

    monkeypatch.setattr(alignment, "load_final_alignment", lambda *_: pytest.fail(
        "S3 sentence duration must not load Alignment"))
    values, status, _tables = _run(root, decisions, intervals, ["s3_sentence_duration_s"])
    assert values.s3_sentence_duration_s.iloc[0] == pytest.approx(2.)
    assert status.iloc[0].input_contract == "FINAL_REVIEWED_TIMING"


def test_s3_legacy_trial_confirmation_is_not_silently_trusted(tmp_path, monkeypatch):
    root, decisions, intervals = _case(tmp_path, monkeypatch, "sentence_s3")
    choices = root / "configs" / "alignment_choices.json"
    choices.write_text(json.dumps({"trials": {"r": [
        {"trial_id": "trial_001", "start_sec": .1, "end_sec": 2.1,
         "prompt_id": "sentence_s3_blue_spot_key"}]}}), encoding="utf-8")
    values, status, _ = _run(root, decisions, intervals, ["s3_sentence_duration_s"])
    assert np.isnan(values.s3_sentence_duration_s.iloc[0])
    assert status.iloc[0].failure_reason == "legacy_unverified_trial_confirmation"


def test_s3_all_vowels_from_final_alignment_and_sum_numerator(tmp_path, monkeypatch):
    root, decisions, intervals = _case(tmp_path, monkeypatch, "sentence_s3")

    class Final:
        manifest = {"alignment_run_id": "synthetic-v1"}
        trials = pd.DataFrame([{"recording_id": "r", "trial_id": "trial_001",
                                "sequence_id": "seq1", "task_id": "sentence_s3",
                                "prompt_id": "sentence_s3_blue_spot_key",
                                "start_sec": .1, "end_sec": 2.1}])

        @staticmethod
        def phones_for_trial(_recording_id, _trial_id):
            durations = [.05, .06, .07, .08, .09, .10, .11, .12, .12]
            return pd.DataFrame([{"recording_id": "r", "trial_id": "trial_001",
                "sequence_id": "seq1", "phone_index_in_trial": index,
                "phone_id": f"p{index}", "parent_word_id": f"w{index}",
                "parent_word_text": word, "phone_raw": f"{phone}1",
                "phone_normalized": phone, "start_sec": .2 + .18 * (index - 1),
                "end_sec": .2 + .18 * (index - 1) + duration,
                "alignment_review_status": "HUMAN_ACCEPTED", "boundary_source": "MFA"}
                for index, ((word, phone), duration) in enumerate(
                    zip(S3_EXPECTED_WORD_VOWELS, durations, strict=True), 1)])

    import vslp.acoustic.alignment as alignment

    monkeypatch.setattr(alignment, "load_final_alignment", lambda *_: (Final(), ""))
    selected = ["s3_vocalic_duration_mean_s", "s3_vocalic_duration_sd_s",
                "s3_vocalic_proportion_pct", "s3_sentence_duration_s"]
    values, status, tables = _run(root, decisions, intervals, selected)
    assert values.s3_vocalic_proportion_pct.iloc[0] == pytest.approx(40.)
    assert values.s3_vocalic_duration_mean_s.iloc[0] == pytest.approx(.8 / 9)
    assert status.set_index("feature_id").loc[
        "s3_vocalic_proportion_pct", "input_contract"] == "FINAL_ALIGNMENT_CONTRACT_V1"
    tokens = pd.read_csv(tables / "native_measurements" / "family07_vocalic_tokens.csv")
    trials = pd.read_csv(tables / "native_measurements" / "family07_trials.csv")
    assert len(tokens) == 9 and tokens.valid.all()
    assert trials.loc[trials.feature_id.eq("s3_vocalic_proportion_pct"),
                      "s3_total_vocalic_duration_s"].iloc[0] == pytest.approx(.8)


def test_s3_hard_exclusion_blocks_sentence_duration_without_alignment(tmp_path, monkeypatch):
    root, decisions, intervals = _case(tmp_path, monkeypatch, "sentence_s3")
    table = pd.read_csv(intervals)
    table = pd.concat([table, pd.DataFrame([{"recording_id": "r", "view": "authoritative",
        "segment_role": "manual_exclusion", "start_sec": 1., "end_sec": 1.1}])], ignore_index=True)
    table.to_csv(intervals, index=False)
    choices = root / "configs" / "alignment_choices.json"
    payload = json.loads(choices.read_text(encoding="utf-8"))
    payload["trials"]["r"][0]["segmentation_review_hash"] = sha256(
        decisions.read_bytes() + intervals.read_bytes()).hexdigest()
    choices.write_text(json.dumps(payload), encoding="utf-8")
    values, status, _ = _run(root, decisions, intervals, ["s3_sentence_duration_s"])
    assert np.isnan(values.s3_sentence_duration_s.iloc[0])
    assert status.iloc[0].failure_reason == "sentence_contains_hard_exclusion"


def test_mixed_dispatch_preserves_timing_when_alignment_is_missing(tmp_path, monkeypatch):
    from vslp.acoustic.features.mixed_dispatch import run_mixed_feature_stage
    from vslp.acoustic.features.stage import FeatureExtractionConfig, _select_registry

    root, decisions, intervals = _case(tmp_path, monkeypatch, "bamboo_passage")
    import vslp.acoustic.alignment as alignment

    monkeypatch.setattr(alignment, "load_final_alignment", lambda *_: (None, "missing_alignment"))
    selected = ["bamboo_speech_duration_s", "bamboo_total_duration_s", "delta_v_s"]
    config = FeatureExtractionConfig(selected_features=selected)
    result = run_mixed_feature_stage(decisions, root, config, intervals,
                                    _select_registry(config))
    status = pd.read_csv(result.summary_table.parent / "acoustic_feature_status_long.csv",
                         keep_default_na=False).set_index("feature_id")
    assert status.loc["bamboo_speech_duration_s", "status"] == "computed"
    assert status.loc["bamboo_total_duration_s", "status"] == "computed"
    assert status.loc["delta_v_s", "failure_reason"] == "missing_final_alignment"


def test_s3_proportion_rejects_mismatched_confirmed_trial_bounds(tmp_path, monkeypatch):
    root, decisions, intervals = _case(tmp_path, monkeypatch, "sentence_s3")

    class Final:
        manifest = {"alignment_run_id": "synthetic-v1"}
        trials = pd.DataFrame([{"recording_id": "r", "trial_id": "trial_001",
                                "sequence_id": "seq1", "task_id": "sentence_s3",
                                "prompt_id": "sentence_s3_blue_spot_key",
                                "start_sec": .2, "end_sec": 2.1}])

        @staticmethod
        def phones_for_trial(_recording_id, _trial_id):
            return pd.DataFrame([{"sequence_id": "seq1", "phone_index_in_trial": index,
                "phone_id": f"p{index}", "parent_word_id": f"w{index}",
                "parent_word_text": word, "phone_raw": f"{phone}1",
                "phone_normalized": phone, "start_sec": .25 + .18 * (index - 1),
                "end_sec": .30 + .18 * (index - 1),
                "alignment_review_status": "HUMAN_ACCEPTED", "boundary_source": "MFA"}
                for index, (word, phone) in enumerate(S3_EXPECTED_WORD_VOWELS, 1)])

    import vslp.acoustic.alignment as alignment

    monkeypatch.setattr(alignment, "load_final_alignment", lambda *_: (Final(), ""))
    values, status, _ = _run(root, decisions, intervals,
                             ["s3_sentence_duration_s", "s3_vocalic_proportion_pct"])
    assert values.s3_sentence_duration_s.iloc[0] == pytest.approx(2.)
    assert np.isnan(values.s3_vocalic_proportion_pct.iloc[0])
    assert status.set_index("feature_id").loc[
        "s3_vocalic_proportion_pct", "failure_reason"] == "invalid_trial_bounds"
