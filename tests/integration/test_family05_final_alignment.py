"""F05 reads only frozen canonical V1 tokens and preserves trial aggregation."""

import json

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.features.catalog import load_feature_catalog
from vslp.acoustic.features.family05_stage import run_family05_stage
from vslp.acoustic.features.stage import FeatureExtractionConfig, _select_registry


class FinalStore:
    def __init__(self, task_id, trials, phones, words):
        self.trials = pd.DataFrame(trials)
        self._phones = pd.DataFrame(phones)
        self._words = pd.DataFrame(words)
        self.manifest = {"alignment_run_id": "frozen_v1"}
        self.task_id = task_id

    def words_for_trial(self, recording_id, trial_id):
        return self._words.loc[self._words.trial_id.eq(trial_id)].copy()

    def phones_for_trial(self, recording_id, trial_id):
        return self._phones.loc[self._phones.trial_id.eq(trial_id)].copy()


def _run_case(tmp_path, monkeypatch, task_id, targets, trial_count=2,
              issue="", exclusion=False):
    root = tmp_path / "project"
    review = root / "acoustic" / "003_segmentation_review" / "final"
    review.mkdir(parents=True)
    decisions = review / "final_segmentation_decisions.csv"
    intervals = review / "final_segmentation_intervals.csv"
    decisions.write_text("recording_id\nr\n", encoding="utf-8")
    rows = []
    for index in range(trial_count):
        rows.append({"recording_id": "r", "segment_role": "speech",
                     "start_sec": index * 1.0, "end_sec": index * 1.0 + .8})
    if exclusion:
        rows.append({"recording_id": "r", "segment_role": "manual_exclusion",
                     "start_sec": .13, "end_sec": .15})
    pd.DataFrame(rows).to_csv(intervals, index=False)
    (root / "project_manifest.json").write_text(json.dumps({"task_id": task_id,
        "task_type": "Sentence / controlled speech"}), encoding="utf-8")
    audio_path = root / "native.wav"
    sf.write(audio_path, np.zeros(32000, dtype=np.float32), 16000)
    kept = pd.DataFrame([{"recording_id": "r", "file_name": "native.wav",
                          "source_file_path": str(audio_path),
                          "analysis_wav_path": str(audio_path)}])
    monkeypatch.setattr("vslp.acoustic.features.family05_stage.load_final_segmentation",
                        lambda *_: kept)
    trial_rows, phone_rows, word_rows = [], [], []
    for index in range(trial_count):
        trial_id = f"trial_{index + 1}"
        trial_rows.append({"recording_id": "r", "trial_id": trial_id,
                           "task_id": task_id, "sequence_id": f"r:{trial_id}",
                           "prompt_id": ("sentence_s2_buy_bobby_a_puppy" if task_id ==
                                         "sentence_s2" else "sentence_s5_we_were_away_a_year_ago"),
                           "start_sec": float(index), "end_sec": index + .8})
        for offset, (target_id, word, phone) in enumerate(targets):
            word_id = f"{trial_id}:{word}:{offset}"
            word_rows.append({"trial_id": trial_id, "word_id": word_id,
                              "canonical_word_text": word})
            start = index + .1 + offset * .2
            phone_rows.append({"trial_id": trial_id, "sequence_id": f"r:{trial_id}",
                               "task_item": task_id, "parent_word_id": word_id,
                               "parent_word_text": word, "phone_id": f"{trial_id}:{target_id}:{offset}",
                               "phone_normalized": phone, "phone_raw": phone,
                               "start_sec": start, "end_sec": start + .1,
                               "boundary_source": "REVIEWED_MANUAL_CORRECTION",
                               "alignment_review_status": "HUMAN_ACCEPTED"})
    store = FinalStore(task_id, trial_rows, phone_rows, word_rows)
    monkeypatch.setattr("vslp.acoustic.features.family05_stage.load_final_alignment",
                        lambda *_: (None, issue) if issue else (store, ""))

    def frames_from_shared_burg(audio, rate, phone_tokens, profile, **kwargs):
        phone = phone_tokens.iloc[0]
        start = float(phone.start_sec)
        times = start + np.arange(0, .101, .005)
        local = times - start
        return pd.DataFrame(), pd.DataFrame({
            "token_index": 1, "time_sec": times,
            "f1_hz": 400 + 1000 * local,
            "f2_hz": 1800 - 500 * local,
            "f3_hz": np.full(len(times), 2800.),
            "valid_ordered": True,
        })

    monkeypatch.setattr("vslp.acoustic.features.family05_stage.track_aligned_vowels",
                        frames_from_shared_burg)
    catalog = load_feature_catalog()
    ids = {target for target, *_ in targets}
    specs = [x for x in catalog["outputs"] if x["family_id"] == "F05"
             and x["target_id"] in ids]
    config = FeatureExtractionConfig(selected_features=[x["feature_id"] for x in specs])
    registry = _select_registry(config)  # real stage registry lacks target_id columns
    result = run_family05_stage(decisions, root, config, intervals, registry,
                                execution_stage_dir=root / "f05")
    tables = root / "f05" / "tables"
    return result, pd.read_csv(tables / "acoustic_feature_status_long.csv",
                               keep_default_na=False), pd.read_csv(
        tables / "native_measurements" / "formant_dynamics_tokens.csv"), pd.read_csv(
        tables / "native_measurements" / "formant_dynamics_trials.csv")


def test_s2_final_v1_native_waveform_repeated_trials(tmp_path, monkeypatch):
    _, status, tokens, trials = _run_case(tmp_path, monkeypatch, "sentence_s2",
                                          [("buy_ay", "buy", "AY")])
    assert len(status) == 4
    assert set(status.status) == {"computed"}
    assert len(tokens) == 2
    assert set(tokens.native_sample_rate_hz) == {16000}
    assert set(tokens.boundary_source) == {"REVIEWED_MANUAL_CORRECTION"}
    np.testing.assert_allclose(tokens.analysis_start_sec.iloc[0], .12, atol=1e-6)
    assert set(tokens.fit_method) == {"bisquare_irls_mad_4.685_v1"}
    assert trials.groupby("feature_id").trial_id.nunique().eq(2).all()
    assert status.loc[status.feature_id.eq("f2_transition_slope_buy_ay_hz_s"),
                      "value"].iloc[0] < 0
    assert status.loc[status.feature_id.eq("f1_range_buy_ay_hz"),
                      "value"].iloc[0] > 0


def test_s5_away_and_ago_remain_separate(tmp_path, monkeypatch):
    _, status, tokens, _ = _run_case(tmp_path, monkeypatch, "sentence_s5",
        [("away_ey", "away", "EY"), ("ago_ow", "ago", "OW")], trial_count=1)
    assert len(status) == 4
    assert set(tokens.target_id) == {"away_ey", "ago_ow"}
    assert not status.feature_id.astype(str).str.contains("range").any()


def test_equivalent_tokens_median_within_trial(tmp_path, monkeypatch):
    _, status, tokens, trials = _run_case(tmp_path, monkeypatch, "sentence_s2",
        [("buy_ay", "buy", "AY"), ("buy_ay", "buy", "AY")], trial_count=1)
    assert len(tokens) == 2 and tokens.phone_id.nunique() == 2
    assert trials.groupby("feature_id").n_valid_tokens.first().eq(2).all()
    assert set(status.status) == {"computed"}


def test_missing_final_alignment_fails_locally(tmp_path, monkeypatch):
    _, status, tokens, trials = _run_case(tmp_path, monkeypatch, "sentence_s2",
        [("buy_ay", "buy", "AY")], issue="missing_alignment")
    assert set(status.failure_reason) == {"missing_final_alignment"}
    assert tokens.empty and trials.empty


def test_manual_exclusion_blocks_target_before_formant_measurement(tmp_path, monkeypatch):
    _, status, tokens, _ = _run_case(tmp_path, monkeypatch, "sentence_s2",
        [("buy_ay", "buy", "AY")], trial_count=1, exclusion=True)
    assert set(status.failure_reason) == {"target_outside_eligible_audio"}
    assert set(tokens.qc_flags) == {"target_outside_eligible_audio"}
    assert tokens.n_valid_frames.isna().all()


def test_stale_final_alignment_rejected(tmp_path, monkeypatch):
    _, status, tokens, _ = _run_case(tmp_path, monkeypatch, "sentence_s2",
        [("buy_ay", "buy", "AY")], trial_count=1, issue="stale_alignment_trial_revision")
    assert set(status.failure_reason) == {"stale_final_alignment"}
    assert set(status.alignment_prerequisite_detail) == {"stale_alignment_trial_revision"}
    assert tokens.empty
