"""Reviewed repeated nonlinguistic intervals remain distinct without MFA."""

from __future__ import annotations

import pandas as pd

from vslp.acoustic.alignment.trials import reviewed_trial_candidates
from vslp.acoustic.features.family10 import derive_ddk_feature_events


def test_sustained_reviewed_productions_are_separate_candidates(tmp_path):
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    pd.DataFrame([{"recording_id": "r1", "segment_role": "speech",
                   "start_sec": left, "end_sec": right}
                  for left, right in ((.5, 1.5), (2.5, 3.5), (4.5, 5.5))]).to_csv(
                      final / "final_segmentation_intervals.csv", index=False)
    assert reviewed_trial_candidates(tmp_path, "r1") == [
        (.5, 1.5), (2.5, 3.5), (4.5, 5.5)]
    # These are only reviewed speech candidates; this test does not call MFA
    # or claim a per-trial Family 01/02 output contract.


def test_ddk_reviewed_event_trains_remain_separate_without_mfa():
    rows = [{"view": "authoritative", "segment_role": "speech",
             "start_sec": start, "end_sec": start + .07,
             "boundary_source": "reviewed"}
            for start in (.1, .3, .5, 1.5, 1.7, 1.9)]
    rows.append({"view": "authoritative", "segment_role": "manual_exclusion",
                 "start_sec": .9, "end_sec": 1.1,
                 "boundary_source": "manual"})
    events, summary, reason = derive_ddk_feature_events(
        pd.DataFrame(rows), recording_id="r1", file_name="synthetic.wav",
        analysis_start_sec=0, analysis_end_sec=2.1)
    assert reason == ""
    assert summary["n_valid_sequences"] == 2
    assert events.loc[events.event_kind.eq("ddk_event"), "sequence_id"].tolist() == [
        1, 1, 1, 2, 2, 2]
