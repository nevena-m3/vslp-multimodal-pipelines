"""Synthetic scale check for exception-only Alignment QC artifacts."""

import json

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.alignment.qc import create_alignment_qc, generate_selected_trial_plot


def test_200_recording_qc_generates_only_ten_flagged_plots(tmp_path):
    root = tmp_path / "synthetic_project"
    run = root / "acoustic" / "004_alignment" / "runs" / "synthetic_run"
    (run / "configs").mkdir(parents=True)
    (run / "tables").mkdir()
    reviewed = root / "acoustic" / "003_segmentation_review" / "final"
    reviewed.mkdir(parents=True)
    wav = root / "synthetic.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    trials, reviews, diagnostics, decisions = [], [], [], []
    expected, observed = {}, {}
    for record_index in range(200):
        identity = f"synthetic_{record_index:03d}"
        expected[identity] = observed[identity] = 3
        decisions.append({"recording_id": identity, "analysis_wav_path": str(wav)})
        for trial_index in range(3):
            trial_id = f"trial_{trial_index + 1:03d}"
            flagged = record_index >= 190 and trial_index == 0
            trials.append({"recording_id": identity, "trial_id": trial_id,
                           "start_sec": 0., "end_sec": 1.,
                           "prompt_id": "wstg_we_see_three_geese"})
            reviews.append({"recording_id": identity, "trial_id": trial_id,
                            "review_status": "NEEDS_REVIEW" if flagged else
                            "AUTO_ACCEPTED_STRUCTURAL", "review_mode": "AUTO",
                            "flags_triggered": json.dumps(
                                ["INCOMPLETE_WORD_COVERAGE"] if flagged else [])})
            diagnostics.append({"recording_id": identity, "trial_id": trial_id,
                                "status": "ALIGNED", "expected_word_count": 4,
                                "n_words": 3 if flagged else 4, "n_phones": 8})
    (run / "configs" / "alignment_trials.json").write_text(
        json.dumps({"trials": trials}), encoding="utf-8")
    pd.DataFrame(reviews).to_csv(run / "tables" / "alignment_trial_review.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(
        run / "tables" / "alignment_trial_diagnostics.csv", index=False)
    pd.DataFrame(decisions).to_csv(
        reviewed / "final_segmentation_decisions.csv", index=False)
    manifest = {"alignment_run_id": "synthetic_run", "source": "synthetic",
                "expected_trials_by_recording": expected,
                "observed_trials_by_recording": observed,
                "alignment_review_policy_version": "synthetic_policy"}
    create_alignment_qc(root, run, manifest)
    index = pd.read_csv(run / "tables" / "alignment_qc_index.csv")
    assert len(index) == 600
    flagged_records = set(index.loc[index.review_status.eq("NEEDS_REVIEW"), "recording_id"])
    assert len(flagged_records) == 10
    assert index.recording_id.nunique() - len(flagged_records) == 190
    assert len(list((run / "diagnostics" / "flagged_trials").glob("*.png"))) == 10
    assert (run / "diagnostics" / "alignment_qc_summary.png").is_file()
    (run / "logs").mkdir()
    (run / "logs" / "stage_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    selected = generate_selected_trial_plot(root, "synthetic_run",
                                            "synthetic_000", "trial_001")
    assert selected.is_file()
    assert len(list((run / "diagnostics" / "flagged_trials").glob("*.png"))) == 11
    updated_manifest = json.loads((run / "logs" / "stage_manifest.json").read_text(
        encoding="utf-8"))
    assert updated_manifest["alignment_qc_artifacts"]["qc_index_path"]["sha256"]
