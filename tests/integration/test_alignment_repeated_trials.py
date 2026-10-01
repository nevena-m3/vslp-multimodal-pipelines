"""Nonclinical three-trial corpus contract and human freeze gate."""

from __future__ import annotations

import json
import hashlib

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vslp.acoustic.alignment.stage import (
    AlignmentConfig, freeze_alignment, list_alignment_runs, load_final_alignment,
    run_acoustic_alignment,
)
from vslp.acoustic.alignment.trial_review import set_trial_review
from vslp.acoustic.alignment.trial_review import mark_recording_stale, trial_review_table
from vslp.acoustic.alignment.corrections import (load_corrections, reset_corrections,
                                                 set_correction)


def _fixture(tmp_path, n_trials=3):
    root = tmp_path / "synthetic_wstg"
    root.mkdir()
    audio = np.zeros(8 * 16000, dtype="float32")
    intervals = []
    trials = []
    for number, start in enumerate((.5, 3., 5.5)[:n_trials], 1):
        begin, end = round(start * 16000), round((start + 1) * 16000)
        audio[begin:end] = .08 * np.sin(2 * np.pi * 210 * np.arange(end - begin) / 16000)
        intervals.append({"recording_id": "r1", "segment_role": "speech",
                          "start_sec": start, "end_sec": start + 1})
        trials.append({"recording_id": "r1", "trial_id": f"trial_{number:03d}",
                       "prompt_id": "wstg_we_see_three_geese",
                       "start_sec": start, "end_sec": start + 1})
    wav = root / "synthetic.wav"
    sf.write(wav, audio, 16000)
    (root / "project_manifest.json").write_text(json.dumps({"task_id": "wstg"}))
    review = root / "acoustic" / "003_segmentation_review" / "final"
    review.mkdir(parents=True)
    pd.DataFrame([{"recording_id": "r1", "file_name": "synthetic.wav",
                   "final_decision": "KEEP_MANUAL", "analysis_wav_path": str(wav),
                   "analysis_start_sec": 0, "analysis_end_sec": 8,
                   "segmentation_run_id": "seg", "review_run_id": "review"}]).to_csv(
                       review / "final_segmentation_decisions.csv", index=False)
    pd.DataFrame(intervals).to_csv(review / "final_segmentation_intervals.csv", index=False)
    prompt = root / "prompt.json"
    prompt.write_text(json.dumps({"manifest_version": "1", "task_id": "wstg",
        "prompt_version": "v1", "language": "en", "transcript": "We see three geese.",
        "expected_words": ["WE", "SEE", "THREE", "GEESE"],
        "phone_set": "ARPABET_CMU_39"}))
    speakers = root / "speakers.json"
    speakers.write_text(json.dumps({"recordings": [{"recording_id": "r1",
        "speaker_id": "technical_fixture", "speaker_id_source": "technical_recording_key"}]}))
    trial_path = root / "trials.json"
    trial_path.write_text(json.dumps({"schema_version": "1", "trials": trials}))
    return root, prompt, speakers, trial_path


class FakeCorpusProvider:
    name = "synthetic_mock"
    version = "1"

    def align_corpus(self, corpus, output, profile, logs):
        files = sorted(corpus.rglob("*.wav"))
        self.aligned_units = [file.stem for file in files]
        output.mkdir(parents=True)
        result = {}
        for wav in files:
            assert wav.with_suffix(".lab").read_text() == "WE SEE THREE GEESE"
            grid = output / f"{wav.stem}.TextGrid"
            lines = ['name = "words"']
            for number, label in enumerate(("WE", "SEE", "THREE", "GEESE"), 1):
                start = (number - 1) * .2 + .05
                lines += [f"intervals [{number}]:", f"xmin = {start}",
                          f"xmax = {start + .15}", f'text = "{label}"']
            lines += ['name = "phones"']
            for number, label in enumerate(("W", "S", "TH", "G"), 1):
                start = (number - 1) * .2 + .06
                lines += [f"intervals [{number}]:", f"xmin = {start}",
                          f"xmax = {start + .1}", f'text = "{label}"']
            grid.write_text("\n".join(lines), encoding="utf-8")
            result[wav.stem] = grid
        return result


@pytest.mark.parametrize("n_trials", [2, 3])
def test_repeated_sentence_trials_align_independently_and_freeze_after_review(tmp_path, n_trials):
    root, prompt, speakers, trials = _fixture(tmp_path, n_trials)
    result = run_acoustic_alignment(root, AlignmentConfig(
        source="mfa", prompt_manifest_path=str(prompt),
        speaker_manifest_path=str(speakers), trial_manifest_path=str(trials),
        provider=FakeCorpusProvider()))
    assert result.status == "completed"
    run_id = list_alignment_runs(root)[0]["alignment_run_id"]
    base = root / "acoustic" / "004_alignment" / "runs" / run_id / "tables"
    words = pd.read_csv(base / "alignment_words.csv")
    phones = pd.read_csv(base / "alignment_phones.csv")
    trials_table = pd.read_csv(base / "alignment_trial_diagnostics.csv")
    assert len(words) == 4 * n_trials
    assert len(phones) == 4 * n_trials
    assert len(trials_table) == n_trials
    assert trials_table.n_words.tolist() == [4] * n_trials
    assert words.start_sec.iloc[0] == pytest.approx(.55, abs=.002)
    assert words.start_sec.iloc[-1] == pytest.approx((.5, 3., 5.5)[n_trials - 1] + .65, abs=.002)
    assert words.word_index.is_unique
    with pytest.raises(ValueError, match="review_required"):
        freeze_alignment(root, run_id)
    for number in range(1, n_trials + 1):
        set_trial_review(root, run_id, "r1", f"trial_{number:03d}", "ACCEPTED")
    freeze_alignment(root, run_id)
    store, issue = load_final_alignment(root)
    assert issue == "" and store is not None
    assert all(len(store.get_trial_tokens("r1", f"trial_{number:03d}")) == 4
               for number in range(1, n_trials + 1))
    assert store.manifest["trial_contract_version"] == "reviewed_trials_v1"


def test_needs_review_trial_cannot_freeze(tmp_path):
    root, prompt, speakers, trials = _fixture(tmp_path, 2)
    run_acoustic_alignment(root, AlignmentConfig(source="mfa",
        prompt_manifest_path=str(prompt), speaker_manifest_path=str(speakers),
        trial_manifest_path=str(trials), provider=FakeCorpusProvider()))
    run_id = list_alignment_runs(root)[0]["alignment_run_id"]
    set_trial_review(root, run_id, "r1", "trial_001", "ACCEPTED")
    set_trial_review(root, run_id, "r1", "trial_002", "NEEDS_REVIEW")
    with pytest.raises(ValueError, match="all_alignment_trials_must_be_accepted"):
        freeze_alignment(root, run_id)


@pytest.mark.parametrize("excluded_role", ["cough", "other_speaker"])
def test_manual_exclusion_inside_trial_never_reaches_provider(tmp_path, excluded_role):
    root, prompt, speakers, trials = _fixture(tmp_path, 1)
    path = (root / "acoustic" / "003_segmentation_review" / "final" /
            "final_segmentation_intervals.csv")
    intervals = pd.read_csv(path)
    intervals = pd.concat([intervals, pd.DataFrame([{
        "recording_id": "r1", "segment_role": "manual_exclusion",
        "start_sec": .8, "end_sec": .9, "review_label": excluded_role,
    }])], ignore_index=True)
    intervals.to_csv(path, index=False)
    provider = FakeCorpusProvider()
    with pytest.raises(ValueError, match="TRIAL_CROSSES_EXCLUSION"):
        run_acoustic_alignment(root, AlignmentConfig(source="mfa",
            prompt_manifest_path=str(prompt), speaker_manifest_path=str(speakers),
            trial_manifest_path=str(trials), provider=provider))
    assert not hasattr(provider, "aligned_units")


@pytest.mark.parametrize("n_trials,expected_auto", [(2, False), (3, True)])
def test_structural_review_policy_auto_passes_only_matching_complete_trials(
        tmp_path, n_trials, expected_auto):
    root, prompt, speakers, trials = _fixture(tmp_path, n_trials)
    content = json.loads(prompt.read_text(encoding="utf-8"))
    content["expected_repetitions"] = 3
    prompt.write_text(json.dumps(content), encoding="utf-8")
    run_acoustic_alignment(root, AlignmentConfig(source="mfa",
        prompt_manifest_path=str(prompt), speaker_manifest_path=str(speakers),
        trial_manifest_path=str(trials), provider=FakeCorpusProvider(),
        structural_auto_review=True))
    run_id = list_alignment_runs(root)[-1]["alignment_run_id"]
    reviews = trial_review_table(root, run_id)
    assert len(reviews) == n_trials
    if expected_auto:
        assert reviews.review_status.eq("AUTO_ACCEPTED_STRUCTURAL").all()
        assert reviews.review_mode.eq("AUTO").all()
        run = root / "acoustic" / "004_alignment" / "runs" / run_id
        qc = pd.read_csv(run / "tables" / "alignment_qc_index.csv")
        assert qc.review_status.eq("AUTO_ACCEPTED_STRUCTURAL").all()
        assert qc.plot_path.isna().all() or qc.plot_path.eq("").all()
        assert (run / "diagnostics" / "alignment_qc_summary.png").is_file()
        assert not list((run / "diagnostics" / "flagged_trials").glob("*.png"))
        freeze_alignment(root, run_id)
        final = pd.read_csv(root / "acoustic" / "004_alignment" / "final" /
                            "final_alignment_trial_review.csv")
        assert final.final_status.eq("AUTO_ACCEPTED_STRUCTURAL").all()
        final_root = root / "acoustic" / "004_alignment" / "final"
        assert len(pd.read_csv(final_root / "final_alignment_trials.csv")) == 3
        assert (final_root / "final_alignment_qc_index.csv").is_file()
        assert (final_root / "diagnostics" / "alignment_qc_summary.png").is_file()
    else:
        assert reviews.review_status.eq("NEEDS_REVIEW").all()
        assert reviews.flags_triggered.str.contains("PROTOCOL_TRIAL_COUNT_MISMATCH").all()
        run = root / "acoustic" / "004_alignment" / "runs" / run_id
        qc = pd.read_csv(run / "tables" / "alignment_qc_index.csv")
        assert qc["flags"].str.contains("PROTOCOL_TRIAL_COUNT_MISMATCH").all()
        assert all((run / path).is_file() for path in qc.plot_path)
        from PySide6.QtWidgets import QApplication
        from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget
        QApplication.instance() or QApplication([])
        widget = AlignmentWidget(lambda: root)
        widget.refresh()
        assert "REVIEW 2 ISSUES" == widget.primary.text()
        assert "protocol trial count mismatch" in widget.normal_summary.text()
        widget.close()
        with pytest.raises(ValueError, match="alignment_trial_review_required"):
            freeze_alignment(root, run_id)


@pytest.mark.parametrize("n_trials", [2, 3])
def test_reviewed_speech_proposes_only_exact_protocol_count(tmp_path, n_trials):
    from vslp.acoustic.alignment.task_workflow import project_alignment_context
    from vslp.acoustic.alignment.trials import structurally_proposed_trials

    root, _prompt, _speakers, _trials = _fixture(tmp_path, n_trials)
    context = project_alignment_context(root)
    proposed = structurally_proposed_trials(root, context).get("trials", {}).get("r1", [])
    assert len(proposed) == (3 if n_trials == 3 else 0)
    if proposed:
        assert all(item["proposal_source"] == "AUTO_STRUCTURAL_CANDIDATE"
                   for item in proposed)
        assert [(item["start_sec"], item["end_sec"]) for item in proposed] == [
            (.5, 1.5), (3., 4.), (5.5, 6.5)]


def test_manual_correction_removes_structural_auto_acceptance(tmp_path):
    root, prompt, speakers, trials = _fixture(tmp_path, 3)
    raw = json.loads(prompt.read_text(encoding="utf-8"))
    raw["expected_repetitions"] = 3
    prompt.write_text(json.dumps(raw), encoding="utf-8")
    run_acoustic_alignment(root, AlignmentConfig(source="mfa",
        prompt_manifest_path=str(prompt), speaker_manifest_path=str(speakers),
        trial_manifest_path=str(trials), provider=FakeCorpusProvider(),
        structural_auto_review=True))
    run_id = list_alignment_runs(root)[-1]["alignment_run_id"]
    word = pd.read_csv(root / "acoustic" / "004_alignment" / "runs" / run_id /
                       "tables" / "alignment_words.csv").iloc[0]
    set_correction(root, run_id, "r1", "trial_001", "word", int(word.word_index),
                   float(word.start_sec) - .01, float(word.end_sec))
    review = trial_review_table(root, run_id).set_index("trial_id")
    assert review.loc["trial_001", "review_status"] == "NEEDS_REVIEW"
    assert review.loc["trial_001", "review_mode"] == "HUMAN_CORRECTION_PENDING"
    set_trial_review(root, run_id, "r1", "trial_001", "ACCEPTED")
    freeze_alignment(root, run_id)
    final = pd.read_csv(root / "acoustic" / "004_alignment" / "final" /
                        "final_alignment_trial_review.csv").set_index("trial_id")
    assert final.loc["trial_001", "final_status"] == "ACCEPTED_MANUAL"
    assert final.loc["trial_002", "final_status"] == "AUTO_ACCEPTED_STRUCTURAL"


@pytest.mark.parametrize("outside_sec,should_freeze", [(9e-6, True), (.001, False)])
def test_freeze_trial_boundary_uses_working_sample_tolerance_and_identifies_token(
        tmp_path, outside_sec, should_freeze):
    root, prompt, speakers, trials = _fixture(tmp_path, 3)
    run_acoustic_alignment(root, AlignmentConfig(source="mfa",
        prompt_manifest_path=str(prompt), speaker_manifest_path=str(speakers),
        trial_manifest_path=str(trials), provider=FakeCorpusProvider()))
    run_id = list_alignment_runs(root)[-1]["alignment_run_id"]
    run = root / "acoustic" / "004_alignment" / "runs" / run_id
    words_path = run / "tables" / "alignment_words.csv"
    words = pd.read_csv(words_path)
    words.loc[0, "start_sec"] = .5 - outside_sec
    words.to_csv(words_path, index=False)
    manifest_path = run / "logs" / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["words_sha256"] = hashlib.sha256(words_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for number in (1, 2, 3):
        set_trial_review(root, run_id, "r1", f"trial_{number:03d}", "ACCEPTED")
    if should_freeze:
        freeze_alignment(root, run_id)
        assert load_final_alignment(root)[1] == ""
    else:
        with pytest.raises(ValueError, match="trial_token_boundary_mismatch") as error:
            freeze_alignment(root, run_id)
        detail = json.loads(str(error.value).split(":", 1)[1])
        assert detail["recording_id"] == "r1"
        assert detail["trial_id"] == "trial_001"
        assert detail["tier"] == "word"
        assert detail["token_index"] == 1
        assert detail["trial_start_sec"] == .5
        assert detail["source"]


def test_repeated_trial_viewer_navigation_playback_and_review_persist(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget

    QApplication.instance() or QApplication([])
    root, prompt, speakers, trials = _fixture(tmp_path)
    run_acoustic_alignment(root, AlignmentConfig(source="mfa",
        prompt_manifest_path=str(prompt), speaker_manifest_path=str(speakers),
        trial_manifest_path=str(trials), provider=FakeCorpusProvider()))
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    assert widget.trials.count() == 3
    assert widget.recordings.count() == 1
    assert "UNREVIEWED" in widget.normal_summary.text()
    widget._step_trial(1)
    assert widget.trials.currentData()["trial_id"] == "trial_002"
    widget._play_pause()
    assert widget._play_end_ms == 4000
    widget.player.pause()
    widget._set_review("ACCEPTED")
    assert widget.trials.currentData()["trial_id"] == "trial_003"
    assert "1 / 3 reviewed" in widget.review_progress.text()
    widget.close()
    reopened = AlignmentWidget(lambda: root)
    reopened.refresh()
    reopened.trials.setCurrentIndex(1)
    assert "ACCEPTED" in reopened.normal_summary.text()
    assert not reopened.freeze_button.isEnabled()
    reopened.close()


def test_four_step_gui_confirms_proposals_runs_and_requires_review(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget
    from vslp.acoustic.alignment.task_workflow import load_project_choices

    QApplication.instance() or QApplication([])
    root, prompt, speakers, trials = _fixture(tmp_path)
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs" / "alignment_choices.json").write_text(
        json.dumps({"schema_version": "1", "recordings": {},
                    "manual_trial_review_required": True}), encoding="utf-8")
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    assert widget._view_step == "trials"
    assert len(widget.proposal_regions) == 3
    assert widget.primary.text() == "CONFIRM TRIALS"
    widget._select_proposal(1)
    widget._play_pause()
    assert widget._play_end_ms == 4000
    widget.player.pause()
    widget._select_proposal(2)
    widget._omit_proposal()
    assert len(widget.proposal_regions) == 2
    widget._select_proposal(0)
    widget._proposal_cursor = 1.0
    widget._split_proposal()
    assert len(widget.proposal_regions) == 3
    assert not widget.advanced.isVisible()
    assert "technical_" not in widget.action_hint.text()
    widget._primary_action()
    assert len(load_project_choices(root)["trials"]["r1"]) == 3
    widget.show_preflight_result({"status": "READY"})
    assert widget._view_step == "run"
    assert widget.primary.text() == "RUN ALIGNMENT"
    widget.alignment_started()
    assert widget._view_step == "running"
    assert widget.run_progress.maximum() == 0
    widget.alignment_progress(0, 0, "Running MFA corpus alignment...")
    assert "ALIGNMENT RUNNING" in widget.action_hint.text()
    widget.alignment_finished()
    widget.close()

    run_acoustic_alignment(root, AlignmentConfig(source="mfa",
        prompt_manifest_path=str(prompt), speaker_manifest_path=str(speakers),
        trial_manifest_path=str(trials), provider=FakeCorpusProvider()))
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    widget.show()
    assert widget._view_step == "review"
    assert not widget.primary.text().startswith("FREEZE")
    widget._set_review("NEEDS_REVIEW")
    assert "1 needs review" in widget.review_progress.text()
    assert widget.trials.currentData()["trial_id"] == "trial_002"
    assert "flagged cases await review" in widget.action_hint.text()
    widget.focus_trial("r1", "trial_001")
    assert not widget.keep_needs_review_button.isHidden()
    run_id = widget.runs.currentData()
    for trial_id in ("trial_001", "trial_002", "trial_003"):
        set_trial_review(root, run_id, "r1", trial_id, "ACCEPTED")
    widget.refresh()
    widget._show_step()
    assert widget._view_step == "freeze"
    assert widget.primary.text() == "FREEZE ALIGNMENT"
    widget.close()


def test_five_fragments_merge_to_three_visual_trials_then_align_and_freeze(
        tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from vslp.acoustic.alignment.task_workflow import (
        load_project_choices, project_alignment_context,
    )
    from vslp.acoustic.alignment.trials import trial_manifest_for_alignment
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget

    QApplication.instance() or QApplication([])
    root, prompt, speakers, _old_trials = _fixture(tmp_path)
    review = root / "acoustic" / "003_segmentation_review" / "final"
    pd.DataFrame([{"recording_id": "r1", "segment_role": "speech",
                   "start_sec": left, "end_sec": right}
                  for left, right in ((.5, 1.5), (3, 4), (5.5, 5.8),
                                      (5.8, 6.1), (6.1, 6.5))]).to_csv(
                                          review / "final_segmentation_intervals.csv", index=False)
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    widget.show()
    QApplication.processEvents()
    assert len(widget.proposal_regions) == 5
    assert "Expected repetitions: 3" in widget.trial_count.text()
    assert "Current trials: 5" in widget.trial_count.text()
    for when, modifiers in ((5.65, Qt.NoModifier), (5.95, Qt.ControlModifier)):
        scene = widget.plot.plotItem.vb.mapViewToScene(pg.Point(when, 0))
        QTest.mouseClick(widget.plot.viewport(), Qt.LeftButton, modifiers,
                         widget.plot.mapFromScene(scene))
    assert widget._selected_proposals == {2, 3}
    assert widget.merge_button.isEnabled()
    widget._merge_proposals()
    assert len(widget.proposal_regions) == 4
    widget._select_proposal(2)
    widget._select_proposal(3, additive=True)
    widget._merge_proposals()
    assert len(widget.proposal_regions) == 3
    assert [label.textItem.toPlainText() for label in widget.proposal_labels] == [
        "Trial 1", "Trial 2", "Trial 3"]
    region = widget.proposal_regions[2]
    region.setRegion((5.5, 6.49))
    widget._proposal_region_changed(region)
    assert widget._current_proposal_bounds()[2] == pytest.approx((5.5, 6.49))
    assert "Current trials: 3" in widget.trial_count.text()
    assert widget.primary.text() == "CONFIRM TRIALS"
    widget._confirm_proposals()
    confirmed = load_project_choices(root)["trials"]["r1"]
    assert len(confirmed) == 3
    widget.show_preflight_result({"status": "READY"})
    assert widget.primary.text() == "RUN ALIGNMENT"
    widget.close()


    provider = FakeCorpusProvider()
    manifest = trial_manifest_for_alignment(root, project_alignment_context(root))
    run_acoustic_alignment(root, AlignmentConfig(source="mfa",
        prompt_manifest_path=str(prompt), speaker_manifest_path=str(speakers),
        trial_manifest_path=str(manifest), provider=provider))
    assert len(provider.aligned_units) == 3
    run_id = list_alignment_runs(root)[-1]["alignment_run_id"]
    words = pd.read_csv(root / "acoustic" / "004_alignment" / "runs" / run_id /
                        "tables" / "alignment_words.csv")
    assert len(words) == 12
    for number in range(1, 4):
        assert len(words.loc[words.sequence_id.between((number - 1) * 100000,
                                                     number * 100000 - 1)]) == 4
        set_trial_review(root, run_id, "r1", f"trial_{number:03d}", "ACCEPTED")
    freeze_alignment(root, run_id)
    store, issue = load_final_alignment(root)
    assert issue == "" and store is not None
    assert len(store.words) == 12


def test_merge_enabled_for_one_selected_trial(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget

    QApplication.instance() or QApplication([])
    root, _prompt, _speakers, _old_trials = _fixture(tmp_path)
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs" / "alignment_choices.json").write_text(
        json.dumps({"schema_version": "1", "recordings": {},
                    "manual_trial_review_required": True}), encoding="utf-8")
    review = root / "acoustic" / "003_segmentation_review" / "final"
    pd.DataFrame([{"recording_id": "r1", "segment_role": "speech",
                   "start_sec": left, "end_sec": right}
                  for left, right in ((.5, 1.5), (3, 4), (5.5, 6.0))]).to_csv(
                      review / "final_segmentation_intervals.csv", index=False)
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    widget._select_proposal(1)
    assert widget.merge_button.isEnabled()
    widget._merge_proposals()
    assert widget._current_proposal_bounds() == [(0.5, 1.5), (3.0, 6.0)]
    widget.close()


def test_visual_trial_editor_delete_split_add_undo_and_exclusion(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import pyqtgraph as pg
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    from vslp.acoustic.alignment.trials import validate_trial_draft
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget

    QApplication.instance() or QApplication([])
    root, _prompt, _speakers, _trials = _fixture(tmp_path)
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    widget.show()
    QApplication.processEvents()
    assert len(widget.proposal_regions) == 3
    widget._select_proposal(2)
    widget._omit_proposal()
    assert len(widget.proposal_regions) == 2
    widget._undo_proposal()
    assert len(widget.proposal_regions) == 3
    widget._select_proposal(2)
    widget._proposal_cursor = 6.0
    widget._split_proposal()
    assert len(widget.proposal_regions) == 4
    widget._undo_proposal()
    assert len(widget.proposal_regions) == 3
    widget._begin_add_trial()
    assert widget._add_mode and widget.add_button.text() == "DRAG ON WAVEFORM…"
    for event_type, when in ((QEvent.MouseButtonPress, 7.0),
                             (QEvent.MouseMove, 7.4),
                             (QEvent.MouseButtonRelease, 7.4)):
        scene = widget.plot.plotItem.vb.mapViewToScene(pg.Point(when, 0))
        point = QPointF(widget.plot.mapFromScene(scene))
        event = QMouseEvent(event_type, point, Qt.LeftButton, Qt.LeftButton,
                            Qt.NoModifier)
        assert widget.eventFilter(widget.plot.viewport(), event)
    assert len(widget.proposal_regions) == 4
    widget._undo_proposal()
    assert len(widget.proposal_regions) == 3
    widget.close()

    review = root / "acoustic" / "003_segmentation_review" / "final"
    intervals = pd.read_csv(review / "final_segmentation_intervals.csv")
    intervals.loc[len(intervals)] = ["r1", "manual_exclusion", 5.95, 6.05]
    intervals.to_csv(review / "final_segmentation_intervals.csv", index=False)
    with pytest.raises(ValueError, match="TRIAL_CROSSES_EXCLUSION"):
        validate_trial_draft(root, "r1", [(5.5, 6.5)])
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    assert not widget._apply_proposal_bounds([(5.5, 6.5)], {0})
    assert len(widget.proposal_regions) == 3
    widget.close()


def test_trial_boundary_handle_drag_is_undoable(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget

    QApplication.instance() or QApplication([])
    root, _prompt, _speakers, _trials = _fixture(tmp_path)
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    widget.show()
    QApplication.processEvents()
    before = widget._current_proposal_bounds()
    assert len(widget.proposal_regions[2].lines) == 2
    start = widget.plot.mapFromScene(
        widget.plot.plotItem.vb.mapViewToScene(pg.Point(5.5, 0)))
    end = widget.plot.mapFromScene(
        widget.plot.plotItem.vb.mapViewToScene(pg.Point(5.45, 0)))
    QTest.mousePress(widget.plot.viewport(), Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(widget.plot.viewport(), end)
    QTest.mouseRelease(widget.plot.viewport(), Qt.LeftButton, Qt.NoModifier, end)
    assert widget._current_proposal_bounds()[2][0] < before[2][0]
    widget._undo_proposal()
    assert widget._current_proposal_bounds() == pytest.approx(before)
    widget.close()


def test_new_reviewed_run_can_supersede_frozen_legacy_without_losing_archive(tmp_path):
    root, prompt, speakers, trials = _fixture(tmp_path, 2)
    config = AlignmentConfig(source="mfa", prompt_manifest_path=str(prompt),
        speaker_manifest_path=str(speakers), trial_manifest_path=str(trials),
        provider=FakeCorpusProvider())
    run_acoustic_alignment(root, config)
    first = list_alignment_runs(root)[-1]["alignment_run_id"]
    for trial in ("trial_001", "trial_002"):
        set_trial_review(root, first, "r1", trial, "ACCEPTED")
    freeze_alignment(root, first)
    run_acoustic_alignment(root, config)
    second = list_alignment_runs(root)[-1]["alignment_run_id"]
    for trial in ("trial_001", "trial_002"):
        set_trial_review(root, second, "r1", trial, "ACCEPTED")
    with pytest.raises(FileExistsError, match="immutable"):
        freeze_alignment(root, second)
    freeze_alignment(root, second, supersede=True)
    store, issue = load_final_alignment(root)
    assert issue == "" and store is not None
    assert store.manifest["alignment_run_id"] == second
    assert store.manifest["supersedes_alignment_run_id"] == first
    archive = root / "acoustic" / "004_alignment" / "archived_final"
    assert len(list(archive.glob("*/final_alignment_manifest.json"))) == 1
    assert len(pd.read_csv(next(archive.glob("*/final_alignment_words.csv")))) == 8


def test_manual_word_and_phone_corrections_keep_mfa_immutable_and_freeze(tmp_path):
    root, prompt, speakers, trials = _fixture(tmp_path)
    run_acoustic_alignment(root, AlignmentConfig(source="mfa",
        prompt_manifest_path=str(prompt), speaker_manifest_path=str(speakers),
        trial_manifest_path=str(trials), provider=FakeCorpusProvider()))
    run_id = list_alignment_runs(root)[0]["alignment_run_id"]
    base = root / "acoustic" / "004_alignment" / "runs" / run_id / "tables"
    original_words = pd.read_csv(base / "alignment_words.csv")
    original_phones = pd.read_csv(base / "alignment_phones.csv")
    word = original_words.iloc[0]
    phone = original_phones.iloc[0]
    set_correction(root, run_id, "r1", "trial_001", "word", int(word.word_index),
                   float(word.start_sec) - .01, float(word.end_sec), "reviewer timing")
    set_correction(root, run_id, "r1", "trial_001", "phone", int(phone.phone_index),
                   float(phone.start_sec) + .01, float(phone.end_sec), "reviewer timing")
    assert len(load_corrections(root, run_id)) == 2
    reset_corrections(root, run_id, "r1", "trial_001", "phone", int(phone.phone_index))
    assert len(load_corrections(root, run_id)) == 1
    set_correction(root, run_id, "r1", "trial_001", "phone", int(phone.phone_index),
                   float(phone.start_sec) + .01, float(phone.end_sec), "reviewer timing")
    set_trial_review(root, run_id, "r1", "trial_001", "ACCEPTED")
    reset_corrections(root, run_id, "r1", "trial_001", "phone", int(phone.phone_index))
    assert trial_review_table(root, run_id).iloc[0].review_status == "NEEDS_REVIEW"
    set_correction(root, run_id, "r1", "trial_001", "phone", int(phone.phone_index),
                   float(phone.start_sec) + .01, float(phone.end_sec), "reviewer timing")
    set_trial_review(root, run_id, "r1", "trial_001", "ACCEPTED")
    for number in (2, 3):
        set_trial_review(root, run_id, "r1", f"trial_{number:03d}", "ACCEPTED")
    freeze_alignment(root, run_id)
    store, issue = load_final_alignment(root)
    assert issue == "" and store is not None
    assert store.words.iloc[0].start_sec == pytest.approx(float(word.start_sec) - .01)
    assert store.phones.iloc[0].start_sec == pytest.approx(float(phone.start_sec) + .01)
    assert store.words.iloc[0].start_sample == round(store.words.iloc[0].start_sec * 16000)
    assert store.words.iloc[0].alignment_source == "reviewed_manual_correction"
    assert pd.read_csv(base / "alignment_words.csv").equals(original_words)
    assert pd.read_csv(base / "alignment_phones.csv").equals(original_phones)
    final_reviews = pd.read_csv(store.manifest["final_trial_review_path"])
    assert final_reviews.final_status.tolist() == ["ACCEPTED_MANUAL", "ACCEPTED_MFA", "ACCEPTED_MFA"]


def test_stale_trial_cannot_be_accepted_or_frozen(tmp_path):
    root, prompt, speakers, trials = _fixture(tmp_path)
    run_acoustic_alignment(root, AlignmentConfig(source="mfa",
        prompt_manifest_path=str(prompt), speaker_manifest_path=str(speakers),
        trial_manifest_path=str(trials), provider=FakeCorpusProvider()))
    run_id = list_alignment_runs(root)[0]["alignment_run_id"]
    mark_recording_stale(root, run_id, "r1")
    assert trial_review_table(root, run_id).review_status.eq("STALE_AFTER_TRIAL_EDIT").all()
    with pytest.raises(ValueError, match="stale_alignment_requires_rerun"):
        set_trial_review(root, run_id, "r1", "trial_001", "ACCEPTED")
    with pytest.raises(ValueError, match="all_alignment_trials_must_be_accepted"):
        freeze_alignment(root, run_id)


def test_targeted_recording_rerun_preserves_unrelated_review(tmp_path):
    root, prompt, speakers, trials_path = _fixture(tmp_path)
    review = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = pd.read_csv(review / "final_segmentation_decisions.csv")
    decisions.loc[len(decisions)] = {**decisions.iloc[0].to_dict(),
                                    "recording_id": "r2", "file_name": "synthetic_2.wav"}
    (root / "synthetic_2.wav").write_bytes((root / "synthetic.wav").read_bytes())
    decisions.loc[1, "analysis_wav_path"] = str(root / "synthetic_2.wav")
    decisions.to_csv(review / "final_segmentation_decisions.csv", index=False)
    intervals = pd.read_csv(review / "final_segmentation_intervals.csv")
    second = intervals.copy()
    second["recording_id"] = "r2"
    pd.concat([intervals, second]).to_csv(review / "final_segmentation_intervals.csv", index=False)
    speaker_data = json.loads(speakers.read_text(encoding="utf-8"))
    speaker_data["recordings"].append({"recording_id": "r2", "speaker_id": "technical_r2"})
    speakers.write_text(json.dumps(speaker_data), encoding="utf-8")
    trial_data = json.loads(trials_path.read_text(encoding="utf-8"))
    trial_data["trials"] += [{**item, "recording_id": "r2"} for item in trial_data["trials"]]
    trials_path.write_text(json.dumps(trial_data), encoding="utf-8")
    config = AlignmentConfig(source="mfa", prompt_manifest_path=str(prompt),
        speaker_manifest_path=str(speakers), trial_manifest_path=str(trials_path),
        provider=FakeCorpusProvider())
    run_acoustic_alignment(root, config)
    first = list_alignment_runs(root)[-1]["alignment_run_id"]
    for trial in ("trial_001", "trial_002", "trial_003"):
        set_trial_review(root, first, "r2", trial, "ACCEPTED")
    original = pd.read_csv(root / "acoustic" / "004_alignment" / "runs" / first /
                           "tables" / "alignment_words.csv")
    trial_data["trials"] = [item for item in trial_data["trials"]
                            if item["recording_id"] != "r1" or item["trial_id"] != "trial_003"]
    trial_data["trials"].append({"recording_id": "r1", "trial_id": "trial_003",
        "prompt_id": "wstg_we_see_three_geese", "start_sec": 5.4, "end_sec": 6.5})
    trials_path.write_text(json.dumps(trial_data), encoding="utf-8")
    mark_recording_stale(root, first, "r1")
    from dataclasses import replace
    rerun_provider = FakeCorpusProvider()
    run_acoustic_alignment(root, replace(config, provider=rerun_provider,
                                         rerun_recording_id="r1", reuse_run_id=first))
    assert len(rerun_provider.aligned_units) == 3
    second_run = list_alignment_runs(root)[-1]["alignment_run_id"]
    assert second_run != first
    second_base = root / "acoustic" / "004_alignment" / "runs" / second_run
    current = pd.read_csv(second_base / "tables" / "alignment_words.csv")
    assert len(current) == len(original)
    assert current.loc[current.recording_id.eq("r2"), "start_sec"].tolist() == original.loc[
        original.recording_id.eq("r2"), "start_sec"].tolist()
    reviews = trial_review_table(root, second_run)
    assert reviews.loc[reviews.recording_id.eq("r2"), "review_status"].eq("ACCEPTED").all()
    assert reviews.loc[reviews.recording_id.eq("r1"), "review_status"].eq("UNREVIEWED").all()
    manifest = json.loads((second_base / "logs" / "stage_manifest.json").read_text(encoding="utf-8"))
    assert manifest["reused_alignment_run_id"] == first
    assert manifest["rerun_recording_id"] == "r1"
    from vslp.acoustic.alignment.task_workflow import load_project_choices, save_project_choices
    choices = load_project_choices(root)
    choices["selected_run_id"] = second_run
    save_project_choices(root, choices)
    from PySide6.QtWidgets import QApplication
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget
    QApplication.instance() or QApplication([])
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    assert widget.runs.currentData() == second_run
    widget.runs.setCurrentIndex(widget.runs.findData(first))
    choices = load_project_choices(root)
    choices["selected_run_id"] = second_run
    save_project_choices(root, choices)
    widget.refresh()
    assert widget.runs.currentData() == second_run
    widget.close()


def test_gui_repair_wrong_trial_count_marks_old_run_stale(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget
    from vslp.acoustic.alignment.task_workflow import load_project_choices, save_project_choices

    QApplication.instance() or QApplication([])
    root, prompt, speakers, trials_path = _fixture(tmp_path)
    choices = load_project_choices(root)
    trials = json.loads(trials_path.read_text(encoding="utf-8"))["trials"]
    fragments = [{**trials[-1], "trial_id": f"trial_{number:03d}",
                  "start_sec": left, "end_sec": right}
                 for number, left, right in ((3, 5.5, 5.8), (4, 5.8, 6.1), (5, 6.1, 6.5))]
    choices["trials"] = {"r1": trials[:2] + fragments}
    save_project_choices(root, choices)
    trials_path.write_text(json.dumps({"schema_version": "1", "trials": trials[:2] + fragments}),
                           encoding="utf-8")
    run_acoustic_alignment(root, AlignmentConfig(source="mfa",
        prompt_manifest_path=str(prompt), speaker_manifest_path=str(speakers),
        trial_manifest_path=str(trials_path), provider=FakeCorpusProvider()))
    run_id = list_alignment_runs(root)[-1]["alignment_run_id"]
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    assert "Trial count differs" not in widget.action_hint.text()  # historical run is reviewable
    widget._edit_trial_boundaries()
    assert widget._view_step == "trials"
    assert len(widget.proposal_regions) == 5
    widget._select_proposal(4)
    widget._omit_proposal()
    widget._select_proposal(3)
    widget._omit_proposal()
    assert len(widget.proposal_regions) == 3
    # Restore the intended third sentence region by dragging its remaining handles.
    widget.proposal_regions[2].setRegion((5.5, 6.5))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)
    widget._confirm_proposals()
    assert widget._repair_ready, widget.action_hint.text()
    assert widget.primary.text() == "RERUN ALIGNMENT FOR THIS RECORDING"
    assert len(load_project_choices(root)["trials"]["r1"]) == 3
    assert trial_review_table(root, run_id).review_status.eq("STALE_AFTER_TRIAL_EDIT").all()
    widget.close()


def test_trial_count_mismatch_visible_before_run(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget
    from vslp.acoustic.alignment.task_workflow import load_project_choices, save_project_choices

    QApplication.instance() or QApplication([])
    root, _prompt, _speakers, trials_path = _fixture(tmp_path)
    original = json.loads(trials_path.read_text(encoding="utf-8"))["trials"]
    extra = [{**original[-1], "trial_id": "trial_004", "start_sec": 6.5, "end_sec": 6.8},
             {**original[-1], "trial_id": "trial_005", "start_sec": 6.8, "end_sec": 7.0}]
    choices = load_project_choices(root)
    choices["trials"] = {"r1": original + extra}
    save_project_choices(root, choices)
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    assert "expected 3, confirmed 5" in widget.action_hint.text()
    assert widget.primary.text() == "CONFIRM TRIALS"
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)
    widget._primary_action()
    assert load_project_choices(root)["trial_count_ack"]["r1"] == 5
    widget.close()


def test_gui_manual_boundary_edit_persists_across_reopen(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    import pyqtgraph as pg
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget

    QApplication.instance() or QApplication([])
    root, prompt, speakers, trials = _fixture(tmp_path)
    run_acoustic_alignment(root, AlignmentConfig(source="mfa",
        prompt_manifest_path=str(prompt), speaker_manifest_path=str(speakers),
        trial_manifest_path=str(trials), provider=FakeCorpusProvider()))
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    run_id = widget.runs.currentData()
    widget._toggle_alignment_edit()
    words = pd.read_csv(root / "acoustic" / "004_alignment" / "runs" / run_id /
                        "tables" / "alignment_words.csv")
    first = words.iloc[0]
    region = pg.LinearRegionItem([float(first.start_sec) - .01, float(first.end_sec)])
    widget._save_token_region("word", int(first.word_index), "trial_001", region)
    assert len(load_corrections(root, run_id)) == 1
    widget.close()
    reopened = AlignmentWidget(lambda: root)
    reopened.refresh()
    assert "Manually corrected" in [item[4] for item in reopened._tokens]
    reopened._reset_current_trial()
    assert load_corrections(root, run_id).empty
    reopened.close()


def test_frozen_trial_repair_creates_pending_revision_without_mutating_final(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget
    from vslp.acoustic.alignment.task_workflow import load_project_choices

    QApplication.instance() or QApplication([])
    root, prompt, speakers, trials = _fixture(tmp_path)
    run_acoustic_alignment(root, AlignmentConfig(source="mfa",
        prompt_manifest_path=str(prompt), speaker_manifest_path=str(speakers),
        trial_manifest_path=str(trials), provider=FakeCorpusProvider()))
    first = list_alignment_runs(root)[-1]["alignment_run_id"]
    for number in (1, 2, 3):
        set_trial_review(root, first, "r1", f"trial_{number:03d}", "ACCEPTED")
    freeze_alignment(root, first)
    widget = AlignmentWidget(lambda: root)
    widget.refresh()
    assert widget.primary.text() == "ALIGNMENT FROZEN"
    widget._edit_trial_boundaries()
    widget.proposal_regions[2].setRegion((5.4, 6.5))
    widget._confirm_proposals()
    assert widget._repair_ready
    assert load_project_choices(root)["pending_trial_revision"]["r1"]["source_run_id"] == first
    store, issue = load_final_alignment(root)
    assert store is None and issue == "stale_alignment_trial_revision"
    archived_store, archived_issue = load_final_alignment(root, require_current_trials=False)
    assert archived_issue == "" and archived_store.manifest["alignment_run_id"] == first
    widget.close()
