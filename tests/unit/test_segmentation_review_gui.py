import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from PySide6.QtWidgets import QApplication, QFileDialog

from vslp.acoustic.segment.review import load_review_state, save_segmentation_review_entry
from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow
from vslp.gui.acoustic_app.review_widget import SegmentationReviewWidget


def test_review_gui_filters_and_persists_after_widget_restart(tmp_path: Path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    wav = tmp_path / "x.wav"
    sf.write(wav, np.zeros(16000, dtype="float32"), 16000, subtype="FLOAT")
    segments = tmp_path / "auto.csv"
    pd.DataFrame([{"segment_type": "speech", "start_sec": 0.1, "end_sec": 0.9,
                   "duration_sec": 0.8}]).to_csv(segments, index=False)
    table = tmp_path / "acoustic" / "002_segmentation" / "tables" / "acoustic_segmentation_summary.csv"
    table.parent.mkdir(parents=True)
    pd.DataFrame([
        {"recording_id": "accepted", "file_name": "a.wav", "task_name": "Bamboo Passage",
         "segmentation_method": "silero_vad", "automatic_status": "ACCEPTED",
         "review_required": False, "duration_sec": 1, "analysis_wav_path": str(wav),
         "segments_csv_path": str(segments)},
        {"recording_id": "review", "file_name": "b.wav", "task_name": "Bamboo Passage",
         "segmentation_method": "silero_vad", "automatic_status": "REVIEW",
         "review_required": True, "duration_sec": 1, "analysis_wav_path": str(wav),
         "segments_csv_path": str(segments)},
    ]).to_csv(table, index=False)
    widget = SegmentationReviewWidget(lambda: tmp_path)
    assert not hasattr(widget, "plot_label")  # Review uses the live pyqtgraph editor.
    widget.refresh()
    assert widget.recording_list.count() == 1  # Pending review is default.
    widget.filter_combo.setCurrentText("All")
    assert widget.recording_list.count() == 2
    save_segmentation_review_entry(tmp_path, "review", "KEEP_AUTO", "Reviewer", "Remove cough",
        analysis_start_sec=.05, analysis_end_sec=.95,
        exclusion_intervals=[{"start_sec": .4, "end_sec": .5,
            "exclusion_reason": "Cough / throat clear", "notes": "Cough"}])
    widget.close()
    restarted = SegmentationReviewWidget(lambda: tmp_path)
    restarted.refresh()
    assert restarted.recording_list.count() == 0
    restarted.filter_combo.setCurrentText("Kept automatic after review")
    assert restarted.recording_list.count() == 1
    assert restarted.editor.analysis_start_sec == .05
    assert restarted.editor.analysis_end_sec == .95
    assert restarted.editor.exclusions[0]["exclusion_reason"] == "Cough / throat clear"
    restarted._enter_edit_mode()
    restarted.editor._regions[0].setRegion((.2, .8))
    restarted.editor.set_selection(.55, .6)
    restarted.notes_edit.setPlainText("Adjusted speech and cough")
    restarted._exclude_selection()
    restarted._save("KEEP_MANUAL")
    decisions, overrides = load_review_state(tmp_path)
    assert decisions.set_index("recording_id").loc["review", "final_decision"] == "KEEP_MANUAL"
    assert [(float(r.start_sec), float(r.end_sec)) for r in overrides.itertuples()] == [(.2, .8)]
    restarted.close()
    assert app is not None


def test_acoustic_gui_has_separate_review_stage_and_final_paths():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = AcousticPipelineWindow()
    tabs = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert tabs.index("Segmentation manual review") == tabs.index("Segmentation") + 1
    assert tabs.index("Quality Control") == tabs.index("Segmentation manual review") + 1
    assert "review" in window.stage_records
    window.close()
    assert app is not None


def test_reopen_existing_run_restores_manual_review(tmp_path: Path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "acoustic" / "002_segmentation" / "tables" / "acoustic_segmentation_summary.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame([{"recording_id": "r", "file_name": "r.wav", "task_name": "Bamboo Passage",
                   "segmentation_method": "silero_vad", "automatic_status": "REVIEW",
                   "review_required": True, "duration_sec": 1}]).to_csv(source, index=False)
    (tmp_path / "project_manifest.json").write_text(json.dumps({
        "modality": "acoustic", "run_root": str(tmp_path.resolve()),
        "project_name": "P", "task_name": "Bamboo Passage",
        "input_folder": str(tmp_path / "input"), "output_parent": str(tmp_path.parent),
    }), encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: str(tmp_path))
    window = AcousticPipelineWindow()
    window.open_existing_run()
    assert window._run_root == tmp_path.resolve()
    assert window.task_name_edit.text() == "Bamboo Passage"
    assert window.task_name_edit.isReadOnly()
    assert window.review_widget.recording_list.count() == 1
    window.close()
    assert app is not None
