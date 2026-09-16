import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from PySide6.QtWidgets import QApplication

from vslp.acoustic.segment.review import save_segmentation_review_entry
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
    widget.refresh()
    assert widget.recording_list.count() == 1  # Pending review is default.
    widget.filter_combo.setCurrentText("All")
    assert widget.recording_list.count() == 2
    save_segmentation_review_entry(tmp_path, "review", "KEEP_AUTO", "Reviewer")
    widget.close()
    restarted = SegmentationReviewWidget(lambda: tmp_path)
    restarted.refresh()
    assert restarted.recording_list.count() == 0
    restarted.filter_combo.setCurrentText("Kept automatic after review")
    assert restarted.recording_list.count() == 1
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
