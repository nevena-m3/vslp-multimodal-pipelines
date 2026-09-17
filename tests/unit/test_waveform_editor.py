import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from matplotlib.image import imread
from matplotlib.axes import Axes
from PySide6.QtWidgets import QApplication

from vslp.acoustic.segment.pipeline import _plot
from vslp.acoustic.segment.silero_reference import Interval
from vslp.acoustic.features.plugins.audio_utils import mask_from_segments
from vslp.gui.acoustic_app.waveform_editor import WaveformEditor, display_waveform, playback_bounds


def test_display_decimation_preserves_peaks_without_setting_boundary_grid():
    x = np.zeros(240_000, dtype="float32")
    x[12345] = 1
    times, values = display_waveform(x, 16000, max_points=1000)
    assert len(times) <= 1000
    assert values.max() == 1
    assert playback_bounds(.123456, 15) == (123, None)


def test_playback_selection_bounds_and_rejection():
    assert playback_bounds(2, 10, (1.25, 2.75)) == (1250, 2750)
    with pytest.raises(ValueError):
        playback_bounds(0, 10, (3, 2))
    with pytest.raises(ValueError):
        playback_bounds(0, 10, (0, 11))


def test_live_editor_add_delete_seek_window_and_selection(tmp_path: Path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    wav = tmp_path / "x.wav"
    sf.write(wav, np.zeros(48000, dtype="float32"), 16000, subtype="FLOAT")
    editor = WaveformEditor()
    editor.load_recording(wav, None, [(0.5, 1.0)], [(0.5, 1.0)])
    editor.set_editing(True)
    before_drag = []
    editor.edit_started.connect(before_drag.append)
    editor._regions[0].setRegion((.4, 1.1))
    assert before_drag[0] == [(0.5, 1.0)]
    editor.set_intervals(before_drag[0])
    editor.set_cursor(1.75)
    assert editor.cursor_sec == 1.75
    seeks = []
    editor.seek_requested.connect(seeks.append)
    editor.seek_requested.emit(1.25)
    assert seeks == [1.25]
    editor.set_selection(1.5, 2)
    editor.add_interval(*editor.selection)
    assert editor.intervals() == [(0.5, 1.0), (1.5, 2.0)]
    editor.delete_selected()
    assert editor.intervals() == [(0.5, 1.0)]
    editor.set_analysis_window(.25, 2.5)
    assert (editor.analysis_start_sec, editor.analysis_end_sec) == (.25, 2.5)
    with pytest.raises(ValueError):
        editor.add_interval(.75, 1.25)
    editor.close()
    assert app is not None


def test_static_plot_distinguishes_roles_and_review_overlays(tmp_path: Path, monkeypatch):
    path = tmp_path / "roles.png"
    labels = []
    original = Axes.legend
    def capture_legend(self, *args, **kwargs):
        labels.extend(handle.get_label() for handle in kwargs.get("handles", []))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Axes, "legend", capture_legend)
    _plot(path, np.zeros(4000, dtype="float32"), 1000, "silero_vad",
          [Interval(1, 1.5), Interval(2.5, 3)], "KEEP_MANUAL", ["reviewed"],
          automatic_intervals=[Interval(.8, 1.6), Interval(2.4, 3.1)],
          analysis_window=(.5, 3.5), excluded_intervals=[(1.8, 2.1)])
    pixels = imread(path)[..., :3]
    assert path.is_file()
    # Distinct role fills produce different RGB clusters in the saved artifact.
    samples = pixels.reshape(-1, 3)
    assert len(np.unique((samples * 15).astype(int), axis=0)) > 30
    assert {"Speech", "Leading nonspeech", "Internal nonspeech / pause",
            "Trailing nonspeech", "Automatic boundary", "Manual boundary",
            "Outside analysis window", "Manual exclusion"}.issubset(labels)


def test_full_file_policy_respects_reviewed_exclusions():
    segments = pd.DataFrame([
        {"segment_type": "outside_analysis", "segment_role": "outside_analysis_window", "start_sec": 0, "end_sec": 1},
        {"segment_type": "speech", "segment_role": "speech", "start_sec": 1, "end_sec": 2},
        {"segment_type": "excluded", "segment_role": "manual_exclusion", "start_sec": 2, "end_sec": 2.5},
        {"segment_type": "nonspeech", "segment_role": "internal_nonspeech", "start_sec": 2.5, "end_sec": 3},
    ])
    mask = mask_from_segments(3000, 1000, segments, region="full_file")
    assert not mask[:1000].any()
    assert mask[1000:2000].all()
    assert not mask[2000:2500].any()
    assert mask[2500:3000].all()
