"""Live waveform and exact-time interval editor for acoustic review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from vslp.acoustic.segment.review import validate_analysis_window, validate_manual_interval_list
from vslp.acoustic.segment.stage import _read_canonical_audio


ROLE_COLORS = {
    "speech": (93, 182, 125, 65),
    "leading_nonspeech": (145, 180, 201, 70),
    "internal_nonspeech": (232, 172, 145, 80),
    "trailing_nonspeech": (177, 159, 201, 70),
    "outside_analysis_window": (132, 141, 151, 95),
    "manual_exclusion": (210, 128, 100, 105),
}


def display_waveform(x: np.ndarray, sample_rate: int, *, max_points: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """Min/max decimation for display only; review times stay in seconds."""
    data = np.asarray(x, dtype=np.float32)
    if len(data) <= max_points:
        return np.arange(len(data), dtype=float) / sample_rate, data
    block = max(1, int(np.ceil(len(data) / (max_points // 2))))
    padded = np.pad(data, (0, (-len(data)) % block), constant_values=np.nan)
    chunks = padded.reshape(-1, block)
    lo = np.nanmin(chunks, axis=1)
    hi = np.nanmax(chunks, axis=1)
    indices = np.arange(len(chunks), dtype=float) * block / sample_rate
    times = np.repeat(indices, 2)
    values = np.column_stack((lo, hi)).reshape(-1)
    return times, values


def playback_bounds(cursor_sec: float, duration_sec: float,
                    selection: tuple[float, float] | None = None) -> tuple[int, int | None]:
    """Convert selected exact times to media milliseconds without changing saved boundaries."""
    duration = float(duration_sec)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Recording duration must be positive")
    if selection is None:
        if not np.isfinite(cursor_sec):
            raise ValueError("Playback cursor must be finite")
        return round(max(0.0, min(duration, cursor_sec)) * 1000), None
    start, end = validate_analysis_window(selection[0], selection[1], duration)
    return round(start * 1000), round(end * 1000)


class ReviewViewBox(pg.ViewBox):
    time_selected = Signal(float, float)

    def mouseDragEvent(self, event, axis=None):
        if event.button() == Qt.LeftButton:
            a = float(self.mapSceneToView(event.buttonDownScenePos()).x())
            b = float(self.mapSceneToView(event.scenePos()).x())
            event.accept()
            if event.isFinish() and abs(b - a) > 0.005:
                self.time_selected.emit(min(a, b), max(a, b))
        else:
            super().mouseDragEvent(event, axis)


class WaveformEditor(QWidget):
    seek_requested = Signal(float)
    selection_changed = Signal(float, float)
    intervals_changed = Signal()
    edit_started = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.duration_sec = 0.0
        self.cursor_sec = 0.0
        self.analysis_start_sec = 0.0
        self.analysis_end_sec = 0.0
        self.selection: tuple[float, float] | None = None
        self.exclusions: list[dict] = []
        self.automatic_intervals: list[tuple[float, float]] = []
        self._regions: list[pg.LinearRegionItem] = []
        self._background: list[pg.LinearRegionItem] = []
        self._auto_lines: list[pg.InfiniteLine] = []
        self._selected_region = -1
        self._editing = False
        self._active_drag = False
        self._last_intervals: list[tuple[float, float]] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.view = ReviewViewBox(enableMenu=False)
        self.waveform = pg.PlotWidget(viewBox=self.view, background="w")
        self.waveform.setToolTip("Click to seek · drag to select · wheel to zoom · middle drag to pan · drag boundary lines to edit")
        self.waveform.setLabel("left", "Amplitude")
        self.waveform.showGrid(x=True, y=True, alpha=0.15)
        self.waveform.setMinimumHeight(300)
        self.support = pg.PlotWidget(background="w")
        self.support.setLabel("left", "RMS")
        self.support.setLabel("bottom", "Time", "s")
        self.support.setXLink(self.waveform)
        self.support.setMaximumHeight(145)
        self.support.setMinimumHeight(105)
        layout.addWidget(self.waveform, 4)
        layout.addWidget(self.support, 1)
        self.playback_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#253858", width=2))
        self.support_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#253858", width=1))
        self.waveform.addItem(self.playback_line)
        self.support.addItem(self.support_line)
        self.selection_region = pg.LinearRegionItem(values=(0, 0), movable=False,
            brush=pg.mkBrush(100, 120, 155, 40), pen=pg.mkPen("#657d9a"))
        self.selection_region.hide()
        self.waveform.addItem(self.selection_region)
        self.view.time_selected.connect(self.set_selection)
        self.waveform.scene().sigMouseClicked.connect(self._clicked)

    def load_recording(self, wav_path: Path, frame_path: Path | None,
                       automatic: list[tuple[float, float]],
                       reviewed: list[tuple[float, float]],
                       analysis_window: tuple[float, float] | None = None,
                       exclusions: list[dict] | None = None) -> None:
        x, sr = _read_canonical_audio(wav_path)
        self.duration_sec = len(x) / sr
        self.automatic_intervals = list(automatic)
        self.analysis_start_sec, self.analysis_end_sec = analysis_window or (0.0, self.duration_sec)
        self.exclusions = [dict(item) for item in (exclusions or [])]
        self._editing = False
        self.selection = None
        self.selection_region.hide()
        self._clear_regions()
        t, y = display_waveform(x, sr)
        self.waveform.plot(t, y, pen=pg.mkPen("#315a89", width=0.8), skipFiniteCheck=True)
        if frame_path and frame_path.is_file():
            frames = pd.read_csv(frame_path)
            if {"mid_sec", "rms"}.issubset(frames.columns):
                self.support.plot(frames.mid_sec.to_numpy(float), frames.rms.to_numpy(float),
                                  pen=pg.mkPen("#435a9b", width=1.2))
        else:
            hop = max(1, round(sr * .02))
            n = len(x) // hop
            if n:
                rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1))
                self.support.plot((np.arange(n) + .5) * hop / sr, rms,
                                  pen=pg.mkPen("#435a9b", width=1.2))
        self.set_intervals(reviewed)
        for start, end in automatic:
            for boundary in (start, end):
                line = pg.InfiniteLine(pos=boundary, angle=90, movable=False,
                                       pen=pg.mkPen("#27814f", width=1))
                line.setZValue(-2)
                self.waveform.addItem(line)
                self._auto_lines.append(line)
        self.set_cursor(0.0)
        self.waveform.setXRange(0, min(self.duration_sec, 20.0), padding=0)
        self.waveform.enableAutoRange(axis="y")

    def _clear_regions(self) -> None:
        for item in [*self._regions, *self._background, *self._auto_lines]:
            self.waveform.removeItem(item)
        self._regions.clear()
        self._background.clear()
        self._auto_lines.clear()
        self.waveform.clear()
        self.support.clear()
        self.waveform.addItem(self.playback_line)
        self.support.addItem(self.support_line)
        self.waveform.addItem(self.selection_region)

    def set_intervals(self, intervals: list[tuple[float, float]]) -> None:
        if intervals:
            validate_manual_interval_list(intervals, duration_sec=self.duration_sec)
        for region in self._regions:
            self.waveform.removeItem(region)
        self._regions = []
        self._selected_region = -1
        self._active_drag = False
        for index, (start, end) in enumerate(intervals):
            region = pg.LinearRegionItem(values=(start, end), movable=False,
                brush=pg.mkBrush(*ROLE_COLORS["speech"]),
                pen=pg.mkPen("#7735a4" if self._editing else "#27814f", width=2,
                             style=Qt.DashLine if self._editing else Qt.SolidLine))
            region.setBounds((0, self.duration_sec))
            # Drag boundary handles while a drag in the region body selects time.
            for line in region.lines:
                line.setMovable(self._editing)
            region.setZValue(-5)
            region.sigRegionChanged.connect(self._region_changing)
            region.sigRegionChangeFinished.connect(self._region_edited)
            self.waveform.addItem(region)
            self._regions.append(region)
        self._paint_roles()
        self._last_intervals = self.intervals()

    def intervals(self) -> list[tuple[float, float]]:
        return sorted((float(a), float(b)) for a, b in (region.getRegion() for region in self._regions))

    def _region_edited(self, region) -> None:
        self._active_drag = False
        self._selected_region = self._regions.index(region)
        self._paint_roles()
        self._last_intervals = self.intervals()
        self.intervals_changed.emit()

    def _region_changing(self, _region) -> None:
        if self._editing and not self._active_drag:
            self._active_drag = True
            self.edit_started.emit(self._last_intervals.copy())

    def set_editing(self, editing: bool) -> None:
        self._editing = editing
        self.set_intervals(self.intervals())

    def add_interval(self, start: float, end: float) -> None:
        candidate = [*self.intervals(), (start, end)]
        validate_manual_interval_list(candidate, duration_sec=self.duration_sec)
        self.set_intervals(candidate)
        self._selected_region = len(self._regions) - 1
        self.intervals_changed.emit()

    def delete_selected(self) -> None:
        if self._selected_region < 0:
            raise ValueError("Select a speech interval first")
        candidate = [item for index, item in enumerate(self.intervals()) if index != self._selected_region]
        if not candidate:
            raise ValueError("At least one speech interval is required; exclude the recording instead")
        self.set_intervals(candidate)
        self.intervals_changed.emit()

    def set_cursor(self, seconds: float) -> None:
        self.cursor_sec = max(0.0, min(self.duration_sec, float(seconds)))
        self.playback_line.setValue(self.cursor_sec)
        self.support_line.setValue(self.cursor_sec)

    def follow_cursor(self) -> None:
        left, right = self.waveform.viewRange()[0]
        if self.cursor_sec > right or self.cursor_sec < left:
            width = max(1.0, right - left)
            self.waveform.setXRange(max(0.0, self.cursor_sec - .2 * width),
                                    min(self.duration_sec, self.cursor_sec + .8 * width), padding=0)

    def set_selection(self, start: float, end: float) -> None:
        start, end = validate_analysis_window(max(0, start), min(self.duration_sec, end), self.duration_sec)
        self.selection = (start, end)
        self.selection_region.setRegion(self.selection)
        self.selection_region.show()
        self.selection_changed.emit(start, end)

    def clear_selection(self) -> None:
        self.selection = None
        self.selection_region.hide()

    def zoom_to_selection(self) -> None:
        if self.selection:
            self.waveform.setXRange(*self.selection, padding=.05)

    def set_analysis_window(self, start: float, end: float) -> None:
        self.analysis_start_sec, self.analysis_end_sec = validate_analysis_window(start, end, self.duration_sec)
        self._paint_roles()

    def set_exclusions(self, exclusions: list[dict]) -> None:
        self.exclusions = [dict(item) for item in exclusions]
        self._paint_roles()

    def _paint_roles(self) -> None:
        for item in self._background:
            self.waveform.removeItem(item)
        self._background = []
        intervals = [(max(a, self.analysis_start_sec), min(b, self.analysis_end_sec))
                     for a, b in self.intervals() if b > self.analysis_start_sec and a < self.analysis_end_sec]
        if not intervals:
            intervals = []
        spans = [(0, self.analysis_start_sec, "outside_analysis_window"),
                 (self.analysis_end_sec, self.duration_sec, "outside_analysis_window")]
        if intervals:
            spans.extend([(self.analysis_start_sec, intervals[0][0], "leading_nonspeech"),
                          (intervals[-1][1], self.analysis_end_sec, "trailing_nonspeech")])
        else:
            spans.append((self.analysis_start_sec, self.analysis_end_sec, "leading_nonspeech"))
        spans.extend((a[1], b[0], "internal_nonspeech") for a, b in zip(intervals, intervals[1:]))
        spans.extend((float(item["start_sec"]), float(item["end_sec"]), "manual_exclusion")
                     for item in self.exclusions)
        for start, end, role in spans:
            if end <= start:
                continue
            item = pg.LinearRegionItem(values=(start, end), movable=False,
                brush=pg.mkBrush(*ROLE_COLORS[role]), pen=pg.mkPen(None))
            item.setZValue(-3 if role in {"manual_exclusion", "outside_analysis_window"} else -10)
            self.waveform.addItem(item)
            self._background.append(item)

    def _clicked(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        point = self.waveform.plotItem.vb.mapSceneToView(event.scenePos())
        time = max(0.0, min(self.duration_sec, float(point.x())))
        if self._editing:
            for index, region in enumerate(self._regions):
                a, b = region.getRegion()
                if a <= time <= b:
                    self._selected_region = index
                    break
        self.set_cursor(time)
        self.seek_requested.emit(time)
        if event.double():
            self.zoom_to_selection()
