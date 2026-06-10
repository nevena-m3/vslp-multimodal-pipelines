"""VSLP Kinematics Pipeline GUI v0.77.

This GUI intentionally mirrors the acoustic pipeline layout: left stage sidebar,
institutional branding strip, top tabs, run log, and compact scientific workflow
panels. Heavy MediaPipe extraction and final feature computation are connected in
later patches; this pass establishes the production-quality outline and user flow.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

try:
    from PySide6.QtCore import QPointF, QRectF, QObject, QThread, Qt, Signal, QTimer
    from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QAbstractSpinBox,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QDoubleSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTreeWidget,
        QTreeWidgetItem,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PySide6 is required to launch the kinematics GUI. Install with: pip install -e '.[gui]'"
    ) from exc

from vslp.analysis.kinematics import (
    AGGREGATION_PROFILES,
    LANDMARK_PRESETS,
    NORMALIZATION_METHODS,
    NORMALIZATION_METHOD_DETAILS,
    LandmarkRunConfig,
    VideoIngestConfig,
    bootstrap_mediapipe_runtime,
    download_default_model,
    run_mediapipe_landmarks,
    verify_mediapipe_runtime,
    mediapipe_environment_status,
    link_metadata,
    mediapipe_capability_note,
    run_ingest,
    summarize_ingest_manifest,
    build_format_summary,
    build_warning_summary,
    analyze_landmark_selection,
    write_selected_landmarks,
    write_landmark_plan,
    write_normalization_config,
    run_normalization_from_selection,
    run_video_qc,
    DEFAULT_KINEMATIC_FEATURE_IDS,
    KINEMATIC_FEATURE_GROUPS,
    KINEMATIC_FEATURE_SPECS,
    QC_FEATURE_REQUIREMENTS,
    FeatureComputationConfig,
    run_feature_computation,
    TemporalAggregationConfig,
    run_temporal_aggregation,
    write_scaffold_report,
)
from vslp.analysis.kinematics.schemas import DEFAULT_VIDEO_EXTENSIONS, parse_int_list

APP_VERSION = "v0.77"
BRAND_DIR = Path(__file__).resolve().parent / "assets" / "branding"
LAB_LOGO = BRAND_DIR / "lab_logo.png"
UOFT_LOGO = BRAND_DIR / "uoft_logo.png"


@dataclass
class StageRecord:
    status: str = "Not run"
    summary_path: str | None = None
    report_path: str | None = None
    manifest_path: str | None = None
    errors_path: str | None = None


class Worker(QObject):
    started = Signal(str)
    finished = Signal(str, object)
    failed = Signal(str, str)
    message = Signal(str)

    def __init__(self, name: str, func: Callable, kwargs: dict):
        super().__init__()
        self.name = name
        self.func = func
        self.kwargs = kwargs

    def run(self) -> None:
        self.started.emit(self.name)
        self.message.emit(f"Starting {self.name}...")
        try:
            result = self.func(**self.kwargs)
        except Exception as exc:  # noqa: BLE001
            import traceback

            self.failed.emit(self.name, f"{exc}\n\n{traceback.format_exc()}")
            return
        self.message.emit(f"Finished {self.name}.")
        self.finished.emit(self.name, result)


class LandmarkMeshCanvas(QWidget):
    """Real-frame MediaPipe overlay selector.

    Unlike the earlier abstract mesh view, this canvas is designed for clinical
    review: it displays an actual frame from the selected video, overlays the
    landmarks extracted by MediaPipe for that same frame, and lets the user click
    directly on points to add/remove them from the working landmark set.
    """

    selection_changed = Signal(str)

    REGION_SETS: dict[str, set[int]] = {
        "mouth/lips": {0, 13, 14, 17, 37, 40, 57, 61, 78, 81, 82, 87, 88, 95, 146, 164, 178, 181, 185, 191, 267, 270, 287, 291, 308, 311, 312, 317, 318, 324, 375, 402, 405, 409, 415},
        "jaw/chin/lower face": {17, 152, 175, 199, 200, 148, 176, 149, 150, 136, 172, 58, 132, 361, 288, 397, 365, 379, 378, 400},
        "eye/canthus anchors": {33, 133, 159, 145, 263, 362, 386, 374, 246, 161, 160, 144, 163, 7, 466, 388, 387, 373, 390, 249},
        "nose/midline": {1, 2, 4, 5, 6, 8, 9, 10, 94, 97, 98, 168, 195, 197, 326, 327},
        "brows/upper face": {70, 105, 107, 336, 334, 300, 46, 52, 53, 65, 55, 285, 295, 282, 283, 276},
        "cheeks/contour": {10, 21, 54, 58, 67, 93, 103, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234, 251, 284, 288, 297, 323, 332, 356, 361, 365, 377, 378, 379, 389, 397, 454},
    }

    REGION_COLORS = {
        "mouth/lips": QColor("#E11D48"),
        "jaw/chin/lower face": QColor("#F59E0B"),
        "eye/canthus anchors": QColor("#2563EB"),
        "nose/midline": QColor("#7C3AED"),
        "brows/upper face": QColor("#0E9F6E"),
        "cheeks/contour": QColor("#64748B"),
        "other": QColor("#94A3B8"),
    }

    LANDMARK_LABELS = {
        13: "upper inner lip / mouth aperture",
        14: "lower inner lip / mouth aperture",
        61: "left mouth corner / commissure",
        291: "right mouth corner / commissure",
        78: "left inner mouth corner",
        308: "right inner mouth corner",
        81: "upper lip support",
        311: "upper lip support",
        0: "midline lip/face reference",
        17: "lower lip / lower-face support",
        152: "chin / jaw anchor",
        199: "lower-face reference",
        33: "left outer eye/canthus anchor",
        263: "right outer eye/canthus anchor",
        1: "nose tip / midline reference",
        70: "left brow/upper-face motion",
        300: "right brow/upper-face motion",
        105: "left brow support",
        334: "right brow support",
        159: "left upper eyelid",
        386: "right upper eyelid",
        145: "left lower eyelid",
        374: "right lower eyelid",
    }

    MOUTH_EDGES = [(61, 78), (78, 13), (13, 308), (308, 291), (291, 14), (14, 61), (13, 14), (61, 291)]
    EYE_EDGES = [(33, 133), (263, 362), (33, 159), (159, 133), (33, 145), (145, 133), (263, 386), (386, 362), (263, 374), (374, 362)]
    FACE_GUIDE_EDGES = [(33, 263), (1, 152), (61, 291), (13, 14), (17, 152)]

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(760, 560)
        self.setMouseTracking(True)
        self.points: dict[int, tuple[float, float]] = {}
        self.selected: set[int] = set()
        self.hover_idx: int | None = None
        self.frame_pixmap: QPixmap | None = None
        self.source_label = "No real frame loaded yet"
        self.frame_label = ""
        self.show_all_labels = False
        self.show_selected_labels = True
        self.show_mesh_edges = False
        self.landmark_display_mode = "All faint"
        self.overlay_style = "Balanced"
        self.point_radius = 3.8
        self.selected_point_radius = 7.0
        self.high_contrast_landmarks = False
        self.zoom_factor = 1.0
        self.auto_zoom_face = True
        self.view_center = (0.5, 0.5)
        self._dragging = False
        self._drag_start = None
        self._drag_center = self.view_center

    @classmethod
    def region_for(cls, idx: int) -> str:
        for region, vals in cls.REGION_SETS.items():
            if idx in vals:
                return region
        return "other"

    @classmethod
    def label_for(cls, idx: int) -> str:
        return cls.LANDMARK_LABELS.get(idx, cls.region_for(idx))

    def set_overlay_style(self, style: str) -> None:
        self.overlay_style = style if style in {"Subtle", "Balanced", "High contrast"} else "Balanced"
        if self.overlay_style == "Subtle":
            self.point_radius = 2.4
            self.selected_point_radius = 5.6
        elif self.overlay_style == "High contrast":
            self.point_radius = 3.8
            self.selected_point_radius = 7.2
        else:
            self.point_radius = 3.0
            self.selected_point_radius = 6.2
        self.update()

    def set_selected_labels_visible(self, visible: bool) -> None:
        self.show_selected_labels = bool(visible)
        self.update()

    def set_mesh_edges_visible(self, visible: bool) -> None:
        self.show_mesh_edges = bool(visible)
        self.update()

    def set_landmark_display_mode(self, mode: str) -> None:
        allowed = {"All faint", "Selected + anchors", "Selected only"}
        self.landmark_display_mode = mode if mode in allowed else "All faint"
        self.update()

    def set_overlay(self, frame: QPixmap | None, points: dict[int, tuple[float, float]], source_label: str, frame_label: str = "") -> None:
        self.frame_pixmap = frame
        self.points = points or {}
        self.source_label = source_label
        self.frame_label = frame_label
        self.hover_idx = None
        if self.auto_zoom_face and self.points:
            self.zoom_to_face(margin=0.18, emit=False)
        self.update()

    def set_points(self, points: dict[int, tuple[float, float]], source_label: str) -> None:
        self.set_overlay(None, points, source_label)

    def set_selected(self, indices: list[int] | tuple[int, ...] | set[int]) -> None:
        allowed = set(self.points.keys()) if self.points else set(range(0, 500))
        self.selected = {int(i) for i in indices if int(i) in allowed}
        self.selection_changed.emit(", ".join(map(str, sorted(self.selected))))
        self.update()

    def selected_text(self) -> str:
        return ", ".join(map(str, sorted(self.selected)))

    def _view_window(self) -> tuple[float, float, float, float]:
        z = max(1.0, float(self.zoom_factor))
        width = 1.0 / z
        height = 1.0 / z
        cx, cy = self.view_center
        x0 = min(max(0.0, cx - width / 2.0), max(0.0, 1.0 - width))
        y0 = min(max(0.0, cy - height / 2.0), max(0.0, 1.0 - height))
        return x0, y0, x0 + width, y0 + height

    def set_zoom(self, factor: float, *, emit: bool = True) -> None:
        self.zoom_factor = float(max(1.0, min(12.0, factor)))
        if emit:
            self.selection_changed.emit(self.selected_text())
        self.update()

    def zoom_in(self) -> None:
        self.set_zoom(self.zoom_factor * 1.35)

    def zoom_out(self) -> None:
        self.set_zoom(self.zoom_factor / 1.35)

    def reset_view(self) -> None:
        self.zoom_factor = 1.0
        self.view_center = (0.5, 0.5)
        self.update()

    def zoom_to_face(self, margin: float = 0.16, *, emit: bool = True) -> None:
        if not self.points:
            self.reset_view()
            return
        xs = [p[0] for p in self.points.values()]
        ys = [p[1] for p in self.points.values()]
        x0, x1 = max(0.0, min(xs) - margin), min(1.0, max(xs) + margin)
        y0, y1 = max(0.0, min(ys) - margin), min(1.0, max(ys) + margin)
        w = max(0.08, x1 - x0)
        h = max(0.08, y1 - y0)
        # Use one isotropic normalized zoom so clicking stays simple and the image is not distorted.
        self.zoom_factor = float(max(1.0, min(12.0, min(1.0 / w, 1.0 / h))))
        self.view_center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        if emit:
            self.selection_changed.emit(self.selected_text())
        self.update()

    def _image_rect(self):
        margin = 22
        title_h = 38
        footer_h = 34
        avail_w = max(20, self.width() - 2 * margin)
        avail_h = max(20, self.height() - title_h - footer_h - margin)
        left = margin
        top = title_h
        if self.frame_pixmap is None or self.frame_pixmap.isNull():
            return left, top, avail_w, avail_h
        iw, ih = self.frame_pixmap.width(), self.frame_pixmap.height()
        scale = min(avail_w / iw, avail_h / ih)
        w = int(iw * scale)
        h = int(ih * scale)
        return int(left + (avail_w - w) / 2), int(top + (avail_h - h) / 2), w, h

    def _to_screen(self, x: float, y: float) -> QPointF:
        left, top, w, h = self._image_rect()
        vx0, vy0, vx1, vy1 = self._view_window()
        return QPointF(left + ((x - vx0) / max(1e-9, vx1 - vx0)) * w, top + ((y - vy0) / max(1e-9, vy1 - vy0)) * h)

    def _to_norm_delta(self, dx: float, dy: float) -> tuple[float, float]:
        _left, _top, w, h = self._image_rect()
        vx0, vy0, vx1, vy1 = self._view_window()
        return dx / max(1.0, w) * (vx1 - vx0), dy / max(1.0, h) * (vy1 - vy0)

    def _nearest(self, pos) -> tuple[int | None, float]:
        best_idx = None
        best_d = 1e9
        for idx, (x, y) in self.points.items():
            p = self._to_screen(x, y)
            d = ((p.x() - pos.x()) ** 2 + (p.y() - pos.y()) ** 2) ** 0.5
            if d < best_d:
                best_idx, best_d = idx, d
        return best_idx, best_d

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._dragging and self._drag_start is not None:
            dx = event.position().x() - self._drag_start.x()
            dy = event.position().y() - self._drag_start.y()
            ndx, ndy = self._to_norm_delta(dx, dy)
            cx, cy = self._drag_center
            self.view_center = (cx - ndx, cy - ndy)
            self.update()
            return
        idx, d = self._nearest(event.position()) if self.points else (None, 1e9)
        self.hover_idx = idx if d <= max(14, 22 / max(1.0, self.zoom_factor ** 0.25)) else None
        if self.hover_idx is not None:
            self.setToolTip(f"Landmark {self.hover_idx}: {self.label_for(self.hover_idx)}\nClick to add/remove. Drag empty space to pan; use zoom controls to inspect the mouth/face.")
        else:
            self.setToolTip("Load a real video frame with MediaPipe overlay. Drag empty space to pan; use zoom controls to inspect points.")
        self.update()

    def mousePressEvent(self, event):  # noqa: N802
        if not self.points:
            return
        if event.button() == Qt.RightButton:
            self._dragging = True
            self._drag_start = event.position()
            self._drag_center = self.view_center
            return
        if event.button() != Qt.LeftButton:
            return
        idx, d = self._nearest(event.position())
        threshold = 18 if self.zoom_factor <= 2 else 22
        if idx is not None and d <= threshold:
            if idx in self.selected:
                self.selected.remove(idx)
            else:
                self.selected.add(idx)
            self.selection_changed.emit(self.selected_text())
            self.update()
        else:
            self._dragging = True
            self._drag_start = event.position()
            self._drag_center = self.view_center

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._dragging = False
        self._drag_start = None

    def wheelEvent(self, event):  # noqa: N802
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in()
        elif delta < 0:
            self.zoom_out()

    def _draw_edge(self, painter: QPainter, a: int, b: int, color: QColor, width: float = 1.1) -> None:
        if a not in self.points or b not in self.points:
            return
        pa = self._to_screen(*self.points[a])
        pb = self._to_screen(*self.points[b])
        painter.setPen(QPen(color, width))
        painter.drawLine(pa, pb)

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#07111F"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#F8FAFC"))
        painter.drawText(18, 24, "Real video frame + Google MediaPipe overlay")
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#CBD5E1"))
        painter.drawText(18, 42, self.frame_label or "Load an extracted video/frame to inspect actual landmark placement.")

        left, top, w, h = self._image_rect()
        painter.setPen(QPen(QColor("#1E3A5F"), 1))
        painter.setBrush(QBrush(QColor("#0F172A")))
        painter.drawRoundedRect(left, top, w, h, 8, 8)
        if self.frame_pixmap is not None and not self.frame_pixmap.isNull():
            vx0, vy0, vx1, vy1 = self._view_window()
            src = QRectF(vx0 * self.frame_pixmap.width(), vy0 * self.frame_pixmap.height(), (vx1 - vx0) * self.frame_pixmap.width(), (vy1 - vy0) * self.frame_pixmap.height())
            dst = QRectF(left, top, w, h)
            painter.drawPixmap(dst, self.frame_pixmap, src)
        else:
            painter.setPen(QColor("#CBD5E1"))
            painter.drawText(left + 24, top + 42, "No real video frame loaded. Run Landmarks, then use Representative Detected Frame or Load Selected Frame.")

        if self.show_mesh_edges and self.points:
            for a, b in self.FACE_GUIDE_EDGES:
                self._draw_edge(painter, a, b, QColor(255, 255, 255, 165), 1.4)
            for a, b in self.EYE_EDGES:
                self._draw_edge(painter, a, b, QColor(37, 99, 235, 155), 1.2)
            for a, b in self.MOUTH_EDGES:
                self._draw_edge(painter, a, b, QColor(225, 29, 72, 180), 1.6)

        style = getattr(self, "overlay_style", "Balanced")
        if style == "Subtle":
            point_alpha, ring_alpha, halo_width, rim_width = 85, 95, 1.4, 0.7
        elif style == "High contrast":
            point_alpha, ring_alpha, halo_width, rim_width = 185, 195, 2.6, 1.2
        else:
            point_alpha, ring_alpha, halo_width, rim_width = 125, 135, 1.8, 0.9

        # Review-grade rendering: the default keeps all landmarks visible but
        # quiet. Analysts can reduce clutter to selected/anchor points or increase
        # contrast only when a dark/low-quality frame needs it.
        anchor_hint = {33, 133, 263, 362, 1, 13, 14, 61, 291, 17, 152}
        display_mode = getattr(self, "landmark_display_mode", "All faint")
        for idx, (x, y) in self.points.items():
            if idx in self.selected:
                continue
            if display_mode == "Selected only":
                continue
            if display_mode == "Selected + anchors" and idx not in anchor_hint and idx != self.hover_idx:
                continue
            p = self._to_screen(x, y)
            region = self.region_for(idx)
            if display_mode == "All faint":
                fill = QColor("#CBD5E1")
                fill.setAlpha(point_alpha)
            else:
                base = self.REGION_COLORS.get(region, self.REGION_COLORS["other"])
                fill = QColor(base)
                fill.setAlpha(min(210, point_alpha + 30) if idx in anchor_hint else point_alpha)
            if idx == self.hover_idx:
                fill = QColor("#FBBF24")
                fill.setAlpha(235)
            radius = self.point_radius + (2.0 if idx == self.hover_idx else 0.0)
            painter.setPen(QPen(QColor(2, 6, 23, ring_alpha), halo_width))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(p, radius + 0.9, radius + 0.9)
            painter.setPen(QPen(QColor(241, 245, 249, max(85, ring_alpha - 45)), rim_width))
            painter.setBrush(QBrush(fill))
            painter.drawEllipse(p, radius, radius)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        for idx in sorted(self.selected):
            if idx not in self.points:
                continue
            p = self._to_screen(*self.points[idx])
            radius = self.selected_point_radius
            painter.setPen(QPen(QColor(2, 6, 23, 220), 2.6 if style != "Subtle" else 2.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(p, radius + 1.4, radius + 1.4)
            painter.setPen(QPen(QColor("#FCD34D"), 1.8))
            painter.setBrush(QBrush(QColor("#0E7490")))
            painter.drawEllipse(p, radius, radius)
            if self.show_selected_labels:
                label_bg = QRectF(p.x() + 8, p.y() - 21, 32, 17)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(3, 7, 18, 190)))
                painter.drawRoundedRect(label_bg, 4, 4)
                painter.setPen(QColor("#FFF7ED"))
                painter.drawText(label_bg, Qt.AlignCenter, str(idx))

        if self.hover_idx is not None and self.hover_idx not in self.selected:
            p = self._to_screen(*self.points[self.hover_idx])
            painter.setPen(QPen(QColor(3, 7, 18, 235), 3.2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(p, 12, 12)
            painter.setPen(QPen(QColor("#FACC15"), 2.0))
            painter.drawEllipse(p, 10, 10)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            label_bg = QRectF(p.x() + 10, p.y() - 23, 32, 17)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(3, 7, 18, 190)))
            painter.drawRoundedRect(label_bg, 4, 4)
            painter.setPen(QColor("#FACC15"))
            painter.drawText(label_bg, Qt.AlignCenter, str(self.hover_idx))

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#E2E8F0"))
        footer = f"Source: {self.source_label} | visible landmarks: {len(self.points)} | selected: {len(self.selected)} | zoom: {self.zoom_factor:.1f}x | left-click point, right/left-drag empty space to pan, wheel to zoom"
        painter.drawText(18, self.height() - 13, footer)



class NormalizationMethodVisual(QWidget):
    """Compact schematic explaining what normalization changes.

    The widget is intentionally illustrative rather than data-derived: it shows
    that raw coordinates are centered and divided by a stable anatomical scale.
    It avoids pretending that MediaPipe monocular landmarks become true physical
    millimeters without calibration.
    """

    def __init__(self) -> None:
        super().__init__()
        self.method = "intercanthal_distance"
        self.setMinimumSize(360, 150)

    def set_method(self, method: str) -> None:
        self.method = method or "intercanthal_distance"
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect().adjusted(12, 10, -12, -10)
        painter.fillRect(rect, QColor("#101A2A"))
        painter.setPen(QPen(QColor("#28445F"), 1))
        painter.drawRoundedRect(rect, 12, 12)

        details = NORMALIZATION_METHOD_DETAILS.get(self.method, {})
        display_name = str(details.get("display_name", self.method.replace("_", " ")))
        anchor_landmarks = tuple(details.get("anchor_landmarks", ()) or ())
        fallback_landmarks = tuple(details.get("fallback_landmarks", ()) or ())

        painter.setPen(QPen(QColor("#D8E8F8"), 1))
        title_font = QFont("Arial", 9)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(rect.adjusted(12, 4, -12, -4), Qt.AlignTop | Qt.AlignLeft, display_name)

        left = QRectF(rect.left() + 18, rect.top() + 42, rect.width() * 0.40, rect.height() - 58)
        right = QRectF(rect.left() + rect.width() * 0.57, rect.top() + 42, rect.width() * 0.34, rect.height() - 58)
        self._draw_face(painter, left, raw=True, anchor_landmarks=anchor_landmarks, fallback_landmarks=fallback_landmarks)
        self._draw_arrow(painter, QRectF(left.right() + 8, left.center().y() - 8, right.left() - left.right() - 16, 16))
        self._draw_face(painter, right, raw=False, anchor_landmarks=anchor_landmarks, fallback_landmarks=fallback_landmarks)

        painter.setFont(QFont("Arial", 7))
        painter.setPen(QPen(QColor("#AFC8DE"), 1))
        painter.drawText(left.adjusted(0, left.height() - 15, 0, 12), Qt.AlignCenter, "raw frame units")
        painter.drawText(right.adjusted(0, right.height() - 15, 0, 12), Qt.AlignCenter, "centered + scaled")
        painter.end()

    def _draw_arrow(self, painter: QPainter, rect: QRectF) -> None:
        if rect.width() <= 10:
            return
        y = rect.center().y()
        x1 = rect.left()
        x2 = rect.right()
        painter.setPen(QPen(QColor("#7DD3FC"), 2))
        painter.drawLine(QPointF(x1, y), QPointF(x2, y))
        painter.drawLine(QPointF(x2, y), QPointF(x2 - 6, y - 5))
        painter.drawLine(QPointF(x2, y), QPointF(x2 - 6, y + 5))

    def _draw_face(self, painter: QPainter, rect: QRectF, *, raw: bool, anchor_landmarks: tuple, fallback_landmarks: tuple) -> None:
        cx = rect.center().x()
        cy = rect.center().y() - 4
        rx = rect.width() * (0.30 if raw else 0.25)
        ry = rect.height() * (0.36 if raw else 0.30)
        face_rect = QRectF(cx - rx, cy - ry, 2 * rx, 2 * ry)
        painter.setBrush(QBrush(QColor("#17263A")))
        painter.setPen(QPen(QColor("#64748B"), 1.2))
        painter.drawEllipse(face_rect)

        # Stable anchors. Different methods emphasize different denominators.
        if self.method == "face_bbox_width":
            painter.setPen(QPen(QColor("#FBBF24"), 2))
            painter.drawLine(QPointF(face_rect.left(), cy), QPointF(face_rect.right(), cy))
            label = "face width"
        elif self.method == "face_height_nose_chin":
            painter.setPen(QPen(QColor("#FBBF24"), 2))
            painter.drawLine(QPointF(cx, face_rect.top() + 6), QPointF(cx, face_rect.bottom() - 4))
            label = "10/152"
        elif self.method == "raw_normalized_coordinates":
            painter.setPen(QPen(QColor("#94A3B8"), 1))
            label = "no scale"
        elif self.method == "procrustes_head_stabilized":
            painter.setPen(QPen(QColor("#C084FC"), 2))
            painter.drawLine(QPointF(cx - rx * 0.55, cy - ry * 0.16), QPointF(cx + rx * 0.55, cy - ry * 0.16))
            painter.drawLine(QPointF(cx - rx * 0.40, cy + ry * 0.26), QPointF(cx + rx * 0.40, cy + ry * 0.26))
            label = "future rigid fit"
        else:
            painter.setPen(QPen(QColor("#FBBF24"), 2))
            painter.drawLine(QPointF(cx - rx * 0.52, cy - ry * 0.18), QPointF(cx + rx * 0.52, cy - ry * 0.18))
            label = "/".join(map(str, anchor_landmarks or fallback_landmarks or ())) or "anchors"

        # Eyes/mouth/chin reference points.
        point_pen = QPen(QColor("#0B1220"), 1)
        painter.setPen(point_pen)
        for x, y, color in [
            (cx - rx * 0.52, cy - ry * 0.18, "#FBBF24"),
            (cx + rx * 0.52, cy - ry * 0.18, "#FBBF24"),
            (cx - rx * 0.30, cy + ry * 0.22, "#38BDF8"),
            (cx + rx * 0.30, cy + ry * 0.22, "#38BDF8"),
            (cx, cy + ry * 0.55, "#AFC8DE"),
        ]:
            painter.setBrush(QBrush(QColor(color)))
            painter.drawEllipse(QPointF(x, y), 3.2, 3.2)
        painter.setPen(QPen(QColor("#AFC8DE"), 1))
        painter.setFont(QFont("Arial", 7))
        painter.drawText(rect.adjusted(0, 0, 0, -2), Qt.AlignBottom | Qt.AlignCenter, label)

class KinematicsPipelineWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VSLP - Kinematics Pipeline")
        self.resize(1480, 900)
        self.setMinimumSize(1180, 760)

        self.input_root: Path | None = None
        self.output_root: Path | None = None
        self.ingest_manifest_csv: Path | None = None
        self.metadata_path: Path | None = None
        self.last_report_html: Path | None = None
        self.landmark_indices: tuple[int, ...] = LANDMARK_PRESETS["ALS oral-motor core 15"]
        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self._worker_done_callback: Callable[[object], None] | None = None
        self._busy_task_name: str | None = None
        self._landmark_overlay_loaded = False
        self._landmark_frame_reload_timer = QTimer(self)
        self._landmark_frame_reload_timer.setSingleShot(True)
        self._landmark_frame_reload_timer.timeout.connect(self.load_real_frame_landmark_overlay)

        self.stage_records: dict[str, StageRecord] = {
            "project": StageRecord(),
            "metadata": StageRecord(),
            "ingest": StageRecord(),
            "landmarks": StageRecord(),
            "selection": StageRecord(),
            "normalization": StageRecord(),
            "qc": StageRecord(),
            "features": StageRecord(),
            "aggregation": StageRecord(),
            "reports": StageRecord(),
        }
        self._build_ui()
        self._refresh_stage_cards()

    # ---------------------------- UI BUILD ----------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(260)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(7)

        title = QLabel("VSLP")
        title.setObjectName("AppTitleLabel")
        subtitle = QLabel(f"Kinematics Pipeline GUI {APP_VERSION}")
        subtitle.setObjectName("SubtitleLabel")
        ip_notice = QLabel(
            "(c) 2026 Nevena Musikic & Yana Yunusova\n"
            "Speech Production Lab, University of Toronto"
        )
        ip_notice.setObjectName("IPNoticeLabel")
        ip_notice.setWordWrap(True)
        side_layout.addWidget(title)
        side_layout.addWidget(subtitle)
        side_layout.addWidget(ip_notice)

        self.stage_labels: dict[str, QLabel] = {}
        self.stage_status_dots: dict[str, QFrame] = {}
        self.stage_cards: dict[str, QFrame] = {}
        for key, label in [
            ("project", "Project"),
            ("metadata", "Metadata"),
            ("ingest", "Ingest"),
            ("landmarks", "Face Landmarks"),
            ("selection", "Landmark Selection"),
            ("normalization", "Normalization"),
            ("qc", "Video QC"),
            ("features", "Feature Computation"),
            ("aggregation", "Temporal Aggregation"),
            ("reports", "Reports"),
        ]:
            card = QFrame()
            card.setObjectName("Card")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(9, 7, 9, 7)
            label_widget = QLabel(label)
            label_widget.setStyleSheet("font-weight: 700;")
            status_row = QHBoxLayout()
            status_row.setContentsMargins(0, 0, 0, 0)
            status_row.setSpacing(6)
            status_dot = QFrame()
            status_dot.setObjectName("StageStatusDot")
            status_dot.setFixedSize(10, 10)
            status_lbl = QLabel("Not run")
            status_lbl.setObjectName("SubtitleLabel")
            status_row.addWidget(status_dot)
            status_row.addWidget(status_lbl, stretch=1)
            card_layout.addWidget(label_widget)
            card_layout.addLayout(status_row)
            card.setMaximumHeight(62)
            self.stage_labels[key] = status_lbl
            self.stage_status_dots[key] = status_dot
            self.stage_cards[key] = card
            side_layout.addWidget(card)

        side_layout.addStretch(1)
        self.refresh_outputs_btn = QPushButton("Refresh Latest Outputs")
        self.refresh_outputs_btn.clicked.connect(self.refresh_latest_outputs)
        side_layout.addWidget(self.refresh_outputs_btn)

        self.run_all_btn = QPushButton("Run Full Kinematics Workflow")
        self.run_all_btn.setObjectName("RunButton")
        self.run_all_btn.clicked.connect(self.run_all)
        side_layout.addWidget(self.run_all_btn)

        self.open_output_btn = QPushButton("Open Output Project Folder")
        self.open_output_btn.setObjectName("OpenButton")
        self.open_output_btn.clicked.connect(self.open_output_root)
        side_layout.addWidget(self.open_output_btn)
        root.addWidget(sidebar)

        main_col = QVBoxLayout()
        main_col.setSpacing(16)
        main_col.addWidget(self._build_branding_bar())

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.ElideRight)
        self.tabs.addTab(self._build_setup_tab(), "Setup")
        self.tabs.addTab(self._build_metadata_tab(), "Metadata")
        self.tabs.addTab(self._build_landmarks_tab(), "Landmarks")
        self.tabs.addTab(self._build_selection_tab(), "Landmark Selection")
        self.tabs.addTab(self._build_normalization_tab(), "Normalization")
        self.tabs.addTab(self._build_qc_tab(), "Video QC")
        self.tabs.addTab(self._build_features_tab(), "Features")
        self.tabs.addTab(self._build_aggregation_tab(), "Aggregation")
        self.tabs.addTab(self._build_inspector_tab(), "Inspector")
        self.tabs.addTab(self._build_reports_tab(), "Reports & Outputs")
        main_col.addWidget(self.tabs, stretch=1)

        log_group = QGroupBox("Run Log")
        log_layout = QVBoxLayout(log_group)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(90)
        self.log_box.setMaximumHeight(125)
        log_layout.addWidget(self.progress)
        log_layout.addWidget(self.log_box)
        log_group.setMaximumHeight(180)
        main_col.addWidget(log_group)
        root.addLayout(main_col, stretch=1)

    def _build_branding_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("BrandingBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)
        label = QLabel("VSLP Kinematics Pipeline")
        label.setObjectName("BrandingTitle")
        layout.addWidget(label)
        layout.addStretch(1)
        layout.addWidget(self._logo_or_text(LAB_LOGO, "Speech Production Lab", max_width=210, max_height=56))
        layout.addWidget(self._logo_or_text(UOFT_LOGO, "University of Toronto", max_width=250, max_height=54))
        bar.setMaximumHeight(76)
        return bar

    def _logo_or_text(self, logo_path: Path, fallback: str, max_width: int = 170, max_height: int = 42) -> QLabel:
        widget = QLabel()
        widget.setObjectName("LogoPlaceholder")
        widget.setAlignment(Qt.AlignCenter)
        widget.setMinimumWidth(min(max_width, 220))
        widget.setMaximumWidth(max_width)
        widget.setMaximumHeight(max_height)
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            if not pixmap.isNull():
                widget.setObjectName("LogoImage")
                widget.setPixmap(pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                widget.setToolTip(str(logo_path))
                return widget
        widget.setText(fallback)
        widget.setToolTip(f"Optional logo file not found: {logo_path}")
        return widget

    # ---------------------------- HELPERS ----------------------------
    def _info_panel(self, title: str, body: str) -> QFrame:
        panel = QFrame()
        panel.setObjectName("InfoPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        title_label = QLabel(title)
        title_label.setObjectName("InfoTitle")
        body_label = QLabel(body)
        body_label.setObjectName("InfoBody")
        body_label.setWordWrap(True)
        panel.setMaximumHeight(118)
        layout.addWidget(title_label)
        layout.addWidget(body_label)
        return panel

    def _scrollable(self, content: QWidget) -> QWidget:
        wrapper = QWidget()
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return wrapper

    def _set_tooltip(self, widget: QWidget, text: str) -> None:
        widget.setToolTip(text)

    def _path_or_warn(self, edit: QLineEdit, label: str) -> Path | None:
        text = edit.text().strip()
        if not text:
            QMessageBox.warning(self, "Missing path", f"Please select {label}.")
            return None
        return Path(text).expanduser().resolve()

    def _log(self, message: str) -> None:
        self.log_box.appendPlainText(message)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def _set_progress_idle(self, value: int = 0) -> None:
        if hasattr(self, "progress"):
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, int(value))))
            self.progress.setFormat("Idle" if value == 0 else f"{int(value)}%")

    def _set_progress_busy(self, label: str) -> None:
        if hasattr(self, "progress"):
            self.progress.setRange(0, 0)
            self.progress.setFormat(label)

    def _start_worker(self, name: str, func: Callable, kwargs: dict, done_callback: Callable[[object], None]) -> None:
        if self._thread is not None:
            QMessageBox.information(self, "Stage already running", "Wait for the current stage to finish before starting another stage.")
            return
        self._busy_task_name = name
        self._worker_done_callback = done_callback
        self._thread = QThread(self)
        self._worker = Worker(name, func, kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.started.connect(self._on_worker_started)
        self._worker.message.connect(self._log)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.finished.connect(lambda *_: self._thread.quit())
        self._worker.failed.connect(lambda *_: self._thread.quit())
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_worker_state)
        self._thread.start()

    def _on_worker_started(self, name: str) -> None:
        self._set_progress_busy(f"Running: {name}")
        self._log(f"[RUNNING] {name}")

    def _on_worker_failed(self, name: str, error: str) -> None:
        self._set_progress_idle(0)
        self._log(f"[FAILED] {name}: {error}")
        QMessageBox.critical(self, f"{name} failed", error[:4000])

    def _on_worker_finished(self, name: str, result: object) -> None:
        self._set_progress_idle(100)
        self._log(f"[DONE] {name}")
        callback = self._worker_done_callback
        if callback is not None:
            callback(result)
        QTimer.singleShot(900, lambda: self._set_progress_idle(0))

    def _clear_worker_state(self) -> None:
        self._thread = None
        self._worker = None
        self._worker_done_callback = None
        self._busy_task_name = None

    def _stage_status_presentation(self, status: str) -> tuple[str, str]:
        """Return human-readable sidebar status text and indicator color."""
        normalized = str(status or "Not run").strip().lower().replace(" ", "_")
        if normalized in {"completed", "detected"}:
            return "Complete", "#22C55E"
        if normalized == "completed_with_warnings":
            return "Complete - review", "#F59E0B"
        if normalized == "failed":
            return "Failed", "#EF4444"
        if normalized == "running":
            return "Running", "#38BDF8"
        if normalized in {"configured", "planned"}:
            return normalized.capitalize(), "#AFC8DE"
        if normalized == "stale":
            return "Stale", "#A855F7"
        if normalized in {"not_run", "none", ""}:
            return "Not run", "#64748B"
        return str(status), "#64748B"

    def _refresh_stage_cards(self) -> None:
        for key, record in self.stage_records.items():
            if key in self.stage_labels:
                text, color = self._stage_status_presentation(record.status)
                self.stage_labels[key].setText(text)
                if key in self.stage_status_dots:
                    self.stage_status_dots[key].setStyleSheet(
                        f"QFrame#StageStatusDot {{ background-color: {color}; border: 1px solid #D8E8F8; border-radius: 5px; }}"
                    )

    def _fill_table(self, table: QTableWidget, df: pd.DataFrame, max_rows: int = 300) -> None:
        if df is None or df.empty:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        df = df.head(max_rows).copy()
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        table.setRowCount(len(df))
        for r, (_, row) in enumerate(df.iterrows()):
            for c, value in enumerate(row):
                item = QTableWidgetItem("" if pd.isna(value) else str(value))
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)

    # ---------------------------- TABS ----------------------------
    def _build_setup_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        layout.addWidget(self._info_panel(
            "Project dashboard",
            "Define the raw video dataset and output project folder, then run structural ingest. This stage checks video readability, format consistency, FPS, duration, resolution, and early warnings before landmark extraction.",
        ))

        paths_group = QGroupBox("1. Project definition")
        form = QGridLayout(paths_group)
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.project_name_edit = QLineEdit("VSLP Kinematics Project")
        self.task_name_edit = QLineEdit()
        self.task_name_edit.setPlaceholderText("e.g., Bamboo passage video, DDK-pa, DDK-pataka, smile, mouth open-close")
        self._set_tooltip(self.input_edit, "Folder containing raw participant videos. Subfolders are searched recursively.")
        self._set_tooltip(self.output_edit, "Root folder where VSLP writes kinematics outputs: tables, reports, logs, manifests, and future artifacts.")
        self._set_tooltip(self.task_name_edit, "Optional task label used as fallback when filename/folder parsing is inconclusive.")
        browse_in = QPushButton("Browse Input Folder")
        browse_in.clicked.connect(self.browse_input_dir)
        browse_out = QPushButton("Browse Output Folder")
        browse_out.clicked.connect(self.browse_output_dir)
        form.addWidget(QLabel("Input video folder"), 0, 0)
        form.addWidget(self.input_edit, 0, 1)
        form.addWidget(browse_in, 0, 2)
        form.addWidget(QLabel("Output project folder"), 1, 0)
        form.addWidget(self.output_edit, 1, 1)
        form.addWidget(browse_out, 1, 2)
        form.addWidget(QLabel("Project name"), 2, 0)
        form.addWidget(self.project_name_edit, 2, 1, 1, 2)
        form.addWidget(QLabel("Task being analyzed"), 3, 0)
        form.addWidget(self.task_name_edit, 3, 1, 1, 2)
        task_hint = QLabel("Examples: Bamboo passage video, DDK-pa, DDK-pataka, smile, mouth opening/closing, facial expression task")
        task_hint.setObjectName("SubtitleLabel")
        task_hint.setWordWrap(True)
        form.addWidget(task_hint, 4, 1, 1, 2)

        workflow_group = QGroupBox("2. Project gate")
        workflow_layout = QVBoxLayout(workflow_group)
        btn_row = QHBoxLayout()
        self.init_project_btn = QPushButton("Initialize Project")
        self.init_project_btn.setObjectName("RunButton")
        self.init_project_btn.clicked.connect(self.run_project_init)
        self.ingest_btn = QPushButton("Run Video Ingest")
        self.ingest_btn.clicked.connect(self.run_ingest_stage)
        btn_row.addWidget(self.init_project_btn)
        btn_row.addWidget(self.ingest_btn)
        btn_row.addStretch(1)
        workflow_layout.addLayout(btn_row)
        self.project_gate_message = QLabel("Select folders, initialize the project, then run video ingest.")
        self.project_gate_message.setObjectName("SubtitleLabel")
        self.project_gate_message.setWordWrap(True)
        workflow_layout.addWidget(self.project_gate_message)

        readiness_group = QGroupBox("3. Dataset readiness")
        readiness_layout = QGridLayout(readiness_group)
        self.ingest_metric_labels: dict[str, QLabel] = {}
        cards = [
            ("Status", "status"),
            ("Videos", "videos"),
            ("Readable", "readable"),
            ("Warnings", "warnings"),
            ("Median FPS", "median_fps"),
            ("Median duration", "median_duration"),
            ("Estimated frames", "frames"),
            ("FPS range", "fps_range"),
        ]
        for idx, (title, key) in enumerate(cards):
            card = QFrame()
            card.setObjectName("InfoPanel")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 8, 10, 8)
            card_title = QLabel(title)
            card_title.setObjectName("SubtitleLabel")
            value = QLabel("-")
            value.setObjectName("InfoTitle")
            value.setWordWrap(True)
            card_layout.addWidget(card_title)
            card_layout.addWidget(value)
            self.ingest_metric_labels[key] = value
            readiness_layout.addWidget(card, idx // 4, idx % 4)
        self.ingest_next_step_label = QLabel("No ingest results yet.")
        self.ingest_next_step_label.setObjectName("SubtitleLabel")
        self.ingest_next_step_label.setWordWrap(True)
        readiness_layout.addWidget(self.ingest_next_step_label, 2, 0, 1, 4)

        ingest_group = QGroupBox("4. Video format summary")
        ingest_layout = QVBoxLayout(ingest_group)
        self.ingest_summary_label = QLabel("No ingest results yet.")
        self.ingest_summary_label.setWordWrap(True)
        self.ingest_summary_label.setObjectName("SubtitleLabel")
        ingest_layout.addWidget(self.ingest_summary_label)
        self.ingest_format_table = QTableWidget(0, 9)
        self.ingest_format_table.setHorizontalHeaderLabels([
            "Extension", "Codec", "Container", "Resolution", "Files", "Median FPS", "Median duration", "Frames", "Warnings"
        ])
        self.ingest_format_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ingest_format_table.setMaximumHeight(190)
        self.ingest_format_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ingest_format_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        ingest_layout.addWidget(self.ingest_format_table)

        warnings_group = QGroupBox("5. Ingest warnings")
        warnings_layout = QVBoxLayout(warnings_group)
        self.ingest_warning_label = QLabel("No warnings yet.")
        self.ingest_warning_label.setObjectName("SubtitleLabel")
        self.ingest_warning_label.setWordWrap(True)
        warnings_layout.addWidget(self.ingest_warning_label)
        self.ingest_warning_table = QTableWidget(0, 4)
        self.ingest_warning_table.setHorizontalHeaderLabels(["Video", "Relative path", "Status", "Warning"])
        self.ingest_warning_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ingest_warning_table.setMaximumHeight(165)
        self.ingest_warning_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ingest_warning_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        warnings_layout.addWidget(self.ingest_warning_table)

        layout.addWidget(paths_group)
        layout.addWidget(workflow_group)
        layout.addWidget(readiness_group)
        layout.addWidget(ingest_group)
        layout.addWidget(warnings_group)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_metadata_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "Info",
            "Metadata is optional for landmark extraction but essential for later linkage to subject, session, task, diagnosis, severity, and visit structure. The GUI links conservatively and never forces uncertain matches.",
        ))
        group = QGroupBox("Demographics / metadata CSV or XLSX")
        grid = QGridLayout(group)
        self.metadata_edit = QLineEdit()
        self.metadata_edit.setPlaceholderText("optional metadata table; CSV/XLSX supported")
        browse_btn = QPushButton("Browse Metadata")
        browse_btn.clicked.connect(self.browse_metadata_file)
        grid.addWidget(QLabel("Metadata file"), 0, 0)
        grid.addWidget(self.metadata_edit, 0, 1)
        grid.addWidget(browse_btn, 0, 2)
        layout.addWidget(group)
        required = QPlainTextEdit()
        required.setReadOnly(True)
        required.setMaximumHeight(130)
        required.setPlainText(
            "Recommended metadata: subject_id, session_id, visit/date, task, diagnosis/group, severity, video filename, device/camera if available.\n\n"
            "Outputs will include metadata link preview, unmatched videos, unused metadata row samples, and a conservative linkage summary."
        )
        layout.addWidget(required)
        run_btn = QPushButton("Run Metadata Linking")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_metadata_stage)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_landmarks_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "Info",
            "Run Google MediaPipe Face Landmarker over each accepted video and write one per-frame landmark CSV per video. Frames with no detected face are preserved as NaN rows instead of being silently removed, so landmark gaps are auditable.",
        ))

        readiness_group = QGroupBox("1. Landmark extraction readiness")
        readiness_layout = QGridLayout(readiness_group)
        self.landmark_metric_labels: dict[str, QLabel] = {}
        landmark_cards = [
            ("Runtime", "runtime"),
            ("Plan", "plan"),
            ("Videos", "videos"),
            ("OK", "ok"),
            ("Errors", "errors"),
            ("Mean detection", "mean_detection"),
            ("Status", "status"),
            ("Next", "next_step"),
        ]
        for idx, (title, key) in enumerate(landmark_cards):
            card = QFrame()
            card.setObjectName("InfoPanel")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 8, 10, 8)
            card_title = QLabel(title)
            card_title.setObjectName("SubtitleLabel")
            value = QLabel("-")
            value.setObjectName("InfoTitle")
            value.setWordWrap(True)
            card_layout.addWidget(card_title)
            card_layout.addWidget(value)
            self.landmark_metric_labels[key] = value
            readiness_layout.addWidget(card, idx // 4, idx % 4)
        self.landmark_next_step_label = QLabel("Run Setup -> Video Ingest first, then write the landmark plan or run extraction.")
        self.landmark_next_step_label.setObjectName("SubtitleLabel")
        self.landmark_next_step_label.setWordWrap(True)
        readiness_layout.addWidget(self.landmark_next_step_label, 2, 0, 1, 4)
        layout.addWidget(readiness_group)

        group = QGroupBox("2. MediaPipe Face Landmarker configuration")
        form = QGridLayout(group)
        self.model_path_edit = QLineEdit("models/face_landmarker.task")
        self.detection_conf_spin = QDoubleSpinBox(); self.detection_conf_spin.setRange(0.0, 1.0); self.detection_conf_spin.setSingleStep(0.05); self.detection_conf_spin.setValue(0.50)
        self.presence_conf_spin = QDoubleSpinBox(); self.presence_conf_spin.setRange(0.0, 1.0); self.presence_conf_spin.setSingleStep(0.05); self.presence_conf_spin.setValue(0.50)
        self.tracking_conf_spin = QDoubleSpinBox(); self.tracking_conf_spin.setRange(0.0, 1.0); self.tracking_conf_spin.setSingleStep(0.05); self.tracking_conf_spin.setValue(0.50)
        self.expected_landmarks_spin = QSpinBox(); self.expected_landmarks_spin.setRange(468, 500); self.expected_landmarks_spin.setValue(478)
        browse_model = QPushButton("Browse Model")
        browse_model.clicked.connect(self.browse_model_file)
        download_model = QPushButton("Download Default Model")
        download_model.clicked.connect(self.download_landmarker_model)
        install_runtime = QPushButton("Install / Verify MediaPipe Runtime")
        install_runtime.setObjectName("RunButton")
        install_runtime.clicked.connect(self.bootstrap_mediapipe_runtime_stage)
        self.auto_prepare_mediapipe_check = QCheckBox("Auto-install missing runtime and download model before extraction")
        self.auto_prepare_mediapipe_check.setChecked(True)
        self.auto_prepare_mediapipe_check.setToolTip(
            "Recommended for normal users. The GUI will install opencv-python/mediapipe into the active VSLP virtual environment if missing, "
            "then download face_landmarker.task if needed before running extraction."
        )
        form.addWidget(QLabel("FaceLandmarker .task model"), 0, 0)
        form.addWidget(self.model_path_edit, 0, 1)
        form.addWidget(browse_model, 0, 2)
        form.addWidget(download_model, 0, 3)
        form.addWidget(QLabel("Detection confidence"), 1, 0); form.addWidget(self.detection_conf_spin, 1, 1)
        form.addWidget(QLabel("Presence confidence"), 2, 0); form.addWidget(self.presence_conf_spin, 2, 1)
        form.addWidget(QLabel("Tracking confidence"), 3, 0); form.addWidget(self.tracking_conf_spin, 3, 1)
        form.addWidget(QLabel("Expected landmark columns"), 4, 0); form.addWidget(self.expected_landmarks_spin, 4, 1)
        form.addWidget(QLabel("Runtime setup"), 5, 0); form.addWidget(install_runtime, 5, 1, 1, 1); form.addWidget(self.auto_prepare_mediapipe_check, 5, 2, 1, 2)
        layout.addWidget(group)
        note = QPlainTextEdit(); note.setReadOnly(True); note.setMaximumHeight(150)
        note.setPlainText(mediapipe_capability_note())
        layout.addWidget(note)
        btn_row = QHBoxLayout()
        plan_btn = QPushButton("Write Landmark Extraction Plan")
        plan_btn.clicked.connect(self.run_landmark_plan_stage)
        run_btn = QPushButton("Run MediaPipe Landmark Extraction")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_landmark_extraction_stage)
        btn_row.addWidget(plan_btn)
        btn_row.addWidget(run_btn)
        layout.addLayout(btn_row)
        self.landmark_runtime_label = QLabel("Runtime not checked yet.")
        self.landmark_runtime_label.setObjectName("SubtitleLabel")
        self.landmark_runtime_label.setWordWrap(True)
        layout.addWidget(self.landmark_runtime_label)
        self.landmark_summary_label = QLabel("No landmark extraction results yet.")
        self.landmark_summary_label.setObjectName("SubtitleLabel")
        self.landmark_summary_label.setWordWrap(True)
        layout.addWidget(self.landmark_summary_label)
        self.landmark_summary_table = QTableWidget(0, 7)
        self.landmark_summary_table.setHorizontalHeaderLabels(["Video", "Status", "Frames", "Face frames", "Dropped", "Detected %", "Output"])
        self.landmark_summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.landmark_summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.landmark_summary_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.landmark_summary_table)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_selection_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "Info",
            "Select the landmark subset that will drive kinematic normalization and feature computation. This workstation uses the real video frame plus the actual MediaPipe landmark row for that frame. Frame review is intentionally explicit rather than playback-based, so landmark placement can be audited precisely.",
        ))

        selection_dashboard = QGroupBox("1. Selection readiness and scientific coverage")
        selection_dashboard_layout = QGridLayout(selection_dashboard)
        self.selection_metric_labels: dict[str, QLabel] = {}
        selection_cards = [
            ("Preset", "preset"),
            ("Selected", "selected"),
            ("Regions", "regions"),
            ("Anchors", "anchors"),
            ("Mouth", "mouth"),
            ("Jaw", "jaw"),
            ("Status", "status"),
            ("Next", "next_step"),
        ]
        for idx, (title, key) in enumerate(selection_cards):
            card = QFrame()
            card.setObjectName("InfoPanel")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 8, 10, 8)
            card_title = QLabel(title)
            card_title.setObjectName("SubtitleLabel")
            value = QLabel("-")
            value.setObjectName("InfoTitle")
            value.setWordWrap(True)
            card_layout.addWidget(card_title)
            card_layout.addWidget(value)
            self.selection_metric_labels[key] = value
            selection_dashboard_layout.addWidget(card, idx // 4, idx % 4)
        self.selection_next_step_label = QLabel("Choose a preset, inspect it on a real detected frame, then save selected landmarks.")
        self.selection_next_step_label.setObjectName("SubtitleLabel")
        self.selection_next_step_label.setWordWrap(True)
        selection_dashboard_layout.addWidget(self.selection_next_step_label, 2, 0, 1, 4)
        layout.addWidget(selection_dashboard)

        workstation = QGroupBox("2. Landmark selection workstation")
        workstation_layout = QVBoxLayout(workstation)
        main_row = QHBoxLayout()

        controls = QFrame()
        controls.setObjectName("InfoPanel")
        controls.setMinimumWidth(390)
        controls.setMaximumWidth(470)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 12, 12, 12)

        video_title = QLabel("Video and frame review")
        video_title.setObjectName("InfoTitle")
        controls_layout.addWidget(video_title)
        controls_layout.addWidget(QLabel("Use detected-frame bookmarks for fast review, or enter an exact frame and load it explicitly."))
        self.landmark_video_combo = QComboBox()
        self.landmark_video_combo.currentIndexChanged.connect(self._landmark_video_changed)
        refresh_videos_btn = QPushButton("Refresh extracted videos")
        refresh_videos_btn.clicked.connect(self.refresh_landmark_video_choices)
        controls_layout.addWidget(QLabel("Extracted video"))
        controls_layout.addWidget(self.landmark_video_combo)
        controls_layout.addWidget(refresh_videos_btn)

        self.landmark_frame_combo = QComboBox()
        self.landmark_frame_combo.currentIndexChanged.connect(self._landmark_frame_choice_changed)
        self.landmark_frame_spin = QSpinBox()
        self.landmark_frame_spin.setRange(0, 999999)
        self.landmark_frame_spin.setValue(0)
        self.landmark_frame_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.landmark_frame_spin.setToolTip("Type an exact frame number, then click Load Frame.")
        self.landmark_frame_spin.valueChanged.connect(self._landmark_spin_changed)
        self.landmark_frame_label = QLabel("Frame 0 / 0")
        self.landmark_frame_label.setObjectName("SubtitleLabel")
        load_frame_btn = QPushButton("Load Frame")
        load_frame_btn.setObjectName("RunButton")
        load_frame_btn.clicked.connect(self.load_real_frame_landmark_overlay)
        best_frame_btn = QPushButton("Load middle detected frame")
        best_frame_btn.clicked.connect(self.use_representative_landmark_frame)
        controls_layout.addWidget(QLabel("Detected-frame bookmark"))
        controls_layout.addWidget(self.landmark_frame_combo)
        controls_layout.addWidget(QLabel("Exact frame"))
        controls_layout.addWidget(self.landmark_frame_spin)
        controls_layout.addWidget(self.landmark_frame_label)
        controls_layout.addWidget(load_frame_btn)
        controls_layout.addWidget(best_frame_btn)

        preset_title = QLabel("Landmark set")
        preset_title.setObjectName("InfoTitle")
        controls_layout.addWidget(preset_title)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(LANDMARK_PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self.apply_landmark_preset)
        self.landmark_text = QPlainTextEdit()
        self.landmark_text.setMaximumHeight(72)
        self.landmark_text.setPlainText(", ".join(map(str, self.landmark_indices)))
        controls_layout.addWidget(QLabel("Preset"))
        controls_layout.addWidget(self.preset_combo)
        controls_layout.addWidget(QLabel("Selected landmark indices"))
        controls_layout.addWidget(self.landmark_text)

        region_grid = QGridLayout()
        for idx, (region_label, region_key) in enumerate([
            ("Mouth / lips", "mouth/lips"),
            ("Jaw / chin", "jaw/chin/lower face"),
            ("Eye anchors", "eye/canthus anchors"),
            ("Nose / midline", "nose/midline"),
            ("Brows / upper face", "brows/upper face"),
        ]):
            btn = QPushButton(region_label)
            btn.clicked.connect(lambda _=False, rk=region_key: self.add_landmark_region(rk))
            region_grid.addWidget(btn, idx // 2, idx % 2)
        controls_layout.addLayout(region_grid)

        view_title = QLabel("Overlay display")
        view_title.setObjectName("InfoTitle")
        controls_layout.addWidget(view_title)
        self.overlay_style_combo = QComboBox()
        self.overlay_style_combo.addItems(["Subtle", "Balanced", "High contrast"])
        self.overlay_style_combo.setCurrentText("Balanced")
        self.overlay_style_combo.currentTextChanged.connect(lambda text: self.landmark_canvas.set_overlay_style(text) if hasattr(self, "landmark_canvas") else None)
        self.landmark_display_combo = QComboBox()
        self.landmark_display_combo.addItems(["All faint", "Selected + anchors", "Selected only"])
        self.landmark_display_combo.setCurrentText("All faint")
        self.landmark_display_combo.currentTextChanged.connect(lambda text: self.landmark_canvas.set_landmark_display_mode(text) if hasattr(self, "landmark_canvas") else None)
        self.show_selected_labels_checkbox = QCheckBox("Show selected IDs")
        self.show_selected_labels_checkbox.setChecked(True)
        self.show_selected_labels_checkbox.toggled.connect(lambda checked: self.landmark_canvas.set_selected_labels_visible(checked) if hasattr(self, "landmark_canvas") else None)
        self.show_mesh_edges_checkbox = QCheckBox("Show guide edges")
        self.show_mesh_edges_checkbox.setChecked(False)
        self.show_mesh_edges_checkbox.toggled.connect(lambda checked: self.landmark_canvas.set_mesh_edges_visible(checked) if hasattr(self, "landmark_canvas") else None)
        self.auto_zoom_checkbox = QCheckBox("Auto-zoom to detected face")
        self.auto_zoom_checkbox.setChecked(True)
        self.auto_zoom_checkbox.toggled.connect(self._toggle_auto_zoom_landmark_canvas)
        controls_layout.addWidget(QLabel("Contrast"))
        controls_layout.addWidget(self.overlay_style_combo)
        controls_layout.addWidget(QLabel("Landmarks shown"))
        controls_layout.addWidget(self.landmark_display_combo)
        controls_layout.addWidget(self.show_selected_labels_checkbox)
        controls_layout.addWidget(self.show_mesh_edges_checkbox)
        controls_layout.addWidget(self.auto_zoom_checkbox)
        zoom_row = QHBoxLayout()
        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(lambda: self.landmark_canvas.zoom_in())
        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(lambda: self.landmark_canvas.zoom_out())
        zoom_face_btn = QPushButton("Zoom Face")
        zoom_face_btn.clicked.connect(lambda: self.landmark_canvas.zoom_to_face())
        reset_view_btn = QPushButton("Reset")
        reset_view_btn.clicked.connect(lambda: self.landmark_canvas.reset_view())
        for w in [zoom_in_btn, zoom_out_btn, zoom_face_btn, reset_view_btn]:
            zoom_row.addWidget(w)
        controls_layout.addLayout(zoom_row)

        save_title = QLabel("Save / output")
        save_title.setObjectName("InfoTitle")
        controls_layout.addWidget(save_title)
        apply_btn = QPushButton("Apply / Save Selected Landmarks")
        apply_btn.setObjectName("RunButton")
        apply_btn.clicked.connect(self.run_selection_stage)
        clear_btn = QPushButton("Clear Selection")
        clear_btn.clicked.connect(self.clear_visual_landmarks)
        preview_btn = QPushButton("Save Overlay Preview PNG")
        preview_btn.clicked.connect(self.save_landmark_mesh_preview)
        controls_layout.addWidget(apply_btn)
        controls_layout.addWidget(preview_btn)
        controls_layout.addWidget(clear_btn)
        controls_layout.addStretch(1)

        self.landmark_canvas = LandmarkMeshCanvas()
        self.landmark_canvas.set_overlay_style("Balanced")
        self.landmark_canvas.set_landmark_display_mode("All faint")
        self.landmark_canvas.set_mesh_edges_visible(False)
        self.landmark_canvas.set_selected(self.landmark_indices)
        self.landmark_canvas.selection_changed.connect(self._canvas_selection_changed)
        main_row.addWidget(controls, stretch=0)
        main_row.addWidget(self.landmark_canvas, stretch=1)
        workstation_layout.addLayout(main_row)
        layout.addWidget(workstation)

        output_group = QGroupBox("3. Selection output and requirement checks")
        output_layout = QVBoxLayout(output_group)
        self.selection_feedback_label = QLabel("No real overlay loaded yet. Refresh extracted videos, choose a detected-frame bookmark, then load the frame.")
        self.selection_feedback_label.setWordWrap(True)
        self.selection_feedback_label.setObjectName("SubtitleLabel")
        output_layout.addWidget(self.selection_feedback_label)
        tables_row = QHBoxLayout()
        self.selected_landmark_table = QTableWidget(0, 4)
        self.selected_landmark_table.setHorizontalHeaderLabels(["Landmark", "Region", "Meaning / use", "Status"])
        self.selected_landmark_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.selected_landmark_table.setMinimumHeight(190)
        self.selected_landmark_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tables_row.addWidget(self.selected_landmark_table, stretch=1)
        self.selection_requirement_table = QTableWidget(0, 5)
        self.selection_requirement_table.setHorizontalHeaderLabels(["Requirement", "Status", "Missing required", "Missing recommended", "Reason"])
        self.selection_requirement_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.selection_requirement_table.setMinimumHeight(190)
        self.selection_requirement_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tables_row.addWidget(self.selection_requirement_table, stretch=1)
        output_layout.addLayout(tables_row)
        layout.addWidget(output_group)

        guide = QTextBrowser()
        guide.setMaximumHeight(155)
        guide.setHtml(
            "<h3>Scientific use</h3>"
            "<p><b>Frame review:</b> use detected-frame bookmarks for reliable audit points. Full playback is intentionally not used here because it can hide single-frame tracking failures and make landmark placement look better than it is.</p>"
            "<p><b>Display:</b> default rendering shows all landmarks faintly. Use <b>Selected + anchors</b> to reduce clutter, or <b>High contrast</b> only for difficult frames.</p>"
            "<p><b>Normalization:</b> preferred intercanthal scaling uses inner canthus anchors <b>133/362</b>. If unavailable, normalization can fall back to outer-eye anchors <b>33/263</b>, but that fallback is recorded and QC-flagged for review.</p>"
        )
        layout.addWidget(guide)

        self.preset_table = QTableWidget(0, 4)
        self.preset_table.setHorizontalHeaderLabels(["Preset", "N landmarks", "Recommended use", "Indices"])
        rows = []
        use_map = {
            "ALS oral-motor core 15": "Default oral/jaw/lip kinematics and normalization anchors.",
            "Lower-face jaw/lip kinematics": "Jaw opening, lower lip, commissure, and mouth-aperture signals.",
            "Lip symmetry and lateralization": "Left/right commissure asymmetry and lateralized movement.",
            "Parkinson hypomimia / facial expressivity": "Broader facial expressivity and reduced movement amplitude screening.",
            "Broad audit 30": "Exploratory QC/audit set before narrowing to a task-specific subset.",
        }
        for name, vals in LANDMARK_PRESETS.items():
            rows.append({"Preset": name, "N landmarks": len(vals), "Recommended use": use_map.get(name, "Task-specific or exploratory review."), "Indices": ", ".join(map(str, vals))})
        self._fill_table(self.preset_table, pd.DataFrame(rows), max_rows=20)
        layout.addWidget(self.preset_table)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_normalization_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)
        layout.addWidget(self._info_panel(
            "Normalization dashboard",
            "Normalization converts raw MediaPipe image/model coordinates into centered, face-scale units. It reduces camera-distance and face-size effects before ALS/PD kinematic features are computed. It does not create true millimeter biomechanics and it does not fix poor tracking, head rotation, or missing landmarks; those are handled by QC.",
        ))

        self.norm_metric_labels: dict[str, QLabel] = {}
        metric_group = QGroupBox("Normalization readiness")
        metric_grid = QGridLayout(metric_group)
        metric_specs = [
            ("method", "Method"),
            ("anchors", "Anchors"),
            ("scale_valid", "Scale valid"),
            ("qc", "QC status"),
            ("videos", "Videos"),
            ("next_step", "Next"),
        ]
        for i, (key, title) in enumerate(metric_specs):
            card = QFrame(); card.setObjectName("MetricCard")
            card_layout = QVBoxLayout(card); card_layout.setContentsMargins(10, 7, 10, 7)
            title_label = QLabel(title); title_label.setObjectName("MetricTitle")
            value = QLabel("-"); value.setObjectName("MetricValue"); value.setWordWrap(True)
            card_layout.addWidget(title_label); card_layout.addWidget(value)
            self.norm_metric_labels[key] = value
            metric_grid.addWidget(card, 0, i)
        layout.addWidget(metric_group)

        main_row = QHBoxLayout()
        left_group = QGroupBox("1. Policy and controls")
        left = QVBoxLayout(left_group)
        form = QGridLayout()
        self.norm_combo = QComboBox(); self.norm_combo.addItems(list(NORMALIZATION_METHODS.keys()))
        self.norm_combo.setCurrentText("intercanthal_distance")
        self.norm_combo.currentTextChanged.connect(self._update_normalization_method_panel)
        self.center_landmark_spin = QSpinBox(); self.center_landmark_spin.setRange(0, 477); self.center_landmark_spin.setValue(1)
        self.norm_overwrite_check = QCheckBox("Overwrite existing normalized landmark files")
        self.norm_overwrite_check.setChecked(True)
        form.addWidget(QLabel("Method"), 0, 0); form.addWidget(self.norm_combo, 0, 1)
        form.addWidget(QLabel("Center landmark"), 1, 0); form.addWidget(self.center_landmark_spin, 1, 1)
        form.addWidget(QLabel("Output policy"), 2, 0); form.addWidget(self.norm_overwrite_check, 2, 1)
        left.addLayout(form)

        self.norm_visual = NormalizationMethodVisual()
        left.addWidget(self.norm_visual)
        self.norm_desc = QLabel(""); self.norm_desc.setWordWrap(True); self.norm_desc.setObjectName("SubtitleLabel")
        left.addWidget(self.norm_desc)
        self.norm_science_note = QTextBrowser()
        self.norm_science_note.setMaximumHeight(150)
        self.norm_science_note.setOpenExternalLinks(False)
        left.addWidget(self.norm_science_note)
        btn_row = QHBoxLayout()
        config_btn = QPushButton("Write Config")
        config_btn.clicked.connect(self.write_normalization_config_only)
        btn = QPushButton("Run Normalization")
        btn.setObjectName("RunButton"); btn.clicked.connect(self.run_normalization_stage)
        refresh = QPushButton("Refresh Results")
        refresh.clicked.connect(self._load_normalization_results)
        btn_row.addWidget(config_btn); btn_row.addWidget(btn); btn_row.addWidget(refresh)
        left.addLayout(btn_row)
        main_row.addWidget(left_group, 1)

        right_group = QGroupBox("2. Method audit")
        right = QVBoxLayout(right_group)
        self.norm_anchor_table = QTableWidget(0, 2)
        self.norm_anchor_table.setHorizontalHeaderLabels(["Audit item", "Meaning"])
        self.norm_anchor_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right.addWidget(self.norm_anchor_table)
        self.norm_method_table = QTableWidget(0, 5)
        self.norm_method_table.setHorizontalHeaderLabels(["Method", "Anchors", "Best use", "Caution", "Evidence"])
        self.norm_method_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right.addWidget(self.norm_method_table)
        main_row.addWidget(right_group, 1)
        layout.addLayout(main_row)

        results_group = QGroupBox("3. Outputs and diagnostics")
        results_layout = QVBoxLayout(results_group)
        self.norm_results_label = QLabel("No normalized landmark manifest loaded yet.")
        self.norm_results_label.setObjectName("SubtitleLabel"); self.norm_results_label.setWordWrap(True)
        results_layout.addWidget(self.norm_results_label)
        self.norm_results_table = QTableWidget(0, 10)
        self.norm_results_table.setHorizontalHeaderLabels(["Video", "Status", "Frames", "Face %", "Scale source", "Scale valid %", "Scale CV", "Max jump %", "Selected complete %", "QC flags"])
        self.norm_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.norm_results_table)
        layout.addWidget(results_group)
        layout.addStretch(1)
        self._update_normalization_method_panel(self.norm_combo.currentText())
        self._load_normalization_results()
        return self._scrollable(container)

    def _build_qc_tab(self) -> QWidget:
        container = QWidget(); layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "Info",
            "Video QC now computes automated landmark-extraction risk summaries from the MediaPipe CSVs. It flags pass/review/fail for analyst review using face visibility, long no-face gaps, and landmark frame-to-frame stability. QC does not automatically exclude videos.",
        ))
        qc_rows = [
            ("Face visibility QC", "Face-detected fraction and dropped-frame burden."),
            ("Long gap QC", "Maximum consecutive no-face frames and gap fraction."),
            ("Landmark stability QC", "Median and p95 frame-to-frame displacement for the selected landmark set."),
            ("Review status", "Conservative pass/review/fail flag with written rationale; no automatic exclusion."),
        ]
        table = QTableWidget(0, 2); table.setHorizontalHeaderLabels(["QC family", "Current computation"])
        self._fill_table(table, pd.DataFrame(qc_rows, columns=["QC family", "Current computation"]), max_rows=20)
        layout.addWidget(table)
        btn_row = QHBoxLayout()
        btn = QPushButton("Run Landmark / Video QC")
        btn.setObjectName("RunButton"); btn.clicked.connect(self.run_video_qc_stage)
        refresh = QPushButton("Refresh QC Table")
        refresh.clicked.connect(self._load_video_qc_summary)
        btn_row.addWidget(btn); btn_row.addWidget(refresh); btn_row.addStretch(1)
        layout.addLayout(btn_row)
        self.video_qc_label = QLabel("No video QC summary loaded yet.")
        self.video_qc_label.setObjectName("SubtitleLabel")
        self.video_qc_label.setWordWrap(True)
        layout.addWidget(self.video_qc_label)
        self.video_qc_table = QTableWidget(0, 0)
        self.video_qc_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.video_qc_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.video_qc_table)
        layout.addStretch(1)
        return self._scrollable(container)

    def _build_features_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.addWidget(self._info_panel(
            "Info",
            "Compute kinematic features from normalized MediaPipe landmark trajectories. This tab now mirrors the acoustic Features stage: choose feature families, inspect landmark dependencies and QC gates, configure computation policy, run the stage, then review the feature table.",
        ))

        body = QHBoxLayout()
        left = QGroupBox("Feature selector")
        left_layout = QVBoxLayout(left)
        btn_row = QHBoxLayout()
        all_btn = QPushButton("All")
        default_btn = QPushButton("Default oral-motor")
        robust_btn = QPushButton("Tier B only")
        clear_btn = QPushButton("Clear")
        all_btn.clicked.connect(lambda: self._set_feature_selection({spec.feature_id for spec in KINEMATIC_FEATURE_SPECS}))
        default_btn.clicked.connect(lambda: self._set_feature_selection(set(DEFAULT_KINEMATIC_FEATURE_IDS)))
        robust_btn.clicked.connect(lambda: self._set_feature_selection({spec.feature_id for spec in KINEMATIC_FEATURE_SPECS if spec.tier == "B"}))
        clear_btn.clicked.connect(lambda: self._set_feature_selection(set()))
        for b in [all_btn, default_btn, robust_btn, clear_btn]:
            btn_row.addWidget(b)
        left_layout.addLayout(btn_row)

        self.feature_tree = QTreeWidget()
        self.feature_tree.setColumnCount(5)
        self.feature_tree.setHeaderLabels(["Feature / subsystem", "Status", "Tier", "Landmarks", "Native signal / unit"])
        self.feature_tree.setAlternatingRowColors(True)
        self.feature_tree.itemChanged.connect(self._on_feature_tree_item_changed)
        self.feature_tree.itemSelectionChanged.connect(self._refresh_feature_detail_panel)
        left_layout.addWidget(self.feature_tree, stretch=1)

        right = QVBoxLayout()
        selected_group = QGroupBox("Selected features")
        selected_layout = QVBoxLayout(selected_group)
        self.feature_selected_label = QLabel("Selected: 0")
        self.feature_selected_label.setObjectName("InfoTitle")
        selected_layout.addWidget(self.feature_selected_label)
        self.feature_selected_box = QPlainTextEdit()
        self.feature_selected_box.setReadOnly(True)
        self.feature_selected_box.setMaximumHeight(120)
        selected_layout.addWidget(self.feature_selected_box)
        right.addWidget(selected_group)

        param_group = QGroupBox("Computation settings")
        param_grid = QGridLayout(param_group)
        self.feature_smoothing_cutoff = QDoubleSpinBox(); self.feature_smoothing_cutoff.setRange(0.5, 20.0); self.feature_smoothing_cutoff.setSingleStep(0.5); self.feature_smoothing_cutoff.setValue(6.0); self.feature_smoothing_cutoff.setSuffix(" Hz")
        self.feature_sigma_extreme = QDoubleSpinBox(); self.feature_sigma_extreme.setRange(3.0, 10.0); self.feature_sigma_extreme.setSingleStep(0.5); self.feature_sigma_extreme.setValue(5.0)
        self.feature_sigma_tight = QDoubleSpinBox(); self.feature_sigma_tight.setRange(1.5, 6.0); self.feature_sigma_tight.setSingleStep(0.25); self.feature_sigma_tight.setValue(3.0)
        self.feature_onset_frac = QDoubleSpinBox(); self.feature_onset_frac.setRange(0.01, 0.40); self.feature_onset_frac.setSingleStep(0.01); self.feature_onset_frac.setValue(0.10)
        self.feature_offset_frac = QDoubleSpinBox(); self.feature_offset_frac.setRange(0.50, 0.99); self.feature_offset_frac.setSingleStep(0.01); self.feature_offset_frac.setValue(0.90)
        self.feature_use_smoothing = QCheckBox("Smooth cleaned trajectories before feature computation"); self.feature_use_smoothing.setChecked(True)
        param_grid.addWidget(QLabel("Low-pass cutoff"), 0, 0); param_grid.addWidget(self.feature_smoothing_cutoff, 0, 1)
        param_grid.addWidget(QLabel("Extreme outlier sigma"), 0, 2); param_grid.addWidget(self.feature_sigma_extreme, 0, 3)
        param_grid.addWidget(QLabel("Tight outlier sigma"), 1, 0); param_grid.addWidget(self.feature_sigma_tight, 1, 1)
        param_grid.addWidget(QLabel("Movement onset fraction"), 1, 2); param_grid.addWidget(self.feature_onset_frac, 1, 3)
        param_grid.addWidget(QLabel("Movement offset fraction"), 2, 0); param_grid.addWidget(self.feature_offset_frac, 2, 1)
        param_grid.addWidget(self.feature_use_smoothing, 2, 2, 1, 2)
        right.addWidget(param_group)

        detail_group = QGroupBox("Feature interpretation")
        detail_layout = QVBoxLayout(detail_group)
        self.feature_detail_box = QTextBrowser()
        self.feature_detail_box.setMinimumHeight(160)
        self.feature_detail_box.setOpenExternalLinks(False)
        detail_layout.addWidget(self.feature_detail_box)
        right.addWidget(detail_group)

        qc_group = QGroupBox("QC requirements for selected features")
        qc_layout = QVBoxLayout(qc_group)
        self.feature_qc_table = QTableWidget(0, 3)
        self.feature_qc_table.setHorizontalHeaderLabels(["Parameter", "Recommended threshold", "Reason"])
        self._fill_table(self.feature_qc_table, pd.DataFrame(QC_FEATURE_REQUIREMENTS), max_rows=20)
        qc_layout.addWidget(self.feature_qc_table)
        right.addWidget(qc_group)

        run_row = QHBoxLayout()
        plan_btn = QPushButton("Write Feature Computation Plan")
        plan_btn.clicked.connect(self.write_feature_computation_plan)
        run_btn = QPushButton("Run Kinematic Feature Computation")
        run_btn.setObjectName("RunButton")
        run_btn.clicked.connect(self.run_feature_computation_stage)
        refresh_btn = QPushButton("Refresh Feature Outputs")
        refresh_btn.clicked.connect(self._load_feature_results)
        run_row.addWidget(plan_btn); run_row.addWidget(run_btn); run_row.addWidget(refresh_btn); run_row.addStretch(1)
        right.addLayout(run_row)

        body.addWidget(left, stretch=3)
        body.addLayout(right, stretch=2)
        layout.addLayout(body, stretch=1)

        self.feature_results_label = QLabel("No kinematic feature table loaded yet.")
        self.feature_results_label.setObjectName("SubtitleLabel")
        layout.addWidget(self.feature_results_label)
        self.feature_results_table = QTableWidget(0, 0)
        self.feature_results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.feature_results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.feature_results_table)

        self._populate_feature_tree()
        return self._scrollable(container)

    def _populate_feature_tree(self) -> None:
        if not hasattr(self, "feature_tree"):
            return
        if not hasattr(self, "selected_feature_ids"):
            self.selected_feature_ids = set(DEFAULT_KINEMATIC_FEATURE_IDS)
        self.feature_tree.blockSignals(True)
        self.feature_tree.clear()
        for group, specs in KINEMATIC_FEATURE_GROUPS.items():
            group_item = QTreeWidgetItem([group, "subsystem", "", "", f"{len(specs)} feature definitions"])
            group_item.setFlags(group_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            checked = any(spec.feature_id in self.selected_feature_ids for spec in specs)
            group_item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
            font = group_item.font(0)
            font.setBold(True)
            group_item.setFont(0, font)
            self.feature_tree.addTopLevelItem(group_item)
            for spec in specs:
                child = QTreeWidgetItem([
                    spec.feature_id,
                    spec.status,
                    spec.tier,
                    ", ".join(map(str, spec.landmarks)),
                    f"{spec.native_signal} | {spec.unit}",
                ])
                child.setData(0, Qt.UserRole, spec.feature_id)
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setCheckState(0, Qt.Checked if spec.feature_id in self.selected_feature_ids else Qt.Unchecked)
                child.setToolTip(0, spec.label)
                child.setToolTip(4, spec.interpretation)
                group_item.addChild(child)
            group_item.setExpanded(True)
        self.feature_tree.resizeColumnToContents(0)
        self.feature_tree.resizeColumnToContents(1)
        self.feature_tree.resizeColumnToContents(2)
        self.feature_tree.blockSignals(False)
        self._refresh_feature_summary()
        self._refresh_feature_detail_panel()

    def _on_feature_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        self.feature_tree.blockSignals(True)
        if item.parent() is None:
            state = item.checkState(0)
            for i in range(item.childCount()):
                item.child(i).setCheckState(0, state)
        selected: set[str] = set()
        for i in range(self.feature_tree.topLevelItemCount()):
            group = self.feature_tree.topLevelItem(i)
            for j in range(group.childCount()):
                child = group.child(j)
                fid = child.data(0, Qt.UserRole)
                if fid and child.checkState(0) == Qt.Checked:
                    selected.add(str(fid))
        self.selected_feature_ids = selected
        self.feature_tree.blockSignals(False)
        self._refresh_feature_summary()
        self._refresh_feature_detail_panel()

    def _set_feature_selection(self, ids: set[str]) -> None:
        self.selected_feature_ids = set(ids)
        self._populate_feature_tree()

    def _feature_spec_by_id(self, feature_id: str):
        for spec in KINEMATIC_FEATURE_SPECS:
            if spec.feature_id == feature_id:
                return spec
        return None

    def _selected_feature_specs(self):
        return [spec for spec in KINEMATIC_FEATURE_SPECS if spec.feature_id in getattr(self, "selected_feature_ids", set(DEFAULT_KINEMATIC_FEATURE_IDS))]

    def _refresh_feature_summary(self) -> None:
        if not hasattr(self, "feature_selected_label"):
            return
        specs = self._selected_feature_specs()
        group_counts: dict[str, int] = {}
        for spec in specs:
            group_counts[spec.group] = group_counts.get(spec.group, 0) + 1
        self.feature_selected_label.setText(f"Selected: {len(specs)} | groups: {len(group_counts)}")
        lines = []
        for group, count in group_counts.items():
            lines.append(f"{group}: {count}")
        required_landmarks = sorted({idx for spec in specs for idx in spec.landmarks})
        if required_landmarks:
            lines.append("")
            lines.append("Required landmarks: " + ", ".join(map(str, required_landmarks)))
        self.feature_selected_box.setPlainText("\n".join(lines) if lines else "No kinematic features selected.")

    def _refresh_feature_detail_panel(self) -> None:
        if not hasattr(self, "feature_detail_box"):
            return
        item = self.feature_tree.currentItem() if hasattr(self, "feature_tree") else None
        spec = None
        if item is not None:
            fid = item.data(0, Qt.UserRole)
            if fid:
                spec = self._feature_spec_by_id(str(fid))
        if spec is None:
            specs = self._selected_feature_specs()
            selected_ids = ", ".join(spec.feature_id for spec in specs[:12])
            if len(specs) > 12:
                selected_ids += ", ..."
            self.feature_detail_box.setHtml(
                "<b>Kinematic feature selector</b><br>"
                "Choose feature groups on the left. Select a row to inspect landmark dependencies, normalization and interpretation.<br><br>"
                f"<b>Currently selected:</b> {len(specs)}<br>"
                f"<b>Feature IDs:</b> {selected_ids or 'none'}"
            )
            return
        self.feature_detail_box.setHtml(
            f"<b>{spec.label}</b><br>"
            f"<b>ID:</b> {spec.feature_id}<br>"
            f"<b>Group:</b> {spec.group}<br>"
            f"<b>Status:</b> {spec.status} | <b>Tier:</b> {spec.tier}<br>"
            f"<b>Landmarks:</b> {', '.join(map(str, spec.landmarks))}<br>"
            f"<b>Native signal:</b> {spec.native_signal}<br>"
            f"<b>Unit:</b> {spec.unit}<br>"
            f"<b>Normalization:</b> {spec.normalization}<br>"
            f"<b>Scalar aggregation:</b> {spec.aggregation}<br><br>"
            f"<b>Interpretation:</b> {spec.interpretation}<br>"
            f"<b>Computation source:</b> {spec.source_function}"
        )

    def _build_aggregation_tab(self) -> QWidget:
        container = QWidget(); layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel(
            "Info",
            "Collapse frame-level kinematic time series into per-video scalar tables using an explicit, auditable aggregation policy. This stage does not delete time-series evidence; it writes a separate aggregation layer for export and review.",
        ))
        group = QGroupBox("Temporal aggregation profile")
        grid = QGridLayout(group)
        self.agg_combo = QComboBox(); self.agg_combo.addItems(list(AGGREGATION_PROFILES.keys()))
        self.agg_combo.setCurrentText("robust_default")
        self.agg_desc = QLabel(AGGREGATION_PROFILES["robust_default"]); self.agg_desc.setWordWrap(True); self.agg_desc.setObjectName("SubtitleLabel")
        self.agg_combo.currentTextChanged.connect(lambda name: self.agg_desc.setText(AGGREGATION_PROFILES.get(name, "")))
        self.agg_include_raw = QCheckBox("Include raw unsmoothed feature signals")
        self.agg_include_raw.setChecked(False)
        self.agg_include_velocity = QCheckBox("Include velocity-derived signals")
        self.agg_include_velocity.setChecked(True)
        self.agg_min_valid = QDoubleSpinBox(); self.agg_min_valid.setRange(0.0, 1.0); self.agg_min_valid.setSingleStep(0.05); self.agg_min_valid.setDecimals(2); self.agg_min_valid.setValue(0.50)
        self.agg_min_detected = QDoubleSpinBox(); self.agg_min_detected.setRange(0.0, 1.0); self.agg_min_detected.setSingleStep(0.05); self.agg_min_detected.setDecimals(2); self.agg_min_detected.setValue(0.60)
        run_btn = QPushButton("Run Temporal Aggregation")
        run_btn.setObjectName("RunButton"); run_btn.clicked.connect(self.run_temporal_aggregation_stage)
        refresh_btn = QPushButton("Refresh Aggregation Outputs")
        refresh_btn.clicked.connect(self._load_aggregation_results)
        grid.addWidget(QLabel("Aggregation profile"), 0, 0); grid.addWidget(self.agg_combo, 0, 1, 1, 2)
        grid.addWidget(QLabel("Interpretation"), 1, 0); grid.addWidget(self.agg_desc, 1, 1, 1, 2)
        grid.addWidget(QLabel("Minimum valid fraction per signal"), 2, 0); grid.addWidget(self.agg_min_valid, 2, 1)
        grid.addWidget(QLabel("Minimum detected-face fraction"), 3, 0); grid.addWidget(self.agg_min_detected, 3, 1)
        grid.addWidget(self.agg_include_raw, 4, 1, 1, 2)
        grid.addWidget(self.agg_include_velocity, 5, 1, 1, 2)
        grid.addWidget(run_btn, 6, 1); grid.addWidget(refresh_btn, 6, 2)
        layout.addWidget(group)
        table = QTableWidget(0, 2); table.setHorizontalHeaderLabels(["Profile", "Recommended use"])
        self._fill_table(table, pd.DataFrame([{"Profile": k, "Recommended use": v} for k, v in AGGREGATION_PROFILES.items()]), max_rows=20)
        layout.addWidget(table)
        self.aggregation_results_label = QLabel("No temporal aggregation table loaded yet."); self.aggregation_results_label.setObjectName("SubtitleLabel")
        layout.addWidget(self.aggregation_results_label)
        self.aggregation_results_table = QTableWidget(0, 0)
        self.aggregation_results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.aggregation_results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.aggregation_results_table)
        return self._scrollable(container)

    def _build_inspector_tab(self) -> QWidget:
        container = QWidget(); layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel("Info", "Preview the current kinematics stage tables and manifests before trusting downstream outputs."))
        btn_row = QHBoxLayout()
        refresh = QPushButton("Refresh Inspector")
        refresh.clicked.connect(self.refresh_inspector)
        open_out = QPushButton("Open Output Project Folder")
        open_out.clicked.connect(self.open_output_root)
        btn_row.addWidget(refresh); btn_row.addWidget(open_out); btn_row.addStretch(1)
        layout.addLayout(btn_row)
        self.inspector_label = QLabel("No table loaded yet."); self.inspector_label.setObjectName("SubtitleLabel")
        layout.addWidget(self.inspector_label)
        self.inspector_table = QTableWidget(0, 0)
        self.inspector_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.inspector_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.inspector_table)
        return self._scrollable(container)

    def _build_reports_tab(self) -> QWidget:
        container = QWidget(); layout = QVBoxLayout(container)
        layout.addWidget(self._info_panel("Info", "Create a reproducible kinematics workflow report and open output artifacts."))
        btn_row = QHBoxLayout()
        create = QPushButton("Create Scaffold Report")
        create.setObjectName("RunButton"); create.clicked.connect(self.run_report_stage)
        open_report = QPushButton("Open HTML Report")
        open_report.clicked.connect(self.open_report)
        open_folder = QPushButton("Open Output Project Folder")
        open_folder.clicked.connect(self.open_output_root)
        btn_row.addWidget(create); btn_row.addWidget(open_report); btn_row.addWidget(open_folder); btn_row.addStretch(1)
        layout.addLayout(btn_row)
        self.report_box = QTextBrowser(); self.report_box.setMinimumHeight(340)
        self.report_box.setHtml("<b>No report generated yet.</b><br>Run the report stage after configuring the workflow outline.")
        layout.addWidget(self.report_box)
        return self._scrollable(container)

    # ---------------------------- ACTIONS ----------------------------
    def browse_input_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select input video folder")
        if path:
            self.input_edit.setText(path); self.input_root = Path(path)

    def browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output project folder")
        if path:
            self.output_edit.setText(path); self.output_root = Path(path)

    def browse_metadata_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select metadata file", filter="Tables (*.csv *.xlsx *.xls);;All files (*.*)")
        if path:
            self.metadata_edit.setText(path); self.metadata_path = Path(path)

    def browse_model_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select MediaPipe FaceLandmarker model", filter="MediaPipe task (*.task);;All files (*.*)")
        if path:
            self.model_path_edit.setText(path)

    def run_project_init(self) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out is None:
            return
        self.output_root = out
        project_dir = out / "kinematics"
        for sub in ["000_ingest", "001_metadata", "002_landmarks", "003_selection", "004_normalization", "005_video_qc", "006_features", "007_aggregation", "008_inspector", "009_reports"]:
            (project_dir / sub).mkdir(parents=True, exist_ok=True)
        manifest = project_dir / "project_manifest.json"
        payload = {
            "schema": "vslp_kinematics_project_v0.71",
            "app_version": APP_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_name": self.project_name_edit.text().strip() or "VSLP Kinematics Project",
            "task": self.task_name_edit.text().strip(),
            "input_video_folder": self.input_edit.text().strip(),
            "output_project_folder": str(out),
            "stage_order": [
                "000_ingest", "001_metadata", "002_landmarks", "003_selection",
                "004_normalization", "005_video_qc", "006_features",
                "007_aggregation", "008_inspector", "009_reports",
            ],
        }
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.stage_records["project"] = StageRecord(status="completed", manifest_path=str(manifest))
        self._refresh_stage_cards()
        if hasattr(self, "project_gate_message"):
            self.project_gate_message.setText(f"Project initialized. Manifest: {manifest}")
        self._log(f"Initialized kinematics project: {project_dir}")

    def run_ingest_stage(self) -> None:
        inp = self._path_or_warn(self.input_edit, "an input video folder")
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if inp is None or out is None:
            return
        self.input_root, self.output_root = inp, out
        cfg = VideoIngestConfig(input_root=inp, output_root=out, recursive=True, extensions=DEFAULT_VIDEO_EXTENSIONS)
        try:
            res = run_ingest(cfg)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ingest failed", str(exc)); return
        self.ingest_manifest_csv = Path(res["manifest_csv"])
        readiness = str(res.get("readiness", "UNKNOWN")).upper()
        stage_status = "completed" if readiness == "PASS" else "completed_with_warnings" if readiness == "REVIEW" else "failed"
        self.stage_records["ingest"] = StageRecord(status=stage_status, manifest_path=str(self.ingest_manifest_csv), summary_path=str(res.get("summary_csv", "")))
        self._refresh_stage_cards()
        self._log(f"Video ingest completed: {res['n_videos']} video(s); readiness={readiness}.")
        if res.get("log_path"):
            self._log(f"Ingest log: {res['log_path']}")
        self._load_ingest_summary()

    def _load_ingest_summary(self) -> None:
        if not self.ingest_manifest_csv or not self.ingest_manifest_csv.exists():
            return
        df = pd.read_csv(self.ingest_manifest_csv)
        summary = summarize_ingest_manifest(df)
        fmt = build_format_summary(df)
        warnings = build_warning_summary(df)

        def fmt_num(value: object, digits: int = 2, suffix: str = "") -> str:
            if value is None or pd.isna(value):
                return "-"
            try:
                val = float(value)
            except Exception:
                return str(value)
            if abs(val - round(val)) < 1e-9:
                return f"{int(round(val))}{suffix}"
            return f"{val:.{digits}f}{suffix}"

        if hasattr(self, "ingest_metric_labels"):
            self.ingest_metric_labels["status"].setText(str(summary.get("readiness", "-")))
            self.ingest_metric_labels["videos"].setText(str(summary.get("n_videos", 0)))
            self.ingest_metric_labels["readable"].setText(str(summary.get("readable_videos", 0)))
            self.ingest_metric_labels["warnings"].setText(str(summary.get("warning_videos", 0)))
            self.ingest_metric_labels["median_fps"].setText(fmt_num(summary.get("median_fps"), 2))
            self.ingest_metric_labels["median_duration"].setText(fmt_num(summary.get("median_duration_sec"), 1, " s"))
            self.ingest_metric_labels["frames"].setText(fmt_num(summary.get("total_estimated_frames"), 0))
            self.ingest_metric_labels["fps_range"].setText(f"{fmt_num(summary.get('min_fps'), 2)} - {fmt_num(summary.get('max_fps'), 2)}")
        if hasattr(self, "ingest_next_step_label"):
            self.ingest_next_step_label.setText(str(summary.get("next_step", "No ingest results yet.")))
        readiness = str(summary.get("readiness", "UNKNOWN"))
        self.ingest_summary_label.setText(
            f"{summary.get('n_videos', 0)} candidate videos detected; "
            f"{summary.get('readable_videos', 0)} readable; "
            f"{summary.get('warning_videos', 0)} with warnings. Readiness: {readiness}."
        )
        if hasattr(self, "project_gate_message"):
            self.project_gate_message.setText(f"Ingest readiness: {readiness}. {summary.get('next_step', '')}")

        display_fmt = fmt.copy()
        for col in ["median_fps", "median_duration_sec"]:
            if col in display_fmt.columns:
                display_fmt[col] = pd.to_numeric(display_fmt[col], errors="coerce").round(2)
        for col in ["estimated_frames", "warnings"]:
            if col in display_fmt.columns:
                display_fmt[col] = pd.to_numeric(display_fmt[col], errors="coerce").fillna(0).astype(int)
        self._fill_table(self.ingest_format_table, display_fmt, max_rows=80)

        if warnings.empty:
            if hasattr(self, "ingest_warning_label"):
                self.ingest_warning_label.setText("No structural ingest warnings.")
            if hasattr(self, "ingest_warning_table"):
                self._fill_table(self.ingest_warning_table, warnings, max_rows=80)
        else:
            if hasattr(self, "ingest_warning_label"):
                self.ingest_warning_label.setText(f"{len(warnings)} video(s) need review before trusting downstream features.")
            if hasattr(self, "ingest_warning_table"):
                self._fill_table(self.ingest_warning_table, warnings, max_rows=120)

    def run_metadata_stage(self) -> None:
        if not self.ingest_manifest_csv or not self.ingest_manifest_csv.exists():
            QMessageBox.warning(self, "Missing ingest", "Run video ingest before metadata linking."); return
        meta = Path(self.metadata_edit.text()).expanduser().resolve() if self.metadata_edit.text().strip() else None
        try:
            out = link_metadata(self.ingest_manifest_csv, meta, Path(self.output_edit.text()).expanduser().resolve())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Metadata failed", str(exc)); return
        self.stage_records["metadata"] = StageRecord(status="completed", manifest_path=str(out))
        self._refresh_stage_cards(); self._log(f"Metadata link preview written: {out}")

    def apply_landmark_preset(self, name: str) -> None:
        vals = LANDMARK_PRESETS.get(name, ())
        self.landmark_text.setPlainText(", ".join(map(str, vals)))
        if hasattr(self, "landmark_canvas"):
            self.landmark_canvas.set_selected(vals)
        self._update_selected_landmark_feedback()

    def _canvas_selection_changed(self, text: str) -> None:
        if hasattr(self, "landmark_text"):
            self.landmark_text.setPlainText(text)
        self._update_selected_landmark_feedback()

    def clear_visual_landmarks(self) -> None:
        if hasattr(self, "landmark_canvas"):
            self.landmark_canvas.set_selected([])
        self.landmark_text.setPlainText("")
        self._update_selected_landmark_feedback()

    def add_landmark_region(self, region: str) -> None:
        current = set()
        try:
            current = set(parse_int_list(self.landmark_text.toPlainText()))
        except Exception:
            pass
        current.update(LandmarkMeshCanvas.REGION_SETS.get(region, set()))
        # Keep the set readable; selected values outside 0-477 are not expected but guarded.
        vals = sorted(v for v in current if 0 <= int(v) < 478)
        self.landmark_text.setPlainText(", ".join(map(str, vals)))
        if hasattr(self, "landmark_canvas"):
            self.landmark_canvas.set_selected(vals)
        self._update_selected_landmark_feedback()

    def _set_selection_dashboard_value(self, key: str, value: object) -> None:
        if hasattr(self, "selection_metric_labels") and key in self.selection_metric_labels:
            self.selection_metric_labels[key].setText("-" if value is None else str(value))

    def _update_selected_landmark_feedback(self) -> None:
        if not hasattr(self, "selected_landmark_table"):
            return
        try:
            vals = list(parse_int_list(self.landmark_text.toPlainText()))
        except Exception:
            vals = []
        diagnostics = analyze_landmark_selection(vals)
        rows = []
        regions = {}
        for idx in diagnostics.selected_landmarks:
            region = LandmarkMeshCanvas.region_for(int(idx))
            regions[region] = regions.get(region, 0) + 1
            rows.append({
                "Landmark": int(idx),
                "Region": region,
                "Meaning / use": LandmarkMeshCanvas.label_for(int(idx)),
                "Status": "selected",
            })
        self._fill_table(self.selected_landmark_table, pd.DataFrame(rows), max_rows=120)

        req_rows = []
        for result in diagnostics.requirement_results:
            req_rows.append({
                "Requirement": result.label,
                "Status": result.status,
                "Missing required": ", ".join(map(str, result.missing_required)),
                "Missing recommended": ", ".join(map(str, result.missing_recommended)),
                "Reason": result.reason,
            })
        if hasattr(self, "selection_requirement_table"):
            self._fill_table(self.selection_requirement_table, pd.DataFrame(req_rows), max_rows=30)

        anchor_result = next((r for r in diagnostics.requirement_results if r.requirement_id == "normalization_intercanthal"), None)
        mouth_result = next((r for r in diagnostics.requirement_results if r.requirement_id == "mouth_aperture"), None)
        jaw_result = next((r for r in diagnostics.requirement_results if r.requirement_id == "jaw_lower_face"), None)
        self._set_selection_dashboard_value("preset", self.preset_combo.currentText() if hasattr(self, "preset_combo") else "custom")
        self._set_selection_dashboard_value("selected", diagnostics.n_selected)
        self._set_selection_dashboard_value("regions", len(diagnostics.region_counts))
        self._set_selection_dashboard_value("anchors", anchor_result.status if anchor_result else "-")
        self._set_selection_dashboard_value("mouth", mouth_result.status if mouth_result else "-")
        self._set_selection_dashboard_value("jaw", jaw_result.status if jaw_result else "-")
        self._set_selection_dashboard_value("status", diagnostics.status)
        self._set_selection_dashboard_value("next_step", "Normalize" if diagnostics.status == "Complete" else "Review")
        if hasattr(self, "selection_next_step_label"):
            self.selection_next_step_label.setText(diagnostics.next_step)

        if hasattr(self, "selection_feedback_label"):
            self.selection_feedback_label.setText(
                f"Selected {diagnostics.n_selected} landmarks. Region coverage: "
                + (", ".join(f"{k}={v}" for k, v in diagnostics.region_counts.items()) if diagnostics.region_counts else "none")
                + f". Selection status: {diagnostics.status}. "
                + diagnostics.next_step
            )

    def _landmark_tables_dir(self) -> Path | None:
        out_text = self.output_edit.text().strip()
        if not out_text:
            return None
        return Path(out_text).expanduser().resolve() / "kinematics" / "002_landmarks" / "tables"

    def _landmarks_manifest_path(self) -> Path | None:
        tables = self._landmark_tables_dir()
        if tables is None:
            return None
        path = tables / "landmarks_manifest.csv"
        return path if path.exists() else None

    def refresh_landmark_video_choices(self) -> None:
        if not hasattr(self, "landmark_video_combo"):
            return
        self.landmark_video_combo.clear()
        self._landmark_video_rows = []
        manifest = self._landmarks_manifest_path()
        if manifest is None:
            QMessageBox.information(self, "No landmark manifest", "Run Landmarks -> Run MediaPipe Landmark Extraction first.")
            return
        try:
            df = pd.read_csv(manifest)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not read landmark manifest", str(exc)); return
        for _, row in df.iterrows():
            out_csv = str(row.get("output_csv", ""))
            if not out_csv or not Path(out_csv).exists():
                continue
            video_id = str(row.get("video_id") or Path(out_csv).stem.replace("-lmks", ""))
            status = str(row.get("status", ""))
            frac = pd.to_numeric(pd.Series([row.get("face_detected_fraction")]), errors="coerce").iloc[0]
            label = f"{video_id}  |  {status}  |  detected={frac*100:.1f}%" if pd.notna(frac) else f"{video_id}  |  {status}"
            self._landmark_video_rows.append(dict(row))
            self.landmark_video_combo.addItem(label)
        if not self._landmark_video_rows:
            QMessageBox.warning(self, "No usable landmark CSVs", "The manifest exists, but no *-lmks.csv files were found. Check the Landmarks tab result table.")
        else:
            if self.landmark_video_combo.count() > 0:
                self.landmark_video_combo.setCurrentIndex(0)
                self._landmark_video_changed(0)
            self._log(f"Loaded {len(self._landmark_video_rows)} landmark video option(s) from {manifest}")

    def _update_landmark_frame_label(self) -> None:
        if not hasattr(self, "landmark_frame_label") or not hasattr(self, "landmark_frame_spin"):
            return
        current = int(self.landmark_frame_spin.value())
        maximum = int(self.landmark_frame_spin.maximum())
        self.landmark_frame_label.setText(f"Frame {current} / {maximum}")

    def _set_landmark_frame_value(self, frame_idx: int, *, load: bool = False) -> None:
        if not hasattr(self, "landmark_frame_spin"):
            return
        lo = int(self.landmark_frame_spin.minimum())
        hi = int(self.landmark_frame_spin.maximum())
        target = min(hi, max(lo, int(frame_idx)))
        self.landmark_frame_spin.blockSignals(True)
        self.landmark_frame_spin.setValue(target)
        self.landmark_frame_spin.blockSignals(False)
        self._sync_landmark_frame_combo_to_value(target)
        self._update_landmark_frame_label()
        if load:
            self.load_real_frame_landmark_overlay()

    def use_representative_landmark_frame(self) -> None:
        """Jump to the middle detected-face frame and load it.

        This is more reliable for landmark selection than manual +/- buttons: it
        uses the detection mask already produced by MediaPipe and avoids landing
        on no-face or sparse-overlay frames.
        """
        rec = self._selected_landmark_video_row()
        if rec is None:
            return
        landmark_csv = Path(str(rec.get("output_csv", ""))).expanduser()
        if not landmark_csv.exists():
            QMessageBox.warning(self, "Missing landmark CSV", f"Could not find landmark CSV:\n{landmark_csv}")
            return
        try:
            df = pd.read_csv(landmark_csv, usecols=lambda c: c in {"frame", "face_detected"})
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not read landmark frames", str(exc))
            return
        if df.empty:
            return
        if "face_detected" in df.columns:
            detected = df["face_detected"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
            candidates = df.loc[detected].copy()
        else:
            candidates = df.copy()
        if candidates.empty:
            QMessageBox.warning(self, "No detected-face frames", "This landmark CSV contains no face_detected=True rows. Try another video or rerun landmark extraction.")
            return
        chosen = candidates.iloc[len(candidates) // 2]
        frame_idx = int(chosen.get("frame", candidates.index[len(candidates) // 2]))
        self._set_landmark_frame_value(frame_idx, load=True)

    def _landmark_frame_choice_changed(self, idx: int) -> None:
        """Move the exact-frame box to a selected detected-frame bookmark."""
        if idx < 0 or not hasattr(self, "landmark_frame_combo"):
            return
        frame = self.landmark_frame_combo.currentData()
        if frame is None:
            return
        try:
            self._set_landmark_frame_value(int(frame), load=False)
        except Exception:
            return

    def _sync_landmark_frame_combo_to_value(self, frame_idx: int) -> None:
        if not hasattr(self, "landmark_frame_combo"):
            return
        for i in range(self.landmark_frame_combo.count()):
            try:
                if int(self.landmark_frame_combo.itemData(i)) == int(frame_idx):
                    self.landmark_frame_combo.blockSignals(True)
                    self.landmark_frame_combo.setCurrentIndex(i)
                    self.landmark_frame_combo.blockSignals(False)
                    return
            except Exception:
                continue

    def _landmark_spin_changed(self, value: int) -> None:
        """Update frame label only; loading is explicit for reproducible review."""
        self._sync_landmark_frame_combo_to_value(int(value))
        self._update_landmark_frame_label()

    def _schedule_landmark_frame_reload(self) -> None:
        """Kept for compatibility with older tests; frame loading is now explicit."""
        return

    def _frame_choices_from_landmarks(self, landmark_csv: Path, max_frame_hint: int) -> list[tuple[str, int]]:
        try:
            df = pd.read_csv(landmark_csv, usecols=lambda c: c in {"frame", "face_detected", "timestamp_ms"})
        except Exception:
            return [("Middle frame", max(0, int(max_frame_hint) // 2))]
        if df.empty:
            return [("Frame 0", 0)]
        if "frame" in df.columns:
            frames = pd.to_numeric(df["frame"], errors="coerce").dropna().astype(int)
        else:
            frames = pd.Series(range(len(df)), dtype=int)
        if frames.empty:
            return [("Frame 0", 0)]
        if "face_detected" in df.columns:
            detected_mask = df["face_detected"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
            detected_frames = pd.to_numeric(df.loc[detected_mask, "frame"] if "frame" in df.columns else pd.Series(df.index[detected_mask]), errors="coerce").dropna().astype(int).tolist()
        else:
            detected_frames = frames.tolist()
        base_frames = detected_frames if detected_frames else frames.tolist()
        base_frames = sorted(set(int(v) for v in base_frames))
        if not base_frames:
            base_frames = [0]
        quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
        labels = ["Early detected", "25% detected", "Middle detected", "75% detected", "Late detected"] if detected_frames else ["Early frame", "25% frame", "Middle frame", "75% frame", "Late frame"]
        choices: list[tuple[str, int]] = []
        for label, q in zip(labels, quantiles):
            pos = int(round(q * (len(base_frames) - 1)))
            choices.append((f"{label} - frame {base_frames[pos]}", base_frames[pos]))
        choices.insert(0, (f"First {'detected' if detected_frames else 'available'} - frame {base_frames[0]}", base_frames[0]))
        choices.append((f"Last {'detected' if detected_frames else 'available'} - frame {base_frames[-1]}", base_frames[-1]))
        deduped: list[tuple[str, int]] = []
        seen: set[int] = set()
        for label, frame in choices:
            if frame not in seen:
                deduped.append((label, frame))
                seen.add(frame)
        return deduped

    def _populate_landmark_frame_choices(self, row: dict) -> None:
        if not hasattr(self, "landmark_frame_combo") or not hasattr(self, "landmark_frame_spin"):
            return
        n_frames = int(pd.to_numeric(pd.Series([row.get("n_frames")]), errors="coerce").fillna(0).iloc[0])
        max_frame = max(0, n_frames - 1)
        landmark_csv = Path(str(row.get("output_csv", ""))).expanduser()
        choices = self._frame_choices_from_landmarks(landmark_csv, max_frame) if landmark_csv.exists() else [("Middle frame", max_frame // 2)]
        if choices:
            max_frame = max(max_frame, max(frame for _, frame in choices))
        self.landmark_frame_spin.blockSignals(True)
        self.landmark_frame_spin.setRange(0, max_frame)
        self.landmark_frame_spin.blockSignals(False)
        self.landmark_frame_combo.blockSignals(True)
        self.landmark_frame_combo.clear()
        for label, frame in choices:
            self.landmark_frame_combo.addItem(label, int(frame))
        middle_idx = 0
        for i in range(self.landmark_frame_combo.count()):
            if "Middle" in self.landmark_frame_combo.itemText(i):
                middle_idx = i
                break
        self.landmark_frame_combo.setCurrentIndex(middle_idx)
        frame = int(self.landmark_frame_combo.itemData(middle_idx)) if self.landmark_frame_combo.count() else 0
        self.landmark_frame_combo.blockSignals(False)
        self._set_landmark_frame_value(frame, load=False)
        self._update_landmark_frame_label()

    def _landmark_video_changed(self, idx: int) -> None:
        rows = getattr(self, "_landmark_video_rows", [])
        if idx < 0 or idx >= len(rows):
            return
        row = rows[idx]
        self._populate_landmark_frame_choices(row)
        if hasattr(self, "selection_feedback_label"):
            self.selection_feedback_label.setText("Video changed. Choose a detected-frame bookmark or exact frame, then click Load Frame. Loading is explicit so frame review is reproducible.")

    def _toggle_auto_zoom_landmark_canvas(self, checked: bool) -> None:
        if hasattr(self, "landmark_canvas"):
            self.landmark_canvas.auto_zoom_face = bool(checked)
            if checked and self.landmark_canvas.points:
                self.landmark_canvas.zoom_to_face()

    def _selected_landmark_video_row(self) -> dict | None:
        if not hasattr(self, "_landmark_video_rows") or not self._landmark_video_rows:
            self.refresh_landmark_video_choices()
        idx = self.landmark_video_combo.currentIndex() if hasattr(self, "landmark_video_combo") else -1
        if idx < 0 or not getattr(self, "_landmark_video_rows", None):
            return None
        return self._landmark_video_rows[min(idx, len(self._landmark_video_rows) - 1)]

    def _resolve_landmark_source_path(self, rec: dict) -> Path | None:
        """Resolve a source video path from the landmark manifest row."""
        for key in ("source_path", "video_path", "path"):
            value = str(rec.get(key, "") or "").strip()
            if value and value.lower() != "nan":
                path = Path(value).expanduser()
                if path.exists():
                    return path
        rel = str(rec.get("relative_path", "") or "").strip()
        if rel and rel.lower() != "nan":
            for root in [getattr(self, "input_root", None), Path(self.input_edit.text()).expanduser() if hasattr(self, "input_edit") and self.input_edit.text().strip() else None]:
                if root is None:
                    continue
                candidate = Path(root).expanduser() / rel
                if candidate.exists():
                    return candidate
        return None

    def _load_video_frame_pixmap(self, source_path: Path, frame_idx: int) -> QPixmap | None:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "OpenCV unavailable", f"Cannot load the source video frame because OpenCV is not available:\n{exc}")
            return None
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                return None
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            return QPixmap.fromImage(qimg)
        finally:
            cap.release()

    def _points_for_requested_frame(self, landmark_csv: Path, requested_frame: int) -> tuple[dict[int, tuple[float, float]], int, bool, str]:
        df = pd.read_csv(landmark_csv)
        if df.empty:
            raise ValueError(f"Landmark CSV is empty: {landmark_csv}")
        if "frame" in df.columns:
            frames = pd.to_numeric(df["frame"], errors="coerce")
            exact = df[frames == int(requested_frame)]
        else:
            exact = df.iloc[[min(max(0, int(requested_frame)), len(df)-1)]]
        if exact.empty:
            exact = df.iloc[[min(max(0, int(requested_frame)), len(df)-1)]]
        face_detected = False
        if "face_detected" in exact.columns:
            face_detected = str(exact["face_detected"].iloc[0]).lower() in {"true", "1", "yes"}
        used_note = "requested frame"
        if not face_detected and "face_detected" in df.columns:
            detected_mask = df["face_detected"].astype(str).str.lower().isin(["true", "1", "yes"])
            detected_df = df[detected_mask].copy()
            if not detected_df.empty:
                if "frame" in detected_df.columns:
                    detected_df["_dist"] = (pd.to_numeric(detected_df["frame"], errors="coerce") - int(requested_frame)).abs()
                    exact = detected_df.sort_values("_dist").iloc[[0]].drop(columns=["_dist"])
                else:
                    exact = detected_df.iloc[[0]]
                face_detected = True
                used_note = "nearest detected-face frame"
        row = exact.iloc[0]
        actual_frame = int(row.get("frame", requested_frame)) if pd.notna(row.get("frame", requested_frame)) else int(requested_frame)
        points: dict[int, tuple[float, float]] = {}
        max_idx = 0
        for col in df.columns:
            if col.endswith("_x"):
                try:
                    max_idx = max(max_idx, int(col[:-2]))
                except ValueError:
                    pass
        for i in range(max_idx + 1):
            xcol, ycol = f"{i}_x", f"{i}_y"
            if xcol not in df.columns or ycol not in df.columns:
                continue
            x = pd.to_numeric(pd.Series([row.get(xcol)]), errors="coerce").iloc[0]
            y = pd.to_numeric(pd.Series([row.get(ycol)]), errors="coerce").iloc[0]
            if pd.notna(x) and pd.notna(y):
                points[i] = (float(max(0.0, min(1.0, x))), float(max(0.0, min(1.0, y))))
        return points, actual_frame, face_detected, used_note

    def load_real_frame_landmark_overlay(self) -> None:
        rec = self._selected_landmark_video_row()
        if rec is None:
            return
        landmark_csv = Path(str(rec.get("output_csv", ""))).expanduser()
        source_path = self._resolve_landmark_source_path(rec)
        if not landmark_csv.exists():
            QMessageBox.warning(self, "Missing landmark CSV", f"Could not find landmark CSV:\n{landmark_csv}"); return
        requested = int(self.landmark_frame_spin.value()) if hasattr(self, "landmark_frame_spin") else 0
        try:
            points, frame_idx, face_detected, used_note = self._points_for_requested_frame(landmark_csv, requested)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not load landmarks for frame", str(exc)); return
        if len(points) < 20:
            QMessageBox.warning(self, "Sparse overlay", f"Only {len(points)} usable landmarks were available for frame {frame_idx}. Try another frame or inspect Landmark extraction coverage.")
        pixmap = self._load_video_frame_pixmap(source_path, frame_idx) if source_path is not None else None
        video_id = str(rec.get("video_id") or landmark_csv.stem.replace("-lmks", ""))
        self.landmark_canvas.set_overlay(
            pixmap,
            points,
            f"{video_id} / {landmark_csv.name}",
            f"Video: {video_id} | frame {frame_idx} ({used_note}) | face_detected={face_detected} | landmarks visible={len(points)}",
        )
        try:
            current = parse_int_list(self.landmark_text.toPlainText())
        except Exception:
            current = []
        self.landmark_canvas.set_selected(current)
        # The requested frame may fall back to the nearest detected-face frame.
        # Update controls without triggering another reload loop.
        if hasattr(self, "landmark_frame_spin"):
            self.landmark_frame_spin.blockSignals(True)
            self.landmark_frame_spin.setValue(frame_idx)
            self.landmark_frame_spin.blockSignals(False)
            self._sync_landmark_frame_combo_to_value(frame_idx)
            self._update_landmark_frame_label()
        self._landmark_overlay_loaded = True
        self._update_selected_landmark_feedback()
        if pixmap is None and hasattr(self, "selection_feedback_label"):
            self.selection_feedback_label.setText(
                f"Loaded landmark coordinates for frame {frame_idx}, but the source video pixels could not be opened. Check source_path/relative_path in the landmark manifest."
            )
        self._log(f"Loaded real-frame MediaPipe overlay: video={video_id}, frame={frame_idx}, points={len(points)}, source={source_path}")

    def _set_landmark_dashboard_value(self, key: str, value: object) -> None:
        if hasattr(self, "landmark_metric_labels") and key in self.landmark_metric_labels:
            self.landmark_metric_labels[key].setText("-" if value is None else str(value))

    def _update_landmark_dashboard_from_manifest(self, manifest_csv: Path | None = None) -> None:
        """Refresh landmark extraction dashboard cards from the current manifest."""
        manifest = manifest_csv or self._landmarks_manifest_path()
        status = mediapipe_environment_status()
        runtime_text = "Ready" if status.opencv_available and status.mediapipe_available else "Missing"
        self._set_landmark_dashboard_value("runtime", runtime_text)
        if manifest is None or not Path(manifest).exists():
            self._set_landmark_dashboard_value("videos", "0")
            self._set_landmark_dashboard_value("ok", "0")
            self._set_landmark_dashboard_value("errors", "0")
            self._set_landmark_dashboard_value("mean_detection", "-")
            self._set_landmark_dashboard_value("status", "Not run")
            self._set_landmark_dashboard_value("next_step", "Plan/run")
            if hasattr(self, "landmark_next_step_label"):
                if self._require_ingest_manifest_silent() is None:
                    self.landmark_next_step_label.setText("Run Setup -> Video Ingest before landmark extraction.")
                else:
                    self.landmark_next_step_label.setText("Ingest is available. Write the landmark extraction plan or run MediaPipe extraction.")
            return
        try:
            df = pd.read_csv(manifest)
        except Exception:
            self._set_landmark_dashboard_value("status", "Review")
            if hasattr(self, "landmark_next_step_label"):
                self.landmark_next_step_label.setText(f"Could not read landmark manifest: {manifest}")
            return
        n_videos = int(len(df))
        statuses = df.get("status", pd.Series(dtype=str)).astype(str).str.lower() if not df.empty else pd.Series(dtype=str)
        n_ok = int(statuses.isin(["ok", "skipped_existing"]).sum()) if not df.empty else 0
        n_error = int(statuses.eq("error").sum()) if not df.empty else 0
        mean_det = pd.to_numeric(df.get("face_detected_fraction"), errors="coerce").mean() if not df.empty else float("nan")
        if n_videos == 0:
            stage_status = "Not run"
            next_step = "Run MediaPipe extraction."
        elif n_error > 0:
            stage_status = "Review"
            next_step = "Review failed videos and extraction errors before selection/normalization."
        else:
            stage_status = "Complete"
            next_step = "Next: open Landmark Selection and inspect the real-frame overlay."
        self._set_landmark_dashboard_value("videos", n_videos)
        self._set_landmark_dashboard_value("ok", n_ok)
        self._set_landmark_dashboard_value("errors", n_error)
        self._set_landmark_dashboard_value("mean_detection", "-" if pd.isna(mean_det) else f"{mean_det * 100:.1f}%")
        self._set_landmark_dashboard_value("status", stage_status)
        self._set_landmark_dashboard_value("next_step", "Selection" if stage_status == "Complete" else "Review")
        if hasattr(self, "landmark_next_step_label"):
            self.landmark_next_step_label.setText(next_step)

    def _require_ingest_manifest_silent(self) -> Path | None:
        if self.ingest_manifest_csv and self.ingest_manifest_csv.exists():
            return self.ingest_manifest_csv
        out_text = self.output_edit.text().strip() if hasattr(self, "output_edit") else ""
        if out_text:
            candidate = Path(out_text).expanduser().resolve() / "kinematics" / "000_ingest" / "tables" / "video_ingest_manifest.csv"
            if candidate.exists():
                self.ingest_manifest_csv = candidate
                return candidate
        return None

    # Backward-compatible alias for older buttons/docs.
    def load_mesh_from_latest_landmarks(self) -> None:
        self.refresh_landmark_video_choices()
        self.load_real_frame_landmark_overlay()

    def save_landmark_mesh_preview(self) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out is None or not hasattr(self, "landmark_canvas"):
            return
        fig_dir = out / "kinematics" / "003_selection" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        path = fig_dir / "selected_landmark_video_overlay_preview.png"
        self.landmark_canvas.grab().save(str(path))
        QMessageBox.information(self, "Overlay preview saved", f"Saved current real-frame landmark overlay preview:\n{path}")
        self._log(f"Saved landmark overlay preview: {path}")
    def _landmark_config(self) -> LandmarkRunConfig:
        return LandmarkRunConfig(
            model_path=self.model_path_edit.text().strip() or "models/face_landmarker.task",
            selected_preset=self.preset_combo.currentText() if hasattr(self, "preset_combo") else "ALS oral-motor core 15",
            selected_landmarks=parse_int_list(self.landmark_text.toPlainText()),
            min_face_detection_confidence=float(self.detection_conf_spin.value()),
            min_face_presence_confidence=float(self.presence_conf_spin.value()),
            min_tracking_confidence=float(self.tracking_conf_spin.value()),
            normalization_method=self.norm_combo.currentText() if hasattr(self, "norm_combo") else "intercanthal_distance",
            n_landmarks=int(self.expected_landmarks_spin.value()) if hasattr(self, "expected_landmarks_spin") else 478,
        )

    def _require_ingest_manifest(self) -> Path | None:
        if self.ingest_manifest_csv and self.ingest_manifest_csv.exists():
            return self.ingest_manifest_csv
        out_text = self.output_edit.text().strip()
        if out_text:
            candidate = Path(out_text).expanduser().resolve() / "kinematics" / "000_ingest" / "tables" / "video_ingest_manifest.csv"
            if candidate.exists():
                self.ingest_manifest_csv = candidate
                return candidate
        QMessageBox.warning(self, "Missing ingest", "Run Setup -> Run Video Ingest before landmark extraction.")
        return None

    def _update_landmark_runtime_label(self) -> None:
        status = mediapipe_environment_status()
        self.landmark_runtime_label.setText(
            f"OpenCV: {'available' if status.opencv_available else 'missing'}"
            f"{f' ({status.opencv_version})' if status.opencv_version else ''}; "
            f"MediaPipe: {'available' if status.mediapipe_available else 'missing'}"
            f"{f' ({status.mediapipe_version})' if status.mediapipe_version else ''}. "
            f"{status.message}"
        )
        if hasattr(self, "landmark_metric_labels"):
            self._set_landmark_dashboard_value("runtime", "Ready" if status.opencv_available and status.mediapipe_available else "Missing")

    def bootstrap_mediapipe_runtime_stage(self) -> None:
        self._log("Checking/installing MediaPipe runtime into the active Python environment...")
        try:
            result = bootstrap_mediapipe_runtime()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "MediaPipe runtime setup failed", str(exc))
            self._log(f"MediaPipe runtime setup failed: {exc}")
            return
        self._update_landmark_runtime_label()
        if result.get("ok"):
            QMessageBox.information(
                self,
                "MediaPipe runtime ready",
                "opencv-python and mediapipe are available in this environment. "
                "If this was a fresh install, restart the GUI if extraction still reports missing imports.",
            )
        else:
            QMessageBox.warning(
                self,
                "MediaPipe runtime incomplete",
                "The install command finished but the runtime is still not fully available. "
                "Open the run log for details, or restart the GUI and try again.",
            )
        self._log(f"MediaPipe runtime setup command: {result.get('command')}")
        if result.get("stderr"):
            self._log("pip stderr tail: " + str(result.get("stderr"))[-1200:])

    def download_landmarker_model(self) -> None:
        try:
            path = download_default_model(self._path_or_warn(self.output_edit, "an output project folder") or Path.cwd(), self._landmark_config())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Model download failed", str(exc)); return
        self.model_path_edit.setText(str(path))
        QMessageBox.information(self, "Model ready", f"FaceLandmarker model is available at:\n{path}")
        self._log(f"MediaPipe FaceLandmarker model ready: {path}")

    def run_landmark_plan_stage(self) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        manifest = self._require_ingest_manifest()
        if out is None or manifest is None:
            return
        cfg = self._landmark_config()
        path = write_landmark_plan(out, cfg, manifest_csv=manifest)
        self.stage_records["landmarks"] = StageRecord(status="planned", manifest_path=str(path))
        self._set_landmark_dashboard_value("plan", "Written")
        self._update_landmark_runtime_label()
        self._update_landmark_dashboard_from_manifest(None)
        self._refresh_stage_cards(); self._log(f"Landmark extraction plan written: {path}")

    def run_landmark_extraction_stage(self) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        manifest = self._require_ingest_manifest()
        if out is None or manifest is None:
            return
        cfg = self._landmark_config()
        self._update_landmark_runtime_label()
        status = mediapipe_environment_status()
        auto_prepare = bool(getattr(self, "auto_prepare_mediapipe_check", None) and self.auto_prepare_mediapipe_check.isChecked())
        if (not status.opencv_available or not status.mediapipe_available) and auto_prepare:
            reply = QMessageBox.question(
                self,
                "Install MediaPipe runtime?",
                "OpenCV and/or MediaPipe are missing from this VSLP Python environment.\n\n"
                "The GUI can install the required packages into the active virtual environment now:\n"
                "  opencv-python\n"
                "  mediapipe\n\n"
                "This can take a few minutes and requires internet access. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
            self.bootstrap_mediapipe_runtime_stage()
            status = mediapipe_environment_status()
        if not status.opencv_available or not status.mediapipe_available:
            QMessageBox.critical(
                self,
                "MediaPipe runtime missing",
                "Real landmark extraction cannot run until opencv-python and mediapipe are installed.\n\n"
                "Click 'Install / Verify MediaPipe Runtime' on the Landmarks tab, or run:\n"
                "python -m pip install opencv-python mediapipe",
            )
            return
        try:
            res = run_mediapipe_landmarks(out, cfg, manifest)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "MediaPipe extraction failed", str(exc)); return
        manifest_csv = Path(res["manifest_csv"])
        self.stage_records["landmarks"] = StageRecord(status="completed" if int(res.get("n_error", 0)) == 0 else "completed_with_warnings", manifest_path=str(manifest_csv))
        self._refresh_stage_cards()
        self._log(f"MediaPipe landmarks completed: {res['n_ok']} ok, {res['n_error']} error, {res['n_skipped_existing']} skipped. Manifest: {manifest_csv}")
        self._load_landmark_summary(manifest_csv)
        QMessageBox.information(self, "Landmark extraction complete", f"Processed {res['n_ok']} video(s).\nManifest:\n{manifest_csv}")

    def _load_landmark_summary(self, manifest_csv: Path) -> None:
        if not manifest_csv.exists():
            return
        df = pd.read_csv(manifest_csv)
        if df.empty:
            self.landmark_summary_table.setRowCount(0); self.landmark_summary_table.setColumnCount(0); return
        preview = pd.DataFrame({
            "Video": df.get("video_id", ""),
            "Status": df.get("status", ""),
            "Frames": df.get("n_frames", ""),
            "Face frames": df.get("n_faces_detected", ""),
            "Dropped": df.get("n_dropped", ""),
            "Detected %": (pd.to_numeric(df.get("face_detected_fraction"), errors="coerce") * 100).round(1),
            "Output": df.get("output_csv", ""),
        })
        self._fill_table(self.landmark_summary_table, preview, max_rows=100)
        n_error = int((df.get("status", pd.Series(dtype=str)).astype(str).str.lower() == "error").sum()) if not df.empty else 0
        mean_det = pd.to_numeric(df.get("face_detected_fraction"), errors="coerce").mean() if not df.empty else float("nan")
        if hasattr(self, "landmark_summary_label"):
            self.landmark_summary_label.setText(
                f"Loaded landmark manifest: {len(df)} video(s), {n_error} error(s), "
                f"mean detected-frame fraction={'-' if pd.isna(mean_det) else f'{mean_det * 100:.1f}%'}"
            )
        self._set_landmark_dashboard_value("plan", "Written")
        self._update_landmark_dashboard_from_manifest(manifest_csv)

    def run_selection_stage(self) -> None:
        try:
            self.landmark_indices = parse_int_list(self.landmark_text.toPlainText())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Invalid landmarks", str(exc)); return
        if hasattr(self, "landmark_canvas"):
            self.landmark_canvas.set_selected(self.landmark_indices)
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out:
            preview = out / "kinematics" / "003_selection" / "figures" / "selected_landmark_mesh_preview.png"
            preview.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(self, "landmark_canvas"):
                self.landmark_canvas.grab().save(str(preview))
            outputs = write_selected_landmarks(
                out,
                self.landmark_indices,
                preset=self.preset_combo.currentText(),
                mesh_source=getattr(getattr(self, "landmark_canvas", None), "source_label", "not_available"),
                preview_png=str(preview) if preview.exists() else None,
                app_version=APP_VERSION,
            )
            diagnostics = analyze_landmark_selection(self.landmark_indices)
            stage_status = "completed" if diagnostics.status == "Complete" else "completed_with_warnings"
            self.stage_records["selection"] = StageRecord(status=stage_status, manifest_path=str(outputs["selected_json"]), summary_path=str(outputs["summary_csv"]))
            self._refresh_stage_cards()
            self._update_selected_landmark_feedback()
            self._log(f"Selected {len(self.landmark_indices)} landmarks: {outputs['selected_json']} | status={diagnostics.status}")

    def _set_normalization_dashboard_value(self, key: str, value: object) -> None:
        if hasattr(self, "norm_metric_labels") and key in self.norm_metric_labels:
            self.norm_metric_labels[key].setText("-" if value is None else str(value))

    def _format_landmark_tuple(self, values: object) -> str:
        vals = tuple(values or ()) if not isinstance(values, str) else ()
        return ", ".join(map(str, vals)) if vals else "None"

    def _update_normalization_method_panel(self, method: str | None = None) -> None:
        method = method or (self.norm_combo.currentText() if hasattr(self, "norm_combo") else "intercanthal_distance")
        details = NORMALIZATION_METHOD_DETAILS.get(method, {})
        if hasattr(self, "norm_visual"):
            self.norm_visual.set_method(method)
        display = str(details.get("display_name", method.replace("_", " ")))
        anchors = tuple(details.get("anchor_landmarks", ()) or ())
        fallback = tuple(details.get("fallback_landmarks", ()) or ())
        anchor_text = self._format_landmark_tuple(anchors)
        if fallback:
            anchor_text = f"{anchor_text}; fallback {self._format_landmark_tuple(fallback)}"
        self._set_normalization_dashboard_value("method", display)
        self._set_normalization_dashboard_value("anchors", anchor_text)
        self._set_normalization_dashboard_value("next_step", "Run QC after normalization")
        if hasattr(self, "norm_desc"):
            self.norm_desc.setText(str(NORMALIZATION_METHODS.get(method, "")))
        if hasattr(self, "norm_science_note"):
            self.norm_science_note.setHtml(
                "<div style='color:#D8E8F8;'>"
                f"<b>What changes:</b> {details.get('what_changes', '')}<br>"
                f"<b>Best use:</b> {details.get('best_for', '')}<br>"
                f"<b>Caution:</b> {details.get('caution', '')}<br>"
                f"<b>Evidence:</b> {details.get('evidence_level', '')}"
                "</div>"
            )
        if hasattr(self, "norm_anchor_table"):
            rows = [
                {"Audit item": "Primary denominator", "Meaning": anchor_text},
                {"Audit item": "Centering", "Meaning": f"Subtract landmark {self.center_landmark_spin.value() if hasattr(self, 'center_landmark_spin') else 1} before scaling."},
                {"Audit item": "What it controls", "Meaning": "Camera distance / face-size differences, approximately."},
                {"Audit item": "What it does not control", "Meaning": "Head rotation, depth motion, poor tracking, occlusion, or true millimeter scale."},
                {"Audit item": "Custom selections", "Meaning": "Scale anchors are read from the full MediaPipe CSV; include them in selected landmarks when you want visual audit/provenance."},
            ]
            self._fill_table(self.norm_anchor_table, pd.DataFrame(rows), max_rows=10)
        if hasattr(self, "norm_method_table"):
            method_rows = []
            for key, item in NORMALIZATION_METHOD_DETAILS.items():
                a = self._format_landmark_tuple(item.get("anchor_landmarks", ()))
                fb = self._format_landmark_tuple(item.get("fallback_landmarks", ()))
                if fb != "None":
                    a = f"{a}; fallback {fb}"
                method_rows.append({
                    "Method": key,
                    "Anchors": a,
                    "Best use": item.get("best_for", ""),
                    "Caution": item.get("caution", ""),
                    "Evidence": item.get("evidence_level", ""),
                })
            self._fill_table(self.norm_method_table, pd.DataFrame(method_rows), max_rows=20)

    def write_normalization_config_only(self) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out is None:
            return
        path = write_normalization_config(out, self.norm_combo.currentText())
        self.stage_records["normalization"] = StageRecord(status="configured", manifest_path=str(path))
        self._refresh_stage_cards(); self._log(f"Normalization config written: {path}")

    def run_normalization_stage(self) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out is None:
            return
        try:
            selected = tuple(self.landmark_indices) if getattr(self, "landmark_indices", None) else None
            self.progress.setValue(10)
            self._log("Starting computational normalization...")
            result = run_normalization_from_selection(
                out,
                self.norm_combo.currentText(),
                selected_landmarks=selected,
                preset=getattr(self, "preset_combo", None).currentText() if hasattr(self, "preset_combo") else "ALS oral-motor core 15",
                center_landmark=int(self.center_landmark_spin.value()),
                overwrite=bool(self.norm_overwrite_check.isChecked()),
            )
            self.progress.setValue(85)
            self._load_normalization_results(result.get("manifest_csv"))
            manifest = result.get("manifest_csv")
            if int(result.get("n_error", 0) or 0) > 0 or int(result.get("n_qc_flagged", 0) or 0) > 0:
                norm_stage_status = "completed_with_warnings"
            else:
                norm_stage_status = "completed"
            self.stage_records["normalization"] = StageRecord(status=norm_stage_status, manifest_path=str(manifest))
            self._refresh_stage_cards()
            self.progress.setValue(100)
            self._log(
                f"Normalized {result.get('n_videos', 0)} videos "
                f"(ok={result.get('n_ok', 0)}, qc_flagged={result.get('n_qc_flagged', 0)}, errors={result.get('n_error', 0)}): {manifest}"
            )
        except Exception as exc:  # noqa: BLE001
            self.progress.setValue(0)
            self._log(f"Normalization failed: {exc}")
            QMessageBox.critical(self, "Normalization failed", str(exc))

    def _load_normalization_results(self, manifest_csv=None) -> None:
        path = Path(str(manifest_csv)) if manifest_csv else None
        if path is None:
            out_text = self.output_edit.text().strip() if hasattr(self, "output_edit") else ""
            if out_text:
                path = Path(out_text).expanduser().resolve() / "kinematics" / "004_normalization" / "tables" / "normalized_landmarks_manifest.csv"
        if path is None or not path.exists():
            if hasattr(self, "norm_results_label"):
                self.norm_results_label.setText("No normalized landmark manifest loaded yet. Choose a method, write config if needed, then run normalization.")
            self._set_normalization_dashboard_value("videos", "0")
            self._set_normalization_dashboard_value("scale_valid", "-")
            self._set_normalization_dashboard_value("qc", "Not run")
            return
        df = pd.read_csv(path)
        rows = []
        for _, rec in df.iterrows():
            face_pct = pd.to_numeric(pd.Series([rec.get("face_detected_fraction")]), errors="coerce").iloc[0]
            scale_pct = pd.to_numeric(pd.Series([rec.get("scale_valid_fraction")]), errors="coerce").iloc[0]
            sel_pct = pd.to_numeric(pd.Series([rec.get("selected_complete_frame_fraction")]), errors="coerce").iloc[0]
            scale_cv = pd.to_numeric(pd.Series([rec.get("scale_value_cv")]), errors="coerce").iloc[0]
            max_jump = pd.to_numeric(pd.Series([rec.get("scale_frame_to_frame_max_jump_fraction")]), errors="coerce").iloc[0]
            rows.append({
                "Video": rec.get("video_id", ""),
                "Status": rec.get("status", ""),
                "Frames": rec.get("n_frames", ""),
                "Face %": "" if pd.isna(face_pct) else f"{face_pct * 100:.1f}",
                "Scale source": rec.get("scale_source", ""),
                "Scale valid %": "" if pd.isna(scale_pct) else f"{scale_pct * 100:.1f}",
                "Scale CV": "" if pd.isna(scale_cv) else f"{scale_cv:.3f}",
                "Max jump %": "" if pd.isna(max_jump) else f"{max_jump * 100:.1f}",
                "Selected complete %": "" if pd.isna(sel_pct) else f"{sel_pct * 100:.1f}",
                "QC flags": rec.get("qc_flags", ""),
            })
        self._fill_table(self.norm_results_table, pd.DataFrame(rows), max_rows=200)
        n_videos = len(df)
        n_ok = int((df.get("status", pd.Series(dtype=str)).astype(str).str.lower() == "ok").sum()) if not df.empty else 0
        n_qc = int((df.get("status", pd.Series(dtype=str)).astype(str).str.lower() == "qc_flagged").sum()) if not df.empty else 0
        n_error = int((df.get("status", pd.Series(dtype=str)).astype(str).str.lower() == "error").sum()) if not df.empty else 0
        mean_scale = pd.to_numeric(df.get("scale_valid_fraction"), errors="coerce").mean() if not df.empty else float("nan")
        if n_error > 0:
            qc_status = "Review / errors"
        elif n_qc > 0:
            qc_status = "Review"
        elif n_ok > 0:
            qc_status = "Complete"
        else:
            qc_status = "Not run"
        self._set_normalization_dashboard_value("videos", n_videos)
        self._set_normalization_dashboard_value("scale_valid", "-" if pd.isna(mean_scale) else f"{mean_scale * 100:.1f}%")
        self._set_normalization_dashboard_value("qc", qc_status)
        self._set_normalization_dashboard_value("next_step", "Video QC" if qc_status == "Complete" else "Review diagnostics")
        if hasattr(self, "norm_results_label"):
            self.norm_results_label.setText(
                f"Loaded normalized manifest: {n_videos} video(s), ok={n_ok}, qc_flagged={n_qc}, errors={n_error}. "
                "Scale CV and frame-to-frame jump should be reviewed before trusting velocity or amplitude features."
            )

    def _write_placeholder(self, stage_key: str, rel: str, payload: dict, message: str) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out is None:
            return
        path = out / "kinematics" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.stage_records[stage_key] = StageRecord(status="completed", manifest_path=str(path))
        self._refresh_stage_cards(); self._log(f"{message}: {path}")

    def run_video_qc_stage(self) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out is None:
            return
        manifest = self._landmarks_manifest_path()
        if manifest is None:
            QMessageBox.warning(self, "Missing landmarks", "Run Landmarks -> Run MediaPipe Landmark Extraction before video QC.")
            return
        self.stage_records["qc"] = StageRecord(status="running")
        self._refresh_stage_cards()
        self._start_worker("Landmark / video QC", run_video_qc, {"output_root": out, "landmarks_manifest_csv": manifest}, self._finish_video_qc_stage)

    def _finish_video_qc_stage(self, result: object) -> None:
        res = dict(result)
        summary_csv = Path(res["summary_csv"])
        self.stage_records["qc"] = StageRecord(status="completed", manifest_path=str(summary_csv))
        self._refresh_stage_cards()
        self._log(f"Video QC completed: {res.get('n_videos', 0)} video(s); status counts={res.get('status_counts', {})}. Summary: {summary_csv}")
        self._load_video_qc_summary(summary_csv)

    def _load_video_qc_summary(self, path: Path | None = None) -> None:
        out_text = self.output_edit.text().strip() if hasattr(self, "output_edit") else ""
        if path is None and out_text:
            path = Path(out_text).expanduser().resolve() / "kinematics" / "005_video_qc" / "tables" / "landmark_video_qc_summary.csv"
        if path is None or not Path(path).exists():
            if hasattr(self, "video_qc_label"):
                self.video_qc_label.setText("No video QC summary found yet.")
            return
        df = pd.read_csv(path)
        if hasattr(self, "video_qc_label"):
            counts = df.get("qc_status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not df.empty else {}
            self.video_qc_label.setText(f"Loaded QC summary: {path}. Status counts: {counts}")
        cols = [c for c in ["video_id", "qc_status", "face_detected_fraction", "n_frames", "n_faces_detected", "max_no_face_gap_frames", "p95_frame_displacement", "qc_rationale"] if c in df.columns]
        if hasattr(self, "video_qc_table"):
            self._fill_table(self.video_qc_table, df[cols] if cols else df, max_rows=200)

    def write_feature_computation_plan(self) -> None:
        """Write an auditable feature-computation plan before running the stage."""
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out is None:
            return
        selected_features = sorted(getattr(self, "selected_feature_ids", set(DEFAULT_KINEMATIC_FEATURE_IDS)))
        selected_specs = [spec for spec in KINEMATIC_FEATURE_SPECS if spec.feature_id in set(selected_features)]
        tables_dir = out / "kinematics" / "006_features" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        plan_json = tables_dir / "feature_computation_plan.json"
        plan_csv = tables_dir / "feature_computation_plan.csv"
        rows = []
        for spec in selected_specs:
            rows.append({
                "feature_id": spec.feature_id,
                "group": spec.group,
                "label": spec.label,
                "status": spec.status,
                "tier": spec.tier,
                "native_signal": spec.native_signal,
                "unit": spec.unit,
                "landmarks": ",".join(map(str, spec.landmarks)),
                "normalization": spec.normalization,
                "aggregation": spec.aggregation,
                "interpretation": spec.interpretation,
                "source_function": spec.source_function,
            })
        pd.DataFrame(rows).to_csv(plan_csv, index=False)
        payload = {
            "stage": "006_features",
            "version": APP_VERSION,
            "selected_feature_ids": selected_features,
            "n_selected_features": len(selected_features),
            "selected_landmarks": list(getattr(self, "landmark_indices", [])),
            "normalization_dependency": "normalized landmarks from kinematics/004_normalization/tables/normalized_landmarks_manifest.csv",
            "qc_dependency": "landmark/video QC from kinematics/005_video_qc/tables/landmark_video_qc_summary.csv",
            "settings": {
                "smoothing_cutoff_hz": float(self.feature_smoothing_cutoff.value()) if hasattr(self, "feature_smoothing_cutoff") else None,
                "extreme_outlier_sigma": float(self.feature_sigma_extreme.value()) if hasattr(self, "feature_sigma_extreme") else None,
                "tight_outlier_sigma": float(self.feature_sigma_tight.value()) if hasattr(self, "feature_sigma_tight") else None,
                "movement_onset_fraction": float(self.feature_onset_frac.value()) if hasattr(self, "feature_onset_frac") else None,
                "movement_offset_fraction": float(self.feature_offset_frac.value()) if hasattr(self, "feature_offset_frac") else None,
                "use_smoothed_signals": bool(self.feature_use_smoothing.isChecked()) if hasattr(self, "feature_use_smoothing") else None,
            },
            "feature_table_csv": str(plan_csv),
            "qc_requirements": list(QC_FEATURE_REQUIREMENTS),
            "note": "Plan only. It does not compute features or modify landmark/normalization outputs.",
        }
        plan_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._log(f"Feature computation plan written: {plan_json}")
        QMessageBox.information(self, "Feature plan written", f"Feature computation plan written:\n{plan_json}")

    def run_feature_computation_stage(self) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out is None:
            return
        norm_manifest = out / "kinematics" / "004_normalization" / "tables" / "normalized_landmarks_manifest.csv"
        if not norm_manifest.exists():
            QMessageBox.warning(self, "Missing normalization", "Run Normalization -> Run Computational Normalization before feature computation.")
            return
        cfg = FeatureComputationConfig(
            selected_landmarks=tuple(self.landmark_indices),
            selected_preset=self.selection_preset_combo.currentText() if hasattr(self, "selection_preset_combo") else "custom",
            smoothing_cutoff_hz=float(self.feature_smoothing_cutoff.value()),
            outlier_sigma_extreme=float(self.feature_sigma_extreme.value()),
            outlier_sigma_tight=float(self.feature_sigma_tight.value()),
            onset_frac=float(self.feature_onset_frac.value()),
            offset_frac=float(self.feature_offset_frac.value()),
            use_smoothed_signals=bool(self.feature_use_smoothing.isChecked()),
            overwrite=True,
        )
        selected_features = sorted(getattr(self, "selected_feature_ids", set(DEFAULT_KINEMATIC_FEATURE_IDS)))
        self._log(f"Feature computation selected {len(selected_features)} feature definition(s): {', '.join(selected_features) if selected_features else 'none'}")
        self.stage_records["features"] = StageRecord(status="running")
        self._refresh_stage_cards()
        self._start_worker("Kinematic feature computation", run_feature_computation, {"output_root": out, "cfg": cfg}, self._finish_feature_computation_stage)

    def _finish_feature_computation_stage(self, result: object) -> None:
        res = dict(result)
        features_csv = Path(res["features_csv"])
        self.stage_records["features"] = StageRecord(status="completed", manifest_path=str(features_csv))
        self._refresh_stage_cards()
        self._log(f"Feature computation completed: {res.get('n_videos', 0)} video(s); ok={res.get('n_ok', 0)}, qc_flagged={res.get('n_qc_flagged', 0)}, error={res.get('n_error', 0)}. Features: {features_csv}")
        self._load_feature_results(features_csv)

    def _load_feature_results(self, path: Path | None = None) -> None:
        out_text = self.output_edit.text().strip() if hasattr(self, "output_edit") else ""
        if path is None and out_text:
            path = Path(out_text).expanduser().resolve() / "kinematics" / "006_features" / "tables" / "kinematic_features.csv"
        if path is None or not Path(path).exists():
            if hasattr(self, "feature_results_label"):
                self.feature_results_label.setText("No kinematic feature table loaded yet.")
            return
        df = pd.read_csv(path)
        if hasattr(self, "feature_results_label"):
            counts = df.get("status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not df.empty else {}
            self.feature_results_label.setText(f"Loaded kinematic features: {path}. Status counts: {counts}")
        preferred = [
            "video_id", "status", "n_frames", "face_detected_fraction", "n_movements",
            "mouth_aperture_median", "mouth_aperture_range_p05_p95",
            "outer_lip_spread_median", "lip_aspect_ratio_median",
            "jaw_to_nose_median", "corner_vertical_asymmetry_median", "feature_qc_flags",
        ]
        cols = [c for c in preferred if c in df.columns]
        if hasattr(self, "feature_results_table"):
            self._fill_table(self.feature_results_table, df[cols] if cols else df, max_rows=200)

    def run_features_placeholder(self) -> None:
        self.run_feature_computation_stage()

    def run_temporal_aggregation_stage(self) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out is None:
            return
        features_csv = out / "kinematics" / "006_features" / "tables" / "kinematic_features.csv"
        if not features_csv.exists():
            QMessageBox.warning(self, "Missing features", "Run Features -> Run Kinematic Feature Computation before temporal aggregation.")
            return
        cfg = TemporalAggregationConfig(
            profile=self.agg_combo.currentText() if hasattr(self, "agg_combo") else "robust_default",
            include_raw_signals=bool(self.agg_include_raw.isChecked()) if hasattr(self, "agg_include_raw") else False,
            include_velocity_signals=bool(self.agg_include_velocity.isChecked()) if hasattr(self, "agg_include_velocity") else True,
            min_valid_fraction=float(self.agg_min_valid.value()) if hasattr(self, "agg_min_valid") else 0.50,
            min_detected_fraction=float(self.agg_min_detected.value()) if hasattr(self, "agg_min_detected") else 0.60,
            overwrite=True,
        )
        self.stage_records["aggregation"] = StageRecord(status="running")
        self._refresh_stage_cards()
        self._start_worker("Temporal aggregation", run_temporal_aggregation, {"output_root": out, "cfg": cfg}, self._finish_temporal_aggregation_stage)

    def _finish_temporal_aggregation_stage(self, result: object) -> None:
        res = dict(result)
        agg_csv = Path(res["aggregated_features_csv"])
        self.stage_records["aggregation"] = StageRecord(status="completed", manifest_path=str(res.get("manifest_json", agg_csv)))
        self._refresh_stage_cards()
        self._log(f"Temporal aggregation completed: {res.get('n_videos', 0)} video(s); ok={res.get('n_ok', 0)}, qc_flagged={res.get('n_qc_flagged', 0)}, error={res.get('n_error', 0)}. Aggregated table: {agg_csv}")
        self._load_aggregation_results(agg_csv)

    def _load_aggregation_results(self, path: Path | None = None) -> None:
        out_text = self.output_edit.text().strip() if hasattr(self, "output_edit") else ""
        if path is None and out_text:
            path = Path(out_text).expanduser().resolve() / "kinematics" / "007_aggregation" / "tables" / "kinematic_aggregated_features.csv"
        if path is None or not Path(path).exists():
            if hasattr(self, "aggregation_results_label"):
                self.aggregation_results_label.setText("No temporal aggregation table loaded yet.")
            return
        df = pd.read_csv(path)
        if hasattr(self, "aggregation_results_label"):
            counts = df.get("status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not df.empty else {}
            self.aggregation_results_label.setText(f"Loaded temporal aggregation: {path}. Status counts: {counts}")
        preferred = [
            "video_id", "status", "aggregation_profile", "n_frames", "duration_s",
            "face_detected_fraction", "n_signals_aggregated", "n_movements",
            "mouth_aperture_median", "mouth_aperture_iqr", "mouth_aperture_range_p05_p95",
            "outer_lip_spread_median", "lip_aspect_ratio_median",
            "jaw_to_nose_median", "aggregation_qc_flags",
        ]
        cols = [c for c in preferred if c in df.columns]
        if hasattr(self, "aggregation_results_table"):
            self._fill_table(self.aggregation_results_table, df[cols] if cols else df, max_rows=200)

    def run_aggregation_placeholder(self) -> None:
        self.run_temporal_aggregation_stage()

    def refresh_inspector(self) -> None:
        out_text = self.output_edit.text().strip()
        if not out_text:
            return
        root = Path(out_text) / "kinematics"
        candidates = sorted(root.glob("**/*.csv"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        if not candidates:
            self.inspector_label.setText("No CSV outputs found yet."); self.inspector_table.setRowCount(0); self.inspector_table.setColumnCount(0); return
        path = candidates[0]
        df = pd.read_csv(path)
        self.inspector_label.setText(f"Previewing latest CSV: {path}")
        self._fill_table(self.inspector_table, df, max_rows=300)

    def run_report_stage(self) -> None:
        out = self._path_or_warn(self.output_edit, "an output project folder")
        if out is None:
            return
        report = write_scaffold_report(out)
        self.last_report_html = report
        self.stage_records["reports"] = StageRecord(status="completed", report_path=str(report))
        self._refresh_stage_cards()
        self.report_box.setHtml(report.read_text(encoding="utf-8"))
        self._log(f"Kinematics report written: {report}")

    def refresh_latest_outputs(self) -> None:
        self._load_ingest_summary()
        landmark_manifest = self._landmarks_manifest_path()
        if landmark_manifest is not None:
            self._load_landmark_summary(landmark_manifest)
        elif hasattr(self, "landmark_metric_labels"):
            self._update_landmark_dashboard_from_manifest(None)
        self.refresh_inspector()
        self._log("Refreshed latest kinematics outputs.")

    def open_output_root(self) -> None:
        path_text = self.output_edit.text().strip()
        if not path_text:
            QMessageBox.information(self, "No output folder", "Select an output folder first."); return
        path = Path(path_text).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        import webbrowser
        webbrowser.open(path.as_uri())

    def open_report(self) -> None:
        if self.last_report_html and self.last_report_html.exists():
            import webbrowser
            webbrowser.open(self.last_report_html.as_uri())
        else:
            QMessageBox.information(self, "No report", "Create the report first.")

    def run_all(self) -> None:
        self.run_project_init()
        self.run_ingest_stage()
        self.run_metadata_stage()
        self.run_landmark_plan_stage()
        self.run_selection_stage()
        self.run_normalization_stage()
        self.run_video_qc_stage()
        self.run_features_placeholder()
        self.run_aggregation_placeholder()
        self.run_report_stage()


def launch_kinematics_gui() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("VSLP")
    try:
        from vslp.gui.theme import build_dark_stylesheet
        app.setStyleSheet(build_dark_stylesheet())
    except Exception:
        # Fallback mirrors the acoustic dark theme closely enough if the shared theme is unavailable.
        app.setStyleSheet(
            """
            QWidget { background: #071727; color: #D8E8F8; font-family: 'Segoe UI', Arial, sans-serif; font-size: 10pt; }
            QFrame#Sidebar { background: #0E253A; border: 1px solid #1B3A57; border-radius: 10px; }
            QLabel#AppTitleLabel { font-size: 24pt; font-weight: 900; color: #FFFFFF; }
            QLabel#SubtitleLabel, QLabel#IPNoticeLabel { color: #AFC8DE; font-size: 8.5pt; }
            QFrame#Card, QGroupBox { background: #0C2033; border: 1px solid #1B3A57; border-radius: 8px; }
            QFrame#BrandingBar { background: #F6F8FB; border-radius: 10px; }
            QLabel#BrandingTitle { color: #06213A; font-weight: 800; }
            QPushButton { background: #1D6791; color: white; border: 1px solid #2E7DAA; border-radius: 6px; padding: 7px 10px; font-weight: 700; }
            QPushButton#RunButton { background: #16896F; }
            QPushButton#OpenButton { background: #415A75; }
            QLineEdit, QPlainTextEdit, QTextBrowser, QComboBox, QSpinBox, QDoubleSpinBox { background: #0B1D2E; color: #E7F3FF; border: 1px solid #244B6C; border-radius: 5px; padding: 5px; }
            QTableWidget { background: #0B1D2E; color: #E7F3FF; gridline-color: #244B6C; selection-background-color: #254D70; selection-color: white; }
            QHeaderView::section { background: #183B59; color: #FFFFFF; padding: 6px; border: 1px solid #244B6C; }
            QTabWidget::pane { border: 1px solid #244B6C; }
            QTabBar::tab { background: #102A42; color: #D8E8F8; padding: 8px 13px; border-top-left-radius: 5px; border-top-right-radius: 5px; }
            QTabBar::tab:selected { background: #1D6791; color: #FFFFFF; }
            """
        )
    win = KinematicsPipelineWindow()
    win.show()
    return app.exec()


def main() -> int:
    return launch_kinematics_gui()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
