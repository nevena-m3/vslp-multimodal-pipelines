"""The student GUIs keep controls reachable at laptop window sizes."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QScrollArea

from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow
from vslp.gui.alignment_app.workbench_window import WorkbenchWindow


SIZES = ((1280, 720), (1366, 768), (1536, 864),
         (1920, 1080), (2560, 1440), (900, 600))


def _visible_after_scroll(scroll: QScrollArea, widget) -> bool:
    scroll.ensureWidgetVisible(widget)
    QApplication.processEvents()
    top_left = widget.mapTo(scroll.viewport(), QPoint(0, 0))
    return scroll.viewport().rect().contains(top_left)


def test_alignment_window_resizes_and_review_stays_reachable(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(WorkbenchWindow, "_start", lambda *_args, **_kwargs: None)
    window = WorkbenchWindow()
    try:
        window.show()
        setup_scroll = window.findChild(QScrollArea, "AlignmentSetupScroll")
        assert setup_scroll is not None
        for width, height in SIZES:
            window.resize(width, height)
            app.processEvents()
            assert window.size().width() == width
            assert window.size().height() == height
            window.tabs.setCurrentIndex(0)
            assert window.create_button.isVisible()
            window.tabs.setCurrentIndex(2)
            app.processEvents()
            assert window.reviewer.wave_plot.height() >= 135
            assert window.reviewer.word_plot.height() >= 80
            assert window.reviewer.phone_plot.height() >= 80
        assert window.minimumWidth() <= 1280
        assert window.minimumHeight() <= 720
    finally:
        window.close()


def test_acoustic_window_resizes_and_feature_actions_stay_reachable():
    app = QApplication.instance() or QApplication([])
    window = AcousticPipelineWindow()
    try:
        window.show()
        sidebar_scroll = window.findChild(QScrollArea, "AcousticSidebarScroll")
        assert sidebar_scroll is not None
        feature_page = next(window.tabs.widget(i) for i in range(window.tabs.count())
                            if window.tabs.tabText(i) == "Acoustic Features")
        feature_scroll = feature_page.findChild(QScrollArea)
        assert feature_scroll is not None
        for width, height in SIZES:
            window.resize(width, height)
            app.processEvents()
            assert window.size().width() == width
            assert window.size().height() == height
            assert _visible_after_scroll(sidebar_scroll, window.open_output_btn)
            window.tabs.setCurrentWidget(feature_page)
            app.processEvents()
            assert _visible_after_scroll(feature_scroll, window.run_features_btn)
            assert window.feature_tree.height() >= 360
        assert window.minimumWidth() <= 1280
        assert window.minimumHeight() <= 720
    finally:
        window.close()
