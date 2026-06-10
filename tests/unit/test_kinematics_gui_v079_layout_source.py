from pathlib import Path


APP_SOURCE = Path("src/vslp/gui/kinematics/app.py")


def test_v079_selection_uses_spacious_splitter_and_tabbed_outputs():
    source = APP_SOURCE.read_text(encoding="utf-8")
    assert "APP_VERSION = \"v0.86\"" in source
    assert "QSplitter(Qt.Horizontal)" in source
    assert "selection_tabs.addTab(selected_tab, \"Selected landmarks\")" in source
    assert "selection_tabs.addTab(requirements_tab, \"Requirement checks\")" in source
    assert "selection_tabs.addTab(preset_tab, \"Preset reference\")" in source


def test_v079_canvas_has_zoom_context_and_clipping():
    source = APP_SOURCE.read_text(encoding="utf-8")
    assert "def _draw_zoom_minimap" in source
    assert "painter.setClipRect(QRectF(left, top, w, h))" in source
    assert "yellow inset shows current zoom window" in source


def test_v079_normalization_keeps_details_secondary():
    source = APP_SOURCE.read_text(encoding="utf-8")
    assert "details_tabs.addTab(anchor_tab, \"Current method audit\")" in source
    assert "details_tabs.addTab(method_tab, \"Method comparison\")" in source
    assert "details_tabs.addTab(note_tab, \"Why normalize\")" in source
    assert "NormalizationMethodVisual()" not in source
