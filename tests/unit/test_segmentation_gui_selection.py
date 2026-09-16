import os

from PySide6.QtWidgets import QApplication

from vslp.acoustic.segment.selection import DDK, PHONATION, SILERO
from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow


def test_gui_method_selection_and_task_recommendation():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = AcousticPipelineWindow()
    assert window.segmentation_method_combo.count() == 4
    assert window.segmentation_method_combo.currentData() == SILERO
    assert window.speech_pad_spin.value() == 0
    window.task_name_edit.setText("/pataka/")
    assert window.segmentation_method_combo.currentData() == DDK
    assert window.ddk_group.isVisible() or not window.isVisible()
    assert window._segmentation_config_from_gui().method == DDK
    window.task_name_edit.setText("sustained /a/")
    assert window.segmentation_method_combo.currentData() == PHONATION
    assert window._segmentation_config_from_gui().phonation.stable_duration_ms == 2000
    window.task_name_edit.setText("Bamboo Passage")
    assert window.segmentation_method_combo.currentData() == SILERO
    window.close()
    assert app is not None
