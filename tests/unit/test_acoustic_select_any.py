"""Selection permission is independent of task recommendation and execution."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from vslp.acoustic.features.catalog import load_feature_catalog, supports_task
from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow


def test_bamboo_selection_covers_all_registered_leaves_and_preserves_task_switch():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = AcousticPipelineWindow()
    window.task_name_edit.setText("Bamboo Passage")
    selectable = {feature_id for feature_id in window.feature_items
                  if window._feature_eligible(feature_id)}
    registered = {output["feature_id"] for output in window.feature_catalog["outputs"]
                  if output.get("selectable") and output.get("family_spec_approved")}
    assert selectable == registered
    states = {"✓ RECOMMENDED", "— NOT RECOMMENDED"}
    assert states <= {window.feature_items[feature_id].text(1) for feature_id in selectable}
    unsupported = window.feature_items["ddk_pa_rate_syll_s"]
    assert unsupported.flags() & Qt.ItemIsUserCheckable
    assert unsupported.text(2) == "UNSUPPORTED FOR THIS TASK"
    alignment = window.feature_items["delta_v_s"]
    assert alignment.text(2) == "Alignment required"
    for family_id in ("F01", "F04", "F07", "F09", "F12", "F13"):
        family = window.subsystem_items[family_id]
        family.setCheckState(0, Qt.Checked)
        assert all(child.checkState(0) == Qt.Checked for child in
                   window._feature_descendants(family)
                   if window._feature_eligible(child.data(0, Qt.UserRole)[1]))
        family.setCheckState(0, Qt.Unchecked)
    construct = unsupported.parent()
    construct.setCheckState(0, Qt.Checked)
    assert unsupported.checkState(0) == Qt.Checked
    window.task_name_edit.setText("S2")
    assert {feature_id for feature_id in window.feature_items
            if window._feature_eligible(feature_id)} == registered
    assert unsupported.checkState(0) == Qt.Checked
    assert unsupported.text(2) == "UNSUPPORTED FOR THIS TASK"
    window.select_all_features()
    assert set(window._selected_feature_names()) == selectable
    disabled = window.feature_items["pause_pattern_factor_if_frozen"]
    assert not disabled.flags() & Qt.ItemIsUserCheckable
    assert disabled.checkState(0) == Qt.Unchecked
    window.close()
    assert app is not None


def test_f04_execution_routes_are_independent_of_recommendation():
    outputs = {item["feature_id"]: item for item in load_feature_catalog()["outputs"]}
    f2 = {**outputs["f2_median_hz"], "recommended_tasks": []}
    assert supports_task(f2, "Bamboo Passage")
    assert not supports_task(f2, "DDK")
    assert supports_task(outputs["vowel_distance_ai_hz"], "S3")
    assert not supports_task(outputs["vowel_distance_ai_hz"], "Bamboo Passage")
