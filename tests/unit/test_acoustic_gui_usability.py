"""Task-aware Acoustic Features and coordinated review-timeline UI checks."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pyqtgraph as pg
import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow, FEATURE_FAMILY_LABELS
from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget
from vslp.gui.acoustic_app.waveform_editor import WaveformEditor
from vslp.core.process_windows import hidden_console_kwargs


def _app() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def test_exact_family_order_and_task_aware_tristate(tmp_path):
    _app()
    window = AcousticPipelineWindow()
    expected = [f"[F{number}] {name}" for number, name in enumerate(FEATURE_FAMILY_LABELS, 1)]
    assert [window.feature_tree.topLevelItem(i).text(0) for i in range(13)] == expected
    assert not window.feature_tree.isSortingEnabled()
    window._feature_prerequisite_issues = lambda _feature_id: []
    window.task_name_edit.setText("Bamboo Passage")
    family = window.subsystem_items["F13"]
    eligible = [item for item in window._feature_descendants(family)
                if window._feature_eligible(item.data(0, Qt.UserRole)[1])]
    assert len(eligible) >= 2
    family.setCheckState(0, Qt.Checked)
    assert all(item.checkState(0) == Qt.Checked for item in eligible)
    eligible[0].setCheckState(0, Qt.Unchecked)
    assert family.checkState(0) == Qt.PartiallyChecked
    family.setCheckState(0, Qt.Checked)
    assert all(item.checkState(0) == Qt.Checked for item in eligible)
    window.clear_feature_selection()
    assert not window._selected_feature_names()
    window.select_all_features()
    assert set(window._selected_feature_names()) == {
        feature_id for feature_id in window.feature_items if window._feature_eligible(feature_id)}
    window.task_name_edit.setText("S1")
    assert [window.feature_tree.topLevelItem(i).text(0) for i in range(13)] == expected
    assert window._feature_task_name() == "WSTG"
    assert window._feature_eligible("s1_tsfel_shannon_entropy_norm")
    assert all(window._feature_eligible(feature_id) for feature_id in window._selected_feature_names())
    window.feature_view_combo.setCurrentIndex(window.feature_view_combo.findData("runnable"))
    assert all(not window.feature_tree.topLevelItem(i).isHidden() for i in range(13))
    window.close()


def test_missing_alignment_is_selectable_but_unavailable():
    _app()
    window = AcousticPipelineWindow()
    window.task_name_edit.setText("S2")
    target = window.feature_items["f1_transition_slope_buy_ay_hz_s"]
    assert "Requires Alignment Package" in target.text(0)
    assert target.text(2) == "ALIGNMENT PACKAGE REQUIRED"
    assert target.flags() & Qt.ItemIsUserCheckable
    assert window._feature_eligible("f1_transition_slope_buy_ay_hz_s")
    assert not window._feature_ready("f1_transition_slope_buy_ay_hz_s")
    window.close()


def test_bamboo_rates_show_frozen_count_without_picker_or_alignment():
    _app()
    window = AcousticPipelineWindow()
    window.task_name_edit.setText("Bamboo Passage")
    window.clear_feature_selection()
    for feature_id in ("bamboo_speaking_rate_wpm", "bamboo_articulation_rate_wpm"):
        window.feature_items[feature_id].setCheckState(0, Qt.Checked)
    assert window.feature_bamboo_count_row.isHidden() is False
    assert window.feature_bamboo_count_value.text() == "97 words"
    assert window.feature_bamboo_count_value.isReadOnly()
    assert window.feature_prompt_row.isHidden()
    assert window.feature_alignment_row.isHidden()
    for feature_id in ("bamboo_speaking_rate_wpm", "bamboo_articulation_rate_wpm"):
        assert "alignment" not in " ".join(window._feature_prerequisite_issues(feature_id)).lower()
    window.close()


def test_bamboo_f13_uses_built_in_prompt_without_count_picker():
    _app()
    window = AcousticPipelineWindow()
    window.task_name_edit.setText("Bamboo Passage")
    window.clear_feature_selection()
    window.feature_items["bamboo_wpd_shannon_entropy_node0_norm"].setCheckState(0, Qt.Checked)
    assert window.feature_prompt_row.isHidden()
    window.close()


@pytest.mark.parametrize(("family_id", "task"), [
    ("F01", "Sustained /a/"), ("F02", "Sustained /a/"),
    ("F03", "Sustained /a/"), ("F04", "Bamboo Passage"),
    ("F05", "S2"), ("F06", "S3"), ("F07", "Bamboo Passage"),
    ("F08", "Bamboo Passage"), ("F09", "Bamboo Passage"),
    ("F10", "/pa/"), ("F11", "S5"), ("F12", "S2"),
    ("F13", "Bamboo Passage"),
])
def test_family_selection_recurses_over_task_applicable_leaves(family_id, task):
    _app()
    window = AcousticPipelineWindow()
    window.task_name_edit.setText(task)
    family = window.subsystem_items[family_id]
    leaves = [leaf for leaf in window._feature_descendants(family)
              if window._feature_eligible(leaf.data(0, Qt.UserRole)[1])]
    assert leaves
    family.setCheckState(0, Qt.Checked)
    assert all(leaf.checkState(0) == Qt.Checked for leaf in leaves)
    leaves[0].setCheckState(0, Qt.Unchecked)
    assert family.checkState(0) == (
        Qt.PartiallyChecked if len(leaves) > 1 else Qt.Unchecked)
    family.setCheckState(0, Qt.Unchecked)
    assert all(leaf.checkState(0) == Qt.Unchecked for leaf in leaves)
    window.close()


def test_bamboo_f7_construct_tristate_includes_alignment_blocked_leaves():
    _app()
    window = AcousticPipelineWindow()
    window.task_name_edit.setText("Bamboo Passage")
    family = window.subsystem_items["F07"]
    variability = window.construct_items["C047"]
    expected = {"delta_v_s", "varco_v_pct", "npvi_v_pct"}
    leaves = [window.feature_items[feature_id] for feature_id in expected]
    assert variability.flags() & Qt.ItemIsUserCheckable
    assert all(leaf.text(2) == "ALIGNMENT PACKAGE REQUIRED" for leaf in leaves)
    family.setCheckState(0, Qt.Checked)
    applicable = {leaf.data(0, Qt.UserRole)[1] for leaf in
                  window._feature_descendants(family) if window._feature_eligible(
                      leaf.data(0, Qt.UserRole)[1])}
    assert len(applicable) == 11
    assert {"bamboo_speech_duration_s", "bamboo_total_duration_s", *expected} <= applicable
    assert applicable <= set(window._selected_feature_names())
    assert "11 selectable" in family.text(2)
    window.clear_feature_selection()
    variability.setCheckState(0, Qt.Checked)
    assert all(leaf.checkState(0) == Qt.Checked for leaf in leaves)
    window.feature_items["varco_v_pct"].setCheckState(0, Qt.Unchecked)
    assert variability.checkState(0) == Qt.PartiallyChecked
    assert family.checkState(0) == Qt.PartiallyChecked
    variability.setCheckState(0, Qt.Checked)
    assert all(leaf.checkState(0) == Qt.Checked for leaf in leaves)
    assert not (window.feature_items["pause_pattern_factor_if_frozen"].flags() &
                Qt.ItemIsUserCheckable)
    window.close()


def test_three_recommendations_and_separate_availability_and_metadata():
    _app()
    window = AcousticPipelineWindow()
    window.task_name_edit.setText("Bamboo Passage")
    tree = window.feature_tree
    assert [tree.headerItem().text(index) for index in range(7)] == [
        "Feature", "Task recommendation", "Availability", "Recommended task",
        "Unit", "Range", "Analysis unit"]
    assert all(not tree.isColumnHidden(index) for index in range(7))
    recommended = window.feature_items["delta_v_s"]
    # Exercise the optional style without changing the frozen registry metadata.
    window.catalog_outputs["f1_median_hz"] = {
        **window.catalog_outputs["f1_median_hz"],
        "recommended_tasks": [],
        "conditional_tasks": ["bamboo_passage"],
    }
    window._refresh_feature_indicators()
    optional = window.feature_items["f1_median_hz"]
    not_recommended = window.feature_items["ddk_pa_rate_syll_s"]
    assert "RECOMMENDED" in recommended.text(1)
    assert recommended.text(2) == "ALIGNMENT PACKAGE REQUIRED"
    assert "SUPPORTED / OPTIONAL" in optional.text(1)
    assert optional.flags() & Qt.ItemIsUserCheckable
    optional.setCheckState(0, Qt.Checked)
    assert optional.checkState(0) == Qt.Checked
    assert "NOT RECOMMENDED" in not_recommended.text(1)
    assert len({recommended.background(1).color().name(),
                optional.background(1).color().name(),
                not_recommended.background(1).color().name()}) == 3
    source = window.catalog_outputs["delta_v_s"]
    assert recommended.text(3) == "Bamboo Passage"
    assert recommended.text(4) == source["unit"]
    assert recommended.text(5) == source["qc_range_text"]
    assert recommended.text(6) == source["analysis_unit"]
    window.close()


def test_not_recommended_supported_leaf_remains_selectable_at_every_level():
    _app()
    window = AcousticPipelineWindow()
    window.task_name_edit.setText("Bamboo Passage")
    feature_id = "jitter_local_pct"
    leaf = window.feature_items[feature_id]
    construct = leaf.parent()
    family = window.subsystem_items["F02"]
    assert window._feature_eligible(feature_id)
    assert leaf.text(1) == "— NOT RECOMMENDED"
    assert leaf.flags() & Qt.ItemIsUserCheckable
    assert construct.flags() & Qt.ItemIsUserCheckable
    assert family.flags() & Qt.ItemIsUserCheckable
    leaf.setCheckState(0, Qt.Checked)
    assert leaf.checkState(0) == Qt.Checked
    window.clear_feature_selection()
    construct.setCheckState(0, Qt.Checked)
    assert leaf.checkState(0) == Qt.Checked
    window.clear_feature_selection()
    family.setCheckState(0, Qt.Checked)
    assert leaf.checkState(0) == Qt.Checked
    window.clear_feature_selection()
    window.select_all_features()
    assert leaf.checkState(0) == Qt.Checked
    disabled = window.feature_items["pause_pattern_factor_if_frozen"]
    assert not (disabled.flags() & Qt.ItemIsUserCheckable)
    assert disabled.checkState(0) == Qt.Unchecked
    window.close()


def test_task_switch_preserves_applicable_selection_despite_prerequisite_change():
    _app()
    window = AcousticPipelineWindow()
    window.task_name_edit.setText("S2")
    target = window.feature_items["f1_transition_slope_buy_ay_hz_s"]
    target.setCheckState(0, Qt.Checked)
    assert target.text(2) == "ALIGNMENT PACKAGE REQUIRED"
    window._feature_prerequisite_issues = lambda _feature_id: []
    window._refresh_feature_count_label()
    assert target.checkState(0) == Qt.Checked
    assert target.text(2) == "Implemented"
    window.task_name_edit.setText("S5")
    assert target.checkState(0) == Qt.Checked
    assert target.text(2) == "UNSUPPORTED FOR THIS TASK"
    window.close()


def test_review_timelines_share_x_axis_and_readable_tracks(tmp_path):
    _app()
    editor = WaveformEditor()
    assert editor.support.getViewBox().linkedView(0) is (
        editor.waveform.getViewBox())
    assert editor.waveform.getAxis("left").width() == editor.support.getAxis("left").width()
    widget = AlignmentWidget(lambda: tmp_path)
    for track in (widget.word_plot, widget.phone_plot):
        assert track.getViewBox().linkedView(0) is widget.plot.getViewBox()
        assert track.minimumHeight() >= 72
        assert track.getAxis("left").width() == widget.plot.getAxis("left").width()
    widget.close()
    editor.close()


def test_alignment_labels_reappear_on_zoom(tmp_path):
    app = _app()
    widget = AlignmentWidget(lambda: tmp_path)
    widget.resize(1000, 850)
    widget.show()
    label = pg.TextItem("example", anchor=(.5, .5))
    widget.word_plot.addItem(label)
    widget._track_labels.append((label, .2, .8, "Word"))
    widget.plot.setXRange(0, 100, padding=0)
    app.processEvents()
    widget._update_token_labels()
    assert not label.isVisible()
    widget.plot.setXRange(0, 1, padding=0)
    app.processEvents()
    widget._update_token_labels()
    assert label.isVisible()
    widget.close()


def test_child_console_policy_is_platform_specific(monkeypatch):
    import subprocess
    import vslp.core.process_windows as process_windows

    monkeypatch.setattr(process_windows.sys, "platform", "linux")
    assert hidden_console_kwargs() == {}
    monkeypatch.setattr(process_windows.sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(subprocess, "STARTF_USESHOWWINDOW", 0x00000001, raising=False)
    monkeypatch.setattr(subprocess, "SW_HIDE", 0, raising=False)
    monkeypatch.setattr(subprocess, "STARTUPINFO", lambda: type("Startup", (), {"dwFlags": 0})(),
                        raising=False)
    options = hidden_console_kwargs()
    assert options["creationflags"] == 0x08000000
    assert options["startupinfo"].dwFlags & 0x00000001


def test_alignment_progress_uses_completed_recordings_not_preparation(tmp_path):
    _app()
    widget = AlignmentWidget(lambda: tmp_path)
    widget.alignment_started()
    widget.alignment_progress(4, 4, "Recordings prepared: 4 / 4")
    assert "100%" not in widget.run_percent.text()
    widget.alignment_progress(0, 0, "Running MFA corpus alignment...")
    assert "Stage: Aligning" in widget.run_stage.text()
    widget.alignment_progress(1, 4, "Aligning 1 / 4 — first.wav")
    assert "25%" in widget.run_percent.text()
    assert "Recording 2 of 4" in widget.run_queue.text()
    assert "first.wav" in widget.run_queue_status.text()
    widget.alignment_finished()
    widget.close()


def test_alignment_technical_log_defaults_collapsed():
    _app()
    from vslp.gui.alignment_app.main_window import AlignmentWindow

    window = AlignmentWindow()
    assert not window.technical.isChecked()
    window.review.show_technical_button.click()
    assert window.technical.isChecked()
    window.close()


def test_bamboo_sentence_selector_zooms_shared_timeline(tmp_path, monkeypatch):
    _app()
    import vslp.gui.acoustic_app.alignment_widget as alignment_gui
    import vslp.acoustic.features.family13_v1_stage as family13_stage

    alignment = SimpleNamespace(
        canonical_words=pd.DataFrame({"recording_id": ["r1"]}),
        trials=pd.DataFrame([{"recording_id": "r1", "trial_id": "t1",
                              "task_id": "bamboo_passage"}]),
    )
    monkeypatch.setattr(alignment_gui, "load_final_alignment", lambda _root: (alignment, ""))
    monkeypatch.setattr(family13_stage, "_sentence_spans", lambda *_args: [
        {"sentence_index": 1, "start_sec": 1., "end_sec": 2.},
        {"sentence_index": 2, "start_sec": 3., "end_sec": 4.},
    ])
    widget = AlignmentWidget(lambda: tmp_path)
    widget._load_sentence_navigation("r1")
    assert widget.sentence_selector.count() == 2
    widget.sentence_selector.setCurrentIndex(1)
    left, right = widget.plot.viewRange()[0]
    assert left < 3. and right > 4.
    assert widget.word_plot.getViewBox().viewRange()[0] == widget.plot.viewRange()[0]
    widget.close()
