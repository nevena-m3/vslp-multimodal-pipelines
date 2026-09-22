"""Source-only Acoustic Features registry and GUI contract."""

import json
import os
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from vslp.acoustic.features.catalog import (
    TASKS, attach_family_outputs, load_feature_catalog, prerequisite_issues,
    task_fit, task_indicator, task_key,
    validate_feature_ids,
)
from vslp.acoustic.features.stage import FeatureExtractionConfig, _select_registry
from vslp.acoustic.features.family08 import FAMILY08_IDS
from vslp.acoustic.features.family09 import FAMILY09_IDS
from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow


def _window() -> AcousticPipelineWindow:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QApplication.instance() or QApplication([])
    return AcousticPipelineWindow()


def test_exact_master_constructs_and_no_legacy_leaves():
    catalog = load_feature_catalog()
    source = json.loads((Path(__file__).parents[2] / "src/vslp/acoustic/features/data/master_matrix.json").read_text(encoding="utf-8"))
    assert len(catalog["families"]) == 13
    assert len(catalog["constructs"]) == 79
    assert [c["construct_id"] for c in catalog["constructs"]] == [
        c["construct_id"] for c in source["constructs"]]
    assert {o["feature_id"] for o in catalog["outputs"]} == FAMILY08_IDS | FAMILY09_IDS
    assert all(c["outputs"] == [] for c in catalog["constructs"]
               if c["family_id"] not in {"F08", "F09"})
    assert all(c["evidence_level"] is None for c in catalog["constructs"])
    assert catalog["constructs"][4]["construct_name"] == "Jitter"
    with pytest.raises(ValueError, match="Duplicate"):
        validate_feature_ids([{"feature_id": "x"}, {"feature_id": "x"}])
    with pytest.raises(ValueError, match="exact"):
        validate_feature_ids([{"feature_id": "—"}])


def test_task_registry_is_explicit_and_has_no_semantic_classes():
    assert TASKS == {
        "sustained_a": "Sustained /a/", "bamboo_passage": "Bamboo Passage",
        "ddk": "DDK", "wstg": "WSTG",
    }
    assert task_key("WSTG") == "wstg"
    assert task_key("We See Three Geese") is None
    assert task_key("DDK /ta/") is None
    assert task_key("Bamboo-like passage") is None
    assert all("task_class" not in c for c in load_feature_catalog()["constructs"])
    constructs = {c["construct_id"]: c for c in load_feature_catalog()["constructs"]}
    assert task_fit(constructs["C001"], "WSTG") == "Recommended"
    assert task_fit(constructs["C002"], "WSTG") == "Not specified"
    assert task_fit(constructs["C002"], "Sustained /a/") == "Recommended"
    assert task_fit(constructs["C002"], "sustained vowel") == "Not specified"


def test_task_indicator_has_four_source_bound_states():
    base = {"construct_id": "C048", "feature_id": "source_exact_id",
            "family_spec_approved": True, "selectable": True,
            "matrix_status": "PROD", "recommended_tasks": ["bamboo_passage"],
            "conditional_tasks": ["wstg"]}
    assert task_indicator(base, "Bamboo Passage")[0] == "GREEN"
    assert task_indicator(base, "WSTG")[0] == "AMBER"
    assert task_indicator(base, "Sustained /a/")[0] == "GRAY"
    assert task_indicator(base, "We See Three Geese")[0] == "GRAY"
    assert task_indicator({**base, "matrix_status": "BLOCKED"}, "Bamboo Passage")[0] == "RED"
    assert task_indicator(base, "Bamboo Passage",
                          prerequisite_problems=["prompt manifest missing"])[0] == "RED"
    assert task_indicator(base, "Bamboo Passage", has_exact_output=False)[0] == "RED"


def test_legacy_ids_cannot_execute_and_family_spec_is_required():
    for selection in ([], ["f0_mean"], ["f0_mean_hz"], ["mean_pause_dur"], ["jitter"]):
        with pytest.raises(ValueError, match="legacy feature IDs cannot execute"):
            _select_registry(FeatureExtractionConfig(selected_features=selection))
    issues = prerequisite_issues({"feature_id": "f0_mean_hz"}, task_name="WSTG",
        final_decisions=None, final_intervals=None)
    assert "family specification" in issues[0]


def test_family_leaves_attach_with_exact_ids_and_stable_construct_identity():
    catalog = load_feature_catalog()
    construct = catalog["constructs"][4]
    leaf = {"construct_id": "C005", "feature_id": "exact_spec_id",
            "human_name": "Example output", "family_spec_approved": True}
    attach_family_outputs(catalog, "F02", [leaf])
    assert catalog["constructs"][4] is construct
    assert construct["construct_id"] == "C005"
    assert construct["outputs"] == [leaf]
    assert catalog["outputs"][-1] == leaf
    with pytest.raises(ValueError, match="Duplicate"):
        attach_family_outputs(catalog, "F02", [leaf])
    with pytest.raises(ValueError, match="existing construct"):
        attach_family_outputs(load_feature_catalog(), "F01", [leaf])
    with pytest.raises(ValueError, match="approval"):
        attach_family_outputs(load_feature_catalog(), "F02", [{
            "construct_id": "C005", "feature_id": "unapproved"}])


def test_gui_columns_details_search_and_structured_recommendations():
    window = _window()
    tabs = {window.tabs.tabText(i): i for i in range(window.tabs.count())}
    assert not window.tabs.isTabEnabled(tabs["Quality Control"])
    assert window.tabs.isTabEnabled(tabs["Acoustic Features"])
    assert "Physiological Features" not in tabs
    assert [window.feature_tree.headerItem().text(i) for i in range(8)] == [
        "Feature", "Code name", "Evidence", "Use", "Task(s)",
        "Unit", "QC range", "Analysis unit",
    ]
    assert len(window.subsystem_items) == 13
    assert len(window.construct_items) == 79
    assert set(window.feature_items) == FAMILY08_IDS | FAMILY09_IDS
    jitter = window.construct_items["C005"]
    assert jitter.text(1) == "—"
    assert jitter.text(2) == "—"
    assert jitter.data(0, Qt.UserRole + 1) == "RED"
    assert not jitter.icon(0).isNull()
    assert "exact leaf feature ID" in jitter.toolTip(0)
    window.feature_tree.setCurrentItem(jitter)
    fields = {window.feature_detail_box.topLevelItem(i).text(0):
              window.feature_detail_box.topLevelItem(i).text(1)
              for i in range(window.feature_detail_box.topLevelItemCount())}
    assert fields["Code name"] == "—"
    assert fields["Evidence"] == "—"
    window.task_name_edit.setText("WSTG")
    groups = [window.feature_recommendations_box.topLevelItem(i).text(0)
              for i in range(window.feature_recommendations_box.topLevelItemCount())]
    assert any(g.startswith("Recommended (") for g in groups)
    assert any(g.startswith("Conditional (") for g in groups)
    assert any(g.startswith("Unavailable (") for g in groups)
    window.feature_search_edit.setText("Jitter")
    assert not jitter.isHidden()
    window.feature_search_edit.setText("WSTG")
    assert not window.construct_items["C001"].isHidden()
    window.close()


def test_constructs_are_visible_but_unselectable_and_buttons_choose_no_fake_outputs():
    window = _window()
    window.task_name_edit.setText("WSTG")
    assert not window.construct_items["C005"].flags() & Qt.ItemIsUserCheckable
    assert not window.construct_items["C013"].flags() & Qt.ItemIsUserCheckable
    assert not window.subsystem_items["F02"].flags() & Qt.ItemIsUserCheckable
    window.select_recommended_features()
    assert window._selected_feature_names() == []
    window.select_all_features()
    assert set(window._selected_feature_names()) == {
        feature_id for feature_id, item in window.catalog_outputs.items()
        if item["use_status"] == "Production" and item["selectable"]}
    assert not window.run_features_btn.isEnabled()
    assert not window.min_pause_feature_spin.isEnabled()
    assert not window.computation_mode_combo.isEnabled()
    research = window.construct_items["C004"]
    assert research.isHidden()
    window.show_research_check.setChecked(True)
    assert not research.isHidden()
    window.close()


def test_recommended_button_chooses_green_only(tmp_path: Path):
    window = _window()
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    (final / "final_segmentation_decisions.csv").write_text("recording_id\n", encoding="utf-8")
    (final / "final_segmentation_intervals.csv").write_text("recording_id\n", encoding="utf-8")
    manifest = tmp_path / "prompt.json"
    manifest.write_text(json.dumps({
        "schema_version": "1", "task_id": "bamboo_passage",
        "prompt_version": "v1", "count_source": "frozen_prompt",
        "syllable_count": 10, "word_count": 5,
        "applies_to_all_recordings": True,
    }), encoding="utf-8")
    window._run_root = tmp_path
    window.task_name_edit.setText("Bamboo Passage")
    window.feature_prompt_manifest_edit.setText(str(manifest))
    assert window.feature_items["speaking_rate_syll_s"].data(0, Qt.UserRole + 1) == "GREEN"
    conditional = window.catalog_outputs["speaking_rate_words_min"]
    conditional["recommended_tasks"] = []
    conditional["conditional_tasks"] = ["bamboo_passage"]
    window._refresh_feature_count_label()
    assert window.feature_items["speaking_rate_words_min"].data(0, Qt.UserRole + 1) == "AMBER"
    window.select_recommended_features()
    assert set(window._selected_feature_names()) == (
        FAMILY08_IDS - {"speaking_rate_words_min"} |
        (FAMILY09_IDS - {"pause_pattern_components", "pause_pattern_factor_if_frozen"}))
    window.task_name_edit.setText("WSTG")
    assert window.feature_items["speaking_rate_syll_s"].data(0, Qt.UserRole + 1) == "RED"
    window.select_recommended_features()
    assert window._selected_feature_names() == []
    window.close()
