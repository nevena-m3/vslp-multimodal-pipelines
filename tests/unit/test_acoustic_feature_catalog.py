"""Source-only Acoustic Features registry and GUI contract."""

import json
import os
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QPushButton

from vslp.acoustic.features.catalog import (
    TASKS, attach_family_outputs, feature_availability, load_feature_catalog, prerequisite_issues,
    task_fit, task_indicator, task_key,
    validate_feature_ids,
)
from vslp.acoustic.features.stage import FeatureExtractionConfig, _select_registry
from vslp.acoustic.features.family01 import FAMILY01_IDS
from vslp.acoustic.features.family02 import FAMILY02_IDS
from vslp.acoustic.features.family04 import FAMILY04_IDS
from vslp.acoustic.features.family06 import IMPLEMENTED_IDS as FAMILY06_IDS
from vslp.acoustic.features.family06 import UNAVAILABLE_IDS as FAMILY06_UNAVAILABLE
from vslp.acoustic.features.family07 import FAMILY07_IDS
from vslp.acoustic.features.family08 import FAMILY08_IDS
from vslp.acoustic.features.family09 import FAMILY09_IDS
from vslp.acoustic.features.family10 import FAMILY10_IDS
from vslp.acoustic.features.family11 import FAMILY11_IDS
from vslp.acoustic.features.family12 import FAMILY12_IDS
from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow

FAMILY01_UNAVAILABLE = {"intonation_factor_score_if_frozen"}
FAMILY02_UNAVAILABLE = {"gne_source_replication_only", "pvi_9_14hz_source", "cpp_db",
                        "cpps_db", "harmonic_h1_h8_mean", "harmonic_h1_h8_sd",
                        "relh_h1_h8"}
FAMILY04_UNAVAILABLE = {"vowel_envelope_distance_ai_source"}
FAMILY07_UNAVAILABLE = {"vowel_duration_<phone>_s"}
FAMILY11_UNAVAILABLE = set()
FAMILY12_UNAVAILABLE = {"opensmile_1_4khz_<functional>"}
FAMILY13_UNAVAILABLE = {
    "ppe_source_replication_only", "dynamics_det_dmfcc_components",
    "dynamics_articulation_rate", "dynamics_factor_if_frozen",
    "rhythm_factor_if_frozen", "regularity_visibility_density",
    "regularity_psi_components", "regularity_factor_if_frozen",
}


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
    assert {o["feature_id"] for o in catalog["outputs"]} == (
        FAMILY01_IDS | FAMILY01_UNAVAILABLE | FAMILY02_IDS | FAMILY02_UNAVAILABLE |
        FAMILY04_IDS | FAMILY04_UNAVAILABLE | FAMILY06_IDS | FAMILY06_UNAVAILABLE |
        FAMILY07_IDS | FAMILY07_UNAVAILABLE | FAMILY08_IDS | FAMILY09_IDS | FAMILY10_IDS |
        FAMILY11_IDS | FAMILY11_UNAVAILABLE | FAMILY12_IDS | FAMILY12_UNAVAILABLE |
        FAMILY13_UNAVAILABLE)
    assert all(c["outputs"] == [] for c in catalog["constructs"]
                   if c["family_id"] not in {"F01", "F02", "F04", "F06", "F07", "F08", "F09", "F10",
                                             "F11", "F12", "F13"})
    assert all(c["evidence_level"] == "LIMITED" for c in catalog["constructs"]
               if c["family_id"] == "F03")
    assert all(c["evidence_level"] for c in catalog["constructs"]
               if c["family_id"] == "F13")
    assert all(c["evidence_level"] for c in catalog["constructs"]
               if c["family_id"] in {"F01", "F08", "F09"})
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
    assert task_fit(constructs["C001"], "WSTG") == "Not specified"
    assert task_fit(constructs["C002"], "WSTG") == "Not specified"
    assert task_fit(constructs["C002"], "Bamboo Passage") == "Recommended"
    assert task_fit(constructs["C002"], "sustained vowel") == "Not specified"


def test_task_fit_and_availability_are_independent():
    base = {"construct_id": "C048", "feature_id": "source_exact_id",
            "family_spec_approved": True, "selectable": True,
            "matrix_status": "PROD", "recommended_tasks": ["bamboo_passage"],
            "conditional_tasks": ["wstg"]}
    assert task_indicator(base, "Bamboo Passage")[0] == "GREEN"
    assert task_indicator(base, "WSTG")[0] == "AMBER"
    assert task_indicator(base, "Sustained /a/")[0] == "GRAY"
    assert task_indicator(base, "We See Three Geese")[0] == "GRAY"
    assert task_indicator({**base, "matrix_status": "BLOCKED"}, "Bamboo Passage")[0] == "GREEN"
    assert feature_availability({**base, "matrix_status": "BLOCKED"})[0] == "Not implemented"
    implemented = next(item for item in load_feature_catalog()["outputs"]
                       if item["feature_id"] == "speaking_rate_syll_s")
    assert feature_availability(implemented, prerequisite_problems=["prompt manifest missing"])[0] == "Implemented"
    assert task_indicator(base, "Bamboo Passage")[0] == "GREEN"
    assert feature_availability({"construct_id": "C001", "matrix_status": "PROD"})[0] == "Not implemented"
    assert task_fit(base, "WSTG", "Sentence / controlled speech") == "Conditional"
    assert task_fit(base, "WSTG-like", "Sentence / controlled speech") == "Not specified"
    assert task_fit({**base, "recommended_task_types": ["Sentence / controlled speech"]},
                    "WSTG-like", "Sentence / controlled speech") == "Not specified"
    assert task_fit({**base, "recommended_task_types": ["Sentence / controlled speech"],
                     "explicit_task_type_compatibility": True},
                    "WSTG-like", "Sentence / controlled speech") == "Recommended"
    family08 = next(item for item in load_feature_catalog()["outputs"]
                    if item["feature_id"] == "articulation_rate_syll_s")
    assert task_fit(family08, "WSTG", "Sentence / controlled speech") == "Not specified"


def test_legacy_ids_cannot_execute_and_family_spec_is_required():
    for selection in ([], ["f0_mean"], ["mean_pause_dur"], ["jitter"]):
        with pytest.raises(ValueError, match="legacy feature IDs cannot execute"):
            _select_registry(FeatureExtractionConfig(selected_features=selection))
    issues = prerequisite_issues({"feature_id": "f0_mean_hz"}, task_name="WSTG",
        final_decisions=None, final_intervals=None)
    assert "family specification" in issues[0]
    assert _select_registry(FeatureExtractionConfig(selected_features=["f0_mean_hz"])).feature.tolist() == ["f0_mean_hz"]


def test_family_leaves_attach_with_exact_ids_and_stable_construct_identity():
    catalog = load_feature_catalog()
    construct = catalog["constructs"][4]
    leaf = {"construct_id": "C005", "feature_id": "exact_spec_id",
            "human_name": "Example output", "family_spec_approved": True}
    attach_family_outputs(catalog, "F02", [leaf])
    assert catalog["constructs"][4] is construct
    assert construct["construct_id"] == "C005"
    assert construct["outputs"][-1] == leaf
    assert catalog["outputs"][-1] == leaf
    with pytest.raises(ValueError, match="Duplicate"):
        attach_family_outputs(catalog, "F02", [leaf])
    with pytest.raises(ValueError, match="existing construct"):
        attach_family_outputs(load_feature_catalog(), "F01", [leaf])
    with pytest.raises(ValueError, match="approval"):
        attach_family_outputs(load_feature_catalog(), "F02", [{
            "construct_id": "C005", "feature_id": "unapproved"}])


def test_construct_evidence_propagates_to_variants_without_leaf_override():
    catalog = load_feature_catalog()
    construct = next(c for c in catalog["constructs"] if c["construct_id"] == "C005")
    construct.update(evidence_level="MODERATE", evidence_study_count=4,
                     evidence_entry_count=7)
    leaves = [
        {"construct_id": "C005", "feature_id": "test_variant_a",
         "family_spec_approved": True},
        {"construct_id": "C005", "feature_id": "test_variant_b",
         "family_spec_approved": True},
    ]
    attach_family_outputs(catalog, "F02", leaves)
    assert all(leaf["evidence_level"] == "MODERATE" for leaf in leaves)
    assert all(leaf["evidence_study_count"] == 4 for leaf in leaves)


def test_gui_columns_details_search_and_structured_recommendations():
    window = _window()
    tabs = {window.tabs.tabText(i): i for i in range(window.tabs.count())}
    assert not window.tabs.isTabEnabled(tabs["Quality Control"])
    assert window.tabs.isTabEnabled(tabs["Acoustic Features"])
    assert "Physiological Features" not in tabs
    assert [window.feature_tree.headerItem().text(i) for i in range(9)] == [
        "Feature", "Code name", "Task recommendation", "Availability", "Evidence",
        "Recommended tasks", "Unit", "Range", "Analysis unit",
    ]
    assert len(window.subsystem_items) == 13
    assert len(window.construct_items) == 79
    assert set(window.feature_items) == (
        FAMILY01_IDS | FAMILY01_UNAVAILABLE | FAMILY02_IDS | FAMILY02_UNAVAILABLE |
            FAMILY04_IDS | FAMILY04_UNAVAILABLE | FAMILY06_IDS | FAMILY06_UNAVAILABLE |
        FAMILY07_IDS | FAMILY07_UNAVAILABLE | FAMILY08_IDS | FAMILY09_IDS | FAMILY10_IDS |
        FAMILY11_IDS | FAMILY11_UNAVAILABLE | FAMILY12_IDS | FAMILY12_UNAVAILABLE |
        FAMILY13_UNAVAILABLE)
    jitter = window.construct_items["C005"]
    assert jitter.text(1) == "—"
    assert jitter.text(4) == "MODERATE"
    assert jitter.data(0, Qt.UserRole + 1) == "GRAY"
    assert jitter.data(0, Qt.UserRole + 2) == "Implemented"
    assert jitter.text(2) == "" and jitter.icon(3).isNull()
    assert not jitter.icon(2).isNull()
    window.feature_tree.setCurrentItem(jitter)
    fields = {window.feature_detail_box.topLevelItem(i).text(0):
              window.feature_detail_box.topLevelItem(i).text(1)
              for i in range(window.feature_detail_box.topLevelItemCount())}
    assert fields["Code name"] == "—"
    assert fields["Evidence"] == "MODERATE"
    window.task_name_edit.setText("WSTG")
    groups = [window.feature_recommendations_box.topLevelItem(i).text(0)
              for i in range(window.feature_recommendations_box.topLevelItemCount())]
    assert any(g.startswith("Recommended for current task (") for g in groups)
    assert any(g.startswith("Conditional (") for g in groups)
    assert not any(g.startswith("Not specified (") for g in groups)
    assert not any(g.startswith("Execution issues (") for g in groups)
    wstg_construct = window.construct_items["C001"]
    assert wstg_construct.data(0, Qt.UserRole + 1) == "GRAY"
    assert wstg_construct.data(0, Qt.UserRole + 2) == "Implemented"
    assert wstg_construct.text(2) == ""
    window.feature_search_edit.setText("Jitter")
    assert not jitter.isHidden()
    window.feature_search_edit.setText("WSTG")
    assert not window.construct_items["C001"].isHidden()
    window.close()


def test_constructs_are_visible_but_unselectable_and_buttons_choose_no_fake_outputs():
    window = _window()
    window.task_name_edit.setText("WSTG")
    assert window.construct_items["C005"].flags() & Qt.ItemIsUserCheckable
    assert not window.construct_items["C013"].flags() & Qt.ItemIsUserCheckable
    assert window.subsystem_items["F02"].flags() & Qt.ItemIsUserCheckable
    window.select_recommended_features()
    selected = window._selected_feature_names()
    assert all(window.feature_items[name].data(0, Qt.UserRole + 1) == "GREEN"
               and window.feature_items[name].text(3) == "Implemented" for name in selected)
    window.select_all_features()
    assert set(window._selected_feature_names()) == {
        feature_id for feature_id, item in window.catalog_outputs.items()
        if feature_availability(item)[0] == "Implemented"}
    assert window.run_features_btn.isEnabled()
    assert not window.min_pause_feature_spin.isEnabled()
    assert not window.computation_mode_combo.isEnabled()
    assert not window.feature_items["pause_pattern_components"].isHidden()
    assert any(button.text() == "All implemented" for button in window.findChildren(QPushButton))
    assert not any(button.text() == "All production" for button in window.findChildren(QPushButton))
    window.close()


def test_implemented_leaf_rendering_and_advisory_gray_or_amber_selection():
    window = _window()
    window.task_name_edit.setText("WSTG")
    catalog = window.catalog_outputs
    implemented = FAMILY08_IDS | (FAMILY09_IDS - {"pause_pattern_factor_if_frozen"})
    for feature_id in implemented:
        leaf = window.feature_items[feature_id]
        assert leaf.text(1) == feature_id
        assert leaf.text(3) == "Implemented"
        assert leaf.text(4) in {"STRONG", "MODERATE", "LIMITED", "NOT ESTABLISHED"}
        assert leaf.flags() & Qt.ItemIsUserCheckable
        assert leaf.text(2) == "" and not leaf.icon(2).isNull()
        assert leaf.icon(3).isNull()
    for feature_id in {"pause_pattern_factor_if_frozen"}:
        assert window.feature_items[feature_id].text(3) == "Not implemented"
        assert not window.feature_items[feature_id].flags() & Qt.ItemIsUserCheckable
    assert window.construct_items["C050"] is window.feature_items["total_pause_duration_s"]
    assert window.construct_items["C050"].text(1) == "total_pause_duration_s"
    assert window.construct_items["C050"].text(4) == "MODERATE"
    gray = window.feature_items["pause_count"]
    assert gray.data(0, Qt.UserRole + 1) == "GRAY"
    gray.setCheckState(0, Qt.Checked)
    assert "pause_count" in window._selected_feature_names()
    conditional = catalog["speaking_rate_syll_s"]
    conditional["conditional_tasks"] = ["wstg"]
    window._refresh_feature_count_label()
    amber = window.feature_items["speaking_rate_syll_s"]
    assert amber.data(0, Qt.UserRole + 1) == "AMBER"
    amber.setCheckState(0, Qt.Checked)
    assert "speaking_rate_syll_s" in window._selected_feature_names()
    assert amber.text(2) == "" and amber.icon(3).isNull()
    window.close()


def test_wstg_flattening_and_selection_dependent_prerequisites(tmp_path: Path):
    window = _window()
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    (final / "final_segmentation_decisions.csv").write_text("recording_id\n", encoding="utf-8")
    (final / "final_segmentation_intervals.csv").write_text("recording_id\n", encoding="utf-8")
    window._run_root = tmp_path
    window.task_name_edit.setText("WSTG")
    window.task_type_combo.setCurrentText("Sentence / controlled speech")
    implemented_f09 = FAMILY09_IDS - {"pause_pattern_factor_if_frozen"}
    for feature_id in implemented_f09:
        window.feature_items[feature_id].setCheckState(0, Qt.Checked)
    assert window.feature_prompt_row.isHidden()
    assert "Current-run issues: 0" in window.feature_count_label.text()
    assert all(window.feature_items[f].data(0, Qt.UserRole + 1) == "GRAY"
               for f in implemented_f09)
    for feature_id in FAMILY08_IDS:
        window.feature_items[feature_id].setCheckState(0, Qt.Checked)
    assert not window.feature_prompt_row.isHidden()
    assert "Current-run issues: 3" in window.feature_count_label.text()
    assert all(window.feature_items[f].flags() & Qt.ItemIsUserCheckable
               for f in FAMILY08_IDS | implemented_f09)
    assert window.construct_items["C049"] is window.feature_items["articulation_rate_syll_s"]
    assert window.construct_items["C053"] is window.feature_items["pause_count"]
    assert window.construct_items["C048"].childCount() == 2
    assert window.construct_items["C051"].childCount() == 2
    assert all(window.feature_items[f].text(1) == f for f in FAMILY08_IDS | implemented_f09)
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
    expected = {name for name, item in window.feature_items.items()
                if item.data(0, Qt.UserRole + 1) == "GREEN"
                and item.text(3) == "Implemented"}
    assert set(window._selected_feature_names()) == expected
    window.task_name_edit.setText("WSTG")
    assert window.feature_items["speaking_rate_syll_s"].data(0, Qt.UserRole + 1) == "GRAY"
    window.select_recommended_features()
    expected = {name for name, item in window.feature_items.items()
                if item.data(0, Qt.UserRole + 1) == "GREEN"
                and item.text(3) == "Implemented"}
    assert set(window._selected_feature_names()) == expected
    window.close()


def test_family10_gui_implementation_independent_of_reviewed_run(tmp_path: Path):
    window = _window()
    final = tmp_path / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    (final / "final_segmentation_intervals.csv").write_text(
        "recording_id\n", encoding="utf-8")
    window._run_root = tmp_path
    window.task_name_edit.setText("DDK")
    assert window.feature_items["ddk_rate_syll_s"].data(0, Qt.UserRole + 1) == "GREEN"
    assert window.feature_items["ddk_rate_syll_s"].data(0, Qt.UserRole + 2) == "Implemented"
    decisions.write_text("segmentation_method,final_decision\nsilero_vad,KEEP_AUTO\n",
                         encoding="utf-8")
    window._refresh_feature_count_label()
    assert window.feature_items["ddk_rate_syll_s"].data(0, Qt.UserRole + 1) == "GREEN"
    decisions.write_text("segmentation_method,final_decision\nddk_energy,KEEP_MANUAL\n",
                         encoding="utf-8")
    window._refresh_feature_count_label()
    assert window.feature_items["ddk_rate_syll_s"].data(0, Qt.UserRole + 2) == "Implemented"
    window.select_recommended_features()
    assert set(FAMILY10_IDS) <= set(window._selected_feature_names())
    window.feature_tree.setCurrentItem(window.feature_items["ddk_cycle_mad_s"])
    fields = {window.feature_detail_box.topLevelItem(i).text(0):
              window.feature_detail_box.topLevelItem(i).text(1)
              for i in range(window.feature_detail_box.topLevelItemCount())}
    assert fields["Code name"] == "ddk_cycle_mad_s"
    assert fields["Unit"] == "s"
    assert fields["Task recommendation"] == "Recommended for DDK."
    assert fields["Availability"] == "Implemented"
    assert fields["Parameter profile"].endswith("(read-only)")
    window.close()


def test_family01_and_family02_exact_leaf_availability_and_evidence():
    window = _window()
    for feature_id in FAMILY01_IDS | FAMILY02_IDS:
        item = window.feature_items[feature_id]
        assert item.text(1) == feature_id
        assert item.text(3) == "Implemented"
        assert item.text(4) in {"MODERATE", "LIMITED", "NOT ESTABLISHED"}
        assert item.flags() & Qt.ItemIsUserCheckable
    for feature_id in FAMILY01_UNAVAILABLE | FAMILY02_UNAVAILABLE:
        item = window.feature_items[feature_id]
        assert item.text(1) == feature_id
        assert item.text(3) == "Not implemented"
        assert not item.flags() & Qt.ItemIsUserCheckable
    window.close()


def test_family07_registry_gui_and_selection_dependent_alignment():
    window = _window()
    window.task_name_edit.setText("WSTG")
    for feature_id in FAMILY07_IDS:
        item = window.feature_items[feature_id]
        assert item.text(1) == feature_id
        assert item.text(3) == "Implemented"
        assert item.text(4) in {"LIMITED", "NOT ESTABLISHED"}
        assert item.flags() & Qt.ItemIsUserCheckable
    blocked = window.feature_items["vowel_duration_<phone>_s"]
    assert blocked.text(3) == "Not implemented"
    assert not blocked.flags() & Qt.ItemIsUserCheckable
    window.feature_items["speech_time_s"].setCheckState(0, Qt.Checked)
    assert window.feature_alignment_row.isHidden()
    assert "Current-run issues: 1" in window.feature_count_label.text()  # frozen timing absent
    window.feature_items["mean_word_duration_s"].setCheckState(0, Qt.Checked)
    assert not window.feature_alignment_row.isHidden()
    assert window.feature_items["mean_word_duration_s"].data(0, Qt.UserRole + 1) == "GRAY"
    assert window.feature_items["mean_word_duration_s"].flags() & Qt.ItemIsUserCheckable
    window.close()


def test_family11_12_exact_leaves_and_reproducibility_blocks():
    catalog = load_feature_catalog()
    outputs = {item["feature_id"]: item for item in catalog["outputs"]}
    for feature_id in ("absolute_energy_fs2", "sound_power_digital", "wave_amp_skew",
                       "amp_sd_fs", "mfcc01_mean", "mfcc13_sd",
                       "spectral_bandwidth_p2_mean_hz", "zcr_mean_fraction",
                       "spec_m4_kurtosis", "spectral_flatness"):
        assert feature_id in outputs
        assert feature_availability(outputs[feature_id])[0] == "Implemented"
        assert outputs[feature_id]["evidence_level"] in {"NOT ESTABLISHED", "LIMITED"}
    assert feature_availability(outputs["visibility_graph_density_amp"])[0] == "Implemented"
    assert feature_availability(outputs["opensmile_1_4khz_<functional>"])[0] == "Not implemented"
    assert feature_availability(outputs["spectral_contrast_band6_db"])[0] == "Implemented"
    contrast = outputs["spectral_contrast_band6_db"]["parameter_profile"]["active_parameters"]
    assert contrast["n_bands"] == 6
    assert contrast["library_version"] == "0.11.0"
    assert contrast["input"] == "STFT magnitude"
