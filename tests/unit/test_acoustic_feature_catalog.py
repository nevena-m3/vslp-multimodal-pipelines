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
from vslp.acoustic.features.family03 import FAMILY03_IDS
from vslp.acoustic.features.family04 import FAMILY04_IDS, TOKEN_IDS, VOWEL_IDS
from vslp.acoustic.features.family06 import F06_PUBLIC_IDS as FAMILY06_IDS
from vslp.acoustic.features.family06 import UNAVAILABLE_IDS as FAMILY06_UNAVAILABLE
from vslp.acoustic.features.family07 import FAMILY07_IDS
from vslp.acoustic.features.family08_v2 import FAMILY08_IDS
from vslp.acoustic.features.family09_v2 import PUBLIC_IDS as FAMILY09_ACTIVE_IDS
from vslp.acoustic.features.family10_v2 import PUBLIC_IDS as FAMILY10_IDS
from vslp.acoustic.features.family11_v2 import PUBLIC_IDS as FAMILY11_IDS
from vslp.acoustic.features.family12_v2 import PUBLIC_IDS as FAMILY12_IDS
from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow
from vslp.gui.acoustic_app.feature_presentation import (
    construct_presentation_status, output_presentation_status,
)

FAMILY01_UNAVAILABLE = {"intonation_factor_score_if_frozen"}
FAMILY02_UNAVAILABLE = {"cpps_db", "harmonic_h1_h8_mean", "harmonic_h1_h8_sd",
                        "relh_h1_h8"}
FAMILY04_UNAVAILABLE = set()
FAMILY07_UNAVAILABLE = {"vowel_duration_<phone>_s"}
FAMILY11_UNAVAILABLE = set()
FAMILY12_UNAVAILABLE = {"s1_spectral_band_1_4khz_percentile1",
                        "s7_spectral_band_1_4khz_percentile1"}
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


FAMILY09_IDS = FAMILY09_ACTIVE_IDS | {"pause_pattern_factor_if_frozen"}

def test_exact_master_constructs_and_no_legacy_leaves():
    catalog = load_feature_catalog()
    assert len(catalog["families"]) == 13
    assert len(catalog["constructs"]) == 82
    assert len({c["construct_id"] for c in catalog["constructs"]}) == 82
    selectable = {o["feature_id"] for o in catalog["outputs"]
                  if o.get("selectable") and o.get("family_spec_approved")}
    assert len(selectable) == 373
    assert not {"f0_mean", "mean_pause_dur", "generic_signal_entropy"} & selectable
    with pytest.raises(ValueError, match="Duplicate"):
        validate_feature_ids([{"feature_id": "x"}, {"feature_id": "x"}])
    with pytest.raises(ValueError, match="exact"):
        validate_feature_ids([{"feature_id": "—"}])



def test_task_registry_is_explicit_and_has_no_semantic_classes():
    assert TASKS == {
        "sustained_a": "Sustained /a/", "bamboo_passage": "Bamboo Passage",
        "ddk": "DDK", "ddk_pa": "/pa/", "ddk_ta": "/ta/",
        "ddk_ka": "/ka/", "ddk_pataka": "/pataka/", "wstg": "WSTG",
        "sentence_s2": "S2", "sentence_s3": "S3", "sentence_s5": "S5",
        "sentence_s6": "S6", "sentence_s7": "S7", "sentence_s9": "S9",
    }
    assert task_key("WSTG") == "wstg"
    assert task_key("Bamboo Passage") == "bamboo_passage"
    assert task_key("Bamboo-like passage") is None
    assert all("task_class" not in c for c in load_feature_catalog()["constructs"])



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
                       if item["feature_id"] == "bamboo_speaking_rate_wpm")
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
                    if item["feature_id"] == "bamboo_articulation_rate_wpm")
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
    assert [window.feature_tree.headerItem().text(i)
            for i in range(window.feature_tree.columnCount())] == [
        "Feature", "Task recommendation", "Availability",
        "Recommended task", "Unit", "Range", "Analysis unit"]
    assert len(window.subsystem_items) == 13
    assert len(window.construct_items) == 77
    assert len(window.feature_items) == 394
    window.task_name_edit.setText("Bamboo Passage")
    assert window.feature_items["bamboo_speaking_rate_wpm"].text(1) == "✓ RECOMMENDED"
    assert window.feature_items["ddk_pa_rate_syll_s"].text(2) == "UNSUPPORTED FOR THIS TASK"
    window.feature_search_edit.setText("Jitter")
    assert not window.construct_items["C005"].isHidden()
    window.close()



def test_constructs_are_visible_but_unselectable_and_buttons_choose_no_fake_outputs():
    window = _window()
    assert window.construct_items["C005"].flags() & Qt.ItemIsUserCheckable
    assert window.subsystem_items["F02"].flags() & Qt.ItemIsUserCheckable
    window.select_all_features()
    selected = set(window._selected_feature_names())
    expected = {name for name in window.feature_items if window._feature_eligible(name)}
    assert selected == expected and len(selected) == 373
    disabled = window.feature_items["pause_pattern_factor_if_frozen"]
    assert not disabled.flags() & Qt.ItemIsUserCheckable
    assert disabled.checkState(0) == Qt.Unchecked
    assert any(button.text() == "Select all selectable features"
               for button in window.findChildren(QPushButton))
    window.close()



def test_presentation_statuses_are_derived_without_changing_registry_availability():
    catalog = load_feature_catalog()
    constructs = {item["construct_id"]: item for item in catalog["constructs"]}
    outputs = {item["feature_id"]: item for item in catalog["outputs"]}
    assert construct_presentation_status(constructs["C005"]) == "IMPLEMENTED"
    assert construct_presentation_status(constructs["C057"]) == "FACTOR_NOT_FROZEN"
    assert construct_presentation_status(constructs["C033"]) == "IMPLEMENTED"
    assert output_presentation_status(outputs["pause_pattern_factor_if_frozen"]) == "FACTOR_NOT_FROZEN"
    assert feature_availability(outputs["pause_pattern_factor_if_frozen"])[0] == "Not implemented"



def test_runnable_view_shows_only_concrete_selectable_leaves():
    window = _window()
    window.task_name_edit.setText("Bamboo Passage")
    window.feature_view_combo.setCurrentIndex(window.feature_view_combo.findData("runnable"))
    assert window.feature_items["f0_mean_hz"].isHidden()
    assert window.feature_items["pause_pattern_factor_if_frozen"].isHidden()
    window.select_all_features()
    assert len(window._selected_feature_names()) == 373
    window.feature_view_combo.setCurrentIndex(window.feature_view_combo.findData("all"))
    assert not window.feature_items["pause_pattern_factor_if_frozen"].isHidden()
    window.close()



def test_implemented_leaf_rendering_and_advisory_gray_or_amber_selection():
    window = _window()
    window.task_name_edit.setText("WSTG")
    for feature_id in FAMILY08_IDS | FAMILY09_ACTIVE_IDS:
        item = window.feature_items[feature_id]
        assert item.flags() & Qt.ItemIsUserCheckable
        assert item.text(1) == "— NOT RECOMMENDED"
        assert item.text(2) == "UNSUPPORTED FOR THIS TASK"
    gray = window.feature_items["bamboo_pause_count"]
    gray.setCheckState(0, Qt.Checked)
    assert "bamboo_pause_count" in window._selected_feature_names()
    conditional = window.catalog_outputs["bamboo_speaking_rate_wpm"]
    conditional["conditional_tasks"] = ["wstg"]
    window._refresh_feature_count_label()
    amber = window.feature_items["bamboo_speaking_rate_wpm"]
    assert amber.text(1) == "◇ SUPPORTED / OPTIONAL"
    amber.setCheckState(0, Qt.Checked)
    assert "bamboo_speaking_rate_wpm" in window._selected_feature_names()
    window.close()



def test_wstg_flattening_and_selection_dependent_prerequisites(tmp_path: Path):
    window = _window()
    window.task_name_edit.setText("WSTG")
    selected = FAMILY08_IDS | FAMILY09_ACTIVE_IDS
    for feature_id in selected:
        window.feature_items[feature_id].setCheckState(0, Qt.Checked)
    assert selected <= set(window._selected_feature_names())
    assert all(window.feature_items[f].text(2) == "UNSUPPORTED FOR THIS TASK"
               for f in selected)
    window.task_name_edit.setText("Bamboo Passage")
    assert selected <= set(window._selected_feature_names())
    assert all(window.feature_items[f].flags() & Qt.ItemIsUserCheckable
               for f in selected)
    window.close()



def test_specialized_scientific_inputs_appear_only_for_selected_dependencies():
    window = _window()
    window.task_name_edit.setText("Bamboo Passage")
    assert window.feature_prompt_row.isHidden()
    rate = window.feature_items["bamboo_speaking_rate_wpm"]
    rate.setCheckState(0, Qt.Checked)
    assert window.feature_prompt_row.isHidden()
    assert window.feature_bamboo_count_value.text() == "97 words"
    rate.setCheckState(0, Qt.Unchecked)
    assert window.feature_items["f1_median_hz"].flags() & Qt.ItemIsUserCheckable
    window.close()



def test_recommended_button_chooses_green_only(tmp_path: Path):
    window = _window()
    window.task_name_edit.setText("Bamboo Passage")
    window.select_recommended_features()
    selected = set(window._selected_feature_names())
    expected = {name for name in window.feature_items
                if window._feature_eligible(name)
                and window._feature_task_recommendation(
                    window.catalog_outputs[name], window._feature_task_name(),
                    window.task_type_combo.currentData())[0] == "GREEN"}
    assert selected == expected
    window.task_name_edit.setText("WSTG")
    window.select_recommended_features()
    selected = set(window._selected_feature_names())
    assert all(window._feature_task_recommendation(
        window.catalog_outputs[name], window._feature_task_name(),
        window.task_type_combo.currentData())[0] == "GREEN" for name in selected)
    window.close()



def test_family10_gui_implementation_independent_of_reviewed_run(tmp_path: Path):
    window = _window()
    window.task_name_edit.setText("DDK")
    leaf = window.feature_items["ddk_pa_rate_syll_s"]
    assert leaf.flags() & Qt.ItemIsUserCheckable
    assert leaf.text(2) == "Prerequisite required"
    leaf.setCheckState(0, Qt.Checked)
    window.task_name_edit.setText("/pa/")
    assert leaf.checkState(0) == Qt.Checked
    assert leaf.text(2) == "Prerequisite required"
    assert set(FAMILY10_IDS) <= set(window.feature_items)
    window.close()



def test_family01_and_family02_exact_leaf_availability_and_evidence():
    window = _window()
    assert window.feature_items["jitter_local_pct"].text(0).startswith("Jitter")
    assert window.feature_items["shimmer_apq11_pct"].text(0).startswith("Shimmer")
    assert "Cepstral peak prominence" in window.feature_items["cpp_db"].text(0)
    for feature_id in FAMILY01_IDS | FAMILY02_IDS:
        assert window.feature_items[feature_id].flags() & Qt.ItemIsUserCheckable
    for feature_id in FAMILY01_UNAVAILABLE | FAMILY02_UNAVAILABLE:
        assert not window.feature_items[feature_id].flags() & Qt.ItemIsUserCheckable
    window.close()



def test_family07_registry_gui_and_selection_dependent_alignment():
    window = _window()
    window.task_name_edit.setText("Bamboo Passage")
    for feature_id in ("bamboo_speech_duration_s", "bamboo_total_duration_s",
                       "delta_v_s", "varco_v_pct", "npvi_v_pct"):
        assert window.feature_items[feature_id].flags() & Qt.ItemIsUserCheckable
    assert window.feature_items["delta_v_s"].text(2) == "ALIGNMENT PACKAGE REQUIRED"
    assert window.feature_items["bamboo_total_duration_s"].text(2) != "ALIGNMENT PACKAGE REQUIRED"
    assert "vowel_duration_<phone>_s" not in window.feature_items
    window.close()



def test_family11_12_exact_leaves_and_reproducibility_blocks():
    catalog = load_feature_catalog()
    outputs = {item["feature_id"]: item for item in catalog["outputs"]}
    for feature_id in ("vowel_absolute_energy_fs2", "s5_sound_power_digital",
                       "vowel_wave_amp_skew", "s5_amp_sd_fs", "vowel_mfcc01_mean", "s6_mfcc13_sd",
                       "bamboo_spectral_bandwidth_p2_mean_hz", "s1_zcr_mean_fraction",
                       "s7_spec_m4_kurtosis", "bamboo_spectral_flatness"):
        assert feature_id in outputs
        assert feature_availability(outputs[feature_id])[0] == "Implemented"
        assert outputs[feature_id]["evidence_level"] in {"NOT ESTABLISHED", "LIMITED"}
    assert feature_availability(outputs["bamboo_visibility_graph_density_amp"])[0] == "Implemented"
    assert feature_availability(outputs["s1_spectral_band_1_4khz_percentile1"])[0] == "Not implemented"
    assert feature_availability(outputs["vowel_spectral_contrast_band6_db"])[0] == "Implemented"
    contrast = outputs["vowel_spectral_contrast_band6_db"]["parameter_profile"]["active_parameters"]
    assert contrast["n_bands"] == 6
    assert contrast["library_version"] == "0.11.0"
    assert contrast["input"] == "STFT magnitude"
