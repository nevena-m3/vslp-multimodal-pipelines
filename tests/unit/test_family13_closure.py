"""Family 13 source closure: an exact ID is not an executable algorithm."""

import json
import os
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from vslp.acoustic.features.catalog import feature_availability, load_feature_catalog, task_fit
from vslp.acoustic.features.mixed_dispatch import _executors
from vslp.acoustic.features.stage import FeatureExtractionConfig, _select_registry


EXPECTED_UNAVAILABLE = {
    "ppe_source_replication_only", "dynamics_det_dmfcc_components",
    "dynamics_articulation_rate", "dynamics_factor_if_frozen",
    "rhythm_factor_if_frozen", "regularity_visibility_density",
    "regularity_psi_components", "regularity_factor_if_frozen",
}
EXPECTED_TEMPLATES = {
    "shannon_amp_entropy_<binning>", "sample_entropy_m<...>_r<...>",
    "wpd_shannon_entropy_<wavelet>_L<level>", "psd_spectral_entropy_<config>",
    "wavelet_energy_<wavelet>_L<level>_<node>", "rqa_det_mfcc<k>_<config>",
    "rhythm_moddepth_<band>", "rhythm_psi_<coupling>",
}


def test_family13_source_audit_covers_all_eight_constructs_without_defaults():
    path = Path(__file__).parents[2] / "src/vslp/acoustic/features/data/family13.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    assert [c["construct_id"] for c in source["constructs"]] == [
        f"C{n:03d}" for n in range(72, 80)]
    assert {c["status"] for c in source["constructs"]} == {
        "NOT_IMPLEMENTED", "UNRESOLVED_TEMPLATE"}
    assert {feature for c in source["constructs"] for feature in c.get("source_ids", [])} == (
        EXPECTED_UNAVAILABLE)
    assert {feature for c in source["constructs"]
            for feature in c.get("source_templates", [])} == EXPECTED_TEMPLATES
    assert all(c["missing_definition"] and c["evidence_level"] and
               c["evidence_study_count"] > 0 and c["evidence_entry_count"] > 0
               for c in source["constructs"])
    assert "whitening filter" in source["constructs"][0]["missing_definition"].lower()
    assert "engineering bridge" in source["constructs"][4]["missing_definition"].lower()
    assert all("factor" in c["missing_definition"] for c in source["constructs"]
               if any("factor_if_frozen" in feature for feature in c.get("source_ids", [])))


def test_family13_registry_has_evidence_but_no_executable_or_template_leaf():
    catalog = load_feature_catalog()
    constructs = {c["construct_id"]: c for c in catalog["constructs"]}
    leaves = {o["feature_id"]: o for o in catalog["outputs"] if o["family_id"] == "F13"}
    assert len(catalog["constructs"]) == 79
    assert len(catalog["outputs"]) == 138
    assert sum(feature_availability(o)[0] == "Implemented" for o in catalog["outputs"]) == 115
    assert set(leaves) == EXPECTED_UNAVAILABLE
    assert all(feature_availability(o)[0] == "Not implemented" and not o["selectable"]
               and o["blocked_reason"] for o in leaves.values())
    assert all(feature not in leaves for feature in EXPECTED_TEMPLATES)
    assert {feature for c in constructs.values() if c["family_id"] == "F13"
            for feature in c["source_feature_templates"]} == EXPECTED_TEMPLATES
    assert {constructs[f"C{n:03d}"]["evidence_level"] for n in range(72, 80)} == {
        "LIMITED", "NOT ESTABLISHED"}
    assert task_fit(constructs["C072"], "Sustained /a/") == "Recommended"
    assert task_fit(constructs["C072"], "WSTG") == "Not specified"
    assert task_fit(constructs["C073"], "WSTG") == "Not specified"
    assert task_fit(constructs["C076"], "Sustained /a/") == "Not specified"
    assert task_fit(constructs["C076"], "Bamboo Passage") == "Recommended"
    assert all("mfa" not in str(o["prerequisites"]).casefold() for o in leaves.values())


def test_family13_cannot_be_dispatched_or_selected_until_source_contract_is_frozen():
    from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow

    assert all("F13" not in families for families, _ in _executors().values())
    mixed = _select_registry(FeatureExtractionConfig(selected_features=[
        "f0_mean_hz", "pause_count", "mfcc01_mean"]))
    assert set(mixed.feature) == {"f0_mean_hz", "pause_count", "mfcc01_mean"}
    with pytest.raises(ValueError, match="exact, approved output IDs"):
        _select_registry(FeatureExtractionConfig(selected_features=[
            "f0_mean_hz", "ppe_source_replication_only"]))
    with pytest.raises(ValueError, match="exact, approved output IDs"):
        _select_registry(FeatureExtractionConfig(selected_features=[
            "pause_count", "shannon_amp_entropy_<binning>"]))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QApplication.instance() or QApplication([])
    window = AcousticPipelineWindow()
    try:
        assert all(not window.feature_items[name].flags() & Qt.ItemIsUserCheckable
                   for name in EXPECTED_UNAVAILABLE)
        assert not window.subsystem_items["F13"].flags() & Qt.ItemIsUserCheckable
        assert "shannon_amp_entropy_<binning>" in window.construct_items["C073"].text(1)
        assert window.construct_items["C073"].text(3) == "Not implemented"
        assert window.construct_items["C073"].text(4) == "NOT ESTABLISHED"
        assert window.feature_items["ppe_source_replication_only"].text(1) == (
            "ppe_source_replication_only")
        assert window.feature_items["ppe_source_replication_only"].text(3) == (
            "Not implemented")
        window.select_all_features()
        assert not EXPECTED_UNAVAILABLE.intersection(window._selected_feature_names())
        assert all("<" not in name for name in window._selected_feature_names())
        assert {"f0_mean_hz", "pause_count", "mfcc01_mean"} <= set(
            window._selected_feature_names())
    finally:
        window.close()
