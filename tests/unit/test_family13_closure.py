"""F13 active operational leaves and deferred historical source identities."""

import json
import os
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from vslp.acoustic.features.catalog import feature_availability, load_feature_catalog
from vslp.acoustic.features.family13_v1 import PUBLIC_IDS
from vslp.acoustic.features.mixed_dispatch import _executors
from vslp.acoustic.features.stage import FeatureExtractionConfig, _select_registry


def test_family13_historical_constructs_and_operational_status():
    path = Path(__file__).parents[2] / "src/vslp/acoustic/features/data/family13.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    assert [row["construct_id"] for row in source["constructs"]] == [
        f"C{n:03d}" for n in range(72, 80)]
    assert {row["status"] for row in source["constructs"]} == {
        "IMPLEMENTED_VSLP_OPERATIONALIZATION"}
    assert source["registry_version"] == "family13_vslp_operational_v1"


def test_family13_registry_keeps_source_placeholders_unavailable():
    catalog = load_feature_catalog()
    leaves = [row for row in catalog["outputs"] if row["family_id"] == "F13"]
    assert len(leaves) == 96
    assert {row["feature_id"] for row in leaves if row["selectable"]} == PUBLIC_IDS
    deferred = [row for row in leaves if not row["selectable"]]
    assert len(deferred) == 9
    assert all(feature_availability(row)[0] == "Not implemented" for row in deferred)
    assert all(feature_availability(row)[0] == "Implemented" for row in leaves
               if row["selectable"])
    assert "vowel_sample_entropy_pending_definition" in {
        row["feature_id"] for row in deferred}
    assert not any("factor_if_frozen" in row["feature_id"] and row["selectable"]
                   for row in leaves)


def test_family13_exact_ids_dispatch_and_selection():
    assert sum("F13" in families for families, _ in _executors().values()) == 1
    registry = _select_registry(FeatureExtractionConfig(selected_features=[
        "vowel_ppe_bits", "bamboo_mfcc01_rqa_det"]))
    assert set(registry.feature) == {"vowel_ppe_bits", "bamboo_mfcc01_rqa_det"}
    with pytest.raises(ValueError, match="exact, approved output IDs"):
        _select_registry(FeatureExtractionConfig(selected_features=[
            "ppe_source_replication_only"]))
    with pytest.raises(ValueError, match="exact, approved output IDs"):
        _select_registry(FeatureExtractionConfig(selected_features=[
            "vowel_sample_entropy_pending_definition"]))


def test_gui_exposes_exactly_87_f13_leaves_and_keeps_placeholders_disabled():
    from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QApplication.instance() or QApplication([])
    window = AcousticPipelineWindow()
    try:
        selectable = {feature for feature in PUBLIC_IDS
                      if window.feature_items[feature].flags() & Qt.ItemIsUserCheckable}
        assert selectable == PUBLIC_IDS
        assert not (window.feature_items["vowel_sample_entropy_pending_definition"].flags()
                    & Qt.ItemIsUserCheckable)
        assert not (window.feature_items["dynamics_factor_if_frozen"].flags()
                    & Qt.ItemIsUserCheckable)
    finally:
        window.close()
