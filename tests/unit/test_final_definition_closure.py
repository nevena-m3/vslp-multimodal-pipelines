"""Registry identity provenance and deliberately blocked source definitions."""

import csv
from collections import Counter
from pathlib import Path

from vslp.acoustic.features.catalog import load_feature_catalog
from vslp.acoustic.features.catalog import feature_availability
from vslp.acoustic.features.mixed_dispatch import _executors


ROOT = Path(__file__).parents[2]


def test_family03_has_evidence_but_no_invented_public_ids():
    catalog = load_feature_catalog()
    constructs = [c for c in catalog["constructs"] if c["family_id"] == "F03"]
    assert len(constructs) == 8
    assert all(c["id_provenance"] == "NO_PUBLIC_ID" and
               c["source_placeholder"] == "blocked_proprietary" and
               c["source_status"] == "NOT_IMPLEMENTED" and
               c["evidence_level"] == "LIMITED" and
               c["evidence_entry_count"] > 0 and c["outputs"] == []
               for c in constructs)
    assert not any(o["family_id"] == "F03" for o in catalog["outputs"])


def test_inventory_counts_identity_and_unavailable_selection():
    with (ROOT / "docs/acoustic_feature_implementation_inventory.csv").open(
            encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert Counter(r["status"] for r in rows) == {
        "IMPLEMENTED": 115, "NOT_IMPLEMENTED": 21, "UNRESOLVED_TEMPLATE": 19}
    assert Counter(r["id_provenance"] for r in rows) == {
        "SOURCE_DEFINED_ID": 136, "PROJECT_TEMPLATE": 19}
    assert all(r["selectable_in_gui"] == "False" for r in rows
               if r["status"] != "IMPLEMENTED")
    assert all(r["not_implemented_reason_class"] in {
        "PROPRIETARY_UNREPRODUCIBLE", "SOURCE_DEFINITION_INCOMPLETE",
        "FACTOR_TRANSFORM_NOT_FROZEN", "SOURCE_PACKAGE_PARITY_UNAVAILABLE",
        "OTHER_EXPLICIT_REASON"}
        for r in rows if r["status"] == "NOT_IMPLEMENTED")
    catalog = load_feature_catalog()
    assert len(catalog["constructs"]) == 79
    assert {o["feature_id"] for o in catalog["outputs"] if "<" not in o["feature_id"]} == {
        r["feature_id"] for r in rows if r["status"] != "UNRESOLVED_TEMPLATE"}
    assert all(o["id_provenance"] == ("PROJECT_TEMPLATE" if "<" in o["feature_id"]
                                      else "SOURCE_DEFINED_ID") for o in catalog["outputs"])


def test_family13_identity_is_source_defined_not_inferred():
    catalog = load_feature_catalog()
    family13 = [o for o in catalog["outputs"] if o["family_id"] == "F13"]
    assert len(family13) == 8
    assert all(o["id_provenance"] == "SOURCE_DEFINED_ID" and not o["selectable"]
               for o in family13)
    assert not any(o["feature_id"] in {"shannon_amp_entropy_<binning>",
                                        "sample_entropy_m<...>_r<...>"}
                   for o in catalog["outputs"])


def test_release_registry_has_unique_dispatch_and_complete_implemented_metadata():
    catalog = load_feature_catalog()
    outputs = catalog["outputs"]
    assert len({o["feature_id"] for o in outputs}) == len(outputs)
    family_groups = {}
    for group, (families, _executor) in _executors().items():
        for family in families:
            family_groups.setdefault(family, []).append(group)
    for output in outputs:
        implemented = feature_availability(output)[0] == "Implemented"
        assert bool(output["selectable"]) == implemented
        if implemented:
            assert len(family_groups.get(output["family_id"], [])) == 1
            assert all(output.get(field) for field in (
                "feature_id", "evidence_level", "unit", "analysis_unit",
                "algorithm_version", "default_parameter_set_id"))
            assert output.get("qc_range_text") is not None
        else:
            assert not output["selectable"]
