from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff


def test_delivery_separates_main_from_supplementary_outputs(tmp_path: Path) -> None:
    tables = tmp_path / "acoustic" / "004_features" / "tables"
    tables.mkdir(parents=True)
    handoff = write_feature_handoff(
        tables,
        pd.DataFrame([{"recording_id": "r1", "task": "vowel", "f0_mean": 120.0}]),
        pd.DataFrame([{"feature": "f0_mean", "family": "phonatory", "formula": "mean(f0(t))"}]),
        pd.DataFrame([{"recording_id": "r1", "feature": "f0_mean", "status": "computed"}]),
        "acoustic",
        tables / "legacy_values.csv",
        tables / "legacy_registry.csv",
    )
    diagnostic = tmp_path / "acoustic" / "003_quality_control" / "plots" / "qc.png"
    diagnostic.parent.mkdir(parents=True)
    diagnostic.write_bytes(b"png")
    delivery = build_feature_delivery(tmp_path, "acoustic", handoff)

    main = delivery["main_dir"]
    assert all((main / name).exists() for name in (
        "feature_values.csv", "feature_registry.csv", "feature_status.csv", "feature_export_manifest.json", "README.md"
    ))
    assert not (main / "metadata_context.csv").exists()
    assert "metadata_context.csv" not in (main / "README.md").read_text(encoding="utf-8")
    catalog = pd.read_csv(delivery["catalog_csv"])
    assert "acoustic/003_quality_control/plots/qc.png" in catalog["relative_path"].str.replace("\\", "/").tolist()
    manifest = json.loads(delivery["delivery_manifest_json"].read_text(encoding="utf-8"))
    assert manifest["contract"] == "vslp-modality-feature-delivery"
    assert manifest["modality"] == "acoustic"


def test_kinematic_label_uses_plural_output_folder(tmp_path: Path) -> None:
    tables = tmp_path / "kinematics" / "006_features" / "tables"
    tables.mkdir(parents=True)
    handoff = write_feature_handoff(
        tables,
        pd.DataFrame([{"recording_id": "r1", "task": "open_close", "rom_vert_med": 0.2}]),
        pd.DataFrame([{"feature": "rom_vert_med", "family": "lower_lip", "formula": "P95(x)-P05(x)"}]),
        pd.DataFrame(),
        "kinematic",
        tables / "legacy_values.csv",
        tables / "legacy_registry.csv",
    )
    delivery = build_feature_delivery(tmp_path, "kinematic", handoff)
    assert delivery["main_dir"] == tmp_path / "kinematics" / "feature_handoff" / "main"
