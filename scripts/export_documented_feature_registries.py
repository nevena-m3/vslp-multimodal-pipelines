"""Export source-controlled registry snapshots without importing heavy DSP modules."""

from __future__ import annotations

import ast
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "reference"


def _formula(feature: str) -> str:
    if feature.startswith("path_"):
        return "median or IQR across movements of sum(|x[i]-x[i-1]|)"
    if feature.startswith("rom_"):
        return "median or IQR across movements of P95(x)-P05(x)"
    if feature.startswith("sLL_"):
        return "requested summary of |dx/dt|; derivative uses timestamps"
    if feature.startswith("aLL_"):
        return "requested summary of |d2x/dt2|; derivative uses signed dx/dt"
    if feature.startswith("aspect_"):
        return "requested summary of mouth_opening/lip_width"
    if feature.startswith("jaw_lat_"):
        return "requested summary of lower-face left/right reference distance ratio"
    if feature.startswith("lip_symm_ratio_"):
        return "requested summary of left/right commissure-to-midline distance ratio"
    if feature == "lat_xcorr":
        return "max_lag normalized cross-correlation(left_commissure,right_commissure)"
    return "See computation note"


def export_kinematic() -> None:
    source = ROOT / "src" / "vslp" / "analysis" / "kinematics" / "features.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    rows = []
    for node in ast.walk(tree):
        is_plain = isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KINEMATIC_FEATURE_SPECS" for t in node.targets)
        is_annotated = isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "KINEMATIC_FEATURE_SPECS"
        if not (is_plain or is_annotated):
            continue
        for call in node.value.elts:
            values = [ast.literal_eval(arg) for arg in call.args]
            feature, group, label, status, tier, native, unit, landmarks, normalization, aggregation, interpretation, source_function = values
            rows.append({
                "feature": feature, "modality": "kinematic", "family": group,
                "meaning": label, "unit": unit, "formula": _formula(feature),
                "computation_note": native, "task_scope": "repeated oral-motor task; whole-file fallback flagged",
                "aggregation": aggregation, "required_inputs": ", ".join(map(str, landmarks)),
                "normalization": normalization, "evidence_tier": tier,
                "implementation_status": status, "interpretation": interpretation,
                "source_document": "Kinematic Feature Validation.xlsx",
                "source_location": "Main/Summary; mapped by feature family",
                "source_function": source_function,
            })
        break
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "WORKBOOK_VALIDATED_FEATURE_ROWS":
            for row in ast.literal_eval(node.value):
                rows.append({**row, "modality": "kinematic", "source_function": "validated_workbook_geometry"})
            break
    if not rows:
        raise RuntimeError("KINEMATIC_FEATURE_SPECS not found")
    path = OUT / "kinematic_feature_registry.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    export_kinematic()
