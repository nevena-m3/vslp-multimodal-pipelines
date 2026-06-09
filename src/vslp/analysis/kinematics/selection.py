"""Landmark selection diagnostics for the kinematics GUI.

The GUI lets the analyst choose a subset of MediaPipe Face Landmarker points for
normalization, QC, and feature computation. This module keeps the scientific
checks out of the Qt layer so they can be tested directly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

# Region definitions mirror the GUI overlay. They are intentionally small,
# reviewer-facing groups rather than a full MediaPipe ontology.
LANDMARK_REGION_SETS: dict[str, frozenset[int]] = {
    "mouth/lips": frozenset({0, 13, 14, 17, 37, 40, 57, 61, 78, 81, 82, 87, 88, 95, 146, 164, 178, 181, 185, 191, 267, 270, 287, 291, 308, 311, 312, 317, 318, 324, 375, 402, 405, 409, 415}),
    "jaw/chin/lower face": frozenset({17, 152, 175, 199, 200, 148, 176, 149, 150, 136, 172, 58, 132, 361, 288, 397, 365, 379, 378, 400}),
    "eye/canthus anchors": frozenset({33, 133, 159, 145, 263, 362, 386, 374, 246, 161, 160, 144, 163, 7, 466, 388, 387, 373, 390, 249}),
    "nose/midline": frozenset({1, 2, 4, 5, 6, 8, 9, 10, 94, 97, 98, 168, 195, 197, 326, 327}),
    "brows/upper face": frozenset({70, 105, 107, 336, 334, 300, 46, 52, 53, 65, 55, 285, 295, 282, 283, 276}),
    "cheeks/contour": frozenset({10, 21, 54, 58, 67, 93, 103, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234, 251, 284, 288, 297, 323, 332, 356, 361, 365, 377, 378, 379, 389, 397, 454}),
}

# Requirements are conservative GUI checks. They do not prove a feature is valid;
# they tell the analyst whether the current selection contains the landmarks
# needed to audit the major downstream families.
SELECTION_REQUIREMENTS: dict[str, dict[str, object]] = {
    "normalization_intercanthal": {
        "label": "Intercanthal normalization anchors",
        "required_any": ((133, 362),),
        "recommended_any": ((33, 263),),
        "reason": "Default normalization scales by inner eye/canthus distance; outer eye anchors are useful fallback/audit points.",
    },
    "mouth_aperture": {
        "label": "Mouth aperture",
        "required_all": (13, 14),
        "reason": "Upper/lower inner lip points are the minimal aperture pair.",
    },
    "lip_spread": {
        "label": "Lip spread",
        "required_all": (61, 291),
        "reason": "Left/right oral commissures define horizontal lip spread.",
    },
    "inner_lip_spread": {
        "label": "Inner lip spread",
        "required_all": (78, 308),
        "reason": "Inner oral-corner landmarks support secondary spread and aperture checks.",
    },
    "jaw_lower_face": {
        "label": "Jaw/lower-face tracking",
        "required_any": ((17,), (152,), (199,)),
        "reason": "At least one lower-face or chin point is needed to audit jaw/lower-face motion.",
    },
    "midline_reference": {
        "label": "Midline reference",
        "required_any": ((0,), (1,), (8,), (10,)),
        "reason": "A stable midline point helps interpret symmetry and lateralization.",
    },
    "bilateral_oral_symmetry": {
        "label": "Bilateral oral symmetry",
        "required_all": (61, 291),
        "recommended_any": ((0,), (1,), (152,)),
        "reason": "Symmetry requires paired commissures and benefits from a midline/lower-face reference.",
    },
}


@dataclass(frozen=True)
class SelectionRequirementResult:
    requirement_id: str
    label: str
    status: str
    missing_required: tuple[int, ...]
    missing_recommended: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class SelectionDiagnostics:
    selected_landmarks: tuple[int, ...]
    n_selected: int
    region_counts: dict[str, int]
    requirement_results: tuple[SelectionRequirementResult, ...]
    supported_families: tuple[str, ...]
    warning_flags: tuple[str, ...]
    status: str
    next_step: str


def region_for_landmark(idx: int) -> str:
    for region, members in LANDMARK_REGION_SETS.items():
        if int(idx) in members:
            return region
    return "other"


def _dedupe_ints(values: Iterable[int]) -> tuple[int, ...]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        ivalue = int(value)
        if ivalue < 0:
            raise ValueError("Landmark indices must be non-negative.")
        if ivalue not in seen:
            seen.add(ivalue)
            out.append(ivalue)
    return tuple(out)


def _requirement_result(requirement_id: str, selected: set[int]) -> SelectionRequirementResult:
    spec = SELECTION_REQUIREMENTS[requirement_id]
    missing_required: set[int] = set()
    status = "PASS"

    required_all = tuple(int(v) for v in spec.get("required_all", tuple()))
    if required_all:
        missing_required.update(v for v in required_all if v not in selected)

    required_any = tuple(tuple(int(v) for v in group) for group in spec.get("required_any", tuple()))
    if required_any and not any(all(v in selected for v in group) for group in required_any):
        # Show the first preferred set as the missing target so the GUI can give
        # a concise instruction.
        missing_required.update(v for v in required_any[0] if v not in selected)

    recommended_any = tuple(tuple(int(v) for v in group) for group in spec.get("recommended_any", tuple()))
    missing_recommended: set[int] = set()
    if recommended_any and not any(all(v in selected for v in group) for group in recommended_any):
        missing_recommended.update(v for v in recommended_any[0] if v not in selected)

    if missing_required:
        status = "FAIL"
    elif missing_recommended:
        status = "REVIEW"

    return SelectionRequirementResult(
        requirement_id=requirement_id,
        label=str(spec["label"]),
        status=status,
        missing_required=tuple(sorted(missing_required)),
        missing_recommended=tuple(sorted(missing_recommended)),
        reason=str(spec["reason"]),
    )


def analyze_landmark_selection(indices: Iterable[int]) -> SelectionDiagnostics:
    selected_landmarks = _dedupe_ints(indices)
    selected = set(selected_landmarks)

    region_counts: dict[str, int] = {}
    for idx in selected_landmarks:
        region = region_for_landmark(idx)
        region_counts[region] = region_counts.get(region, 0) + 1

    requirement_results = tuple(_requirement_result(key, selected) for key in SELECTION_REQUIREMENTS)
    supported = tuple(result.label for result in requirement_results if result.status in {"PASS", "REVIEW"})
    failed = tuple(result for result in requirement_results if result.status == "FAIL")
    review = tuple(result for result in requirement_results if result.status == "REVIEW")

    warning_flags: list[str] = []
    if len(selected_landmarks) == 0:
        warning_flags.append("empty_selection")
    if len(selected_landmarks) < 4:
        warning_flags.append("very_small_selection")
    if failed:
        warning_flags.append("missing_required_landmarks")
    if review:
        warning_flags.append("missing_recommended_landmarks")
    if region_counts.get("mouth/lips", 0) == 0:
        warning_flags.append("no_mouth_landmarks")
    if region_counts.get("eye/canthus anchors", 0) == 0:
        warning_flags.append("no_eye_anchor_landmarks")

    if len(selected_landmarks) == 0:
        status = "Not configured"
        next_step = "Select a preset or click landmarks on the real-frame overlay."
    elif failed:
        status = "Review"
        first = failed[0]
        missing = ", ".join(map(str, first.missing_required))
        next_step = f"Add missing required landmarks for {first.label}: {missing}."
    elif review:
        status = "Review"
        first = review[0]
        missing = ", ".join(map(str, first.missing_recommended))
        next_step = f"Selection is usable, but add recommended audit landmarks for {first.label}: {missing}."
    else:
        status = "Complete"
        next_step = "Save selection, then proceed to Normalization."

    return SelectionDiagnostics(
        selected_landmarks=selected_landmarks,
        n_selected=len(selected_landmarks),
        region_counts=dict(sorted(region_counts.items())),
        requirement_results=requirement_results,
        supported_families=supported,
        warning_flags=tuple(warning_flags),
        status=status,
        next_step=next_step,
    )


def diagnostics_to_payload(diagnostics: SelectionDiagnostics) -> dict[str, object]:
    return {
        "selected_landmarks": list(diagnostics.selected_landmarks),
        "n_selected": diagnostics.n_selected,
        "region_counts": diagnostics.region_counts,
        "supported_families": list(diagnostics.supported_families),
        "warning_flags": list(diagnostics.warning_flags),
        "selection_status": diagnostics.status,
        "next_step": diagnostics.next_step,
        "requirement_results": [asdict(result) for result in diagnostics.requirement_results],
    }


def write_selected_landmarks(
    output_root: Path | str,
    selected_landmarks: Iterable[int],
    *,
    preset: str,
    mesh_source: str = "not_available",
    preview_png: str | None = None,
    app_version: str | None = None,
) -> dict[str, Path]:
    """Write selected landmarks plus auditable selection diagnostics."""
    diagnostics = analyze_landmark_selection(selected_landmarks)
    out = Path(output_root).expanduser().resolve() / "kinematics" / "003_selection"
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    selected_json = tables / "selected_landmarks.json"
    requirements_csv = tables / "selected_landmark_requirements.csv"
    summary_csv = tables / "selected_landmark_summary.csv"

    payload = diagnostics_to_payload(diagnostics)
    payload.update({
        "preset": preset,
        "mesh_source": mesh_source,
        "preview_png": preview_png,
        "app_version": app_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "Selected landmarks define the default subset for normalization, QC, and kinematic feature computation. Full extracted MediaPipe landmarks remain available for audit.",
    })
    selected_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    req_rows = []
    for result in diagnostics.requirement_results:
        req_rows.append({
            "requirement_id": result.requirement_id,
            "label": result.label,
            "status": result.status,
            "missing_required": ", ".join(map(str, result.missing_required)),
            "missing_recommended": ", ".join(map(str, result.missing_recommended)),
            "reason": result.reason,
        })
    pd.DataFrame(req_rows).to_csv(requirements_csv, index=False)

    summary = {
        "n_selected": diagnostics.n_selected,
        "selection_status": diagnostics.status,
        "warning_flags": ";".join(diagnostics.warning_flags),
        "supported_families": ";".join(diagnostics.supported_families),
        "next_step": diagnostics.next_step,
        **{f"region_{k.replace('/', '_').replace(' ', '_')}": v for k, v in diagnostics.region_counts.items()},
    }
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    return {
        "selected_json": selected_json,
        "requirements_csv": requirements_csv,
        "summary_csv": summary_csv,
    }
