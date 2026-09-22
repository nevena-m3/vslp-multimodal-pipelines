"""Source-bound registry for the 79 master-matrix constructs.

The matrix has no exact output IDs. Family specifications may add leaf outputs;
until then no feature algorithm is approved for production execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DATA = Path(__file__).with_name("data") / "master_matrix.json"
_FAMILY08 = Path(__file__).with_name("data") / "family08.json"
_FAMILY09 = Path(__file__).with_name("data") / "family09.json"
USE = {"PROD": "Production", "VARIANT": "Conditional", "TASK": "Conditional",
       "VALIDATE": "Validation required", "RESEARCH": "Research", "BLOCKED": "Blocked"}
TASKS = {
    "sustained_a": "Sustained /a/",
    "bamboo_passage": "Bamboo Passage",
    "ddk": "DDK",
    "wstg": "WSTG",
}


def validate_feature_ids(outputs: list[dict[str, Any]]) -> None:
    ids = [str(item["feature_id"]) for item in outputs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate feature IDs in acoustic catalog")
    if any(not value or value == "—" for value in ids):
        raise ValueError("Output features require exact code-facing IDs")


def load_feature_catalog() -> dict[str, Any]:
    matrix = json.loads(_DATA.read_text(encoding="utf-8"))
    if len(matrix["constructs"]) != 79 or len(matrix["families"]) != 13:
        raise ValueError("Master matrix must contain 79 constructs in 13 families")
    for construct in matrix["constructs"]:
        construct["recommended_tasks"] = [key for key, mark in construct["tasks"].items() if mark == "✓"]
        construct["conditional_tasks"] = [key for key, mark in construct["tasks"].items() if mark == "C"]
        construct["use_status"] = USE[construct["matrix_status"]]
        construct["evidence_level"] = None
        construct["evidence_study_count"] = None
        construct["evidence_entry_count"] = None
        construct["outputs"] = []
    # Master matrix has no FEATURE ID(S); the supplied Family 08 document does.
    matrix["outputs"] = []
    matrix["tasks_registry"] = TASKS.copy()
    family08 = json.loads(_FAMILY08.read_text(encoding="utf-8"))
    constructs = {item["construct_id"]: item for item in matrix["constructs"]}
    for construct_id in ("C048", "C049"):
        # Detailed family task applicability supersedes the broader matrix cells.
        constructs[construct_id]["recommended_tasks"] = ["bamboo_passage"]
        constructs[construct_id]["conditional_tasks"] = []
    outputs = []
    for spec in family08["outputs"]:
        construct = constructs[spec["construct_id"]]
        outputs.append({
            **spec,
            "construct_name": construct["construct_name"],
            "family_id": "F08",
            "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"],
            "source_use_status": construct["matrix_status"],
            "use_status": "Production",
            "recommended_tasks": ["bamboo_passage"],
            "conditional_tasks": [],
            "qc_range_low": 0,
            "qc_range_high": 10,
            "qc_range_text": "0–10 syllables/s" if spec["unit"] == "syllables/s" else "≥0 words/min",
            "analysis_unit": "Whole utterance/prompt",
            "signal_scope": construct["signal_scope"],
            "analysis_region": "first to last final patient-speech boundary",
            "estimator": "versioned prompt count / reviewed task timing",
            "algorithm": family08["algorithm_version"],
            "algorithm_version": family08["algorithm_version"],
            "default_parameter_set_id": family08["parameter_set_id"],
            "parameter_profile": {
                "name": "Validated default", "read_only": True,
                "active_parameters": {"pause_min_sec": family08["pause_min_sec"],
                                      "boundary_source": "frozen reviewed segmentation"}},
            "prerequisites": {
                "requires_audio": False,
                "requires_segmentation": True,
                "requires_final_reviewed_segmentation": True,
                "requires_alignment": False,
                "requires_task": True,
                "requires_unscaled_audio": False,
                "requires_voiced_region": False,
                "requires_ddk_events": False,
                "requires_vowel_tokens": False,
                "requires_prompt_manifest": True},
            "family_spec_approved": True,
            "selectable": True,
            "source_document": family08["source_document"],
        })
    attach_family_outputs(matrix, "F08", outputs)
    family09 = json.loads(_FAMILY09.read_text(encoding="utf-8"))
    for construct_id in (f"C{number:03d}" for number in range(50, 58)):
        constructs[construct_id]["recommended_tasks"] = ["bamboo_passage"]
        constructs[construct_id]["conditional_tasks"] = []
    outputs09 = []
    for spec in family09["outputs"]:
        construct = constructs[spec["construct_id"]]
        research = spec.get("research_only", False)
        outputs09.append({
            **spec,
            "construct_name": construct["construct_name"],
            "family_id": "F09",
            "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"],
            "source_use_status": construct["matrix_status"],
            "use_status": "Research" if research else "Production",
            "recommended_tasks": ["bamboo_passage"],
            "conditional_tasks": [],
            "qc_range_low": None, "qc_range_high": None,
            "signal_scope": construct["signal_scope"],
            "analysis_region": "reviewed patient utterance and eligible internal pause/phrase events",
            "estimator": "shared reviewed pause/phrase event table",
            "algorithm": family09["algorithm_version"],
            "algorithm_version": family09["algorithm_version"],
            "default_parameter_set_id": family09["parameter_set_id"],
            "parameter_profile": {
                "name": "Validated default" if not research else "Research components",
                "read_only": True,
                "active_parameters": {
                    "minimum_internal_pause_ms": family09["minimum_internal_pause_ms"],
                    "sample_sd_ddof": family09["sample_sd_ddof"],
                    "boundary_source": "frozen reviewed segmentation"}},
            "prerequisites": {
                "requires_audio": False, "requires_segmentation": True,
                "requires_final_reviewed_segmentation": True,
                "requires_alignment": False, "requires_task": True,
                "requires_unscaled_audio": False, "requires_voiced_region": False,
                "requires_ddk_events": False, "requires_vowel_tokens": False,
                "requires_prompt_manifest": False},
            "family_spec_approved": True,
            "selectable": spec.get("selectable", True),
            "blocked_reason": spec.get("blocked_reason", ""),
            "source_document": family09["source_document"],
        })
    attach_family_outputs(matrix, "F09", outputs09)
    return matrix


def attach_family_outputs(catalog: dict[str, Any], family_id: str,
                          outputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach exact family-spec leaves without changing construct identities."""
    constructs = {item["construct_id"]: item for item in catalog["constructs"]}
    validate_feature_ids([*catalog["outputs"], *outputs])
    for output in outputs:
        construct = constructs.get(output["construct_id"])
        if construct is None or construct["family_id"] != family_id:
            raise ValueError("Output must reference an existing construct in its family")
        if not output.get("family_spec_approved"):
            raise ValueError("Output requires family-spec approval before registration")
    for output in outputs:
        constructs[output["construct_id"]]["outputs"].append(output)
    catalog["outputs"].extend(outputs)
    return catalog


def task_key(task_name: str) -> str | None:
    """Resolve only registered display names; never infer task semantics."""
    normalized = task_name.strip().casefold()
    return next((key for key, label in TASKS.items() if label.casefold() == normalized), None)


def task_fit(item: dict[str, Any], task_name: str) -> str:
    key = task_key(task_name)
    if key in item["recommended_tasks"]:
        return "Recommended"
    if key in item["conditional_tasks"]:
        return "Conditional"
    return "Not specified"


def task_indicator(item: dict[str, Any], task_name: str, *,
                   prerequisite_problems: list[str] | None = None,
                   has_exact_output: bool = True) -> tuple[str, str]:
    """Return the display state and its source-grounded explanation."""
    task_label = task_name.strip() or "current task"
    if item.get("matrix_status") == "BLOCKED" or item.get("use_status") == "Blocked":
        return "RED", "Unavailable — exact source algorithm is not reproducibly implementable."
    if not has_exact_output or not (item.get("feature_id") or item.get("construct_id")):
        return "RED", "Unavailable — exact leaf feature ID awaits its family specification."
    if not item.get("family_spec_approved", False) or not item.get("selectable", False):
        return "RED", "Unavailable — output algorithm is not approved for execution."
    if prerequisite_problems:
        return "RED", "Unavailable — " + "; ".join(prerequisite_problems)
    fit = task_fit(item, task_name)
    if fit == "Recommended":
        if item.get("conditional_reason"):
            return "AMBER", f"Conditional for {task_label} — {item['conditional_reason']}"
        return "GREEN", f"Recommended for {task_label} by the feature specification."
    if fit == "Conditional":
        reason = item.get("conditional_reason")
        suffix = f" — {reason}" if reason else "."
        return "AMBER", f"Conditional for {task_label}{suffix}"
    return "GRAY", f"Not specified for {task_label} in the current feature specification."


def prerequisite_issues(output: dict[str, Any], *, task_name: str,
                        final_decisions: Path | None, final_intervals: Path | None,
                        alignment_available: bool = False,
                        prompt_available: bool = False) -> list[str]:
    """Schema reserved for approved exact outputs from future family documents."""
    if not output.get("family_spec_approved", False):
        return ["Exact output algorithm awaits its family specification"]
    requires = output["prerequisites"]
    issues = []
    if requires.get("requires_task") and task_key(task_name) is None:
        issues.append("Task is not in the explicit task registry")
    if requires.get("requires_final_reviewed_segmentation") and not (
            final_decisions and final_decisions.is_file() and final_intervals and final_intervals.is_file()):
        issues.append("Final reviewed segmentation is unavailable")
    if requires.get("requires_alignment") and not alignment_available:
        issues.append("Required vowel/phone alignment is unavailable")
    if requires.get("requires_prompt_manifest") and not prompt_available:
        issues.append("Prompt numerator is unavailable")
    return issues
