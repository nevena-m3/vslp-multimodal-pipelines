"""GUI-only labels for the existing acoustic registry; no execution policy lives here."""

from __future__ import annotations

from vslp.acoustic.features.catalog import feature_availability


LABELS = {
    "CONTAINER": "Container",
    "IMPLEMENTED": "Implemented",
    "PARTIALLY_IMPLEMENTED": "Partially implemented",
    "UNRESOLVED_DEFINITION": "Definition required",
    "NOT_IMPLEMENTED_PROPRIETARY": "Unavailable — proprietary",
    "NOT_IMPLEMENTED_SOURCE_INCOMPLETE": "Unavailable — source incomplete",
    "FACTOR_NOT_FROZEN": "Factor not frozen",
}


def output_presentation_status(output: dict) -> str:
    """Explain a leaf without changing its scientific availability/selectability."""
    if feature_availability(output)[0] == "Implemented":
        return "IMPLEMENTED"
    feature_id = output["feature_id"]
    if "<" in feature_id or ">" in feature_id:
        return "UNRESOLVED_DEFINITION"
    if "factor_if_frozen" in feature_id or "factor_score_if_frozen" in feature_id:
        return "FACTOR_NOT_FROZEN"
    reason = str(output.get("blocked_reason", "")).casefold()
    if "proprietary" in reason or "licensed source" in reason:
        return "NOT_IMPLEMENTED_PROPRIETARY"
    return "NOT_IMPLEMENTED_SOURCE_INCOMPLETE"


def construct_presentation_status(construct: dict) -> str:
    """Derive a construct label from exact leaves plus unresolved templates."""
    outputs = construct.get("outputs", [])
    statuses = [output_presentation_status(output) for output in outputs]
    templates = (construct.get("source_feature_templates") or
                 construct.get("feature_id_templates") or [])
    has_unresolved = bool(templates) or "UNRESOLVED_DEFINITION" in statuses
    implemented = statuses.count("IMPLEMENTED")
    if implemented:
        return "PARTIALLY_IMPLEMENTED" if implemented < len(statuses) or has_unresolved else "IMPLEMENTED"
    if has_unresolved or (not statuses and construct.get("family_id") != "F03"):
        return "UNRESOLVED_DEFINITION"
    if construct.get("family_id") == "F03" or (
            statuses and all(status == "NOT_IMPLEMENTED_PROPRIETARY" for status in statuses)):
        return "NOT_IMPLEMENTED_PROPRIETARY"
    if statuses and all(status == "FACTOR_NOT_FROZEN" for status in statuses):
        return "FACTOR_NOT_FROZEN"
    return "NOT_IMPLEMENTED_SOURCE_INCOMPLETE"
