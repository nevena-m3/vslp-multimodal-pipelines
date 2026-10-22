"""F01 task identity and recommendation lookup; no linguistic Alignment is loaded."""

from __future__ import annotations

import json
import re
from pathlib import Path


_TASK_REGISTRY = Path(__file__).resolve().parents[1] / "alignment" / "task_prompts.json"
_TASK_TYPES = {
    "Sustained phonation": "sustained_vowel",
    "Passage / connected speech": "bamboo_passage",
    "Sentence / controlled speech": "sentence",
    "DDK": "ddk",
}


def task_context(manifest: dict) -> dict[str, str]:
    """Resolve explicit Setup/registry identity, never a WAV name or acoustic guess."""
    task_id = str(manifest.get("task_id") or "").strip().casefold()
    task_name = str(manifest.get("task_name") or "").strip()
    explicit_family = str(manifest.get("task_family") or "")
    if (task_id == "bamboo" and task_name.casefold() == "bamboo" and
            manifest.get("task_type") == "Passage / connected speech"):
        task_id = "bamboo_passage"
        if explicit_family == "other":
            explicit_family = ""
    family = explicit_family or _TASK_TYPES.get(manifest.get("task_type"), "")
    item = str(manifest.get("task_item") or manifest.get("vowel_stimulus") or "").strip()
    registry = json.loads(_TASK_REGISTRY.read_text(encoding="utf-8"))
    entry = next((task for task in registry["tasks"] if task["task_id"] == task_id), None)
    if not family and entry:
        family = (_TASK_TYPES.get(entry.get("task_type"), "") or
                  ("sustained_vowel" if entry.get("vowel_stimulus") else ""))
    if family == "bamboo_passage" and not explicit_family and task_id != "bamboo_passage":
        family = "other"
    if family == "sustained_vowel" and not item:
        item = str((entry or {}).get("vowel_stimulus") or "")
        if not item:
            match = re.fullmatch(r"Sustained /([^/]+)/", task_name, flags=re.IGNORECASE)
            item = match.group(1) if match else ""
    if family == "sentence" and not item:
        if task_id in {"s3", "sentence_s3"} or task_name.casefold() == "s3":
            item = "S3"
        elif task_id in {"s5", "sentence_s5"} or task_name.casefold() == "s5":
            item = "S5"
        elif task_id in {"s6", "sentence_s6"} or task_name.casefold() == "s6":
            item = "S6"
        elif entry and len(entry.get("prompts", [])) == 1:
            item = str(entry["prompts"][0]["prompt_id"])
        else:
            item = task_id or task_name
    if family == "ddk" and not item:
        item = task_name.casefold() if task_name.casefold() in {"pa", "ta", "ka", "pataka"} else ""
    if family == "bamboo_passage" and not item:
        item = "bamboo"
    stimulus = ""
    if family == "sentence" and item.casefold() in {"s3", "s5", "s6"}:
        named = next((task for task in registry["tasks"]
                      if task["task_id"] == f"sentence_{item.casefold()}"), None)
        if named and named.get("prompts"):
            stimulus = str(named["prompts"][0]["exact_expected_text"])
    return {"task_family": family or "other", "task_item": item,
            "task_id": task_id, "task_name": task_name, "stimulus_text": stimulus}


def recommendation_key(context: dict[str, str]) -> str:
    """Keep registered S5/S6 separate from other sentences."""
    family = context["task_family"]
    if family == "sentence":
        return ("sentence:S5" if context["task_item"].casefold() == "s5" else
                "sentence:S6" if context["task_item"].casefold() == "s6" else "sentence:other")
    return family
