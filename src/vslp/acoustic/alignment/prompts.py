"""Versioned task prompt registry (no aligner dependencies)."""

from __future__ import annotations

import json
from pathlib import Path


REGISTRY_PATH = Path(__file__).with_name("task_prompts.json")


def task_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def task_entry(task_id: str) -> dict | None:
    return next((task for task in task_registry()["tasks"]
                 if task["task_id"] == str(task_id).casefold()), None)


def _manifest_task(manifest: dict) -> dict | None:
    """Resolve only explicit Setup metadata, never recording names or audio."""
    registered = task_entry(str(manifest.get("task_id", "")))
    if registered:
        return registered
    # An older Setup version stored the explicit Bamboo label as its task ID.
    if str(manifest.get("task_id", "")).strip().casefold() == "bamboo":
        return task_entry("bamboo_passage")
    names = [str(manifest.get(field, "")).strip().casefold()
             for field in ("task_display_name", "task_name")]
    matches = [task for task in task_registry()["tasks"]
               if task["display_name"].casefold() in names]
    return matches[0] if len(matches) == 1 else None
