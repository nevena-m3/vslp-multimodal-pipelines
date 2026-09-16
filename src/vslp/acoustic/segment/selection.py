"""Task-aware method recommendations without study or clinical metadata."""

from __future__ import annotations

import re

SILERO = "silero_vad"
DDK = "ddk_energy"
PHONATION = "sustained_phonation"
CUSTOM = "custom_plugin"


def recommend_method(task_name: str) -> str:
    name = re.sub(r"[^a-z0-9]+", " ", task_name.casefold()).strip()
    words = set(name.split())
    if {"ddk", "amr", "smr", "pataka", "pa", "ta", "ka", "ba"} & words:
        return DDK
    if "sustained" in words or "phonation" in words or "vowel" in words:
        return PHONATION
    if name in {"a", "i", "ah", "sustained a", "sustained i"}:
        return PHONATION
    return SILERO
