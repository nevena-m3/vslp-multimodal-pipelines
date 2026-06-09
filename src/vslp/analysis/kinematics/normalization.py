"""Normalization configuration writer for kinematic features."""
from __future__ import annotations
from pathlib import Path
import json
from .schemas import NORMALIZATION_METHODS


def write_normalization_config(output_root: Path, method: str, *, notes: str = "") -> Path:
    if method not in NORMALIZATION_METHODS:
        raise KeyError(method)
    out = Path(output_root) / "kinematics" / "003_normalization" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "normalization_config.json"
    path.write_text(json.dumps({"method": method, "interpretation": NORMALIZATION_METHODS[method], "notes": notes}, indent=2), encoding="utf-8")
    return path
