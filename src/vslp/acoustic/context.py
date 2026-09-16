"""Operational identity shared by acoustic stages."""

from __future__ import annotations

import json
from pathlib import Path


def run_context(output_root: str | Path) -> dict[str, str]:
    manifest = json.loads((Path(output_root) / "project_manifest.json").read_text(encoding="utf-8"))
    required = ("project_name", "task_name", "run_id", "created_at_local", "created_at_utc")
    if any(not manifest.get(key) for key in required):
        raise ValueError("Acoustic stage requires initialized Setup provenance")
    return {
        "project_name": manifest["project_name"],
        "task_name": manifest["task_name"],
        "run_id": manifest["run_id"],
        "run_created_at_local": manifest["created_at_local"],
        "run_created_at_utc": manifest["created_at_utc"],
    }


def cleanup_stage(fn):
    """Remove empty acoustic stage directories after success or failure."""
    from functools import wraps
    from inspect import signature
    from vslp.core.project import prune_empty_acoustic_directories

    @wraps(fn)
    def wrapped(*args, **kwargs):
        bound = signature(fn).bind_partial(*args, **kwargs)
        output_root = bound.arguments.get("output_root")
        try:
            return fn(*args, **kwargs)
        finally:
            if output_root is not None:
                prune_empty_acoustic_directories(output_root)

    return wrapped
