"""Project directory creation and stage folder management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime, timezone


STAGE_SUBDIRS = ["tables", "plots", "logs", "reports", "artifacts", "errors"]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "project_manifest.json"

    def stage_dir(self, modality: str, stage_order: int, stage_name: str) -> Path:
        safe = stage_name.lower().replace(" ", "_")
        return self.root / modality / f"{stage_order:03d}_{safe}"


def initialize_project(output_root: str | Path, project_name: str = "vslp_project") -> ProjectPaths:
    """Create a reproducible VSLP project folder."""
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    for top in ["configs", "acoustic", "kinematic", "feature_analysis", "ml", "logs"]:
        (root / top).mkdir(exist_ok=True)

    manifest = {
        "project_name": project_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "research_use_only",
        "privacy": "local_only",
        "schema_version": "0.1.0",
    }
    (root / "project_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ProjectPaths(root=root)


def ensure_stage_folders(stage_dir: str | Path) -> dict[str, Path]:
    """Create standard subdirectories for a stage and return them."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for sub in STAGE_SUBDIRS:
        p = stage_dir / sub
        p.mkdir(parents=True, exist_ok=True)
        out[sub] = p
    return out
