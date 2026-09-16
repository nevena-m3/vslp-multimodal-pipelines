"""Project directory creation and stage folder management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any
from uuid import uuid4


STAGE_SUBDIRS = ["tables", "plots", "logs", "reports", "artifacts", "errors"]
WORKSPACE_DIRS = {
    "configs": "configs",
    "acoustic": "acoustic",
    "kinematics": "kinematics",
    "feature_analysis": "feature_analysis",
    "ml": "ml",
    "logs": "logs",
}
WORKSPACE_SCHEMA = "vslp_workspace"
WORKSPACE_SCHEMA_VERSION = "1.0.0"
ACOUSTIC_ONLY_DIRS = {key: WORKSPACE_DIRS[key] for key in ("configs", "acoustic", "logs")}


def task_run_folder_name(task_name: str, when: datetime | None = None) -> str:
    """Return a filesystem-safe task name followed by a local timestamp."""
    task = re.sub(r"[^\w.-]+", "_", task_name.strip(), flags=re.UNICODE).strip("._-")
    if not task or not any(char.isalnum() for char in task):
        raise ValueError("Enter a task name containing letters or numbers.")
    task = task[:80].rstrip("._-")
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{task}_{stamp}"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "project_manifest.json"

    def component_dir(self, component: str) -> Path:
        if component not in WORKSPACE_DIRS:
            raise ValueError(f"Unknown VSLP workspace component: {component}")
        return self.root / WORKSPACE_DIRS[component]

    def stage_dir(self, modality: str, stage_order: int, stage_name: str) -> Path:
        safe = stage_name.lower().replace(" ", "_")
        return self.root / modality / f"{stage_order:03d}_{safe}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Existing VSLP workspace manifest is not readable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Existing VSLP workspace manifest must contain a JSON object: {path}")
    return payload


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        for attempt in range(6):
            try:
                temporary.replace(path)
                return
            except OSError as exc:
                retryable = isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in {5, 32, 33}
                if not retryable:
                    raise
                if attempt == 5:
                    raise PermissionError(
                        f"Could not update the project manifest after 6 attempts: {path}. "
                        "Close other VSLP windows or file previews using this project."
                    ) from exc
                time.sleep(0.15 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def initialize_project(
    output_root: str | Path,
    project_name: str = "VSLP Study",
    *,
    layout: dict[str, str] | None = None,
) -> ProjectPaths:
    """Create or safely reopen a shared VSLP study workspace.

    All desktop applications receive the same workspace root. Each application
    owns a component directory below that root and may register component
    metadata without replacing records written by another application.
    """
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    active_layout = layout or WORKSPACE_DIRS
    for top in active_layout.values():
        (root / top).mkdir(exist_ok=True)

    manifest_path = root / "project_manifest.json"
    existing = _read_manifest(manifest_path)
    now = _utc_now()
    manifest = {
        **existing,
        "schema": WORKSPACE_SCHEMA,
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "project_name": existing.get("project_name") or project_name or "VSLP Study",
        "created_at_utc": existing.get("created_at_utc") or now,
        "updated_at_utc": now,
        "mode": existing.get("mode") or "research_use_only",
        "privacy": existing.get("privacy") or "local_only",
        "layout": {**(existing.get("layout") or {}), **active_layout},
        "components": existing.get("components") if isinstance(existing.get("components"), dict) else {},
    }
    _write_manifest(manifest_path, manifest)
    return ProjectPaths(root=root)


def register_project_component(
    output_root: str | Path,
    component: str,
    details: dict[str, Any] | None = None,
    *,
    project_name: str = "VSLP Study",
    layout: dict[str, str] | None = None,
) -> ProjectPaths:
    """Register one GUI/component in the shared workspace manifest."""
    paths = initialize_project(output_root, project_name=project_name, layout=layout)
    if component not in WORKSPACE_DIRS:
        raise ValueError(f"Unknown VSLP workspace component: {component}")
    manifest = _read_manifest(paths.manifest)
    components = dict(manifest.get("components") or {})
    previous = components.get(component) if isinstance(components.get(component), dict) else {}
    now = _utc_now()
    components[component] = {
        **previous,
        **(details or {}),
        "directory": WORKSPACE_DIRS[component],
        "initialized_at_utc": previous.get("initialized_at_utc") or now,
        "updated_at_utc": now,
    }
    manifest["components"] = components
    manifest["updated_at_utc"] = now
    _write_manifest(paths.manifest, manifest)
    return paths


def prune_empty_acoustic_directories(output_root: str | Path) -> None:
    """Remove only empty directories within this run's acoustic component."""
    component = Path(output_root).expanduser().resolve() / WORKSPACE_DIRS["acoustic"]
    if not component.is_dir() or component.is_symlink():
        return
    directories = [p for p in component.rglob("*") if p.is_dir() and not p.is_symlink()]
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


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
