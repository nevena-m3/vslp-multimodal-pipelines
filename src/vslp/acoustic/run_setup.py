"""Create one immutable, uniquely named acoustic analysis run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from vslp.core.project import task_run_folder_name


@dataclass(frozen=True)
class AcousticRun:
    root: Path
    run_id: str


def initialize_acoustic_run(
    *,
    project_name: str,
    task_name: str,
    input_folder: str | Path,
    output_parent: str | Path,
    setup_values: dict[str, str],
    gui_version: str = "0.38",
    pipeline_version: str = "0.38.0",
    now: datetime | None = None,
) -> AcousticRun:
    """Validate setup, reserve a unique run directory, and write provenance."""
    project = project_name.strip()
    task = task_name.strip()
    if not project:
        raise ValueError("Project name is required.")
    if not task:
        raise ValueError("Task name is required.")
    source = Path(input_folder).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Input folder does not exist: {source}")
    parent = Path(output_parent).expanduser().resolve()
    if parent.is_relative_to(source):
        raise ValueError("Output parent must be outside the input folder.")
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise ValueError(f"Output parent is not a directory: {parent}")
    with tempfile.TemporaryFile(dir=parent):
        pass
    local_time = now if now is not None else datetime.now().astimezone()
    if local_time.tzinfo is None:
        local_time = local_time.astimezone()
    base_id = task_run_folder_name(task, local_time)
    slug = base_id.rsplit("_", 2)[0]
    if parent.is_relative_to(source):
        raise ValueError("Output parent must be outside the input folder.")
    run_root = None
    run_id = ""
    for attempt in range(1, 1000):
        run_id = base_id if attempt == 1 else f"{base_id}_{attempt:02d}"
        candidate = parent / run_id
        if candidate.is_relative_to(source):
            raise ValueError("Run folder must be outside the input folder.")
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        run_root = candidate.resolve()
        break
    if run_root is None:
        raise FileExistsError(f"Could not reserve a unique run folder under {parent}")

    for name in ("configs", "logs", "acoustic"):
        (run_root / name).mkdir()
    utc_time = local_time.astimezone(timezone.utc)
    manifest = {
        "schema": "vslp_acoustic_run",
        "schema_version": "1.0.0",
        "modality": "acoustic",
        "project_name": project,
        "task_name": task,
        "task_slug": slug,
        "run_id": run_id,
        "created_at_local": local_time.isoformat(),
        "created_at_utc": utc_time.isoformat(),
        "input_folder": str(source),
        "output_parent": str(parent),
        "run_root": str(run_root),
        "gui_version": gui_version,
        "pipeline_version": pipeline_version,
        "layout": {"configs": "configs", "logs": "logs", "acoustic": "acoustic"},
        "components": {"acoustic": {"directory": "acoustic", "primary_task": task, "input_folder": str(source), "gui_version": gui_version}},
    }
    (run_root / "project_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (run_root / "configs" / "setup_config.json").write_text(json.dumps(setup_values, indent=2) + "\n", encoding="utf-8")
    (run_root / "logs" / "setup.log").write_text(
        f"Created acoustic run {run_id} at {utc_time.isoformat()} UTC\nProject: {project}\nTask: {task}\nInput: {source}\nOutput parent: {parent}\n",
        encoding="utf-8",
    )
    return AcousticRun(root=run_root, run_id=run_id)
