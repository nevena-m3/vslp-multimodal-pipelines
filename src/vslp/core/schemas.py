"""Shared typed schemas for stage-based VSLP pipelines.

These schemas are intentionally small and stable. Pipeline stages should exchange
plain files plus JSON/YAML manifests, not hidden Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
import json

StageStatus = Literal["pending", "running", "completed", "completed_with_warnings", "failed", "skipped"]


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to a produced file with optional semantic role."""

    path: str
    role: str
    media_type: str | None = None
    sha256: str | None = None


@dataclass
class StageManifest:
    """Audit record for a single pipeline stage."""

    stage_name: str
    stage_version: str
    status: StageStatus
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    input_artifacts: list[ArtifactRef] = field(default_factory=list)
    output_artifacts: list[ArtifactRef] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path


@dataclass(frozen=True)
class StageResult:
    """Return value used by backend services and CLI commands."""

    status: StageStatus
    manifest_path: Path
    summary_table: Path | None = None
    error_table: Path | None = None
    report_path: Path | None = None
