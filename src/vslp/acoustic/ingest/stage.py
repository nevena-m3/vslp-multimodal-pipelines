"""Acoustic ingest stage: discover files, run ffprobe, write audit outputs."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from vslp.core.io import discover_files
from vslp.core.project import ensure_stage_folders
from vslp.core.provenance import python_environment, sha256_file, tool_versions
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult
from vslp.acoustic.ingest.ffprobe import digest_media_file

DEFAULT_AUDIO_EXTENSIONS = [".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac", ".aac", ".aiff", ".webm", ".mov"]


def run_acoustic_ingest(
    input_path: str | Path,
    output_root: str | Path,
    ffprobe_bin: str = "ffprobe",
    extensions: list[str] | None = None,
) -> StageResult:
    """Run non-destructive audio ingest over a file/folder.

    One bad file is reported in the error table and does not stop the batch.
    """
    extensions = extensions or DEFAULT_AUDIO_EXTENSIONS
    stage_dir = Path(output_root) / "acoustic" / "001_ingest"
    folders = ensure_stage_folders(stage_dir)

    rows: list[dict] = []
    errors: list[dict] = []
    files = discover_files(input_path, extensions)

    for path in files:
        try:
            digest = digest_media_file(path, ffprobe_bin=ffprobe_bin)
            digest["sha256"] = sha256_file(path)
            digest["ingest_status"] = "ok"
            rows.append(digest)
        except Exception as exc:  # noqa: BLE001 - batch should continue
            errors.append({"file_name": path.name, "file_path": str(path), "ingest_status": "failed", "error": str(exc)})

    summary_path = folders["tables"] / "audio_ingest_summary.csv"
    errors_path = folders["errors"] / "audio_ingest_errors.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    pd.DataFrame(errors).to_csv(errors_path, index=False)

    manifest = StageManifest(
        stage_name="acoustic_ingest",
        stage_version="0.1.0",
        status="completed_with_warnings" if errors else "completed",
        output_artifacts=[
            ArtifactRef(path=str(summary_path), role="ingest_summary", media_type="text/csv"),
            ArtifactRef(path=str(errors_path), role="ingest_errors", media_type="text/csv"),
        ],
        config={"input_path": str(input_path), "extensions": extensions, "ffprobe_bin": ffprobe_bin},
        environment={"python": python_environment(), "tools": tool_versions(ffprobe=ffprobe_bin)},
        warnings=[f"{len(errors)} files failed ingest"] if errors else [],
        errors=errors,
    )
    manifest_path = folders["logs"] / "stage_manifest.json"
    manifest.write_json(manifest_path)

    return StageResult(
        status=manifest.status,
        manifest_path=manifest_path,
        summary_table=summary_path,
        error_table=errors_path,
    )
