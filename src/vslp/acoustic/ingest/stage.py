"""Acoustic ingest stage: discover files, run ffprobe, write audit outputs."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from vslp.acoustic.context import run_context, cleanup_stage
from vslp.core.io import discover_files_with_duplicate_report
from vslp.core.project import ensure_stage_folders
from vslp.core.provenance import python_environment, sha256_file, tool_versions
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult
from vslp.acoustic.ingest.ffprobe import digest_media_file

DEFAULT_AUDIO_EXTENSIONS = [".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac", ".aac", ".aiff", ".webm", ".mov"]


@cleanup_stage
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
    stage_dir = Path(output_root) / "acoustic" / "000_ingest"
    folders = ensure_stage_folders(stage_dir, lazy=True)
    context = run_context(output_root)

    rows: list[dict] = []
    errors: list[dict] = []
    files, skipped_duplicates = discover_files_with_duplicate_report(input_path, extensions)

    for path in files:
        try:
            digest = digest_media_file(path, ffprobe_bin=ffprobe_bin)
            digest["sha256"] = sha256_file(path)
            digest["source_sha256"] = digest["sha256"]
            digest["recording_id"] = digest["sha256"]
            digest["source_file_path"] = str(path)
            digest.update(context)
            digest["ingest_status"] = "ok"
            rows.append(digest)
        except Exception as exc:  # noqa: BLE001 - batch should continue
            errors.append({"file_name": path.name, "file_path": str(path), "ingest_status": "failed", "error": str(exc)})

    summary_path = folders["tables"] / "audio_ingest_summary.csv"
    errors_path = folders["errors"] / "audio_ingest_errors.csv"
    duplicates_path = folders["tables"] / "audio_ingest_skipped_duplicates.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    pd.DataFrame(errors).to_csv(errors_path, index=False)
    pd.DataFrame(skipped_duplicates).to_csv(duplicates_path, index=False)

    manifest = StageManifest(
        stage_name="acoustic_ingest",
        stage_version="0.1.0",
        status="completed_with_warnings" if errors or skipped_duplicates else "completed",
        output_artifacts=[
            ArtifactRef(path=str(summary_path), role="ingest_summary", media_type="text/csv"),
            ArtifactRef(path=str(errors_path), role="ingest_errors", media_type="text/csv"),
            ArtifactRef(path=str(duplicates_path), role="ingest_skipped_duplicates", media_type="text/csv"),
        ],
        config={
            "input_path": str(input_path),
            "extensions": extensions,
            "ffprobe_bin": ffprobe_bin,
            "duplicate_policy": ".wav > .mp4 > .webm > other supported extensions for identical filename stems",
            "n_skipped_duplicates": len(skipped_duplicates),
        },
        environment={"python": python_environment(), "tools": tool_versions(ffprobe=ffprobe_bin)},
        warnings=([f"{len(errors)} files failed ingest"] if errors else []) + ([f"{len(skipped_duplicates)} duplicate same-stem files skipped"] if skipped_duplicates else []),
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
