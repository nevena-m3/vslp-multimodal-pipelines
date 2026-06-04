"""Shared file-discovery and CSV metadata helpers."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


# If duplicate files have exactly the same stem but different extensions,
# keep the highest-priority candidate. This handles common browser/recording
# exports where the same recording may exist as .wav, .mp4, and .webm.
DEFAULT_EXTENSION_PRIORITY = [
    ".wav",
    ".mp4",
    ".webm",
    ".m4a",
    ".mp3",
    ".flac",
    ".ogg",
    ".aac",
    ".aiff",
    ".mov",
]


def _normalize_extensions(extensions: list[str]) -> set[str]:
    return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}


def _extension_rank(path: Path, priority: list[str]) -> tuple[int, str]:
    suffix = path.suffix.lower()
    try:
        return (priority.index(suffix), str(path).lower())
    except ValueError:
        return (len(priority), str(path).lower())


def discover_files_with_duplicate_report(
    input_path: str | Path,
    extensions: list[str],
    extension_priority: list[str] | None = None,
) -> tuple[list[Path], list[dict[str, str]]]:
    """Discover supported files and remove same-stem extension duplicates.

    Duplicate rule
    --------------
    If two or more supported files have exactly the same filename stem and differ
    only by extension, VSLP keeps one file using this priority by default:

        .wav > .mp4 > .webm > other supported extensions

    The duplicate comparison is intentionally based on the file stem, not on the
    container detected by ffprobe, because this step happens before media digest.
    Skipped duplicates are returned for audit/logging.
    """
    input_path = Path(input_path).expanduser()
    exts = _normalize_extensions(extensions)
    priority = extension_priority or DEFAULT_EXTENSION_PRIORITY

    if input_path.is_file():
        if input_path.suffix.lower() in exts:
            return [input_path], []
        return [], []

    if not input_path.is_dir():
        return [], []

    candidates = sorted(p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    grouped: dict[str, list[Path]] = {}
    for path in candidates:
        grouped.setdefault(path.stem, []).append(path)

    kept: list[Path] = []
    skipped: list[dict[str, str]] = []
    for stem, paths in grouped.items():
        ordered = sorted(paths, key=lambda p: _extension_rank(p, priority))
        keep = ordered[0]
        kept.append(keep)
        for duplicate in ordered[1:]:
            skipped.append(
                {
                    "duplicate_stem": stem,
                    "kept_file_name": keep.name,
                    "kept_file_path": str(keep),
                    "skipped_file_name": duplicate.name,
                    "skipped_file_path": str(duplicate),
                    "reason": "same filename stem; lower-priority extension",
                }
            )

    return sorted(kept), skipped


def discover_files(input_path: str | Path, extensions: list[str]) -> list[Path]:
    """Discover supported files from a file or directory, recursively.

    Same-stem duplicate files with different extensions are deduplicated using
    the VSLP priority policy: .wav, then .mp4, then .webm, then other supported
    media types.
    """
    files, _skipped = discover_files_with_duplicate_report(input_path, extensions)
    return files


def load_demographics_csv(path: str | Path) -> pd.DataFrame:
    """Load demographics/metadata CSV as strings where possible to preserve IDs."""
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df


def validate_metadata_columns(df: pd.DataFrame, required: list[str]) -> tuple[bool, list[str]]:
    missing = [c for c in required if c not in df.columns]
    return len(missing) == 0, missing
