"""Shared file-discovery and CSV metadata helpers."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def discover_files(input_path: str | Path, extensions: list[str]) -> list[Path]:
    """Discover supported files from a file, directory, or glob-like folder input."""
    input_path = Path(input_path).expanduser()
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in exts else []
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    return []


def load_demographics_csv(path: str | Path) -> pd.DataFrame:
    """Load demographics/metadata CSV as strings where possible to preserve IDs."""
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df


def validate_metadata_columns(df: pd.DataFrame, required: list[str]) -> tuple[bool, list[str]]:
    missing = [c for c in required if c not in df.columns]
    return len(missing) == 0, missing
