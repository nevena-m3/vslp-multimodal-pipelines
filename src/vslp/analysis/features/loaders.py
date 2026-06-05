from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    """Read a tabular file. CSV is primary; TSV is supported by extension.

    This intentionally avoids Excel dependency for the first feature-analysis pass.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {suffix}. Use CSV, TSV, TXT, or Parquet.")


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out
