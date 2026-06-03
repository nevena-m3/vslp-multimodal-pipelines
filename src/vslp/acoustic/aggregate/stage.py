"""Aggregation-stage scaffold."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def aggregate_features(input_csv: str | Path, output_csv: str | Path, group_columns: list[str], numeric_policy: str = "mean") -> Path:
    """Aggregate feature rows with explicit missing-value preservation.

    Current v1 scaffold supports mean aggregation. Advanced algorithms will be added here.
    """
    df = pd.read_csv(input_csv)
    numeric = df.select_dtypes(include="number").columns.tolist()
    if numeric_policy != "mean":
        raise NotImplementedError(f"Aggregation policy not implemented yet: {numeric_policy}")
    out = df.groupby(group_columns, dropna=False)[numeric].mean().reset_index()
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return output_csv
