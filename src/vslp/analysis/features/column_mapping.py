from __future__ import annotations

import re
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .schemas import ColumnRole

ID_PATTERNS = [
    r"^file(_?name)?$", r"record(_?key)?", r"subject", r"participant", r"patient",
    r"session", r"visit", r"iteration", r"task", r"recording", r"date", r"protocol",
]
TARGET_PATTERNS = [r"diagnosis", r"dx", r"severity", r"alsfrs", r"alsbdi", r"label", r"target", r"class", r"score", r"bin"]
TIME_PATTERNS = [r"date", r"time", r"visit", r"iteration", r"session"]
TASK_PATTERNS = [r"task", r"prompt", r"passage", r"ddk"]
QC_PATTERNS = [r"qc", r"quality", r"snr", r"clip", r"noise", r"reverb", r"echo", r"dropout", r"interference", r"warning", r"artifact"]
IGNORE_PATTERNS = [r"path", r"directory", r"folder", r"manifest", r"hash", r"status", r"error", r"message"]


def _matches(name: str, patterns: Iterable[str]) -> bool:
    s = name.lower().strip()
    return any(re.search(p, s) for p in patterns)


def is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series, errors="coerce")
    non_missing = series.notna().sum()
    if non_missing == 0:
        return False
    return converted.notna().sum() / max(non_missing, 1) >= 0.80


def infer_column_roles(df: pd.DataFrame, source_hint: str = "features") -> pd.DataFrame:
    """Infer column roles for a feature/QC/metadata table.

    Conservative default: numeric unknown columns in feature tables become FEATURE;
    numeric unknown columns in QC tables become QC_FEATURE; known IDs/covariates are protected.
    """
    rows: List[Dict[str, object]] = []
    hint = (source_hint or "features").lower()
    for col in df.columns:
        s = df[col]
        name = str(col)
        numeric = is_numeric_like(s)
        lower = name.lower().strip()
        role: ColumnRole
        reason = ""
        if _matches(lower, TASK_PATTERNS):
            role = ColumnRole.TASK
            reason = "task-like column name"
        elif _matches(lower, TIME_PATTERNS) and not numeric:
            role = ColumnRole.TIME
            reason = "time/session/date-like column name"
        elif _matches(lower, ID_PATTERNS) and not (numeric and lower in {"severity", "score"}):
            role = ColumnRole.IDENTIFIER
            reason = "identifier-like column name"
        elif _matches(lower, TARGET_PATTERNS):
            role = ColumnRole.TARGET
            reason = "target/label-like column name"
        elif _matches(lower, QC_PATTERNS):
            role = ColumnRole.QC_FEATURE if numeric else ColumnRole.COVARIATE
            reason = "QC/artifact-like column name"
        elif _matches(lower, IGNORE_PATTERNS):
            role = ColumnRole.IGNORE
            reason = "path/status/error-like column name"
        elif numeric:
            role = ColumnRole.QC_FEATURE if "qc" in hint or "quality" in hint else ColumnRole.FEATURE
            reason = "numeric analysis column"
        else:
            role = ColumnRole.COVARIATE
            reason = "non-numeric descriptive column"
        rows.append({
            "column": name,
            "role": role.value,
            "dtype": str(s.dtype),
            "numeric_like": bool(numeric),
            "missing_fraction": float(s.isna().mean()),
            "unique_values": int(s.nunique(dropna=True)),
            "reason": reason,
        })
    return pd.DataFrame(rows)


def columns_by_role(mapping: pd.DataFrame, role: ColumnRole | str) -> List[str]:
    r = role.value if isinstance(role, ColumnRole) else str(role)
    if mapping.empty:
        return []
    return mapping.loc[mapping["role"].eq(r), "column"].astype(str).tolist()


def choose_join_key(left: pd.DataFrame, right: pd.DataFrame) -> str | None:
    preferred = ["record_key", "file_name", "filename", "source_file", "subject_id", "session_id"]
    left_cols = {c.lower(): c for c in left.columns}
    right_cols = {c.lower(): c for c in right.columns}
    for key in preferred:
        if key in left_cols and key in right_cols:
            return left_cols[key]
    common = [c for c in left.columns if c in right.columns]
    return common[0] if common else None
