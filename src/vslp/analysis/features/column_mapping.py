"""Column-role detection for the VSLP Feature Analysis GUI.

Conservative default: in a primary feature table, numeric columns are treated as
features unless there is strong evidence that they are identifiers, QC variables,
audit/status variables, covariates, time variables, or exact clinical labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .schemas import ColumnRole

ROLE_IDENTIFIER = "Identifier"
ROLE_FEATURE = "Feature"
ROLE_QC = "QC feature"
ROLE_TARGET = "Target / label"
ROLE_COVARIATE = "Covariate"
ROLE_TASK = "Task"
ROLE_TIME = "Time / visit"
ROLE_AUDIT = "Audit / status"
ROLE_IGNORE = "Ignore"

ROLE_OPTIONS = [
    ROLE_IDENTIFIER,
    ROLE_FEATURE,
    ROLE_QC,
    ROLE_TARGET,
    ROLE_COVARIATE,
    ROLE_TASK,
    ROLE_TIME,
    ROLE_AUDIT,
    ROLE_IGNORE,
]

_ENUM_TO_DISPLAY = {
    ColumnRole.IDENTIFIER: ROLE_IDENTIFIER,
    ColumnRole.FEATURE: ROLE_FEATURE,
    ColumnRole.QC_FEATURE: ROLE_QC,
    ColumnRole.TARGET: ROLE_TARGET,
    ColumnRole.COVARIATE: ROLE_COVARIATE,
    ColumnRole.TASK: ROLE_TASK,
    ColumnRole.TIME: ROLE_TIME,
    ColumnRole.IGNORE: ROLE_IGNORE,
}
_VALUE_TO_DISPLAY = {r.value: d for r, d in _ENUM_TO_DISPLAY.items()}
_DISPLAY_TO_VALUE = {d: r.value for r, d in _ENUM_TO_DISPLAY.items()}
_DISPLAY_TO_VALUE[ROLE_AUDIT] = ColumnRole.IGNORE.value

_IDENTIFIER_EXACT = {
    "file_name", "filename", "file", "source_file", "source_path", "path",
    "relative_path", "absolute_path", "record_key", "recording_id", "sample_id",
    "subject_id", "participant_id", "patient_id", "session_id", "visit_id",
    "clinical_visit_id", "protocol_id", "iteration", "trial", "project", "project_name",
}
_TASK_EXACT = {"task", "task_name", "prompt", "elicitation", "passage_name"}
_TIME_EXACT = {
    "date", "recording_date", "visit_date", "session_date", "timestamp",
    "timepoint", "time_point", "days_from_baseline", "months_from_baseline",
}
_TARGET_EXACT = {
    "diagnosis", "dx", "group", "class", "label", "target", "outcome",
    "severity", "severity_bin", "severity_class", "severity_score",
    "alsfrs", "alsfrs_total", "alsfrs_total_score", "alsfrs_r",
    "alsfrs_bulbar", "alsfrs_bulbar_subscore", "alsbdi", "alsbdi_total",
    "progression_rate", "progressor", "disease_progression", "sentence_intelligibility_percent",
}
_COVARIATE_EXACT = {
    "age", "sex", "gender", "site", "device", "microphone", "language",
    "dialect", "education", "handedness", "onset_site", "disease_duration",
}
_AUDIT_PATTERNS = [
    r"(^|_)status($|_)", r"(^|_)warning", r"(^|_)flag", r"(^|_)reason",
    r"implementation_status", r"feature_status", r"computed_proxy", r"validity",
    r"manifest", r"hash", r"sha256", r"version", r"path", r"error", r"message",
]
_QC_PATTERNS = [
    r"(^|_)qc($|_)", r"quality", r"snr", r"clip", r"clipping", r"noise",
    r"hum", r"interference", r"reverb", r"echo", r"dropout", r"gain",
    r"distortion", r"powerline", r"saturation", r"artifact", r"contamination",
]
_FEATURE_NAME_PATTERNS = [
    r"^f0", r"^cpp", r"^hnr", r"jitter", r"shimmer", r"voicebreak",
    r"^h1", r"^h2", r"^f[1-5]($|_)", r"formant", r"bandwidth", r"_bw$",
    r"slope", r"d_dx", r"range$", r"a1p0", r"a1p1", r"a3p0", r"p0", r"p1",
    r"rms", r"speech_dur", r"total_dur", r"pause", r"phrase", r"speech_rate",
    r"intensity", r"fft_", r"nrj_", r"ratio_below_above", r"ddk", r"coord", r"comp$",
    r"velocity", r"distance", r"angle", r"area", r"landmark", r"kinematic",
]

@dataclass
class ColumnRoleRecord:
    column: str
    role: str
    confidence: float
    reason: str
    dtype: str
    missing_fraction: float
    unique_values: int
    numeric_like: bool


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, name) for p in patterns)


def _is_numeric_like(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    converted = pd.to_numeric(s, errors="coerce")
    non_missing = int(s.notna().sum())
    return bool(non_missing and converted.notna().sum() / max(non_missing, 1) >= 0.80)


def _is_low_cardinality(s: pd.Series) -> bool:
    n = max(int(s.notna().sum()), 1)
    return s.nunique(dropna=True) <= min(20, max(3, int(0.20 * n)))


def infer_table_kind(path_or_name: str | Path | None = None, explicit: str | None = None) -> str:
    if explicit and explicit.lower() not in {"auto", ""}:
        e = explicit.lower()
        if "qc" in e or "quality" in e:
            return "qc"
        if "meta" in e or "demo" in e:
            return "metadata"
        return "feature"
    text = normalize_name(Path(str(path_or_name)).name if path_or_name else "")
    if "qc" in text or "quality" in text:
        return "qc"
    if "metadata" in text or "demographic" in text or "file_index" in text:
        return "metadata"
    return "feature"


def classify_columns(df: pd.DataFrame, table_kind: str = "feature", registry: pd.DataFrame | None = None) -> pd.DataFrame:
    kind = infer_table_kind(explicit=table_kind)
    registry_names: set[str] = set()
    if registry is not None and not registry.empty:
        for c in ["feature", "feature_name", "name", "column"]:
            if c in registry.columns:
                registry_names.update(normalize_name(x) for x in registry[c].dropna().astype(str))
                break
    rows: list[ColumnRoleRecord] = []
    for col in df.columns:
        s = df[col]
        n = normalize_name(col)
        numeric = _is_numeric_like(s)
        miss = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        dtype = str(s.dtype)
        if n in _TASK_EXACT:
            role, conf, reason = ROLE_TASK, 0.96, "Task/prompt identity column."
        elif n in _IDENTIFIER_EXACT or (not numeric and n == "id"):
            role, conf, reason = ROLE_IDENTIFIER, 0.95, "Identifier column."
        elif n in _TIME_EXACT:
            role, conf, reason = ROLE_TIME, 0.92, "Time, visit, or recording-date column."
        elif _matches_any(n, _AUDIT_PATTERNS):
            role, conf, reason = ROLE_AUDIT, 0.88, "Audit/status/path/error column; not analyzed as a feature."
        elif kind == "qc" and numeric:
            role, conf, reason = ROLE_QC, 0.90, "Numeric column in QC table."
        elif _matches_any(n, _QC_PATTERNS):
            role = ROLE_QC if numeric else ROLE_COVARIATE
            conf, reason = 0.86, "QC/artifact-related column name."
        elif n in registry_names and numeric:
            role, conf, reason = ROLE_FEATURE, 0.97, "Column found in supplied feature registry/policy table."
        elif n in _TARGET_EXACT:
            role, conf, reason = ROLE_TARGET, 0.90, "Exact clinical label/outcome name."
        elif n in _COVARIATE_EXACT:
            role, conf, reason = ROLE_COVARIATE, 0.88, "Known demographic/device/site covariate."
        elif kind == "metadata":
            if n in _TARGET_EXACT:
                role, conf, reason = ROLE_TARGET, 0.90, "Clinical label in metadata table."
            elif n in _COVARIATE_EXACT or not numeric or _is_low_cardinality(s):
                role, conf, reason = ROLE_COVARIATE, 0.72, "Metadata descriptor/covariate."
            else:
                role, conf, reason = ROLE_COVARIATE, 0.60, "Numeric metadata column; retained as covariate unless manually changed."
        elif _matches_any(n, _FEATURE_NAME_PATTERNS) and numeric:
            role, conf, reason = ROLE_FEATURE, 0.94, "Numeric column with acoustic/kinematic feature-like name."
        elif numeric and kind in {"feature", "generic"}:
            role, conf, reason = ROLE_FEATURE, 0.82, "Numeric primary-table column; conservative default is Feature."
        elif not numeric and _is_low_cardinality(s):
            role, conf, reason = ROLE_COVARIATE, 0.62, "Low-cardinality nonnumeric column; likely grouping/covariate."
        else:
            role, conf, reason = ROLE_IGNORE, 0.50, "Unclassified nonnumeric column."
        rows.append(ColumnRoleRecord(str(col), role, conf, reason, dtype, miss, nunique, numeric))
    return pd.DataFrame([r.__dict__ for r in rows])


def _role_aliases(role: ColumnRole | str) -> set[str]:
    if isinstance(role, ColumnRole):
        return {role.value, _ENUM_TO_DISPLAY.get(role, role.value)}
    r = str(role)
    aliases = {r}
    if r in _DISPLAY_TO_VALUE:
        aliases.add(_DISPLAY_TO_VALUE[r])
    if r in _VALUE_TO_DISPLAY:
        aliases.add(_VALUE_TO_DISPLAY[r])
    return aliases


def columns_by_role(mapping: pd.DataFrame, role: ColumnRole | str) -> list[str]:
    if mapping.empty or "role" not in mapping.columns:
        return []
    aliases = _role_aliases(role)
    return mapping.loc[mapping["role"].astype(str).isin(aliases), "column"].astype(str).tolist()


def role_lists(mapping: pd.DataFrame) -> Mapping[str, list[str]]:
    out: dict[str, list[str]] = {}
    if mapping.empty:
        return out
    for role, sub in mapping.groupby("role"):
        out[str(role)] = sub["column"].astype(str).tolist()
    return out


def summarize_roles(mapping: pd.DataFrame) -> pd.DataFrame:
    if mapping.empty:
        return pd.DataFrame(columns=["role", "n_columns"])
    return mapping.groupby("role", dropna=False).size().reset_index(name="n_columns").sort_values("role")


def infer_column_roles(df: pd.DataFrame, source_hint: str = "features") -> pd.DataFrame:
    # Backward-compatible alias used by tests and the earlier backend.
    kind = "qc" if "qc" in source_hint.lower() or "quality" in source_hint.lower() else ("metadata" if "meta" in source_hint.lower() else "feature")
    return classify_columns(df, table_kind=kind)


def choose_join_key(left: pd.DataFrame, right: pd.DataFrame) -> str | None:
    preferred = ["record_key", "file_name", "filename", "source_file", "subject_id", "session_id"]
    left_cols = {normalize_name(c): c for c in left.columns}
    right_cols = {normalize_name(c): c for c in right.columns}
    for key in preferred:
        if key in left_cols and key in right_cols:
            return left_cols[key]
    common_norm = [k for k in left_cols if k in right_cols]
    return left_cols[common_norm[0]] if common_norm else None
