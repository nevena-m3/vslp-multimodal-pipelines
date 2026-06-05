"""Column-role detection for the VSLP Feature Analysis GUI.

The design is intentionally conservative: in the primary feature table, numeric
columns are treated as features unless there is strong evidence that they are
identifiers, QC variables, audit/status variables, or clinical labels. This
prevents acoustic/kinematic features from being misclassified as targets.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

ROLE_IDENTIFIER = "Identifier"
ROLE_FEATURE = "Feature"
ROLE_QC = "QC feature"
ROLE_TARGET = "Target / label"
ROLE_COVARIATE = "Covariate"
ROLE_TIME = "Time / visit"
ROLE_AUDIT = "Audit / status"
ROLE_IGNORE = "Ignore"

ROLE_OPTIONS = [
    ROLE_IDENTIFIER,
    ROLE_FEATURE,
    ROLE_QC,
    ROLE_TARGET,
    ROLE_COVARIATE,
    ROLE_TIME,
    ROLE_AUDIT,
    ROLE_IGNORE,
]

_IDENTIFIER_EXACT = {
    "file_name", "filename", "file", "source_file", "source_path", "path",
    "relative_path", "absolute_path", "record_key", "recording_id", "sample_id",
    "subject_id", "participant_id", "patient_id", "id", "session_id",
    "visit_id", "clinical_visit_id", "protocol_id", "iteration", "trial",
    "task", "task_name", "modality", "project", "project_name",
}

_TIME_EXACT = {
    "date", "recording_date", "visit_date", "session_date", "timestamp",
    "timepoint", "time_point", "days_from_baseline", "months_from_baseline",
}

_TARGET_EXACT = {
    "diagnosis", "dx", "group", "class", "label", "target", "outcome",
    "severity", "severity_bin", "severity_class", "severity_score",
    "alsfrs", "alsfrs_total", "alsfrs_total_score", "alsfrs_r",
    "alsfrs_bulbar", "alsfrs_bulbar_subscore", "alsbdi", "alsbdi_total",
    "progression_rate", "progressor", "disease_progression",
}

_COVARIATE_EXACT = {
    "age", "sex", "gender", "site", "device", "microphone", "language",
    "dialect", "education", "handedness", "onset_site", "disease_duration",
}

_AUDIT_PATTERNS = [
    r"(^|_)status($|_)", r"(^|_)warning", r"(^|_)flag", r"(^|_)reason",
    r"implementation_status", r"feature_status", r"computed_proxy", r"validity",
    r"manifest", r"hash", r"sha256", r"version", r"path", r"error",
]

_QC_PATTERNS = [
    r"(^|_)qc($|_)", r"quality", r"snr", r"clip", r"clipping", r"noise",
    r"hum", r"interference", r"reverb", r"echo", r"dropout", r"gain",
    r"distortion", r"powerline", r"saturation", r"artifact", r"contamination",
]

_FEATURE_NAME_PATTERNS = [
    # Acoustic timing/rhythm/phonatory/formant/nasality/coordination families
    r"^f0", r"^cpp", r"^hnr", r"jitter", r"shimmer", r"voicebreak",
    r"^h1", r"^h2", r"^f[1-5]($|_)", r"formant", r"bandwidth", r"_bw$",
    r"slope", r"d_dx", r"range$", r"a1p0", r"a1p1", r"a3p0",
    r"p0", r"p1", r"rms", r"speech_dur", r"total_dur", r"pause",
    r"phrase", r"speech_rate", r"intensity", r"fft_", r"nrj_", r"ratio_",
    r"coord", r"comp$", r"vsa", r"vai", r"fcr", r"ddk", r"ems",
    # Kinematic/generic movement families
    r"velocity", r"speed", r"accel", r"acceleration", r"jerk", r"range_of_motion",
    r"rom", r"displacement", r"trajectory", r"landmark", r"opening", r"closing",
    r"lip", r"jaw", r"cheek", r"face", r"facial", r"kinematic",
]

@dataclass(frozen=True)
class ColumnRole:
    column: str
    role: str
    confidence: float
    reason: str
    dtype: str
    missing_fraction: float
    unique_values: int


def _norm(name: str) -> str:
    x = str(name).strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    return x.strip("_")


def _matches_any(norm: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, norm) for p in patterns)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _is_low_cardinality(s: pd.Series, max_unique: int = 12) -> bool:
    non_na = s.dropna()
    if len(non_na) == 0:
        return False
    return non_na.nunique(dropna=True) <= max_unique


def _registry_feature_set(registry: pd.DataFrame | None) -> set[str]:
    if registry is None or registry.empty:
        return set()
    candidates = ["feature", "feature_name", "name", "id", "column", "column_name"]
    features: set[str] = set()
    for col in registry.columns:
        if _norm(col) in candidates:
            features.update(_norm(v) for v in registry[col].dropna().astype(str).tolist())
    return features


def infer_table_kind(path_or_name: str | None = None, explicit: str | None = None) -> str:
    """Return one of feature, qc, metadata, registry, generic."""
    if explicit:
        return explicit
    name = _norm(path_or_name or "")
    if "quality" in name or re.search(r"(^|_)qc($|_)", name):
        return "qc"
    if "metadata" in name or "file_index" in name or "demographic" in name:
        return "metadata"
    if "registry" in name or "policy" in name or "status_long" in name or "reduction_audit" in name:
        return "registry"
    if "feature" in name:
        return "feature"
    return "generic"


def classify_columns(
    df: pd.DataFrame,
    *,
    table_kind: str = "feature",
    registry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify columns into analysis roles with confidence and rationale.

    In a primary feature table, numeric columns default to Feature unless a
    stricter rule catches them first. Targets are only assigned for exact
    clinical-label names, not broad words like "score" alone.
    """
    registry_features = _registry_feature_set(registry)
    rows: list[ColumnRole] = []
    kind = table_kind or "feature"

    for col in df.columns:
        s = df[col]
        n = _norm(col)
        dtype = str(s.dtype)
        miss = float(s.isna().mean()) if len(s) else 0.0
        nunique = int(s.nunique(dropna=True)) if len(s) else 0
        numeric = _is_numeric(s)

        role = ROLE_IGNORE
        conf = 0.50
        reason = "No strong role detected."

        if n in _IDENTIFIER_EXACT:
            role, conf, reason = ROLE_IDENTIFIER, 0.99, "Exact identifier column name."
        elif n in _TIME_EXACT or re.search(r"(^|_)(date|timepoint|timestamp)($|_)", n):
            role, conf, reason = ROLE_TIME, 0.95, "Date, timestamp, or longitudinal time variable."
        elif _matches_any(n, _AUDIT_PATTERNS):
            role, conf, reason = ROLE_AUDIT, 0.90, "Audit/status/flag/version/path-like column."
        elif kind == "qc" and numeric:
            role, conf, reason = ROLE_QC, 0.95, "Numeric column in the supplied QC table."
        elif _matches_any(n, _QC_PATTERNS) and numeric:
            role, conf, reason = ROLE_QC, 0.85, "QC/artifact-like numeric column name."
        elif n in registry_features:
            role, conf, reason = ROLE_FEATURE, 0.99, "Column appears in supplied feature registry/policy."
        elif kind == "metadata":
            if n in _TARGET_EXACT:
                role, conf, reason = ROLE_TARGET, 0.92, "Clinical label/outcome column in metadata."
            elif n in _COVARIATE_EXACT:
                role, conf, reason = ROLE_COVARIATE, 0.90, "Known demographic/device/site covariate."
            else:
                role, conf, reason = ROLE_COVARIATE, 0.65, "Metadata column not otherwise classified."
        elif n in _TARGET_EXACT:
            # Conservative target classification: exact clinical labels only.
            role, conf, reason = ROLE_TARGET, 0.88, "Exact clinical label/outcome name."
        elif n in _COVARIATE_EXACT:
            role, conf, reason = ROLE_COVARIATE, 0.88, "Known covariate column."
        elif _matches_any(n, _FEATURE_NAME_PATTERNS) and numeric:
            role, conf, reason = ROLE_FEATURE, 0.92, "Numeric column with acoustic/kinematic feature-like name."
        elif numeric and kind in {"feature", "generic"}:
            # The main protection against feature -> target misclassification.
            role, conf, reason = ROLE_FEATURE, 0.80, "Numeric primary-table column; treated as feature by conservative default."
        elif kind == "feature" and not numeric and _is_low_cardinality(s):
            if n in _TARGET_EXACT:
                role, conf, reason = ROLE_TARGET, 0.90, "Exact target-like low-cardinality column."
            else:
                role, conf, reason = ROLE_COVARIATE, 0.62, "Low-cardinality nonnumeric column; likely grouping/covariate."
        else:
            role, conf, reason = ROLE_IGNORE, 0.50, "Unclassified nonnumeric column."

        rows.append(ColumnRole(col, role, conf, reason, dtype, miss, nunique))

    return pd.DataFrame([r.__dict__ for r in rows])


def summarize_roles(mapping: pd.DataFrame) -> pd.DataFrame:
    if mapping.empty:
        return pd.DataFrame(columns=["role", "n_columns"])
    return mapping.groupby("role", dropna=False).size().reset_index(name="n_columns").sort_values("role")


def role_lists(mapping: pd.DataFrame) -> Mapping[str, list[str]]:
    out: dict[str, list[str]] = {}
    if mapping.empty:
        return out
    for role, sub in mapping.groupby("role"):
        out[str(role)] = sub["column"].astype(str).tolist()
    return out
