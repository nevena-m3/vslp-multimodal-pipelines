"""ML data-contract and multimodal dataset assembly utilities.

This module builds the first ML-GUI foundation layer. It does not train models.
Its responsibilities are to load acoustic/kinematic ML-ready tables, harmonize
feature manifests, join metadata/labels, compute modality overlap, and write
leakage-aware dataset artifacts for later model training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

ROW_KEY_CANDIDATES: tuple[str, ...] = (
    "subject_id", "session_id", "visit_id", "iteration", "task", "recording_id", "record_key", "video_id", "file_name"
)
DEFAULT_PROTECTED_COLUMNS: tuple[str, ...] = (
    "diagnosis", "disease_group", "label", "target", "severity_score", "severity_bin", "ALSFRS_R", "ALSFRS_R_bulbar", "UPDRS", "days_from_baseline", "visit_date"
)


@dataclass(frozen=True)
class MLDatasetContractConfig:
    """Configuration for ML dataset assembly."""

    acoustic_features_csv: str | None = None
    kinematic_features_csv: str | None = None
    metadata_csv: str | None = None
    acoustic_manifest_csv: str | None = None
    kinematic_manifest_csv: str | None = None
    target_column: str | None = None
    key_columns: list[str] = field(default_factory=list)
    qc_policy: str = "pass_review"  # pass_only, pass_review, include_all
    comparison_mode: str = "maximum_available"  # maximum_available, matched_subjects

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_csv(path: str | Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        return None
    return pd.read_csv(p)


def _normalize_manifest(df: pd.DataFrame | None, modality: str, feature_columns: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        rows = []
        protected = set(ROW_KEY_CANDIDATES) | set(DEFAULT_PROTECTED_COLUMNS)
        for col in feature_columns:
            is_numeric = True
            rows.append({
                "column_name": col,
                "column_role": "metadata" if col in protected else "candidate_feature",
                "modality": modality,
                "family": "metadata" if col in protected else "unmanifested",
                "primitive": "identifier" if col in protected else "unknown",
                "summary_statistic": "n/a",
                "unit": "varies",
                "model_role": "join_key_or_target" if col in protected else "candidate_predictor",
                "include_in_ml_default": bool(col not in protected and is_numeric),
                "interpretation": "Auto-inferred because no manifest was supplied.",
                "caution": "Review before publication-grade modeling.",
            })
        return pd.DataFrame(rows)
    out = df.copy()
    if "feature_name" in out.columns and "column_name" not in out.columns:
        out = out.rename(columns={"feature_name": "column_name"})
    if "column_name" not in out.columns:
        first = out.columns[0]
        out = out.rename(columns={first: "column_name"})
    if "include_in_ml_default" not in out.columns:
        if "model_role" in out.columns:
            out["include_in_ml_default"] = out["model_role"].astype(str).str.contains("candidate", case=False, na=False)
        elif "column_role" in out.columns:
            out["include_in_ml_default"] = out["column_role"].astype(str).isin(["canonical_feature", "candidate_feature"])
        else:
            out["include_in_ml_default"] = False
    if "modality" not in out.columns:
        out["modality"] = modality
    for col, default in {
        "column_role": "candidate_feature",
        "family": "unclassified",
        "primitive": "unknown",
        "summary_statistic": "unknown",
        "unit": "varies",
        "model_role": "candidate_predictor",
        "interpretation": "",
        "caution": "",
    }.items():
        if col not in out.columns:
            out[col] = default
    out["modality"] = out["modality"].fillna(modality).replace("", modality)
    return out


def _truthy_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "include", "included"})


def _infer_key_columns(tables: list[pd.DataFrame], metadata: pd.DataFrame | None, explicit: list[str] | None = None) -> list[str]:
    if explicit:
        keys = [k for k in explicit if all(k in t.columns for t in tables if t is not None and not t.empty)]
        if metadata is not None:
            keys = [k for k in keys if k in metadata.columns]
        if keys:
            return keys
    candidate_tables = [t for t in tables if t is not None and not t.empty]
    if metadata is not None and not metadata.empty:
        candidate_tables.append(metadata)
    keys: list[str] = []
    for k in ROW_KEY_CANDIDATES:
        present_in_feature = any(k in t.columns for t in tables if t is not None and not t.empty)
        present_in_meta = metadata is None or k in metadata.columns
        if present_in_feature and present_in_meta:
            keys.append(k)
    # Use a compact default that joins reliably across modalities; avoid file/video id unless needed.
    preferred = [k for k in ("subject_id", "session_id", "visit_id", "iteration", "task", "record_key", "recording_id") if k in keys]
    return preferred or keys[:3]


def _apply_qc_policy(df: pd.DataFrame, modality: str, policy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return df, pd.DataFrame()
    status_cols = [c for c in ("qc_status", "status", "normalization_status") if c in df.columns]
    if policy == "include_all" or not status_cols:
        return df.copy(), pd.DataFrame()
    allowed = {"pass", "complete", "completed", "ok"}
    if policy == "pass_review":
        allowed |= {"review", "complete - review", "completed_with_warnings", "warning", "warnings"}
    masks = []
    for col in status_cols:
        masks.append(df[col].astype(str).str.strip().str.lower().isin(allowed))
    keep = masks[0]
    for m in masks[1:]:
        keep = keep | m
    excluded = df.loc[~keep].copy()
    if not excluded.empty:
        excluded["exclusion_reason"] = f"{modality}_qc_policy_{policy}"
    return df.loc[keep].copy(), excluded


def _prepare_modality(df: pd.DataFrame | None, manifest: pd.DataFrame | None, modality: str, key_cols: list[str], qc_policy: str) -> tuple[pd.DataFrame | None, pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return None, _normalize_manifest(manifest, modality, []), pd.DataFrame()
    filtered, excluded = _apply_qc_policy(df, modality, qc_policy)
    manifest_norm = _normalize_manifest(manifest, modality, list(filtered.columns))
    manifest_features = manifest_norm.loc[_truthy_series(manifest_norm["include_in_ml_default"]), "column_name"].astype(str).tolist()
    numeric_cols: list[str] = []
    for col in manifest_features:
        if col in filtered.columns:
            vals = pd.to_numeric(filtered[col], errors="coerce")
            if vals.notna().any():
                filtered[col] = vals
                numeric_cols.append(col)
    id_cols = [c for c in key_cols if c in filtered.columns]
    if not id_cols:
        id_cols = [c for c in ROW_KEY_CANDIDATES if c in filtered.columns][:3]
    out = filtered[id_cols + numeric_cols].copy()
    rename = {c: f"{modality}__{c}" for c in numeric_cols}
    out = out.rename(columns=rename)
    manifest_norm["ml_column_name"] = manifest_norm["column_name"].map(lambda c: f"{modality}__{c}" if c in numeric_cols else c)
    manifest_norm["selected_for_ml_v01"] = manifest_norm["column_name"].isin(numeric_cols)
    return out, manifest_norm, excluded


def _join_metadata(df: pd.DataFrame | None, metadata: pd.DataFrame | None, key_cols: list[str]) -> pd.DataFrame | None:
    if df is None:
        return None
    if metadata is None or metadata.empty:
        return df.copy()
    join_keys = [k for k in key_cols if k in df.columns and k in metadata.columns]
    if not join_keys:
        return df.copy()
    meta_cols = [c for c in metadata.columns if c not in df.columns or c in join_keys]
    return df.merge(metadata[meta_cols].drop_duplicates(), on=join_keys, how="left")


def _row_key(df: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    available = [k for k in key_cols if k in df.columns]
    if not available:
        return pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)
    return df[available].astype(str).agg("|".join, axis=1)


def _modality_overlap(acoustic: pd.DataFrame | None, kinematic: pd.DataFrame | None, key_cols: list[str], subject_col: str = "subject_id") -> pd.DataFrame:
    a_keys = set(_row_key(acoustic, key_cols)) if acoustic is not None and not acoustic.empty else set()
    k_keys = set(_row_key(kinematic, key_cols)) if kinematic is not None and not kinematic.empty else set()
    rows = [
        {"overlap_group": "acoustic_only", "n_rows": len(a_keys - k_keys)},
        {"overlap_group": "kinematic_only", "n_rows": len(k_keys - a_keys)},
        {"overlap_group": "both_modalities", "n_rows": len(a_keys & k_keys)},
        {"overlap_group": "any_modality", "n_rows": len(a_keys | k_keys)},
    ]
    if subject_col:
        def subjects(df):
            return set(df[subject_col].dropna().astype(str)) if df is not None and subject_col in df.columns else set()
        asub = subjects(acoustic); ksub = subjects(kinematic)
        rows.extend([
            {"overlap_group": "acoustic_only_subjects", "n_rows": len(asub - ksub)},
            {"overlap_group": "kinematic_only_subjects", "n_rows": len(ksub - asub)},
            {"overlap_group": "both_modality_subjects", "n_rows": len(asub & ksub)},
            {"overlap_group": "any_modality_subjects", "n_rows": len(asub | ksub)},
        ])
    return pd.DataFrame(rows)


def _problem_type(metadata: pd.DataFrame | None, target_column: str | None) -> str:
    if metadata is None or not target_column or target_column not in metadata.columns:
        return "unconfigured"
    vals = metadata[target_column].dropna()
    if vals.empty:
        return "unconfigured"
    numeric = pd.to_numeric(vals, errors="coerce")
    if numeric.notna().mean() > 0.9 and vals.nunique() > 8:
        return "regression"
    n = vals.astype(str).nunique()
    if n == 2:
        return "binary_classification"
    if n > 2:
        return "multiclass_classification"
    return "unconfigured"


def _leakage_precheck(df: pd.DataFrame | None, subject_col: str = "subject_id") -> dict[str, Any]:
    if df is None or df.empty:
        return {"status": "not_available", "message": "Dataset not available."}
    if subject_col not in df.columns:
        return {"status": "review", "message": f"Missing {subject_col}; subject-grouped validation cannot be guaranteed."}
    n_subjects = int(df[subject_col].dropna().astype(str).nunique())
    n_rows = int(len(df))
    if n_subjects < 2:
        return {"status": "review", "message": "Fewer than two subjects; train/test validation is not meaningful.", "n_subjects": n_subjects, "n_rows": n_rows}
    return {"status": "pass", "message": "Subject identifier is available for leakage-safe grouped validation.", "n_subjects": n_subjects, "n_rows": n_rows}


def build_ml_datasets(output_root: str | Path, config: MLDatasetContractConfig) -> dict[str, Any]:
    """Build unimodal and early-fusion ML dataset artifacts.

    This is intentionally a data-contract stage only. It writes datasets and
    manifests but does not train or evaluate models.
    """
    root = Path(output_root).expanduser().resolve()
    tables = root / "ml" / "001_dataset" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    acoustic_df = _read_csv(config.acoustic_features_csv)
    kinematic_df = _read_csv(config.kinematic_features_csv)
    metadata_df = _read_csv(config.metadata_csv)
    acoustic_manifest = _read_csv(config.acoustic_manifest_csv)
    kinematic_manifest = _read_csv(config.kinematic_manifest_csv)

    key_cols = _infer_key_columns([t for t in [acoustic_df, kinematic_df] if t is not None], metadata_df, config.key_columns)
    a_prepared, a_manifest, a_excl = _prepare_modality(acoustic_df, acoustic_manifest, "acoustic", key_cols, config.qc_policy)
    k_prepared, k_manifest, k_excl = _prepare_modality(kinematic_df, kinematic_manifest, "kinematic", key_cols, config.qc_policy)

    a_joined = _join_metadata(a_prepared, metadata_df, key_cols)
    k_joined = _join_metadata(k_prepared, metadata_df, key_cols)

    if a_prepared is not None and k_prepared is not None:
        join_keys = [k for k in key_cols if k in a_prepared.columns and k in k_prepared.columns]
        early = a_prepared.merge(k_prepared, on=join_keys, how="inner") if join_keys else pd.DataFrame()
        early = _join_metadata(early, metadata_df, key_cols)
    else:
        early = pd.DataFrame()

    if config.comparison_mode == "matched_subjects" and "subject_id" in early.columns:
        matched_subjects = set(early["subject_id"].dropna().astype(str))
        if a_joined is not None and "subject_id" in a_joined.columns:
            a_joined = a_joined[a_joined["subject_id"].astype(str).isin(matched_subjects)].copy()
        if k_joined is not None and "subject_id" in k_joined.columns:
            k_joined = k_joined[k_joined["subject_id"].astype(str).isin(matched_subjects)].copy()

    overlap = _modality_overlap(a_prepared, k_prepared, key_cols)
    manifest = pd.concat([a_manifest, k_manifest], ignore_index=True, sort=False)
    manifest_path = tables / "ml_feature_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    acoustic_path = tables / "ml_dataset_acoustic_only.csv"
    kinematic_path = tables / "ml_dataset_kinematic_only.csv"
    early_path = tables / "ml_dataset_early_fusion.csv"
    overlap_path = tables / "ml_modality_overlap.csv"
    exclusions_path = tables / "ml_row_exclusions.csv"
    dataset_manifest_path = tables / "ml_dataset_manifest.json"

    (a_joined if a_joined is not None else pd.DataFrame()).to_csv(acoustic_path, index=False)
    (k_joined if k_joined is not None else pd.DataFrame()).to_csv(kinematic_path, index=False)
    early.to_csv(early_path, index=False)
    overlap.to_csv(overlap_path, index=False)
    exclusions = pd.concat([a_excl.assign(modality="acoustic"), k_excl.assign(modality="kinematic")], ignore_index=True, sort=False)
    exclusions.to_csv(exclusions_path, index=False)

    target_type = _problem_type(metadata_df, config.target_column)
    checks = {
        "acoustic_only": _leakage_precheck(a_joined),
        "kinematic_only": _leakage_precheck(k_joined),
        "early_fusion": _leakage_precheck(early),
    }
    payload = {
        "status": "ML_DATASET_CONTRACT_BUILT",
        "stage": "ml_v0_1_dataset_contract",
        "config": config.to_dict(),
        "key_columns": key_cols,
        "target_column": config.target_column,
        "problem_type": target_type,
        "outputs": {
            "acoustic_only_csv": str(acoustic_path),
            "kinematic_only_csv": str(kinematic_path),
            "early_fusion_csv": str(early_path),
            "feature_manifest_csv": str(manifest_path),
            "modality_overlap_csv": str(overlap_path),
            "row_exclusions_csv": str(exclusions_path),
        },
        "n_rows": {
            "acoustic_only": int(0 if a_joined is None else len(a_joined)),
            "kinematic_only": int(0 if k_joined is None else len(k_joined)),
            "early_fusion": int(len(early)),
        },
        "n_features": {
            "acoustic": int(sum(str(c).startswith("acoustic__") for c in (a_joined.columns if a_joined is not None else []))),
            "kinematic": int(sum(str(c).startswith("kinematic__") for c in (k_joined.columns if k_joined is not None else []))),
            "early_fusion": int(sum(str(c).startswith(("acoustic__", "kinematic__")) for c in early.columns)),
        },
        "leakage_precheck": checks,
        "note": "This stage builds datasets only. Model training is intentionally not performed in ML v0.1.",
    }
    dataset_manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return {
        "acoustic_only_csv": acoustic_path,
        "kinematic_only_csv": kinematic_path,
        "early_fusion_csv": early_path,
        "feature_manifest_csv": manifest_path,
        "modality_overlap_csv": overlap_path,
        "row_exclusions_csv": exclusions_path,
        "dataset_manifest_json": dataset_manifest_path,
        "manifest": payload,
    }


__all__ = [
    "MLDatasetContractConfig",
    "build_ml_datasets",
    "ROW_KEY_CANDIDATES",
]
