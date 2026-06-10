"""Feature-GUI ML export builder.

This module intentionally lives in the Feature Analysis layer, not in the
acoustic or kinematics pipelines. Acoustic and kinematic GUIs produce their own
scientific feature/audit outputs; this layer curates those outputs into clean,
model-facing tables and manifests.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


ID_COLUMN_CANDIDATES = [
    "subject_id", "participant_id", "patient_id", "session_id", "visit_id",
    "iteration", "task", "recording_date", "record_key", "file_name",
    "video_id", "audio_id", "recording_id",
]
TARGET_COLUMN_CANDIDATES = [
    "diagnosis", "disease_group", "group", "severity_score", "severity_bin",
    "alsfrs_r", "alsfrs_r_bulbar", "updrs", "days_from_baseline",
]
PROVENANCE_PATTERNS = [
    "path", "source", "input_", "output_", "csv", "wav", "video", "file_name"
]
QC_PATTERNS = [
    "qc", "status", "flag", "valid_fraction", "detected_fraction", "n_frames",
    "n_signals", "n_movements", "aggregation_profile", "duration_s",
]


@dataclass
class MLExportResult:
    output_dir: Path
    paths: dict[str, Path]
    tables: dict[str, pd.DataFrame]
    summary: dict[str, object]


def _read(path: Optional[str | Path]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(p)
    if p.suffix.lower() == ".tsv":
        return pd.read_csv(p, sep="\t")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported table format: {p.suffix}. Use CSV, TSV, TXT, or Parquet.")


def _norm_cols(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _lower_map(columns: Iterable[str]) -> dict[str, str]:
    return {str(c).lower(): str(c) for c in columns}


def _existing(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    lm = _lower_map(df.columns)
    out = []
    for c in candidates:
        if c.lower() in lm:
            out.append(lm[c.lower()])
    return out


def _is_numeric_feature(df: pd.DataFrame, col: str) -> bool:
    return pd.api.types.is_numeric_dtype(df[col])


def _is_provenance_col(col: str) -> bool:
    lc = col.lower()
    return any(p in lc for p in PROVENANCE_PATTERNS)


def _is_qc_col(col: str) -> bool:
    lc = col.lower()
    return any(p in lc for p in QC_PATTERNS)


def _infer_acoustic_family(feature: str, registry: Optional[pd.DataFrame]) -> str:
    if registry is not None and not registry.empty and "feature" in registry.columns:
        rows = registry[registry["feature"].astype(str).str.lower() == feature.lower()]
        if not rows.empty and "subsystem" in rows.columns:
            val = str(rows.iloc[0]["subsystem"])
            if val and val != "nan":
                return val
    lc = feature.lower()
    if any(k in lc for k in ["pause", "phrase", "speech_rate", "total_dur", "speech_dur"]):
        return "respiratory_timing"
    if any(k in lc for k in ["jitter", "shimmer", "hnr", "cpp", "f0", "pitch"]):
        return "phonatory_voice_quality"
    if re.match(r"f[1-5]", lc) or "formant" in lc:
        return "resonatory_articulatory"
    if any(k in lc for k in ["mfcc", "spectral", "centroid", "rolloff"]):
        return "spectral_cepstral"
    return "acoustic_other"


def _infer_kinematic_family(feature: str) -> str:
    lc = feature.lower()
    if any(k in lc for k in ["aperture", "rom_vert", "sll_vert", "all_vert", "path_vert", "mouth_height"]):
        return "mouth_aperture_opening"
    if any(k in lc for k in ["spread", "rom_horz", "sll_horz", "all_horz", "path_horz", "mouth_width"]):
        return "lip_spread_horizontal"
    if any(k in lc for k in ["jaw", "chin", "lower_lip_to_chin", "nose"]):
        return "jaw_lower_face"
    if any(k in lc for k in ["symm", "asym", "left_", "right_", "xcorr", "coord"]):
        return "symmetry_coordination"
    if "velocity" in lc or lc.startswith("sll"):
        return "velocity"
    if lc.startswith("all") or "accel" in lc:
        return "acceleration"
    return "kinematic_other"


def _infer_primitive(feature: str) -> str:
    lc = feature.lower()
    if "velocity" in lc or lc.startswith("sll") or "speed" in lc:
        return "velocity"
    if lc.startswith("all") or "accel" in lc:
        return "acceleration"
    if "path" in lc:
        return "path"
    if "rom" in lc or "range" in lc or "p95" in lc or "p05" in lc:
        return "range_distribution"
    if "iqr" in lc or "sd" in lc or "cv" in lc:
        return "variability"
    if "ratio" in lc or "symm" in lc or "asym" in lc or "xcorr" in lc:
        return "symmetry_coordination"
    return "scalar"


def _infer_summary_stat(feature: str) -> str:
    lc = feature.lower()
    for stat in ["median", "mean", "iqr", "p05", "p95", "min", "max", "sd", "std", "range_p05_p95"]:
        if lc.endswith("_" + stat) or (stat == "range_p05_p95" and "range_p05_p95" in lc):
            return stat
    if lc.endswith("_med"):
        return "median"
    if "prc_5_95" in lc:
        return "p95_minus_p05"
    if "prc_95" in lc:
        return "p95"
    if "prc_5" in lc:
        return "p05"
    return "file_scalar"


def _registry_lookup(registry: Optional[pd.DataFrame], feature: str, col: str) -> str:
    if registry is None or registry.empty or "feature" not in registry.columns or col not in registry.columns:
        return ""
    rows = registry[registry["feature"].astype(str).str.lower() == feature.lower()]
    if rows.empty:
        return ""
    val = rows.iloc[0][col]
    return "" if pd.isna(val) else str(val)


def classify_feature_table_columns(
    df: pd.DataFrame,
    modality: str,
    registry: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return a column manifest with ML roles for one modality table."""
    rows = []
    id_cols = set(_existing(df, ID_COLUMN_CANDIDATES))
    target_cols = set(_existing(df, TARGET_COLUMN_CANDIDATES))
    registry_features = set()
    if registry is not None and not registry.empty and "feature" in registry.columns:
        registry_features = {str(x) for x in registry["feature"].dropna().astype(str)}
    for col in df.columns:
        role = "other"
        include = False
        family = ""
        unit = ""
        interpretation = ""
        caution = ""
        primitive = _infer_primitive(col)
        if col in id_cols:
            role = "id"
        elif col in target_cols:
            role = "target_candidate"
        elif _is_provenance_col(col):
            role = "provenance"
        elif _is_qc_col(col):
            role = "qc_metric"
        elif _is_numeric_feature(df, col):
            # Registry columns are preferred, but numeric engineering summaries are also
            # allowed as candidate predictors in the full feature-catalog layer.
            role = f"{modality}_predictor"
            include = True
        else:
            role = "metadata"

        if role.endswith("_predictor"):
            if modality == "acoustic":
                family = _infer_acoustic_family(col, registry)
                unit = _registry_lookup(registry, col, "unit")
                interpretation = _registry_lookup(registry, col, "meaning") or _registry_lookup(registry, col, "interpretation_note")
                caution = _registry_lookup(registry, col, "computation_note")
            else:
                family = _infer_kinematic_family(col)
                unit = "normalized units" if any(k in col.lower() for k in ["mouth", "lip", "jaw", "rom", "path", "spread", "aperture"]) else ""
                interpretation = f"Kinematic scalar summary for {family.replace('_', ' ')}."
                caution = "Use with video QC and normalization diagnostics; not a diagnostic cutoff."
        elif role == "qc_metric":
            family = "quality_control"
            caution = "Use for filtering/stratification, not as a disease predictor by default."
        elif role == "target_candidate":
            family = "target_outcome"
        elif role == "id":
            family = "row_identity"
        elif role == "provenance":
            family = "provenance"

        evidence = _registry_lookup(registry, col, "evidence_tier")
        rows.append({
            "column": col,
            "feature_name": col,
            "modality": modality,
            "role": role,
            "family": family,
            "primitive": primitive,
            "summary_statistic": _infer_summary_stat(col),
            "unit": unit,
            "include_in_ml_default": bool(include),
            "is_numeric": bool(_is_numeric_feature(df, col)),
            "evidence_tier": evidence,
            "interpretation": interpretation,
            "caution": caution,
        })
    return pd.DataFrame(rows)


def _apply_metadata(df: pd.DataFrame, metadata: Optional[pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    if metadata is None or metadata.empty:
        return df.copy(), "No metadata supplied. Used labels/IDs already present in the feature table."
    shared = [c for c in ["record_key", "subject_id", "session_id", "visit_id", "task", "file_name", "video_id"] if c in df.columns and c in metadata.columns]
    if "record_key" in shared:
        keys = ["record_key"]
    elif {"subject_id", "session_id", "task"}.issubset(set(shared)):
        keys = ["subject_id", "session_id", "task"]
    elif {"subject_id", "task"}.issubset(set(shared)):
        keys = ["subject_id", "task"]
    elif shared:
        keys = [shared[0]]
    else:
        return df.copy(), "Metadata supplied but no safe shared join key was found; metadata was not merged."
    meta_cols = [c for c in metadata.columns if c not in df.columns or c in keys]
    merged = df.merge(metadata[meta_cols], on=keys, how="left", suffixes=("", "_metadata"))
    return merged, f"Metadata merged using key(s): {', '.join(keys)}."


def _make_ml_ready(df: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_cols = [c for c in manifest.loc[manifest["role"].eq("id"), "column"].tolist() if c in df.columns]
    target_cols = [c for c in manifest.loc[manifest["role"].eq("target_candidate"), "column"].tolist() if c in df.columns]
    feature_cols = [c for c in manifest.loc[manifest["include_in_ml_default"].astype(bool), "column"].tolist() if c in df.columns]
    # Keep row-context columns in ML-ready table, but features-only is numeric predictors only.
    context_cols = []
    for c in ["subject_id", "session_id", "visit_id", "task", "recording_id", "record_key", "file_name", "video_id", "diagnosis", "severity_score", "severity_bin"]:
        if c in df.columns and c not in context_cols:
            context_cols.append(c)
    for c in id_cols + target_cols:
        if c in df.columns and c not in context_cols:
            context_cols.append(c)
    ml_cols = context_cols + [c for c in feature_cols if c not in context_cols]
    return df[ml_cols].copy() if ml_cols else pd.DataFrame(), df[feature_cols].copy() if feature_cols else pd.DataFrame()


def _prefix_feature_columns(df: pd.DataFrame, manifest: pd.DataFrame, modality: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    man = manifest.copy()
    protected = set(_existing(out, ID_COLUMN_CANDIDATES + TARGET_COLUMN_CANDIDATES))
    rename = {}
    for _, row in man.iterrows():
        col = row["column"]
        if col in protected:
            continue
        if row.get("include_in_ml_default", False) and not str(col).startswith(f"{modality}__"):
            rename[col] = f"{modality}__{col}"
    out = out.rename(columns=rename)
    man["column"] = man["column"].map(lambda c: rename.get(c, c))
    man["feature_name"] = man["column"]
    return out, man


def _row_alignment(acoustic: Optional[pd.DataFrame], kinematic: Optional[pd.DataFrame]) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    if acoustic is None or acoustic.empty or kinematic is None or kinematic.empty:
        return pd.DataFrame(), [], pd.DataFrame([{"status": "not_available", "reason": "Both acoustic and kinematic ML-ready tables are required for early fusion."}])
    candidate_sets = [
        ["record_key"],
        ["subject_id", "session_id", "task"],
        ["subject_id", "visit_id", "task"],
        ["subject_id", "task"],
        ["subject_id"],
    ]
    keys = []
    for cand in candidate_sets:
        if all(c in acoustic.columns and c in kinematic.columns for c in cand):
            keys = cand
            break
    if not keys:
        return pd.DataFrame(), [], pd.DataFrame([{"status": "review", "reason": "No safe shared row key found. Provide metadata with subject/session/task alignment."}])
    a_key = acoustic[keys].drop_duplicates()
    k_key = kinematic[keys].drop_duplicates()
    inner = a_key.merge(k_key, on=keys, how="inner")
    report = pd.DataFrame([
        {"metric": "join_keys", "value": ", ".join(keys)},
        {"metric": "acoustic_rows", "value": len(acoustic)},
        {"metric": "kinematic_rows", "value": len(kinematic)},
        {"metric": "matched_key_rows", "value": len(inner)},
        {"metric": "acoustic_unique_keys", "value": len(a_key)},
        {"metric": "kinematic_unique_keys", "value": len(k_key)},
    ])
    return inner, keys, report


def build_ml_export_package(
    output_root: str | Path,
    acoustic_features: Optional[str | Path] = None,
    acoustic_registry: Optional[str | Path] = None,
    acoustic_scale_registry: Optional[str | Path] = None,
    kinematic_features: Optional[str | Path] = None,
    kinematic_manifest: Optional[str | Path] = None,
    metadata: Optional[str | Path] = None,
) -> MLExportResult:
    """Build acoustic, kinematic and early-fusion ML-ready tables.

    Missing modalities are allowed. The function writes all tables it can build
    and records skipped/fallback decisions in reports/manifests.
    """
    output_dir = Path(output_root) / "feature_analysis" / "ml_export_builder"
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    meta = _norm_cols(_read(metadata))
    ac_df = _norm_cols(_read(acoustic_features))
    ac_registry = _norm_cols(_read(acoustic_registry))
    ac_scale_registry = _norm_cols(_read(acoustic_scale_registry))
    kin_df = _norm_cols(_read(kinematic_features))
    kin_manifest_in = _norm_cols(_read(kinematic_manifest))

    tables: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}
    notes: list[str] = []

    ac_ml = ac_only = ac_manifest = pd.DataFrame()
    kin_ml = kin_only = kin_manifest = pd.DataFrame()

    if ac_df is not None:
        ac_df, note = _apply_metadata(ac_df, meta)
        notes.append(f"acoustic: {note}")
        ac_manifest = classify_feature_table_columns(ac_df, "acoustic", ac_registry)
        # Scale registry can add ML recommendation text when available.
        if ac_scale_registry is not None and "feature" in ac_scale_registry.columns:
            rec = ac_scale_registry[[c for c in ["feature", "ml_recommendation", "interpretation_note", "native_scale"] if c in ac_scale_registry.columns]].copy()
            rec = rec.rename(columns={"feature": "column"})
            ac_manifest = ac_manifest.merge(rec, on="column", how="left")
            if "ml_recommendation" in ac_manifest.columns:
                ac_manifest["caution"] = ac_manifest["caution"].fillna("") + ac_manifest["ml_recommendation"].fillna("").map(lambda x: (" | " + x) if x else "")
        ac_ml, ac_only = _make_ml_ready(ac_df, ac_manifest)
        tables["acoustic_ml_ready"] = ac_ml
        tables["acoustic_features_only"] = ac_only
        tables["acoustic_feature_manifest_unified"] = ac_manifest
    else:
        notes.append("acoustic: no acoustic feature table supplied.")

    if kin_df is not None:
        kin_df, note = _apply_metadata(kin_df, meta)
        notes.append(f"kinematic: {note}")
        if kin_manifest_in is not None and not kin_manifest_in.empty and "column" in kin_manifest_in.columns:
            kin_manifest = kin_manifest_in.copy()
        else:
            kin_manifest = classify_feature_table_columns(kin_df, "kinematic", None)
        # Ensure required manifest columns exist if a previous manifest was loaded.
        for c in ["column", "feature_name", "modality", "role", "family", "primitive", "summary_statistic", "unit", "include_in_ml_default", "is_numeric", "evidence_tier", "interpretation", "caution"]:
            if c not in kin_manifest.columns:
                if c == "column" and "feature_name" in kin_manifest.columns:
                    kin_manifest[c] = kin_manifest["feature_name"]
                elif c == "feature_name" and "column" in kin_manifest.columns:
                    kin_manifest[c] = kin_manifest["column"]
                elif c == "modality":
                    kin_manifest[c] = "kinematic"
                elif c == "include_in_ml_default":
                    role = kin_manifest.get("role", pd.Series([""] * len(kin_manifest))).astype(str)
                    kin_manifest[c] = role.str.contains("predictor", case=False, na=False)
                elif c == "is_numeric":
                    kin_manifest[c] = kin_manifest.get("column", pd.Series(dtype=str)).astype(str).map(lambda x: x in kin_df.columns and _is_numeric_feature(kin_df, x))
                else:
                    kin_manifest[c] = ""
        # If manifest is very permissive, still suppress obvious QC/support columns from ML default.
        suppress = kin_manifest["column"].astype(str).map(lambda x: _is_qc_col(x) or _is_provenance_col(x))
        kin_manifest.loc[suppress, "include_in_ml_default"] = False
        kin_manifest.loc[suppress & kin_manifest["role"].astype(str).str.contains("predictor", case=False, na=False), "role"] = "qc_metric"
        kin_ml, kin_only = _make_ml_ready(kin_df, kin_manifest)
        tables["kinematic_ml_ready"] = kin_ml
        tables["kinematic_features_only"] = kin_only
        tables["kinematic_feature_manifest_unified"] = kin_manifest
    else:
        notes.append("kinematic: no kinematic feature table supplied.")

    # Prefix predictors in early fusion to prevent acoustic/kinematic column collisions.
    ac_pref, ac_man_pref = _prefix_feature_columns(ac_ml, ac_manifest, "acoustic") if not ac_ml.empty else (pd.DataFrame(), pd.DataFrame())
    kin_pref, kin_man_pref = _prefix_feature_columns(kin_ml, kin_manifest, "kinematic") if not kin_ml.empty else (pd.DataFrame(), pd.DataFrame())
    _, keys, align_report = _row_alignment(ac_pref, kin_pref)
    tables["row_alignment_report"] = align_report
    if keys:
        ac_cols = keys + [c for c in ac_pref.columns if c not in keys and not c.endswith("_metadata")]
        kin_feature_cols = [c for c in kin_pref.columns if c not in keys and c not in TARGET_COLUMN_CANDIDATES and c not in ID_COLUMN_CANDIDATES]
        # Preserve targets/context from acoustic side first, then add kinematic predictors.
        early = ac_pref[ac_cols].merge(kin_pref[keys + kin_feature_cols], on=keys, how="inner")
        tables["multimodal_early_fusion_ml_ready"] = early
    else:
        tables["multimodal_early_fusion_ml_ready"] = pd.DataFrame()

    unified_manifest_parts = []
    if not ac_manifest.empty:
        unified_manifest_parts.append(ac_manifest)
    if not kin_manifest.empty:
        unified_manifest_parts.append(kin_manifest)
    unified_manifest = pd.concat(unified_manifest_parts, ignore_index=True, sort=False) if unified_manifest_parts else pd.DataFrame()
    tables["feature_manifest_unified"] = unified_manifest

    row_exclusions = []
    if ac_df is None:
        row_exclusions.append({"source": "acoustic", "status": "missing", "reason": "No acoustic feature table supplied."})
    if kin_df is None:
        row_exclusions.append({"source": "kinematic", "status": "missing", "reason": "No kinematic feature table supplied."})
    if ac_df is not None and kin_df is not None and not keys:
        row_exclusions.append({"source": "multimodal", "status": "review", "reason": "Early fusion not built because no safe join key was found."})
    tables["row_exclusions"] = pd.DataFrame(row_exclusions)

    for name, df in tables.items():
        p = tables_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p

    summary = {
        "app_layer": "Feature Analysis GUI ML Export Builder",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "acoustic_rows": 0 if ac_df is None else int(len(ac_df)),
        "kinematic_rows": 0 if kin_df is None else int(len(kin_df)),
        "acoustic_predictors": 0 if ac_manifest.empty else int(ac_manifest["include_in_ml_default"].astype(bool).sum()),
        "kinematic_predictors": 0 if kin_manifest.empty else int(kin_manifest["include_in_ml_default"].astype(bool).sum()),
        "early_fusion_rows": int(len(tables["multimodal_early_fusion_ml_ready"])),
        "join_keys": keys,
        "notes": notes,
        "boundary": "Feature curation/export only. Model training, imputation, scaling and feature selection must occur later inside ML validation folds.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    manifest_path = tables_dir / "ml_export_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["ml_export_manifest_json"] = manifest_path
    return MLExportResult(output_dir=output_dir, paths=paths, tables=tables, summary=summary)
