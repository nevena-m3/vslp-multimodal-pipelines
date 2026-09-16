"""Shared, nonbreaking export contract for acoustic and kinematic features."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


FEATURE_REGISTRY_COLUMNS = (
    "feature", "modality", "family", "meaning", "unit", "formula",
    "computation_note", "task_scope", "aggregation", "required_inputs",
    "normalization", "evidence_tier", "implementation_status",
    "interpretation", "source_document", "source_location",
)

MAIN_HANDOFF_FILENAMES = (
    "feature_values.csv",
    "feature_registry.csv",
    "feature_status.csv",
    "feature_export_manifest.json",
)


def normalize_feature_registry(registry: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Return a registry with the common cross-modal columns and stable order."""
    out = registry.copy()
    aliases = {
        "subsystem": "family",
        "group": "family",
        "definition": "meaning",
        "tier": "evidence_tier",
        "status": "implementation_status",
    }
    for source, target in aliases.items():
        if target not in out.columns and source in out.columns:
            out[target] = out[source]
    out["modality"] = modality
    for col in FEATURE_REGISTRY_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    leading = list(FEATURE_REGISTRY_COLUMNS)
    return out.loc[:, leading + [c for c in out.columns if c not in leading]]


def write_feature_handoff(
    output_dir: Path | str,
    values: pd.DataFrame,
    registry: pd.DataFrame,
    status: pd.DataFrame,
    modality: str,
    source_values: Path | str,
    source_registry: Path | str,
) -> dict[str, Path]:
    """Write canonical aliases consumed by the Feature GUI without renaming legacy outputs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    values_path = out / "feature_values.csv"
    registry_path = out / "feature_registry.csv"
    status_path = out / "feature_status.csv"
    manifest_path = out / "feature_export_manifest.json"
    if status.empty and not len(status.columns):
        status = pd.DataFrame(columns=("recording_id", "task", "modality", "feature", "status", "note"))
    values.to_csv(values_path, index=False)
    normalize_feature_registry(registry, modality).to_csv(registry_path, index=False)
    status.to_csv(status_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "contract": "vslp-feature-handoff",
                "contract_version": "1.0.0",
                "modality": modality,
                "row_granularity": "one row per source recording and task",
                "canonical_outputs": {
                    "feature_values": str(values_path),
                    "feature_registry": str(registry_path),
                    "feature_status": str(status_path),
                },
                "legacy_sources": {
                    "feature_values": str(source_values),
                    "feature_registry": str(source_registry),
                },
                "warning": "Identifiers and QC fields are context, not default model predictors.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "feature_values_csv": values_path,
        "feature_registry_csv": registry_path,
        "feature_status_csv": status_path,
        "feature_export_manifest_json": manifest_path,
    }


def build_feature_delivery(
    output_root: Path | str,
    modality: str,
    handoff: dict[str, Path | str],
    optional_main: dict[str, Path | str] | None = None,
) -> dict[str, Path]:
    """Create a human-facing main/supplementary delivery without moving stage outputs."""
    root = Path(output_root).expanduser().resolve()
    modality_folder = {"kinematic": "kinematics"}.get(modality, modality)
    modality_root = root / modality_folder
    delivery = modality_root / "feature_handoff"
    main_dir = delivery / "main"
    supplementary_dir = delivery / "supplementary"
    main_dir.mkdir(parents=True, exist_ok=True)
    supplementary_dir.mkdir(parents=True, exist_ok=True)

    source_map = {
        "feature_values.csv": Path(handoff["feature_values_csv"]),
        "feature_registry.csv": Path(handoff["feature_registry_csv"]),
        "feature_status.csv": Path(handoff["feature_status_csv"]),
        "feature_export_manifest.json": Path(handoff["feature_export_manifest_json"]),
    }
    for name, source in (optional_main or {}).items():
        source_map[name] = Path(source)

    copied: dict[str, str] = {}
    unavailable: dict[str, str] = {}
    for name, source in source_map.items():
        if source.exists() and source.is_file():
            target = main_dir / name
            temporary = target.with_suffix(target.suffix + ".part")
            shutil.copy2(source, temporary)
            temporary.replace(target)
            copied[name] = str(target.relative_to(root))
        else:
            unavailable[name] = str(source)

    main_readme = main_dir / "README.md"
    main_readme.write_text(
        "# Main downstream outputs\n\n"
        "Use this folder when loading the modality into the VSLP Feature GUI.\n\n"
        "- `feature_values.csv`: one recording-task row with context, QC, and feature values.\n"
        "- `feature_registry.csv`: formula, unit, scope, aggregation, evidence, and implementation metadata.\n"
        "- `feature_status.csv`: per-recording feature availability and warnings.\n"
        "- `feature_export_manifest.json`: machine-readable schema and provenance.\n"
        "- `qc_features.csv`: optional modality QC covariates when available.\n"
        "- `metadata_context.csv`: optional mapped recording context when available.\n\n"
        "Do not use identifiers, paths, or QC fields as disease predictors by default.\n",
        encoding="utf-8",
    )

    main_source_paths = {path.resolve() for path in source_map.values() if path.exists()}
    rows: list[dict[str, object]] = []
    for path in sorted(modality_root.rglob("*")):
        if not path.is_file() or delivery in path.parents:
            continue
        rel = path.relative_to(root)
        rows.append({
            "importance": "main_source" if path.resolve() in main_source_paths else "supplementary",
            "artifact": path.name,
            "type": path.suffix.lower().lstrip(".") or "file",
            "relative_path": str(rel),
            "size_bytes": path.stat().st_size,
            "purpose": "canonical downstream input" if path.resolve() in main_source_paths else "stage-specific audit, diagnostic, report, plot, log, or intermediate",
        })
    catalog = pd.DataFrame(rows, columns=("importance", "artifact", "type", "relative_path", "size_bytes", "purpose"))
    catalog_csv = supplementary_dir / "artifact_catalog.csv"
    catalog.to_csv(catalog_csv, index=False)
    supplementary_readme = supplementary_dir / "README.md"
    supplementary_readme.write_text(
        "# Supplementary outputs\n\n"
        "`artifact_catalog.csv` indexes the modality-specific stage outputs in their original provenance-preserving folders. "
        "These files support QC review, troubleshooting, plots, reports, and method audit. They are not the default Feature GUI input.\n",
        encoding="utf-8",
    )

    manifest_path = delivery / "delivery_manifest.json"
    manifest_path.write_text(json.dumps({
        "contract": "vslp-modality-feature-delivery",
        "contract_version": "1.0.0",
        "modality": modality,
        "main_directory": str(main_dir.relative_to(root)),
        "supplementary_directory": str(supplementary_dir.relative_to(root)),
        "main_outputs": copied,
        "optional_outputs_unavailable": unavailable,
        "supplementary_catalog": str(catalog_csv.relative_to(root)),
    }, indent=2), encoding="utf-8")
    return {
        "delivery_root": delivery,
        "main_dir": main_dir,
        "supplementary_dir": supplementary_dir,
        "catalog_csv": catalog_csv,
        "delivery_manifest_json": manifest_path,
    }
