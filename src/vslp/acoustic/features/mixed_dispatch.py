"""Registry-driven execution and one authoritative handoff for selected families."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


def _executors():
    from vslp.acoustic.features.family01_stage import run_family01_stage
    from vslp.acoustic.features.family04_stage import run_family04_stage
    from vslp.acoustic.features.family06_stage import run_family06_stage
    from vslp.acoustic.features.family08 import run_reviewed_timing_stage
    from vslp.acoustic.features.family10 import run_family10_stage
    from vslp.acoustic.features.family11_12_stage import run_family11_12_stage

    # Executor grouping is deliberate: F01/F02 share audio and voicing work;
    # F07/F08/F09 share reviewed timing; F11/F12 share waveform framing.
    return {
        "voice": ({"F01", "F02"}, run_family01_stage),
        "formants": ({"F04"}, run_family04_stage),
        "segmental": ({"F06"}, run_family06_stage),
        "timing": ({"F07", "F08", "F09"}, run_reviewed_timing_stage),
        "ddk": ({"F10"}, run_family10_stage),
        "spectrum": ({"F11", "F12"}, run_family11_12_stage),
    }


def _record_context(row, task_id: str, task_type: str) -> dict:
    return {"recording_id": str(row.recording_id), "file_name": str(row.file_name),
            "task_id": task_id, "task_type": task_type,
            "segmentation_run_id": str(row.segmentation_run_id),
            "review_run_id": str(row.review_run_id)}


def run_mixed_feature_stage(decisions_csv: str | Path, root: str | Path, config,
                            intervals_csv: str | Path, registry: pd.DataFrame,
                            progress_callback=None) -> StageResult:
    """Resolve each approved ID once, run each executor once, merge selected results."""
    import json

    root = Path(root)
    stage = root / "acoustic" / "005_features"
    tables = stage / "tables"
    decisions, intervals = Path(decisions_csv), Path(intervals_csv)
    kept = load_final_segmentation(intervals, decisions)
    project = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    from vslp.acoustic.features.catalog import task_key

    task_id = str(project.get("task_id") or task_key(str(project.get("task_name", ""))) or "")
    task_type = str(project.get("task_type", ""))
    selected = list(config.selected_features)
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate selected feature ID")
    lookup = registry.set_index("feature")
    if not lookup.index.is_unique:
        raise ValueError("Duplicate feature ID in registry snapshot")
    if set(selected) != set(lookup.index):
        raise ValueError("Selected feature IDs do not match the approved registry snapshot")
    executors = _executors()
    family_to_group = {}
    for group, (families, _) in executors.items():
        for family in families:
            if family in family_to_group:
                raise ValueError(f"Ambiguous executor for {family}")
            family_to_group[family] = group
    groups: dict[str, list[str]] = {}
    for feature_id in selected:
        family = str(lookup.loc[feature_id, "family_id"])
        if family not in family_to_group:
            raise ValueError(f"No executor for {feature_id} ({family})")
        groups.setdefault(family_to_group[family], []).append(feature_id)

    total_units = len(kept) * len(groups)
    if progress_callback:
        progress_callback(0, total_units, "Acoustic Features")
    rows: list[dict] = []
    from vslp.acoustic.features.formant_service import FormantService

    formant_service = FormantService()
    family_manifests: dict[str, str] = {}
    family_audits: list[ArtifactRef] = []
    completed_units = 0
    for group, ids in groups.items():
        group_registry = registry.loc[registry.feature.isin(ids)].copy()
        group_config = replace(config, selected_features=ids)
        group_dir = stage / "family_runs" / group
        families, executor = executors[group]

        def report(done: int, total: int, message: str) -> None:
            if progress_callback:
                progress_callback(completed_units + min(done, len(kept)), total_units, message)

        try:
            extra = {"formant_service": formant_service} if group == "formants" else {}
            result = executor(decisions, root, group_config, intervals, group_registry,
                              progress_callback=report, execution_stage_dir=group_dir,
                              **extra)
            family_manifests[group] = str(result.manifest_path)
            family_manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
            for artifact in family_manifest.get("output_artifacts", []):
                path = artifact.get("path", "")
                if path and Path(path).is_file() and "native_measurements" in Path(path).parts:
                    family_audits.append(ArtifactRef(path, artifact.get("role", "family_audit"),
                                                     artifact.get("media_type", "application/octet-stream")))
            status_path = group_dir / "tables" / "acoustic_feature_status_long.csv"
            family_rows = pd.read_csv(status_path, keep_default_na=False).to_dict("records")
            for item in family_rows:
                feature_id = str(item.get("feature_id") or item.get("feature") or "")
                if feature_id not in ids:
                    raise ValueError(f"Executor {group} emitted unselected output {feature_id}")
                item["feature_id"] = feature_id
                item["family_id"] = str(lookup.loc[feature_id, "family_id"])
                item["task_id"] = task_id
                item["task_type"] = task_type
                item["unit"] = str(lookup.loc[feature_id, "unit"])
                item["algorithm_version"] = str(item.get("algorithm_version") or
                                                lookup.loc[feature_id, "algorithm_version"])
                item["algorithm"] = str(item.get("algorithm") or item["algorithm_version"])
                item["parameter_set_id"] = str(item.get("parameter_set_id") or
                                               lookup.loc[feature_id, "parameter_set_id"])
                item["failure_reason"] = str(item.get("failure_reason") or item.get("reason") or "")
                item["provenance_ref"] = str(result.manifest_path)
                native_name = str(item.get("native_measurement_table") or "")
                item["native_measurement_path"] = (
                    str(group_dir / "tables" / "native_measurements" / native_name)
                    if native_name else "")
                rows.append(item)
        except Exception as exc:  # an executor failure must not erase unrelated families
            reason = f"family_executor_failed:{type(exc).__name__}:{str(exc).splitlines()[0]}"
            family_manifests[group] = reason
            for record in kept.itertuples():
                for feature_id in ids:
                    rows.append({**_record_context(record, task_id, task_type),
                                 "feature_id": feature_id, "family_id": str(lookup.loc[feature_id, "family_id"]),
                                 "value": np.nan, "unit": str(lookup.loc[feature_id, "unit"]),
                                 "status": "unavailable", "failure_reason": reason,
                                 "algorithm": str(lookup.loc[feature_id, "algorithm_version"]),
                                 "algorithm_version": str(lookup.loc[feature_id, "algorithm_version"]),
                                 "parameter_set_id": str(lookup.loc[feature_id, "parameter_set_id"]),
                                 "provenance_ref": ""})
        completed_units += len(kept)
        if progress_callback:
            progress_callback(completed_units, total_units, f"Acoustic Features — {group}")

    long = pd.DataFrame(rows)
    if not long.empty and long.duplicated(["recording_id", "feature_id"]).any():
        raise ValueError("Duplicate recording/feature result from family executors")
    expected = {(str(record.recording_id), feature_id)
                for record in kept.itertuples() for feature_id in selected}
    produced = set(zip(long.recording_id.astype(str), long.feature_id.astype(str))) if not long.empty else set()
    for record in kept.itertuples():
        for feature_id in selected:
            if (str(record.recording_id), feature_id) not in produced:
                long = pd.concat([long, pd.DataFrame([{**_record_context(record, task_id, task_type),
                    "feature_id": feature_id, "family_id": str(lookup.loc[feature_id, "family_id"]),
                    "value": np.nan, "unit": str(lookup.loc[feature_id, "unit"]),
                    "status": "unavailable", "failure_reason": "executor_missing_result",
                    "algorithm": str(lookup.loc[feature_id, "algorithm_version"]),
                    "algorithm_version": str(lookup.loc[feature_id, "algorithm_version"]),
                    "parameter_set_id": str(lookup.loc[feature_id, "parameter_set_id"]),
                    "provenance_ref": ""}])], ignore_index=True)
    assert len(long) == len(expected)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.sort_values(["recording_id", "feature_id"]).reset_index(drop=True)
    base = pd.DataFrame([_record_context(record, task_id, task_type) for record in kept.itertuples()])
    scalar_selected = [feature_id for feature_id in selected
                       if str(lookup.loc[feature_id, "output_granularity"]) == "recording"]
    if not base.empty:
        pivot = long.loc[long.feature_id.isin(scalar_selected)].pivot(
            index="recording_id", columns="feature_id", values="value")
        wide = base.merge(pivot, left_on="recording_id", right_index=True, how="left")
        wide = wide.reindex(columns=[*base.columns, *scalar_selected])
    else:
        wide = base.reindex(columns=[*base.columns, *scalar_selected])
    # Preserve the established DDK denominator audit beside its scalar values.
    if "F10" in set(registry.family_id.astype(str)):
        ddk_context = ("n_ddk_events", "n_valid_sequences", "raw_analysis_duration_sec",
                       "excluded_contamination_duration_sec", "effective_ddk_duration_sec")
        ddk_rows = long.loc[long.family_id.eq("F10")]
        for column in ddk_context:
            if column in ddk_rows:
                by_record = ddk_rows.groupby("recording_id")[column].first()
                wide[column] = wide.recording_id.map(by_record)
    tables.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    wide.to_csv(values_path, index=False)
    long.to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    handoff = write_feature_handoff(tables, wide, registry, long,
                                    "acoustic", values_path, registry_path)
    build_feature_delivery(root, "acoustic", handoff)
    failures = int(long.failure_reason.astype(str).ne("").sum())
    success = len(long) - failures
    from vslp.acoustic.features.catalog import load_feature_catalog
    from vslp.acoustic.alignment import load_final_alignment

    approved = {item["feature_id"]: item for item in load_feature_catalog()["outputs"]}
    alignment, _alignment_issue = load_final_alignment(root)
    alignment_provenance = (alignment.manifest if alignment is not None else {})
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version="mixed-family-dispatch-1.0.0",
        status="completed_with_warnings" if failures else "completed",
        input_artifacts=[ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
                         ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")],
        output_artifacts=[ArtifactRef(str(values_path), "feature_values", "text/csv"),
                          ArtifactRef(str(status_path), "feature_status", "text/csv"),
                          ArtifactRef(str(registry_path), "feature_registry", "text/csv"),
                          *family_audits],
        config={"selected_feature_ids": selected,
                "families_invoked": sorted(set(registry.family_id.astype(str))),
                "executor_groups": list(groups), "family_manifests": family_manifests,
                "shared_formant_tracks_cached": formant_service.cached_tracks,
                "successful_outputs": success, "failed_outputs": failures,
                "parameter_set_ids": dict(zip(registry.feature, registry.parameter_set_id)),
                "prerequisites": {feature_id: approved[feature_id]["prerequisites"]
                                  for feature_id in selected},
                "alignment_run_id": alignment_provenance.get("alignment_run_id", ""),
                "alignment_words_sha256": alignment_provenance.get("final_words_sha256", ""),
                "alignment_phones_sha256": alignment_provenance.get("final_phones_sha256", ""),
                "source_segmentation_decisions_sha256": sha256_file(decisions),
                "source_segmentation_intervals_sha256": sha256_file(intervals)})
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
