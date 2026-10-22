"""F10 executor over the explicit final reviewed DDK-event artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vslp.acoustic.features.family10_v2 import (
    AUDIT_COLUMNS, PARAMETER_SET_ID, PUBLIC_IDS, RATE_IDS, TARGETS, VARIABILITY_IDS,
    VERSION, load_reviewed_ddk_events, measure_trial,
)
from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


def run_family10_stage(decisions_csv: str | Path, root: str | Path, config,
                       intervals_csv: str | Path | None, registry: pd.DataFrame,
                       progress_callback=None, execution_stage_dir: Path | None = None,
                       recording_scope=None) -> StageResult:
    root = Path(root)
    final = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    if (Path(decisions_csv).resolve() != decisions.resolve() or intervals_csv is None
            or Path(intervals_csv).resolve() != intervals.resolve()):
        raise ValueError("Family 10 requires frozen reviewed DDK context")
    kept = (recording_scope.kept if recording_scope is not None else
            load_final_segmentation(intervals, decisions))
    event_path = final / "final_ddk_events.csv"
    project = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    all_events, event_issue = load_reviewed_ddk_events(event_path)
    from vslp.acoustic.features.catalog import task_key

    task_id = str(project.get("task_id") or task_key(str(project.get("task_name", ""))) or "")
    if task_id.casefold() != "ddk":
        all_events, event_issue = pd.DataFrame(), "unsupported_ddk_target"
    selected = set(config.selected_features) & PUBLIC_IDS
    specs = registry.set_index("feature")
    stage = execution_stage_dir or root / "acoustic" / "005_features"
    tables = stage / "tables"
    native = tables / "native_measurements"
    values_rows, status_rows, trial_rows, audit_rows = [], [], [], []
    for index, record in enumerate(kept.itertuples(), 1):
        recording_id = str(record.recording_id)
        own = all_events.loc[all_events.recording_id.astype(str).eq(recording_id)] if (
            not event_issue) else pd.DataFrame()
        base = {"recording_id": recording_id, "file_name": str(record.file_name),
                "task_id": "ddk", "algorithm_version": VERSION,
                "parameter_set_id": PARAMETER_SET_ID,
                "input_contract": "FINAL_DDK_EVENTS",
                "final_decisions_sha256": sha256_file(decisions),
                "final_intervals_sha256": sha256_file(intervals),
                "final_ddk_events_sha256": sha256_file(event_path) if not event_issue else ""}
        values = dict(base)
        for target, mode in TARGETS.items():
            rate_id, variability_id = RATE_IDS[target], VARIABILITY_IDS[target]
            requested = {rate_id, variability_id} & selected
            if not requested:
                continue
            target_rows = own.loc[own.ddk_target.astype(str).str.casefold().str.strip("/").eq(target)] if (
                not own.empty) else pd.DataFrame()
            groups = list(target_rows.groupby("trial_id", sort=False)) if not target_rows.empty else []
            for trial_id, trial_events in groups:
                result, audit, summary = measure_trial(trial_events, target)
                audit_rows.extend(audit.to_dict("records"))
                trial_rows.append({**base, "trial_id": str(trial_id), "ddk_target": target,
                                   "ddk_mode": mode, **summary,
                                   rate_id: result["rate"][0],
                                   f"{rate_id}_reason": result["rate"][1],
                                   variability_id: result["variability"][0],
                                   f"{variability_id}_reason": result["variability"][1]})
            for feature_id, key in ((rate_id, "rate"), (variability_id, "variability")):
                if feature_id not in requested:
                    continue
                candidates = [row for row in trial_rows if row["recording_id"] == recording_id
                              and row["ddk_target"] == target]
                valid = [float(row[feature_id]) for row in candidates if (
                    np.isfinite(row[feature_id]) and not row[f"{feature_id}_reason"])]
                reason = "" if valid else (event_issue or next(
                    (row[f"{feature_id}_reason"] for row in candidates
                     if row[f"{feature_id}_reason"]),
                    "missing_reviewed_ddk_events" if own.empty else "unsupported_ddk_target"))
                value = float(np.median(valid)) if valid else np.nan
                values[feature_id] = value
                status_rows.append({**base, "feature": feature_id, "feature_id": feature_id,
                                    "family_id": "F10", "value": value,
                                    "status": "computed" if not reason else "unavailable",
                                    "reason": reason, "failure_reason": reason,
                                    "unit": str(specs.loc[feature_id, "unit"]),
                                    "ddk_target": target, "ddk_mode": mode,
                                    "trial_count": len(candidates),
                                    "valid_trial_count": len(valid)})
        values_rows.append(values)
        if progress_callback:
            progress_callback(index, len(kept), f"Acoustic Features — {record.file_name}")
    native.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    trial_path = native / "family10_trial_values.csv"
    audit_path = native / "family10_selected_events.csv"
    values_df, status_df = pd.DataFrame(values_rows), pd.DataFrame(status_rows)
    values_df.to_csv(values_path, index=False)
    status_df.to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    pd.DataFrame(trial_rows).to_csv(trial_path, index=False)
    pd.DataFrame(audit_rows).reindex(columns=AUDIT_COLUMNS).to_csv(audit_path, index=False)
    if execution_stage_dir is None:
        handoff = write_feature_handoff(tables, values_df, registry, status_df,
                                        "acoustic", values_path, registry_path)
        build_feature_delivery(root, "acoustic", handoff,
                               optional_main={trial_path.name: trial_path,
                                              audit_path.name: audit_path})
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version=VERSION,
        status="completed_with_warnings" if status_df.failure_reason.astype(str).ne("").any()
        else "completed",
        input_artifacts=[ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
                         ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")]
        + ([ArtifactRef(str(event_path), "final_ddk_events", "text/csv")]
           if not event_issue else []),
        output_artifacts=[ArtifactRef(str(values_path), "feature_values", "text/csv"),
                          ArtifactRef(str(status_path), "feature_status", "text/csv"),
                          ArtifactRef(str(trial_path), "family10_trial_values", "text/csv"),
                          ArtifactRef(str(audit_path), "family10_selected_events", "text/csv")],
        config={"selected_features": sorted(selected), "input_contract": "FINAL_DDK_EVENTS",
                "analysis_target_n": 10, "event_issue": event_issue,
                "rate_denominator": "first_accepted_onset_to_tenth_accepted_offset"})
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
