"""Trial-first Bamboo F09 executor with a final-reviewed-timing prerequisite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vslp.acoustic.features.catalog import task_key
from vslp.acoustic.features.family07_v2 import bamboo_reviewed_trial_domains
from vslp.acoustic.features.family09_v2 import (
    EVENT_COLUMNS, PUBLIC_IDS, VERSION, reviewed_pause_events,
)
from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


def _trial_domains(root: Path, recording_id: str, own: pd.DataFrame) -> tuple[list[dict], str]:
    return bamboo_reviewed_trial_domains(root, recording_id, own)


def _clip(own: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    frame = own.copy()
    frame["start_sec"] = frame.start_sec.astype(float).clip(lower=start)
    frame["end_sec"] = frame.end_sec.astype(float).clip(upper=end)
    return frame.loc[frame.end_sec.gt(frame.start_sec)].copy()


def run_family09_stage(decisions_csv: str | Path, root: str | Path, config,
                       intervals_csv: str | Path | None, registry: pd.DataFrame,
                       progress_callback=None, execution_stage_dir: Path | None = None) -> StageResult:
    root = Path(root)
    final = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    if (intervals_csv is None or Path(intervals_csv).resolve() != intervals.resolve()
            or Path(decisions_csv).resolve() != decisions.resolve()):
        raise ValueError("Family 09 requires frozen reviewed segmentation")
    kept = load_final_segmentation(intervals, decisions)
    timeline = pd.read_csv(intervals, keep_default_na=False)
    project = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    task_id = str(project.get("task_id") or task_key(str(project.get("task_name", ""))) or "")
    selected = set(config.selected_features) & PUBLIC_IDS
    specs = registry.set_index("feature")
    stage = execution_stage_dir or root / "acoustic" / "005_features"
    tables = stage / "tables"
    native = tables / "native_measurements"
    values_rows, status_rows, trial_rows, event_rows = [], [], [], []
    for done, record in enumerate(kept.itertuples(), 1):
        recording_id = str(record.recording_id)
        own = timeline.loc[timeline.recording_id.astype(str).eq(recording_id)].copy()
        own = own.loc[own["view"].astype(str).eq("authoritative")] if "view" in own else own
        trials, trial_issue = (_trial_domains(root, recording_id, own)
                               if task_id == "bamboo_passage" else ([], "task_not_applicable"))
        for trial in trials:
            start, end = float(trial["start_sec"]), float(trial["end_sec"])
            events, results, support = reviewed_pause_events(
                _clip(own, start, end), recording_id, str(trial["trial_id"]))
            if not events.empty:
                event_rows.extend(events.to_dict("records"))
            trial_rows.append({"recording_id": recording_id,
                               "trial_id": str(trial["trial_id"]),
                               "sequence_id": str(trial["sequence_id"]),
                               "task_id": task_id, "trial_start_sec": start,
                               "trial_end_sec": end, **support,
                               **{feature: value for feature, (value, _) in results.items()},
                               **{f"{feature}_reason": reason
                                  for feature, (_, reason) in results.items()}})
        base = {"recording_id": recording_id, "file_name": str(record.file_name),
                "task_id": task_id, "algorithm_version": VERSION,
                "final_decisions_sha256": sha256_file(decisions),
                "final_intervals_sha256": sha256_file(intervals)}
        value_row = dict(base)
        for feature in sorted(selected):
            valid = [float(row[feature]) for row in trial_rows
                     if row["recording_id"] == recording_id
                     and np.isfinite(row[feature]) and not row[f"{feature}_reason"]]
            reason = ("" if valid else next((row[f"{feature}_reason"]
                                            for row in trial_rows if row["recording_id"] == recording_id
                                            and row[f"{feature}_reason"]),
                                           trial_issue or "missing_reviewed_segmentation"))
            value = float(np.median(valid)) if valid else np.nan
            value_row[feature] = value
            status_rows.append({**base, "feature": feature, "feature_id": feature,
                                "family_id": "F09", "value": value,
                                "status": "computed" if not reason else "unavailable",
                                "failure_reason": reason, "unit": str(specs.loc[feature, "unit"]),
                                "input_contract": "FINAL_REVIEWED_TIMING",
                                "trial_count": len(trials), "valid_trial_count": len(valid)})
        values_rows.append(value_row)
        if progress_callback:
            progress_callback(done, len(kept), f"Acoustic Features — {record.file_name}")
    native.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    event_path = native / "family09_pause_phrase_events.csv"
    trial_path = native / "family09_trial_values.csv"
    values_df, status_df = pd.DataFrame(values_rows), pd.DataFrame(status_rows)
    values_df.to_csv(values_path, index=False)
    status_df.to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    pd.DataFrame(event_rows).reindex(columns=EVENT_COLUMNS).to_csv(event_path, index=False)
    pd.DataFrame(trial_rows).to_csv(trial_path, index=False)
    if execution_stage_dir is None:
        handoff = write_feature_handoff(tables, values_df, registry, status_df,
                                        "acoustic", values_path, registry_path)
        build_feature_delivery(root, "acoustic", handoff,
                               optional_main={event_path.name: event_path, trial_path.name: trial_path})
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version=VERSION,
        status="completed_with_warnings" if status_df.failure_reason.astype(str).ne("").any()
        else "completed",
        input_artifacts=[ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
                         ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")],
        output_artifacts=[ArtifactRef(str(values_path), "feature_values", "text/csv"),
                          ArtifactRef(str(status_path), "feature_status", "text/csv"),
                          ArtifactRef(str(event_path), "family09_pause_phrase_events", "text/csv"),
                          ArtifactRef(str(trial_path), "family09_trial_values", "text/csv")],
        config={"selected_features": sorted(selected), "pause_threshold_sec": 0.300,
                "input_contract": "FINAL_REVIEWED_TIMING"})
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
