"""F03 sustained-/a/ stage from final reviewed segmentation and trials."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.alignment.trials import confirmed_task_trials
from vslp.acoustic.features.family01_stage import _region_tracks
from vslp.acoustic.features.family01_task import task_context
from vslp.acoustic.features.family03 import (
    ALGORITHM_VERSION, BLOCKAGE_IDS, FAMILY03_IDS, GIF_IDS, PARAMETER_SET_ID,
    TREMOR_IDS, amplitude_tremor_trial, glottal_trial, vibration_blockage,
)
from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


def run_family03_stage(decisions_csv: str | Path, root: str | Path, config,
                       intervals_csv: str | Path, registry: pd.DataFrame,
                       progress_callback=None,
                       execution_stage_dir: Path | None = None) -> StageResult:
    root = Path(root)
    final = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    if (Path(decisions_csv).resolve() != decisions.resolve() or
            Path(intervals_csv).resolve() != intervals.resolve()):
        raise ValueError("F03 requires frozen reviewed segmentation")
    kept = load_final_segmentation(intervals, decisions)
    timeline = pd.read_csv(intervals, keep_default_na=False)
    project = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    task = task_context(project)
    selected = set(config.selected_features) & FAMILY03_IDS
    stage = execution_stage_dir or root / "acoustic" / "005_features"
    tables = stage / "tables"
    native = tables / "native_measurements"
    native.mkdir(parents=True, exist_ok=True)
    values, statuses, trial_rows, cycle_rows = [], [], [], []
    if progress_callback:
        progress_callback(0, len(kept), "F03 glottal-source review build")
    for index, row in enumerate(kept.itertuples(), 1):
        recording_id = str(row.recording_id)
        result = {feature: (np.nan, "analysis_unavailable") for feature in selected}
        sample_rate = np.nan
        try:
            if task["task_family"] != "sustained_vowel" or task["task_item"] != "a":
                raise ValueError("requires_reviewed_sustained_a")
            audio, sample_rate = sf.read(row.analysis_wav_path, dtype="float32",
                                         always_2d=False)
            if audio.ndim != 1 or sample_rate < 16000:
                raise ValueError("invalid_native_gif_audio")
            trials = confirmed_task_trials(root, recording_id)
            current = timeline.loc[timeline.recording_id.astype(str).eq(recording_id)]
            _, _, spans = _region_tracks(row, current, audio, sample_rate,
                                         task["task_family"], trials)
            groups = trials or [{"trial_id": "", "start_sec": 0.,
                                 "end_sec": len(audio) / sample_rate}]
            by_trial: list[dict[str, tuple[float, str]]] = []
            for trial in groups:
                trial_id = str(trial["trial_id"])
                parts = [track for track, identity, *_ in spans if identity == trial_id]
                trial_values = {feature: (np.nan, "no_eligible_trial_frames")
                                for feature in selected}
                audit = {}
                if parts:
                    if selected & (GIF_IDS | BLOCKAGE_IDS):
                        measured, cycles, gif_audit = glottal_trial(
                            parts, audio, sample_rate, selected & GIF_IDS)
                        trial_values.update(measured)
                        audit.update(gif_audit)
                        cycle_rows.extend({"recording_id": recording_id,
                                           "task_family": task["task_family"],
                                           "task_item": task["task_item"],
                                           "trial_id": trial_id,
                                           "gif_method": "qcp",
                                           "source_equivalent": False,
                                           **cycle} for cycle in cycles)
                    if selected & TREMOR_IDS:
                        # The final reviewed /a/ region is contiguous and <=2 s.
                        track = parts[0]
                        first = round(track.analysis_start_sec * sample_rate)
                        last = round(track.analysis_end_sec * sample_rate)
                        tremor, tremor_audit = amplitude_tremor_trial(
                            audio[first:last], sample_rate)
                        trial_values.update({key: tremor[key]
                                             for key in selected & TREMOR_IDS})
                        audit.update({f"tremor_{key}": value
                                      for key, value in tremor_audit.items()})
                    if selected & BLOCKAGE_IDS:
                        track = parts[0]
                        first = round(track.analysis_start_sec * sample_rate)
                        last = round(track.analysis_end_sec * sample_rate)
                        segment = audio[first:last]
                        pulses = np.asarray(audit.get("pulse_times_sec", []), dtype=float)
                        if not audit.get("valid_cycle_intervals"):
                            blockage, reason, blockage_audit = (
                                np.nan, "gif_cycles_unavailable", {})
                        else:
                            blockage, reason, blockage_audit = vibration_blockage(
                                track, segment, sample_rate, pulses)
                        trial_values["vibration_blockage_pct"] = (blockage, reason)
                        audit.update(blockage_audit)
                by_trial.append(trial_values)
                trial_rows.append({
                    "recording_id": recording_id, "task_family": task["task_family"],
                    "task_item": task["task_item"], "trial_id": trial_id,
                    "analysis_start_sec": min((p.analysis_start_sec for p in parts),
                                              default=np.nan),
                    "analysis_end_sec": max((p.analysis_end_sec for p in parts),
                                            default=np.nan),
                    "analysis_duration_sec": sum(p.analysis_end_sec - p.analysis_start_sec
                                                 for p in parts),
                    "native_sample_rate_hz": sample_rate,
                    "gif_method": audit.get("gif_method", "qcp"),
                    "gif_algorithm_version": audit.get("gif_algorithm_version", ""),
                    "gif_parameter_set_id": audit.get("gif_parameter_set_id", ""),
                    "internal_sample_rate_hz": audit.get("internal_sample_rate_hz", np.nan),
                    "n_candidate_cycles": audit.get("n_candidate_cycles", 0),
                    "n_valid_cycles": audit.get("n_valid_cycles", 0),
                    "valid_cycle_fraction": audit.get("valid_cycle_fraction", 0.),
                    "cycle_event_definition": audit.get("cycle_event_definition", ""),
                    "modulation_sample_rate_hz": audit.get("tremor_modulation_sample_rate_hz", np.nan),
                    "n_tremor_windows": audit.get("tremor_n_windows", 0),
                    "dominant_tremor_frequency_hz": audit.get(
                        "tremor_dominant_tremor_frequency_hz", np.nan),
                    "blockage_count": audit.get("blockage_count", 0),
                    "qc_flags": json.dumps(audit.get("qc_flags", [])),
                    "feature_failures": json.dumps({feature: trial_values[feature][1]
                                                    for feature in selected
                                                    if trial_values[feature][1]}, sort_keys=True),
                    "aggregation_method": "median_valid_frames_or_cycles_within_trial",
                    "source_equivalent": False,
                    "vslp_adaptation_notes": "centered <=2-s stable /a/",
                    **{feature: trial_values[feature][0] for feature in selected},
                })
            for feature in selected:
                good = [value[feature][0] for value in by_trial
                        if not value[feature][1] and np.isfinite(value[feature][0])]
                reason = next((value[feature][1] for value in by_trial
                               if value[feature][1]), "no_valid_trial_measurement")
                result[feature] = ((float(np.median(good)), "") if good else
                                   (np.nan, reason))
        except (OSError, ValueError, RuntimeError) as exc:
            result = {feature: (np.nan, str(exc).splitlines()[0]) for feature in selected}
        base = {"recording_id": recording_id, "file_name": row.file_name,
                "task_id": project.get("task_id", ""),
                "task_type": project.get("task_type", ""),
                "task_family": task["task_family"], "task_item": task["task_item"],
                "run_id": project["run_id"], "source_sha256": row.source_sha256,
                "source_file_path": row.source_file_path,
                "analysis_wav_path": row.analysis_wav_path,
                "segmentation_run_id": row.segmentation_run_id,
                "review_run_id": row.review_run_id,
                "native_sample_rate_hz": sample_rate,
                "final_decisions_sha256": sha256_file(decisions),
                "final_intervals_sha256": sha256_file(intervals)}
        values.append({**base, **{feature: result[feature][0] for feature in selected}})
        for feature in sorted(selected):
            spec = registry.loc[registry.feature.eq(feature)].iloc[0]
            value, reason = result[feature]
            statuses.append({**base, "feature": feature, "feature_id": feature,
                             "family_id": "F03", "value": value,
                             "unit": spec.unit, "status": "unavailable" if reason else "computed",
                             "failure_reason": reason,
                             "algorithm": ALGORITHM_VERSION,
                             "algorithm_version": ALGORITHM_VERSION,
                             "parameter_set_id": PARAMETER_SET_ID,
                             "source_method_class": spec.implementation_class,
                             "source_equivalent": False,
                             "source_reference": "VCS concept; open VSLP estimator",
                             "aggregation_method": "median_valid_trials_v1",
                             "trial_measurement_table": "family03_trial_summaries.csv",
                             "native_measurement_table": "family03_cycle_measurements.csv"
                             if feature in GIF_IDS else "family03_trial_summaries.csv"})
        if progress_callback:
            progress_callback(index, len(kept), "F03 sustained-vowel measurements")
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    trial_path = native / "family03_trial_summaries.csv"
    cycle_path = native / "family03_cycle_measurements.csv"
    pd.DataFrame(values).to_csv(values_path, index=False)
    pd.DataFrame(statuses).to_csv(status_path, index=False)
    pd.DataFrame(trial_rows).to_csv(trial_path, index=False)
    pd.DataFrame(cycle_rows).to_csv(cycle_path, index=False)
    registry.to_csv(registry_path, index=False)
    manifest = StageManifest(
        stage_name="acoustic_features_f03", stage_version=ALGORITHM_VERSION,
        status="completed_with_warnings" if any(s["failure_reason"] for s in statuses)
        else "completed",
        input_artifacts=[ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
                         ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")],
        output_artifacts=[ArtifactRef(str(values_path), "feature_values", "text/csv"),
                          ArtifactRef(str(status_path), "feature_status", "text/csv"),
                          ArtifactRef(str(trial_path), "family03_trial_summaries", "text/csv"),
                          ArtifactRef(str(cycle_path), "family03_cycle_measurements", "text/csv")],
        config={"selected_features": sorted(selected),
                "task_family": task["task_family"], "task_item": task["task_item"],
                "source_equivalent": False,
                "aggregation_method": "median valid trials",
                "vslp_adaptation_notes": "centered <=2-s reviewed stable /a/"})
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
