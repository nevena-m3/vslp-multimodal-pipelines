"""Acoustic feature extraction stage.

V0.7/V0.8 introduces a plugin architecture and computes the first validated-safe
feature families:
- respiratory/timing features from Silero segments;
- rhythm/envelope modulation features from canonical segmentation WAVs;
- phonatory F0/HNR/CPP plus Praat PointProcess cycle perturbation measures;
- global RMS amplitude.

Features still requiring formula-level validation against the uploaded notebook remain
registered, written as NaN, and marked `not_implemented_yet`. This avoids silent
approximation of clinical features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vslp.acoustic.features.plugins import build_default_plugins, implemented_feature_names
from vslp.acoustic.features.plugins.base import FeatureContext, FeatureValue
from vslp.acoustic.features.registry import build_acoustic_feature_registry
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.acoustic.features.scales import build_feature_computation_policy, build_feature_scale_registry
from vslp.acoustic.context import cleanup_stage
from vslp.core.project import ensure_stage_folders
from vslp.core.provenance import python_environment
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult

IMPLEMENTED_FEATURES = implemented_feature_names()
PROXY_FEATURES = set()


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Configuration for acoustic feature extraction.

    selected_subsystems
        Optional subsystem whitelist. Empty means all registry subsystems.
    selected_features
        Optional feature whitelist. Empty means all features from selected subsystems.
    task_word_counts
        Optional map from task name to known word count. Used only for speech_rate.
        If absent, speech_rate is emitted as NaN rather than guessed.
    minimum_pause_duration_sec
        Internal nonspeech runs shorter than this are ignored for pause summary features.
    """

    selected_subsystems: list[str] = field(default_factory=list)
    selected_features: list[str] = field(default_factory=list)
    prompt_manifest_path: str | None = None
    task_word_counts: dict[str, float] = field(default_factory=dict)
    minimum_pause_duration_sec: float = 0.30
    acoustic_region_policy: str = "speech_only"  # speech_only, effective_task, full_file
    # v0.35: explicit computation/reduction policy controls.
    # validated_default keeps family-specific defaults. Other modes are applied only
    # where scientifically allowed and are recorded in the reduction audit table.
    computation_mode: str = "validated_default"
    phonatory_mode: str = "voiced_default"
    formant_mode: str = "valid_frame_default"
    resonatory_mode: str = "valid_spectral_default"
    rhythm_mode: str = "effective_task_default"
    coordination_mode: str = "trajectory_default"
    rhythm_region_policy: str = "effective_task"  # rhythm needs internal pauses preserved
    rhythm_envelope_bandpass_low_hz: float = 300.0
    rhythm_envelope_bandpass_high_hz: float = 1000.0
    rhythm_envelope_sample_rate_hz: float = 100.0
    coordination_region_policy: str = "effective_task"
    coordination_frame_ms: float = 40.0
    coordination_hop_ms: float = 10.0
    coordination_max_lag_ms: float = 250.0
    coordination_min_valid_fraction: float = 0.35
    formant_lpc_target_sr_hz: int = 10000
    formant_lpc_order: int | None = None
    formant_preemphasis: float = 0.97
    phonatory_f0_min_hz: float = 60.0
    phonatory_f0_max_hz: float = 400.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _select_registry(cfg: FeatureExtractionConfig) -> pd.DataFrame:
    from vslp.acoustic.features.catalog import load_feature_catalog

    approved = {output["feature_id"] for output in load_feature_catalog()["outputs"]
                if output.get("family_spec_approved") and output.get("selectable")}
    requested = set(cfg.selected_features)
    if not requested or not requested <= approved:
        raise ValueError(
            "Acoustic Features requires exact, approved output IDs from the "
            "79-construct family specifications; legacy feature IDs cannot execute."
        )
    catalog = load_feature_catalog()
    rows = []
    for output in catalog["outputs"]:
        if output["feature_id"] in requested:
            rows.append({"feature": output["feature_id"], "subsystem": output["family_name"],
                         "family": output["family_name"], "family_id": output["family_id"],
                         "construct_id": output["construct_id"],
                         "meaning": output["human_name"], "unit": output["unit"],
                         "formula": output["formula"], "qc_range_text": output["qc_range_text"],
                         "evidence_tier": output["evidence_level"],
                         "implementation_status": output["use_status"],
                         "algorithm_version": output["algorithm_version"],
                         "parameter_set_id": output["default_parameter_set_id"],
                         "analysis_region": output["analysis_region"],
                         "source_document": output["source_document"]})
    return pd.DataFrame(rows)


def _make_context(row: pd.Series, cfg: FeatureExtractionConfig) -> FeatureContext:
    segments_raw = row.get("segments_csv_path", "")
    # Feature plugins must consume the canonical native-rate preprocessing waveform.
    # The FeatureContext field retains its historical name for plugin compatibility.
    wav_raw = row.get("analysis_wav_path", "")
    segments_path = Path(str(segments_raw)) if str(segments_raw) and str(segments_raw) != "nan" else None
    wav_path = Path(str(wav_raw)) if str(wav_raw) and str(wav_raw) != "nan" else None
    duration = row.get("duration_sec", np.nan)
    try:
        duration_float = float(duration)
    except Exception:
        duration_float = np.nan
    return FeatureContext(
        file_name=str(row.get("file_name", "")),
        row=row,
        segments_csv=segments_path,
        segmentation_wav_path=wav_path,
        task=str(row.get("task")) if pd.notna(row.get("task", np.nan)) else None,
        duration_sec=duration_float,
        config=cfg,
        analysis_region=str(getattr(cfg, "acoustic_region_policy", "speech_only")),
    )


def _operational_prefix(row: pd.Series, run: dict[str, Any]) -> dict[str, Any]:
    source_sha = row.get("source_sha256", "")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ValueError("Segmentation summary must retain the original source SHA-256")
    return {
        "recording_id": source_sha,
        "file_name": row.get("file_name", ""),
        "source_file_path": row.get("source_file_path", row.get("file_path")),
        "source_sha256": source_sha,
        "analysis_wav_path": row.get("analysis_wav_path"),
        "project_name": run["project_name"],
        "task_name": run["task_name"],
        "task": run["task_name"],
        "run_id": run["run_id"],
        "run_created_at_local": run["created_at_local"],
        "run_created_at_utc": run["created_at_utc"],
        "duration_sec": row.get("duration_sec", np.nan),
    }


def _normalize_computation_mode(value: str | None) -> str:
    allowed = {
        "validated_default",
        "speech_only_concatenated",
        "effective_task_with_pauses",
        "per_segment_robust",
        "full_file_exploratory",
    }
    text = str(value or "validated_default").strip().lower()
    return text if text in allowed else "validated_default"


def _apply_computation_mode_defaults(cfg: FeatureExtractionConfig) -> FeatureExtractionConfig:
    """Apply v0.35 global computation-mode presets safely.

    This function intentionally does *not* expose every possible mode to every
    feature. Timing remains segment-event based. Rhythm remains effective-task
    by default. Unsupported mode requests are recorded later in the audit table.
    """
    mode = _normalize_computation_mode(getattr(cfg, "computation_mode", "validated_default"))
    updates: dict[str, object] = {"computation_mode": mode}
    if mode == "validated_default":
        return replace(cfg, **updates)
    if mode == "speech_only_concatenated":
        updates.update({
            "acoustic_region_policy": "speech_only",
            "phonatory_mode": "speech_only_concatenated",
            "formant_mode": "speech_only_concatenated",
            "resonatory_mode": "speech_only_concatenated",
        })
        return replace(cfg, **updates)
    if mode == "effective_task_with_pauses":
        updates.update({
            "acoustic_region_policy": "effective_task",
            "rhythm_region_policy": "effective_task",
            "coordination_region_policy": "effective_task",
            "phonatory_mode": "effective_task_exploratory",
            "formant_mode": "effective_task_exploratory",
            "resonatory_mode": "effective_task_exploratory",
        })
        return replace(cfg, **updates)
    if mode == "full_file_exploratory":
        updates.update({
            "acoustic_region_policy": "full_file",
            "rhythm_region_policy": "full_file",
            "coordination_region_policy": "full_file",
            "phonatory_mode": "full_file_exploratory",
            "formant_mode": "full_file_exploratory",
            "resonatory_mode": "full_file_exploratory",
        })
        return replace(cfg, **updates)
    if mode == "per_segment_robust":
        updates.update({
            "phonatory_mode": "per_speech_segment_planned",
            "formant_mode": "per_speech_segment_planned",
            "resonatory_mode": "per_speech_segment_planned",
        })
        return replace(cfg, **updates)
    return replace(cfg, **updates)

def _feature_family(feature: str) -> str:
    f = str(feature)
    timing = {"total_dur", "speech_dur", "percent_pause", "num_pause", "mean_pause_dur", "mean_phrase_dur", "cv_pause_dur", "cv_phrase_dur", "total_pause_dur", "speech_rate"}
    rhythm = {"intensity_CV", "fft_peaks1", "fft_peaks2", "fft_ampli1", "fft_ampli2", "nrj_below_boundary", "nrj_above_boundary", "nrj_3_6", "ratio_below_above"}
    phon = {"f0_mean", "f0_std", "CPP_mean", "HNR", "localJitter", "localabsoluteJitter", "rapJitter", "ppq5Jitter", "ddpJitter", "localShimmer", "localdbShimmer", "apq3Shimmer", "apq5Shimmer", "apq11Shimmer", "num_voicebreaks", "H1freq", "H1amp", "H2freq", "H2amp"}
    coord = {"CPP_F1_comp", "CPP_F2_comp", "F1_F2_comp"}
    reson = {"A1P0", "A1P0comp", "A1P1", "A1P1comp", "A3P0", "P0freq", "P0amp", "P0prom", "P1amp", "F1freq", "F1amp", "F1width", "F2freq", "F2amp", "F2width", "F3freq", "F3amp", "F3width", "RMSamp"}
    if f in timing: return "timing_respiratory"
    if f in rhythm: return "rhythm_ems"
    if f in phon: return "phonatory"
    if f in coord: return "coordination"
    if f in reson: return "resonatory_nasality"
    return "articulatory_formant"


def _build_reduction_audit(registry: pd.DataFrame, cfg: FeatureExtractionConfig) -> pd.DataFrame:
    rows = []
    for _, row in registry.iterrows():
        feature = str(row.get("feature", ""))
        family = _feature_family(feature)
        mode = str(getattr(cfg, "computation_mode", "validated_default"))
        if family == "timing_respiratory":
            native = "speech/pause segment events"
            region = "segment table: speech + internal nonspeech events"
            reducer = "event sums, counts, durations, percent, mean/CV over event distributions"
            applied = "segment_event_default"
            warning = "mode_locked: waveform modes are not applicable to timing physiology"
        elif family == "rhythm_ems":
            native = "effective-task amplitude envelope and 0-10 Hz modulation spectrum"
            region = str(getattr(cfg, "rhythm_region_policy", "effective_task"))
            reducer = "dominant peaks, normalized band powers, slow/fast ratio"
            applied = str(getattr(cfg, "rhythm_mode", "effective_task_default"))
            warning = "speech-only concatenation is discouraged because it destroys internal pause timing"
        elif family == "phonatory":
            native = "voiced frames / period and amplitude support"
            region = str(getattr(cfg, "acoustic_region_policy", "speech_only"))
            reducer = "F0 mean/SD, CPP/HNR summaries, jitter/shimmer perturbation formulas, voice-break count"
            applied = str(getattr(cfg, "phonatory_mode", "voiced_default"))
            warning = "requires adequate voiced support; values are not full-file silence summaries"
        elif family == "articulatory_formant":
            native = "valid LPC formant-frame trajectories"
            region = str(getattr(cfg, "acoustic_region_policy", "speech_only"))
            reducer = "mean formants, median bandwidths, max-min ranges, slope percentiles"
            applied = str(getattr(cfg, "formant_mode", "valid_frame_default"))
            warning = "requires valid LPC frames; low-validity tracks must be reviewed"
        elif family == "resonatory_nasality":
            native = "valid spectral frames with formant/nasal-pole support"
            region = str(getattr(cfg, "acoustic_region_policy", "speech_only"))
            reducer = "median spectral contrasts/support values with overlap and validity warnings"
            applied = str(getattr(cfg, "resonatory_mode", "valid_spectral_default"))
            warning = "vowel/F0/device sensitive; use controlled tasks when possible"
        else:
            native = "aligned CPP/F1/F2 trajectory windows"
            region = str(getattr(cfg, "coordination_region_policy", "effective_task"))
            reducer = "lagged correlation eigenspectrum normalized participation ratio"
            applied = str(getattr(cfg, "coordination_mode", "trajectory_default"))
            warning = "coordination index is not monotonic severity; requires valid aligned tracks"
        rows.append({
            "feature": feature,
            "subsystem": row.get("subsystem", ""),
            "family": family,
            "requested_global_mode": mode,
            "applied_family_mode": applied,
            "analysis_region": region,
            "native_measurement_scale": native,
            "file_level_scalar_reduction": reducer,
            "scientific_warning": warning,
        })
    return pd.DataFrame(rows)

@cleanup_stage
def run_acoustic_feature_extraction(
    segmentation_summary_csv: str | Path,
    output_root: str | Path,
    config: FeatureExtractionConfig | None = None,
    final_segmentation_intervals_csv: str | Path | None = None,
    progress_callback=None,
) -> StageResult:
    """Extract acoustic features from segmentation outputs."""
    authoritative = Path(output_root) / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_decisions.csv"
    if authoritative.is_file():
        expected_intervals = authoritative.with_name("final_segmentation_intervals.csv")
        if (Path(segmentation_summary_csv).resolve() != authoritative.resolve() or
                final_segmentation_intervals_csv is None or
                Path(final_segmentation_intervals_csv).resolve() != expected_intervals.resolve()):
            raise ValueError("Frozen reviewed segmentation exists; Features requires its authoritative final decisions and intervals")
    cfg = _apply_computation_mode_defaults(config or FeatureExtractionConfig())
    registry = _select_registry(cfg)
    from vslp.acoustic.features.family08 import FAMILY08_IDS, run_reviewed_timing_stage
    from vslp.acoustic.features.family09 import FAMILY09_IDS

    if set(cfg.selected_features) <= FAMILY08_IDS | FAMILY09_IDS:
        return run_reviewed_timing_stage(segmentation_summary_csv, output_root, cfg,
                                         final_segmentation_intervals_csv, registry,
                                         progress_callback=progress_callback)
    segmentation_summary_csv = Path(segmentation_summary_csv)
    stage_dir = Path(output_root) / "acoustic" / "005_features"
    folders = ensure_stage_folders(stage_dir, lazy=True)

    selected_names = set(registry["feature"].astype(str).tolist())
    if final_segmentation_intervals_csv is not None:
        from vslp.acoustic.segment.review import load_final_segmentation
        seg_summary = load_final_segmentation(final_segmentation_intervals_csv, segmentation_summary_csv)
    else:
        seg_summary = pd.read_csv(segmentation_summary_csv)
    if final_segmentation_intervals_csv is None and "automatic_status" in seg_summary:
        seg_summary = seg_summary.loc[
            ~seg_summary["automatic_status"].astype(str).str.upper().isin({"EXCLUDED", "FAILED"})
        ].copy()
    if seg_summary.empty:
        raise ValueError("No accepted or review-required segmented recordings are available for feature extraction")
    run = json.loads((Path(output_root) / "project_manifest.json").read_text(encoding="utf-8"))
    if not run.get("task_name") or not run.get("run_id") or not run.get("project_name"):
        raise ValueError("Acoustic feature extraction requires initialized Setup provenance")
    seg_summary["task"] = run["task_name"]

    plugins = [p for p in build_default_plugins() if selected_names.intersection(set(p.feature_names))]

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    long_status_rows: list[dict[str, Any]] = []

    if progress_callback:
        progress_callback(0, len(seg_summary), "Acoustic Features")
    for index, (_, row) in enumerate(seg_summary.iterrows(), start=1):
        file_name = str(row.get("file_name", ""))
        out_row: dict[str, Any] = _operational_prefix(row, run)
        feature_results: dict[str, FeatureValue] = {}
        context = _make_context(row, cfg)
        try:
            for plugin in plugins:
                try:
                    feature_results.update(plugin.compute(context))
                except Exception as exc:  # noqa: BLE001 - one plugin should not kill the file
                    for name in plugin.feature_names:
                        feature_results[name] = FeatureValue(name, np.nan, "failed", f"plugin_failed: {exc}")

            for _, feat in registry.iterrows():
                name = str(feat["feature"])
                subsystem = str(feat["subsystem"])
                result = feature_results.get(name)
                if result is not None:
                    out_row[name] = result.value
                    status = result.status
                    note = result.note
                elif name in IMPLEMENTED_FEATURES:
                    out_row[name] = np.nan
                    status = "not_selected_or_missing_input"
                    note = "feature_has_plugin_but_required_input_was_unavailable"
                else:
                    out_row[name] = np.nan
                    status = "not_implemented_yet"
                    note = "registered_from_uploaded_feature_notebook_pending_validated_implementation"
                long_status_rows.append(
                    {
                        "file_name": file_name,
                        "recording_id": out_row["recording_id"],
                        "project_name": run["project_name"],
                        "task_name": run["task_name"],
                        "run_id": run["run_id"],
                        "run_created_at_local": run["created_at_local"],
                        "run_created_at_utc": run["created_at_utc"],
                        "feature": name,
                        "subsystem": subsystem,
                        "status": status,
                        "note": note,
                    }
                )
            out_row["feature_extraction_status"] = "ok"
            rows.append(out_row)
        except Exception as exc:  # noqa: BLE001
            out_row["feature_extraction_status"] = "failed"
            rows.append(out_row)
            errors.append({"file_name": file_name, "status": "failed", "error": str(exc)})
        finally:
            if progress_callback:
                progress_callback(index, len(seg_summary), f"Acoustic Features — {file_name}")

    features_path = folders["tables"] / "acoustic_features_per_file.csv"
    status_path = folders["tables"] / "acoustic_feature_status_long.csv"
    registry_path = folders["tables"] / "selected_acoustic_feature_registry.csv"
    scale_registry_path = folders["tables"] / "acoustic_feature_measurement_scale_registry.csv"
    computation_policy_path = folders["tables"] / "acoustic_feature_computation_policy.csv"
    reduction_audit_path = folders["tables"] / "acoustic_feature_scalar_reduction_audit.csv"
    native_segments_path = folders["tables"] / "native_measurements" / "acoustic_native_segment_events.csv"
    errors_path = folders["errors"] / "acoustic_feature_errors.csv"

    pd.DataFrame(rows).to_csv(features_path, index=False)
    pd.DataFrame(long_status_rows).to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    handoff_values = pd.DataFrame(rows).copy()
    if not handoff_values.empty:
        handoff_values["source_file"] = handoff_values.get("file_name", "")
        handoff_values["modality"] = "acoustic"
        handoff_values["aggregation_level"] = "recording_task"
    handoff_status = pd.DataFrame(long_status_rows).copy()
    if not handoff_status.empty:
        handoff_status["modality"] = "acoustic"
    handoff_registry = registry.copy()
    handoff_registry["aggregation"] = "file-level scalar; see acoustic_feature_computation_policy.csv"
    handoff_registry["normalization"] = "feature-specific native scale; see acoustic_feature_measurement_scale_registry.csv"
    handoff_registry["required_inputs"] = "segmentation and/or task-scoped waveform; see computation_note"
    handoff_registry["source_document"] = "Features Formulas (2).docx; Features Research  (2).xlsx"
    handoff_registry["source_location"] = "docs/reference/ACOUSTIC_FEATURE_SOURCE_TRACEABILITY.md"
    handoff = write_feature_handoff(
        folders["tables"], handoff_values, handoff_registry, handoff_status,
        "acoustic", features_path, registry_path,
    )
    delivery = build_feature_delivery(
        output_root,
        "acoustic",
        handoff,
        optional_main={
            "qc_features.csv": Path(output_root) / "acoustic" / "004_quality_control" / "tables" / "acoustic_quality_processed_features.csv",
        },
    )
    build_feature_scale_registry(registry).to_csv(scale_registry_path, index=False)
    build_feature_computation_policy(registry).to_csv(computation_policy_path, index=False)
    _build_reduction_audit(registry, cfg).to_csv(reduction_audit_path, index=False)
    _write_native_segment_events(seg_summary, native_segments_path)
    pd.DataFrame(errors).to_csv(errors_path, index=False)

    missingness_plot = folders["plots"] / "feature_missingness.png"
    subsystem_plot = folders["plots"] / "feature_subsystem_implementation_status.png"
    distribution_plot = folders["plots"] / "implemented_feature_distributions.png"
    task_plot = folders["plots"] / "task_feature_overview.png"
    range_plot = folders["plots"] / "feature_expected_range_flags.png"
    audit_plot = folders["plots"] / "feature_distribution_audit.png"
    corr_plot = folders["plots"] / "feature_correlation_heatmap.png"
    subsystem_dist_plot = folders["plots"] / "feature_subsystem_distributions.png"

    _plot_feature_missingness(features_path, registry, missingness_plot)
    _plot_subsystem_status(status_path, subsystem_plot)
    _plot_implemented_feature_distributions(features_path, registry, distribution_plot)
    _plot_task_feature_overview(features_path, task_plot)
    range_flags_path = folders["tables"] / "acoustic_feature_expected_range_flags.csv"
    distribution_audit_path = folders["tables"] / "acoustic_feature_distribution_audit.csv"
    _write_feature_distribution_audit(features_path, registry, distribution_audit_path, range_flags_path)
    _plot_expected_range_flags(range_flags_path, range_plot)
    _plot_feature_distribution_audit(features_path, registry, audit_plot)
    _plot_feature_correlation_heatmap(features_path, registry, corr_plot)
    _plot_subsystem_distribution_summary(features_path, registry, subsystem_dist_plot)

    report_path = folders["reports"] / "acoustic_feature_report.html"
    _write_feature_html_report(
        report_path,
        rows,
        long_status_rows,
        errors,
        cfg,
        missingness_plot,
        subsystem_plot,
        distribution_plot,
        task_plot,
        range_plot,
        audit_plot,
        corr_plot,
        subsystem_dist_plot,
    )

    computed_statuses = {"computed", "computed_proxy", "computed_with_warning"}
    status_df = pd.DataFrame(long_status_rows)
    computed_features = sorted(status_df.loc[status_df["status"].isin(computed_statuses), "feature"].unique().tolist()) if not status_df.empty else []
    proxy_features = sorted(status_df.loc[status_df["status"].eq("computed_proxy"), "feature"].unique().tolist()) if not status_df.empty else []

    warnings: list[str] = []
    if proxy_features:
        warnings.append(f"Proxy features require reference validation before clinical interpretation: {proxy_features}")
    if errors:
        warnings.append(f"{len(errors)} files failed feature extraction")

    manifest = StageManifest(
        stage_name="acoustic_feature_extraction",
        stage_version="0.12.0",
        status="completed_with_warnings" if warnings else "completed",
        input_artifacts=[ArtifactRef(path=str(segmentation_summary_csv), role="final_segmentation_decisions" if final_segmentation_intervals_csv else "segmentation_summary", media_type="text/csv")]
        + ([ArtifactRef(path=str(final_segmentation_intervals_csv), role="final_segmentation_intervals", media_type="text/csv")] if final_segmentation_intervals_csv else []),
        output_artifacts=[
            ArtifactRef(path=str(features_path), role="features_per_file", media_type="text/csv"),
            ArtifactRef(path=str(status_path), role="feature_status_long", media_type="text/csv"),
            ArtifactRef(path=str(registry_path), role="selected_feature_registry", media_type="text/csv"),
            ArtifactRef(path=str(handoff["feature_values_csv"]), role="canonical_feature_values", media_type="text/csv"),
            ArtifactRef(path=str(handoff["feature_registry_csv"]), role="canonical_feature_registry", media_type="text/csv"),
            ArtifactRef(path=str(handoff["feature_status_csv"]), role="canonical_feature_status", media_type="text/csv"),
            ArtifactRef(path=str(delivery["delivery_manifest_json"]), role="feature_delivery_manifest", media_type="application/json"),
            ArtifactRef(path=str(scale_registry_path), role="feature_measurement_scale_registry", media_type="text/csv"),
            ArtifactRef(path=str(computation_policy_path), role="feature_computation_policy", media_type="text/csv"),
            ArtifactRef(path=str(native_segments_path), role="native_segment_events", media_type="text/csv"),
            ArtifactRef(path=str(range_flags_path), role="expected_range_flags", media_type="text/csv"),
            ArtifactRef(path=str(distribution_audit_path), role="distribution_audit", media_type="text/csv"),
            ArtifactRef(path=str(report_path), role="feature_html_report", media_type="text/html"),
        ],
        config=cfg.to_dict(),
        environment={"python": python_environment()},
        warnings=warnings,
        errors=errors,
        notes=[
            "Feature extraction uses subsystem plugins and region-aware signal policy.",
            "Respiratory/timing features were validated in v0.26 from segmentation tables.",
            "Rhythm/EMS features were validated in v0.27 from effective-task envelope modulation spectrum.",
            f"Computed feature families in this pass: {computed_features}",
            "Coordination features were validated in v0.31 as time-delay cross-correlation eigenspectrum complexity over CPP/F1/F2 trajectories.",
            "Feature measurement scale registry documents the native physiologic scale and recommended reducers.",
            "Feature computation policy is written to define, for each feature, the default analysis region and exact file-level scalar reduction strategy.",
            "Scalar reduction audit is written to document the requested mode, applied family mode, and native measurement scale for every selected feature.",
            "Native segment-event measurements are preserved for timing features; frame/trajectory persistence for signal features is planned as the next architecture extension.",
            "Registered-but-not-yet-implemented features remain explicit NaN placeholders.",
            "Expected-range flags are descriptive screening aids, not clinical cutoffs.",
        ],
    )
    manifest_path = folders["logs"] / "stage_manifest.json"
    manifest.write_json(manifest_path)

    return StageResult(
        status=manifest.status,
        manifest_path=manifest_path,
        summary_table=features_path,
        error_table=errors_path,
        report_path=report_path,
    )


def _write_native_segment_events(seg_summary: pd.DataFrame, output_path: Path) -> None:
    """Persist segment-level native measurements used by timing features.

    This is the first native-scale preservation table. It prevents the pipeline from
    treating pause/phrase physiology as if it only ever existed as one file-level
    mean. Signal-level track persistence for F0, CPP, formants, EMS, and
    coordination is handled in the measurement-scale registry and planned as a
    subsequent architecture extension.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    meta_cols = [
        "recording_id", "file_name", "source_file_path", "source_sha256", "project_name", "task_name", "run_id", "task",
    ]
    for _, file_row in seg_summary.iterrows():
        seg_path_raw = file_row.get("segments_csv_path", "")
        if not isinstance(seg_path_raw, str) or not seg_path_raw or seg_path_raw == "nan":
            continue
        seg_path = Path(seg_path_raw)
        if not seg_path.exists():
            continue
        try:
            segs = pd.read_csv(seg_path)
        except Exception:
            continue
        if segs.empty:
            continue
        for idx, seg in segs.reset_index(drop=True).iterrows():
            out = {c: file_row.get(c, np.nan) for c in meta_cols}
            out.update({
                "segment_index": int(idx),
                "segment_type": seg.get("segment_type", np.nan),
                "segment_role": seg.get("segment_role", np.nan),
                "start_sec": seg.get("start_sec", np.nan),
                "end_sec": seg.get("end_sec", np.nan),
                "duration_sec": seg.get("duration_sec", np.nan),
                "native_scale": "segment_event",
                "physiologic_interpretation": "speech_phrase" if seg.get("segment_type", "") == "speech" else "pause_or_nonspeech",
            })
            rows.append(out)
    cols = meta_cols + ["segment_index", "segment_type", "segment_role", "start_sec", "end_sec", "duration_sec", "native_scale", "physiologic_interpretation"]
    pd.DataFrame(rows, columns=cols).to_csv(output_path, index=False)


def _feature_names_in_table(features_csv: Path, registry: pd.DataFrame) -> list[str]:
    df_cols = pd.read_csv(features_csv, nrows=0).columns
    return [f for f in registry["feature"].tolist() if f in df_cols]


def _plot_feature_missingness(features_csv: Path, registry: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_csv)
    feature_names = [f for f in registry["feature"].tolist() if f in df.columns]
    if not feature_names:
        return
    miss = df[feature_names].isna().mean().sort_values(ascending=True)
    height = max(5.0, min(18.0, 0.18 * len(miss) + 2.0))
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(miss.index, miss.values)
    ax.set_xlabel("Fraction missing / not implemented")
    ax.set_xlim(0, 1)
    ax.set_title("VSLP Acoustic Feature Missingness")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_subsystem_status(status_csv: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(status_csv)
    if df.empty:
        return
    tmp = df.drop_duplicates(["feature", "subsystem", "status"])
    counts = tmp.groupby(["subsystem", "status"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Number of selected features")
    ax.set_title("Feature implementation status by subsystem")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_implemented_feature_distributions(features_csv: Path, registry: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_csv)
    names = [f for f in registry["feature"].astype(str).tolist() if f in df.columns and f in IMPLEMENTED_FEATURES]
    numeric_names = [n for n in names if pd.to_numeric(df[n], errors="coerce").notna().any()]
    if not numeric_names:
        return
    chosen = numeric_names[:16]
    n = len(chosen)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, max(3, 2.8 * nrows)))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, name in zip(axes_arr, chosen, strict=False):
        vals = pd.to_numeric(df[name], errors="coerce").dropna().values
        if vals.size:
            # Nearly identical measurements can make NumPy's equally spaced
            # bin edges collapse at floating point precision.
            spread = float(np.max(vals) - np.min(vals))
            scale = max(1.0, float(np.max(np.abs(vals))))
            bins = 1 if spread <= np.finfo(float).eps * scale * 16 else min(20, max(5, int(np.sqrt(vals.size))))
            ax.hist(vals, bins=bins)
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
    for ax in axes_arr[len(chosen):]:
        ax.axis("off")
    fig.suptitle("Implemented acoustic feature distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_task_feature_overview(features_csv: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_csv)
    if "task" not in df.columns or df.empty:
        return
    candidate_features = [f for f in ["percent_pause", "speech_dur", "f0_mean", "intensity_CV", "fft_peaks1", "RMSamp"] if f in df.columns]
    if not candidate_features:
        return
    task_counts = df["task"].fillna("unknown").astype(str).value_counts().head(12)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(task_counts.index, task_counts.values)
    ax.set_ylabel("Files")
    ax.set_title("Files by task in feature table")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)



def _numeric_feature_frame(features_csv: Path, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(features_csv)
    feature_names = [str(f) for f in registry["feature"].tolist() if str(f) in df.columns]
    if not feature_names:
        return df, pd.DataFrame()
    numeric = pd.DataFrame({name: pd.to_numeric(df[name], errors="coerce") for name in feature_names})
    return df, numeric


def _write_feature_distribution_audit(features_csv: Path, registry: pd.DataFrame, audit_path: Path, flags_path: Path) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    df, numeric = _numeric_feature_frame(features_csv, registry)
    rows: list[dict[str, Any]] = []
    flag_rows: list[dict[str, Any]] = []
    meta = registry.set_index("feature", drop=False)
    for name in numeric.columns:
        vals = numeric[name].dropna()
        r = meta.loc[name] if name in meta.index else pd.Series(dtype=object)
        lo = pd.to_numeric(pd.Series([r.get("expected_low")]), errors="coerce").iloc[0] if "expected_low" in r.index else np.nan
        hi = pd.to_numeric(pd.Series([r.get("expected_high")]), errors="coerce").iloc[0] if "expected_high" in r.index else np.nan
        below = int((vals < lo).sum()) if np.isfinite(lo) and not vals.empty else 0
        above = int((vals > hi).sum()) if np.isfinite(hi) and not vals.empty else 0
        out = below + above
        rows.append({
            "feature": name,
            "subsystem": r.get("subsystem", ""),
            "unit": r.get("unit", ""),
            "n_available": int(vals.size),
            "missing_fraction": float(numeric[name].isna().mean()) if len(numeric) else np.nan,
            "mean": float(vals.mean()) if vals.size else np.nan,
            "median": float(vals.median()) if vals.size else np.nan,
            "sd": float(vals.std(ddof=0)) if vals.size else np.nan,
            "q05": float(vals.quantile(0.05)) if vals.size else np.nan,
            "q25": float(vals.quantile(0.25)) if vals.size else np.nan,
            "q75": float(vals.quantile(0.75)) if vals.size else np.nan,
            "q95": float(vals.quantile(0.95)) if vals.size else np.nan,
            "expected_low": lo,
            "expected_high": hi,
            "n_below_expected": below,
            "n_above_expected": above,
            "out_of_expected_fraction": float(out / vals.size) if vals.size else np.nan,
            "screening_note": "descriptive expected range only; rederive study-specific ranges" if np.isfinite(lo) or np.isfinite(hi) else "no expected range configured",
        })
        if out > 0:
            flag_rows.append({
                "feature": name,
                "subsystem": r.get("subsystem", ""),
                "n_flagged": out,
                "n_available": int(vals.size),
                "flagged_fraction": float(out / vals.size) if vals.size else np.nan,
                "expected_low": lo,
                "expected_high": hi,
                "note": "values outside orientation range; inspect distribution and acquisition/QC before interpretation",
            })
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    pd.DataFrame(flag_rows, columns=["feature","subsystem","n_flagged","n_available","flagged_fraction","expected_low","expected_high","note"]).to_csv(flags_path, index=False)


def _plot_expected_range_flags(flags_csv: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not flags_csv.exists() or flags_csv.stat().st_size == 0:
        return
    df = pd.read_csv(flags_csv)
    if df.empty or "flagged_fraction" not in df.columns:
        return
    df = df.sort_values("flagged_fraction", ascending=True).tail(25)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(df) + 1)))
    ax.barh(df["feature"], df["flagged_fraction"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of available values outside orientation range")
    ax.set_title("Feature expected-range review flags")
    ax.axvline(0.20, linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_feature_distribution_audit(features_csv: Path, registry: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _df, numeric = _numeric_feature_frame(features_csv, registry)
    meta = registry.set_index("feature", drop=False)
    available = [c for c in numeric.columns if numeric[c].notna().sum() >= 2]
    if not available:
        return
    # Prioritize computed/proxy features with expected ranges, then high-coverage features.
    def score(name: str) -> tuple[int, int]:
        r = meta.loc[name] if name in meta.index else pd.Series(dtype=object)
        has_range = int(pd.notna(r.get("expected_low", np.nan)) or pd.notna(r.get("expected_high", np.nan)))
        return (has_range, int(numeric[name].notna().sum()))
    chosen = sorted(available, key=score, reverse=True)[:12]
    ncols = 3
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, max(4, 3.2 * nrows)))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, name in zip(axes_arr, chosen, strict=False):
        vals = numeric[name].dropna().values.astype(float)
        r = meta.loc[name] if name in meta.index else pd.Series(dtype=object)
        lo = pd.to_numeric(pd.Series([r.get("expected_low")]), errors="coerce").iloc[0] if "expected_low" in r.index else np.nan
        hi = pd.to_numeric(pd.Series([r.get("expected_high")]), errors="coerce").iloc[0] if "expected_high" in r.index else np.nan
        spread = float(np.max(vals) - np.min(vals))
        scale = max(1.0, float(np.max(np.abs(vals))))
        bins = 1 if spread <= np.finfo(float).eps * scale * 16 else min(20, max(5, int(np.sqrt(vals.size) + 2)))
        ax.hist(vals, bins=bins, alpha=0.85)
        if np.isfinite(lo):
            ax.axvline(lo, linestyle="--", linewidth=1)
        if np.isfinite(hi):
            ax.axvline(hi, linestyle="--", linewidth=1)
        out_frac = np.nan
        if vals.size:
            mask = np.zeros(vals.shape, dtype=bool)
            if np.isfinite(lo): mask |= vals < lo
            if np.isfinite(hi): mask |= vals > hi
            out_frac = float(mask.mean())
        title = name if not np.isfinite(out_frac) or out_frac == 0 else f"{name}  REVIEW {out_frac:.0%}"
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
    for ax in axes_arr[len(chosen):]:
        ax.axis("off")
    fig.suptitle("Acoustic feature distribution audit", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_feature_correlation_heatmap(features_csv: Path, registry: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _df, numeric = _numeric_feature_frame(features_csv, registry)
    cols = [c for c in numeric.columns if numeric[c].notna().sum() >= 3 and numeric[c].nunique(dropna=True) >= 2]
    if len(cols) < 2:
        return
    cols = cols[:30]
    corr = numeric[cols].corr(method="spearman", min_periods=3)
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(cols) + 3), max(7, 0.35 * len(cols) + 3)))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=7)
    ax.set_title("Spearman correlation among available acoustic features")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_subsystem_distribution_summary(features_csv: Path, registry: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _df, numeric = _numeric_feature_frame(features_csv, registry)
    if numeric.empty:
        return
    meta = registry.set_index("feature", drop=False)
    rows = []
    for name in numeric.columns:
        r = meta.loc[name] if name in meta.index else pd.Series(dtype=object)
        vals = numeric[name]
        rows.append({"feature": name, "subsystem": r.get("subsystem", "unknown"), "coverage": float(vals.notna().mean())})
    cov = pd.DataFrame(rows)
    if cov.empty:
        return
    summary = cov.groupby("subsystem")["coverage"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(summary.index.astype(str), summary.values)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean feature availability within subsystem")
    ax.set_title("Feature coverage by subsystem")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def _write_feature_html_report(
    path: Path,
    rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    cfg: FeatureExtractionConfig,
    missingness_plot: Path,
    subsystem_plot: Path,
    distribution_plot: Path,
    task_plot: Path,
    range_plot: Path,
    audit_plot: Path,
    corr_plot: Path,
    subsystem_dist_plot: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status_df = pd.DataFrame(status_rows)
    computed = int(status_df["status"].isin(["computed", "computed_proxy", "computed_with_warning"]).sum()) if not status_df.empty else 0
    proxies = int((status_df["status"] == "computed_proxy").sum()) if not status_df.empty else 0
    pending = int((status_df["status"] == "not_implemented_yet").sum()) if not status_df.empty else 0
    files_ok = int(sum(1 for r in rows if r.get("feature_extraction_status") == "ok"))
    failed = len(errors)
    implemented = sorted(status_df.loc[status_df["status"].isin(["computed", "computed_proxy", "computed_with_warning"]), "feature"].unique().tolist()) if not status_df.empty else []

    def img_block(title: str, image_path: Path) -> str:
        if not image_path.exists():
            return ""
        rel = Path("../plots") / image_path.name
        return f"<div class='card'><h2>{title}</h2><img src='{rel.as_posix()}'></div>"

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>VSLP Acoustic Feature Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; background:#071A2D; color:#EAF2F8; margin:32px; }}
.card {{ background:#0B253D; border:1px solid #315D7C; border-radius:12px; padding:18px; margin:14px 0; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:6px; background:#0E3A5B; margin-right:8px; margin-top:4px; }}
pre {{ white-space:pre-wrap; background:#102A43; padding:12px; border-radius:8px; }}
code {{ background:#102A43; padding:2px 5px; border-radius:4px; }}
img {{ max-width:100%; border-radius:10px; border:1px solid #315D7C; background:white; }}
.warning {{ color:#FFDFA8; }}
</style></head><body>
<h1>VSLP Acoustic Feature Extraction Report</h1>
<div class='card'><span class='badge'>Files OK: {files_ok}</span><span class='badge'>Files failed: {failed}</span><span class='badge'>Computed feature values: {computed}</span><span class='badge'>Proxy values: {proxies}</span><span class='badge'>Pending placeholders: {pending}</span></div>
<div class='card'><h2>Current implementation scope</h2><p>The region-aware plugin architecture records task region, estimator settings, and feature status. Cycle-based jitter, shimmer, and voice breaks use Praat PointProcess measures; local CPP/HNR, LPC formants, nasality, EMS, and automatic DDK event detection retain explicit validation notes.</p><p class='warning'>Registered features that are not yet implemented remain explicit <code>NaN</code> placeholders with status <code>not_implemented_yet</code>.</p></div>
<div class='card'><h2>Computed feature names</h2><pre>{json.dumps(implemented, indent=2)}</pre></div>
{img_block('Feature missingness', missingness_plot)}
{img_block('Subsystem implementation status', subsystem_plot)}
{img_block('Implemented feature distributions', distribution_plot)}
{img_block('Task overview', task_plot)}
{img_block('Expected-range flags', range_plot)}
{img_block('Feature distribution audit', audit_plot)}
{img_block('Feature correlation heatmap', corr_plot)}
{img_block('Subsystem distribution summary', subsystem_dist_plot)}
<div class='card'><h2>Configuration</h2><pre>{json.dumps(cfg.to_dict(), indent=2)}</pre></div>
</body></html>"""
    path.write_text(html, encoding="utf-8")
