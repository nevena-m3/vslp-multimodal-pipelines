"""Task-aware acoustic segmentation stage and auditable artifact writer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from vslp.acoustic.context import cleanup_stage
from vslp.acoustic.segment.selection import DDK, PHONATION, SILERO
from vslp.acoustic.segment.silero_reference import (
    Interval, boundary_alignment_diagnostics, classify_reading_segmentation,
    erode_intervals, internal_nonspeech, load_silero_model, normalize_intervals,
    segmentation_frame_diagnostics, silero_speech_timestamps, summarize_segmentation,
)
from vslp.acoustic.segment.stage import _prepare_silero_audio, _read_canonical_audio
from vslp.acoustic.segment.task_methods import DDKConfig, PhonationConfig, segment_ddk, segment_phonation
from vslp.core.project import ensure_stage_folders
from vslp.core.provenance import python_environment, sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


@dataclass(frozen=True)
class SegmentationConfig:
    method: str = SILERO
    threshold: float = 0.50
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 0
    frame_ms: int = 30
    strict_speech_edge_ms: int = 50
    strict_nonspeech_edge_ms: int = 200
    boundary_audit_window_ms: int = 120
    boundary_audit_guard_ms: int = 20
    boundary_audit_minimum_contrast_db: float = 3.0
    sensitivity_profile: str = "default"
    ddk: DDKConfig = field(default_factory=DDKConfig)
    phonation: PhonationConfig = field(default_factory=PhonationConfig)


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def _segments(intervals: list[Interval], duration: float) -> pd.DataFrame:
    """Exact interval segments for QC and timing features; no frame quantization."""
    intervals = normalize_intervals(intervals, duration)
    rows = []
    cursor = 0.0
    for i, item in enumerate(intervals):
        if item.start_sec > cursor:
            role = "leading_nonspeech" if i == 0 else "internal_nonspeech"
            rows.append({"segment_type": "nonspeech", "segment_role": role,
                         "start_sec": cursor, "end_sec": item.start_sec})
        rows.append({"segment_type": "speech", "segment_role": "speech",
                     "start_sec": item.start_sec, "end_sec": item.end_sec})
        cursor = item.end_sec
    if cursor < duration:
        rows.append({"segment_type": "nonspeech", "segment_role": "trailing_nonspeech",
                     "start_sec": cursor, "end_sec": duration})
    if not rows:
        rows.append({"segment_type": "nonspeech", "segment_role": "leading_nonspeech",
                     "start_sec": 0.0, "end_sec": duration})
    result = pd.DataFrame(rows)
    result["duration_sec"] = result.end_sec - result.start_sec
    return result


def _frames(x: np.ndarray, sr: int, views: dict[str, list[Interval]], frame_ms: int) -> pd.DataFrame:
    frames = segmentation_frame_diagnostics(x, sr, views, frame_ms=frame_ms, hop_ms=frame_ms)
    frames["rms_db"] = frames["rms_dbfs"]
    frames["rms"] = np.power(10, frames["rms_db"] / 20)
    frames["speech_mask_strict"] = frames["strict_speech"].astype(bool)
    frames["nonspeech_mask_strict"] = frames["strict_internal_nonspeech"].astype(bool)
    frames["speech_vad_raw"] = frames["raw_speech"].astype(bool)
    frames["speech_vad_smooth"] = frames["primary_speech"].astype(bool)
    frames["frame_idx"] = np.arange(len(frames))
    frames["frame_ms"] = frame_ms
    return frames


def _common_summary(intervals: list[Interval], x: np.ndarray, sr: int, frames: pd.DataFrame) -> dict:
    duration = len(x) / sr
    pauses = internal_nonspeech(intervals, duration)
    speech = sum(p.duration_sec for p in intervals)
    return {
        "duration_sec": duration, "speech_duration_sec": speech,
        "speech_fraction": speech / duration,
        "n_segments_total": len(_segments(intervals, duration)),
        "n_speech_segments": len(intervals),
        "n_internal_nonspeech_segments": len(pauses),
        "leading_nonspeech_sec": intervals[0].start_sec if intervals else duration,
        "trailing_nonspeech_sec": duration - intervals[-1].end_sec if intervals else duration,
        "longest_internal_nonspeech_sec": max((p.duration_sec for p in pauses), default=0.0),
        "rms_db_median": float(frames.rms_db.median()),
        "rms_db_std": float(frames.rms_db.std(ddof=0)),
    }


def _plot(path: Path, x: np.ndarray | None, sr: int | None, method: str,
          intervals: list[Interval], status: str, flags: list[str],
          trace: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
          nuclei: list[float] | None = None, stable: Interval | None = None,
          breaks: list[Interval] | None = None,
          automatic_intervals: list[Interval] | None = None,
          trace_label: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2 if trace else 1, 1, figsize=(13, 5.8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]} if trace else None)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    if x is not None and sr:
        t = np.arange(len(x)) / sr
        axes[0].plot(t, x, lw=0.45, color="#315a89")
        axes[0].set_xlim(0, len(x) / sr)
        duration = len(x) / sr
        if intervals:
            if intervals[0].start_sec > 0:
                axes[0].axvspan(0, intervals[0].start_sec, color="#d97876", alpha=0.08)
            if intervals[-1].end_sec < duration:
                axes[0].axvspan(intervals[-1].end_sec, duration, color="#d97876", alpha=0.08)
        else:
            axes[0].axvspan(0, duration, color="#d97876", alpha=0.08)
    for interval in intervals:
        axes[0].axvspan(interval.start_sec, interval.end_sec, color="#4abf80", alpha=0.26)
        if automatic_intervals is None:
            axes[0].axvline(interval.start_sec, color="#27814f", lw=0.8)
            axes[0].axvline(interval.end_sec, color="#27814f", lw=0.8)
        else:
            axes[0].axvline(interval.start_sec, color="#7735a4", ls="--", lw=1.1)
            axes[0].axvline(interval.end_sec, color="#7735a4", ls="--", lw=1.1)
    for item in automatic_intervals or []:
        axes[0].axvline(item.start_sec, color="#27814f", lw=0.8, alpha=0.8)
        axes[0].axvline(item.end_sec, color="#27814f", lw=0.8, alpha=0.8)
    for left, right in zip(intervals[:-1], intervals[1:]):
        axes[0].axvspan(left.end_sec, right.start_sec, color="#d97876", alpha=0.14)
    if stable:
        axes[0].axvspan(stable.start_sec, stable.end_sec, color="#efaa33", alpha=0.4,
                        label="Stable analysis region")
    for item in breaks or []:
        axes[0].axvspan(item.start_sec, item.end_sec, color="#d94055", alpha=0.32)
    if nuclei:
        for n in nuclei:
            axes[0].axvline(n, color="#8520af", ls="--", lw=0.8)
    axes[0].set_ylabel("Amplitude")
    legend = [Patch(facecolor="#4abf80", alpha=0.35, label="Speech"),
              Patch(facecolor="#d97876", alpha=0.13, label="Leading nonspeech"),
              Patch(facecolor="#d97876", alpha=0.30, label="Internal nonspeech / pause"),
              Patch(facecolor="#d97876", alpha=0.13, label="Trailing nonspeech"),
              Line2D([0], [0], color="#27814f", lw=1, label="Automatic boundary")]
    if automatic_intervals is not None:
        legend.append(Line2D([0], [0], color="#7735a4", ls="--", lw=1.1,
                             label="Manual boundary"))
    axes[0].legend(handles=legend, loc="upper right", fontsize=7, ncol=2, framealpha=0.88)
    if trace:
        times, values, threshold = trace
        label = trace_label or ("RMS support" if method == SILERO else ("Energy envelope" if method == DDK else "Activity RMS"))
        axes[1].plot(times, values, label=label, color="#374e9c", lw=1)
        if method != SILERO and trace_label is None:
            axes[1].plot(times, threshold, label="Adaptive threshold", color="#ce4b41", lw=1)
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].set_ylabel("Support")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{method} | {status} | {'; '.join(flags) or 'No review flags'}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _method_result(x: np.ndarray, sr: int, cfg: SegmentationConfig, model: Any):
    duration = len(x) / sr
    extras: dict[str, Any] = {}
    trace = None
    nuclei = []
    stable = None
    breaks = []
    if cfg.method == SILERO:
        work, resample = _prepare_silero_audio(x, sr)
        profiles = {"default": (cfg.threshold, cfg.min_speech_duration_ms, cfg.min_silence_duration_ms),
                    "conservative": (0.65, 250, 100), "permissive": (0.35, 100, 200)}
        if cfg.sensitivity_profile not in profiles:
            raise ValueError(f"Unknown Silero sensitivity profile: {cfg.sensitivity_profile}")
        threshold, min_speech, min_silence = profiles[cfg.sensitivity_profile]
        kwargs = {"threshold": threshold, "min_speech_ms": min_speech,
                  "min_silence_ms": min_silence, "speech_pad_ms": cfg.speech_pad_ms,
                  "model": model}
        stamps = silero_speech_timestamps(work, **kwargs)
        raw = [Interval(p["start"] / 16000, p["end"] / 16000) for p in stamps]
        primary = normalize_intervals(raw, duration)  # No second bridge/filter pass.
        extras.update(resample)
        extras["boundary_source"] = "silero_vad_6.2.1_onnx_sample_indices"
        extras["algorithm_version"] = "silero-vad 6.2.1"
        extras["silero_sample_timestamps"] = json.dumps(stamps)
        extras["sensitivity_profile"] = cfg.sensitivity_profile
    elif cfg.method == DDK:
        result = segment_ddk(x, sr, cfg.ddk)
        raw = [Interval(a / sr, b / sr) for a, b in result["intervals_samples"]]
        primary = raw
        trace = (result["frame_times_sec"], result["energy_envelope"], result["adaptive_threshold"])
        nuclei = [n / sr for n in result["nuclei_samples"]]
        extras.update({key: result[key] for key in ("n_events", "ddk_rate_hz", "cycle_mean_sec",
                                                  "cycle_sd_sec", "minimum_peak_threshold_ratio")})
        extras["threshold_perturbation_event_counts"] = json.dumps(result["threshold_perturbation_event_counts"])
        extras.update({"boundary_source": "ddk_energy_envelope_threshold_crossings",
                       "algorithm_version": "vslp-ddk-energy-1", "processing_provenance": json.dumps(result["processing"])})
    elif cfg.method == PHONATION:
        result = segment_phonation(x, sr, cfg.phonation)
        raw = [Interval(a / sr, b / sr) for a, b in [result["full_samples"]] if a is not None] if result["full_samples"] else []
        primary = raw
        if result["stable_samples"]:
            a, b = result["stable_samples"]
            stable = Interval(a / sr, b / sr)
        breaks = [Interval(a / sr, b / sr) for a, b in result["internal_breaks_samples"]]
        trace = (result["frame_times_sec"], result["activity_rms"], result["activity_threshold"])
        extras.update({"full_phonation_start": primary[0].start_sec if primary else np.nan,
                       "full_phonation_end": primary[0].end_sec if primary else np.nan,
                       "stable_region_start": stable.start_sec if stable else np.nan,
                       "stable_region_end": stable.end_sec if stable else np.nan,
                       "n_internal_voice_breaks": len(breaks),
                       "boundary_source": "sustained_phonation_energy_episode",
                       "algorithm_version": "vslp-phonation-1",
                       "processing_provenance": json.dumps(result["processing"])})
    else:
        raise ValueError(f"Unsupported segmentation method: {cfg.method}")
    strict = erode_intervals(primary, cfg.strict_speech_edge_ms / 1000)
    strict_pause = erode_intervals(internal_nonspeech(primary, duration),
                                   cfg.strict_nonspeech_edge_ms / 1000)
    views = {"raw_speech": raw, "primary_speech": primary, "strict_speech": strict,
             "strict_internal_nonspeech": strict_pause}
    frames = _frames(x, sr, views, cfg.frame_ms)
    summary = _common_summary(primary, x, sr, frames)
    if cfg.method == SILERO:
        triage = classify_reading_segmentation(summary)
        flags = [f for f in triage["qc_flags"].split(";") if f]
        status = {"accepted": "ACCEPTED", "flagged": "REVIEW", "excluded": "EXCLUDED"}[triage["qc_status"]]
        display = _segments([Interval(float(r.start_sec), float(r.end_sec)) for _, r in
            frames.loc[frames.speech_vad_smooth].iterrows()], duration)
        audit = boundary_alignment_diagnostics(
            x, sr, primary, displayed_segments=display,
            window_ms=cfg.boundary_audit_window_ms,
            guard_ms=cfg.boundary_audit_guard_ms,
            minimum_contrast_db=cfg.boundary_audit_minimum_contrast_db)
        if audit.boundary_review_flag.any():
            flags.append("weak_boundary_energy_contrast")
            if status == "ACCEPTED":
                status = "REVIEW"
        extras["low_contrast_boundary_fraction"] = float(audit.boundary_review_flag.mean()) if len(audit) else np.nan
        guardrails = (
            summary["speech_fraction"] < 0.10,
            summary["n_speech_segments"] >= 20,
            summary["longest_internal_nonspeech_sec"] >= 4.0,
            summary["duration_sec"] < 2.0,
            np.isfinite(extras["low_contrast_boundary_fraction"])
            and extras["low_contrast_boundary_fraction"] >= 0.50,
        )
        if status == "ACCEPTED" and any(guardrails):
            flags.append("reading_review_guardrail")
            status = "REVIEW"
        trace = (frames.mid_sec.to_numpy(), frames.rms.to_numpy(),
                 np.zeros(len(frames), dtype=float))
    else:
        flags = result["flags"]
        status = result["automatic_status"]
        audit = pd.DataFrame()
    summary.update(extras)
    return primary, views, frames, audit, summary, status, flags, trace, nuclei, stable, breaks


@cleanup_stage
def run_acoustic_segmentation(preprocess_summary_csv: str | Path, output_root: str | Path,
                              config: SegmentationConfig | None = None) -> StageResult:
    cfg = config or SegmentationConfig()
    source = Path(preprocess_summary_csv)
    if not source.is_file():
        raise FileNotFoundError(source)
    df = pd.read_csv(source)
    required = {"status", "analysis_wav_path", "recording_id", "source_sha256"}
    if missing := required - set(df.columns):
        raise ValueError(f"Preprocess summary missing: {sorted(missing)}")
    stage = Path(output_root) / "acoustic" / "002_segmentation"
    folders = ensure_stage_folders(stage, lazy=True)
    rows = []
    errors = []
    model = None
    for _, row in df.iterrows():
        item = {key: row.get(key) for key in ("recording_id", "file_name", "source_file_path",
                "source_sha256", "project_name", "task_name", "run_id", "run_created_at_local",
                "run_created_at_utc", "analysis_wav_path")}
        item["segmentation_method"] = cfg.method
        item["method"] = cfg.method
        base = str(row.get("recording_id", row.get("file_name", "recording"))).replace("/", "_").replace("\\", "_")
        x = None
        sr = None
        intervals = []
        trace = None
        nuclei = []
        stable = None
        breaks = []
        try:
            if str(row["status"]).lower() != "ok":
                raise ValueError(f"Preprocessing status: {row['status']}")
            wav = Path(str(row["analysis_wav_path"]))
            x, sr = _read_canonical_audio(wav)
            if cfg.method == SILERO and model is None:
                model = load_silero_model(onnx=True)
            (intervals, views, frames, audit, metrics, status, flags,
             trace, nuclei, stable, breaks) = _method_result(x, sr, cfg, model)
            item.update(metrics)
            item["analysis_wav_sha256"] = sha256_file(wav)
            segments = _segments(intervals, len(x) / sr)
            boundary_rows = [{"interval_index": i, "start_sec": p.start_sec,
                              "end_sec": p.end_sec, "start_sample": round(p.start_sec * sr),
                              "end_sample": round(p.end_sec * sr), "boundary_source": item["boundary_source"]}
                             for i, p in enumerate(intervals)]
            item["frame_csv_path"] = _write_csv(frames, folders["tables"] / "frames" / f"{base}__frames.csv")
            item["segments_csv_path"] = _write_csv(segments, folders["tables"] / "segments" / f"{base}__segments.csv")
            item["boundaries_csv_path"] = _write_csv(pd.DataFrame(boundary_rows, columns=[
                "interval_index", "start_sec", "end_sec", "start_sample", "end_sample", "boundary_source"]),
                folders["tables"] / "boundaries" / f"{base}__boundaries.csv")
            if cfg.method == SILERO:
                item["boundary_audit_csv_path"] = _write_csv(audit, folders["tables"] / "boundaries" / f"{base}__boundary_audit.csv")
                view_rows = [{"view": name, "start_sec": p.start_sec, "end_sec": p.end_sec}
                             for name, periods in views.items() for p in periods]
                item["interval_views_csv_path"] = _write_csv(pd.DataFrame(view_rows, columns=["view", "start_sec", "end_sec"]),
                    folders["tables"] / "interval_views" / f"{base}__views.csv")
            elif cfg.method == DDK:
                item["nuclei_csv_path"] = _write_csv(pd.DataFrame({"nucleus_index": range(len(nuclei)), "time_sec": nuclei}),
                    folders["tables"] / "boundaries" / f"{base}__nuclei.csv")
            elif cfg.method == PHONATION:
                item["phonation_regions_csv_path"] = _write_csv(pd.DataFrame([
                    {"region": "full_phonation", "start_sec": p.start_sec, "end_sec": p.end_sec} for p in intervals] +
                    ([{"region": "stable_analysis", "start_sec": stable.start_sec, "end_sec": stable.end_sec}] if stable else []) +
                    [{"region": "internal_break", "start_sec": p.start_sec, "end_sec": p.end_sec} for p in breaks]),
                    folders["tables"] / "interval_views" / f"{base}__phonation_regions.csv")
        except Exception as exc:  # Batch review retains every input.
            status = "FAILED" if str(row["status"]).lower() == "ok" else "EXCLUDED"
            flags = ["segmentation_error" if status == "FAILED" else "preprocess_not_accepted"]
            item["error"] = str(exc)
            if status == "FAILED":
                errors.append({"recording_id": item["recording_id"], "file_name": item["file_name"], "error": str(exc)})
        item["automatic_status"] = status
        item["status"] = "ok" if status in {"ACCEPTED", "REVIEW"} else status.lower()
        item["flags"] = ";".join(flags)
        item["review_required"] = status in {"REVIEW", "EXCLUDED", "FAILED"}
        plot_group = {"ACCEPTED": "accepted", "REVIEW": "flagged",
                      "EXCLUDED": "excluded", "FAILED": "excluded"}[status]
        plot = folders["plots"] / plot_group / f"{base}__{cfg.method}.png"
        _plot(plot, x, sr, cfg.method, intervals, status, flags, trace, nuclei, stable, breaks)
        item["plot_png_path"] = str(plot)
        rows.append(item)
    summary = pd.DataFrame(rows)
    if summary.empty:
        summary = pd.DataFrame(columns=["recording_id", "file_name", "task_name",
                                        "segmentation_method", "automatic_status", "flags",
                                        "review_required", "status", "frame_csv_path",
                                        "segments_csv_path", "boundaries_csv_path", "plot_png_path"])
    summary["accepted_outlier"] = False
    summary["accepted_outlier_max_abs_robust_z"] = np.nan
    if cfg.method == SILERO and len(summary) >= 20:
        accepted = summary.automatic_status.eq("ACCEPTED")
        scores = []
        for column in ("duration_sec", "speech_fraction", "n_speech_segments",
                       "n_internal_nonspeech_segments", "leading_nonspeech_sec",
                       "trailing_nonspeech_sec", "longest_internal_nonspeech_sec",
                       "rms_db_median", "rms_db_std"):
            if column not in summary:
                continue
            values = pd.to_numeric(summary[column], errors="coerce")
            reference = values.loc[accepted & values.notna()]
            if len(reference) < 20:
                continue
            median = float(reference.median())
            mad = float(np.median(np.abs(reference.to_numpy() - median)))
            if mad > 0 and np.isfinite(mad):
                scores.append((0.6744897501960817 * (values - median) / mad).abs())
        if scores:
            maximum = pd.concat(scores, axis=1).max(axis=1)
            summary["accepted_outlier_max_abs_robust_z"] = maximum
            summary["accepted_outlier"] = accepted & maximum.ge(4.5)
            summary.loc[summary.accepted_outlier, "review_required"] = True
    summary_path = Path(_write_csv(summary, folders["tables"] / "acoustic_segmentation_summary.csv"))
    main_path = Path(_write_csv(summary, folders["tables"] / "acoustic_segmentation_main_summary.csv"))
    queue_path = Path(_write_csv(summary, folders["tables"] / "segmentation_review_queue.csv"))
    error_path = Path(_write_csv(pd.DataFrame(errors), folders["errors"] / "acoustic_segmentation_errors.csv")) if errors else None
    stage_status = "failed" if not rows or all(r["status"] in {"failed", "excluded"} for r in rows) else ("completed_with_warnings" if summary.review_required.any() else "completed")
    outputs = [ArtifactRef(path=str(p), role=role, media_type="text/csv") for p, role in
               ((summary_path, "segmentation_summary"), (main_path, "segmentation_main_summary"),
                (queue_path, "segmentation_review_queue"))]
    if error_path:
        outputs.append(ArtifactRef(path=str(error_path), role="segmentation_errors", media_type="text/csv"))
    manifest = StageManifest(stage_name="acoustic_segmentation", stage_version="2.0.0", status=stage_status,
        input_artifacts=[ArtifactRef(path=str(source), role="preprocess_summary", media_type="text/csv", sha256=sha256_file(source))],
        output_artifacts=outputs, config=asdict(cfg), environment={"python": python_environment()},
        warnings=[f"{int(summary.review_required.sum())} recordings require review"], errors=errors,
        notes=["Canonical native-rate FLOAT32 audio was read only.", "Display frames never set primary boundaries."])
    manifest_path = folders["logs"] / "stage_manifest.json"
    manifest.write_json(manifest_path)
    return StageResult(status=stage_status, manifest_path=manifest_path, summary_table=summary_path,
                       error_table=error_path, report_path=None)
