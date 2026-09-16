"""Acoustic segmentation stage.

Silero consumes the canonical native-rate mono waveform produced by Preprocess.
Model-specific resampling to 16 kHz happens in memory inside this stage; no separate
"segmentation WAV" is created.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

from vslp.acoustic.context import cleanup_stage
from vslp.acoustic.segment.silero_wrapper import (
    build_silero_stage_from_audio,
    plot_silero_stage,
    summarize_silero_stage,
)
from vslp.core.project import ensure_stage_folders
from vslp.core.provenance import python_environment, sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


SILERO_SAMPLE_RATE_HZ = 16000


def load_silero_model(repo_or_dir: str = "snakers4/silero-vad", force_reload: bool = False):
    """Load Silero VAD through torch.hub with explicit runtime diagnostics."""
    try:
        import torch  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Silero segmentation requires PyTorch. Activate your VSLP virtual environment "
            "and run: pip install -e '.[silero]'"
        ) from exc

    try:
        import torchaudio  # noqa: F401, PLC0415
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Silero segmentation requires torchaudio. Activate your VSLP virtual environment "
            "and run: pip install -e '.[silero]'"
        ) from exc

    try:
        model, utils = torch.hub.load(
            repo_or_dir,
            "silero_vad",
            force_reload=force_reload,
            trust_repo=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not load Silero VAD through torch.hub. The first run may require "
            "network access unless the model/repository is already cached or vendored. "
            f"Original error: {exc}"
        ) from exc

    (get_speech_timestamps, *_rest) = utils
    return model, get_speech_timestamps


def _read_canonical_audio(path: Path) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical analysis WAV: {path}")
    info = sf.info(path)
    if str(info.format).upper() != "WAV":
        raise ValueError(f"Canonical analysis file must be WAV, got {info.format}: {path}")
    if str(info.subtype).upper() != "FLOAT":
        raise ValueError(
            f"Canonical analysis WAV must use FLOAT subtype, got {info.subtype}: {path}"
        )
    if int(info.channels) != 1:
        raise ValueError(f"Canonical analysis WAV must be mono, got {info.channels}: {path}")

    x, sr = sf.read(path, dtype="float32", always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"Canonical analysis WAV must contain non-empty mono audio: {path}")
    if not np.isfinite(x).all():
        raise ValueError(f"Canonical analysis WAV contains non-finite samples: {path}")
    return x, int(sr)


def _prepare_silero_audio(
    x: np.ndarray,
    sr: int,
    target_sr: int = SILERO_SAMPLE_RATE_HZ,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create the private Silero working signal with explicit audit metadata."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Silero preparation requires non-empty mono audio")
    if not np.isfinite(x).all():
        raise ValueError("Silero preparation received non-finite samples")
    if int(sr) <= 0 or int(target_sr) <= 0:
        raise ValueError("Sample rates must be positive")

    source_duration = len(x) / float(sr)
    if int(sr) == int(target_sr):
        y = x.copy()
        resampled = False
    else:
        divisor = math.gcd(int(sr), int(target_sr))
        up = int(target_sr) // divisor
        down = int(sr) // divisor
        y = signal.resample_poly(x, up=up, down=down).astype(np.float32)
        resampled = True

    expected_length = int(round(source_duration * target_sr))
    if abs(len(y) - expected_length) > 1:
        raise RuntimeError(
            "Silero resampling produced an unexpected sample count: "
            f"expected≈{expected_length}, observed={len(y)}"
        )
    output_duration = len(y) / float(target_sr)
    duration_error = abs(output_duration - source_duration)
    duration_tolerance = (1.0 / float(sr)) + (1.0 / float(target_sr)) + 1e-9
    if duration_error > duration_tolerance:
        raise RuntimeError(
            "Silero resampling changed duration beyond one-sample tolerance: "
            f"error={duration_error:.9f}s, tolerance={duration_tolerance:.9f}s"
        )
    if not np.isfinite(y).all():
        raise RuntimeError("Silero resampling produced non-finite samples")

    # Silero's waveform convention is normalized floating-point audio. Polyphase
    # resampling can create tiny overshoots near sharp peaks; clipping here is
    # algorithm-specific, never written back to the canonical waveform, and is
    # explicitly quantified below.
    clip_mask = np.abs(y) > 1.0
    clip_count = int(np.count_nonzero(clip_mask))
    if clip_count:
        y = np.clip(y, -1.0, 1.0).astype(np.float32)

    return y, {
        "source_analysis_sample_rate_hz": int(sr),
        "silero_input_sample_rate_hz": int(target_sr),
        "silero_input_resampled": bool(resampled),
        "silero_input_n_samples": int(len(y)),
        "silero_input_duration_sec": float(len(y) / target_sr),
        "silero_input_clipped_sample_count": clip_count,
        "silero_input_clipped_sample_fraction": (
            float(clip_count / len(y)) if len(y) else 0.0
        ),
        "silero_resampling_duration_error_sec": float(duration_error),
    }


@cleanup_stage
def run_acoustic_segmentation_silero(
    preprocess_summary_csv: str | Path,
    output_root: str | Path,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 50,
    frame_ms: int = 30,
    silero_repo_or_dir: str = "snakers4/silero-vad",
    force_reload: bool = False,
) -> StageResult:
    """Run Silero VAD on successfully preprocessed canonical waveforms."""
    preprocess_summary_csv = Path(preprocess_summary_csv)
    stage_dir = Path(output_root) / "acoustic" / "002_segmentation"
    folders = ensure_stage_folders(stage_dir, lazy=True)

    if not preprocess_summary_csv.is_file():
        raise FileNotFoundError(f"Preprocess summary not found: {preprocess_summary_csv}")
    df = pd.read_csv(preprocess_summary_csv)
    required = {"status", "analysis_wav_path", "source_sha256", "recording_id"}
    if not required.issubset(df.columns):
        missing = sorted(required.difference(df.columns))
        raise ValueError(f"Preprocess summary is missing required columns: {missing}")

    eligible = df.loc[df["status"].astype(str).str.lower().eq("ok")].copy()
    if eligible.empty:
        raise ValueError("No successfully preprocessed canonical waveforms are available")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    model = None
    get_speech_timestamps = None

    for _, row in eligible.iterrows():
        wav_path = Path(str(row["analysis_wav_path"]))
        file_name = str(row.get("file_name", wav_path.name))
        try:
            canonical_x, canonical_sr = _read_canonical_audio(wav_path)
            if model is None or get_speech_timestamps is None:
                model, get_speech_timestamps = load_silero_model(
                    silero_repo_or_dir,
                    force_reload=force_reload,
                )

            silero_x, resample_info = _prepare_silero_audio(
                canonical_x,
                canonical_sr,
                SILERO_SAMPLE_RATE_HZ,
            )
            stage = build_silero_stage_from_audio(
                x=silero_x,
                sr=SILERO_SAMPLE_RATE_HZ,
                model=model,
                get_speech_timestamps_fn=get_speech_timestamps,
                threshold=threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
                return_seconds=False,
                frame_ms=frame_ms,
            )

            base = wav_path.stem.removesuffix("__analysis")
            frame_csv = folders["tables"] / "frames" / f"{base}__frames.csv"
            segments_csv = folders["tables"] / "segments" / f"{base}__segments.csv"
            boundaries_csv = folders["tables"] / "boundaries" / f"{base}__boundaries.csv"
            plot_png = folders["plots"] / f"{base}__silero_segmentation.png"

            frame_csv.parent.mkdir(parents=True, exist_ok=True)
            segments_csv.parent.mkdir(parents=True, exist_ok=True)
            boundaries_csv.parent.mkdir(parents=True, exist_ok=True)
            stage["frame_df"].to_csv(frame_csv, index=False)
            stage["segments_df"].to_csv(segments_csv, index=False)
            stage["boundaries_df"].to_csv(boundaries_csv, index=False)
            plot_png.parent.mkdir(parents=True, exist_ok=True)
            plot_silero_stage(stage, file_name=file_name, save_path=plot_png, show=False)

            summary = summarize_silero_stage(stage, row=row)
            summary.update(
                {
                    "file_name": file_name,
                    "source_file_path": row.get("source_file_path"),
                    "source_sha256": row.get("source_sha256"),
                    "recording_id": row.get("recording_id"),
                    "project_name": row.get("project_name"),
                    "task_name": row.get("task_name"),
                    "run_id": row.get("run_id"),
                    "run_created_at_local": row.get("run_created_at_local"),
                    "run_created_at_utc": row.get("run_created_at_utc"),
                    "analysis_wav_path": str(wav_path),
                    "analysis_wav_sha256": sha256_file(wav_path),
                    **resample_info,
                    "frame_csv_path": str(frame_csv),
                    "segments_csv_path": str(segments_csv),
                    "boundaries_csv_path": str(boundaries_csv),
                    "plot_png_path": str(plot_png),
                    "status": "ok",
                }
            )
            rows.append(summary)
        except Exception as exc:  # noqa: BLE001 - batch must continue
            errors.append(
                {
                    "file_name": file_name,
                    "analysis_wav_path": str(wav_path),
                    "recording_id": row.get("recording_id"),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    summary_path = folders["tables"] / "acoustic_segmentation_summary.csv"
    main_summary_path = folders["tables"] / "acoustic_segmentation_main_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    _write_main_segmentation_summary(pd.DataFrame(rows), main_summary_path)

    errors_path: Path | None = None
    if errors:
        errors_path = folders["errors"] / "acoustic_segmentation_errors.csv"
        pd.DataFrame(errors).to_csv(errors_path, index=False)

    report_path = folders["reports"] / "acoustic_segmentation_report.html"
    _write_segmentation_html_report(report_path, rows, errors)

    status = (
        "failed"
        if not rows
        else ("completed_with_warnings" if errors else "completed")
    )
    outputs = [
        ArtifactRef(path=str(summary_path), role="segmentation_summary", media_type="text/csv"),
        ArtifactRef(
            path=str(main_summary_path),
            role="segmentation_main_summary",
            media_type="text/csv",
        ),
        ArtifactRef(path=str(report_path), role="segmentation_html_report", media_type="text/html"),
    ]
    if errors_path is not None:
        outputs.append(
            ArtifactRef(path=str(errors_path), role="segmentation_errors", media_type="text/csv")
        )

    manifest = StageManifest(
        stage_name="acoustic_segmentation_silero",
        stage_version="1.0.0",
        status=status,
        input_artifacts=[
            ArtifactRef(
                path=str(preprocess_summary_csv),
                role="preprocess_summary",
                media_type="text/csv",
                sha256=sha256_file(preprocess_summary_csv),
            )
        ],
        output_artifacts=outputs,
        config={
            "method": "silero_vad",
            "method_family": "speech_pause_vad",
            "silero_input_sample_rate_hz": SILERO_SAMPLE_RATE_HZ,
            "resampling": "scipy.signal.resample_poly; in-memory; model-specific",
            "threshold": threshold,
            "min_speech_duration_ms": min_speech_duration_ms,
            "min_silence_duration_ms": min_silence_duration_ms,
            "speech_pad_ms": speech_pad_ms,
            "frame_ms": frame_ms,
            "silero_repo_or_dir": silero_repo_or_dir,
            "force_reload": force_reload,
        },
        environment={"python": python_environment()},
        warnings=[f"{len(errors)} files failed segmentation"] if errors else [],
        errors=errors,
        notes=[
            "Silero operates on an in-memory 16 kHz working signal.",
            "The canonical native-rate FLOAT32 waveform is not overwritten or replaced.",
            "Any model-input clipping caused by resampling overshoot is recorded per file.",
        ],
    )
    manifest_path = folders["logs"] / "stage_manifest.json"
    manifest.write_json(manifest_path)

    return StageResult(
        status=status,
        manifest_path=manifest_path,
        summary_table=summary_path,
        error_table=errors_path,
        report_path=report_path,
    )


MAIN_SEGMENTATION_SUMMARY_COLUMNS = [
    "recording_id",
    "file_name",
    "source_file_path",
    "source_sha256",
    "project_name",
    "task_name",
    "run_id",
    "run_created_at_local",
    "run_created_at_utc",
    "status",
    "method",
    "duration_sec",
    "source_analysis_sample_rate_hz",
    "silero_input_sample_rate_hz",
    "silero_input_resampled",
    "silero_input_clipped_sample_count",
    "silero_input_clipped_sample_fraction",
    "n_segments_total",
    "n_speech_segments",
    "n_internal_nonspeech_segments",
    "speech_fraction",
    "leading_nonspeech_sec",
    "trailing_nonspeech_sec",
    "longest_internal_nonspeech_sec",
    "rms_db_median",
    "rms_db_std",
    "analysis_wav_path",
    "analysis_wav_sha256",
    "frame_csv_path",
    "segments_csv_path",
    "boundaries_csv_path",
    "plot_png_path",
]


def _write_main_segmentation_summary(summary_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        pd.DataFrame(columns=MAIN_SEGMENTATION_SUMMARY_COLUMNS).to_csv(path, index=False)
        return
    for col in MAIN_SEGMENTATION_SUMMARY_COLUMNS:
        if col not in summary_df.columns:
            summary_df[col] = np.nan
    summary_df.loc[:, MAIN_SEGMENTATION_SUMMARY_COLUMNS].to_csv(path, index=False)


def _write_segmentation_html_report(
    path: Path,
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = len(rows)
    failed = len(errors)
    vals = [
        float(r["speech_fraction"])
        for r in rows
        if pd.notna(r.get("speech_fraction"))
    ]
    mean_speech_fraction = float(np.mean(vals)) if vals else None
    resampled = sum(bool(r.get("silero_input_resampled")) for r in rows)
    clipped = sum(
        int(r.get("silero_input_clipped_sample_count") or 0) > 0 for r in rows
    )
    html_text = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>VSLP Acoustic Segmentation Report</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif; background:#071A2D; color:#EAF2F8; margin:32px; }}
.card {{ background:#0B253D; border:1px solid #315D7C; border-radius:12px; padding:18px; margin:14px 0; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:6px; background:#0E3A5B; margin-right:8px; }}
</style></head><body>
<h1>VSLP Acoustic Segmentation Report</h1>
<div class='card'>
<span class='badge'>Segmented: {ok}</span>
<span class='badge'>Failed: {failed}</span>
<span class='badge'>Internally resampled to 16 kHz: {resampled}</span>
<span class='badge'>Model-input overshoot clipped: {clipped}</span>
<span class='badge'>Mean speech fraction: {mean_speech_fraction}</span>
</div>
<div class='card'>
Silero consumes the canonical native-rate preprocessing waveform. Resampling to 16 kHz
is performed only in memory for the VAD model; no secondary segmentation WAV is created.
</div>
</body></html>"""
    path.write_text(html_text, encoding="utf-8")
