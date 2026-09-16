"""Canonical audio reader and pinned Silero compatibility entry point."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

from vslp.core.schemas import StageResult


SILERO_SAMPLE_RATE_HZ = 16000


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


def run_acoustic_segmentation_silero(
    preprocess_summary_csv: str | Path,
    output_root: str | Path,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 0,
    frame_ms: int = 30,
    silero_repo_or_dir: str = "snakers4/silero-vad",
    force_reload: bool = False,
) -> StageResult:
    """Compatibility entry point for the version-pinned ONNX Silero stage."""
    from vslp.acoustic.segment.pipeline import SegmentationConfig, run_acoustic_segmentation

    return run_acoustic_segmentation(
        preprocess_summary_csv, output_root,
        SegmentationConfig(threshold=threshold, min_speech_duration_ms=min_speech_duration_ms,
                           min_silence_duration_ms=min_silence_duration_ms,
                           speech_pad_ms=speech_pad_ms, frame_ms=frame_ms),
    )


MAIN_SEGMENTATION_SUMMARY_COLUMNS = [
    "recording_id", "file_name", "task_name", "run_id", "status", "method",
    "automatic_status", "speech_fraction", "analysis_wav_path", "frame_csv_path",
    "segments_csv_path", "boundaries_csv_path", "plot_png_path",
]


def _write_main_segmentation_summary(summary_df: pd.DataFrame, path: Path) -> None:
    """Maintain the existing public empty-summary CSV contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.reindex(columns=MAIN_SEGMENTATION_SUMMARY_COLUMNS).to_csv(path, index=False)
