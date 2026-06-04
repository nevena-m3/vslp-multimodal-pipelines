"""Acoustic segmentation stage.

V1 supports:
- Silero VAD, using the uploaded reference wrapper contract and diagnostic plot style;
- Energy/RMS placeholder fallback, implemented separately.

The stage expects preprocessed segmentation WAV files from 002_preprocess by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf

from vslp.acoustic.segment.silero_wrapper import build_silero_stage_from_audio, plot_silero_stage, summarize_silero_stage
from vslp.core.project import ensure_stage_folders
from vslp.core.provenance import python_environment, sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


def load_silero_model(repo_or_dir: str = "snakers4/silero-vad", force_reload: bool = False):
    """Load Silero VAD through torch.hub with production-grade error messages.

    Notes
    -----
    - Patient data is never uploaded by this function.
    - The first call may download the public Silero model/code from GitHub unless a
      local repo/cache is used. For locked-down clinical/research installs, cache or
      vendor the model before runtime.
    - Torch/Torchaudio are optional dependencies so ingest/preprocess can run without
      ML/VAD installed.
    """
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
            "Could not load Silero VAD through torch.hub. Common causes: no internet on "
            "first run, GitHub blocked, or the Silero repo/model is not cached locally. "
            "Try once while online, or pass --silero-repo-or-dir /path/to/local/silero-vad. "
            f"Original error: {exc}"
        ) from exc

    (get_speech_timestamps, *_rest) = utils
    return model, get_speech_timestamps


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
    """Run Silero segmentation for each successfully preprocessed file."""
    preprocess_summary_csv = Path(preprocess_summary_csv)
    stage_dir = Path(output_root) / "acoustic" / "003_segmentation"
    folders = ensure_stage_folders(stage_dir)

    df = pd.read_csv(preprocess_summary_csv)
    if "segmentation_wav_path" not in df.columns:
        raise ValueError("Preprocess summary must contain segmentation_wav_path")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    model = None
    get_speech_timestamps = None

    for _, row in df.iterrows():
        wav_path = Path(str(row.get("segmentation_wav_path", "")))
        file_name = str(row.get("file_name", wav_path.name))
        try:
            if not wav_path.exists():
                raise FileNotFoundError(f"Missing segmentation WAV: {wav_path}")
            if model is None or get_speech_timestamps is None:
                model, get_speech_timestamps = load_silero_model(silero_repo_or_dir, force_reload=force_reload)

            x, sr = sf.read(wav_path, always_2d=False, dtype="float32")
            stage = build_silero_stage_from_audio(
                x=x,
                sr=int(sr),
                model=model,
                get_speech_timestamps_fn=get_speech_timestamps,
                threshold=threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
                return_seconds=False,
                frame_ms=frame_ms,
            )

            base = wav_path.stem.replace("__seg16k", "")
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
            plot_silero_stage(stage, file_name=file_name, save_path=plot_png, show=False)

            summary = summarize_silero_stage(stage, row=row)
            summary.update(
                {
                    "file_name": file_name,
                    "source_file_path": row.get("file_path"),
                    "segmentation_wav_path": str(wav_path),
                    "segmentation_wav_sha256": sha256_file(wav_path),
                    "frame_csv_path": str(frame_csv),
                    "segments_csv_path": str(segments_csv),
                    "boundaries_csv_path": str(boundaries_csv),
                    "plot_png_path": str(plot_png),
                    "status": "ok",
                }
            )
            rows.append(summary)
        except Exception as exc:  # noqa: BLE001 - batch should continue
            errors.append({"file_name": file_name, "segmentation_wav_path": str(wav_path), "status": "failed", "error": str(exc)})

    summary_path = folders["tables"] / "acoustic_segmentation_summary.csv"
    main_summary_path = folders["tables"] / "acoustic_segmentation_main_summary.csv"
    errors_path = folders["errors"] / "acoustic_segmentation_errors.csv"
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_path, index=False)
    _write_main_segmentation_summary(summary_df, main_summary_path)
    pd.DataFrame(errors).to_csv(errors_path, index=False)

    report_path = folders["reports"] / "acoustic_segmentation_report.html"
    _write_segmentation_html_report(report_path, rows, errors)

    manifest = StageManifest(
        stage_name="acoustic_segmentation_silero",
        stage_version="0.1.0",
        status="completed_with_warnings" if errors else "completed",
        input_artifacts=[ArtifactRef(path=str(preprocess_summary_csv), role="preprocess_summary", media_type="text/csv")],
        output_artifacts=[
            ArtifactRef(path=str(summary_path), role="segmentation_summary", media_type="text/csv"),
            ArtifactRef(path=str(main_summary_path), role="segmentation_main_summary", media_type="text/csv"),
            ArtifactRef(path=str(errors_path), role="segmentation_errors", media_type="text/csv"),
            ArtifactRef(path=str(report_path), role="segmentation_html_report", media_type="text/html"),
        ],
        config={
            "method": "silero_vad",
            "method_family": "speech_pause_vad",
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
        notes=["Segmentation plots follow the uploaded Silero reference style."],
    )
    manifest_path = folders["logs"] / "stage_manifest.json"
    manifest.write_json(manifest_path)

    return StageResult(
        status=manifest.status,
        manifest_path=manifest_path,
        summary_table=summary_path,
        error_table=errors_path,
        report_path=report_path,
    )


def _write_main_segmentation_summary(summary_df: pd.DataFrame, path: Path) -> None:
    """Write a compact user-facing segmentation summary table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    preferred = [
        "file_name",
        "status",
        "method",
        "duration_sec",
        "sample_rate_analysis",
        "n_segments_total",
        "n_speech_segments",
        "n_internal_nonspeech_segments",
        "speech_fraction",
        "leading_nonspeech_sec",
        "trailing_nonspeech_sec",
        "longest_internal_nonspeech_sec",
        "rms_db_median",
        "rms_db_std",
        "frame_csv_path",
        "segments_csv_path",
        "boundaries_csv_path",
        "plot_png_path",
    ]
    cols = [c for c in preferred if c in summary_df.columns]
    summary_df.loc[:, cols].to_csv(path, index=False)


def _write_segmentation_html_report(path: Path, rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = len(rows)
    failed = len(errors)
    mean_speech_fraction = None
    if rows:
        vals = [r.get("speech_fraction") for r in rows if pd.notna(r.get("speech_fraction"))]
        if vals:
            mean_speech_fraction = sum(float(v) for v in vals) / len(vals)
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>VSLP Acoustic Segmentation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; background:#071A2D; color:#EAF2F8; margin:32px; }}
.card {{ background:#0B253D; border:1px solid #315D7C; border-radius:12px; padding:18px; margin:14px 0; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:6px; background:#0E3A5B; margin-right:8px; }}
a {{ color:#8FCBFF; }}
</style></head><body>
<h1>VSLP Acoustic Segmentation Report</h1>
<div class='card'><span class='badge'>Segmented: {ok}</span><span class='badge'>Failed: {failed}</span><span class='badge'>Mean speech fraction: {mean_speech_fraction}</span></div>
<div class='card'><p>Per-file frame tables, segment tables, boundary tables, and diagnostic plots are written under the segmentation stage folders.</p></div>
</body></html>"""
    path.write_text(html, encoding="utf-8")
