"""VSLP command line interface.

Every GUI action should eventually map to an equivalent CLI command/config.
"""

from __future__ import annotations

from pathlib import Path

import typer

from vslp.acoustic.features.registry import build_acoustic_feature_registry
from vslp.acoustic.features.stage import FeatureExtractionConfig, run_acoustic_feature_extraction
from vslp.acoustic.ingest.stage import run_acoustic_ingest
from vslp.acoustic.metadata.stage import MetadataConfig, run_acoustic_metadata
from vslp.acoustic.pipeline.run_preprocess_to_segmentation import run_acoustic_ingest_preprocess_segment
from vslp.acoustic.preprocess.stage import FilterConfig, PreprocessConfig, run_acoustic_preprocess
from vslp.acoustic.segment.stage import run_acoustic_segmentation_silero
from vslp.core.project import initialize_project
from vslp.core.doctor import run_doctor

app = typer.Typer(help="VSLP multimodal pipelines")
project_app = typer.Typer(help="Project management")
acoustic_app = typer.Typer(help="Acoustic pipeline")
features_app = typer.Typer(help="Feature utilities")
gui_app = typer.Typer(help="Desktop GUIs")
app.add_typer(project_app, name="project")
app.add_typer(acoustic_app, name="acoustic")
app.add_typer(features_app, name="features")
app.add_typer(gui_app, name="gui")


@app.command("doctor")
def doctor(include_silero: bool = True):
    """Check whether the local VSLP environment is ready."""
    checks = run_doctor(include_silero=include_silero)
    all_ok = True
    for c in checks:
        status = "OK" if c.ok else "FAIL"
        typer.echo(f"[{status}] {c.name}: {c.detail}")
        if not c.ok:
            all_ok = False
            if c.fix:
                typer.echo(f"      Fix: {c.fix}")
    if not all_ok:
        raise typer.Exit(code=1)


@project_app.command("init")
def project_init(output_root: Path, project_name: str = "vslp_project"):
    """Create a VSLP project/output directory."""
    paths = initialize_project(output_root=output_root, project_name=project_name)
    typer.echo(f"Created VSLP project: {paths.root}")


@acoustic_app.command("metadata")
def acoustic_metadata(input: Path, output_root: Path, demographics_csv: Path | None = None):
    """Create a project file index from optional demographics CSV and filename parsing."""
    cfg = MetadataConfig(demographics_csv=str(demographics_csv) if demographics_csv else None)
    result = run_acoustic_metadata(input_path=input, output_root=output_root, config=cfg)
    typer.echo(f"Status: {result.status}")
    typer.echo(f"File index: {result.summary_table}")
    typer.echo(f"Report: {result.report_path}")
    typer.echo(f"Manifest: {result.manifest_path}")


@acoustic_app.command("ingest")
def acoustic_ingest(input: Path, output_root: Path, ffprobe_bin: str = "ffprobe"):
    """Digest input audio/video files using ffprobe."""
    result = run_acoustic_ingest(input_path=input, output_root=output_root, ffprobe_bin=ffprobe_bin)
    typer.echo(f"Status: {result.status}")
    typer.echo(f"Summary: {result.summary_table}")
    typer.echo(f"Errors: {result.error_table}")
    typer.echo(f"Manifest: {result.manifest_path}")


@acoustic_app.command("preprocess")
def acoustic_preprocess(
    input: Path,
    output_root: Path,
    segmentation_sample_rate_hz: int = 16000,
    feature_sample_rate_hz: int | None = None,
    ffmpeg_bin: str = "ffmpeg",
    filter_kind: str = "none",
    low_hz: float | None = None,
    high_hz: float | None = None,
    notch_hz: float | None = None,
):
    """Create canonical WAVs and QC tables/reports."""
    filter_cfg = FilterConfig(
        enabled=filter_kind != "none",
        kind=filter_kind,
        low_hz=low_hz,
        high_hz=high_hz,
        notch_hz=notch_hz,
    )
    cfg = PreprocessConfig(
        segmentation_sample_rate_hz=segmentation_sample_rate_hz,
        feature_sample_rate_hz=feature_sample_rate_hz,
        filter=filter_cfg,
        ffmpeg_bin=ffmpeg_bin,
    )
    result = run_acoustic_preprocess(input_path=input, output_root=output_root, config=cfg)
    typer.echo(f"Status: {result.status}")
    typer.echo(f"Summary: {result.summary_table}")
    typer.echo(f"Errors: {result.error_table}")
    typer.echo(f"Report: {result.report_path}")
    typer.echo(f"Manifest: {result.manifest_path}")


@acoustic_app.command("segment-silero")
def acoustic_segment_silero(
    preprocess_summary_csv: Path,
    output_root: Path,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 50,
    frame_ms: int = 30,
    silero_repo_or_dir: str = "snakers4/silero-vad",
    force_reload: bool = False,
):
    """Run Silero VAD segmentation from a preprocess summary CSV."""
    result = run_acoustic_segmentation_silero(
        preprocess_summary_csv=preprocess_summary_csv,
        output_root=output_root,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        frame_ms=frame_ms,
        silero_repo_or_dir=silero_repo_or_dir,
        force_reload=force_reload,
    )
    typer.echo(f"Status: {result.status}")
    typer.echo(f"Summary: {result.summary_table}")
    typer.echo(f"Errors: {result.error_table}")
    typer.echo(f"Report: {result.report_path}")
    typer.echo(f"Manifest: {result.manifest_path}")


@acoustic_app.command("run-v1")
def acoustic_run_v1(
    input: Path,
    output_root: Path,
    project_name: str = "vslp_project",
    skip_segmentation: bool = False,
):
    """Run the first backend path: ingest -> preprocess -> optional Silero segmentation."""
    results = run_acoustic_ingest_preprocess_segment(
        input_path=input,
        output_root=output_root,
        project_name=project_name,
        run_segmentation=not skip_segmentation,
    )
    for name, result in results.items():
        if result is None:
            typer.echo(f"{name}: skipped")
        else:
            typer.echo(f"{name}: {result.status} | summary={result.summary_table} | manifest={result.manifest_path}")


@acoustic_app.command("extract-features")
def acoustic_extract_features(
    segmentation_summary_csv: Path,
    output_root: Path,
    minimum_pause_duration_sec: float = 0.15,
    metadata_csv: Path | None = None,
):
    """Extract currently implemented acoustic features from segmentation outputs."""
    cfg = FeatureExtractionConfig(
        minimum_pause_duration_sec=minimum_pause_duration_sec,
        metadata_csv=str(metadata_csv) if metadata_csv else None,
    )
    result = run_acoustic_feature_extraction(
        segmentation_summary_csv=segmentation_summary_csv,
        output_root=output_root,
        config=cfg,
    )
    typer.echo(f"Status: {result.status}")
    typer.echo(f"Features: {result.summary_table}")
    typer.echo(f"Errors: {result.error_table}")
    typer.echo(f"Report: {result.report_path}")
    typer.echo(f"Manifest: {result.manifest_path}")


@gui_app.command("acoustic")
def gui_acoustic():
    """Launch the acoustic pipeline desktop GUI."""
    try:
        from vslp.gui.acoustic_app.app import launch_acoustic_gui
        raise typer.Exit(code=launch_acoustic_gui())
    except RuntimeError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)


@features_app.command("registry")
def features_registry(output_csv: Path = Path("acoustic_feature_registry.csv")):
    """Export the acoustic feature registry."""
    df = build_acoustic_feature_registry()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    typer.echo(f"Wrote {len(df)} features to {output_csv}")


if __name__ == "__main__":
    app()
