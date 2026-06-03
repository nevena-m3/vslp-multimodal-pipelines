"""VSLP command line interface.

Every GUI action should eventually map to an equivalent CLI command/config.
"""

from __future__ import annotations

from pathlib import Path
import typer

from vslp.core.project import initialize_project
from vslp.acoustic.ingest.stage import run_acoustic_ingest
from vslp.acoustic.features.registry import build_acoustic_feature_registry

app = typer.Typer(help="VSLP multimodal pipelines")
project_app = typer.Typer(help="Project management")
acoustic_app = typer.Typer(help="Acoustic pipeline")
features_app = typer.Typer(help="Feature utilities")
app.add_typer(project_app, name="project")
app.add_typer(acoustic_app, name="acoustic")
app.add_typer(features_app, name="features")


@project_app.command("init")
def project_init(output_root: Path, project_name: str = "vslp_project"):
    paths = initialize_project(output_root=output_root, project_name=project_name)
    typer.echo(f"Created VSLP project: {paths.root}")


@acoustic_app.command("ingest")
def acoustic_ingest(input: Path, output_root: Path, ffprobe_bin: str = "ffprobe"):
    result = run_acoustic_ingest(input_path=input, output_root=output_root, ffprobe_bin=ffprobe_bin)
    typer.echo(f"Status: {result.status}")
    typer.echo(f"Summary: {result.summary_table}")
    typer.echo(f"Manifest: {result.manifest_path}")


@features_app.command("registry")
def features_registry(output_csv: Path = Path("acoustic_feature_registry.csv")):
    df = build_acoustic_feature_registry()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    typer.echo(f"Wrote {len(df)} features to {output_csv}")


if __name__ == "__main__":
    app()
