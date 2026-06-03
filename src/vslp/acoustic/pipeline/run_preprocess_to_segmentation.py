"""Convenience orchestration for the first runnable acoustic backend path."""

from __future__ import annotations

from pathlib import Path

from vslp.acoustic.ingest.stage import run_acoustic_ingest
from vslp.acoustic.preprocess.stage import PreprocessConfig, run_acoustic_preprocess
from vslp.acoustic.segment.stage import run_acoustic_segmentation_silero
from vslp.core.project import initialize_project


def run_acoustic_ingest_preprocess_segment(
    input_path: str | Path,
    output_root: str | Path,
    project_name: str = "vslp_project",
    run_segmentation: bool = True,
    preprocess_config: PreprocessConfig | None = None,
):
    """Run project init -> ingest -> preprocess -> optional Silero segmentation."""
    initialize_project(output_root=output_root, project_name=project_name)
    ingest_result = run_acoustic_ingest(input_path=input_path, output_root=output_root)
    preprocess_result = run_acoustic_preprocess(
        input_path=input_path,
        output_root=output_root,
        config=preprocess_config or PreprocessConfig(),
    )
    segmentation_result = None
    if run_segmentation and preprocess_result.summary_table is not None:
        segmentation_result = run_acoustic_segmentation_silero(
            preprocess_summary_csv=preprocess_result.summary_table,
            output_root=output_root,
        )
    return {
        "ingest": ingest_result,
        "preprocess": preprocess_result,
        "segmentation": segmentation_result,
    }
