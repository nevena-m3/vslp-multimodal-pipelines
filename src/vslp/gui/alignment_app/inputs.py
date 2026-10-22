"""Prepare an alignment domain without running acoustic speech segmentation.

The legacy alignment engine reads the reviewed-domain CSV names.  In an
Alignment-only project these rows mean *possible trial extent*, not a claim
that the whole file is speech.  A human must confirm trial bounds before MFA.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vslp.acoustic.alignment.task_workflow import load_project_choices, save_project_choices
from vslp.acoustic.ingest.stage import run_acoustic_ingest
from vslp.acoustic.preprocess.stage import PreprocessConfig, run_acoustic_preprocess


def prepare_alignment_inputs(input_path: Path, output_root: Path,
                             progress_callback=None) -> dict:
    ingest = run_acoustic_ingest(input_path, output_root, progress_callback=progress_callback)
    preprocess = run_acoustic_preprocess(
        ingest.summary_table, output_root, PreprocessConfig(),
        progress_callback=progress_callback)
    table = pd.read_csv(preprocess.summary_table, keep_default_na=False)
    decisions: list[dict] = []
    intervals: list[dict] = []
    for row in table.itertuples():
        wav = Path(str(row.analysis_wav_path)) if row.analysis_wav_path else None
        if str(row.status).lower() not in {"success", "completed", "ok"} or not wav or not wav.is_file():
            continue
        duration = float(row.analysis_duration_sec)
        if duration <= 0:
            continue
        decisions.append({
            "recording_id": row.recording_id,
            "file_name": row.file_name,
            "final_decision": "KEEP_MANUAL",
            "analysis_wav_path": str(wav),
            "analysis_start_sec": 0.0,
            "analysis_end_sec": duration,
            "segmentation_method": "alignment_full_recording_scope",
            "review_run_id": "alignment_scope_v1",
            "source_sha256": row.source_sha256,
            "source_file_path": str(getattr(row, "source_file_path", "")),
        })
        intervals.append({
            "recording_id": row.recording_id,
            "file_name": row.file_name,
            "view": "authoritative",
            "segment_role": "speech",
            "start_sec": 0.0,
            "end_sec": duration,
            "duration_sec": duration,
            "boundary_source": "ALIGNMENT_SCOPE_UNREVIEWED",
            "segmentation_method": "alignment_full_recording_scope",
        })
    if not decisions:
        raise ValueError("No audio could be prepared for alignment")
    final = output_root / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(decisions).to_csv(final / "final_segmentation_decisions.csv", index=False)
    pd.DataFrame(intervals).to_csv(final / "final_segmentation_intervals.csv", index=False)
    choices = load_project_choices(output_root)
    choices["manual_trial_review_required"] = True
    save_project_choices(output_root, choices)
    return {"ingest": ingest, "preprocess": preprocess, "recording_count": len(decisions)}
