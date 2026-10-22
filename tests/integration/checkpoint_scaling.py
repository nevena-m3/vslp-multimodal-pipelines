"""Run checkpoint orchestration scaling without invoking scientific features.

This probes cache/CSV memory and overhead only. It is deliberately not a
substitute for a full feature-family or Windows CPP process benchmark.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
import soundfile as sf

from vslp.acoustic.features import checkpoint_v1
from vslp.acoustic.features.stage import FeatureExtractionConfig
from vslp.core.schemas import StageResult


def run_size(root: Path, n: int, *, real_timing: bool = False) -> dict:
    root.mkdir(parents=True)
    wav = root / "engineering.wav"
    sr = 16000
    sf.write(wav, .1 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr), sr)
    project = {"task_id": "bamboo_passage", "task_type": "Passage / connected speech",
               "task_name": "Bamboo Passage", "run_id": "engineering"}
    (root / "project_manifest.json").write_text(json.dumps(project), encoding="utf-8")
    final = root / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    kept = pd.DataFrame([{
        "recording_id": f"r{i:04d}", "file_name": f"engineering_{i:04d}.wav",
        "analysis_wav_path": str(wav), "source_file_path": str(wav),
        "segmentation_run_id": "seg", "review_run_id": "review",
        "final_decision": "KEEP_AUTO", "boundary_source": "HUMAN",
    } for i in range(n)])
    kept.to_csv(decisions, index=False)
    pd.DataFrame([{"recording_id": f"r{i:04d}", "view": "authoritative",
                   "segment_role": role, "start_sec": start, "end_sec": end,
                   "boundary_source": "HUMAN", "analysis_start_sec": 0,
                   "analysis_end_sec": 1}
                  for i in range(n)
                  for role, start, end in (("speech", .1, .35),
                                           ("internal_nonspeech", .35, .65),
                                           ("speech", .65, .9))]).to_csv(intervals, index=False)
    feature_id = "bamboo_pause_count" if real_timing else "engineering_value"
    if real_timing:
        from vslp.acoustic.features.catalog import load_feature_catalog
        from vslp.acoustic.features.stage import _select_registry

        registry = _select_registry(FeatureExtractionConfig(selected_features=[feature_id]))
        approved = {item["feature_id"]: item for item in load_feature_catalog()["outputs"]}
        from vslp.acoustic.features.family09_v2_stage import run_family09_stage

        execute_stage = run_family09_stage
    else:
        registry = pd.DataFrame([{"feature": feature_id, "family_id": "F09",
                                  "unit": "count", "algorithm_version": "engineering-only",
                                  "parameter_set_id": "engineering-only",
                                  "output_granularity": "recording"}])
        approved = {feature_id: {"family_id": "F09"}}
        execute_stage = None
    lookup = registry.set_index("feature")
    stage = root / "acoustic" / "005_features"

    def execute(_decisions, _root, _config, _intervals, _registry,
                execution_stage_dir=None, recording_scope=None, **_kwargs):
        record_id = str(recording_scope.kept.iloc[0].recording_id)
        tables = execution_stage_dir / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        status = tables / "acoustic_feature_status_long.csv"
        pd.DataFrame([{"recording_id": record_id, "file_name": f"{record_id}.wav",
                       "feature_id": feature_id, "value": int(record_id[1:]),
                       "status": "computed", "failure_reason": ""}]).to_csv(status, index=False)
        manifest = execution_stage_dir / "logs" / "stage_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}", encoding="utf-8")
        return StageResult("completed", manifest, status)

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    peak_children = 0

    def progress(done, _total, _message):
        nonlocal peak_rss, peak_children
        peak_rss = max(peak_rss, process.memory_info().rss)
        if done % 10 == 0:
            peak_children = max(peak_children, len(process.children(recursive=True)))

    config = FeatureExtractionConfig(selected_features=[feature_id],
                                     resume_existing_run=True)

    def call():
        return checkpoint_v1.run_checkpointed_stage(
            decisions, intervals, root, stage, stage / "tables", config, registry,
            kept, project, "bamboo_passage", "Passage / connected speech", "Bamboo Passage",
            [feature_id], approved, lookup,
            {"timing": ({"F09"}, execute_stage or execute)}, {"timing": [feature_id]},
            [], [], False, progress,
        )

    original = checkpoint_v1.requires_alignment_for_task
    checkpoint_v1.requires_alignment_for_task = lambda *_args: False
    try:
        started = perf_counter()
        result = call()
        first_seconds = perf_counter() - started
        first_metrics = json.loads(result.manifest_path.read_text(encoding="utf-8"))["config"]
        started = perf_counter()
        resumed = call()
        resume_seconds = perf_counter() - started
    finally:
        checkpoint_v1.requires_alignment_for_task = original
    metrics = json.loads(resumed.manifest_path.read_text(encoding="utf-8"))["config"]
    rows = sum(len(chunk) for chunk in pd.read_csv(result.summary_table, chunksize=64))
    computed = sum(int(chunk.status.eq("computed").sum()) for chunk in pd.read_csv(
        result.summary_table.parent / "acoustic_feature_status_long.csv", chunksize=64))
    if real_timing and computed != n:
        raise AssertionError(f"Only {computed}/{n} reviewed-timing outputs computed")
    return {"mode": "real_F09" if real_timing else "trivial_executor",
            "recordings": n, "first_wall_s": round(first_seconds, 3),
            "resume_wall_s": round(resume_seconds, 3),
            "peak_parent_rss_mb": round(peak_rss / 1024**2, 2),
            "peak_child_count": peak_children,
            "checkpoint_count": len(list((stage / "checkpoints_v1" / "timing").glob("*.json"))),
            "result_count": rows, "computed_count": computed,
            "resume_counts": metrics["checkpoint_counts"],
            "first_timing": first_metrics["performance_timing_seconds"],
            "resume_timing": metrics["performance_timing_seconds"]}


if __name__ == "__main__":
    real = "--real-timing" in sys.argv
    with tempfile.TemporaryDirectory(prefix="vslp-checkpoint-scale-") as directory:
        for size in (10, 25, 50, 100):
            print(json.dumps(run_size(Path(directory) / str(size), size,
                                      real_timing=real)), flush=True)
