"""Opt-in, metadata-free audit of the Tanchip Energy implementation.

This utility never runs automatically. It reads only audio in a caller-provided
folder and writes summaries to a caller-provided output folder.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from vslp.acoustic.preprocess.audio import decode_audio_ffmpeg, resolve_mono_channel
from vslp.acoustic.segment.task_methods import DDKConfig, segment_ddk


EXTENSIONS = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".aiff"}


def audit_ddk_corpus(input_folder: str | Path, output_folder: str | Path,
                     config: DDKConfig = DDKConfig()) -> tuple[Path, Path]:
    source = Path(input_folder)
    if not source.is_dir():
        raise NotADirectoryError(source)
    output = Path(output_folder)
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Audit output must be outside the input corpus")
    rows = []
    events = []
    for path in sorted(p for p in source.rglob("*") if p.suffix.lower() in EXTENSIONS):
        row = {"file_name": path.name, "relative_path": str(path.relative_to(source))}
        for syllable in ("ba", "pa", "ta"):
            if syllable in path.stem.lower().replace("-", "_").split("_"):
                row["task_syllable"] = syllable
                break
        try:
            channels, sr = decode_audio_ffmpeg(path, target_sr=None, mono=False)
            resolution = resolve_mono_channel(channels)
            if resolution.selected_channel is None:
                raise ValueError("Unresolved multichannel audio")
            result = segment_ddk(channels[:, resolution.selected_channel], sr, config)
            row.update({"sample_rate_hz": sr, "duration_sec": len(channels) / sr,
                        "automatic_status": result["automatic_status"],
                        "flags": ";".join(result["flags"]), "n_events": result["n_events"],
                        "raw_threshold_crossings": len(result["raw_threshold_crossings_samples"]),
                        "raw_candidates": result["raw_candidate_count"],
                        "merged_candidates": len(result["merged_candidates"]),
                        "removed_by_debounce": result["raw_candidate_count"] - len(result["merged_candidates"]),
                        "removed_by_minimum_duration": len(result["rejected_events"]),
                        "minimum_peak_threshold_ratio": result["minimum_peak_threshold_ratio"]})
            peaks = [e["energy_peak_sample"] / sr for e in result["final_events"]]
            for i, event in enumerate(result["final_events"]):
                events.append({"file_name": path.name, "event_index": i,
                               "duration_sec": (event["end_sample"] - event["start_sample"]) / sr,
                               "gap_sec": (event["start_sample"] - result["final_events"][i - 1]["end_sample"]) / sr if i else np.nan,
                               "cycle_sec": peaks[i] - peaks[i - 1] if i else np.nan,
                               "peak_to_threshold_ratio": event["support_ratio"]})
        except Exception as exc:  # Keep every file in the audit.
            row.update({"automatic_status": "FAILED", "error": str(exc)})
        rows.append(row)
    file_table = pd.DataFrame(rows)
    event_table = pd.DataFrame(events)
    output.mkdir(parents=True, exist_ok=True)
    files_path = output / "ddk_tanchip_audit_files.csv"
    events_path = output / "ddk_tanchip_audit_events.csv"
    file_table.to_csv(files_path, index=False)
    event_table.to_csv(events_path, index=False)
    valid = file_table.loc[file_table.automatic_status.ne("FAILED")] if len(file_table) else file_table
    report = {
        "profile": "DDK Tanchip 2022 Energy",
        "paper_defined": {"lowpass_cutoff_hz": 200, "energy_frame_ms": 20,
                          "threshold_window_ms": 20},
        "implementation_defaults": asdict(config),
        "n_files": len(file_table), "n_executed": len(valid),
        "sample_rate_distribution": file_table.get("sample_rate_hz", pd.Series(dtype=float)).value_counts().to_dict(),
        "task_distribution": file_table.get("task_syllable", pd.Series(dtype=str)).value_counts().to_dict(),
        "zero_event_percentage": float(100 * valid.n_events.eq(0).mean()) if len(valid) else None,
        "fragmentation_proxy_percentage": float(100 * valid.flags.fillna("").str.contains("fragmentation").mean()) if len(valid) else None,
        "duration_quantiles_sec": valid.duration_sec.quantile([.1, .5, .9]).to_dict() if len(valid) else {},
        "event_duration_quantiles_sec": event_table.duration_sec.quantile([.1, .5, .9]).to_dict() if len(event_table) else {},
        "cycle_quantiles_sec": event_table.cycle_sec.quantile([.1, .5, .9]).to_dict() if len(event_table) else {},
        "gap_quantiles_sec": event_table.gap_sec.quantile([.1, .5, .9]).to_dict() if len(event_table) else {},
        "interpretation": "Descriptive audit only. No clinical ground truth or optimization is used. Review event plots before changing engineering defaults.",
    }
    (output / "ddk_tanchip_audit_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return files_path, events_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Opt-in DDK Energy corpus audit")
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    args = parser.parse_args()
    audit_ddk_corpus(args.input_folder, args.output_folder)
