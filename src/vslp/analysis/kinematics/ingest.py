"""Video discovery and structural ingest for the kinematics GUI."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .schemas import VideoIngestConfig, VideoRecord


def _safe_rel(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def discover_videos(input_root: Path, *, recursive: bool, extensions: Iterable[str]) -> list[Path]:
    root = Path(input_root).expanduser().resolve()
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    pattern = "**/*" if recursive else "*"
    return sorted([p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in exts], key=lambda p: p.as_posix().lower())


def _ffprobe(path: Path, ffprobe_bin: str) -> dict:
    cmd = [
        ffprobe_bin, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    if out.returncode != 0:
        raise RuntimeError((out.stderr or out.stdout or "ffprobe failed").strip())
    return json.loads(out.stdout)


def _parse_rate(rate: str | None) -> float | None:
    if not rate or rate == "0/0":
        return None
    try:
        a, b = rate.split("/")
        return float(a) / float(b) if float(b) else None
    except Exception:
        try:
            return float(rate)
        except Exception:
            return None


def _task_guess(path: Path) -> str:
    text = " ".join(path.parts).lower()
    rules = [
        ("bamboo", "bamboo_passage"),
        ("pa-ta-ka", "ddk_pataka"), ("pataka", "ddk_pataka"),
        ("pa", "ddk_pa"), ("papa", "ddk_pa"),
        ("puppy", "buy_bobby_a_puppy"),
        ("smile", "smile"), ("open", "mouth_open_close"),
    ]
    for needle, label in rules:
        if needle in text:
            return label
    return "unknown"


def probe_video(path: Path, *, root: Path, ffprobe_bin: str) -> VideoRecord:
    rel = _safe_rel(path, root)
    base = VideoRecord(
        source_path=str(path), relative_path=rel.as_posix(), video_id=path.stem,
        extension=path.suffix.lower(), size_bytes=path.stat().st_size,
        task_guess=_task_guess(path),
    )
    try:
        info = _ffprobe(path, ffprobe_bin)
        streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        st = streams[0] if streams else {}
        fmt = info.get("format", {})
        fps = _parse_rate(st.get("avg_frame_rate") or st.get("r_frame_rate"))
        duration = None
        for val in (st.get("duration"), fmt.get("duration")):
            try:
                if val is not None:
                    duration = float(val); break
            except Exception:
                pass
        n_frames = None
        try:
            n_frames = int(st.get("nb_frames")) if st.get("nb_frames") else None
        except Exception:
            n_frames = None
        if n_frames is None and fps and duration:
            n_frames = int(round(fps * duration))
        return VideoRecord(
            **{**asdict(base),
               "width": int(st["width"]) if st.get("width") else None,
               "height": int(st["height"]) if st.get("height") else None,
               "fps": fps,
               "duration_sec": duration,
               "n_frames_estimated": n_frames,
               "codec_name": st.get("codec_name"),
               "container_name": fmt.get("format_name"),
               "status": "pass" if streams else "warning",
               "warning": "" if streams else "No video stream found."}
        )
    except Exception as e:
        return VideoRecord(**{**asdict(base), "status": "warning", "warning": f"Probe failed: {e}"})


def summarize_ingest_manifest(df: pd.DataFrame) -> dict[str, object]:
    """Return dashboard-ready ingest metrics from a video manifest table.

    The Setup tab uses these metrics as a structural readiness gate. This
    intentionally stays operational: it describes video readability and format
    consistency, not biological/kinematic signal quality.
    """
    columns = [
        "status", "warning", "fps", "duration_sec", "n_frames_estimated",
        "width", "height", "extension", "codec_name", "container_name",
    ]
    work = df.copy() if df is not None else pd.DataFrame(columns=columns)
    for col in columns:
        if col not in work.columns:
            work[col] = pd.NA

    n_videos = int(len(work))
    status_text = work["status"].fillna("").astype(str).str.lower()
    warning_text = work["warning"].fillna("").astype(str).str.strip()
    readable_mask = status_text.eq("pass")
    warning_mask = (~readable_mask) | warning_text.ne("")

    readable_videos = int(readable_mask.sum())
    warning_videos = int(warning_mask.sum())
    unreadable_videos = int(n_videos - readable_videos)

    fps = pd.to_numeric(work["fps"], errors="coerce")
    duration = pd.to_numeric(work["duration_sec"], errors="coerce")
    frames = pd.to_numeric(work["n_frames_estimated"], errors="coerce")

    if n_videos == 0:
        readiness = "FAIL"
        next_step = "No supported videos were found. Check the input folder and supported extensions, then run ingest again."
    elif readable_videos == 0:
        readiness = "FAIL"
        next_step = "No videos were structurally readable. Review ffprobe/OpenCV support, codecs, and file paths before landmark extraction."
    elif warning_videos > 0:
        readiness = "REVIEW"
        next_step = "Some videos produced warnings. You can continue to landmark extraction, but review warnings before trusting features."
    else:
        readiness = "PASS"
        next_step = "Dataset structure looks ready. Next: run MediaPipe landmark extraction."

    def finite_stat(series: pd.Series, op: str) -> float | None:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        if vals.empty:
            return None
        if op == "median":
            return float(vals.median())
        if op == "min":
            return float(vals.min())
        if op == "max":
            return float(vals.max())
        if op == "sum":
            return float(vals.sum())
        raise ValueError(op)

    summary = {
        "readiness": readiness,
        "n_videos": n_videos,
        "readable_videos": readable_videos,
        "warning_videos": warning_videos,
        "unreadable_videos": unreadable_videos,
        "total_estimated_frames": int(frames.fillna(0).sum()) if not frames.empty else 0,
        "median_fps": finite_stat(fps, "median"),
        "min_fps": finite_stat(fps, "min"),
        "max_fps": finite_stat(fps, "max"),
        "median_duration_sec": finite_stat(duration, "median"),
        "min_duration_sec": finite_stat(duration, "min"),
        "max_duration_sec": finite_stat(duration, "max"),
        "total_duration_sec": finite_stat(duration, "sum"),
        "next_step": next_step,
    }
    return summary


def build_format_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Group ingested videos by container/codec/resolution with robust medians."""
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "extension", "codec_name", "container_name", "resolution", "files",
            "median_fps", "median_duration_sec", "estimated_frames", "warnings",
        ])
    work = df.copy()
    for col in ["width", "height", "fps", "duration_sec", "n_frames_estimated", "warning", "status", "extension", "codec_name", "container_name"]:
        if col not in work.columns:
            work[col] = pd.NA
    width = pd.to_numeric(work["width"], errors="coerce")
    height = pd.to_numeric(work["height"], errors="coerce")
    work["resolution"] = width.fillna(0).astype(int).astype(str) + " x " + height.fillna(0).astype(int).astype(str)
    work.loc[width.isna() | height.isna(), "resolution"] = "unknown"
    work["has_warning"] = (~work["status"].fillna("").astype(str).str.lower().eq("pass")) | work["warning"].fillna("").astype(str).str.strip().ne("")
    grouped = (
        work.groupby(["extension", "codec_name", "container_name", "resolution"], dropna=False)
        .agg(
            files=("source_path", "size"),
            median_fps=("fps", "median"),
            median_duration_sec=("duration_sec", "median"),
            estimated_frames=("n_frames_estimated", "sum"),
            warnings=("has_warning", "sum"),
        )
        .reset_index()
        .sort_values(["files", "extension"], ascending=[False, True])
    )
    return grouped


def build_warning_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per video that requires review after structural ingest."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["video_id", "relative_path", "status", "warning"])
    work = df.copy()
    for col in ["video_id", "relative_path", "status", "warning"]:
        if col not in work.columns:
            work[col] = ""
    warning_mask = (~work["status"].fillna("").astype(str).str.lower().eq("pass")) | work["warning"].fillna("").astype(str).str.strip().ne("")
    return work.loc[warning_mask, ["video_id", "relative_path", "status", "warning"]].reset_index(drop=True)


def run_ingest(cfg: VideoIngestConfig) -> dict[str, Path | int | str]:
    out_root = Path(cfg.output_root) / "kinematics" / "000_ingest"
    tables = out_root / "tables"
    logs = out_root / "logs"
    tables.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    root = Path(cfg.input_root).expanduser().resolve()
    videos = discover_videos(root, recursive=cfg.recursive, extensions=cfg.extensions)
    rows = [probe_video(p, root=root, ffprobe_bin=cfg.ffprobe_bin) for p in videos]
    df = pd.DataFrame([asdict(r) for r in rows])
    manifest_csv = tables / "video_ingest_manifest.csv"
    manifest_json = tables / "video_ingest_manifest.json"
    summary_csv = tables / "video_ingest_summary.csv"
    summary_json = tables / "video_ingest_summary.json"
    format_csv = tables / "video_format_summary.csv"
    warnings_csv = tables / "video_ingest_warnings.csv"
    log_path = logs / "video_ingest.log"

    df.to_csv(manifest_csv, index=False)
    summary = summarize_ingest_manifest(df)
    format_summary = build_format_summary(df)
    warning_summary = build_warning_summary(df)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    format_summary.to_csv(format_csv, index=False)
    warning_summary.to_csv(warnings_csv, index=False)
    payload = {
        "schema": "vslp_kinematics_video_ingest_v0.71",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_videos": len(rows),
        "input_root": str(root),
        "readiness": summary.get("readiness"),
        "summary": summary,
        "rows": [asdict(r) for r in rows],
    }
    manifest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps({"schema": "vslp_kinematics_ingest_dashboard_v0.71", **payload}, indent=2), encoding="utf-8")
    log_lines = [
        f"created_at_utc={payload['created_at_utc']}",
        f"input_root={root}",
        f"n_videos={summary['n_videos']}",
        f"readable_videos={summary['readable_videos']}",
        f"warning_videos={summary['warning_videos']}",
        f"readiness={summary['readiness']}",
        f"next_step={summary['next_step']}",
    ]
    if not warning_summary.empty:
        log_lines.append("warnings:")
        for _, row in warning_summary.iterrows():
            log_lines.append(f"- {row.get('video_id', '')}: {row.get('warning', '') or row.get('status', '')}")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return {
        "output_root": out_root,
        "manifest_csv": manifest_csv,
        "manifest_json": manifest_json,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "format_csv": format_csv,
        "warnings_csv": warnings_csv,
        "log_path": log_path,
        "n_videos": len(rows),
        "readiness": str(summary.get("readiness", "UNKNOWN")),
    }
