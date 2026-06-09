"""Video discovery and structural ingest for the kinematics GUI."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
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


def run_ingest(cfg: VideoIngestConfig) -> dict[str, Path | int]:
    out_root = Path(cfg.output_root) / "kinematics" / "000_ingest"
    tables = out_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    root = Path(cfg.input_root).expanduser().resolve()
    videos = discover_videos(root, recursive=cfg.recursive, extensions=cfg.extensions)
    rows = [probe_video(p, root=root, ffprobe_bin=cfg.ffprobe_bin) for p in videos]
    df = pd.DataFrame([asdict(r) for r in rows])
    manifest_csv = tables / "video_ingest_manifest.csv"
    manifest_json = tables / "video_ingest_manifest.json"
    df.to_csv(manifest_csv, index=False)
    manifest_json.write_text(json.dumps({"n_videos": len(rows), "input_root": str(root), "rows": [asdict(r) for r in rows]}, indent=2), encoding="utf-8")
    return {"output_root": out_root, "manifest_csv": manifest_csv, "manifest_json": manifest_json, "n_videos": len(rows)}
