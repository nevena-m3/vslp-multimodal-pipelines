"""Audio/video container inspection using ffprobe."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def ffprobe_json(path: str | Path, ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    """Return raw ffprobe JSON for an input media file."""
    path = Path(path)
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-print_format", "json",
        str(path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {p.stderr}")
    return json.loads(p.stdout)


def digest_media_file(path: str | Path, ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    """Extract a stable, flat digest for a media file."""
    path = Path(path)
    raw = ffprobe_json(path, ffprobe_bin=ffprobe_bin)
    fmt = raw.get("format", {})
    streams = raw.get("streams", [])
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    first_audio = audio_streams[0] if audio_streams else {}

    def as_float(x):
        try:
            return float(x)
        except Exception:
            return None

    def as_int(x):
        try:
            return int(float(x))
        except Exception:
            return None

    return {
        "file_name": path.name,
        "file_path": str(path),
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "format_name": fmt.get("format_name"),
        "format_long_name": fmt.get("format_long_name"),
        "duration_sec": as_float(fmt.get("duration") or first_audio.get("duration")),
        "bit_rate": as_int(fmt.get("bit_rate")),
        "n_streams": len(streams),
        "n_audio_streams": len(audio_streams),
        "audio_stream_index": as_int(first_audio.get("index")),
        "audio_stream_selector": "0:a:0" if audio_streams else None,
        "audio_codec": first_audio.get("codec_name"),
        "audio_codec_long_name": first_audio.get("codec_long_name"),
        "sample_rate_hz": as_int(first_audio.get("sample_rate")),
        "channels": as_int(first_audio.get("channels")),
        "channel_layout": first_audio.get("channel_layout"),
        "bits_per_sample": as_int(first_audio.get("bits_per_sample")),
    }
