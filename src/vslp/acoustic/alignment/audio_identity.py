"""Decoded acoustic identity, independent of non-signal container metadata."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from pathlib import Path
import struct

import numpy as np
import soundfile as sf


def audio_content_sha256(path: str | Path) -> str:
    """Hash rate, channel/frame counts, then interleaved little-endian float64 samples.

    libsndfile decodes PCM to normalized floating-point values; IEEE float audio
    retains its decoded values. No channel mixing, gain, or resampling occurs.
    """
    digest = sha256()
    with sf.SoundFile(path) as audio:
        digest.update(f"{audio.samplerate}:{audio.channels}:{len(audio)}".encode("ascii"))
        while len(block := audio.read(65536, dtype="float64", always_2d=True)):
            digest.update(np.ascontiguousarray(block, dtype="<f8").tobytes())
    return digest.hexdigest()


def legacy_peak_timestamp_only_match(path: Path, expected_sha256: str,
                                     created_at: str, aligned_at: str) -> bool:
    """Prove a legacy WAV differs only in its PEAK timestamp using its frozen SHA.

    Search is bounded by project creation and alignment completion. A successful
    SHA match reconstructs the *exact* original serialized WAV. Other metadata
    changes, changed samples, and unbounded/unknown history remain stale.
    """
    try:
        start = int(datetime.fromisoformat(created_at).timestamp())
        end = int(datetime.fromisoformat(aligned_at).timestamp())
        if not 0 <= end - start <= 3600:
            return False
        data = path.read_bytes()
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return False
        offset = 12
        peak_timestamp = None
        while offset + 8 <= len(data):
            tag, size = struct.unpack_from("<4sI", data, offset)
            if tag == b"PEAK":
                if size < 8 or peak_timestamp is not None:
                    return False
                peak_timestamp = offset + 12
            offset += 8 + size + (size & 1)
        if peak_timestamp is None or offset != len(data):
            return False
        prefix, suffix = data[:peak_timestamp], data[peak_timestamp + 4:]
        for timestamp in range(start, end + 1):
            digest = sha256()
            digest.update(prefix)
            digest.update(struct.pack("<I", timestamp))
            digest.update(suffix)
            if digest.hexdigest() == expected_sha256:
                return True
    except (OSError, ValueError, OverflowError, struct.error):
        return False
    return False
