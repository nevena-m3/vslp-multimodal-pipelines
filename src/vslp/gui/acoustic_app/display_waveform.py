"""Small, display-only waveform cache on the original recording clock."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import Lock

import numpy as np
import soundfile as sf


class DisplayWaveformCache:
    """Never return analysis audio; cache only a bounded plotting envelope."""

    def __init__(self, capacity: int = 8, points: int = 15000) -> None:
        self.capacity = capacity
        self.points = points
        self._entries: OrderedDict[tuple[str, int, int],
                                   tuple[np.ndarray, np.ndarray, float]] = OrderedDict()
        self._lock = Lock()

    def get(self, path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
        source = Path(path)
        stat = source.stat()
        key = (str(source.resolve()), stat.st_size, stat.st_mtime_ns)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
        with sf.SoundFile(source) as wave:
            rate = wave.samplerate
            duration = wave.frames / rate
            block_size = max(1, (wave.frames + self.points - 1) // self.points)
            times, samples = [], []
            sample_index = 0
            for block in wave.blocks(blocksize=block_size, dtype="float32",
                                     always_2d=True):
                mono = block.mean(axis=1)
                times.extend((sample_index / rate,
                              (sample_index + len(mono) - 1) / rate))
                samples.extend((float(np.min(mono)), float(np.max(mono))))
                sample_index += len(mono)
        result = (np.asarray(times, dtype="float32"),
                  np.asarray(samples, dtype="float32"), duration)
        for array in result[:2]:
            array.flags.writeable = False
        with self._lock:
            self._entries[key] = result
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
        return result
