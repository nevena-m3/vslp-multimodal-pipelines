"""Gross utterance bounds for MFA preparation, independent of clinical timing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def propose_scope(source: str | Path) -> dict:
    """Find the first/last sustained energetic region; preserve internal pauses.

    This is an auditable alignment proposal, never a reviewed speech interval.
    An uncertain/empty proposal falls back to the full file and needs review.
    """
    with sf.SoundFile(source) as stream:
        rate, frames = stream.samplerate, stream.frames
        hop = max(1, round(rate * .02))
        levels = []
        while True:
            block = stream.read(hop, dtype="float32", always_2d=True)
            if not len(block):
                break
            mono = block.mean(axis=1, dtype=np.float64)
            levels.append(float(np.sqrt(np.mean(mono * mono))))
    duration = frames / rate
    if not levels or not np.isfinite(levels).all():
        return {"start_sec": 0.0, "end_sec": duration, "source": "full_recording_fallback",
                "uncertain": True, "reason": "no_finite_alignment_scope_energy"}
    energy = np.asarray(levels)
    floor = float(np.percentile(energy, 15))
    peak = float(np.percentile(energy, 95))
    threshold = max(floor * 4, peak * .08, 1e-5)
    active = energy >= threshold
    # Reject isolated clicks; retain one complete passage from first to last run.
    minimum_frames = max(1, round(.12 * rate / hop))
    runs = np.flatnonzero(np.diff(np.r_[False, active, False].astype(np.int8)))
    candidates = [(int(a), int(b)) for a, b in zip(runs[::2], runs[1::2])
                  if b - a >= minimum_frames]
    if not candidates:
        return {"start_sec": 0.0, "end_sec": duration, "source": "full_recording_fallback",
                "uncertain": True, "reason": "no_sustained_energy_for_alignment_scope",
                "energy_threshold_rms": threshold}
    start = max(0, candidates[0][0] * hop - round(.20 * rate))
    end = min(frames, candidates[-1][1] * hop + round(.20 * rate))
    uncertain = end - start < round(.5 * rate)
    return {"start_sec": start / rate, "end_sec": end / rate,
            "source": "alignment_energy_proposal_v1", "uncertain": uncertain,
            "reason": "scope_too_short" if uncertain else "",
            "energy_threshold_rms": threshold, "pad_sec": .20,
            "min_active_run_sec": .12}


EDGE_SEC = .05


def speech_at_edges(source: str | Path, threshold: float | None = None,
                    edge_sec: float = EDGE_SEC) -> tuple[bool, bool]:
    """Whether speech-level energy reaches the first / last ``edge_sec``.

    A recording cut tightly in a pause (split CAPE-V sentences, trimmed files)
    has its speech near the edge but silence at it: nothing is clipped. Speech
    energy in the outermost 50 ms means the recording may cut into a word.
    ``threshold`` is the scope proposal's RMS threshold (recomputed if absent).
    """
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1, dtype=np.float64)
    hop = max(1, round(rate * .02))
    if threshold is None:
        frames = len(mono) // hop
        if not frames:
            return False, False
        levels = np.sqrt((mono[:frames * hop].reshape(frames, hop) ** 2).mean(axis=1))
        peak = float(np.percentile(levels, 95))
        # As in propose_scope, but capped at peak - 10 dB: a file that is
        # almost all speech has its "floor" percentile inside the speech.
        threshold = min(max(float(np.percentile(levels, 15)) * 4, peak * .08, 1e-5),
                        max(peak * .3, 1e-5))
    width = max(hop, round(edge_sec * rate))

    def loud(part: np.ndarray) -> bool:
        blocks = [part[i:i + hop] for i in range(0, len(part), hop) if len(part[i:i + hop])]
        return any(float(np.sqrt(np.mean(block * block))) >= threshold for block in blocks)

    return loud(mono[:width]), loud(mono[-width:])


def write_scoped_audio(source: str | Path, destination: str | Path,
                       start_sec: float, end_sec: float) -> tuple[float, float]:
    """Write a feature-independent working WAV and return sample-exact bounds."""
    with sf.SoundFile(source) as stream:
        first = round(start_sec * stream.samplerate)
        last = round(end_sec * stream.samplerate)
        if not 0 <= first < last <= stream.frames:
            raise ValueError("invalid_alignment_scope_bounds")
        stream.seek(first)
        with sf.SoundFile(destination, mode="w", samplerate=stream.samplerate,
                          channels=stream.channels, subtype=stream.subtype,
                          format="WAV") as output:
            remaining = last - first
            while remaining:
                block = stream.read(min(262144, remaining), dtype="float64", always_2d=True)
                if not len(block):
                    raise ValueError("alignment_scope_source_truncated")
                output.write(block)
                remaining -= len(block)
    return first / stream.samplerate, last / stream.samplerate
