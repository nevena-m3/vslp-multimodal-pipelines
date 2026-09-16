"""Deterministic audio decoding and canonical preprocessing primitives.

This module intentionally contains only transforms that are part of the canonical
preprocessing contract. Artifact/QC measurements belong to the Quality Control stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class ChannelResolution:
    """Auditable decision for converting decoded audio to one canonical channel."""

    status: str
    selected_channel: int | None
    reason: str
    n_channels: int
    per_channel_rms: tuple[float, ...]
    per_channel_peak_abs: tuple[float, ...]
    correlation_to_reference: tuple[float | None, ...]
    gain_difference_db_to_reference: tuple[float | None, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_decoded_audio(x: np.ndarray, sr: int, *, file_path: Path) -> None:
    if int(sr) <= 0:
        raise ValueError(f"Decoded audio has invalid sample rate {sr}: {file_path}")
    if x.ndim != 2:
        raise ValueError(f"Decoded audio must be frames x channels, got shape {x.shape}: {file_path}")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"Decoded audio is empty: {file_path}")
    if not np.isfinite(x).all():
        raise ValueError(f"Decoded audio contains NaN or infinite samples: {file_path}")


def decode_audio_ffmpeg(
    file_path: str | Path,
    target_sr: int | None = None,
    mono: bool = False,
    ffmpeg_bin: str = "ffmpeg",
    audio_stream_selector: str = "0:a:0",
) -> tuple[np.ndarray, int]:
    """Decode one deterministic audio stream to float32 PCM without amplitude clipping.

    The canonical preprocessing stage calls this with ``target_sr=None`` and
    ``mono=False`` so the native sample rate and channel structure are preserved
    until channel resolution is performed explicitly.

    ``target_sr`` and ``mono`` remain available for algorithm-specific callers, but
    they are not part of canonical preprocessing.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Source media file does not exist: {file_path}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)

    try:
        cmd = [
            str(ffmpeg_bin),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(file_path),
            "-map",
            str(audio_stream_selector),
            "-vn",
        ]
        if mono:
            cmd += ["-ac", "1"]
        if target_sr is not None:
            if int(target_sr) <= 0:
                raise ValueError("target_sr must be a positive integer")
            cmd += ["-ar", str(int(target_sr))]
        cmd += ["-c:a", "pcm_f32le", "-f", "wav", str(tmp_wav)]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            message = proc.stderr.strip() or "unknown ffmpeg error"
            raise RuntimeError(f"ffmpeg failed for {file_path}: {message}")

        x2d, sr = sf.read(tmp_wav, dtype="float32", always_2d=True)
        x2d = np.asarray(x2d, dtype=np.float32)
        _validate_decoded_audio(x2d, int(sr), file_path=file_path)
        if mono:
            return x2d[:, 0].copy(), int(sr)
        return x2d, int(sr)
    finally:
        try:
            os.remove(tmp_wav)
        except OSError:
            pass


def dc_offset_remove(x: np.ndarray) -> np.ndarray:
    """Remove only the constant (zero-frequency) offset from a mono waveform."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"DC-offset removal requires mono 1-D audio, got shape {x.shape}")
    if x.size == 0:
        raise ValueError("DC-offset removal received empty audio")
    if not np.isfinite(x).all():
        raise ValueError("DC-offset removal received NaN or infinite samples")
    offset = float(np.mean(x, dtype=np.float64))
    return (x.astype(np.float64) - offset).astype(np.float32)


def apply_dc_offset_policy(
    x: np.ndarray,
    *,
    remove_dc_offset: bool,
) -> tuple[np.ndarray, float, float]:
    """Apply the user-selected DC policy without any other waveform transform."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"DC policy requires non-empty mono audio, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("DC policy received NaN or infinite samples")

    before = float(np.mean(x, dtype=np.float64))
    if remove_dc_offset:
        y = dc_offset_remove(x)
    else:
        y = x.copy()
    after = float(np.mean(y, dtype=np.float64))
    return y, before, after


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64), dtype=np.float64)))


def _centered_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    ac = a64 - np.mean(a64)
    bc = b64 - np.mean(b64)
    denom = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    if not math.isfinite(denom) or denom <= np.finfo(np.float64).tiny:
        return None
    value = float(np.dot(ac, bc) / denom)
    return float(np.clip(value, -1.0, 1.0))


def _gain_difference_db(rms_value: float, rms_reference: float) -> float | None:
    if rms_value <= 0.0 or rms_reference <= 0.0:
        return None
    return float(20.0 * np.log10(rms_value / rms_reference))


def resolve_mono_channel(
    x: np.ndarray,
    *,
    duplicate_correlation_min: float = 0.999,
    duplicate_gain_difference_db_max: float = 0.10,
    silent_channel_rms_max: float = 1e-6,
) -> ChannelResolution:
    """Resolve multichannel audio conservatively.

    Rules
    -----
    - mono: use channel 0;
    - all channels effectively silent: use channel 0 deterministically;
    - exactly one non-silent channel: use it;
    - multiple non-silent channels: select the first only when they are effectively
      duplicate channels under strict correlation and gain criteria;
    - otherwise: return ``needs_channel_review`` and do not choose a channel.

    The thresholds are engineering duplicate-channel criteria, not clinical or
    physiological thresholds.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError(f"Channel resolution requires frames x channels audio, got {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("Channel resolution received empty audio")
    if not np.isfinite(x).all():
        raise ValueError("Channel resolution received NaN or infinite samples")
    if not (0.0 < duplicate_correlation_min <= 1.0):
        raise ValueError("duplicate_correlation_min must be in (0, 1]")
    if duplicate_gain_difference_db_max < 0.0:
        raise ValueError("duplicate_gain_difference_db_max must be >= 0")
    if silent_channel_rms_max < 0.0:
        raise ValueError("silent_channel_rms_max must be >= 0")

    n_channels = int(x.shape[1])
    rms = tuple(_rms(x[:, ch]) for ch in range(n_channels))
    peaks = tuple(float(np.max(np.abs(x[:, ch]))) for ch in range(n_channels))

    if n_channels == 1:
        return ChannelResolution(
            status="resolved_mono",
            selected_channel=0,
            reason="source_is_mono",
            n_channels=1,
            per_channel_rms=rms,
            per_channel_peak_abs=peaks,
            correlation_to_reference=(1.0,),
            gain_difference_db_to_reference=(0.0,),
        )

    usable = [ch for ch, value in enumerate(rms) if value > silent_channel_rms_max]
    if not usable:
        return ChannelResolution(
            status="resolved_all_channels_effectively_silent",
            selected_channel=0,
            reason=f"all_channel_rms<=silent_threshold({silent_channel_rms_max:g})",
            n_channels=n_channels,
            per_channel_rms=rms,
            per_channel_peak_abs=peaks,
            correlation_to_reference=tuple(1.0 if ch == 0 else None for ch in range(n_channels)),
            gain_difference_db_to_reference=tuple(0.0 if ch == 0 else None for ch in range(n_channels)),
        )

    if len(usable) == 1:
        selected = int(usable[0])
        return ChannelResolution(
            status="resolved_single_usable_channel",
            selected_channel=selected,
            reason="all_other_channels_effectively_silent",
            n_channels=n_channels,
            per_channel_rms=rms,
            per_channel_peak_abs=peaks,
            correlation_to_reference=tuple(1.0 if ch == selected else None for ch in range(n_channels)),
            gain_difference_db_to_reference=tuple(0.0 if ch == selected else None for ch in range(n_channels)),
        )

    reference = int(usable[0])
    correlations: list[float | None] = []
    gains: list[float | None] = []
    duplicate = True
    for ch in range(n_channels):
        if ch == reference:
            correlations.append(1.0)
            gains.append(0.0)
            continue
        if ch not in usable:
            correlations.append(None)
            gains.append(None)
            continue
        corr = _centered_correlation(x[:, reference], x[:, ch])
        gain_db = _gain_difference_db(rms[ch], rms[reference])
        correlations.append(corr)
        gains.append(gain_db)
        if (
            corr is None
            or corr < duplicate_correlation_min
            or gain_db is None
            or abs(gain_db) > duplicate_gain_difference_db_max
        ):
            duplicate = False

    if duplicate:
        return ChannelResolution(
            status="resolved_duplicate_multichannel",
            selected_channel=reference,
            reason=(
                "non_silent_channels_are_effectively_duplicate;"
                f"corr>={duplicate_correlation_min:g};"
                f"|gain_db|<={duplicate_gain_difference_db_max:g}"
            ),
            n_channels=n_channels,
            per_channel_rms=rms,
            per_channel_peak_abs=peaks,
            correlation_to_reference=tuple(correlations),
            gain_difference_db_to_reference=tuple(gains),
        )

    return ChannelResolution(
        status="needs_channel_review",
        selected_channel=None,
        reason=(
            "multiple_non_silent_channels_are_not_equivalent_under_duplicate_channel_criteria"
        ),
        n_channels=n_channels,
        per_channel_rms=rms,
        per_channel_peak_abs=peaks,
        correlation_to_reference=tuple(correlations),
        gain_difference_db_to_reference=tuple(gains),
    )
