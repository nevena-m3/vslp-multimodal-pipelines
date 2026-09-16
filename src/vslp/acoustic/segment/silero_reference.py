"""Scientific Silero interval and boundary diagnostics ported from quality_framework_features.

Source: https://github.com/nevena-m3/quality_framework_features/blob/main/src/paper1_qc/segmentation.py
Primary boundaries are exact Silero sample-index timestamps; display frames are diagnostic only.
"""


from __future__ import annotations


from dataclasses import dataclass


from typing import Iterable


import numpy as np


import pandas as pd


@dataclass(frozen=True)
class Interval:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def normalize_intervals(intervals: Iterable[Interval], duration_sec: float) -> list[Interval]:
    clipped = [
        Interval(max(0.0, float(item.start_sec)), min(duration_sec, float(item.end_sec)))
        for item in intervals
        if float(item.end_sec) > float(item.start_sec)
    ]
    clipped = [item for item in clipped if item.duration_sec > 0]
    clipped.sort(key=lambda item: (item.start_sec, item.end_sec))
    merged: list[Interval] = []
    for item in clipped:
        if merged and item.start_sec <= merged[-1].end_sec:
            merged[-1] = Interval(merged[-1].start_sec, max(merged[-1].end_sec, item.end_sec))
        else:
            merged.append(item)
    return merged


def erode_intervals(intervals: Iterable[Interval], edge_sec: float) -> list[Interval]:
    return [
        Interval(item.start_sec + edge_sec, item.end_sec - edge_sec)
        for item in intervals
        if item.duration_sec > 2 * edge_sec
    ]


def complement_intervals(intervals: Iterable[Interval], duration_sec: float) -> list[Interval]:
    speech = normalize_intervals(intervals, duration_sec)
    result: list[Interval] = []
    cursor = 0.0
    for item in speech:
        if item.start_sec > cursor:
            result.append(Interval(cursor, item.start_sec))
        cursor = max(cursor, item.end_sec)
    if cursor < duration_sec:
        result.append(Interval(cursor, duration_sec))
    return result


def internal_nonspeech(intervals: Iterable[Interval], duration_sec: float) -> list[Interval]:
    speech = normalize_intervals(intervals, duration_sec)
    if len(speech) < 2:
        return []
    return [
        Interval(left.end_sec, right.start_sec)
        for left, right in zip(speech[:-1], speech[1:])
        if right.start_sec > left.end_sec
    ]


def load_silero_model(*, onnx: bool = True):
    """Load one version-pinned Silero model for reuse across recordings/profiles."""
    from silero_vad import load_silero_vad

    return load_silero_vad(onnx=onnx)


def silero_speech_timestamps(
    waveform_16k: np.ndarray,
    *,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 100,
    speech_pad_ms: int = 0,
    onnx: bool = True,
    model=None,
) -> list[dict[str, int]]:
    """Return sample-index timestamps without display-frame quantization."""
    import torch
    from silero_vad import get_speech_timestamps

    if model is None:
        model = load_silero_model(onnx=onnx)
    waveform = torch.from_numpy(np.asarray(waveform_16k, dtype=np.float32))
    stamps = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False,
    )
    return [{"start": int(item["start"]), "end": int(item["end"])} for item in stamps]


BOUNDARY_AUDIT_COLUMNS = [
    "segment_index",
    "start_sec",
    "end_sec",
    "duration_sec",
    "display_start_sec",
    "display_end_sec",
    "display_onset_delta_ms",
    "display_offset_delta_ms",
    "onset_pre_rms_dbfs",
    "onset_inside_rms_dbfs",
    "onset_contrast_db",
    "offset_inside_rms_dbfs",
    "offset_post_rms_dbfs",
    "offset_contrast_db",
    "ambiguous_onset",
    "ambiguous_offset",
    "boundary_review_flag",
]


def _window_rms_dbfs(
    signal: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> float:
    start = max(0, int(round(start_sec * sample_rate)))
    stop = min(len(signal), int(round(end_sec * sample_rate)))
    if stop <= start:
        return float("nan")
    rms = float(np.sqrt(np.mean(np.square(signal[start:stop], dtype=np.float64))))
    return float(20.0 * np.log10(max(rms, 1e-12)))


def boundary_alignment_diagnostics(
    waveform: np.ndarray,
    sample_rate: int,
    speech_intervals: Iterable[Interval],
    *,
    displayed_segments: pd.DataFrame | None = None,
    window_ms: float = 120,
    guard_ms: float = 20,
    minimum_contrast_db: float = 3.0,
) -> pd.DataFrame:
    """Audit exact Silero boundaries without moving them using an energy heuristic.

    Local RMS contrast is a review signal only. Low-intensity/breathy ALS speech can
    have weak energy contrast, so this function never snaps, trims, or expands a
    boundary automatically. ``display_*`` fields quantify the separate 30-ms
    compatibility-frame representation used by the original-style plot.
    """
    signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
    intervals = list(speech_intervals)
    display_speech = pd.DataFrame()
    if displayed_segments is not None and not displayed_segments.empty:
        display_speech = displayed_segments.loc[
            displayed_segments["segment_type"].astype(str).eq("speech")
        ].reset_index(drop=True)
    window_sec = float(window_ms) / 1000.0
    guard_sec = float(guard_ms) / 1000.0
    if window_sec <= guard_sec:
        raise ValueError("boundary audit window_ms must be greater than guard_ms")

    rows = []
    for index, interval in enumerate(intervals):
        start = float(interval.start_sec)
        end = float(interval.end_sec)
        onset_pre = _window_rms_dbfs(
            signal, sample_rate, start - window_sec, start - guard_sec
        )
        onset_inside = _window_rms_dbfs(
            signal, sample_rate, start + guard_sec, min(end, start + window_sec)
        )
        offset_inside = _window_rms_dbfs(
            signal, sample_rate, max(start, end - window_sec), end - guard_sec
        )
        offset_post = _window_rms_dbfs(
            signal, sample_rate, end + guard_sec, end + window_sec
        )
        onset_contrast = onset_inside - onset_pre
        offset_contrast = offset_inside - offset_post
        ambiguous_onset = bool(
            np.isfinite(onset_contrast) and onset_contrast < minimum_contrast_db
        )
        ambiguous_offset = bool(
            np.isfinite(offset_contrast) and offset_contrast < minimum_contrast_db
        )

        display_start = float("nan")
        display_end = float("nan")
        if index < len(display_speech):
            display_start = float(display_speech.loc[index, "start_sec"])
            display_end = float(display_speech.loc[index, "end_sec"])
        rows.append(
            {
                "segment_index": index,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": end - start,
                "display_start_sec": display_start,
                "display_end_sec": display_end,
                "display_onset_delta_ms": (
                    1000.0 * (display_start - start)
                    if np.isfinite(display_start)
                    else np.nan
                ),
                "display_offset_delta_ms": (
                    1000.0 * (display_end - end)
                    if np.isfinite(display_end)
                    else np.nan
                ),
                "onset_pre_rms_dbfs": onset_pre,
                "onset_inside_rms_dbfs": onset_inside,
                "onset_contrast_db": onset_contrast,
                "offset_inside_rms_dbfs": offset_inside,
                "offset_post_rms_dbfs": offset_post,
                "offset_contrast_db": offset_contrast,
                "ambiguous_onset": ambiguous_onset,
                "ambiguous_offset": ambiguous_offset,
                "boundary_review_flag": ambiguous_onset or ambiguous_offset,
            }
        )
    return pd.DataFrame(rows, columns=BOUNDARY_AUDIT_COLUMNS)


def intervals_to_frame(views: dict[str, list[Interval]], file_name: str) -> pd.DataFrame:
    rows = []
    for view, intervals in views.items():
        for index, item in enumerate(intervals):
            rows.append(
                {
                    "file_name": file_name,
                    "view": view,
                    "interval_index": index,
                    "start_sec": item.start_sec,
                    "end_sec": item.end_sec,
                    "duration_sec": item.duration_sec,
                }
            )
    return pd.DataFrame(rows)


def segmentation_frame_diagnostics(
    waveform: np.ndarray,
    sample_rate: int,
    views: dict[str, list[Interval]],
    *,
    frame_ms: float = 30,
    hop_ms: float = 10,
) -> pd.DataFrame:
    """Compute auditable frame RMS and interval-membership traces for plotting/QC."""
    signal = np.asarray(waveform, dtype=float).reshape(-1)
    frame = max(1, int(round(sample_rate * frame_ms / 1000)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000)))
    if signal.size == 0:
        return pd.DataFrame(
            columns=[
                "start_sec",
                "end_sec",
                "mid_sec",
                "rms_dbfs",
                "raw_speech",
                "primary_speech",
                "strict_speech",
                "strict_internal_nonspeech",
            ]
        )
    starts = np.arange(0, max(1, signal.size - frame + 1), hop, dtype=int)
    if starts.size == 0:
        starts = np.array([0], dtype=int)
    rows = []
    for start in starts:
        stop = min(signal.size, start + frame)
        chunk = signal[start:stop]
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        rows.append(
            {
                "start_sec": start / sample_rate,
                "end_sec": stop / sample_rate,
                "mid_sec": (start + stop) / (2 * sample_rate),
                "rms_dbfs": 20 * np.log10(max(rms, 1e-12)),
            }
        )
    output = pd.DataFrame(rows)
    midpoints = output["mid_sec"].to_numpy()
    for view in [
        "raw_speech",
        "primary_speech",
        "strict_speech",
        "strict_internal_nonspeech",
    ]:
        output[view] = frame_membership(midpoints, views.get(view, []))
    return output


def summarize_segmentation(
    waveform: np.ndarray,
    sample_rate: int,
    views: dict[str, list[Interval]],
) -> tuple[dict[str, object], pd.DataFrame]:
    """Summarize the primary Silero view using the original reading-task QC quantities."""
    signal = np.asarray(waveform, dtype=float).reshape(-1)
    duration = signal.size / sample_rate if sample_rate > 0 else 0.0
    primary = views.get("primary_speech", [])
    internal = internal_nonspeech(primary, duration)
    frames = segmentation_frame_diagnostics(signal, sample_rate, views)
    speech_duration = float(sum(interval.duration_sec for interval in primary))
    summary = {
        "duration_sec": float(duration),
        "speech_duration_sec": speech_duration,
        "speech_fraction": speech_duration / duration if duration > 0 else np.nan,
        "n_speech_segments": int(len(primary)),
        "n_internal_nonspeech_segments": int(len(internal)),
        "longest_internal_nonspeech_sec": float(
            max((interval.duration_sec for interval in internal), default=0.0)
        ),
        "rms_db_median": float(frames["rms_dbfs"].median()) if not frames.empty else np.nan,
        "rms_db_std": float(frames["rms_dbfs"].std(ddof=0)) if not frames.empty else np.nan,
    }
    return summary, frames


def classify_reading_segmentation(summary: dict[str, object]) -> dict[str, str]:
    """Preserve the original ALS-reading triage: hard failures vs soft review flags."""
    flags: list[str] = []
    speech_fraction = pd.to_numeric(
        pd.Series([summary.get("speech_fraction")]), errors="coerce"
    ).iloc[0]
    duration = pd.to_numeric(pd.Series([summary.get("duration_sec")]), errors="coerce").iloc[0]
    longest_pause = pd.to_numeric(
        pd.Series([summary.get("longest_internal_nonspeech_sec")]), errors="coerce"
    ).iloc[0]
    rms_median = pd.to_numeric(pd.Series([summary.get("rms_db_median")]), errors="coerce").iloc[0]
    rms_std = pd.to_numeric(pd.Series([summary.get("rms_db_std")]), errors="coerce").iloc[0]
    n_segments = int(summary.get("n_speech_segments", 0) or 0)

    if n_segments == 0:
        flags.append("no_speech_detected")
    if pd.notna(speech_fraction) and speech_fraction < 0.05:
        flags.append("very_low_speech_fraction")
    if pd.notna(duration) and duration < 1.0:
        flags.append("very_short_file")
    if pd.notna(rms_median) and pd.notna(rms_std) and rms_median < -60 and rms_std < 3:
        flags.append("near_silent_or_noise_only")
    if n_segments > 25:
        flags.append("extreme_fragmentation")
    if pd.notna(longest_pause) and longest_pause > 5.0:
        flags.append("extreme_internal_pause")

    hard = {
        "no_speech_detected",
        "very_low_speech_fraction",
        "very_short_file",
        "near_silent_or_noise_only",
    }
    soft = {"extreme_fragmentation", "extreme_internal_pause"}
    if hard.intersection(flags):
        status = "excluded"
    elif soft.intersection(flags):
        status = "flagged"
    else:
        status = "accepted"
    return {"qc_status": status, "qc_flags": ";".join(flags)}


def frame_membership(midpoints_sec: np.ndarray, intervals: Iterable[Interval]) -> np.ndarray:
    midpoints = np.asarray(midpoints_sec, dtype=float)
    mask = np.zeros(midpoints.shape, dtype=bool)
    for item in intervals:
        mask |= (midpoints >= item.start_sec) & (midpoints < item.end_sec)
    return mask
