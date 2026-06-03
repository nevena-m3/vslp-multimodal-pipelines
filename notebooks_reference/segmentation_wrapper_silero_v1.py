from pathlib import Path
import numpy as np
import pandas as pd
import torch


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def boolean_runs(mask):
    """
    Return contiguous runs of a boolean mask as:
    (value, start_idx, end_idx_exclusive)
    """
    mask = np.asarray(mask, dtype=bool)

    if mask.size == 0:
        return []

    runs = []
    start = 0
    current = mask[0]

    for i in range(1, len(mask)):
        if mask[i] != current:
            runs.append((bool(current), int(start), int(i)))
            start = i
            current = mask[i]

    runs.append((bool(current), int(start), int(len(mask))))
    return runs


def timestamps_to_mask(timestamps, n_samples, sr):
    """
    Convert Silero speech timestamps into a sample-level boolean mask.
    """
    mask = np.zeros(int(n_samples), dtype=bool)

    for ts in timestamps:
        s = int(max(0, ts["start"]))
        e = int(min(n_samples, ts["end"]))
        if e > s:
            mask[s:e] = True

    return mask


def sample_mask_to_frame_mask(sample_mask, sr, frame_ms=30):
    """
    Downsample sample-level mask into non-overlapping frame-level mask.
    A frame is marked speech if >50% of samples in the frame are speech.
    """
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = frame_len

    if frame_len <= 0:
        raise ValueError("frame_len must be positive")

    if len(sample_mask) == 0:
        return np.array([], dtype=bool), frame_len, hop_len

    n_frames = int(np.ceil(len(sample_mask) / hop_len))
    total_len = n_frames * hop_len

    if total_len > len(sample_mask):
        sample_mask = np.pad(sample_mask, (0, total_len - len(sample_mask)), mode="constant")

    frame_mask = []
    for start in range(0, len(sample_mask), hop_len):
        chunk = sample_mask[start:start + frame_len]
        frac = float(np.mean(chunk)) if len(chunk) > 0 else 0.0
        frame_mask.append(frac > 0.5)

    return np.asarray(frame_mask, dtype=bool), frame_len, hop_len


def frame_rms_and_db(x, sr, frame_len, hop_len):
    """
    Compute RMS and RMS dB per frame.
    """
    eps = 1e-10

    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])

    n_frames = int(np.ceil(len(x) / hop_len))
    total_len = n_frames * hop_len

    if total_len > len(x):
        x = np.pad(x, (0, total_len - len(x)), mode="constant")

    mids = []
    rms = []
    rms_db = []

    for i, start in enumerate(range(0, len(x), hop_len)):
        frame = x[start:start + frame_len]
        r = float(np.sqrt(np.mean(frame ** 2))) if len(frame) > 0 else 0.0
        mids.append((start + frame_len / 2.0) / sr)
        rms.append(r)
        rms_db.append(20.0 * np.log10(max(r, eps)))

    return np.asarray(mids), np.asarray(rms), np.asarray(rms_db)


def mask_to_segments(mask, frame_hop_sec):
    """
    Convert frame-level speech mask into speech/non-speech segments
    with roles compatible with the plotting/debug workflow.
    """
    rows = []

    for val, s, e in boolean_runs(mask):
        start_sec = s * frame_hop_sec
        end_sec = e * frame_hop_sec
        dur = end_sec - start_sec

        rows.append({
            "segment_type": "speech" if val else "nonspeech",
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "duration_sec": float(dur),
            "run_start_frame": int(s),
            "run_end_frame": int(e),
        })

    segs = pd.DataFrame(rows)

    if len(segs) == 0:
        return segs

    roles = []
    for i, row in segs.iterrows():
        if row["segment_type"] == "speech":
            roles.append("speech")
        else:
            if i == segs.index.min():
                roles.append("leading_nonspeech")
            elif i == segs.index.max():
                roles.append("trailing_nonspeech")
            else:
                roles.append("internal_nonspeech")

    segs["segment_role"] = roles
    return segs


# --------------------------------------------------
# Main Silero wrapper
# --------------------------------------------------

def build_silero_stage_from_audio(
    x,
    sr,
    model,
    get_speech_timestamps_fn,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=100,
    speech_pad_ms=50,
    return_seconds=False,
    frame_ms=30,
):
    """
    Run Silero VAD on a pre-decoded waveform and convert the result into
    a stage-like object similar to the WebRTC notebook - for comparison.

    Parameters
    ----------
    x : np.ndarray
        Mono float waveform in [-1, 1].
    sr : int
        Sample rate. Silero commonly expects 8k or 16k. Use 16k in the decode path.
    model : torch.nn.Module
        Loaded Silero model.
    get_speech_timestamps_fn : callable
        Function returned by silero utils.
    threshold : float
        Silero VAD speech threshold.
    min_speech_duration_ms : int
        Minimum speech chunk duration.
    min_silence_duration_ms : int
        Minimum silence between speech chunks.
    speech_pad_ms : int
        Padding around detected speech chunks.
    return_seconds : bool
        Leave False so timestamps stay in samples.
    frame_ms : int
        Diagnostic frame size for frame_df and plotting.

    Returns
    -------
    dict with:
        info, audio, frame_df, segments_df, boundaries_df
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)

    if sr not in {8000, 16000}:
        raise ValueError(f"Silero VAD usually expects 8000 or 16000 Hz, got {sr}")

    wav_tensor = torch.from_numpy(x)

    speech_timestamps = get_speech_timestamps_fn(
        wav_tensor,
        model,
        threshold=threshold,
        sampling_rate=sr,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=return_seconds,
    )

    sample_mask = timestamps_to_mask(speech_timestamps, len(x), sr)
    frame_mask, frame_len, hop_len = sample_mask_to_frame_mask(sample_mask, sr, frame_ms=frame_ms)

    mids, rms, rms_db = frame_rms_and_db(x, sr, frame_len, hop_len)

    frame_df = pd.DataFrame({
        "frame_idx": np.arange(len(frame_mask), dtype=int),
        "mid_sec": mids,
        "rms": rms,
        "rms_db": rms_db,
        "speech_vad_raw": frame_mask,
        "speech_vad_smooth": frame_mask,
        "speech_mask_strict": frame_mask,
        "nonspeech_mask_strict": ~frame_mask,
        "threshold": threshold,
        "frame_ms": frame_ms,
    })

    frame_hop_sec = hop_len / sr
    segments_df = mask_to_segments(frame_mask, frame_hop_sec=frame_hop_sec)

    if len(segments_df) > 1:
        boundaries_df = pd.DataFrame({
            "boundary_sec": segments_df["end_sec"].iloc[:-1].values
        })
    else:
        boundaries_df = pd.DataFrame(columns=["boundary_sec"])

    info = {
        "method": "silero_vad",
        "threshold": threshold,
        "frame_ms": frame_ms,
        "sample_rate_analysis": sr,
        "min_speech_duration_ms": min_speech_duration_ms,
        "min_silence_duration_ms": min_silence_duration_ms,
        "speech_pad_ms": speech_pad_ms,
    }

    audio = {
        "x_analysis": x,
        "sr_analysis": sr,
        "duration_analysis_sec": float(len(x) / sr) if sr > 0 else np.nan,
        "peak_abs_raw": float(np.max(np.abs(x))) if len(x) > 0 else np.nan,
        "peak_abs_dc": float(np.max(np.abs(x - np.mean(x)))) if len(x) > 0 else np.nan,
        "rms_raw": float(np.sqrt(np.mean(x ** 2))) if len(x) > 0 else np.nan,
    }

    return {
        "info": info,
        "audio": audio,
        "frame_df": frame_df,
        "segments_df": segments_df,
        "boundaries_df": boundaries_df,
        "speech_timestamps": speech_timestamps,
    }


def summarize_silero_stage(stage, row=None):
    """
    Build a flat summary dictionary for saving and comparison.
    """
    frame_df = stage["frame_df"]
    segs = stage["segments_df"]
    audio = stage["audio"]
    info = stage["info"]

    n_speech_segments = int((segs["segment_type"] == "speech").sum()) if len(segs) else 0
    n_internal_nonspeech = int((segs["segment_role"] == "internal_nonspeech").sum()) if len(segs) else 0

    leading_ns = 0.0
    trailing_ns = 0.0
    longest_internal_ns = 0.0

    if len(segs) > 0:
        tmp = segs.loc[segs["segment_role"] == "leading_nonspeech", "duration_sec"]
        if len(tmp) > 0:
            leading_ns = float(tmp.iloc[0])

        tmp = segs.loc[segs["segment_role"] == "trailing_nonspeech", "duration_sec"]
        if len(tmp) > 0:
            trailing_ns = float(tmp.iloc[0])

        tmp = segs.loc[segs["segment_role"] == "internal_nonspeech", "duration_sec"]
        if len(tmp) > 0:
            longest_internal_ns = float(tmp.max())

    speech_fraction = float(frame_df["speech_vad_smooth"].mean()) if len(frame_df) else np.nan
    rms_db_median = float(np.median(frame_df["rms_db"])) if len(frame_df) else np.nan
    rms_db_std = float(np.std(frame_df["rms_db"])) if len(frame_df) else np.nan

    out = {
        "method": "silero_vad",
        "duration_sec": audio["duration_analysis_sec"],
        "sample_rate_analysis": audio["sr_analysis"],
        "threshold": info["threshold"],
        "frame_ms": info["frame_ms"],
        "min_speech_duration_ms": info["min_speech_duration_ms"],
        "min_silence_duration_ms": info["min_silence_duration_ms"],
        "speech_pad_ms": info["speech_pad_ms"],

        "n_frames": int(len(frame_df)),
        "n_segments_total": int(len(segs)),
        "n_speech_segments": n_speech_segments,
        "n_internal_nonspeech_segments": n_internal_nonspeech,

        "speech_fraction": speech_fraction,
        "leading_nonspeech_sec": leading_ns,
        "trailing_nonspeech_sec": trailing_ns,
        "longest_internal_nonspeech_sec": longest_internal_ns,

        "rms_db_median": rms_db_median,
        "rms_db_std": rms_db_std,
    }

    if row is not None:
        out.update({
            "file_name": row.get("file_name"),
            "file_path": row.get("file_path"),
            "ID_norm": row.get("ID_norm"),
            "Diagnosis": row.get("Diagnosis"),
            "severity_bin": row.get("severity_bin"),
            "Recording date": row.get("Recording date"),
        })

    return out