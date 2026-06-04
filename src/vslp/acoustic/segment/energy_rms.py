"""Simple RMS/energy segmentation placeholder.

This is intentionally conservative and mainly useful as a baseline/fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from vslp.acoustic.segment.silero_wrapper import sample_mask_to_frame_mask, frame_rms_and_db, mask_to_segments


def build_energy_rms_stage_from_audio(x, sr, frame_ms=30, threshold_db=None, low_energy_quantile=0.15):
    x = np.asarray(x, dtype=np.float32)
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = frame_len
    mids, rms, rms_db = frame_rms_and_db(x, sr, frame_len, hop_len)
    if threshold_db is None:
        finite = np.isfinite(rms_db)
        threshold_db = float(np.nanpercentile(rms_db[finite], 35)) if finite.any() else -35.0
    frame_mask = rms_db > threshold_db
    segments_df = mask_to_segments(frame_mask, frame_hop_sec=hop_len / sr)
    boundaries_df = pd.DataFrame({"boundary_sec": segments_df["end_sec"].iloc[:-1].values}) if len(segments_df) > 1 else pd.DataFrame(columns=["boundary_sec"])
    frame_df = pd.DataFrame({
        "frame_idx": np.arange(len(frame_mask), dtype=int),
        "mid_sec": mids,
        "rms": rms,
        "rms_db": rms_db,
        "speech_vad_raw": frame_mask,
        "speech_vad_smooth": frame_mask,
        "speech_mask_strict": frame_mask,
        "nonspeech_mask_strict": ~frame_mask,
        "threshold": threshold_db,
        "frame_ms": frame_ms,
    })
    return {
        "info": {"method": "energy_rms", "threshold_db": threshold_db, "frame_ms": frame_ms, "sample_rate_analysis": sr},
        "audio": {"x_analysis": x, "sr_analysis": sr, "duration_analysis_sec": float(len(x) / sr) if sr else np.nan},
        "frame_df": frame_df,
        "segments_df": segments_df,
        "boundaries_df": boundaries_df,
    }
