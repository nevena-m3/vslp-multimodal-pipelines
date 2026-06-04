"""Segmentation-informed acoustic quality-control feature extraction.

This stage sits immediately after Data Segmentation and before acoustic feature extraction.
It computes interpretable quality features grouped into six artifact families derived from
provided QC notebooks:

1. additive_interference
2. gain_dynamics
3. reverberation_echo
4. channel_device
5. nonlinear_distortion
6. temporal_discontinuity

The implementation is intentionally conservative. Features are designed as QC/proxy
screening measurements, not clinical exclusion criteria.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal, stats

from vslp.core.project import ensure_stage_folders
from vslp.core.provenance import python_environment, sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult

QC_FAMILIES: dict[str, dict[str, str]] = {
    "additive_interference": {
        "label": "Additive interference",
        "meaning": "Background noise or hum added to the recording, especially visible during pauses.",
    },
    "gain_dynamics": {
        "label": "Gain dynamics",
        "meaning": "Level changes, automatic gain control, or unstable recording amplitude during speech.",
    },
    "reverberation_echo": {
        "label": "Reverberation / echo",
        "meaning": "Speech energy continuing into pauses after speech offset, suggesting room echo or reverberation.",
    },
    "channel_device": {
        "label": "Channel / device",
        "meaning": "Spectral coloration, bandwidth limitation, or microphone/device response differences.",
    },
    "nonlinear_distortion": {
        "label": "Nonlinear distortion",
        "meaning": "Clipping, saturation, peak flattening, or overload-like distortion at high amplitudes.",
    },
    "temporal_discontinuity": {
        "label": "Temporal discontinuity",
        "meaning": "Dropouts, silent holes inside speech, frozen repeated windows, or abrupt waveform breaks.",
    },
}

FAMILY_FEATURES: dict[str, list[str]] = {
    "additive_interference": [
        "qadd_pause_rms_db_median", "qadd_pause_rms_db_std", "qadd_pause_peak_db_median",
        "qadd_speech_pause_level_diff_db", "qadd_pause_hum50_ratio", "qadd_pause_hum60_ratio",
        "qadd_pause_lowband_ratio", "qadd_pause_midband_ratio", "qadd_pause_highband_ratio",
        "qadd_pause_flatness_median", "qadd_pause_centroid_hz_median", "qadd_status", "qadd_flags",
    ],
    "gain_dynamics": [
        "qgain_speech_rms_db_median", "qgain_speech_rms_db_std", "qgain_speech_rms_db_iqr",
        "qgain_speech_peak95_db", "qgain_speech_peak99_db", "qgain_speech_crest_factor_db",
        "qgain_rolling_rms_instability", "qgain_start_end_diff_db", "qgain_level_slope_db_per_frame",
        "qgain_segment_level_std_db", "qgain_status", "qgain_flags",
    ],
    "reverberation_echo": [
        "qrev_total_offset_pause_sec", "qrev_median_offset_pause_sec", "qrev_post_offset_tail_db_above_floor",
        "qrev_tail_duration_above_floor_sec", "qrev_decay_slope_db_per_sec", "qrev_boundary_blur_index",
        "qrev_early_vs_late_pause_ratio", "qrev_status", "qrev_flags",
    ],
    "channel_device": [
        "qchan_speech_lowband_ratio", "qchan_speech_midband_ratio", "qchan_speech_highband_ratio",
        "qchan_speech_rolloff95_hz", "qchan_speech_centroid_hz", "qchan_speech_flatness",
        "qchan_speech_spectral_tilt_db_per_oct", "qchan_bandwidth_proxy_hz", "qchan_status", "qchan_flags",
    ],
    "nonlinear_distortion": [
        "qdist_n_high_level_frames", "qdist_high_level_frame_fraction", "qdist_total_high_level_sec",
        "qdist_clipped_sample_fraction", "qdist_near_clipped_sample_fraction", "qdist_edge_pileup_fraction",
        "qdist_clipped_frame_fraction", "qdist_near_clipped_frame_fraction", "qdist_flat_top_peak_fraction",
        "qdist_peak_asymmetry_index", "qdist_peak_curvature_abnormality", "qdist_status", "qdist_flags",
    ],
    "temporal_discontinuity": [
        "qtemp_n_speech_samples", "qtemp_speech_sample_fraction", "qtemp_zero_run_count",
        "qtemp_dropout_fraction", "qtemp_silent_hole_inside_speech_count", "qtemp_abrupt_energy_jump_count",
        "qtemp_energy_jump_fraction", "qtemp_duplicate_window_run_count", "qtemp_frozen_window_fraction",
        "qtemp_waveform_continuity_break_score", "qtemp_status", "qtemp_flags",
    ],
}

@dataclass(frozen=True)
class QualityControlConfig:
    selected_families: list[str] | None = None
    minimum_internal_pause_sec: float = 0.15
    high_level_percentile: float = 90.0
    hard_clip_threshold: float = 0.995
    near_clip_threshold: float = 0.95
    zero_threshold: float = 1e-5
    abrupt_jump_db: float = 18.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return np.nan
    return v if np.isfinite(v) else np.nan


def _safe_median(x) -> float:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else np.nan


def _iqr(x) -> float:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 75) - np.percentile(vals, 25)) if vals.size else np.nan


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = np.mean(x, axis=1).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, int(sr)


def _band_ratio(freqs, psd, low, high) -> float:
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    total = np.trapezoid(psd[np.isfinite(psd)], freqs[np.isfinite(psd)]) if np.any(np.isfinite(psd)) else 0.0
    if not np.isfinite(total) or total <= 0:
        return np.nan
    mask = (freqs >= low) & (freqs < high) & np.isfinite(psd)
    if not mask.any():
        return np.nan
    return float(np.trapezoid(psd[mask], freqs[mask]) / total)


def _spectral_flatness(psd) -> float:
    p = np.asarray(psd, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return np.nan
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def _spectral_centroid(freqs, psd) -> float:
    freqs = np.asarray(freqs, dtype=float)
    p = np.asarray(psd, dtype=float)
    mask = np.isfinite(freqs) & np.isfinite(p) & (p > 0)
    if not mask.any() or np.sum(p[mask]) <= 0:
        return np.nan
    return float(np.sum(freqs[mask] * p[mask]) / np.sum(p[mask]))


def _rolloff(freqs, psd, pct=0.95) -> float:
    freqs = np.asarray(freqs, dtype=float)
    p = np.asarray(psd, dtype=float)
    mask = np.isfinite(freqs) & np.isfinite(p) & (p >= 0)
    if not mask.any() or np.sum(p[mask]) <= 0:
        return np.nan
    f = freqs[mask]
    c = np.cumsum(p[mask])
    idx = np.searchsorted(c, pct * c[-1])
    idx = min(max(idx, 0), len(f) - 1)
    return float(f[idx])


def _frame_half_width(frames: pd.DataFrame, sr: int) -> float:
    if "frame_ms" in frames.columns:
        v = _safe_float(frames["frame_ms"].iloc[0]) / 1000.0 / 2.0
        if np.isfinite(v) and v > 0:
            return v
    mids = pd.to_numeric(frames.get("mid_sec", pd.Series(dtype=float)), errors="coerce").dropna().values
    if len(mids) > 1:
        return float(np.median(np.diff(mids)) / 2.0)
    return 0.015


def _mask_samples_from_segments(n: int, sr: int, segments: pd.DataFrame, kind: str, min_pause: float=0.15) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    if segments is None or segments.empty:
        if kind == "full":
            mask[:] = True
        return mask
    if kind == "speech":
        sel = segments[segments["segment_type"].astype(str).str.lower().eq("speech")]
    elif kind == "internal_pause":
        if "segment_role" in segments.columns:
            sel = segments[segments["segment_role"].astype(str).str.lower().eq("internal_nonspeech")]
        else:
            sel = segments[segments["segment_type"].astype(str).str.lower().ne("speech")]
        if "duration_sec" in sel.columns:
            sel = sel[pd.to_numeric(sel["duration_sec"], errors="coerce") >= min_pause]
    else:
        sel = segments
    for _, r in sel.iterrows():
        s = max(0, int(round(float(r["start_sec"]) * sr)))
        e = min(n, int(round(float(r["end_sec"]) * sr)))
        if e > s:
            mask[s:e] = True
    return mask


def _compute_additive(frames: pd.DataFrame, segments: pd.DataFrame, x: np.ndarray, sr: int, cfg: QualityControlConfig) -> dict[str, Any]:
    out = {"qadd_status": "computed", "qadd_flags": ""}
    pause_mask = frames.get("nonspeech_mask_strict", pd.Series(False, index=frames.index)).astype(bool)
    speech_mask = frames.get("speech_mask_strict", pd.Series(False, index=frames.index)).astype(bool)
    pause_db = pd.to_numeric(frames.loc[pause_mask, "rms_db"], errors="coerce").dropna().values if "rms_db" in frames else np.array([])
    speech_db = pd.to_numeric(frames.loc[speech_mask, "rms_db"], errors="coerce").dropna().values if "rms_db" in frames else np.array([])
    if pause_db.size < 3:
        out["qadd_status"] = "insufficient_pause_support"; out["qadd_flags"] = "few_pause_frames"; return out
    out["qadd_pause_rms_db_median"] = _safe_median(pause_db)
    out["qadd_pause_rms_db_std"] = float(np.std(pause_db)) if pause_db.size else np.nan
    out["qadd_pause_peak_db_median"] = float(np.percentile(pause_db, 95)) if pause_db.size else np.nan
    out["qadd_speech_pause_level_diff_db"] = _safe_median(speech_db) - _safe_median(pause_db) if speech_db.size else np.nan
    p_mask = _mask_samples_from_segments(len(x), sr, segments, "internal_pause", cfg.minimum_internal_pause_sec)
    xp = x[p_mask]
    if len(xp) < max(256, int(0.05 * sr)):
        out["qadd_flags"] = "insufficient_pause_audio"
        return out
    nper = min(2048, max(256, len(xp)))
    freqs, psd = signal.welch(xp, fs=sr, nperseg=nper)
    out["qadd_pause_lowband_ratio"] = _band_ratio(freqs, psd, 0, 300)
    out["qadd_pause_midband_ratio"] = _band_ratio(freqs, psd, 300, 3000)
    out["qadd_pause_highband_ratio"] = _band_ratio(freqs, psd, 3000, sr/2)
    out["qadd_pause_hum50_ratio"] = _band_ratio(freqs, psd, 48, 52)
    out["qadd_pause_hum60_ratio"] = _band_ratio(freqs, psd, 58, 62)
    out["qadd_pause_flatness_median"] = _spectral_flatness(psd)
    out["qadd_pause_centroid_hz_median"] = _spectral_centroid(freqs, psd)
    return out


def _compute_gain(frames: pd.DataFrame, segments: pd.DataFrame) -> dict[str, Any]:
    out = {"qgain_status": "computed", "qgain_flags": ""}
    speech_mask = frames.get("speech_mask_strict", pd.Series(False, index=frames.index)).astype(bool)
    speech_db = pd.to_numeric(frames.loc[speech_mask, "rms_db"], errors="coerce").dropna().values if "rms_db" in frames else np.array([])
    if speech_db.size < 5:
        out["qgain_status"] = "insufficient_speech_support"; out["qgain_flags"] = "few_speech_frames"; return out
    out["qgain_speech_rms_db_median"] = _safe_median(speech_db)
    out["qgain_speech_rms_db_std"] = float(np.std(speech_db))
    out["qgain_speech_rms_db_iqr"] = _iqr(speech_db)
    out["qgain_speech_peak95_db"] = float(np.percentile(speech_db, 95))
    out["qgain_speech_peak99_db"] = float(np.percentile(speech_db, 99))
    out["qgain_speech_crest_factor_db"] = out["qgain_speech_peak99_db"] - out["qgain_speech_rms_db_median"]
    w = max(3, min(15, len(speech_db)//4))
    roll = pd.Series(speech_db).rolling(w, center=True, min_periods=2).median().dropna().values
    out["qgain_rolling_rms_instability"] = float(np.std(roll)) if len(roll) else np.nan
    k = max(1, int(0.1 * len(speech_db)))
    out["qgain_start_end_diff_db"] = float(np.mean(speech_db[-k:]) - np.mean(speech_db[:k]))
    out["qgain_level_slope_db_per_frame"] = float(np.polyfit(np.arange(len(speech_db)), speech_db, 1)[0]) if len(speech_db) >= 2 else np.nan
    if not segments.empty and "segment_type" in segments.columns:
        speech_segments = segments[segments["segment_type"].astype(str).str.lower().eq("speech")]
        mids = pd.to_numeric(frames.get("mid_sec", pd.Series(np.arange(len(frames)))), errors="coerce").values
        seg_meds = []
        for _, seg in speech_segments.iterrows():
            mask = (mids >= float(seg["start_sec"])) & (mids <= float(seg["end_sec"]))
            vals = pd.to_numeric(frames.loc[mask, "rms_db"], errors="coerce").dropna().values
            if vals.size:
                seg_meds.append(float(np.median(vals)))
        if seg_meds:
            out["qgain_segment_level_std_db"] = float(np.std(seg_meds))
    return out


def _compute_reverb(frames: pd.DataFrame, segments: pd.DataFrame) -> dict[str, Any]:
    out = {"qrev_status": "computed_proxy", "qrev_flags": ""}
    if segments.empty or "segment_type" not in segments.columns or "rms_db" not in frames.columns:
        out["qrev_status"] = "insufficient_boundary_support"; out["qrev_flags"] = "missing_segments_or_frames"; return out
    mids = pd.to_numeric(frames.get("mid_sec", pd.Series(dtype=float)), errors="coerce").values
    rms = pd.to_numeric(frames["rms_db"], errors="coerce").values
    tails = []; durations=[]; ratios=[]
    for i in range(len(segments)-1):
        a=segments.iloc[i]; b=segments.iloc[i+1]
        if str(a.get("segment_type","")).lower()=="speech" and str(b.get("segment_role","")).lower()=="internal_nonspeech":
            pause_dur=float(b["duration_sec"]); durations.append(pause_dur)
            early=(mids>=float(b["start_sec"])) & (mids<min(float(b["end_sec"]), float(b["start_sec"])+0.25))
            late=(mids>max(float(b["start_sec"]), float(b["end_sec"])-0.25)) & (mids<=float(b["end_sec"]))
            speech_tail=(mids>=max(float(a["start_sec"]), float(a["end_sec"])-0.25)) & (mids<=float(a["end_sec"]))
            floor=np.nanmedian(rms[late]) if np.any(late) else np.nanmedian(rms)
            early_db=np.nanmedian(rms[early]) if np.any(early) else np.nan
            speech_end_db=np.nanmedian(rms[speech_tail]) if np.any(speech_tail) else np.nan
            tails.append(early_db - floor if np.isfinite(early_db) and np.isfinite(floor) else np.nan)
            ratios.append(early_db - speech_end_db if np.isfinite(early_db) and np.isfinite(speech_end_db) else np.nan)
    if not durations:
        out["qrev_status"]="insufficient_offset_pause_support"; out["qrev_flags"]="no_speech_to_internal_pause_offsets"; return out
    out["qrev_total_offset_pause_sec"] = float(np.sum(durations))
    out["qrev_median_offset_pause_sec"] = _safe_median(durations)
    out["qrev_post_offset_tail_db_above_floor"] = _safe_median(tails)
    out["qrev_early_vs_late_pause_ratio"] = _safe_median(ratios)
    out["qrev_boundary_blur_index"] = _safe_median(tails)
    out["qrev_tail_duration_above_floor_sec"] = float(np.sum([d for d,t in zip(durations,tails) if np.isfinite(t) and t > 6.0]))
    return out


def _compute_channel(segments: pd.DataFrame, x: np.ndarray, sr: int) -> dict[str, Any]:
    out={"qchan_status":"computed", "qchan_flags":""}
    mask=_mask_samples_from_segments(len(x), sr, segments, "speech")
    xs=x[mask]
    if len(xs)<max(512, int(0.2*sr)):
        out["qchan_status"]="insufficient_speech_spectral_support"; out["qchan_flags"]="few_speech_samples"; return out
    nper=min(4096, max(512, len(xs)))
    freqs, psd=signal.welch(xs, fs=sr, nperseg=nper)
    out["qchan_speech_lowband_ratio"]=_band_ratio(freqs, psd, 0, 300)
    out["qchan_speech_midband_ratio"]=_band_ratio(freqs, psd, 300, 3000)
    out["qchan_speech_highband_ratio"]=_band_ratio(freqs, psd, 3000, sr/2)
    out["qchan_speech_rolloff95_hz"]=_rolloff(freqs, psd)
    out["qchan_speech_centroid_hz"]=_spectral_centroid(freqs, psd)
    out["qchan_speech_flatness"]=_spectral_flatness(psd)
    valid=(freqs>=100)&(freqs<=min(4000,sr/2))&(psd>0)&np.isfinite(psd)
    if np.sum(valid)>5:
        out["qchan_speech_spectral_tilt_db_per_oct"] = float(np.polyfit(np.log2(freqs[valid]), 10*np.log10(psd[valid]), 1)[0])
    above=np.where((psd >= np.nanmax(psd)*0.01) & np.isfinite(psd))[0] if np.any(np.isfinite(psd)) else []
    out["qchan_bandwidth_proxy_hz"] = float(freqs[above[-1]]) if len(above) else np.nan
    return out


def _compute_distortion(frames: pd.DataFrame, x: np.ndarray, sr: int, cfg: QualityControlConfig) -> dict[str, Any]:
    out={"qdist_status":"computed", "qdist_flags":""}
    speech_mask = frames.get("speech_mask_strict", pd.Series(False, index=frames.index)).astype(bool)
    rms = pd.to_numeric(frames.get("rms_db", pd.Series(dtype=float)), errors="coerce")
    vals = rms[speech_mask].dropna().values
    if vals.size < 3:
        out["qdist_status"]="insufficient_high_level_speech_support"; out["qdist_flags"]="few_speech_frames"; return out
    thr=np.percentile(vals, cfg.high_level_percentile)
    high=(speech_mask.values)&(rms.values>=thr)
    out["qdist_n_high_level_frames"] = int(np.sum(high))
    out["qdist_high_level_frame_fraction"] = float(np.mean(high)) if len(high) else np.nan
    half=_frame_half_width(frames, sr)
    mids=pd.to_numeric(frames.get("mid_sec", pd.Series(dtype=float)), errors="coerce").values
    masks=[]
    for mid, keep in zip(mids, high, strict=False):
        if not keep or not np.isfinite(mid): continue
        s=max(0,int(round((mid-half)*sr))); e=min(len(x),int(round((mid+half)*sr)))
        if e>s: masks.append(x[s:e])
    if not masks:
        out["qdist_status"]="insufficient_high_level_speech_support"; out["qdist_flags"]="no_high_level_windows"; return out
    y=np.concatenate(masks)
    ab=np.abs(y)
    out["qdist_total_high_level_sec"] = float(len(y)/sr)
    out["qdist_clipped_sample_fraction"] = float(np.mean(ab>=cfg.hard_clip_threshold))
    out["qdist_near_clipped_sample_fraction"] = float(np.mean(ab>=cfg.near_clip_threshold))
    out["qdist_edge_pileup_fraction"] = float(np.mean(ab>=0.98))
    # window-level proxies
    clipped=[]; near=[]; flat=[]; curv=[]
    for w in masks:
        a=np.abs(w); clipped.append(np.any(a>=cfg.hard_clip_threshold)); near.append(np.any(a>=cfg.near_clip_threshold))
        flat.append(np.mean(np.abs(np.diff(w)) < 1e-4) if len(w)>2 else np.nan)
        if len(w)>3 and np.max(a)>0:
            idx=int(np.argmax(a));
            if 0<idx<len(w)-1:
                curv.append(abs(w[idx-1]-2*w[idx]+w[idx+1])/max(a[idx],1e-6))
    out["qdist_clipped_frame_fraction"] = float(np.mean(clipped))
    out["qdist_near_clipped_frame_fraction"] = float(np.mean(near))
    out["qdist_flat_top_peak_fraction"] = _safe_median(flat)
    pos=np.percentile(y[y>0],99) if np.any(y>0) else np.nan; neg=np.percentile(np.abs(y[y<0]),99) if np.any(y<0) else np.nan
    out["qdist_peak_asymmetry_index"] = float((pos-neg)/(pos+neg)) if np.isfinite(pos) and np.isfinite(neg) and (pos+neg)>0 else np.nan
    out["qdist_peak_curvature_abnormality"] = _safe_median(curv)
    return out


def _compute_temporal(frames: pd.DataFrame, x: np.ndarray, sr: int, cfg: QualityControlConfig) -> dict[str, Any]:
    out={"qtemp_status":"computed", "qtemp_flags":""}
    speech_mask_samples = np.ones(len(x), dtype=bool)
    frame_speech = frames.get("speech_mask_strict", pd.Series(False, index=frames.index)).astype(bool)
    mids=pd.to_numeric(frames.get("mid_sec", pd.Series(dtype=float)), errors="coerce").values
    half=_frame_half_width(frames, sr)
    speech_sample_mask=np.zeros(len(x), dtype=bool)
    for mid, sp in zip(mids, frame_speech.values, strict=False):
        if sp and np.isfinite(mid):
            s=max(0,int(round((mid-half)*sr))); e=min(len(x),int(round((mid+half)*sr))); speech_sample_mask[s:e]=True
    if not speech_sample_mask.any(): speech_sample_mask[:] = True
    out["qtemp_n_speech_samples"] = int(np.sum(speech_sample_mask))
    out["qtemp_speech_sample_fraction"] = float(np.mean(speech_sample_mask)) if len(x) else np.nan
    y=x[speech_sample_mask]
    if len(y)<sr*0.1:
        out["qtemp_status"]="insufficient_signal_support"; out["qtemp_flags"]="few_samples"; return out
    zero=np.abs(y)<=cfg.zero_threshold
    edges=np.diff(np.r_[False, zero, False].astype(int)); starts=np.flatnonzero(edges==1); ends=np.flatnonzero(edges==-1)
    runs=[e-s for s,e in zip(starts,ends,strict=False) if (e-s)>=max(5,int(0.01*sr))]
    out["qtemp_zero_run_count"] = int(len(runs))
    out["qtemp_dropout_fraction"] = float(np.sum(runs)/len(y)) if len(y) else np.nan
    rms=pd.to_numeric(frames.get("rms_db", pd.Series(dtype=float)), errors="coerce").values
    valid=frame_speech.values & np.isfinite(rms)
    jumps=np.abs(np.diff(rms))>=cfg.abrupt_jump_db if len(rms)>1 else np.array([])
    valid_pairs=valid[:-1]&valid[1:] if len(valid)>1 else np.array([])
    out["qtemp_abrupt_energy_jump_count"] = int(np.sum(jumps & valid_pairs)) if len(jumps) else 0
    out["qtemp_energy_jump_fraction"] = float(np.mean(jumps & valid_pairs)) if len(jumps) and np.any(valid_pairs) else 0.0
    # duplicate/frozen windows proxy
    frame_len=max(1,int(0.04*sr)); hop=max(1,int(0.04*sr))
    n=min(200, max(0,(len(y)-frame_len)//hop))
    dups=0; frozen=[]
    last=None
    for i in range(n):
        w=y[i*hop:i*hop+frame_len]
        if last is not None and len(w)==len(last):
            corr=np.corrcoef(w,last)[0,1] if np.std(w)>0 and np.std(last)>0 else np.nan
            if np.isfinite(corr) and corr>0.999: dups+=1
        frozen.append(float(np.std(w)<1e-5))
        last=w
    out["qtemp_duplicate_window_run_count"] = int(dups)
    out["qtemp_frozen_window_fraction"] = float(np.nanmean(frozen)) if frozen else np.nan
    out["qtemp_silent_hole_inside_speech_count"] = int(len(runs))
    out["qtemp_waveform_continuity_break_score"] = float(out["qtemp_dropout_fraction"] + out["qtemp_energy_jump_fraction"] + (out["qtemp_frozen_window_fraction"] if np.isfinite(out.get("qtemp_frozen_window_fraction",np.nan)) else 0))
    return out


def _compute_one(row: pd.Series, cfg: QualityControlConfig, families: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    file_name=str(row.get("file_name", ""))
    wav=Path(str(row.get("segmentation_wav_path", "")))
    frames_path=Path(str(row.get("frame_csv_path", "")))
    segments_path=Path(str(row.get("segments_csv_path", "")))
    out={"file_name":file_name, "segmentation_wav_path":str(wav), "frame_csv_path":str(frames_path), "segments_csv_path":str(segments_path)}
    status_rows=[]
    if not wav.exists(): raise FileNotFoundError(f"Missing segmentation WAV: {wav}")
    if not frames_path.exists(): raise FileNotFoundError(f"Missing frame CSV: {frames_path}")
    if not segments_path.exists(): raise FileNotFoundError(f"Missing segments CSV: {segments_path}")
    x,sr=_read_audio(wav); frames=pd.read_csv(frames_path); segments=pd.read_csv(segments_path)
    out["sample_rate_hz"]=sr; out["duration_sec"]=float(len(x)/sr) if sr else np.nan; out["segmentation_wav_sha256"]=sha256_file(wav)
    fam_funcs={
        "additive_interference": lambda: _compute_additive(frames, segments, x, sr, cfg),
        "gain_dynamics": lambda: _compute_gain(frames, segments),
        "reverberation_echo": lambda: _compute_reverb(frames, segments),
        "channel_device": lambda: _compute_channel(segments, x, sr),
        "nonlinear_distortion": lambda: _compute_distortion(frames, x, sr, cfg),
        "temporal_discontinuity": lambda: _compute_temporal(frames, x, sr, cfg),
    }
    for fam in QC_FAMILIES:
        if fam in families:
            try:
                vals=fam_funcs[fam](); out.update(vals); fam_status=vals.get(f"{fam.split('_')[0] if False else ''}", "")
                status_col = next((k for k in vals if k.endswith("_status")), None)
                flag_col = next((k for k in vals if k.endswith("_flags")), None)
                status_rows.append({"file_name":file_name,"family":fam,"family_label":QC_FAMILIES[fam]["label"],"status":vals.get(status_col,"computed"),"flags":vals.get(flag_col,"")})
            except Exception as exc:
                status_rows.append({"file_name":file_name,"family":fam,"family_label":QC_FAMILIES[fam]["label"],"status":"failed","flags":str(exc)})
        else:
            for k in FAMILY_FEATURES[fam]:
                out[k]=np.nan if not k.endswith("status") and not k.endswith("flags") else ("not_selected" if k.endswith("status") else "")
            status_rows.append({"file_name":file_name,"family":fam,"family_label":QC_FAMILIES[fam]["label"],"status":"not_selected","flags":""})
    return out, status_rows


def run_acoustic_quality_control(segmentation_summary_csv: str | Path, output_root: str | Path, config: QualityControlConfig | None=None) -> StageResult:
    cfg=config or QualityControlConfig()
    families=cfg.selected_families or list(QC_FAMILIES.keys())
    families=[f for f in families if f in QC_FAMILIES]
    output_root=Path(output_root); segmentation_summary_csv=Path(segmentation_summary_csv)
    stage_dir=output_root/"acoustic"/"003_quality_control"
    folders=ensure_stage_folders(stage_dir)
    rows=[]; status_rows=[]; errors=[]
    seg=pd.read_csv(segmentation_summary_csv) if segmentation_summary_csv.exists() else pd.DataFrame()
    for _, row in seg.iterrows():
        try:
            vals, st=_compute_one(row, cfg, families); rows.append(vals); status_rows.extend(st)
        except Exception as exc:
            errors.append({"file_name":row.get("file_name", ""), "status":"failed", "error":str(exc)})
    features_csv=folders["tables"]/"acoustic_quality_features.csv"
    main_csv=folders["tables"]/"acoustic_quality_main_summary.csv"
    status_csv=folders["tables"]/"acoustic_quality_family_status.csv"
    family_csv=folders["tables"]/"acoustic_quality_family_summary.csv"
    errors_csv=folders["errors"]/"acoustic_quality_errors.csv"
    feat_df=pd.DataFrame(rows)
    for fam, keys in FAMILY_FEATURES.items():
        for k in keys:
            if k not in feat_df.columns: feat_df[k]=np.nan
    feat_df.to_csv(features_csv,index=False)
    status_df=pd.DataFrame(status_rows, columns=["file_name","family","family_label","status","flags"])
    status_df.to_csv(status_csv,index=False)
    pd.DataFrame(errors).to_csv(errors_csv,index=False)
    fam_summary=status_df.groupby(["family","family_label","status"]).size().reset_index(name="count") if not status_df.empty else pd.DataFrame(columns=["family","family_label","status","count"])
    fam_summary.to_csv(family_csv,index=False)
    main_cols=["file_name","duration_sec","sample_rate_hz","qadd_pause_rms_db_median","qadd_speech_pause_level_diff_db","qgain_speech_rms_db_std","qrev_post_offset_tail_db_above_floor","qchan_speech_centroid_hz","qdist_near_clipped_sample_fraction","qtemp_waveform_continuity_break_score"]
    for c in main_cols:
        if c not in feat_df.columns: feat_df[c]=np.nan
    feat_df[main_cols].to_csv(main_csv,index=False)
    # plots
    p1=folders["plots"]/"quality_family_status.png"; p2=folders["plots"]/"quality_main_features_overview.png"
    _plot_family_status(status_df,p1); _plot_overview(feat_df,p2)
    report=folders["reports"]/"acoustic_quality_control_report.html"
    _write_report(report, feat_df, fam_summary, cfg, p1, p2)
    manifest=StageManifest(
        stage_name="acoustic_quality_control",
        stage_version="0.21.0",
        status="completed_with_warnings" if errors else "completed",
        input_artifacts=[ArtifactRef(path=str(segmentation_summary_csv), role="segmentation_summary", media_type="text/csv")],
        output_artifacts=[ArtifactRef(path=str(features_csv), role="quality_features", media_type="text/csv"), ArtifactRef(path=str(main_csv), role="quality_main_summary", media_type="text/csv"), ArtifactRef(path=str(report), role="quality_report", media_type="text/html")],
        config=cfg.to_dict() | {"selected_families": families},
        environment={"python": python_environment()},
        errors=errors,
        notes=["Quality features are screening/proxy measurements; they do not automatically reject recordings."],
    )
    manifest_path=folders["logs"]/"stage_manifest.json"; manifest.write_json(manifest_path)
    return StageResult(status=manifest.status, manifest_path=manifest_path, summary_table=features_csv, error_table=errors_csv, report_path=report)


def _plot_family_status(status_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax=plt.subplots(figsize=(8,4.5))
    if status_df.empty:
        ax.text(0.5,0.5,"No QC family status rows", ha="center", va="center"); ax.axis("off")
    else:
        counts=status_df.groupby("family_label").size().sort_values()
        ax.barh(counts.index, counts.values)
        ax.set_xlabel("Files/family evaluations"); ax.set_title("QC family coverage")
    fig.tight_layout(); fig.savefig(path,dpi=170); plt.close(fig)


def _plot_overview(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax=plt.subplots(figsize=(8,4.5))
    cols=["qadd_pause_rms_db_median","qgain_speech_rms_db_std","qdist_near_clipped_sample_fraction","qtemp_waveform_continuity_break_score"]
    vals=[]; labels=[]
    for c in cols:
        if c in df:
            v=pd.to_numeric(df[c], errors="coerce").dropna()
            if len(v): vals.append(float(np.nanmedian(v))); labels.append(c.replace("_","\n"))
    if not vals:
        ax.text(0.5,0.5,"No numeric QC overview values", ha="center", va="center"); ax.axis("off")
    else:
        ax.bar(labels, vals); ax.set_title("Median QC feature overview"); ax.tick_params(axis='x', labelsize=8)
    fig.tight_layout(); fig.savefig(path,dpi=170); plt.close(fig)


def _write_report(path: Path, feat_df: pd.DataFrame, fam_summary: pd.DataFrame, cfg: QualityControlConfig, p1: Path, p2: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n=len(feat_df)
    family_rows="".join(f"<tr><td>{r.family_label}</td><td>{r.status}</td><td>{r['count']}</td></tr>" for _,r in fam_summary.iterrows()) if not fam_summary.empty else "<tr><td colspan='3'>No family rows</td></tr>"
    html=f"""<!doctype html><html><head><meta charset='utf-8'><title>VSLP Acoustic Quality Control</title>
<style>body{{font-family:Arial,sans-serif;background:#0B1624;color:#EEF6FC;margin:28px}}.card{{background:#122235;border:1px solid #253B52;border-radius:12px;padding:16px;margin:14px 0}}img{{max-width:100%;background:#fff;border-radius:8px}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #30495F;padding:6px;text-align:left}}</style></head><body>
<h1>VSLP Acoustic Quality Control</h1><div class='card'><b>Files evaluated:</b> {n}<br><b>Selected families:</b> {', '.join(cfg.selected_families or QC_FAMILIES.keys())}</div>
<div class='card'><h2>Family summary</h2><table><tr><th>Family</th><th>Status</th><th>Count</th></tr>{family_rows}</table></div>
<div class='card'><h2>QC family coverage</h2><img src='../plots/{p1.name}'></div><div class='card'><h2>QC overview</h2><img src='../plots/{p2.name}'></div>
</body></html>"""
    path.write_text(html, encoding="utf-8")
