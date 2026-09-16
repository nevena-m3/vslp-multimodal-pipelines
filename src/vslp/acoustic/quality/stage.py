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
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - optional at runtime
    PCA = None
    StandardScaler = None

from vslp.acoustic.context import cleanup_stage
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
    selected_features: list[str] | None = None
    minimum_internal_pause_sec: float = 0.15
    high_level_percentile: float = 90.0
    hard_clip_threshold: float = 0.995
    near_clip_threshold: float = 0.95
    zero_threshold: float = 1e-5
    abrupt_jump_db: float = 18.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def quality_feature_registry() -> pd.DataFrame:
    """Return a transparent registry of QC features, families, and parameter dependencies."""
    rows: list[dict[str, Any]] = []
    parameter_map = {
        "additive_interference": "minimum_internal_pause_sec",
        "gain_dynamics": "high_level_percentile",
        "reverberation_echo": "minimum_internal_pause_sec",
        "channel_device": "none",
        "nonlinear_distortion": "high_level_percentile; hard_clip_threshold; near_clip_threshold",
        "temporal_discontinuity": "zero_threshold; abrupt_jump_db",
    }
    for family, features in FAMILY_FEATURES.items():
        for feat in features:
            role = "support" if feat.endswith("_status") or feat.endswith("_flags") else "quality_feature"
            rows.append({
                "family": family,
                "family_label": QC_FAMILIES[family]["label"],
                "feature": feat,
                "role": role,
                "default_selected": bool(role == "quality_feature"),
                "parameter_dependencies": parameter_map.get(family, "none"),
                "meaning": _infer_feature_meaning(feat),
            })
    return pd.DataFrame(rows)


def _infer_feature_meaning(feature: str) -> str:
    f = feature.lower()
    if f.endswith("_status"):
        return "Computation status for this QC family."
    if f.endswith("_flags"):
        return "Semicolon-delimited support or warning flags for this QC family."
    if "pause" in f and "rms" in f:
        return "Pause-region level/noise estimate; higher values can indicate additive background contamination."
    if "speech_pause" in f:
        return "Contrast between speech and pause levels; low contrast can indicate noise or weak signal separation."
    if "hum50" in f or "hum60" in f:
        return "Relative power near 50/60 Hz during pauses; higher values suggest line-frequency interference."
    if "rms_db_std" in f or "instability" in f or "slope" in f:
        return "Amplitude variability/drift proxy; high values may reflect gain changes or unstable recording level."
    if "tail" in f or "reverb" in f or "blur" in f:
        return "Post-speech energy persistence proxy; high values may reflect reverberation or echo."
    if "centroid" in f or "rolloff" in f or "band" in f or "tilt" in f:
        return "Spectral/device response proxy; useful for detecting bandwidth or channel differences."
    if "clip" in f or "flat" in f or "pileup" in f or "asymmetry" in f:
        return "Nonlinear distortion or clipping proxy; higher values indicate potential overload."
    if "dropout" in f or "zero" in f or "jump" in f or "frozen" in f or "continuity" in f:
        return "Temporal discontinuity proxy; higher values suggest dropouts, glitches, or waveform breaks."
    return "Segmentation-informed acoustic quality feature."


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
    """Read canonical preprocessing audio without silently transforming it."""
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Canonical QC audio must be mono; got shape {x.shape}: {path}")
    if x.size == 0:
        raise ValueError(f"Canonical QC audio is empty: {path}")
    if not np.isfinite(x).all():
        raise ValueError(f"Canonical QC audio contains NaN/Inf samples: {path}")
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


def _compute_one(
    row: pd.Series,
    cfg: QualityControlConfig,
    families: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    file_name = str(row.get("file_name", ""))
    wav = Path(str(row.get("analysis_wav_path", "")))
    frames_path = Path(str(row.get("frame_csv_path", "")))
    segments_path = Path(str(row.get("segments_csv_path", "")))
    out = {
        "file_name": file_name,
        "analysis_wav_path": str(wav),
        "frame_csv_path": str(frames_path),
        "segments_csv_path": str(segments_path),
    }
    out.update(
        {
            key: row.get(key)
            for key in (
                "recording_id",
                "source_file_path",
                "source_sha256",
                "project_name",
                "task_name",
                "run_id",
                "run_created_at_local",
                "run_created_at_utc",
            )
        }
    )
    status_rows = []
    if not wav.exists():
        raise FileNotFoundError(f"Missing canonical analysis WAV: {wav}")
    if not frames_path.exists():
        raise FileNotFoundError(f"Missing frame CSV: {frames_path}")
    if not segments_path.exists():
        raise FileNotFoundError(f"Missing segments CSV: {segments_path}")

    x, sr = _read_audio(wav)
    frames = pd.read_csv(frames_path)
    segments = pd.read_csv(segments_path)
    out["sample_rate_hz"] = sr
    out["duration_sec"] = float(len(x) / sr) if sr else np.nan
    out["analysis_wav_sha256"] = sha256_file(wav)

    fam_funcs = {
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
                vals = fam_funcs[fam]()
                out.update(vals)
                status_col = next((k for k in vals if k.endswith("_status")), None)
                flag_col = next((k for k in vals if k.endswith("_flags")), None)
                status_rows.append(
                    {
                        "file_name": file_name,
                        "family": fam,
                        "family_label": QC_FAMILIES[fam]["label"],
                        "status": vals.get(status_col, "computed"),
                        "flags": vals.get(flag_col, ""),
                    }
                )
            except Exception as exc:
                status_rows.append(
                    {
                        "file_name": file_name,
                        "family": fam,
                        "family_label": QC_FAMILIES[fam]["label"],
                        "status": "failed",
                        "flags": str(exc),
                    }
                )
        else:
            for key in FAMILY_FEATURES[fam]:
                out[key] = (
                    np.nan
                    if not key.endswith("status") and not key.endswith("flags")
                    else ("not_selected" if key.endswith("status") else "")
                )
            status_rows.append(
                {
                    "file_name": file_name,
                    "family": fam,
                    "family_label": QC_FAMILIES[fam]["label"],
                    "status": "not_selected",
                    "flags": "",
                }
            )
    return out, status_rows


def _apply_feature_selection(feat_df: pd.DataFrame, cfg: QualityControlConfig) -> pd.DataFrame:
    """Mask unselected numeric QC features while preserving status/flags for audit."""
    selected = set(cfg.selected_features or [])
    if not selected:
        return feat_df
    registry = quality_feature_registry()
    quality_features = registry.loc[registry["role"].eq("quality_feature"), "feature"].tolist()
    for feat in quality_features:
        if feat in feat_df.columns and feat not in selected:
            feat_df[feat] = np.nan
    return feat_df


def _quality_warning_rows(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Generate conservative per-file QC warnings from interpretable proxy thresholds.

    Thresholds are intentionally screening defaults. They are not clinical rejection criteria.
    """
    rows: list[dict[str, Any]] = []
    rules = [
        ("additive_interference", "qadd_speech_pause_level_diff_db", "low", 10.0, "Low speech-pause level contrast; speech/pause separation may be noisy."),
        ("additive_interference", "qadd_pause_hum50_ratio", "high", 0.05, "Elevated 50 Hz pause-region energy; possible line interference."),
        ("additive_interference", "qadd_pause_hum60_ratio", "high", 0.05, "Elevated 60 Hz pause-region energy; possible line interference."),
        ("gain_dynamics", "qgain_speech_rms_db_std", "high", 6.0, "High speech-level variability; possible gain instability or variable distance."),
        ("reverberation_echo", "qrev_post_offset_tail_db_above_floor", "high", 6.0, "Post-offset speech tail above floor; possible reverberation/echo."),
        ("nonlinear_distortion", "qdist_near_clipped_sample_fraction", "high", 0.01, "Near-clipped samples present in high-level speech frames."),
        ("temporal_discontinuity", "qtemp_waveform_continuity_break_score", "high", 0.02, "Waveform continuity-break proxy elevated; possible dropout/glitch."),
    ]
    if feat_df.empty:
        return pd.DataFrame(columns=["file_name", "family", "feature", "value", "threshold", "direction", "warning", "severity"])
    for _, row in feat_df.iterrows():
        fn = row.get("file_name", "")
        for family, feat, direction, thr, msg in rules:
            if feat not in feat_df.columns:
                continue
            val = _safe_float(row.get(feat))
            if not np.isfinite(val):
                continue
            triggered = val > thr if direction == "high" else val < thr
            if not triggered:
                continue
            severity = "review"
            if direction == "high" and val > thr * 2:
                severity = "strong_review"
            if direction == "low" and val < max(0.0, thr / 2):
                severity = "strong_review"
            rows.append({
                "file_name": fn,
                "family": family,
                "feature": feat,
                "value": val,
                "threshold": thr,
                "direction": direction,
                "warning": msg,
                "severity": severity,
            })
    return pd.DataFrame(rows, columns=["file_name", "family", "feature", "value", "threshold", "direction", "warning", "severity"])


def _quality_recommendations(warnings_df: pd.DataFrame, n_files: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    family_guidance = {
        "additive_interference": "Inspect noise-sensitive spectral, phonatory, and low-energy timing features. Pause-derived measures may be contaminated.",
        "gain_dynamics": "Use caution with amplitude/intensity and energy-envelope features; consider reporting gain instability alongside biomarkers.",
        "reverberation_echo": "Use caution with pause duration, pause energy, timing boundaries, and rhythm/prosody features sensitive to speech tails.",
        "channel_device": "Use caution comparing spectral/formant/channel-sensitive features across devices or microphones.",
        "nonlinear_distortion": "Use caution with amplitude, CPP, spectral, and phonatory features; clipped files may need exclusion or sensitivity analysis.",
        "temporal_discontinuity": "Use caution with temporal/rhythm and signal-continuity-sensitive features; inspect files with dropouts/glitches.",
    }
    if n_files <= 0:
        return pd.DataFrame(columns=["family", "n_files_flagged", "fraction_files_flagged", "recommendation"])
    if warnings_df.empty:
        return pd.DataFrame([{
            "family": "overall",
            "n_files_flagged": 0,
            "fraction_files_flagged": 0.0,
            "recommendation": "No QC warning thresholds were triggered. Continue feature extraction and inspect acoustic QC plots.",
        }])
    for fam, grp in warnings_df.groupby("family"):
        n = int(grp["file_name"].nunique())
        rows.append({
            "family": fam,
            "n_files_flagged": n,
            "fraction_files_flagged": n / n_files,
            "recommendation": family_guidance.get(fam, "Review affected files before interpreting downstream acoustic features."),
        })
    return pd.DataFrame(rows)


@cleanup_stage
def run_acoustic_quality_control(segmentation_summary_csv: str | Path, output_root: str | Path, config: QualityControlConfig | None=None) -> StageResult:
    cfg=config or QualityControlConfig()
    families=cfg.selected_families or list(QC_FAMILIES.keys())
    families=[f for f in families if f in QC_FAMILIES]
    output_root=Path(output_root); segmentation_summary_csv=Path(segmentation_summary_csv)
    stage_dir=output_root/"acoustic"/"003_quality_control"
    folders=ensure_stage_folders(stage_dir, lazy=True)
    rows=[]; status_rows=[]; errors=[]
    seg=pd.read_csv(segmentation_summary_csv) if segmentation_summary_csv.exists() else pd.DataFrame()
    for _, row in seg.iterrows():
        if str(row.get("automatic_status", "ACCEPTED")).upper() in {"EXCLUDED", "FAILED"}:
            continue
        try:
            vals, st=_compute_one(row, cfg, families); rows.append(vals); status_rows.extend([{**item, **{key: row.get(key) for key in ("recording_id", "task_name", "run_id")}} for item in st])
        except Exception as exc:
            errors.append({"file_name":row.get("file_name", ""), "status":"failed", "error":str(exc)})
    features_csv=folders["tables"]/"acoustic_quality_features.csv"
    main_csv=folders["tables"]/"acoustic_quality_main_summary.csv"
    status_csv=folders["tables"]/"acoustic_quality_family_status.csv"
    family_csv=folders["tables"]/"acoustic_quality_family_summary.csv"
    registry_csv=folders["tables"]/"acoustic_quality_feature_registry.csv"
    warnings_csv=folders["tables"]/"acoustic_quality_warnings.csv"
    recommendations_csv=folders["tables"]/"acoustic_quality_recommendations.csv"
    distribution_csv=folders["tables"]/"acoustic_quality_distribution_summary.csv"
    processed_csv=folders["tables"]/"acoustic_quality_processed_features.csv"
    family_scores_csv=folders["tables"]/"acoustic_quality_family_scores.csv"
    feature_corr_csv=folders["tables"]/"acoustic_quality_feature_spearman_correlation.csv"
    family_corr_csv=folders["tables"]/"acoustic_quality_family_spearman_correlation.csv"
    review_rank_csv=folders["tables"]/"acoustic_quality_recording_review_rank.csv"
    pca_variance_csv=folders["tables"]/"acoustic_quality_pca_variance.csv"
    pca_scores_csv=folders["tables"]/"acoustic_quality_pca_scores.csv"
    errors_csv=folders["errors"]/"acoustic_quality_errors.csv"

    feat_df=pd.DataFrame(rows)
    for fam, keys in FAMILY_FEATURES.items():
        for k in keys:
            if k not in feat_df.columns: feat_df[k]=np.nan
    feat_df = _apply_feature_selection(feat_df, cfg)

    warnings_df = _quality_warning_rows(feat_df)
    recommendations_df = _quality_recommendations(warnings_df, n_files=len(feat_df))

    # attach compact review summary to per-file table
    if not feat_df.empty:
        warn_counts = warnings_df.groupby("file_name").size().to_dict() if not warnings_df.empty else {}
        warn_fams = warnings_df.groupby("file_name")["family"].apply(lambda s: ";".join(sorted(set(map(str, s))))).to_dict() if not warnings_df.empty else {}
        feat_df["quality_n_warnings"] = feat_df["file_name"].map(warn_counts).fillna(0).astype(int) if "file_name" in feat_df else 0
        feat_df["quality_warning_families"] = feat_df["file_name"].map(warn_fams).fillna("") if "file_name" in feat_df else ""
        feat_df["quality_review_level"] = np.where(feat_df["quality_n_warnings"] >= 2, "review", np.where(feat_df["quality_n_warnings"] == 1, "minor_review", "no_threshold_warning"))

    registry_df = quality_feature_registry()
    if cfg.selected_features:
        selected_set = set(cfg.selected_features)
        registry_df["selected_in_run"] = registry_df["feature"].isin(selected_set) | registry_df["role"].eq("support")
    else:
        registry_df["selected_in_run"] = registry_df["default_selected"] | registry_df["role"].eq("support")

    feat_df.to_csv(features_csv,index=False)
    registry_df.to_csv(registry_csv,index=False)
    warnings_df.to_csv(warnings_csv,index=False)
    recommendations_df.to_csv(recommendations_csv,index=False)
    status_df=pd.DataFrame(status_rows, columns=["file_name","recording_id","task_name","run_id","family","family_label","status","flags"])
    status_df.to_csv(status_csv,index=False)
    pd.DataFrame(errors).to_csv(errors_csv,index=False)
    fam_summary=status_df.groupby(["family","family_label","status"]).size().reset_index(name="count") if not status_df.empty else pd.DataFrame(columns=["family","family_label","status","count"])
    fam_summary.to_csv(family_csv,index=False)
    main_cols=["recording_id","file_name","source_file_path","source_sha256","project_name","task_name","run_id","run_created_at_local","run_created_at_utc","analysis_wav_path","analysis_wav_sha256","quality_review_level","quality_n_warnings","quality_warning_families","duration_sec","sample_rate_hz","qadd_pause_rms_db_median","qadd_speech_pause_level_diff_db","qgain_speech_rms_db_std","qrev_post_offset_tail_db_above_floor","qchan_speech_centroid_hz","qdist_near_clipped_sample_fraction","qtemp_waveform_continuity_break_score"]
    for c in main_cols:
        if c not in feat_df.columns: feat_df[c]=np.nan
    feat_df[main_cols].to_csv(main_csv,index=False)

    # Descriptive/statistical QC layer. These tables/plots are for data audit and
    # visualization only; no inferential group claims are made here.
    analysis = _build_quality_analysis_tables(feat_df, warnings_df)
    analysis["distribution_summary"].to_csv(distribution_csv, index=False)
    analysis["processed_features"].to_csv(processed_csv, index=False)
    analysis["family_scores"].to_csv(family_scores_csv, index=False)
    analysis["feature_corr"].to_csv(feature_corr_csv)
    analysis["family_corr"].to_csv(family_corr_csv)
    analysis["review_rank"].to_csv(review_rank_csv, index=False)
    analysis["pca_variance"].to_csv(pca_variance_csv, index=False)
    analysis["pca_scores"].to_csv(pca_scores_csv, index=False)

    # plots
    p1=folders["plots"]/"quality_family_status.png"
    p2=folders["plots"]/"quality_main_features_overview.png"
    p3=folders["plots"]/"quality_feature_distributions.png"
    p4=folders["plots"]/"quality_warning_heatmap.png"
    p5=folders["plots"]/"quality_recommendation_summary.png"
    p6=folders["plots"]/"quality_family_score_distributions.png"
    p7=folders["plots"]/"quality_recording_review_rank.png"
    p8=folders["plots"]/"quality_feature_correlation_heatmap.png"
    p9=folders["plots"]/"quality_family_correlation_heatmap.png"
    p10=folders["plots"]/"quality_missingness_feature_coverage.png"
    p11=folders["plots"]/"quality_pca_scree.png"
    p12=folders["plots"]/"quality_pca_embedding.png"
    _plot_family_status(status_df,p1); _plot_overview(feat_df,p2)
    _plot_feature_distributions(feat_df,p3); _plot_warning_heatmap(warnings_df, feat_df, p4); _plot_recommendations(recommendations_df,p5)
    _plot_family_score_distributions(analysis["family_scores"], p6)
    _plot_review_rank(analysis["review_rank"], p7)
    _plot_correlation_heatmap(analysis["feature_corr"], p8, title="QC feature Spearman correlation")
    _plot_correlation_heatmap(analysis["family_corr"], p9, title="QC family-score Spearman correlation")
    _plot_missingness_coverage(analysis["distribution_summary"], p10)
    _plot_pca_scree(analysis["pca_variance"], p11)
    _plot_pca_embedding(analysis["pca_scores"], p12)
    report=folders["reports"]/"acoustic_quality_control_report.html"
    _write_report(report, feat_df, fam_summary, cfg, p1, p2, p3, p4, p5, recommendations_df, extra_plots=[p6,p7,p8,p9,p10,p11,p12])
    manifest=StageManifest(
        stage_name="acoustic_quality_control",
        stage_version="0.23.0",
        status="completed_with_warnings" if errors else "completed",
        input_artifacts=[ArtifactRef(path=str(segmentation_summary_csv), role="segmentation_summary", media_type="text/csv")],
        output_artifacts=[
            ArtifactRef(path=str(features_csv), role="quality_features", media_type="text/csv"),
            ArtifactRef(path=str(main_csv), role="quality_main_summary", media_type="text/csv"),
            ArtifactRef(path=str(registry_csv), role="quality_feature_registry", media_type="text/csv"),
            ArtifactRef(path=str(warnings_csv), role="quality_warnings", media_type="text/csv"),
            ArtifactRef(path=str(recommendations_csv), role="quality_recommendations", media_type="text/csv"),
            ArtifactRef(path=str(distribution_csv), role="quality_distribution_summary", media_type="text/csv"),
            ArtifactRef(path=str(family_scores_csv), role="quality_family_scores", media_type="text/csv"),
            ArtifactRef(path=str(review_rank_csv), role="quality_review_rank", media_type="text/csv"),
            ArtifactRef(path=str(report), role="quality_report", media_type="text/html"),
        ],
        config=cfg.to_dict() | {"selected_families": families, "selected_features": cfg.selected_features},
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


def _plot_feature_distributions(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        ("qadd_speech_pause_level_diff_db", "Speech-pause\ncontrast (dB)", "low", 10.0),
        ("qgain_speech_rms_db_std", "Speech level\nSD (dB)", "high", 6.0),
        ("qrev_post_offset_tail_db_above_floor", "Reverb tail\nabove floor (dB)", "high", 6.0),
        ("qdist_near_clipped_sample_fraction", "Near-clipped\nsample fraction", "high", 0.01),
        ("qtemp_waveform_continuity_break_score", "Continuity\nbreak score", "high", 0.02),
    ]
    fig, axes = plt.subplots(1, len(cols), figsize=(15, 4.2))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for ax, (col, label, direction, thr) in zip(axes, cols, strict=False):
        vals = pd.to_numeric(df[col], errors="coerce").dropna() if col in df.columns else pd.Series(dtype=float)
        if vals.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue
        y = vals.values
        x = np.zeros_like(y, dtype=float)
        rng = np.random.default_rng(13)
        jitter = rng.normal(0, 0.035, size=len(y))
        bad = y > thr if direction == "high" else y < thr
        ax.scatter(x[~bad] + jitter[~bad], y[~bad], s=46, alpha=0.82, label="within screen")
        ax.scatter(x[bad] + jitter[bad], y[bad], s=58, alpha=0.90, label="review")
        ax.axhline(thr, linestyle="--", linewidth=1.2)
        ax.set_xticks([])
        ax.set_title(label, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("QC feature distributions across uploaded recordings", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_warning_heatmap(warnings_df: pd.DataFrame, feat_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    families = list(QC_FAMILIES.keys())
    files = feat_df["file_name"].astype(str).tolist() if "file_name" in feat_df.columns else []
    mat = np.zeros((len(files), len(families)))
    if not warnings_df.empty and files:
        f_index = {f: i for i, f in enumerate(files)}
        fam_index = {f: i for i, f in enumerate(families)}
        for _, r in warnings_df.iterrows():
            fi = f_index.get(str(r.get("file_name", "")))
            fj = fam_index.get(str(r.get("family", "")))
            if fi is not None and fj is not None:
                mat[fi, fj] += 1
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.35 * max(1, len(files)))))
    if not files:
        ax.text(0.5, 0.5, "No files evaluated", ha="center", va="center"); ax.axis("off")
    else:
        im = ax.imshow(mat, aspect="auto")
        ax.set_yticks(range(len(files))); ax.set_yticklabels(files, fontsize=7)
        ax.set_xticks(range(len(families))); ax.set_xticklabels([QC_FAMILIES[f]["label"] for f in families], rotation=35, ha="right", fontsize=8)
        ax.set_title("QC warning heatmap by file and artifact family")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] > 0:
                    ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.75, label="warning count")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_recommendations(rec_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if rec_df.empty or "family" not in rec_df.columns:
        ax.text(0.5, 0.5, "No QC recommendations", ha="center", va="center"); ax.axis("off")
    else:
        df = rec_df[rec_df["family"].ne("overall")].copy()
        if df.empty:
            ax.text(0.5, 0.5, "No warning thresholds triggered", ha="center", va="center"); ax.axis("off")
        else:
            df = df.sort_values("fraction_files_flagged")
            labels = [QC_FAMILIES.get(f, {}).get("label", f) for f in df["family"]]
            ax.barh(labels, df["fraction_files_flagged"].astype(float) * 100)
            ax.set_xlabel("Files flagged (%)")
            ax.set_title("QC families most likely to affect downstream acoustic features")
            ax.set_xlim(0, max(100, float(df["fraction_files_flagged"].max() * 100) * 1.15))
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _write_report(path: Path, feat_df: pd.DataFrame, fam_summary: pd.DataFrame, cfg: QualityControlConfig, p1: Path, p2: Path, p3: Path, p4: Path, p5: Path, recommendations_df: pd.DataFrame, extra_plots: list[Path] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n=len(feat_df)
    family_rows="".join(f"<tr><td>{r.family_label}</td><td>{r.status}</td><td>{r['count']}</td></tr>" for _,r in fam_summary.iterrows()) if not fam_summary.empty else "<tr><td colspan='3'>No family rows</td></tr>"
    rec_rows="".join(f"<tr><td>{r.family}</td><td>{r.n_files_flagged}</td><td>{r.fraction_files_flagged:.2f}</td><td>{r.recommendation}</td></tr>" for _,r in recommendations_df.iterrows()) if not recommendations_df.empty else "<tr><td colspan='4'>No recommendation rows</td></tr>"
    html=f"""<!doctype html><html><head><meta charset='utf-8'><title>VSLP Acoustic Quality Control</title>
<style>body{{font-family:Arial,sans-serif;background:#0B1624;color:#EEF6FC;margin:28px}}.card{{background:#122235;border:1px solid #253B52;border-radius:12px;padding:16px;margin:14px 0}}img{{max-width:100%;background:#fff;border-radius:8px}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #30495F;padding:6px;text-align:left}}</style></head><body>
<h1>VSLP Acoustic Quality Control</h1><div class='card'><b>Files evaluated:</b> {n}<br><b>Selected families:</b> {', '.join(cfg.selected_families or QC_FAMILIES.keys())}</div>
<div class='card'><h2>Family summary</h2><table><tr><th>Family</th><th>Status</th><th>Count</th></tr>{family_rows}</table></div>
<div class='card'><h2>Recommendations</h2><table><tr><th>Family</th><th>Files flagged</th><th>Fraction</th><th>Recommendation</th></tr>{rec_rows}</table></div>
<div class='card'><h2>QC feature distributions</h2><img src='../plots/{p3.name}'></div>
<div class='card'><h2>QC warning heatmap</h2><img src='../plots/{p4.name}'></div>
<div class='card'><h2>QC recommendation summary</h2><img src='../plots/{p5.name}'></div>
<div class='card'><h2>QC family coverage</h2><img src='../plots/{p1.name}'></div><div class='card'><h2>QC overview</h2><img src='../plots/{p2.name}'></div>
{''.join(f"<div class='card'><h2>{plot.stem.replace('_',' ').title()}</h2><img src='../plots/{plot.name}'></div>" for plot in (extra_plots or []))}
</body></html>"""
    path.write_text(html, encoding="utf-8")

# -----------------------------------------------------------------------------
# v0.23 statistical/visual QC audit helpers
# -----------------------------------------------------------------------------

def _selected_numeric_quality_features(df: pd.DataFrame) -> list[str]:
    """Return numeric QC feature columns, excluding identity/status/support fields."""
    cols: list[str] = []
    registry = quality_feature_registry()
    eligible = set(registry.loc[registry["role"].eq("quality_feature"), "feature"].astype(str))
    for c in df.columns:
        if c not in eligible:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 2 and s.nunique(dropna=True) >= 2:
            cols.append(c)
    return cols


def _feature_family(feature: str) -> str:
    for fam, feats in FAMILY_FEATURES.items():
        if feature in feats or feature.startswith({
            "additive_interference": "qadd_",
            "gain_dynamics": "qgain_",
            "reverberation_echo": "qrev_",
            "channel_device": "qchan_",
            "nonlinear_distortion": "qdist_",
            "temporal_discontinuity": "qtemp_",
        }.get(fam, "__")):
            return fam
    return "other"


def _choose_transform_for_qc(s: pd.Series) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 3:
        return "none"
    skew = float(x.skew()) if np.isfinite(x.skew()) else 0.0
    if (x >= 0).all() and skew > 1.0:
        return "log1p"
    if abs(skew) > 1.0:
        return "asinh"
    return "none"


def _apply_qc_transform(s: pd.Series, method: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    if method == "log1p":
        return np.log1p(np.clip(x, a_min=0, a_max=None))
    if method == "asinh":
        return np.arcsinh(x)
    return x


def _robust_scale_qc(s: pd.Series) -> tuple[pd.Series, float, float, str]:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    med = float(np.nanmedian(x)) if x.notna().any() else np.nan
    q25 = float(np.nanpercentile(x, 25)) if x.notna().any() else np.nan
    q75 = float(np.nanpercentile(x, 75)) if x.notna().any() else np.nan
    iqr = q75 - q25 if np.isfinite(q75) and np.isfinite(q25) else np.nan
    if not np.isfinite(iqr) or iqr <= 1e-12:
        return x * np.nan, med, iqr, "not_scaled_low_variation"
    return (x - med) / iqr, med, iqr, "robust_scaled"


def _build_quality_analysis_tables(feat_df: pd.DataFrame, warnings_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build descriptive QC analysis tables.

    This mirrors the uploaded notebooks conceptually: transform skewed variables,
    winsorize extremes, robust-scale, aggregate family scores, then inspect
    correlation/PCA structure. It deliberately avoids inferential hypothesis tests
    in this acoustic-only pipeline.
    """
    numeric_features = _selected_numeric_quality_features(feat_df)
    dist_rows: list[dict[str, Any]] = []
    processed = feat_df[[c for c in ["recording_id", "file_name", "task_name", "run_id", "quality_review_level", "quality_n_warnings"] if c in feat_df.columns]].copy() if not feat_df.empty else pd.DataFrame()

    processed_blocks: list[pd.DataFrame] = []
    for feature in numeric_features:
        raw = pd.to_numeric(feat_df[feature], errors="coerce").astype(float)
        method = _choose_transform_for_qc(raw)
        transformed = _apply_qc_transform(raw, method)
        if transformed.notna().any():
            lo, hi = np.nanpercentile(transformed, [1, 99])
            winsorized = transformed.clip(lower=lo, upper=hi)
        else:
            lo = hi = np.nan
            winsorized = transformed
        proc, med, iqr, scale_status = _robust_scale_qc(winsorized)
        processed_blocks.append(pd.DataFrame({
            f"{feature}_tr": transformed,
            f"{feature}_w": winsorized,
            f"{feature}_proc": proc,
        }))
        x = raw.dropna()
        dist_rows.append({
            "feature": feature,
            "family": _feature_family(feature),
            "family_label": QC_FAMILIES.get(_feature_family(feature), {}).get("label", _feature_family(feature)),
            "n_nonmissing": int(x.size),
            "missing_fraction": float(raw.isna().mean()) if len(raw) else np.nan,
            "n_unique": int(x.nunique(dropna=True)),
            "min": float(x.min()) if x.size else np.nan,
            "q25": float(x.quantile(0.25)) if x.size else np.nan,
            "median": float(x.median()) if x.size else np.nan,
            "q75": float(x.quantile(0.75)) if x.size else np.nan,
            "max": float(x.max()) if x.size else np.nan,
            "mad": float(np.median(np.abs(x - x.median()))) if x.size else np.nan,
            "skew": float(x.skew()) if x.size >= 3 else np.nan,
            "transform": method,
            "winsor_lower_q01": float(lo) if np.isfinite(lo) else np.nan,
            "winsor_upper_q99": float(hi) if np.isfinite(hi) else np.nan,
            "robust_center_median": med,
            "robust_iqr": iqr,
            "scaling_status": scale_status,
        })

    if processed_blocks:
        processed = pd.concat([processed.reset_index(drop=True)] + processed_blocks, axis=1)
    elif processed.empty:
        processed = pd.DataFrame(columns=["file_name"])

    distribution = pd.DataFrame(dist_rows)

    family_scores = feat_df[[c for c in ["recording_id", "file_name", "task_name", "run_id", "quality_review_level", "quality_n_warnings", "quality_warning_families"] if c in feat_df.columns]].copy() if not feat_df.empty else pd.DataFrame(columns=["file_name"])
    for fam in QC_FAMILIES:
        feats = [f for f in numeric_features if _feature_family(f) == fam and f"{f}_proc" in processed.columns]
        proc_cols = [f"{f}_proc" for f in feats]
        n_col = f"qfamily_{fam}_n_features_available"
        score_col = f"qfamily_{fam}_score"
        if proc_cols:
            family_scores[n_col] = processed[proc_cols].notna().sum(axis=1)
            family_scores[score_col] = processed[proc_cols].median(axis=1, skipna=True)
        else:
            family_scores[n_col] = 0
            family_scores[score_col] = np.nan

    proc_cols = [c for c in processed.columns if c.endswith("_proc")]
    if len(proc_cols) >= 2:
        feature_corr = processed[proc_cols].corr(method="spearman", min_periods=3)
        feature_corr.index = [c.replace("_proc", "") for c in feature_corr.index]
        feature_corr.columns = [c.replace("_proc", "") for c in feature_corr.columns]
    else:
        feature_corr = pd.DataFrame()

    family_score_cols = [c for c in family_scores.columns if c.endswith("_score")]
    if len(family_score_cols) >= 2:
        family_corr = family_scores[family_score_cols].corr(method="spearman", min_periods=3)
        label_map = {f"qfamily_{fam}_score": QC_FAMILIES[fam]["label"] for fam in QC_FAMILIES}
        family_corr = family_corr.rename(index=label_map, columns=label_map)
    else:
        family_corr = pd.DataFrame()

    if not family_scores.empty:
        burden_cols = [c for c in family_scores.columns if c.endswith("_score")]
        burden = family_scores[burden_cols].abs().median(axis=1, skipna=True) if burden_cols else pd.Series(np.nan, index=family_scores.index)
        review_rank = family_scores[[c for c in ["recording_id", "file_name", "task_name", "run_id", "quality_review_level", "quality_n_warnings", "quality_warning_families"] if c in family_scores.columns]].copy()
        review_rank["quality_burden_score"] = burden
        review_rank = review_rank.sort_values(["quality_n_warnings", "quality_burden_score"], ascending=[False, False], na_position="last").reset_index(drop=True)
        review_rank.insert(0, "review_rank", np.arange(1, len(review_rank) + 1))
    else:
        review_rank = pd.DataFrame(columns=["review_rank", "file_name", "quality_burden_score"])

    pca_variance = pd.DataFrame(columns=["component", "variance_explained", "cumulative_variance_explained"])
    pca_scores = pd.DataFrame(columns=["file_name", "PC1", "PC2"])
    if PCA is not None and StandardScaler is not None and len(proc_cols) >= 2 and len(processed) >= 3:
        X = processed[proc_cols].copy()
        valid_cols = [c for c in X.columns if X[c].notna().sum() >= 3 and X[c].nunique(dropna=True) >= 2]
        if len(valid_cols) >= 2:
            X = X[valid_cols]
            X = X.fillna(X.median(numeric_only=True))
            if len(X) >= 3:
                Xz = StandardScaler().fit_transform(X)
                n_components = min(5, Xz.shape[0], Xz.shape[1])
                if n_components >= 2:
                    pca = PCA(n_components=n_components, random_state=13)
                    scores = pca.fit_transform(Xz)
                    pca_variance = pd.DataFrame({
                        "component": [f"PC{i+1}" for i in range(n_components)],
                        "variance_explained": pca.explained_variance_ratio_,
                        "cumulative_variance_explained": np.cumsum(pca.explained_variance_ratio_),
                    })
                    pca_scores = processed[[c for c in ["recording_id", "file_name", "task_name", "run_id", "quality_review_level", "quality_n_warnings"] if c in processed.columns]].copy()
                    for i in range(n_components):
                        pca_scores[f"PC{i+1}"] = scores[:, i]

    return {
        "distribution_summary": distribution,
        "processed_features": processed,
        "family_scores": family_scores,
        "feature_corr": feature_corr,
        "family_corr": family_corr,
        "review_rank": review_rank,
        "pca_variance": pca_variance,
        "pca_scores": pca_scores,
    }


def _plot_family_score_distributions(family_scores: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    score_cols = [f"qfamily_{fam}_score" for fam in QC_FAMILIES if f"qfamily_{fam}_score" in family_scores.columns]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    vals = [pd.to_numeric(family_scores[c], errors="coerce").dropna().values for c in score_cols]
    labels = [QC_FAMILIES[c.replace("qfamily_", "").replace("_score", "")]["label"] for c in score_cols]
    vals_nonempty = [(values, label) for values, label in zip(vals, labels, strict=False) if len(values)]
    if not vals_nonempty:
        ax.text(0.5, 0.5, "No family scores available", ha="center", va="center"); ax.axis("off")
    else:
        vals, labels = zip(*vals_nonempty)
        parts = ax.violinplot(vals, showmeans=False, showmedians=True, widths=0.75)
        for body in parts["bodies"]:
            body.set_alpha(0.55)
        rng = np.random.default_rng(17)
        for i, v in enumerate(vals, start=1):
            jitter = rng.normal(0, 0.045, len(v))
            ax.scatter(np.full(len(v), i) + jitter, v, s=28, alpha=0.70)
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Robust family score (median of processed QC features)")
        ax.set_title("QC artifact-family burden distributions")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _plot_review_rank(review_rank: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, max(4.2, 0.35 * max(1, min(25, len(review_rank))))))
    if review_rank.empty or "quality_burden_score" not in review_rank.columns:
        ax.text(0.5, 0.5, "No recording review ranking available", ha="center", va="center"); ax.axis("off")
    else:
        df = review_rank.head(25).copy().iloc[::-1]
        labels = df["file_name"].astype(str).str.slice(0, 52) if "file_name" in df else df.index.astype(str)
        vals = pd.to_numeric(df["quality_burden_score"], errors="coerce").fillna(0)
        ax.barh(labels, vals)
        ax.set_xlabel("QC burden score (median |family score|)")
        ax.set_title("Recordings ranked for QC review")
        for i, (_, r) in enumerate(df.iterrows()):
            nw = int(r.get("quality_n_warnings", 0)) if pd.notna(r.get("quality_n_warnings", np.nan)) else 0
            ax.text(vals.iloc[i], i, f"  warnings={nw}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _plot_correlation_heatmap(corr: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = corr.shape[0] if not corr.empty else 0
    fig, ax = plt.subplots(figsize=(max(6.5, min(15, 0.35 * n + 4)), max(5.5, min(14, 0.35 * n + 4))))
    if corr.empty or n < 2:
        ax.text(0.5, 0.5, "Not enough variable QC features for correlation", ha="center", va="center"); ax.axis("off")
    else:
        plot_corr = corr.astype(float).copy()
        # Group features by family/prefix to keep the matrix interpretable without fragile clustering.
        ordered = sorted(plot_corr.index, key=lambda x: (_feature_family(str(x)), str(x)))
        plot_corr = plot_corr.loc[ordered, ordered]
        im = ax.imshow(plot_corr.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="equal")
        labels = [str(x).replace("q", "q\n", 1) if len(str(x)) > 18 and n <= 20 else str(x) for x in plot_corr.index]
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=6 if n > 18 else 8)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=6 if n > 18 else 8)
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("Spearman ρ")
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _plot_missingness_coverage(distribution: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, max(5.0, 0.22 * max(1, len(distribution)))))
    if distribution.empty or "missing_fraction" not in distribution.columns:
        ax.text(0.5, 0.5, "No feature coverage information", ha="center", va="center"); ax.axis("off")
    else:
        df = distribution.sort_values("missing_fraction", ascending=True).copy()
        labels = df["feature"].astype(str).str.replace("_", " ").str.slice(0, 55)
        ax.barh(labels, (1 - pd.to_numeric(df["missing_fraction"], errors="coerce").fillna(1)) * 100)
        ax.set_xlabel("Available recordings (%)")
        ax.set_title("QC feature coverage across uploaded recordings")
        ax.set_xlim(0, 100)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _plot_pca_scree(pca_variance: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if pca_variance.empty:
        ax.text(0.5, 0.5, "PCA not available: insufficient recordings/features", ha="center", va="center"); ax.axis("off")
    else:
        comps = pca_variance["component"].astype(str)
        var = pd.to_numeric(pca_variance["variance_explained"], errors="coerce") * 100
        cum = pd.to_numeric(pca_variance["cumulative_variance_explained"], errors="coerce") * 100
        ax.bar(comps, var, alpha=0.75, label="Component")
        ax.plot(comps, cum, marker="o", linewidth=2, label="Cumulative")
        ax.set_ylabel("Variance explained (%)")
        ax.set_title("PCA QC-structure scree plot")
        ax.legend(frameon=False)
        ax.set_ylim(0, max(100, float(cum.max()) * 1.05 if len(cum) else 100))
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _plot_pca_embedding(pca_scores: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    if pca_scores.empty or not {"PC1", "PC2"}.issubset(pca_scores.columns):
        ax.text(0.5, 0.5, "PCA embedding not available", ha="center", va="center"); ax.axis("off")
    else:
        x = pd.to_numeric(pca_scores["PC1"], errors="coerce")
        y = pd.to_numeric(pca_scores["PC2"], errors="coerce")
        warnings = pd.to_numeric(pca_scores.get("quality_n_warnings", pd.Series(0, index=pca_scores.index)), errors="coerce").fillna(0)
        sc = ax.scatter(x, y, s=55 + 18 * warnings, c=warnings, cmap="viridis", alpha=0.82)
        ax.axhline(0, color="0.7", linewidth=0.7); ax.axvline(0, color="0.7", linewidth=0.7)
        ax.set_xlabel("PC1") ; ax.set_ylabel("PC2")
        ax.set_title("PCA embedding of QC feature space")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.80); cbar.set_label("QC warning count")
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)
