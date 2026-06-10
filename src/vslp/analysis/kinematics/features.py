"""Kinematic feature computation for normalized MediaPipe landmark trajectories.

This module is the GUI-facing Block 06 feature layer. It consumes the normalized
landmark tables produced by Block 04 and emits:

* one frame-level kinematic time-series CSV per video; and
* one scalar feature table with one row per video.

The implementation intentionally ports the structure of the uploaded laboratory
feature script: interpolate gaps, remove distributional outliers, smooth
trajectories, compute geometry/kinematic signals, segment open/close movements,
and aggregate robust summary statistics. It does not mutate the raw MediaPipe
landmark outputs or the normalized landmark layer.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from .schemas import LANDMARK_PRESETS




@dataclass(frozen=True)
class KinematicFeatureSpec:
    """Declarative feature registry entry used by the GUI feature selector.

    This registry is intentionally separate from the numerical implementation
    below. It documents what each feature means, which landmarks it needs, how it
    should be normalized, and which QC checks must be satisfied before the
    feature is trusted.
    """

    feature_id: str
    group: str
    label: str
    status: str
    tier: str
    native_signal: str
    unit: str
    landmarks: tuple[int, ...]
    normalization: str
    aggregation: str
    interpretation: str
    source_function: str


ICD_NORMALIZATION_DESCRIPTION = "intercanthal distance; landmarks 243 and 463"
ROBUST_AGGREGATION_DESCRIPTION = "median, IQR, 5th percentile, 95th percentile, 5-95 spread"
MEDIAN_IQR_AGGREGATION_DESCRIPTION = "median and IQR"


KINEMATIC_FEATURE_SPECS: tuple[KinematicFeatureSpec, ...] = (
    KinematicFeatureSpec(
        "path_vert_med", "Vertical lip/jaw displacement", "Vertical path length", "implemented-formula", "B",
        "lower lip vertical aperture trajectory", "ICD-normalized distance", (17, 8, 243, 463), ICD_NORMALIZATION_DESCRIPTION, MEDIAN_IQR_AGGREGATION_DESCRIPTION,
        "Total absolute vertical movement over the analysis window or repetition.", "vertical aperture trajectory plus path length summary",
    ),
    KinematicFeatureSpec(
        "rom_vert_med", "Vertical lip/jaw displacement", "Vertical range of motion", "implemented-formula", "B",
        "lower lip vertical aperture trajectory", "ICD-normalized distance", (17, 8, 243, 463), ICD_NORMALIZATION_DESCRIPTION, MEDIAN_IQR_AGGREGATION_DESCRIPTION,
        "95th minus 5th percentile vertical opening range.", "vertical aperture trajectory plus robust range summary",
    ),
    KinematicFeatureSpec(
        "sLL_vert", "Vertical lip/jaw displacement", "Vertical speed", "implemented-formula", "B",
        "gradient of vertical aperture trajectory", "ICD-normalized distance/s", (17, 8, 243, 463), ICD_NORMALIZATION_DESCRIPTION, ROBUST_AGGREGATION_DESCRIPTION,
        "Magnitude of lower-lip/jaw vertical movement speed.", "vertical speed summary",
    ),
    KinematicFeatureSpec(
        "aLL_vert", "Vertical lip/jaw displacement", "Vertical acceleration", "implemented-formula", "C",
        "gradient of vertical speed", "ICD-normalized distance/s2", (17, 8, 243, 463), ICD_NORMALIZATION_DESCRIPTION, ROBUST_AGGREGATION_DESCRIPTION,
        "Magnitude of vertical movement acceleration; sensitive to smoothing and timestamps.", "vertical acceleration summary",
    ),
    KinematicFeatureSpec(
        "path_horz_med", "Horizontal lip spread", "Horizontal path length", "implemented-formula", "B",
        "left-right commissure spread trajectory", "ICD-normalized distance", (61, 291, 243, 463), ICD_NORMALIZATION_DESCRIPTION, MEDIAN_IQR_AGGREGATION_DESCRIPTION,
        "Total absolute horizontal lip-spread movement over the window or repetition.", "horizontal spread trajectory plus path length summary",
    ),
    KinematicFeatureSpec(
        "rom_horz_med", "Horizontal lip spread", "Horizontal range of motion", "implemented-formula", "B",
        "left-right commissure spread trajectory", "ICD-normalized distance", (61, 291, 243, 463), ICD_NORMALIZATION_DESCRIPTION, MEDIAN_IQR_AGGREGATION_DESCRIPTION,
        "95th minus 5th percentile horizontal lip-spread range.", "horizontal spread trajectory plus robust range summary",
    ),
    KinematicFeatureSpec(
        "sLL_horz", "Horizontal lip spread", "Horizontal speed", "implemented-formula", "B",
        "gradient of horizontal spread trajectory", "ICD-normalized distance/s", (61, 291, 243, 463), ICD_NORMALIZATION_DESCRIPTION, ROBUST_AGGREGATION_DESCRIPTION,
        "Magnitude of horizontal lip-spread speed.", "horizontal speed summary",
    ),
    KinematicFeatureSpec(
        "aLL_horz", "Horizontal lip spread", "Horizontal acceleration", "implemented-formula", "C",
        "gradient of horizontal speed", "ICD-normalized distance/s2", (61, 291, 243, 463), ICD_NORMALIZATION_DESCRIPTION, ROBUST_AGGREGATION_DESCRIPTION,
        "Magnitude of horizontal spread acceleration; sensitive to smoothing and timestamps.", "horizontal acceleration summary",
    ),
    KinematicFeatureSpec(
        "lip_aspect", "Lip aperture geometry", "Lip aspect ratio", "implemented-formula", "B",
        "vertical lip distance divided by horizontal commissure distance", "ratio", (0, 17, 61, 291, 243, 463), ICD_NORMALIZATION_DESCRIPTION, ROBUST_AGGREGATION_DESCRIPTION,
        "Relative mouth opening shape: vertical aperture scaled by horizontal lip spread.", "lip aspect ratio summary",
    ),
    KinematicFeatureSpec(
        "jaw_lateralization", "Jaw lateralization", "Jaw lateralization ratio", "implemented-formula", "B",
        "distance lower lip/jaw point to left canthus divided by distance to right canthus", "ratio", (17, 243, 463), ICD_NORMALIZATION_DESCRIPTION, ROBUST_AGGREGATION_DESCRIPTION,
        "Asymmetry/lateral deviation of the lower jaw or lower lip relative to eye anchors.", "jaw lateralization summary",
    ),
    KinematicFeatureSpec(
        "lip_symmetry", "Lip symmetry", "Left-right lip motion symmetry", "implemented-formula", "B",
        "left commissure-to-reference distance divided by right commissure-to-reference distance", "ratio", (61, 291, 8, 243, 463), ICD_NORMALIZATION_DESCRIPTION, ROBUST_AGGREGATION_DESCRIPTION,
        "Symmetry of left and right oral commissure movement relative to a midline reference.", "lip symmetry summary",
    ),
    KinematicFeatureSpec(
        "lat_xcorr", "Bilateral coordination", "Left-right oral commissure cross-correlation", "implemented-formula", "C",
        "normalized cross-correlation of left and right commissure distance trajectories", "correlation", (61, 291, 8, 243, 463), ICD_NORMALIZATION_DESCRIPTION, "maximum normalized cross-correlation",
        "Coordination between left and right oral commissure trajectories.", "lateral cross-correlation summary",
    ),
)


KINEMATIC_FEATURE_GROUPS: dict[str, tuple[KinematicFeatureSpec, ...]] = {}
for _spec in KINEMATIC_FEATURE_SPECS:
    KINEMATIC_FEATURE_GROUPS[_spec.group] = (*KINEMATIC_FEATURE_GROUPS.get(_spec.group, tuple()), _spec)


DEFAULT_KINEMATIC_FEATURE_IDS: tuple[str, ...] = tuple(
    spec.feature_id for spec in KINEMATIC_FEATURE_SPECS if spec.tier in {"B", "C"}
)


QC_FEATURE_REQUIREMENTS: tuple[dict[str, str], ...] = (
    {
        "Parameter": "Face-detected fraction",
        "Recommended threshold": ">= 80% detected frames",
        "Reason": "Low detected-frame coverage makes all landmark trajectories unstable and increases interpolation burden.",
    },
    {
        "Parameter": "ICD availability",
        "Recommended threshold": "landmarks 243 and 463 valid for most detected frames",
        "Reason": "The current formulas normalize distances by intercanthal distance; unstable anchors corrupt every ICD-normalized feature.",
    },
    {
        "Parameter": "Long no-face gaps",
        "Recommended threshold": "flag contiguous gaps > 0.5 s or task-specific movement gap",
        "Reason": "Long gaps can erase a movement cycle or force invalid interpolation.",
    },
    {
        "Parameter": "Coordinate jumps / jitter",
        "Recommended threshold": "flag abrupt frame-to-frame jumps after scaling",
        "Reason": "Jitter inflates speed, acceleration, path length and cross-correlation features.",
    },
    {
        "Parameter": "Timestamp/frame-rate stability",
        "Recommended threshold": "no duplicate timestamps; stable frame interval",
        "Reason": "Velocity and acceleration use gradients over time and are invalid if timestamps are irregular.",
    },
)



# GUI-facing feature framework derived from the uploaded feature/field maps.
# These rows are deliberately family-level: they organize what should be built
# without pretending that every field-map scalar has already been validated in
# the current computational kernel.
KINEMATIC_FEATURE_FRAMEWORK: tuple[dict[str, str], ...] = (
    {
        "Feature family": "Vertical lip/jaw opening",
        "Primary constructs": "aperture, range of motion, cumulative path, speed, acceleration",
        "Current implementation": "implemented kernel: mouth_aperture + velocity/path/range summaries",
        "Best tasks": "open-close, DDK, Buy Bobby a Puppy, sentence/passage",
        "ALS/PD relevance": "ALS bulbar slowing and reduced/compensatory range; PD speech/facial bradykinesia exploratory",
        "Evidence status": "literature-informed research feature family",
    },
    {
        "Feature family": "Horizontal lip spreading/retraction",
        "Primary constructs": "outer/inner lip spread, horizontal ROM, speed, acceleration",
        "Current implementation": "implemented kernel: outer_lip_spread, inner_lip_spread + summary derivatives",
        "Best tasks": "smile/spread, /i/-loaded sentence, connected speech",
        "ALS/PD relevance": "ALS lower-face weakness and PD hypomimia/masked facial movement",
        "Evidence status": "literature-informed; task-specific validation needed",
    },
    {
        "Feature family": "Mouth shape / aspect",
        "Primary constructs": "aperture-to-spread ratio and robust shape summaries",
        "Current implementation": "implemented kernel: lip_aspect_ratio",
        "Best tasks": "open-close, speech tasks with alternating vowels/consonants",
        "ALS/PD relevance": "configuration change, reduced oral shaping, compensatory jaw/lip strategy",
        "Evidence status": "translational feature derived from geometric oral-motor constructs",
    },
    {
        "Feature family": "Jaw/lower-face displacement",
        "Primary constructs": "chin/nose distance, lower-lip-to-chin distance, lower-face motion",
        "Current implementation": "implemented kernel: jaw_to_nose, lower_lip_to_chin",
        "Best tasks": "max open, speech with large jaw excursion",
        "ALS/PD relevance": "jaw compensation, slowing, reduced or excessive excursion",
        "Evidence status": "literature-informed, but MediaPipe chin points are anatomical proxies",
    },
    {
        "Feature family": "Symmetry and lateralization",
        "Primary constructs": "left-right corner asymmetry, jaw/lip lateral balance",
        "Current implementation": "implemented kernel: corner_vertical_asymmetry, corner_lateral_asymmetry",
        "Best tasks": "smile/spread, pucker, speech, non-speech facial tasks",
        "ALS/PD relevance": "unilateral lower-face weakness, asymmetric recruitment, exploratory PD asymmetry",
        "Evidence status": "moderate; requires visual QC and task-specific interpretation",
    },
    {
        "Feature family": "Coordination and timing",
        "Primary constructs": "bilateral correlation, movement segmentation, repetition consistency",
        "Current implementation": "partial: movement windows and repetition range summaries; full correlation library to be built",
        "Best tasks": "DDK/AMR/SMR, repeated open-close, connected speech",
        "ALS/PD relevance": "incoordination, irregularity, bradykinesia, reduced movement synchrony",
        "Evidence status": "planned expansion; needs dataset calibration",
    },
)


KINEMATIC_FEATURE_IMPLEMENTATION_AUDIT: tuple[dict[str, str], ...] = (
    {
        "Audit item": "Uploaded feature maps",
        "Decision": "Use as feature-family roadmap, not as a blindly copied code contract",
        "Rationale": "The maps organize ALS/video kinematic constructs and a 65-feature script lineage, but the GUI must expose what is currently implemented versus planned.",
    },
    {
        "Audit item": "Landmark convention",
        "Decision": "Keep current GUI normalization convention explicit",
        "Rationale": "The uploaded feature map uses an ICD convention based on landmarks 243/463; the current GUI normalization uses 133/362 with 33/263 fallback. Exact legacy compatibility should be a later locked decision, not hidden.",
    },
    {
        "Audit item": "Disease specificity",
        "Decision": "Do not label any feature as diagnostic",
        "Rationale": "ALS/PD relevance comes from feature families and task design; cutoffs require labeled validation data.",
    },
    {
        "Audit item": "Current computation",
        "Decision": "Compute the implemented oral-motor kernel after normalization and QC",
        "Rationale": "The current backend emits frame-level signals and robust per-video summaries from normalized trajectories; field-map expansion should be implemented incrementally with tests.",
    },
)


def feature_framework_dataframe() -> pd.DataFrame:
    """Return the family-level kinematic feature framework table for the GUI."""
    return pd.DataFrame(KINEMATIC_FEATURE_FRAMEWORK)


def feature_implementation_audit_dataframe() -> pd.DataFrame:
    """Return an audit table separating implemented features from roadmap items."""
    return pd.DataFrame(KINEMATIC_FEATURE_IMPLEMENTATION_AUDIT)


def write_feature_framework_catalog(output_root: Path | str) -> dict[str, Path]:
    """Write feature framework and implementation-audit tables for provenance.

    This is not a replacement for computed feature outputs. It documents which
    feature families are currently implemented, partially implemented, or planned
    so the GUI stays scientifically honest while the feature library expands.
    """
    root = Path(output_root).expanduser().resolve()
    tables = root / "kinematics" / "006_features" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    framework_csv = tables / "kinematic_feature_framework.csv"
    audit_csv = tables / "kinematic_feature_implementation_audit.csv"
    framework_json = tables / "kinematic_feature_framework.json"
    feature_framework_dataframe().to_csv(framework_csv, index=False)
    feature_implementation_audit_dataframe().to_csv(audit_csv, index=False)
    payload = {
        "status": "FEATURE_FRAMEWORK_ROADMAP_WITH_IMPLEMENTED_KERNEL",
        "purpose": "Document kinematic feature families, current implementation coverage, and scientific cautions.",
        "implemented_now": [
            "mouth aperture and robust summaries",
            "lip spread and robust summaries",
            "mouth aspect ratio",
            "jaw/lower-face proxy distances",
            "mouth-corner asymmetry",
            "movement-window summaries",
        ],
        "planned_expansion": [
            "full 65-feature field-map parity",
            "task-specific feature presets",
            "complete coordination/correlation library",
            "jerk, stiffness, duration and repetition timing features",
            "dataset-calibrated validity and interpretability ranges",
        ],
        "framework_rows": list(KINEMATIC_FEATURE_FRAMEWORK),
        "implementation_audit": list(KINEMATIC_FEATURE_IMPLEMENTATION_AUDIT),
        "warning": "This catalog documents feature families and implementation status; it is not an ALS/PD diagnostic model.",
    }
    framework_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"framework_csv": framework_csv, "audit_csv": audit_csv, "framework_json": framework_json}

def feature_registry_dataframe():
    """Return the kinematic feature registry as a pandas DataFrame."""
    rows = []
    for spec in KINEMATIC_FEATURE_SPECS:
        row = asdict(spec)
        row["landmarks"] = ", ".join(map(str, spec.landmarks))
        rows.append(row)
    return pd.DataFrame(rows)


def selected_specs(feature_ids: Iterable[str]) -> tuple[KinematicFeatureSpec, ...]:
    """Return registry specs matching a sequence of feature ids."""
    wanted = set(feature_ids)
    return tuple(spec for spec in KINEMATIC_FEATURE_SPECS if spec.feature_id in wanted)


@dataclass(frozen=True)
class FeatureComputationConfig:
    selected_landmarks: tuple[int, ...] = LANDMARK_PRESETS["ALS oral-motor core 15"]
    selected_preset: str = "ALS oral-motor core 15"
    smoothing_cutoff_hz: float = 6.0
    smoothing_order: int = 4
    outlier_sigma_extreme: float = 5.0
    outlier_sigma_tight: float = 3.0
    onset_frac: float = 0.10
    offset_frac: float = 0.90
    min_movement_len_frames: int = 3
    overwrite: bool = True
    use_smoothed_signals: bool = True


_FEATURE_DEFINITIONS: dict[str, dict] = {
    "mouth_aperture": {
        "kind": "distance",
        "points": (13, 14),
        "family": "oral_aperture",
        "description": "Upper-lower lip opening distance.",
    },
    "outer_lip_spread": {
        "kind": "distance",
        "points": (61, 291),
        "family": "lip_spread",
        "description": "Left-right outer commissure spread.",
    },
    "inner_lip_spread": {
        "kind": "distance",
        "points": (78, 308),
        "family": "lip_spread",
        "description": "Left-right inner lip spread.",
    },
    "jaw_to_nose": {
        "kind": "distance",
        "points": (152, 1),
        "family": "jaw_displacement",
        "description": "Chin-to-midface/nose reference distance.",
    },
    "lower_lip_to_chin": {
        "kind": "distance",
        "points": (14, 152),
        "family": "lower_face_geometry",
        "description": "Lower lip-to-chin distance.",
    },
    "lip_aspect_ratio": {
        "kind": "ratio",
        "numerator": "mouth_aperture",
        "denominator": "outer_lip_spread",
        "family": "oral_aperture",
        "description": "Mouth aperture divided by outer lip spread.",
    },
    "corner_vertical_asymmetry": {
        "kind": "abs_delta_axis",
        "points": (61, 291),
        "axis": "y_norm",
        "family": "lip_symmetry",
        "description": "Absolute vertical mismatch between left and right mouth corners.",
    },
    "corner_lateral_asymmetry": {
        "kind": "bilateral_abs_balance",
        "points": (61, 291),
        "axis": "x_norm",
        "family": "lip_symmetry",
        "description": "Absolute left-right lateral imbalance of mouth corners after centering.",
    },
}


def _features_dir(output_root: Path | str) -> Path:
    out = Path(output_root).expanduser().resolve() / "kinematics" / "006_features"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "timeseries").mkdir(parents=True, exist_ok=True)
    return out


def _normalization_manifest(output_root: Path | str) -> Path:
    return Path(output_root).expanduser().resolve() / "kinematics" / "004_normalization" / "tables" / "normalized_landmarks_manifest.csv"


def _selection_json_path(output_root: Path | str) -> Path:
    return Path(output_root).expanduser().resolve() / "kinematics" / "003_selection" / "tables" / "selected_landmarks.json"


def load_feature_selection(output_root: Path | str, fallback_preset: str = "ALS oral-motor core 15") -> tuple[tuple[int, ...], str]:
    path = _selection_json_path(output_root)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            vals = tuple(int(v) for v in payload.get("selected_landmarks", []) if int(v) >= 0)
            if vals:
                return tuple(dict.fromkeys(vals)), str(payload.get("preset") or "custom_visual_selection")
        except Exception:
            pass
    return tuple(LANDMARK_PRESETS[fallback_preset]), fallback_preset


def _coord_cols(idx: int) -> tuple[str, str, str]:
    return f"{idx}_x_norm", f"{idx}_y_norm", f"{idx}_z_norm"


def _has_point(df: pd.DataFrame, idx: int) -> bool:
    return all(c in df.columns for c in _coord_cols(idx))


def _time_seconds(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    if "timestamp_ms" in df.columns:
        t = pd.to_numeric(df["timestamp_ms"], errors="coerce").to_numpy(dtype=float) / 1000.0
        if n > 1 and np.isfinite(t).sum() > 1:
            finite = t[np.isfinite(t)]
            if finite.size > 1 and np.all(np.diff(finite) > 0):
                return t
    return np.arange(n, dtype=float)


def _fps_from_time(t: np.ndarray) -> float:
    finite = t[np.isfinite(t)]
    if finite.size > 1:
        d = np.diff(finite)
        d = d[d > 0]
        if d.size:
            return float(1.0 / np.nanmedian(d))
    return 30.0


def _interpolate_nans(x: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    return s.interpolate(method="linear", limit_direction="both").to_numpy(dtype=float)


def _remove_distribution_outliers(x: np.ndarray, extreme: float, tight: float) -> tuple[np.ndarray, int, int]:
    y = np.asarray(x, dtype=float).copy()

    def mask(sigma: float) -> np.ndarray:
        mean = np.nanmean(y)
        sd = np.nanstd(y)
        if not np.isfinite(sd) or sd <= 0:
            return np.zeros_like(y, dtype=bool)
        return np.abs(y - mean) > sigma * sd

    m1 = mask(extreme)
    y[m1] = np.nan
    m2 = mask(tight)
    y[m2] = np.nan
    return y, int(m1.sum()), int(m2.sum())


def _smooth_signal(x: np.ndarray, fps: float, cutoff_hz: float, order: int) -> np.ndarray:
    if len(x) < max(9, order * 3 + 2):
        return x
    nyq = max(fps / 2.0, 1e-6)
    cutoff = min(float(cutoff_hz), nyq * 0.95)
    if cutoff <= 0:
        return x
    try:
        b, a = butter(int(order), cutoff / nyq, btype="low")
        return filtfilt(b, a, x)
    except Exception:
        return x


def clean_signal(raw: np.ndarray, fps: float, cfg: FeatureComputationConfig) -> tuple[np.ndarray, dict]:
    arr = np.asarray(raw, dtype=float)
    n_missing = int(np.isnan(arr).sum())
    filled = _interpolate_nans(arr)
    deouted, n_extreme, n_tight = _remove_distribution_outliers(filled, cfg.outlier_sigma_extreme, cfg.outlier_sigma_tight)
    refilled = _interpolate_nans(deouted)
    smoothed = _smooth_signal(refilled, fps, cfg.smoothing_cutoff_hz, cfg.smoothing_order) if cfg.use_smoothed_signals else refilled
    return smoothed, {
        "n_missing_interpolated": n_missing,
        "n_outliers_extreme": n_extreme,
        "n_outliers_tight": n_tight,
    }


def _point_matrix(df: pd.DataFrame, idx: int) -> np.ndarray:
    cols = _coord_cols(idx)
    return df.loc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _distance(df: pd.DataFrame, a: int, b: int) -> np.ndarray:
    return np.sqrt(np.nansum((_point_matrix(df, a) - _point_matrix(df, b)) ** 2, axis=1))


def _axis(df: pd.DataFrame, idx: int, axis: str) -> np.ndarray:
    col = f"{idx}_{axis}"
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den = np.asarray(den, dtype=float)
    out = np.full_like(np.asarray(num, dtype=float), np.nan, dtype=float)
    ok = np.isfinite(den) & (np.abs(den) > 1e-12)
    out[ok] = np.asarray(num, dtype=float)[ok] / den[ok]
    return out


def _iqr(x: np.ndarray) -> float:
    if np.isfinite(x).sum() == 0:
        return math.nan
    return float(np.nanquantile(x, 0.75) - np.nanquantile(x, 0.25))


def _summaries(x: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    if np.isfinite(x).sum() == 0:
        return {f"{prefix}_{name}": math.nan for name in ["mean", "sd", "median", "iqr", "p05", "p95", "range_p05_p95", "min", "max"]}
    p05 = float(np.nanquantile(x, 0.05))
    p95 = float(np.nanquantile(x, 0.95))
    return {
        f"{prefix}_mean": float(np.nanmean(x)),
        f"{prefix}_sd": float(np.nanstd(x)),
        f"{prefix}_median": float(np.nanmedian(x)),
        f"{prefix}_iqr": _iqr(x),
        f"{prefix}_p05": p05,
        f"{prefix}_p95": p95,
        f"{prefix}_range_p05_p95": p95 - p05,
        f"{prefix}_min": float(np.nanmin(x)),
        f"{prefix}_max": float(np.nanmax(x)),
    }


def _velocity(sig: np.ndarray, t: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(sig) < 3:
        return np.full_like(sig, np.nan, dtype=float)
    if not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        return np.gradient(sig)
    return np.gradient(sig, t)


def _movement_windows(aperture: np.ndarray, cfg: FeatureComputationConfig) -> list[tuple[int, int, str]]:
    x = np.asarray(aperture, dtype=float)
    if len(x) < max(6, cfg.min_movement_len_frames * 2):
        return [(0, len(x), "whole")] if len(x) else []
    if np.isfinite(x).sum() < max(6, cfg.min_movement_len_frames * 2):
        return [(0, len(x), "whole")]
    x = _interpolate_nans(x)
    spread = float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5))
    prominence = spread * 0.10 if np.isfinite(spread) and spread > 0 else None
    peaks, _ = find_peaks(x, prominence=prominence)
    troughs, _ = find_peaks(-x, prominence=prominence)
    extrema = sorted([(int(i), "peak") for i in peaks] + [(int(i), "trough") for i in troughs])
    windows: list[tuple[int, int, str]] = []
    for (i0, t0), (i1, t1) in zip(extrema, extrema[1:]):
        if t0 == t1 or i1 - i0 < cfg.min_movement_len_frames:
            continue
        kind = "open" if t0 == "trough" and t1 == "peak" else "close"
        windows.append((i0, i1 + 1, kind))
    return windows or [(0, len(x), "whole")]


def _required_points_for_features() -> set[int]:
    pts: set[int] = set()
    for definition in _FEATURE_DEFINITIONS.values():
        if "points" in definition:
            pts.update(int(v) for v in definition["points"])
    return pts


def compute_feature_timeseries(df: pd.DataFrame, cfg: FeatureComputationConfig) -> tuple[pd.DataFrame, dict]:
    if df.empty:
        raise ValueError("Normalized landmark table is empty.")
    t = _time_seconds(df)
    fps = _fps_from_time(t)
    out = pd.DataFrame({
        "frame": df.get("frame", pd.Series(range(len(df)))),
        "timestamp_ms": df.get("timestamp_ms", pd.Series([np.nan] * len(df))),
        "time_s": t,
        "face_detected": df.get("face_detected", pd.Series([True] * len(df))).astype(bool),
    })
    cleaning_rows: dict[str, int] = {"n_missing_interpolated_total": 0, "n_outliers_extreme_total": 0, "n_outliers_tight_total": 0}
    missing_feature_inputs: set[str] = set()

    raw_signals: dict[str, np.ndarray] = {}
    for name, definition in _FEATURE_DEFINITIONS.items():
        kind = definition["kind"]
        try:
            if kind == "distance":
                a, b = definition["points"]
                if not (_has_point(df, a) and _has_point(df, b)):
                    missing_feature_inputs.add(name)
                    continue
                raw_signals[name] = _distance(df, a, b)
            elif kind == "ratio":
                num = raw_signals.get(definition["numerator"])
                den = raw_signals.get(definition["denominator"])
                if num is None or den is None:
                    missing_feature_inputs.add(name)
                    continue
                raw_signals[name] = _safe_ratio(num, den)
            elif kind == "abs_delta_axis":
                a, b = definition["points"]
                axis = definition["axis"]
                if f"{a}_{axis}" not in df.columns or f"{b}_{axis}" not in df.columns:
                    missing_feature_inputs.add(name)
                    continue
                raw_signals[name] = np.abs(_axis(df, a, axis) - _axis(df, b, axis))
            elif kind == "bilateral_abs_balance":
                a, b = definition["points"]
                axis = definition["axis"]
                if f"{a}_{axis}" not in df.columns or f"{b}_{axis}" not in df.columns:
                    missing_feature_inputs.add(name)
                    continue
                raw_signals[name] = np.abs(np.abs(_axis(df, a, axis)) - np.abs(_axis(df, b, axis)))
        except Exception:
            missing_feature_inputs.add(name)

    for name, values in raw_signals.items():
        cleaned, stats = clean_signal(values, fps, cfg)
        out[f"{name}_raw"] = values
        out[name] = cleaned
        out[f"{name}_velocity"] = _velocity(cleaned, t)
        for k, v in stats.items():
            cleaning_rows[k + "_total"] = cleaning_rows.get(k + "_total", 0) + int(v)

    if "mouth_aperture" in out.columns:
        windows = _movement_windows(out["mouth_aperture"].to_numpy(dtype=float), cfg)
    else:
        windows = [(0, len(out), "whole")]
    movement_id = np.full(len(out), -1, dtype=int)
    movement_kind = np.array([""] * len(out), dtype=object)
    for i, (start, stop, kind) in enumerate(windows):
        movement_id[start:stop] = i
        movement_kind[start:stop] = kind
    out["movement_id"] = movement_id
    out["movement_kind"] = movement_kind

    meta = {
        "fps_estimated": fps,
        "n_frames": int(len(out)),
        "n_detected_frames": int(out["face_detected"].sum()),
        "face_detected_fraction": float(out["face_detected"].mean()) if len(out) else math.nan,
        "n_movements": int(len(windows)),
        "n_open_movements": int(sum(1 for _, _, k in windows if k == "open")),
        "n_close_movements": int(sum(1 for _, _, k in windows if k == "close")),
        "n_whole_movements": int(sum(1 for _, _, k in windows if k == "whole")),
        "missing_feature_inputs": ";".join(sorted(missing_feature_inputs)),
        **cleaning_rows,
    }
    return out, meta


def summarize_timeseries(ts: pd.DataFrame, meta: dict) -> dict:
    row: dict[str, object] = dict(meta)
    signal_cols = [c for c in ts.columns if c in _FEATURE_DEFINITIONS]
    for col in signal_cols:
        row.update(_summaries(ts[col].to_numpy(dtype=float), col))
        vcol = f"{col}_velocity"
        if vcol in ts.columns:
            row.update(_summaries(np.abs(ts[vcol].to_numpy(dtype=float)), f"{col}_speed_abs"))
            row[f"{col}_path_length"] = float(np.trapezoid(np.abs(ts[vcol].to_numpy(dtype=float)))) if np.isfinite(ts[vcol]).any() else math.nan
    if "movement_id" in ts.columns and signal_cols:
        for col in signal_cols:
            ranges: list[float] = []
            for mid in sorted(int(v) for v in pd.unique(ts["movement_id"]) if int(v) >= 0):
                vals = ts.loc[ts["movement_id"] == mid, col].to_numpy(dtype=float)
                if np.isfinite(vals).sum() >= 2:
                    ranges.append(float(np.nanpercentile(vals, 95) - np.nanpercentile(vals, 5)))
            if ranges:
                row[f"{col}_movement_range_median"] = float(np.nanmedian(ranges))
                row[f"{col}_movement_range_iqr"] = _iqr(np.asarray(ranges, dtype=float))
            else:
                row[f"{col}_movement_range_median"] = math.nan
                row[f"{col}_movement_range_iqr"] = math.nan
    flags: list[str] = []
    if row.get("face_detected_fraction", 1.0) < 0.75:
        flags.append("low_face_detection")
    if row.get("missing_feature_inputs"):
        flags.append("missing_feature_inputs")
    if int(row.get("n_movements") or 0) <= 1:
        flags.append("no_clear_repetitions")
    row["feature_qc_flags"] = ";".join(flags)
    return row


def compute_features_for_normalized_file(input_csv: Path | str, output_timeseries_csv: Path | str, cfg: FeatureComputationConfig) -> dict:
    input_csv = Path(input_csv).expanduser().resolve()
    output_timeseries_csv = Path(output_timeseries_csv).expanduser().resolve()
    df = pd.read_csv(input_csv)
    ts, meta = compute_feature_timeseries(df, cfg)
    output_timeseries_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_timeseries_csv.with_suffix(".part.csv")
    ts.to_csv(tmp, index=False)
    tmp.replace(output_timeseries_csv)
    row = summarize_timeseries(ts, meta)
    row["input_normalized_csv"] = str(input_csv)
    row["output_timeseries_csv"] = str(output_timeseries_csv)
    row["status"] = "ok" if not row.get("feature_qc_flags") else "qc_flagged"
    return row


def run_feature_computation(
    output_root: Path | str,
    cfg: FeatureComputationConfig | None = None,
    normalized_manifest_csv: Path | str | None = None,
) -> dict:
    cfg = cfg or FeatureComputationConfig()
    root = Path(output_root).expanduser().resolve()
    manifest_path = Path(normalized_manifest_csv).expanduser().resolve() if normalized_manifest_csv else _normalization_manifest(root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Normalized landmark manifest not found: {manifest_path}. Run Normalization first.")
    manifest = pd.read_csv(manifest_path)
    out_root = _features_dir(root)
    timeseries_dir = out_root / "timeseries"
    rows: list[dict] = []
    for _, rec in manifest.iterrows():
        video_id = str(rec.get("video_id", "") or Path(str(rec.get("output_csv", ""))).stem.replace("-norm-lmks", ""))
        input_csv = Path(str(rec.get("output_csv", "")))
        if not input_csv.exists():
            input_csv = manifest_path.parent / f"{video_id}-norm-lmks.csv"
        row = {
            "video_id": video_id,
            "source_path": rec.get("source_path", ""),
            "task_guess": rec.get("task_guess", ""),
            "normalization_status": rec.get("status", ""),
        }
        try:
            if str(rec.get("status", "")) == "error":
                raise ValueError("Normalization status is error; feature computation skipped for this video.")
            ts_csv = timeseries_dir / f"{video_id}-kinematic-timeseries.csv"
            feat_row = compute_features_for_normalized_file(input_csv, ts_csv, cfg)
            row.update(feat_row)
        except Exception as exc:  # noqa: BLE001
            row.update({"status": "error", "error": repr(exc), "input_normalized_csv": str(input_csv), "output_timeseries_csv": ""})
        rows.append(row)
    features_df = pd.DataFrame(rows)
    features_csv = out_root / "tables" / "kinematic_features.csv"
    features_df.to_csv(features_csv, index=False)
    registry = pd.DataFrame([
        {"feature": name, "family": d.get("family"), "definition": d.get("description"), "required_inputs": ",".join(map(str, d.get("points", ())))}
        for name, d in _FEATURE_DEFINITIONS.items()
    ])
    registry_csv = out_root / "tables" / "kinematic_feature_registry.csv"
    registry.to_csv(registry_csv, index=False)
    framework_paths = write_feature_framework_catalog(root)
    manifest_json = out_root / "tables" / "feature_computation_manifest.json"
    status_counts = features_df.get("status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not features_df.empty else {}
    payload = {
        "config": asdict(cfg),
        "input_normalized_manifest": str(manifest_path),
        "features_csv": str(features_csv),
        "feature_registry_csv": str(registry_csv),
        "feature_framework_csv": str(framework_paths["framework_csv"]),
        "feature_implementation_audit_csv": str(framework_paths["audit_csv"]),
        "feature_framework_json": str(framework_paths["framework_json"]),
        "n_videos": int(len(features_df)),
        "n_ok": int((features_df.get("status") == "ok").sum()) if not features_df.empty else 0,
        "n_qc_flagged": int((features_df.get("status") == "qc_flagged").sum()) if not features_df.empty else 0,
        "n_error": int((features_df.get("status") == "error").sum()) if not features_df.empty else 0,
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "note": "Feature rows are computed from normalized landmark trajectories. QC flags are retained, not automatically excluded.",
    }
    manifest_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return {"features_csv": features_csv, "feature_registry_csv": registry_csv, "manifest_json": manifest_json, **{k: payload[k] for k in ["n_videos", "n_ok", "n_qc_flagged", "n_error", "status_counts"]}}


__all__ = [
    "FeatureComputationConfig",
    "load_feature_selection",
    "compute_feature_timeseries",
    "compute_features_for_normalized_file",
    "feature_framework_dataframe",
    "feature_implementation_audit_dataframe",
    "write_feature_framework_catalog",
    "run_feature_computation",
]
