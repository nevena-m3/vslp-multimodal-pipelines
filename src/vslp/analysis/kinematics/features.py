"""Kinematic feature registry for the VSLP kinematics GUI.

The registry mirrors the validated acoustic feature selector pattern: features are
organized by biomechanical subsystem, each feature records its native signal,
landmark dependencies, normalization requirement, aggregation rule and current
implementation status. The formulas are adapted from the uploaded legacy
kinematics analysis script and kept declarative here so the GUI can expose the
computation policy before numerical execution is connected.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class KinematicFeatureSpec:
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


ICD = "intercanthal distance; landmarks 243 and 463"
ROBUST = "median, IQR, 5th percentile, 95th percentile, 5-95 spread"
MED_IQR = "median and IQR"


KINEMATIC_FEATURE_SPECS: tuple[KinematicFeatureSpec, ...] = (
    KinematicFeatureSpec(
        "path_vert_med", "Vertical lip/jaw displacement", "Vertical path length", "implemented-formula", "B",
        "lower lip vertical aperture trajectory", "ICD-normalized distance", (17, 8, 243, 463), ICD, MED_IQR,
        "Total absolute vertical movement over the analysis window or repetition.", "get_dist_normed(task='vertical') + path_vert",
    ),
    KinematicFeatureSpec(
        "rom_vert_med", "Vertical lip/jaw displacement", "Vertical range of motion", "implemented-formula", "B",
        "lower lip vertical aperture trajectory", "ICD-normalized distance", (17, 8, 243, 463), ICD, MED_IQR,
        "95th minus 5th percentile vertical opening range.", "get_dist_normed(task='vertical') + rom_vert",
    ),
    KinematicFeatureSpec(
        "sLL_vert", "Vertical lip/jaw displacement", "Vertical speed", "implemented-formula", "B",
        "gradient of vertical aperture trajectory", "ICD-normalized distance/s", (17, 8, 243, 463), ICD, ROBUST,
        "Magnitude of lower-lip/jaw vertical movement speed.", "get_sLL_vars(sLL_vert)",
    ),
    KinematicFeatureSpec(
        "aLL_vert", "Vertical lip/jaw displacement", "Vertical acceleration", "implemented-formula", "C",
        "gradient of vertical speed", "ICD-normalized distance/s2", (17, 8, 243, 463), ICD, ROBUST,
        "Magnitude of vertical movement acceleration; sensitive to smoothing and timestamps.", "get_aLL_vars(aLL_vert)",
    ),
    KinematicFeatureSpec(
        "path_horz_med", "Horizontal lip spread", "Horizontal path length", "implemented-formula", "B",
        "left-right commissure spread trajectory", "ICD-normalized distance", (61, 291, 243, 463), ICD, MED_IQR,
        "Total absolute horizontal lip-spread movement over the window or repetition.", "get_dist_normed(task='horizontal') + path_horz",
    ),
    KinematicFeatureSpec(
        "rom_horz_med", "Horizontal lip spread", "Horizontal range of motion", "implemented-formula", "B",
        "left-right commissure spread trajectory", "ICD-normalized distance", (61, 291, 243, 463), ICD, MED_IQR,
        "95th minus 5th percentile horizontal lip-spread range.", "get_dist_normed(task='horizontal') + rom_horz",
    ),
    KinematicFeatureSpec(
        "sLL_horz", "Horizontal lip spread", "Horizontal speed", "implemented-formula", "B",
        "gradient of horizontal spread trajectory", "ICD-normalized distance/s", (61, 291, 243, 463), ICD, ROBUST,
        "Magnitude of horizontal lip-spread speed.", "get_sLL_vars(sLL_horz)",
    ),
    KinematicFeatureSpec(
        "aLL_horz", "Horizontal lip spread", "Horizontal acceleration", "implemented-formula", "C",
        "gradient of horizontal speed", "ICD-normalized distance/s2", (61, 291, 243, 463), ICD, ROBUST,
        "Magnitude of horizontal spread acceleration; sensitive to smoothing and timestamps.", "get_aLL_vars(aLL_horz)",
    ),
    KinematicFeatureSpec(
        "lip_aspect", "Lip aperture geometry", "Lip aspect ratio", "implemented-formula", "B",
        "vertical lip distance divided by horizontal commissure distance", "ratio", (0, 17, 61, 291, 243, 463), ICD, ROBUST,
        "Relative mouth opening shape: vertical aperture scaled by horizontal lip spread.", "get_lip_aspect_ratio + get_lip_aspect_vars",
    ),
    KinematicFeatureSpec(
        "jaw_lateralization", "Jaw lateralization", "Jaw lateralization ratio", "implemented-formula", "B",
        "distance lower lip/jaw point to left canthus divided by distance to right canthus", "ratio", (17, 243, 463), ICD, ROBUST,
        "Asymmetry/lateral deviation of the lower jaw or lower lip relative to eye anchors.", "get_dist_jawlat + get_jaw_lat_vars",
    ),
    KinematicFeatureSpec(
        "lip_symmetry", "Lip symmetry", "Left-right lip motion symmetry", "implemented-formula", "B",
        "left commissure-to-reference distance divided by right commissure-to-reference distance", "ratio", (61, 291, 8, 243, 463), ICD, ROBUST,
        "Symmetry of left and right oral commissure movement relative to a midline/forehead reference.", "get_lat_lip_symm + get_lip_symm_vars",
    ),
    KinematicFeatureSpec(
        "lat_xcorr", "Bilateral coordination", "Left-right oral commissure cross-correlation", "implemented-formula", "C",
        "normalized cross-correlation of left and right commissure distance trajectories", "correlation", (61, 291, 8, 243, 463), ICD, "maximum normalized cross-correlation",
        "Coordination between left and right oral commissure trajectories.", "get_lat_xcorr",
    ),
)


KINEMATIC_FEATURE_GROUPS: dict[str, tuple[KinematicFeatureSpec, ...]] = {}
for spec in KINEMATIC_FEATURE_SPECS:
    KINEMATIC_FEATURE_GROUPS.setdefault(spec.group, tuple())
    KINEMATIC_FEATURE_GROUPS[spec.group] = (*KINEMATIC_FEATURE_GROUPS[spec.group], spec)


DEFAULT_KINEMATIC_FEATURE_IDS = tuple(spec.feature_id for spec in KINEMATIC_FEATURE_SPECS if spec.tier in {"B", "C"})


QC_FEATURE_REQUIREMENTS = (
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


def feature_registry_dataframe():
    """Return the registry as a pandas DataFrame without importing pandas at module import."""
    import pandas as pd

    rows = []
    for spec in KINEMATIC_FEATURE_SPECS:
        row = asdict(spec)
        row["landmarks"] = ", ".join(map(str, spec.landmarks))
        rows.append(row)
    return pd.DataFrame(rows)


def selected_specs(feature_ids: Iterable[str]) -> tuple[KinematicFeatureSpec, ...]:
    wanted = set(feature_ids)
    return tuple(spec for spec in KINEMATIC_FEATURE_SPECS if spec.feature_id in wanted)
