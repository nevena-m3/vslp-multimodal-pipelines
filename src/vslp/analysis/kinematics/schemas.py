"""Schemas and defaults for the VSLP kinematics GUI scaffold.

This module is deliberately dependency-light. Heavy video/MediaPipe imports are
kept out of schemas so the GUI and tests can import on machines that have not
installed the kinematic extras yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

DEFAULT_VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".wmv", ".mpg", ".mpeg", ".3gp"
})

# These are reviewer-facing, not claims of clinical optimality. They provide a
# reproducible starting point and are expected to be revised as the lab finalizes
# the visible articulator feature set.
LANDMARK_PRESETS: dict[str, tuple[int, ...]] = {
    "ALS oral-motor core 17": (13, 14, 61, 291, 78, 308, 81, 311, 0, 17, 152, 199, 33, 133, 263, 362, 1),
    # Backward-compatible alias kept for old configs; includes the two inner-canthus anchors required by default normalization.
    "ALS oral-motor core 15": (13, 14, 61, 291, 78, 308, 81, 311, 0, 17, 152, 199, 33, 133, 263, 362, 1),
    "Lower-face jaw/lip kinematics": (13, 14, 17, 152, 175, 199, 200, 61, 291, 78, 308, 0, 164, 37, 267, 133, 362),
    "Lip symmetry and lateralization": (61, 291, 78, 308, 57, 287, 40, 270, 33, 263, 133, 362, 152, 10, 1),
    "Parkinson hypomimia / facial expressivity": (70, 300, 105, 334, 159, 386, 145, 374, 61, 291, 13, 14, 0, 17, 152),
    "Broad audit 31": (0, 1, 10, 13, 14, 17, 33, 37, 40, 57, 61, 70, 78, 81, 105, 133, 145, 152, 159, 164, 175, 199, 200, 263, 267, 270, 287, 291, 300, 308, 362),
    "Broad audit 30": (0, 1, 10, 13, 14, 17, 33, 37, 40, 57, 61, 70, 78, 81, 105, 133, 145, 152, 159, 164, 175, 199, 200, 263, 267, 270, 287, 291, 300, 308, 362),
}

NORMALIZATION_METHODS: dict[str, str] = {
    "intercanthal_distance": "Scale distances by left-right eye/canthus distance; default for mouth/jaw features because it reduces camera-distance effects.",
    "interpupillary_or_outer_eye": "Scale by eye-width proxy. Useful when eye landmarks are stable and face is frontal.",
    "face_bbox_width": "Scale by detected face bounding-box width; robust fallback when canthus landmarks are unreliable.",
    "face_height_nose_chin": "Scale by vertical face-height proxy; can be useful for jaw-opening features but more pose-sensitive.",
    "procrustes_head_stabilized": "Future option: rigid/head-pose stabilization before feature computation.",
    "raw_normalized_coordinates": "Audit only: MediaPipe normalized coordinates without anatomical scaling; not recommended for default analysis.",
}

AGGREGATION_PROFILES: dict[str, str] = {
    "robust_default": "Median, IQR, 5th/95th percentiles, tail spread, and missing/quality coverage; default for per-video scalar export.",
    "movement_segmented": "Segment opening/closing movements first, then summarize per movement and across movements.",
    "full_timeseries_summary": "Summarize the entire frame-level trajectory without movement segmentation; useful for connected speech or non-repetitive tasks.",
    "clinically_sensitive": "Export robust summaries plus peak/range/rate measures; higher interpretability, more QC-sensitive.",
    "exploratory_dense": "Large set of scalar reductions for discovery; not recommended as ML default without feature recommendation review.",
}

@dataclass(frozen=True)
class VideoIngestConfig:
    input_root: Path
    output_root: Path
    recursive: bool = True
    extensions: frozenset[str] = field(default_factory=lambda: DEFAULT_VIDEO_EXTENSIONS)
    ffprobe_bin: str = "ffprobe"
    run_name: str = "kinematics"

@dataclass(frozen=True)
class VideoRecord:
    source_path: str
    relative_path: str
    video_id: str
    extension: str
    size_bytes: int
    task_guess: str
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_sec: float | None = None
    n_frames_estimated: int | None = None
    codec_name: str | None = None
    container_name: str | None = None
    status: str = "discovered"
    warning: str = ""


def parse_int_list(text: str) -> tuple[int, ...]:
    """Parse comma/space-separated landmark indices with stable ordering."""
    raw = text.replace(";", ",").replace("\n", ",").replace("\t", ",")
    vals: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        vals.append(int(token))
    seen = set()
    out = []
    for v in vals:
        if v < 0:
            raise ValueError("Landmark indices must be non-negative.")
        if v not in seen:
            seen.add(v); out.append(v)
    return tuple(out)


def selected_landmarks_from_preset(name: str) -> tuple[int, ...]:
    if name not in LANDMARK_PRESETS:
        raise KeyError(f"Unknown landmark preset: {name}")
    return LANDMARK_PRESETS[name]

WORKFLOW_STAGES = [
    ("Setup", "Discover and structurally probe videos", "000_ingest"),
    ("Metadata", "Link subjects, sessions, tasks, clinical labels", "001_metadata"),
    ("Landmarks", "Extract MediaPipe Face Landmarker trajectories", "002_landmarks"),
    ("Selection", "Choose clinically meaningful landmark subsets", "003_selection"),
    ("Normalization", "Scale/stabilize coordinates before features", "004_normalization"),
    ("Video QC", "Quantify landmark/visibility/acquisition quality", "005_video_qc"),
    ("Features", "Compute frame/movement-level kinematics", "006_features"),
    ("Aggregation", "Collapse time series transparently", "007_aggregation"),
    ("Inspector", "Review tables, manifests, and QC evidence", "008_inspector"),
    ("Reports", "Export reproducible package and report", "009_reports"),
]

STAGE_GUIDANCE = {
    "setup": {"purpose": "Create a reproducible inventory of all candidate videos before any transformation.", "decision": "Confirm expected videos are present, readable, and assigned plausible task guesses before moving on."},
    "metadata": {"purpose": "Attach participant/session/task/clinical context without requiring it for landmark extraction.", "decision": "Verify IDs/tasks/sessions align; unresolved metadata should be documented rather than forced."},
    "landmarks": {"purpose": "Configure the MediaPipe Face Landmarker stage and document extraction assumptions.", "decision": "Keep confidence thresholds traceable; higher thresholds increase missing frames, lower thresholds may accept uncertain frames."},
    "selection": {"purpose": "Reduce the full face mesh to reproducible, anatomically meaningful landmark sets.", "decision": "Use ALS oral-motor core by default; choose broader sets for exploratory facial expressivity or hypomimia analysis."},
    "normalization": {"purpose": "Define how distances and movements are scaled so values are comparable across camera distance and face size.", "decision": "Intercanthal distance is default for oral/jaw kinematics; raw normalized coordinates are audit only."},
    "qc": {"purpose": "Separate visual/acquisition problems from true facial movement signals.", "decision": "QC should flag and explain risk; it should not automatically exclude videos without analyst review."},
    "features": {"purpose": "Compute interpretable kinematic signals from cleaned, normalized landmark trajectories.", "decision": "Document which features are raw trajectories, movement-derived summaries, or exploratory outputs."},
    "aggregation": {"purpose": "Turn time-series features into one row per video while preserving clinically relevant variability.", "decision": "Use robust default summaries for ML-ready exports; retain movement/time-series evidence for audit."},
    "inspector": {"purpose": "Let the analyst inspect stage outputs before trusting downstream tables.", "decision": "Use this to detect wrong paths, metadata mismatch, missing landmark outputs, or unexpected warnings."},
    "reports": {"purpose": "Package the run into a reproducible, SOP-aligned report.", "decision": "Use the report as the handoff artifact before feature-analysis/ML stages."},
}
